"""CLI and state machine for the five-stage budgeted preprocessing campaign."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time
from types import MappingProxyType
from typing import Mapping, Sequence
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from .budgeted_contracts import BudgetedCampaign, BudgetedJob, load_budgeted_campaign
from .budgeted_decisions import (
    DecisionError,
    choose_dl_representative,
    decide_tabnet,
    final_preprocessing_status,
    fixed_blend_metrics,
    promote_single,
)
from .budgeted_scheduler import BudgetedScheduler, SchedulerSummary


class StageNeedsReview(RuntimeError):
    """Raised when automatic stage selection would require guessing."""


@dataclass(frozen=True)
class ResumeSelection:
    path: Path
    completed_stage: int
    manifest_sha256: str


@dataclass(frozen=True)
class StageFiveDecision:
    status: str
    reason: str
    common_epochs: int
    campaign_terminal: bool = True
    baseline_best_brier: float | None = None
    candidate_best_brier: float | None = None
    predictions_comparable: bool = False


@dataclass(frozen=True)
class AutoResult:
    stage_id: int
    terminal_state: str
    completed_stage: int
    resume_bundle: Path | None = None
    review_bundle: Path | None = None


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _campaign_identity(config: Path, data_dir: Path) -> dict[str, str]:
    train = data_dir / "train.csv"
    history = data_dir / "trackman_history.csv"
    if not train.is_file() or not history.is_file():
        raise ValueError("train.csv and trackman_history.csv are required")
    experiments_root = Path(__file__).parents[1]
    code_paths = sorted(
        [
            *(
                path
                for package in (
                    experiments_root / "independent_dl",
                    experiments_root / "catboost_preprocessing",
                    experiments_root / "preprocessing_campaign",
                )
                for path in package.rglob("*.py")
                if not path.name.startswith("KAGGLE_")
            ),
            *(experiments_root / "preprocessing_campaign").glob("requirements-*.txt"),
        ],
        key=lambda path: path.as_posix(),
    )
    code_digest = sha256()
    for path in code_paths:
        code_digest.update(path.name.encode("utf-8"))
        code_digest.update(path.read_bytes())
    return {
        "config_sha256": _file_sha256(config),
        "train_sha256": _file_sha256(train),
        "trackman_history_sha256": _file_sha256(history),
        "code_sha256": code_digest.hexdigest(),
    }


def _read_last_member(archive: ZipFile, name: str) -> bytes:
    matches = [info for info in archive.infolist() if info.filename == name]
    if not matches:
        raise KeyError(name)
    return archive.read(matches[-1])


def _archive_member_sha256(archive: ZipFile, name: str) -> str:
    digest = sha256()
    with archive.open(name) as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def inspect_resume_bundles(
    input_root: str | Path,
    *,
    campaign_id: str = "budgeted_preprocessing_campaign_v1",
) -> ResumeSelection | None:
    """Choose the highest internally valid resume bundle without guessing ties."""

    selections: list[ResumeSelection] = []
    for path in sorted(Path(input_root).rglob("*resume_bundle.zip")):
        try:
            with ZipFile(path) as archive:
                names = [info.filename for info in archive.infolist()]
                if len(names) != len(set(names)):
                    continue
                metadata = json.loads(
                    _read_last_member(archive, "resume_metadata.json").decode("utf-8")
                )
                manifest = _read_last_member(archive, "campaign_manifest.json")
                file_records = metadata.get("files") if isinstance(metadata, dict) else None
                if not isinstance(file_records, list) or not file_records:
                    continue
                expected_names = {"resume_metadata.json"}
                files_valid = True
                for record in file_records:
                    if (
                        not isinstance(record, dict)
                        or set(record) != {"path", "size_bytes", "sha256"}
                    ):
                        files_valid = False
                        break
                    name = str(record["path"])
                    expected_names.add(name)
                    if (
                        name not in names
                        or int(record["size_bytes"]) != archive.getinfo(name).file_size
                        or str(record["sha256"])
                        != _archive_member_sha256(archive, name)
                    ):
                        files_valid = False
                        break
                if not files_valid or expected_names != set(names):
                    continue
        except (BadZipFile, KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema_version") != 1
            or metadata.get("campaign_id") != campaign_id
        ):
            continue
        completed_stage = metadata.get("completed_stage")
        manifest_hash = metadata.get("manifest_sha256")
        if (
            isinstance(completed_stage, bool)
            or not isinstance(completed_stage, int)
            or completed_stage not in range(0, 6)
            or not isinstance(manifest_hash, str)
            or sha256(manifest).hexdigest() != manifest_hash
        ):
            continue
        selections.append(ResumeSelection(path, completed_stage, manifest_hash))
    metadata_paths = sorted(Path(input_root).rglob("resume_metadata.json"))
    valid_extracted_paths: set[Path] = set()
    for metadata_path in metadata_paths:
        path = metadata_path.parent
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            manifest_path = path / "campaign_manifest.json"
            manifest = manifest_path.read_bytes()
            file_records = metadata.get("files") if isinstance(metadata, dict) else None
            if not isinstance(file_records, list) or not file_records:
                continue
            expected_names = {"resume_metadata.json"}
            files_valid = True
            for record in file_records:
                if (
                    not isinstance(record, dict)
                    or set(record) != {"path", "size_bytes", "sha256"}
                ):
                    files_valid = False
                    break
                name = str(record["path"])
                expected_names.add(name)
                candidate = path / name
                if (
                    not candidate.is_file()
                    or int(record["size_bytes"]) != candidate.stat().st_size
                    or str(record["sha256"]) != _file_sha256(candidate)
                ):
                    files_valid = False
                    break
            observed_names = {
                candidate.relative_to(path).as_posix()
                for candidate in path.rglob("*")
                if candidate.is_file()
            }
            if not files_valid or expected_names != observed_names:
                continue
        except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if (
            metadata.get("schema_version") != 1
            or metadata.get("campaign_id") != campaign_id
        ):
            continue
        completed_stage = metadata.get("completed_stage")
        manifest_hash = metadata.get("manifest_sha256")
        if (
            isinstance(completed_stage, bool)
            or not isinstance(completed_stage, int)
            or completed_stage not in range(0, 6)
            or not isinstance(manifest_hash, str)
            or sha256(manifest).hexdigest() != manifest_hash
        ):
            continue
        selections.append(ResumeSelection(path, completed_stage, manifest_hash))
        valid_extracted_paths.add(path)
    invalid_extracted = [
        metadata_path.parent
        for metadata_path in metadata_paths
        if metadata_path.parent not in valid_extracted_paths
    ]
    if invalid_extracted:
        raise StageNeedsReview(
            f"invalid extracted resume dataset(s): {invalid_extracted}"
        )
    if not selections:
        return None
    highest = max(item.completed_stage for item in selections)
    finalists = [item for item in selections if item.completed_stage == highest]
    hashes = {item.manifest_sha256 for item in finalists}
    if len(hashes) != 1:
        raise StageNeedsReview(
            f"conflicting hash-valid resume bundles exist for completed stage {highest}"
        )
    return finalists[-1]


def _selected_model(state: Mapping[str, object]) -> Mapping[str, object]:
    selected = state.get("selected_model")
    if not isinstance(selected, dict):
        raise StageNeedsReview("selected_model is missing from stage state")
    required = {"family", "profile_id", "model", "training"}
    if not required.issubset(selected):
        raise StageNeedsReview("selected_model contract is incomplete")
    return selected


def _bind_selected(job: BudgetedJob, selected: Mapping[str, object]) -> BudgetedJob:
    family = str(selected["family"])
    return replace(
        job,
        job_id=job.job_id.replace("selected_dl", family),
        family=family,
        profile_id=str(selected["profile_id"]),
        model=MappingProxyType(dict(selected["model"])),
        training=MappingProxyType(dict(selected["training"])),
    )


def _catboost_template(campaign: BudgetedCampaign) -> BudgetedJob:
    return next(job for job in campaign.stage_jobs(1) if job.family == "catboost")


def _candidate_descriptor(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise StageNeedsReview("promoted preprocessing descriptor is invalid")
    components = raw.get("components")
    if not isinstance(components, list):
        raise StageNeedsReview("promoted preprocessing components are invalid")
    return {
        "setting_id": str(raw["setting_id"]),
        "preprocessing_profile": str(raw["preprocessing_profile"]),
        "components": tuple(str(item) for item in components),
    }


def _stage_four_jobs(
    campaign: BudgetedCampaign, state: Mapping[str, object]
) -> tuple[BudgetedJob, ...]:
    selected = _selected_model(state)
    dl_candidates = [
        _candidate_descriptor(item) for item in state.get("promoted_dl", [])
    ][:2]
    cat_candidates = [
        _candidate_descriptor(item) for item in state.get("promoted_catboost", [])
    ][:2]
    jobs: list[BudgetedJob] = []
    dl_base = _bind_selected(campaign.stage_jobs(2)[0], selected)
    descriptors = [
        {
            "setting_id": "dl_standard",
            "preprocessing_profile": "dl_standard",
            "components": (),
        },
        *dl_candidates[:1],
    ]
    if len(dl_candidates) == 2:
        descriptors.append(
            {
                "setting_id": "combo__"
                + "__".join(str(item["setting_id"]) for item in dl_candidates),
                "preprocessing_profile": (
                    "dl_selective_transform"
                    if any(
                        item["preprocessing_profile"] == "dl_selective_transform"
                        for item in dl_candidates
                    )
                    else "dl_standard"
                ),
                "components": tuple(
                    sorted(
                        {
                            component
                            for item in dl_candidates
                            for component in item["components"]
                        }
                    )
                ),
            }
        )
    for descriptor in descriptors:
        jobs.append(
            replace(
                dl_base,
                job_id=(
                    f"s4__{selected['family']}__{descriptor['setting_id']}"
                    "__tr2022__va2023__s42"
                ),
                stage_id=4,
                setting_id=str(descriptor["setting_id"]),
                preprocessing_profile=str(descriptor["preprocessing_profile"]),
                components=tuple(descriptor["components"]),
                train_end_year=2022,
                valid_year=2023,
                max_seconds=1800,
            )
        )
    if len(dl_candidates) == 2:
        combo = descriptors[-1]
        jobs.append(
            replace(
                dl_base,
                job_id=(
                    f"s4__{selected['family']}__{combo['setting_id']}"
                    "__tr2023__va2024__s42__proxy"
                ),
                stage_id=4,
                setting_id=str(combo["setting_id"]),
                preprocessing_profile=str(combo["preprocessing_profile"]),
                components=tuple(combo["components"]),
                max_seconds=1800,
            )
        )
    cat_base = _catboost_template(campaign)
    cat_descriptors = [
        {
            "setting_id": "tree_native",
            "preprocessing_profile": "tree_native",
            "components": (),
        },
        *cat_candidates[:1],
    ]
    if len(cat_candidates) == 2:
        cat_descriptors.append(
            {
                "setting_id": "combo__"
                + "__".join(str(item["setting_id"]) for item in cat_candidates),
                "preprocessing_profile": "tree_native",
                "components": tuple(
                    sorted(
                        {
                            component
                            for item in cat_candidates
                            for component in item["components"]
                        }
                    )
                ),
            }
        )
    combo_setting = (
        str(cat_descriptors[-1]["setting_id"])
        if len(cat_candidates) == 2
        else None
    )
    for descriptor in cat_descriptors:
        folds = [(2022, 2023, "proxy"), (2023, 2024, "full")]
        if descriptor["setting_id"] == combo_setting:
            folds.insert(1, (2023, 2024, "proxy"))
        for train_end, valid, sample_mode in folds:
            jobs.append(
                replace(
                    cat_base,
                    job_id=(
                        f"s4__catboost__{descriptor['setting_id']}"
                        f"__tr{train_end}__va{valid}__s42__{sample_mode}"
                    ),
                    stage_id=4,
                    setting_id=str(descriptor["setting_id"]),
                    components=tuple(descriptor["components"]),
                    train_end_year=train_end,
                    valid_year=valid,
                    max_seconds=350,
                    sample_mode=sample_mode,
                )
            )
    return tuple(jobs)


def _stage_five_jobs(
    campaign: BudgetedCampaign, state: Mapping[str, object]
) -> tuple[BudgetedJob, ...]:
    selected = _selected_model(state)
    candidate_raw = state.get("final_dl")
    if not isinstance(candidate_raw, dict):
        return ()
    candidate = _candidate_descriptor(candidate_raw)
    base = _bind_selected(campaign.stage_jobs(2)[0], selected)
    baseline = replace(
        base,
        job_id=f"s5__{selected['family']}__dl_standard__tr2023__va2024__s42__full",
        stage_id=5,
        max_seconds=4800,
        sample_mode="full",
    )
    treatment = replace(
        baseline,
        job_id=(
            f"s5__{selected['family']}__{candidate['setting_id']}"
            "__tr2023__va2024__s42__full"
        ),
        setting_id=str(candidate["setting_id"]),
        preprocessing_profile=str(candidate["preprocessing_profile"]),
        components=tuple(candidate["components"]),
    )
    return baseline, treatment


def build_stage_jobs(
    campaign: BudgetedCampaign,
    stage_id: int,
    state: Mapping[str, object],
) -> tuple[BudgetedJob, ...]:
    if stage_id == 1:
        return campaign.stage_jobs(1)
    if stage_id in {2, 3}:
        selected = _selected_model(state)
        jobs = [_bind_selected(job, selected) for job in campaign.stage_jobs(stage_id)]
        if stage_id == 3:
            jobs = [
                replace(
                    job,
                    job_id=job.job_id.replace(
                        "id_frequency_and_oov", "id_frequency_log1p"
                    ),
                    setting_id="id_frequency_log1p",
                    components=("entity_frequency_log1p",),
                )
                if job.setting_id == "id_frequency_and_oov"
                else job
                for job in jobs
            ]
            cat_base = _catboost_template(campaign)
            candidates = (
                ("pitcher_smoothing_k100", ("pitcher_smooth_k100",)),
                ("batter_smoothing_k250", ("batter_smooth_k250",)),
                ("id_frequency_and_oov", ("entity_frequency_and_oov",)),
                ("hand_matchup", ("hand_matchup",)),
            )
            jobs.extend(
                replace(
                    cat_base,
                    job_id=f"s3__catboost__{setting}__tr2023__va2024__s42",
                    stage_id=3,
                    setting_id=setting,
                    components=components,
                    max_seconds=500,
                )
                for setting, components in candidates
            )
        return tuple(jobs)
    if stage_id == 4:
        return _stage_four_jobs(campaign, state)
    if stage_id == 5:
        return _stage_five_jobs(campaign, state)
    raise ValueError("stage_id must be between 1 and 5")


def _curve_best(metric: Mapping[str, object], common_epochs: int) -> float:
    values = []
    for item in metric.get("validation_curve", []):
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and int(item[0]) < common_epochs
        ):
            values.append(float(item[1]))
    if not values:
        raise DecisionError("validation curve has no common epoch point")
    return min(values)


def _time_curve_best(
    metric: Mapping[str, object], common_seconds: float
) -> tuple[float, int]:
    values = []
    for item in metric.get("validation_time_curve", []):
        if (
            isinstance(item, (list, tuple))
            and len(item) == 3
            and float(item[1]) <= common_seconds
        ):
            values.append((float(item[2]), int(item[0])))
    if not values:
        raise DecisionError("validation time curve has no common elapsed-time point")
    return min(values)


def finalize_stage_five(
    *,
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> StageFiveDecision:
    baseline_epochs = int(baseline.get("completed_epochs", 0))
    candidate_epochs = int(candidate.get("completed_epochs", 0))
    common = min(baseline_epochs, candidate_epochs)
    if common < 10:
        return StageFiveDecision(
            "inconclusive", "minimum_ten_common_epochs_not_met", common
        )
    try:
        baseline_best = _curve_best(baseline, common)
        candidate_best = _curve_best(candidate, common)
    except (DecisionError, TypeError, ValueError):
        return StageFiveDecision("inconclusive", "common_curve_invalid", common)
    if candidate_best < baseline_best:
        comparable = (
            int(baseline.get("best_epoch", -1)) < common
            and int(candidate.get("best_epoch", -1)) < common
        )
        return StageFiveDecision(
            "recommended",
            "candidate_improved_capped_brier",
            common,
            True,
            baseline_best,
            candidate_best,
            comparable,
        )
    return StageFiveDecision(
        "not_recommended",
        "candidate_did_not_improve_capped_brier",
        common,
        True,
        baseline_best,
        candidate_best,
        (
            int(baseline.get("best_epoch", -1)) < common
            and int(candidate.get("best_epoch", -1)) < common
        ),
    )


def _metrics(root: Path, job_id: str) -> dict[str, object]:
    path = root / "jobs" / job_id / "metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StageNeedsReview(f"metrics are invalid for {job_id}")
    return payload


def _metrics_or_none(root: Path, job_id: str) -> dict[str, object] | None:
    path = root / "jobs" / job_id / "metrics.json"
    return _metrics(root, job_id) if path.is_file() else None


def _valid_dl_metric(metric: Mapping[str, object]) -> bool:
    return (
        int(metric.get("completed_epochs", 0)) >= 10
        and int(metric.get("validation_points", 0)) >= 3
    )


def _valid_catboost_metric(metric: Mapping[str, object]) -> bool:
    return (
        int(metric.get("best_epoch", -1)) >= 0
        and int(metric.get("validation_points", 0)) > 0
    )


def _predictions(root: Path, job_id: str) -> pd.DataFrame:
    return pd.read_csv(root / "jobs" / job_id / "predictions.csv")


def _blend_gain(
    root: Path,
    catboost_id: str,
    candidate_id: str,
    weights: Sequence[float],
) -> float:
    cat = _predictions(root, catboost_id)
    candidate = _predictions(root, candidate_id)
    merged = candidate.merge(
        cat[["row_id", "target", "probability"]],
        on=["row_id", "target"],
        how="inner",
        validate="one_to_one",
        suffixes=("_candidate", "_catboost"),
    )
    if len(merged) != len(candidate) or len(merged) != len(cat):
        raise StageNeedsReview("blend predictions are not row-aligned")
    rows = fixed_blend_metrics(
        row_id=merged["row_id"].to_numpy(),
        target=merged["target"].to_numpy(),
        catboost_row_id=merged["row_id"].to_numpy(),
        catboost_probability=merged["probability_catboost"].to_numpy(),
        candidate_probability=merged["probability_candidate"].to_numpy(),
        weights=weights,
    )
    return max(float(row["gain"]) for row in rows)


def _oov_delta(root: Path, baseline_id: str, candidate_id: str) -> float:
    base = _predictions(root, baseline_id)
    candidate = _predictions(root, candidate_id)
    merged = candidate.merge(
        base[["row_id", "probability"]],
        on="row_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_candidate", "_baseline"),
    )
    deltas = []
    for column in ("pitcher_oov", "batter_oov"):
        subset = merged.loc[pd.to_numeric(merged[column], errors="coerce").eq(1)]
        if subset.empty:
            continue
        target = subset["target"].to_numpy(dtype="float64")
        candidate_brier = np.mean(
            np.square(subset["probability_candidate"].to_numpy() - target)
        )
        baseline_brier = np.mean(
            np.square(subset["probability_baseline"].to_numpy() - target)
        )
        deltas.append(float(candidate_brier - baseline_brier))
    return min(deltas, default=0.0)


def _final_segment_deltas(
    root: Path, baseline_id: str, candidate_id: str
) -> dict[str, float]:
    base = _predictions(root, baseline_id)
    candidate = _predictions(root, candidate_id)
    merged = candidate.merge(
        base[["row_id", "probability"]],
        on="row_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_candidate", "_baseline"),
    )
    if len(merged) != len(base) or len(merged) != len(candidate):
        raise StageNeedsReview("final predictions are not row-aligned")

    def worst_delta(groups: Sequence[pd.DataFrame]) -> float:
        deltas = []
        for subset in groups:
            if subset.empty:
                continue
            target = subset["target"].to_numpy(dtype="float64")
            candidate_brier = np.mean(
                np.square(subset["probability_candidate"].to_numpy() - target)
            )
            baseline_brier = np.mean(
                np.square(subset["probability_baseline"].to_numpy() - target)
            )
            deltas.append(float(candidate_brier - baseline_brier))
        return max(deltas, default=0.0)

    return {
        "pitcher_oov": worst_delta(
            [merged.loc[pd.to_numeric(merged["pitcher_oov"], errors="coerce").eq(1)]]
        ),
        "batter_oov": worst_delta(
            [merged.loc[pd.to_numeric(merged["batter_oov"], errors="coerce").eq(1)]]
        ),
        "game_type": worst_delta(
            [subset for _, subset in merged.groupby("game_type", dropna=False)]
        ),
    }


def _descriptor(job: BudgetedJob, delta: float) -> dict[str, object]:
    return {
        "setting_id": job.setting_id,
        "preprocessing_profile": job.preprocessing_profile,
        "components": list(job.components),
        "delta": delta,
    }


def evaluate_stage(
    campaign: BudgetedCampaign,
    stage_id: int,
    jobs: Sequence[BudgetedJob],
    root: str | Path,
    previous_state: Mapping[str, object],
) -> dict[str, object]:
    campaign_root = Path(root)
    state = dict(previous_state)
    if stage_id == 1:
        cat_job = next(job for job in jobs if job.family == "catboost")
        cat_metric = _metrics(campaign_root, cat_job.job_id)
        if not _valid_catboost_metric(cat_metric):
            raise StageNeedsReview("CatBoost baseline lacks minimum valid evidence")
        rows = []
        candidate_jobs = []
        for job in jobs:
            metric = _metrics(campaign_root, job.job_id)
            if job.family == "catboost":
                continue
            rows.append(
                {
                    **metric,
                    "candidate_id": job.job_id,
                    "family": job.family,
                    "hashes_valid": True,
                }
            )
            candidate_jobs.append(job)
        try:
            common_seconds = min(
                float(row["validation_time_curve"][-1][1])
                for row in rows
                if _valid_dl_metric(row)
            )
            common_rows = []
            common_best_epochs: dict[str, int] = {}
            for row in rows:
                if not _valid_dl_metric(row):
                    common_rows.append(row)
                    continue
                common_brier, common_best_epoch = _time_curve_best(
                    row, common_seconds
                )
                common_best_epochs[str(row["candidate_id"])] = common_best_epoch
                common_rows.append(
                    {**row, "brier": common_brier, "comparison_seconds": common_seconds}
                )
            rows = common_rows
        except (DecisionError, KeyError, TypeError, ValueError) as error:
            raise StageNeedsReview(
                "DL elapsed-time comparison evidence is incomplete"
            ) from error
        blends = []
        for job, row in zip(candidate_jobs, rows, strict=True):
            best_epoch = int(row.get("best_epoch", -1))
            comparable = best_epoch == common_best_epochs.get(job.job_id)
            blends.append(
                {
                    "candidate_id": job.job_id,
                    "blend_gain": _blend_gain(
                        campaign_root, cat_job.job_id, job.job_id, campaign.blend_weights
                    ) if comparable else 0.0,
                }
            )
        try:
            decision = choose_dl_representative(rows, blends)
        except DecisionError as error:
            raise StageNeedsReview(str(error)) from error
        selected_job = next(
            job for job in jobs if job.job_id == decision.selected_candidate_id
        )
        state["selected_model"] = {
            "candidate_id": selected_job.job_id,
            "family": selected_job.family,
            "profile_id": selected_job.profile_id,
            "model": dict(selected_job.model),
            "training": dict(selected_job.training),
            "reason": decision.reason,
        }
        tabnet_job = next(job for job in jobs if job.family == "tabnet")
        tabnet_row = next(row for row in rows if row["family"] == "tabnet")
        tabnet_blend = next(
            float(row["blend_gain"])
            for row in blends
            if row["candidate_id"] == tabnet_job.job_id
        )
        eligible_rows = [row for row in rows if _valid_dl_metric(row)]
        common_epochs = min(
            (int(row["completed_epochs"]) for row in eligible_rows), default=0
        )
        other_rows = [row for row in eligible_rows if row["family"] != "tabnet"]
        if _valid_dl_metric(tabnet_row) and common_epochs >= 10 and other_rows:
            tabnet_common, tabnet_common_epoch = _time_curve_best(
                tabnet_row, common_seconds
            )
            other_common = [
                (*_time_curve_best(row, common_seconds), row) for row in other_rows
            ]
            best_other_brier, best_other_epoch, best_other_row = min(
                other_common, key=lambda item: (item[0], str(item[2]["candidate_id"]))
            )
            comparable_tabnet = {**tabnet_row, "brier": tabnet_common}
            predictions_comparable = (
                int(tabnet_row.get("best_epoch", -1)) == tabnet_common_epoch
                and int(best_other_row.get("best_epoch", -1)) == best_other_epoch
            )
            oov_gain = (
                -_oov_delta(
                    campaign_root,
                    str(best_other_row["candidate_id"]),
                    tabnet_job.job_id,
                )
                if predictions_comparable
                else 0.0
            )
            tabnet_decision = decide_tabnet(
                comparable_tabnet,
                best_dl_brier=best_other_brier,
                blend_gain=tabnet_blend,
                oov_gain=oov_gain,
                overall_delta=tabnet_common - best_other_brier,
            )
            state["tabnet_common_epochs"] = common_epochs
            state["tabnet_common_seconds"] = common_seconds
            state["tabnet_common_brier"] = tabnet_common
            state["tabnet_reference_brier"] = best_other_brier
            state["tabnet_oov_gain"] = oov_gain
            state["tabnet_segment_predictions_comparable"] = predictions_comparable
        else:
            tabnet_decision = decide_tabnet(
                tabnet_row,
                best_dl_brier=min(float(row["brier"]) for row in other_rows)
                if other_rows
                else float("inf"),
                blend_gain=tabnet_blend,
                oov_gain=0.0,
                overall_delta=0.0,
            )
        state["tabnet_decision"] = tabnet_decision.__dict__
    elif stage_id in {2, 3}:
        selected = _selected_model(state)
        family = str(selected["family"])
        all_jobs = []
        manifest = json.loads(
            (campaign_root / "campaign_manifest.json").read_text(encoding="utf-8")
        )
        for entry in manifest["jobs"].values():
            raw = entry.get("job")
            if isinstance(raw, dict):
                all_jobs.append(_job_from_payload(raw))
        dl_baseline = next(
            job
            for job in all_jobs
            if job.family == family and job.setting_id == "dl_standard" and job.stage_id == 2
        )
        cat_baseline = next(
            job
            for job in all_jobs
            if job.family == "catboost" and job.setting_id == "tree_native"
        )
        dl_candidates = [
            job
            for job in all_jobs
            if job.family == family
            and job.stage_id in {2, 3}
            and job.setting_id != "dl_standard"
        ]
        promoted_dl = []
        baseline_metric = _metrics(campaign_root, dl_baseline.job_id)
        if not _valid_dl_metric(baseline_metric):
            raise StageNeedsReview("selected DL baseline lacks minimum valid evidence")
        base_brier = float(baseline_metric["brier"])
        for job in dl_candidates:
            candidate_metric = _metrics(campaign_root, job.job_id)
            if not _valid_dl_metric(candidate_metric):
                continue
            delta = float(candidate_metric["brier"]) - base_brier
            blend_gain = _blend_gain(
                campaign_root, cat_baseline.job_id, job.job_id, campaign.blend_weights
            )
            if promote_single(
                delta=delta,
                oov_delta=_oov_delta(campaign_root, dl_baseline.job_id, job.job_id),
                blend_gain=blend_gain,
            ):
                promoted_dl.append(_descriptor(job, delta))
        state["promoted_dl"] = sorted(
            promoted_dl, key=lambda item: (float(item["delta"]), str(item["setting_id"]))
        )[:2]
        if stage_id == 3:
            cat_brier = float(_metrics(campaign_root, cat_baseline.job_id)["brier"])
            promoted_cat = []
            for job in jobs:
                if job.family != "catboost":
                    continue
                metric = _metrics_or_none(campaign_root, job.job_id)
                if metric is None or not _valid_catboost_metric(metric):
                    continue
                delta = float(metric["brier"]) - cat_brier
                if delta <= -0.0001:
                    promoted_cat.append(_descriptor(job, delta))
            state["promoted_catboost"] = sorted(
                promoted_cat,
                key=lambda item: (float(item["delta"]), str(item["setting_id"])),
            )[:2]
    elif stage_id == 4:
        stability_jobs = [
            job
            for job in jobs
            if job.family != "catboost" and job.valid_year == 2023
        ]
        baseline = next(
            job for job in stability_jobs if job.setting_id == "dl_standard"
        )
        baseline_metric_2023 = _metrics(campaign_root, baseline.job_id)
        if not _valid_dl_metric(baseline_metric_2023):
            raise StageNeedsReview("DL stability baseline lacks minimum valid evidence")
        baseline_brier_2023 = float(baseline_metric_2023["brier"])
        manifest = json.loads(
            (campaign_root / "campaign_manifest.json").read_text(encoding="utf-8")
        )
        all_jobs = [
            _job_from_payload(entry["job"])
            for entry in manifest["jobs"].values()
            if isinstance(entry, dict) and isinstance(entry.get("job"), dict)
        ]
        baseline_2024 = next(
            job
            for job in all_jobs
            if job.family == baseline.family
            and job.setting_id == "dl_standard"
            and job.valid_year == 2024
            and job.sample_mode == "proxy"
        )
        baseline_metric_2024 = _metrics(campaign_root, baseline_2024.job_id)
        if not _valid_dl_metric(baseline_metric_2024):
            raise StageNeedsReview("DL primary baseline lacks minimum valid evidence")
        baseline_brier_2024 = float(baseline_metric_2024["brier"])
        improving = []
        for job in stability_jobs:
            if job.setting_id == "dl_standard":
                continue
            primary = next(
                (
                    item
                    for item in all_jobs
                    if item.family == job.family
                    and item.setting_id == job.setting_id
                    and item.valid_year == 2024
                    and item.sample_mode == "proxy"
                ),
                None,
            )
            if primary is None:
                continue
            metric_2023 = _metrics(campaign_root, job.job_id)
            metric_2024 = _metrics(campaign_root, primary.job_id)
            if not (
                _valid_dl_metric(metric_2023) and _valid_dl_metric(metric_2024)
            ):
                continue
            delta_2023 = float(metric_2023["brier"]) - baseline_brier_2023
            delta_2024 = float(metric_2024["brier"]) - baseline_brier_2024
            total_rows = int(metric_2023["valid_rows"]) + int(metric_2024["valid_rows"])
            weighted = (
                delta_2023 * int(metric_2023["valid_rows"])
                + delta_2024 * int(metric_2024["valid_rows"])
            ) / total_rows
            if delta_2023 < 0 and delta_2024 < 0 and weighted <= -0.0001:
                improving.append((weighted, job, delta_2023, delta_2024, metric_2023, metric_2024))
        improving.sort(key=lambda item: (item[0], item[1].setting_id))
        if improving:
            weighted, job, delta_2023, delta_2024, metric_2023, metric_2024 = improving[0]
            state["final_dl"] = {
                **_descriptor(job, weighted),
                "proxy_deltas": {"2023": delta_2023, "2024": delta_2024},
                "proxy_rows": {
                    "2023": int(metric_2023["valid_rows"]),
                    "2024": int(metric_2024["valid_rows"]),
                },
            }
        else:
            state["final_dl"] = None
            state["preprocessing_status"] = "not_recommended"

        cat_jobs = [job for job in jobs if job.family == "catboost"]
        all_cat_jobs = [job for job in all_jobs if job.family == "catboost"]

        def find_cat(setting: str, valid_year: int, sample_mode: str) -> BudgetedJob:
            matches = [
                job
                for job in all_cat_jobs
                if job.setting_id == setting
                and job.valid_year == valid_year
                and job.sample_mode == sample_mode
            ]
            if len(matches) != 1:
                raise StageNeedsReview(
                    f"CatBoost evidence is ambiguous: {setting}/{valid_year}/{sample_mode}"
                )
            return matches[0]

        baseline_cat = {
            (2023, "proxy"): find_cat("tree_native", 2023, "proxy"),
            (2024, "proxy"): find_cat("tree_native", 2024, "proxy"),
            (2024, "full"): find_cat("tree_native", 2024, "full"),
        }
        cat_statuses: dict[str, str] = {}
        candidate_settings = sorted(
            {
                job.setting_id
                for job in cat_jobs
                if job.setting_id != "tree_native"
            }
        )
        for setting in candidate_settings:
            evidence = {
                key: find_cat(setting, *key) for key in baseline_cat
            }
            metrics = {
                key: _metrics_or_none(campaign_root, job.job_id)
                for key, job in evidence.items()
            }
            baseline_metrics = {
                key: _metrics_or_none(campaign_root, job.job_id)
                for key, job in baseline_cat.items()
            }
            if not all(
                metric is not None and _valid_catboost_metric(metric)
                for metric in [*metrics.values(), *baseline_metrics.values()]
            ):
                cat_statuses[setting] = "inconclusive"
                continue
            cat_statuses[setting] = final_preprocessing_status(
                proxy_deltas={
                    year: float(metrics[(year, "proxy")]["brier"])
                    - float(baseline_metrics[(year, "proxy")]["brier"])
                    for year in (2023, 2024)
                },
                proxy_rows={
                    year: int(metrics[(year, "proxy")]["valid_rows"])
                    for year in (2023, 2024)
                },
                full_2024_delta=(
                    float(metrics[(2024, "full")]["brier"])
                    - float(baseline_metrics[(2024, "full")]["brier"])
                ),
                segment_deltas=_final_segment_deltas(
                    campaign_root,
                    baseline_cat[(2024, "full")].job_id,
                    evidence[(2024, "full")].job_id,
                ),
                hashes_valid=True,
                cross_family_confirmed=False,
            )
        state["catboost_preprocessing_status"] = cat_statuses
    elif stage_id == 5:
        if len(jobs) != 2:
            state["preprocessing_status"] = "not_recommended"
        else:
            decision = finalize_stage_five(
                baseline=_metrics(campaign_root, jobs[0].job_id),
                candidate=_metrics(campaign_root, jobs[1].job_id),
            )
            if (
                decision.status == "recommended"
                and decision.predictions_comparable
                and isinstance(
                state.get("final_dl"), dict
                )
            ):
                final_dl = state["final_dl"]
                state["preprocessing_status"] = final_preprocessing_status(
                    proxy_deltas={
                        int(year): float(value)
                        for year, value in final_dl["proxy_deltas"].items()
                    },
                    proxy_rows={
                        int(year): int(value)
                        for year, value in final_dl["proxy_rows"].items()
                    },
                    full_2024_delta=(
                        float(decision.candidate_best_brier)
                        - float(decision.baseline_best_brier)
                    ),
                    segment_deltas=_final_segment_deltas(
                        campaign_root, jobs[0].job_id, jobs[1].job_id
                    ),
                    hashes_valid=True,
                )
            else:
                state["preprocessing_status"] = (
                    "inconclusive"
                    if decision.status == "recommended"
                    and not decision.predictions_comparable
                    else decision.status
                )
            state["stage_five_reason"] = decision.reason
            state["common_epochs"] = decision.common_epochs
    state["completed_stage"] = stage_id
    return state


def _job_from_payload(payload: Mapping[str, object]) -> BudgetedJob:
    return BudgetedJob(
        job_id=str(payload["job_id"]),
        stage_id=int(payload["stage_id"]),
        family=str(payload["family"]),
        profile_id=str(payload["profile_id"]),
        setting_id=str(payload["setting_id"]),
        preprocessing_profile=str(payload["preprocessing_profile"]),
        components=tuple(str(item) for item in payload["components"]),
        model=MappingProxyType(dict(payload["model"])),
        training=MappingProxyType(dict(payload["training"])),
        train_end_year=int(payload["train_end_year"]),
        valid_year=int(payload["valid_year"]),
        seed=int(payload["seed"]),
        max_seconds=int(payload["max_seconds"]),
        sample_mode=str(payload.get("sample_mode", "proxy")),
    )


def _safe_restore(selection: ResumeSelection, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if selection.path.is_dir():
        for source in selection.path.rglob("*"):
            if not source.is_file() or source.name == "resume_metadata.json":
                continue
            relative = source.relative_to(selection.path)
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(source.read_bytes())
            os.replace(temporary, destination)
        return
    with ZipFile(selection.path) as archive:
        for info in archive.infolist():
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise StageNeedsReview("resume bundle contains an unsafe path")
            destination = output_root.joinpath(*relative.parts)
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(archive.read(info))
            os.replace(temporary, destination)


def run_auto(
    *,
    config: str | Path,
    input_root: str | Path,
    data_dir: str | Path,
    output_root: str | Path,
    session_started_unix: float | None = None,
    scheduler: BudgetedScheduler | None = None,
) -> AutoResult:
    campaign = load_budgeted_campaign(config)
    config_path = Path(config).resolve()
    data_path = Path(data_dir).resolve()
    identity = _campaign_identity(config_path, data_path)
    root = Path(output_root).resolve()
    selection = inspect_resume_bundles(
        input_root, campaign_id=campaign.campaign_id
    )
    if selection is None:
        print("RESUME_NOT_FOUND starting_fresh_campaign=true", flush=True)
    else:
        print(
            f"RESUME_SELECTED path={selection.path} "
            f"completed_stage={selection.completed_stage}",
            flush=True,
        )
    if selection is not None and not (root / "campaign_manifest.json").is_file():
        _safe_restore(selection, root)
    manifest_path = root / "campaign_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("identity") != identity:
            raise StageNeedsReview(
                "restored campaign identity differs from current data, config, or code"
            )
    state_path = root / "stage_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {}
    completed_stage = int(state.get("completed_stage", selection.completed_stage if selection else 0))
    if completed_stage >= 5:
        return AutoResult(5, "CAMPAIGN_COMPLETE", 5)
    stage_id = completed_stage + 1
    jobs = build_stage_jobs(campaign, stage_id, state)
    started = time.time() if session_started_unix is None else session_started_unix
    deadline = started + campaign.session_seconds
    print(
        f"STAGE_START stage={stage_id} jobs={len(jobs)} deadline_unix={deadline:.0f}",
        flush=True,
    )
    print(f"DATA_READY data_dir={Path(data_dir).resolve()}", flush=True)
    if jobs:
        runner = scheduler or BudgetedScheduler(
            stop_new_jobs_seconds=campaign.stop_new_jobs_seconds,
            archive_reserve_seconds=campaign.archive_reserve_seconds,
            campaign_identity=identity,
            required_gpu_name="T4",
        )
        try:
            summary = runner.run(jobs, root, deadline=deadline)
        except RuntimeError as error:
            if "devices are required" in str(error):
                raise StageNeedsReview(str(error)) from error
            raise
    else:
        summary = SchedulerSummary((), (), ())
    if summary.pending or summary.failed:
        _atomic_json(
            state_path,
            {
                **state,
                "completed_stage": completed_stage,
                "incomplete_stage": stage_id,
                "pending_jobs": list(summary.pending),
                "failed_jobs": list(summary.failed),
            },
        )
        terminal = "STAGE_INCOMPLETE_RESUME_SAME_STAGE"
    else:
        state = evaluate_stage(campaign, stage_id, jobs, root, state)
        _atomic_json(state_path, state)
        completed_stage = stage_id
        terminal = (
            "CAMPAIGN_COMPLETE"
            if stage_id == 5
            else "STAGE_COMPLETE_READY_FOR_NEXT"
        )
    from .budgeted_artifacts import write_stage_bundles

    bundles = write_stage_bundles(
        campaign_root=root,
        stage_id=completed_stage,
        campaign_id=campaign.campaign_id,
    )
    print(
        f"STAGE_SUMMARY stage={stage_id} completed_stage={completed_stage} "
        f"completed_jobs={len(summary.completed)} pending_jobs={len(summary.pending)} "
        f"failed_jobs={len(summary.failed)}",
        flush=True,
    )
    print(terminal, flush=True)
    return AutoResult(
        stage_id, terminal, completed_stage, bundles.resume, bundles.review
    )


def run_worker(
    *,
    job_json: str | Path,
    output_dir: str | Path,
    data_dir: str | Path,
    cache_root: str | Path,
    deadline_unix: float,
) -> None:
    from .budgeted_runtime import BudgetedRuntime

    job = _job_from_payload(json.loads(Path(job_json).read_text(encoding="utf-8")))
    output = Path(output_dir).resolve()
    import torch

    device_count = int(torch.cuda.device_count())
    device_names = [str(torch.cuda.get_device_name(index)) for index in range(device_count)]
    if device_count != 1 or "t4" not in device_names[0].casefold():
        raise StageNeedsReview(
            f"worker must see exactly one T4 GPU; count={device_count} names={device_names}"
        )
    os.environ["PREPROCESSING_SESSION_DEADLINE_UNIX"] = str(deadline_unix)
    runtime = BudgetedRuntime(data_dir, cache_root=cache_root)
    result = runtime.run_job(job, output)
    artifacts = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name in {"job.json", "worker_result.json"}:
            continue
        artifacts.append(
            {
                "path": path.relative_to(output).as_posix(),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
        )
    _atomic_json(
        output / "worker_result.json",
        {
            "schema_version": 1,
            "job_id": job.job_id,
            "state": "completed",
            "artifacts": artifacts,
            "worker_gpu": {"device_count": device_count, "devices": device_names},
        },
    )
    if result.metrics_path.is_file():
        print(f"CHECKPOINT_SAVED job={job.job_id}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    auto = subparsers.add_parser("auto")
    auto.add_argument("--config", required=True)
    auto.add_argument("--input-root", required=True)
    auto.add_argument("--data-dir", required=True)
    auto.add_argument("--output-root", required=True)
    auto.add_argument("--session-started-unix", required=True, type=float)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--job-json", required=True)
    worker.add_argument("--output-dir", required=True)
    worker.add_argument("--deadline-unix", required=True, type=float)
    worker.add_argument("--data-dir", default=os.environ.get("PREPROCESSING_DATA_DIR"))
    worker.add_argument("--cache-root", default=os.environ.get("PREPROCESSING_CACHE_ROOT"))
    status = subparsers.add_parser("status")
    status.add_argument("--output-root")
    status.add_argument("--config")
    status.add_argument("--dry-contract", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "auto":
        os.environ["PREPROCESSING_DATA_DIR"] = str(Path(args.data_dir).resolve())
        cache_root = Path(args.output_root).resolve() / "cache"
        os.environ["PREPROCESSING_CACHE_ROOT"] = str(cache_root)
        run_auto(
            config=args.config,
            input_root=args.input_root,
            data_dir=args.data_dir,
            output_root=args.output_root,
            session_started_unix=args.session_started_unix,
        )
        return 0
    if args.action == "worker":
        if not args.data_dir or not args.cache_root:
            raise ValueError("worker data and cache roots are required")
        run_worker(
            job_json=args.job_json,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            cache_root=args.cache_root,
            deadline_unix=args.deadline_unix,
        )
        return 0
    if args.dry_contract:
        if not args.config:
            raise ValueError("--config is required with --dry-contract")
        campaign = load_budgeted_campaign(args.config)
        print(
            json.dumps(
                {
                    "archive_reserve_seconds": campaign.archive_reserve_seconds,
                    "campaign_id": campaign.campaign_id,
                    "gpu_workers": 2,
                    "session_seconds": campaign.session_seconds,
                    "stage_count": 5,
                    "stop_new_jobs_seconds": campaign.stop_new_jobs_seconds,
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.output_root:
        raise ValueError("--output-root is required unless --dry-contract is used")
    root = Path(args.output_root)
    state_path = root / "stage_state.json"
    print(
        state_path.read_text(encoding="utf-8")
        if state_path.is_file()
        else "NO_STAGE_STATE"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StageNeedsReview as error:
        print(f"STAGE_NEEDS_REVIEW {error}", flush=True)
        raise SystemExit(2) from error
    except Exception as error:
        print(
            f"STAGE_ERROR type={type(error).__name__} message={str(error).replace(' ', '_')}",
            flush=True,
        )
        raise
