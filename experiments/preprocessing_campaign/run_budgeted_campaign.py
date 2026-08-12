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


def _read_last_member(archive: ZipFile, name: str) -> bytes:
    matches = [info for info in archive.infolist() if info.filename == name]
    if not matches:
        raise KeyError(name)
    return archive.read(matches[-1])


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
                metadata = json.loads(
                    _read_last_member(archive, "resume_metadata.json").decode("utf-8")
                )
                manifest = _read_last_member(archive, "campaign_manifest.json")
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
        *dl_candidates,
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
            )
        )
    cat_base = _catboost_template(campaign)
    cat_descriptors = [
        {
            "setting_id": "tree_native",
            "preprocessing_profile": "tree_native",
            "components": (),
        },
        *cat_candidates,
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
    for descriptor in cat_descriptors:
        for train_end, valid, sample_mode in ((2022, 2023, "proxy"), (2023, 2024, "full")):
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
        return StageFiveDecision("recommended", "candidate_improved_capped_brier", common)
    return StageFiveDecision(
        "not_recommended", "candidate_did_not_improve_capped_brier", common
    )


def _metrics(root: Path, job_id: str) -> dict[str, object]:
    path = root / "jobs" / job_id / "metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StageNeedsReview(f"metrics are invalid for {job_id}")
    return payload


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
        rows = []
        blends = []
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
            blends.append(
                {
                    "candidate_id": job.job_id,
                    "blend_gain": _blend_gain(
                        campaign_root, cat_job.job_id, job.job_id, campaign.blend_weights
                    ),
                }
            )
        decision = choose_dl_representative(rows, blends)
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
        best_brier = min(float(row["brier"]) for row in rows)
        state["tabnet_decision"] = decide_tabnet(
            tabnet_row,
            best_dl_brier=best_brier,
            blend_gain=tabnet_blend,
            oov_gain=0.0,
            overall_delta=float(tabnet_row["brier"]) - best_brier,
        ).__dict__
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
        base_brier = float(_metrics(campaign_root, dl_baseline.job_id)["brier"])
        for job in dl_candidates:
            delta = float(_metrics(campaign_root, job.job_id)["brier"]) - base_brier
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
                delta = float(_metrics(campaign_root, job.job_id)["brier"]) - cat_brier
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
    elif stage_id == 5:
        if len(jobs) != 2:
            state["preprocessing_status"] = "not_recommended"
        else:
            decision = finalize_stage_five(
                baseline=_metrics(campaign_root, jobs[0].job_id),
                candidate=_metrics(campaign_root, jobs[1].job_id),
            )
            if decision.status == "recommended" and isinstance(
                state.get("final_dl"), dict
            ):
                baseline_metric = _metrics(campaign_root, jobs[0].job_id)
                candidate_metric = _metrics(campaign_root, jobs[1].job_id)
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
                        float(candidate_metric["brier"])
                        - float(baseline_metric["brier"])
                    ),
                    segment_deltas=_final_segment_deltas(
                        campaign_root, jobs[0].job_id, jobs[1].job_id
                    ),
                    hashes_valid=True,
                )
            else:
                state["preprocessing_status"] = decision.status
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
    root = Path(output_root).resolve()
    selection = inspect_resume_bundles(
        input_root, campaign_id=campaign.campaign_id
    )
    if selection is not None and not (root / "campaign_manifest.json").is_file():
        _safe_restore(selection, root)
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
        )
        summary = runner.run(jobs, root, deadline=deadline)
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
    status.add_argument("--output-root", required=True)
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
