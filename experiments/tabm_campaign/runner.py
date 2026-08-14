from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping, Protocol
from zipfile import ZipFile

from .artifacts import BundlePaths, StageEvidence, verify_resume_bundle, write_stage_bundles
from .contracts import Campaign, Candidate, load_campaign
from .decisions import (
    CandidateScore,
    TemporalEvidence,
    choose_temporal_champion,
    choose_version_a_survivors,
    ensemble_verdict,
    temporal_verdict,
)


VERSION_B_REFERENCE_CANDIDATE_ID = "a__p2__piecewise_linear__bce__plateau__s42"
VERSION_C_TEMPORAL_SENTINEL_CANDIDATE_ID = (
    "a__p2__piecewise_linear__bce__one_cycle__s42"
)


class CampaignRunnerError(RuntimeError):
    """Raised when a stage cannot be advanced from trusted evidence."""


@dataclass(frozen=True)
class CampaignJob:
    candidate_id: str
    capacity: str
    k: int
    width: int
    blocks: int
    dropout: float
    num_embedding: str
    loss: str
    scheduler: str
    learning_rate: float
    seed: int
    train_end_year: int
    valid_year: int
    sample_mode: str
    max_epochs: int
    min_epochs: int
    patience: int


@dataclass(frozen=True)
class CampaignJobResult:
    candidate_id: str
    status: str
    brier: float | None
    best_epoch: int | None
    completed_epochs: int
    checkpoint: Path | None
    predictions_path: Path | None
    resource_evidence: Mapping[str, object]
    failure: str | None


@dataclass(frozen=True)
class StageRunResult:
    version: str
    bundles: BundlePaths
    completed: tuple[str, ...]
    failed: tuple[str, ...]
    inconclusive: tuple[str, ...]


class CampaignRuntime(Protocol):
    def run_jobs(
        self,
        version: str,
        jobs: tuple[CampaignJob, ...],
        output_dir: Path,
        *,
        gpu_count: int,
        job_deadline: float,
    ) -> tuple[CampaignJobResult, ...]: ...


_DEFAULT_CONFIG = Path(__file__).with_name("configs") / "champion_v1.json"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _config_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _candidate_payload(candidate: Candidate) -> dict[str, object]:
    return asdict(candidate)


def _job_from_candidate(
    candidate: Candidate,
    *,
    candidate_id: str | None = None,
    train_end_year: int = 2023,
    valid_year: int = 2024,
    sample_mode: str = "proxy",
    max_epochs: int = 8,
    min_epochs: int = 3,
    patience: int = 3,
) -> CampaignJob:
    return CampaignJob(
        candidate_id or candidate.candidate_id,
        candidate.capacity,
        candidate.k,
        candidate.width,
        candidate.blocks,
        candidate.dropout,
        candidate.num_embedding,
        candidate.loss,
        candidate.scheduler,
        candidate.learning_rate,
        candidate.seed,
        train_end_year,
        valid_year,
        sample_mode,
        max_epochs,
        min_epochs,
        patience,
    )


def _candidate_from_payload(payload: Mapping[str, object]) -> Candidate:
    return Candidate(
        candidate_id=str(payload["candidate_id"]),
        family="tabm",
        capacity=str(payload["capacity"]),
        k=int(payload["k"]),
        width=int(payload["width"]),
        blocks=int(payload["blocks"]),
        dropout=float(payload["dropout"]),
        num_embedding=str(payload["num_embedding"]),
        loss=str(payload["loss"]),
        scheduler=str(payload["scheduler"]),
        learning_rate=float(payload["learning_rate"]),
        seed=int(payload["seed"]),
    )


def _result_payload(result: CampaignJobResult) -> dict[str, object]:
    return {
        "candidate_id": result.candidate_id,
        "status": result.status,
        "brier": result.brier,
        "best_epoch": result.best_epoch,
        "completed_epochs": result.completed_epochs,
        "checkpoint": None if result.checkpoint is None else result.checkpoint.name,
        "predictions": None if result.predictions_path is None else result.predictions_path.name,
        "resource_evidence": dict(result.resource_evidence),
        "failure": result.failure,
    }


def _result_from_payload(payload: Mapping[str, object]) -> CampaignJobResult:
    return CampaignJobResult(
        candidate_id=str(payload["candidate_id"]),
        status=str(payload["status"]),
        brier=None if payload.get("brier") is None else float(payload["brier"]),
        best_epoch=None if payload.get("best_epoch") is None else int(payload["best_epoch"]),
        completed_epochs=int(payload.get("completed_epochs", 0)),
        checkpoint=None if payload.get("checkpoint") is None else Path(str(payload["checkpoint"])),
        predictions_path=None if payload.get("predictions") is None else Path(str(payload["predictions"])),
        resource_evidence=dict(payload.get("resource_evidence", {})),
        failure=None if payload.get("failure") is None else str(payload["failure"]),
    )


def _prior_completed(prior: dict[str, object]) -> dict[str, CampaignJobResult]:
    rows = prior.get("results", ())
    if not isinstance(rows, list):
        return {}
    parsed = [_result_from_payload(row) for row in rows if isinstance(row, dict)]
    return {row.candidate_id: row for row in parsed if row.status == "completed"}


def _run_with_completed_reuse(
    runtime: CampaignRuntime,
    version: str,
    jobs: tuple[CampaignJob, ...],
    output_dir: Path,
    deadline: float,
    prior: dict[str, object],
    gpu_count: int,
) -> tuple[CampaignJobResult, ...]:
    completed = _prior_completed(prior)
    pending = tuple(job for job in jobs if job.candidate_id not in completed)
    prior_rows = {
        str(row.get("candidate_id")): row
        for row in prior.get("results", [])
        if isinstance(row, dict)
    }
    for job in pending:
        training_dir = prior_rows.get(job.candidate_id, {}).get("training_dir")
        if training_dir is None:
            continue
        source = Path(str(training_dir))
        if source.is_dir():
            target = output_dir / job.candidate_id
            target.mkdir(parents=True, exist_ok=True)
            for path in source.iterdir():
                if path.is_file():
                    shutil.copy2(path, target / path.name)
    fresh = (
        runtime.run_jobs(
            version,
            pending,
            output_dir,
            gpu_count=gpu_count,
            job_deadline=deadline,
        )
        if pending
        else ()
    )
    by_id = {**completed, **{row.candidate_id: row for row in fresh}}
    return tuple(by_id[job.candidate_id] for job in jobs)


def _completed(results: tuple[CampaignJobResult, ...]) -> list[CampaignJobResult]:
    return [result for result in results if result.status == "completed" and result.brier is not None]


def _prediction_frame(result: CampaignJobResult):
    import pandas as pd

    if result.predictions_path is None or not result.predictions_path.is_file():
        raise CampaignRunnerError(f"completed result has no prediction evidence: {result.candidate_id}")
    frame = pd.read_csv(result.predictions_path)
    required = {"row_id", "target", "probability"}
    if not required.issubset(frame) or frame["row_id"].astype("string").duplicated().any():
        raise CampaignRunnerError(f"prediction evidence is not uniquely aligned: {result.candidate_id}")
    return frame


def _mean_prediction(results: tuple[CampaignJobResult, ...]):
    import numpy as np
    import pandas as pd

    frames = [_prediction_frame(result) for result in results]
    base = frames[0].copy()
    base["row_id"] = base["row_id"].astype(str)
    probabilities = [base["probability"].to_numpy(dtype="float64")]
    for index, frame in enumerate(frames[1:], start=1):
        candidate = frame.loc[:, ["row_id", "target", "probability"]].copy()
        candidate["row_id"] = candidate["row_id"].astype(str)
        aligned = base.loc[:, ["row_id", "target"]].merge(
            candidate,
            on=["row_id", "target"],
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if len(aligned) != len(base) or not aligned["_merge"].eq("both").all():
            raise CampaignRunnerError("ensemble prediction rows or targets differ")
        probabilities.append(aligned["probability"].to_numpy(dtype="float64"))
    base["probability"] = np.mean(np.vstack(probabilities), axis=0)
    return base


def _frame_brier(frame) -> float:
    import numpy as np

    return float(
        np.mean(
            np.square(
                frame["probability"].to_numpy(dtype="float64")
                - frame["target"].to_numpy(dtype="float64")
            )
        )
    )


def _ensemble_segment_pass(candidate, reference) -> bool:
    merged = reference.loc[:, ["row_id", "target", "probability"]].rename(
        columns={"probability": "reference_probability"}
    )
    segment_columns = [
        column
        for column in ("game_type", "game_month", "pitcher_id_known", "batter_id_known")
        if column in candidate
    ]
    merged = candidate.merge(merged, on=["row_id", "target"], how="inner", validate="one_to_one")
    for column in segment_columns:
        for _, group in merged.groupby(column, dropna=False, sort=True):
            if len(group) < 1000:
                continue
            delta = _frame_brier(group) - float(
                ((group["reference_probability"] - group["target"]) ** 2).mean()
            )
            if delta > 0.00050:
                return False
    return True


def _read_resume_state(
    path: Path,
    expected_config_sha: str,
    extraction_root: Path,
) -> tuple[str, str, dict[str, object]]:
    verified = verify_resume_bundle(path)
    if verified.campaign_config_sha256 != expected_config_sha:
        raise CampaignRunnerError("resume bundle campaign config SHA-256 differs")
    with ZipFile(path, "r") as archive:
        if "stage_state.json" not in archive.namelist():
            raise CampaignRunnerError("resume bundle has no stage_state.json")
        state = json.loads(archive.read("stage_state.json"))
        artifacts = state.get("resume_artifacts", {})
        if not isinstance(artifacts, dict):
            raise CampaignRunnerError("resume artifact map is invalid")
        extraction_root.mkdir(parents=True, exist_ok=True)
        rows = state.get("results", [])
        for candidate_id, binding in artifacts.items():
            if not isinstance(binding, dict):
                raise CampaignRunnerError("resume artifact binding is invalid")
            for kind in ("checkpoint", "predictions"):
                member = binding.get(kind)
                if member is None:
                    continue
                member = str(member)
                if member not in verified.member_sha256:
                    raise CampaignRunnerError(f"resume artifact is absent from manifest: {member}")
                target = extraction_root / member
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
                for row in rows:
                    if isinstance(row, dict) and row.get("candidate_id") == candidate_id:
                        row[kind if kind == "checkpoint" else "predictions"] = str(target)
            training_files = binding.get("training_files", [])
            if not isinstance(training_files, list):
                raise CampaignRunnerError("resume training-file binding is invalid")
            if training_files:
                training_dir = extraction_root / "training" / str(candidate_id)
                training_dir.mkdir(parents=True, exist_ok=True)
                for member in training_files:
                    member = str(member)
                    if member not in verified.member_sha256:
                        raise CampaignRunnerError(f"resume training file is absent: {member}")
                    (training_dir / Path(member).name).write_bytes(archive.read(member))
                for row in rows:
                    if isinstance(row, dict) and row.get("candidate_id") == candidate_id:
                        row["training_dir"] = str(training_dir)
    if state.get("version") != verified.version:
        raise CampaignRunnerError("resume stage state differs from its manifest")
    return verified.version, verified.manifest_sha256, state


def _run_a(
    campaign: Campaign,
    runtime: CampaignRuntime,
    output_dir: Path,
    deadline: float,
    prior: dict[str, object],
    gpu_count: int,
) -> tuple[dict[str, object], tuple[CampaignJobResult, ...], tuple[Path, ...]]:
    jobs = tuple(_job_from_candidate(candidate) for candidate in campaign.version_a_candidates)
    results = _run_with_completed_reuse(
        runtime, "A", jobs, output_dir / "jobs", deadline, prior, gpu_count
    )
    rows = _completed(results)
    if len(rows) < 4:
        state = {
            "version": "A",
            "stage_complete": False,
            "reason": "fewer_than_four_completed_candidates",
            "results": [_result_payload(result) for result in results],
        }
        return state, results, tuple(
            row.checkpoint for row in results if row.checkpoint is not None
        )
    by_id = {candidate.candidate_id: candidate for candidate in campaign.version_a_candidates}
    scores = [
        CandidateScore(
            row.candidate_id,
            by_id[row.candidate_id].capacity,
            by_id[row.candidate_id].num_embedding,
            by_id[row.candidate_id].loss,
            by_id[row.candidate_id].scheduler,
            float(row.brier),
            int(row.resource_evidence.get("parameter_count", 0)),
            float(row.resource_evidence.get("inference_seconds", 0.0)),
        )
        for row in rows
    ]
    survivors = choose_version_a_survivors(scores)
    survivor_payloads = [_candidate_payload(by_id[item.score.candidate_id]) for item in survivors]
    state = {
        "version": "A",
        "stage_complete": True,
        "survivors": survivor_payloads,
        "survivor_reasons": {item.score.candidate_id: item.reason for item in survivors},
        "results": [_result_payload(result) for result in results],
    }
    return state, results, ()


def _run_b(
    campaign: Campaign,
    prior: dict[str, object],
    runtime: CampaignRuntime,
    output_dir: Path,
    deadline: float,
    gpu_count: int,
) -> tuple[dict[str, object], tuple[CampaignJobResult, ...], tuple[Path, ...]]:
    survivors = tuple(_candidate_from_payload(item) for item in prior.get("survivors", ()))
    if len(survivors) != 4:
        raise CampaignRunnerError("Version B requires four Version A survivors")
    primary_jobs = tuple(
        _job_from_candidate(
            candidate,
            candidate_id=f"b__{candidate.candidate_id[3:]}__tr2023__va2024",
            sample_mode="full",
            max_epochs=40,
            patience=10,
        )
        for candidate in survivors
    )
    primary = _run_with_completed_reuse(
        runtime, "B", primary_jobs, output_dir / "jobs", deadline, prior, gpu_count
    )
    completed_primary = sorted(_completed(primary), key=lambda row: (float(row.brier), row.candidate_id))
    if len(completed_primary) < 2:
        state = {
            "version": "B",
            "stage_complete": False,
            "reason": "fewer_than_two_completed_primary_candidates",
            "survivors": [_candidate_payload(candidate) for candidate in survivors],
            "results": [_result_payload(result) for result in primary],
        }
        return state, primary, tuple(row.checkpoint for row in primary if row.checkpoint is not None)
    original_by_job = {job.candidate_id: candidate for job, candidate in zip(primary_jobs, survivors)}
    older_candidates = [original_by_job[row.candidate_id] for row in completed_primary[:2]]
    try:
        reference = next(
            candidate
            for candidate in survivors
            if candidate.candidate_id == VERSION_B_REFERENCE_CANDIDATE_ID
        )
    except StopIteration as error:
        raise CampaignRunnerError(
            "Version B declared temporal reference is not a survivor"
        ) from error
    if reference.candidate_id not in {
        candidate.candidate_id for candidate in older_candidates
    }:
        older_candidates.append(reference)
    older_jobs = tuple(
        _job_from_candidate(
            candidate,
            candidate_id=f"b__{candidate.candidate_id[3:]}__tr2022__va2023",
            train_end_year=2022,
            valid_year=2023,
            sample_mode="full",
            max_epochs=40,
            patience=10,
        )
        for candidate in older_candidates
    )
    older = _run_with_completed_reuse(
        runtime, "B", older_jobs, output_dir / "jobs", deadline, prior, gpu_count
    )
    all_results = (*primary, *older)
    result_by_id = {row.candidate_id: row for row in _completed(all_results)}
    reference_primary_id = next(job.candidate_id for job, candidate in zip(primary_jobs, survivors) if candidate.candidate_id == reference.candidate_id)
    reference_older_id = next(job.candidate_id for job, candidate in zip(older_jobs, older_candidates) if candidate.candidate_id == reference.candidate_id)
    if reference_primary_id not in result_by_id or reference_older_id not in result_by_id:
        state = {
            "version": "B",
            "stage_complete": False,
            "reason": "declared_temporal_reference_incomplete",
            "survivors": [_candidate_payload(candidate) for candidate in survivors],
            "results": [_result_payload(result) for result in all_results],
        }
        return state, all_results, tuple(
            row.checkpoint for row in all_results if row.checkpoint is not None
        )
    fold_briers: dict[str, tuple[float, float]] = {}
    for candidate in older_candidates:
        primary_id = next(job.candidate_id for job, item in zip(primary_jobs, survivors) if item.candidate_id == candidate.candidate_id)
        older_id = next(job.candidate_id for job, item in zip(older_jobs, older_candidates) if item.candidate_id == candidate.candidate_id)
        if not all(key in result_by_id for key in (primary_id, older_id)):
            continue
        fold_briers[candidate.candidate_id] = (
            float(result_by_id[primary_id].brier),
            float(result_by_id[older_id].brier),
        )
    champion_id, champion_score = choose_temporal_champion(
        fold_briers,
        reference_id=reference.candidate_id,
    )
    champion = next(
        candidate for candidate in older_candidates if candidate.candidate_id == champion_id
    )
    checkpoints = tuple(row.checkpoint for row in all_results if row.checkpoint is not None and champion.candidate_id in row.candidate_id)
    champion_primary_id = next(
        job.candidate_id
        for job, candidate in zip(primary_jobs, survivors)
        if candidate.candidate_id == champion.candidate_id
    )
    champion_older_id = next(
        (job.candidate_id for job, candidate in zip(older_jobs, older_candidates) if candidate.candidate_id == champion.candidate_id),
        reference_older_id,
    )
    state = {
        "version": "B",
        "stage_complete": True,
        "survivors": [_candidate_payload(candidate) for candidate in survivors],
        "champion": _candidate_payload(champion),
        "selection_delta": champion_score,
        "champion_fold_briers": {
            "2024": float(result_by_id[champion_primary_id].brier),
            "2023": float(result_by_id[champion_older_id].brier),
        },
        "results": [_result_payload(result) for result in all_results],
    }
    return state, all_results, checkpoints


def _version_c_confirmation_jobs(
    finalists: tuple[Candidate, ...],
    sentinel: Candidate,
) -> tuple[CampaignJob, ...]:
    candidates = (*finalists, sentinel)
    return tuple(
        _job_from_candidate(
            item,
            candidate_id=f"{item.candidate_id}__tr{train_end}__va{valid}",
            train_end_year=train_end,
            valid_year=valid,
            sample_mode="full",
            max_epochs=40,
            patience=10,
        )
        for item in candidates
        for train_end, valid in ((2023, 2024), (2022, 2023))
    )


def _run_c(
    campaign: Campaign,
    prior: dict[str, object],
    runtime: CampaignRuntime,
    output_dir: Path,
    deadline: float,
    gpu_count: int,
) -> tuple[dict[str, object], tuple[CampaignJobResult, ...], tuple[Path, ...]]:
    champion_raw = prior.get("refinement_base_candidate", prior.get("champion"))
    if not isinstance(champion_raw, dict):
        raise CampaignRunnerError("Version C requires a Version B champion")
    champion = _candidate_from_payload(champion_raw)
    baselines = prior.get("champion_fold_briers")
    if not isinstance(baselines, dict) or not {"2024", "2023"}.issubset(baselines):
        raise CampaignRunnerError("Version C requires both Version B champion fold metrics")
    sentinel_raw = prior.get("temporal_sentinel_candidate")
    if isinstance(sentinel_raw, dict):
        sentinel = _candidate_from_payload(sentinel_raw)
    else:
        survivor_rows = prior.get("survivors")
        if not isinstance(survivor_rows, list):
            raise CampaignRunnerError("Version C requires Version B survivor evidence")
        try:
            sentinel = next(
                _candidate_from_payload(item)
                for item in survivor_rows
                if isinstance(item, dict)
                and item.get("candidate_id")
                == VERSION_C_TEMPORAL_SENTINEL_CANDIDATE_ID
            )
        except StopIteration as error:
            raise CampaignRunnerError(
                "Version C temporal one-cycle sentinel is missing"
            ) from error
    if sentinel.candidate_id != VERSION_C_TEMPORAL_SENTINEL_CANDIDATE_ID:
        raise CampaignRunnerError("Version C temporal sentinel identity changed")
    refinements: list[Candidate] = []
    for item in campaign.refinements:
        dropout = min(0.30, max(0.0, champion.dropout + item.dropout_offset))
        suffix = f"lr{item.learning_rate:.4f}".replace(".", "p") + f"__d{dropout:.2f}".replace(".", "p")
        refinements.append(
            replace(
                champion,
                candidate_id=f"c__{champion.capacity}__{champion.num_embedding}__{champion.loss}__{champion.scheduler}__{suffix}__s42",
                dropout=dropout,
                learning_rate=item.learning_rate,
            )
        )
    proxy_jobs = tuple(_job_from_candidate(item) for item in refinements)
    proxy = _run_with_completed_reuse(
        runtime, "C", proxy_jobs, output_dir / "jobs", deadline, prior, gpu_count
    )
    proxy_completed = sorted(_completed(proxy), key=lambda row: (float(row.brier), row.candidate_id))
    if len(proxy_completed) < 2:
        state = {
            "version": "C",
            "stage_complete": False,
            "reason": "fewer_than_two_completed_proxy_refinements",
            "champion": _candidate_payload(champion),
            "refinement_base_candidate": _candidate_payload(champion),
            "champion_fold_briers": baselines,
            "temporal_sentinel_candidate": _candidate_payload(sentinel),
            "results": [_result_payload(result) for result in proxy],
        }
        return state, proxy, tuple(row.checkpoint for row in proxy if row.checkpoint is not None)
    finalists = tuple(
        next(item for item in refinements if item.candidate_id == row.candidate_id)
        for row in proxy_completed[:2]
    )
    confirmation_jobs = _version_c_confirmation_jobs(finalists, sentinel)
    confirmation = _run_with_completed_reuse(
        runtime,
        "C",
        confirmation_jobs,
        output_dir / "jobs",
        deadline,
        prior,
        gpu_count,
    )
    complete_by_id = {row.candidate_id: row for row in _completed(confirmation)}
    if len(complete_by_id) < len(confirmation_jobs):
        combined = (*proxy, *confirmation)
        state = {
            "version": "C",
            "stage_complete": False,
            "reason": "both_fold_refinement_or_sentinel_confirmation_incomplete",
            "champion": _candidate_payload(champion),
            "refinement_base_candidate": _candidate_payload(champion),
            "champion_fold_briers": baselines,
            "temporal_sentinel_candidate": _candidate_payload(sentinel),
            "results": [_result_payload(result) for result in combined],
        }
        return state, combined, tuple(
            row.checkpoint for row in combined if row.checkpoint is not None
        )
    alternative_candidates = (*finalists, sentinel)
    candidate_by_id = {
        item.candidate_id: item for item in (champion, *alternative_candidates)
    }
    fold_briers: dict[str, tuple[float, float]] = {
        champion.candidate_id: (float(baselines["2024"]), float(baselines["2023"]))
    }
    confirmed_refinements: list[Candidate] = []
    sentinel_verdict = None
    for item in alternative_candidates:
        primary_id = f"{item.candidate_id}__tr2023__va2024"
        older_id = f"{item.candidate_id}__tr2022__va2023"
        if primary_id in complete_by_id and older_id in complete_by_id:
            primary_brier = float(complete_by_id[primary_id].brier)
            older_brier = float(complete_by_id[older_id].brier)
            fold_briers[item.candidate_id] = (primary_brier, older_brier)
            verdict = temporal_verdict(
                TemporalEvidence(
                    item.candidate_id,
                    primary_brier - float(baselines["2024"]),
                    older_brier - float(baselines["2023"]),
                )
            )
            if item in finalists and verdict.accepted:
                confirmed_refinements.append(item)
            if item.candidate_id == sentinel.candidate_id:
                sentinel_verdict = verdict
    selected_id, _ = choose_temporal_champion(
        fold_briers,
        reference_id=champion.candidate_id,
    )
    selected = candidate_by_id[selected_id]
    seed_jobs = tuple(
        _job_from_candidate(
            replace(selected, seed=seed),
            candidate_id=f"c_final__{selected.candidate_id}__s{seed}__tr2023__va2024",
            sample_mode="full",
            max_epochs=40,
            patience=10,
        )
        for seed in campaign.seeds
    )
    seeds = _run_with_completed_reuse(
        runtime, "C", seed_jobs, output_dir / "jobs", deadline, prior, gpu_count
    )
    seed_completed = sorted(_completed(seeds), key=lambda row: (float(row.brier), row.candidate_id))
    if len(seed_completed) < len(seed_jobs):
        combined = (*proxy, *confirmation, *seeds)
        state = {
            "version": "C",
            "stage_complete": False,
            "reason": "primary_fold_seed_confirmation_incomplete",
            "champion": _candidate_payload(selected),
            "refinement_base_candidate": _candidate_payload(champion),
            "champion_fold_briers": baselines,
            "temporal_sentinel_candidate": _candidate_payload(sentinel),
            "results": [_result_payload(result) for result in combined],
        }
        return state, combined, tuple(
            row.checkpoint for row in combined if row.checkpoint is not None
        )
    primary_by_seed = {job.seed: result for job, result in zip(seed_jobs, seeds)}
    primary_seed_order = sorted(
        campaign.seeds,
        key=lambda seed: (float(primary_by_seed[seed].brier), seed),
    )
    eligible_seed_groups: list[tuple[int, ...]] = []
    for group in (tuple(primary_seed_order[:2]), tuple(primary_seed_order)):
        ensemble_primary = _frame_brier(
            _mean_prediction(tuple(primary_by_seed[seed] for seed in group))
        )
        best_member = min(float(primary_by_seed[seed].brier) for seed in group)
        if best_member - ensemble_primary >= 0.00003:
            eligible_seed_groups.append(group)
    older_needed = {42}
    for group in eligible_seed_groups:
        older_needed.update(group)
    older_seed_jobs = tuple(
        _job_from_candidate(
            replace(selected, seed=seed),
            candidate_id=f"c_final__{selected.candidate_id}__s{seed}__tr2022__va2023",
            train_end_year=2022,
            valid_year=2023,
            sample_mode="full",
            max_epochs=40,
            patience=10,
        )
        for seed in campaign.seeds
        if seed in older_needed
    )
    older_seeds = _run_with_completed_reuse(
        runtime,
        "C",
        older_seed_jobs,
        output_dir / "jobs",
        deadline,
        prior,
        gpu_count,
    )
    if len(_completed(older_seeds)) < len(older_seed_jobs):
        combined = (*proxy, *confirmation, *seeds, *older_seeds)
        state = {
            "version": "C",
            "stage_complete": False,
            "reason": "older_fold_seed_confirmation_incomplete",
            "champion": _candidate_payload(selected),
            "refinement_base_candidate": _candidate_payload(champion),
            "champion_fold_briers": baselines,
            "temporal_sentinel_candidate": _candidate_payload(sentinel),
            "results": [_result_payload(result) for result in combined],
        }
        return state, combined, tuple(
            row.checkpoint for row in combined if row.checkpoint is not None
        )

    older_by_seed = {job.seed: result for job, result in zip(older_seed_jobs, older_seeds)}
    seed_order = sorted(
        older_by_seed,
        key=lambda seed: (
            0.70 * float(primary_by_seed[seed].brier)
            + 0.30 * float(older_by_seed[seed].brier),
            seed,
        ),
    )
    predictor_options: list[
        tuple[
            str,
            tuple[tuple[Candidate, int, CampaignJobResult], ...],
            tuple[CampaignJobResult, ...],
            tuple[CampaignJobResult, ...],
        ]
    ] = [
        (
            f"single_s{seed}",
            ((replace(selected, seed=seed), seed, primary_by_seed[seed]),),
            (primary_by_seed[seed],),
            (older_by_seed[seed],),
        )
        for seed in seed_order
    ]
    for group in eligible_seed_groups:
        if all(seed in older_by_seed for seed in group):
            predictor_options.append(
                (
                    "mean_best_two_seeds" if len(group) == 2 else "mean_three_seeds",
                    tuple((replace(selected, seed=seed), seed, primary_by_seed[seed]) for seed in group),
                    tuple(primary_by_seed[seed] for seed in group),
                    tuple(older_by_seed[seed] for seed in group),
                )
            )
    if selected.candidate_id != sentinel.candidate_id:
        sentinel_primary = complete_by_id[
            f"{sentinel.candidate_id}__tr2023__va2024"
        ]
        sentinel_older = complete_by_id[
            f"{sentinel.candidate_id}__tr2022__va2023"
        ]
        predictor_options.append(
            (
                "mean_selected_s42_and_one_cycle_sentinel",
                (
                    (selected, 42, primary_by_seed[42]),
                    (sentinel, 42, sentinel_primary),
                ),
                (primary_by_seed[42], sentinel_primary),
                (older_by_seed[42], sentinel_older),
            )
        )
    if len(confirmed_refinements) == 2:
        distinct_primary = tuple(
            complete_by_id[f"{item.candidate_id}__tr2023__va2024"]
            for item in confirmed_refinements
        )
        distinct_older = tuple(
            complete_by_id[f"{item.candidate_id}__tr2022__va2023"]
            for item in confirmed_refinements
        )
        predictor_options.append(
            (
                "mean_two_distinct_structures_s42",
                tuple(
                    (item, 42, result)
                    for item, result in zip(confirmed_refinements, distinct_primary)
                ),
                distinct_primary,
                distinct_older,
            )
        )

    option_evidence: list[dict[str, object]] = []
    accepted_options: list[
        tuple[
            float,
            int,
            float,
            str,
            tuple[tuple[Candidate, int, CampaignJobResult], ...],
        ]
    ] = []
    for name, final_specs, primary_members, older_members in predictor_options:
        primary_frame = _mean_prediction(primary_members)
        older_frame = _mean_prediction(older_members)
        primary_brier = _frame_brier(primary_frame)
        older_brier = _frame_brier(older_frame)
        reference_index = min(
            range(len(primary_members)),
            key=lambda index: (float(primary_members[index].brier), primary_members[index].candidate_id),
        )
        reference_primary = _prediction_frame(primary_members[reference_index])
        reference_older = _prediction_frame(older_members[reference_index])
        primary_delta = primary_brier - _frame_brier(reference_primary)
        older_delta = older_brier - _frame_brier(reference_older)
        weighted_delta = 0.70 * primary_delta + 0.30 * older_delta
        segments_passed = _ensemble_segment_pass(primary_frame, reference_primary)
        verdict = (
            temporal_verdict(TemporalEvidence(name, 0.0, 0.0))
            if len(final_specs) == 1
            else ensemble_verdict(
                candidate_id=name,
                primary_gain=-primary_delta,
                older_delta=older_delta,
                weighted_delta=weighted_delta,
                segment_passed=segments_passed,
            )
        )
        accepted = len(final_specs) == 1 or verdict.accepted
        inference_seconds = sum(
            float(member.resource_evidence.get("inference_seconds", 0.0))
            for member in primary_members
        )
        option_evidence.append(
            {
                "predictor_id": name,
                "members": [
                    {"candidate_id": candidate.candidate_id, "seed": seed}
                    for candidate, seed, _ in final_specs
                ],
                "primary_brier": primary_brier,
                "older_brier": older_brier,
                "primary_gain_vs_best_member": -primary_delta,
                "older_delta_vs_best_member": older_delta,
                "weighted_delta_vs_best_member": weighted_delta,
                "segment_passed": segments_passed,
                "accepted": accepted,
                "reason": "single_member" if len(final_specs) == 1 else verdict.reason,
                "inference_seconds_sum": inference_seconds,
            }
        )
        if accepted:
            accepted_options.append(
                (
                    0.70 * primary_brier + 0.30 * older_brier,
                    len(final_specs),
                    inference_seconds,
                    name,
                    final_specs,
                )
            )
    accepted_options.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    best_metric = accepted_options[0][0]
    tied = [item for item in accepted_options if item[0] <= best_metric + 0.00002]
    chosen_option = min(tied, key=lambda item: (item[1], item[2], item[3]))
    _, _, _, chosen_name, chosen_specs = chosen_option
    checkpoints = tuple(
        result.checkpoint for _, _, result in chosen_specs if result.checkpoint is not None
    )
    combined_results = (*proxy, *confirmation, *seeds, *older_seeds)
    final_members: list[dict[str, object]] = []
    for candidate, seed, result in chosen_specs:
        if result.candidate_id.startswith("c_final__"):
            older_result = older_by_seed[seed]
        else:
            older_result = complete_by_id[f"{candidate.candidate_id}__tr2022__va2023"]
        final_members.append(
            {
                **_result_payload(result),
                "seed": seed,
                "candidate": _candidate_payload(candidate),
                "temporal_best_epochs": [result.best_epoch, older_result.best_epoch],
            }
        )
    state = {
        "version": "C",
        "stage_complete": True,
        "champion": _candidate_payload(selected),
        "refinement_base_candidate": _candidate_payload(champion),
        "final_members": final_members,
        "selected_predictor": chosen_name,
        "predictor_evidence": option_evidence,
        "temporal_sentinel": {
            "candidate": _candidate_payload(sentinel),
            "fold_briers": {
                "2024": fold_briers[sentinel.candidate_id][0],
                "2023": fold_briers[sentinel.candidate_id][1],
            },
            "accepted_as_single": bool(
                sentinel_verdict is not None and sentinel_verdict.accepted
            ),
            "reason": (
                "missing_verdict"
                if sentinel_verdict is None
                else sentinel_verdict.reason
            ),
            "ensemble_option": (
                None
                if selected.candidate_id == sentinel.candidate_id
                else "mean_selected_s42_and_one_cycle_sentinel"
            ),
        },
        "results": [_result_payload(result) for result in combined_results],
    }
    return state, combined_results, checkpoints


def _bundle_members(
    state: dict[str, object],
    results: tuple[CampaignJobResult, ...],
    checkpoints: tuple[Path, ...],
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    state = dict(state)
    artifact_bindings: dict[str, dict[str, str]] = {}
    include_completed_predictions = state.get("version") == "C" and state.get("stage_complete") is False
    for result in results:
        binding: dict[str, str] = {}
        if include_completed_predictions and result.status == "completed" and result.predictions_path is not None and result.predictions_path.is_file():
            binding["predictions"] = f"predictions/{result.candidate_id}.csv"
        if result.status == "inconclusive" and result.checkpoint is not None:
            training_files = [
                path
                for path in (
                    result.checkpoint.parent / "checkpoint.pt",
                    result.checkpoint.parent / "checkpoint_meta.json",
                    result.checkpoint.parent / "best_checkpoint.pt",
                )
                if path.is_file()
            ]
            if training_files:
                binding["training_files"] = [
                    f"training/{result.candidate_id}/{path.name}" for path in training_files
                ]
        if binding:
            artifact_bindings[result.candidate_id] = binding
    state["resume_artifacts"] = artifact_bindings
    state_bytes = _canonical_json(state)
    review: dict[str, bytes] = {
        "stage_state.json": state_bytes,
        "metrics/job_results.json": _canonical_json([_result_payload(result) for result in results]),
        "logs/stage.log": f"version={state['version']} completed={sum(r.status == 'completed' for r in results)}\n".encode(),
    }
    resume: dict[str, bytes] = {"stage_state.json": state_bytes}
    for result in results:
        if result.predictions_path is not None and result.predictions_path.is_file():
            name = f"predictions/{result.candidate_id}.csv"
            review[name] = result.predictions_path.read_bytes()
            if include_completed_predictions and result.status == "completed":
                resume[name] = result.predictions_path.read_bytes()
        binding = artifact_bindings.get(result.candidate_id, {})
        for member, path in zip(
            binding.get("training_files", []),
            (
                result.checkpoint.parent / name
                for name in ("checkpoint.pt", "checkpoint_meta.json", "best_checkpoint.pt")
                if result.checkpoint is not None and (result.checkpoint.parent / name).is_file()
            ),
        ):
            resume[str(member)] = path.read_bytes()
    return review, resume


def run_one_version(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    resume_bundle: str | Path | None = None,
    runtime: CampaignRuntime | None = None,
    config_path: str | Path = _DEFAULT_CONFIG,
    gpu_count: int = 2,
    now: Callable[[], float] = time.time,
) -> StageRunResult:
    """Advance exactly one trusted A-D stage; never create a submission artifact."""

    from competition_rules.contract import assert_experiment_runnable

    config = Path(config_path)
    project_root = Path(__file__).resolve().parents[2]
    assert_experiment_runnable(
        project_root=project_root,
        contract_path=Path(__file__).with_name("experiment_contract.json"),
        config_path=config,
    )
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 1:
        raise CampaignRunnerError("gpu_count must be a positive integer")
    campaign = load_campaign(config)
    config_sha = _config_sha(config)
    prior_manifest_sha: str | None = None
    prior: dict[str, object] = {}
    if resume_bundle is None:
        version = "A"
    else:
        previous, prior_manifest_sha, prior = _read_resume_state(
            Path(resume_bundle),
            config_sha,
            Path(output_dir).resolve() / "resume_import",
        )
        version = (
            previous
            if prior.get("stage_complete") is False
            else {"A": "B", "B": "C", "C": "D"}.get(previous, "")
        )
        if not version:
            raise CampaignRunnerError("the supplied resume bundle cannot advance another stage")
    if runtime is None:
        from .worker import SubprocessCampaignRuntime

        runtime = SubprocessCampaignRuntime(Path(data_dir))
    root = Path(output_dir).resolve() / f"stage_{version}"
    started = now()
    job_deadline = started + campaign.wall_seconds[version] - campaign.finalization_reserve_seconds
    print(f"STAGE_SELECTED version={version} wall_deadline_unix={int(started + campaign.wall_seconds[version])}", flush=True)
    if version == "A":
        state, results, checkpoints = _run_a(
            campaign, runtime, root, job_deadline, prior, gpu_count
        )
    elif version == "B":
        state, results, checkpoints = _run_b(
            campaign, prior, runtime, root, job_deadline, gpu_count
        )
    elif version == "C":
        state, results, checkpoints = _run_c(
            campaign, prior, runtime, root, job_deadline, gpu_count
        )
    else:
        from .final_review import run_final_review

        return run_final_review(
            campaign,
            prior,
            Path(data_dir),
            root,
            prior_manifest_sha,
            config_sha,
            job_deadline,
        )
    review, resume = _bundle_members(state, results, checkpoints)
    bundles = write_stage_bundles(
        root,
        StageEvidence(version, config_sha, prior_manifest_sha, review, resume),
    )
    print(f"BUNDLE_SUCCESS version={version} review={bundles.review} resume={bundles.resume}", flush=True)
    return StageRunResult(
        version,
        bundles,
        tuple(row.candidate_id for row in results if row.status == "completed"),
        tuple(row.candidate_id for row in results if row.status == "failed"),
        tuple(row.candidate_id for row in results if row.status == "inconclusive"),
    )
