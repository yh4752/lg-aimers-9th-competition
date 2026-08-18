from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from .artifacts import CampaignBundles, CampaignEvidence, write_campaign_bundles
from .calibration import (
    CalibrationSelection,
    CalibrationState,
    fit_h2,
    fit_h3,
)
from .contracts import DEFAULT_CONTRACT, HierarchicalJob, build_jobs, load_contract
from .metrics import CandidateDecision, candidate_decision_payload
from .training import TrainingJobResult, training_result_payload


class HierarchicalRunnerError(ValueError):
    """Raised when the restartable campaign state cannot advance safely."""


@dataclass(frozen=True)
class CampaignState:
    schema_version: int
    status: str
    bindings: Mapping[str, str]
    selected_k: float | None
    completed_job_ids: tuple[str, ...]
    active_job_id: str | None
    decisions: Mapping[str, CandidateDecision]
    delivery_candidate_ids: tuple[str, ...]
    final_epochs: int | None
    final_fit_completed: bool


@dataclass(frozen=True)
class CampaignRun:
    state: CampaignState
    bundles: CampaignBundles
    candidate_delivery: Path | None


class CampaignRuntime(Protocol):
    bindings: Mapping[str, str]

    def select_k(self, fit_rows: pd.DataFrame, valid_rows: pd.DataFrame) -> object: ...
    def run_oof(self, job: HierarchicalJob, **kwargs: object) -> TrainingJobResult: ...
    def calibrate(self, kind: str, **kwargs: object) -> object: ...
    def decide(self, candidate_id: str, **kwargs: object) -> CandidateDecision: ...
    def run_full_fit(self, job: HierarchicalJob, **kwargs: object) -> TrainingJobResult: ...


def choose_final_epochs(
    weighted_epochs: Sequence[tuple[int, int]], *, minimum: int, maximum: int
) -> int:
    if (
        isinstance(minimum, bool)
        or isinstance(maximum, bool)
        or not isinstance(minimum, int)
        or not isinstance(maximum, int)
        or minimum <= 0
        or maximum < minimum
        or not weighted_epochs
    ):
        raise HierarchicalRunnerError("final epoch inputs are invalid")
    parsed: list[tuple[int, int]] = []
    for rows, epoch in weighted_epochs:
        if (
            isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0
            or isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0
        ):
            raise HierarchicalRunnerError("weighted epoch entry is invalid")
        parsed.append((rows, epoch))
    total = sum(rows for rows, _ in parsed)
    cumulative = 0
    selected = parsed[-1][1]
    for rows, epoch in sorted(parsed, key=lambda item: item[1]):
        cumulative += rows
        if cumulative * 2 >= total:
            selected = epoch
            break
    return min(max(selected, minimum), maximum)


def fit_final_calibration(
    selection: CalibrationSelection | object,
    oof_by_fold: Mapping[str, pd.DataFrame],
) -> CalibrationState:
    if tuple(oof_by_fold) != ("2022->2023", "2023->2024"):
        raise HierarchicalRunnerError("final calibration fold set differs")
    combined = pd.concat(
        [oof_by_fold["2022->2023"], oof_by_fold["2023->2024"]],
        ignore_index=True,
    )
    required = {"row_id", "probability", "target"}
    if not required.issubset(combined.columns):
        raise HierarchicalRunnerError("final calibration evidence is incomplete")
    if combined["row_id"].isna().any() or combined["row_id"].astype(str).duplicated().any():
        raise HierarchicalRunnerError("final calibration row_id is invalid")
    kind = str(selection.kind)
    regularization = float(selection.selected_regularization)
    clip = float(getattr(selection.state, "clip", 1e-6))
    probability = combined["probability"].to_numpy(dtype="float64")
    target = combined["target"].to_numpy(dtype="float64")
    if kind == "H2":
        state = fit_h2(
            probability, target, regularization=regularization, clip=clip
        )
    elif kind == "H3":
        state = fit_h3(
            probability, target, combined,
            regularization=regularization, clip=clip,
        )
    else:
        raise HierarchicalRunnerError("final calibration kind differs")
    from hashlib import sha256
    expected = sha256("\n".join(combined["row_id"].astype(str)).encode()).hexdigest()
    if state.fit_row_ids_sha256 != expected:
        if kind != "H2":
            raise HierarchicalRunnerError("final calibration row binding differs")
        state = CalibrationState(
            state.schema_version, state.kind, state.regularization, state.clip,
            state.bias, state.slope, state.effects, expected,
        )
    return state


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _selection_payload(selection: object) -> dict[str, object]:
    return {
        "fold": str(getattr(selection, "fold", "2021->2022")),
        "selected_k": float(selection.selected_k),
        "scores": {str(key): float(value) for key, value in dict(selection.scores).items()},
        "row_count": int(selection.row_count),
    }


def _state_payload(state: CampaignState) -> dict[str, object]:
    return {
        "schema_version": state.schema_version,
        "status": state.status,
        "bindings": dict(state.bindings),
        "selected_k": state.selected_k,
        "completed_job_ids": list(state.completed_job_ids),
        "active_job_id": state.active_job_id,
        "decisions": {
            candidate: candidate_decision_payload(decision)
            for candidate, decision in state.decisions.items()
        },
        "delivery_candidate_ids": list(state.delivery_candidate_ids),
        "final_epochs": state.final_epochs,
        "final_fit_completed": state.final_fit_completed,
    }


def _deadline(deadline: float, *, guard_seconds: float = 0.0) -> None:
    if not math.isfinite(deadline) or time.time() + guard_seconds >= deadline:
        raise HierarchicalRunnerError("campaign deadline leaves no time for the next job")


def _ensure_worker_result(result: TrainingJobResult, directory: Path) -> None:
    path = directory / "worker_result.json"
    if not path.is_file():
        _atomic_json(path, training_result_payload(result))


def run_campaign(
    verified: object,
    output_dir: Path,
    *,
    resume_bundle: Path | None,
    absolute_deadline: float,
    runtime: CampaignRuntime,
    on_verified_resume,
) -> CampaignRun:
    del resume_bundle, on_verified_resume  # Resume restoration is wired by the Colab supervisor.
    contract = load_contract()
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise HierarchicalRunnerError("campaign output already exists")
    output.mkdir(parents=True)
    log_path = output / "campaign.log"
    log_path.write_text("CAMPAIGN_START\n", encoding="utf-8")
    bindings = MappingProxyType(dict(runtime.bindings))
    train = pd.read_csv(verified.training.data_dir / "train.csv")
    _deadline(absolute_deadline)
    selection = runtime.select_k(
        train.loc[train["season"] <= 2021].copy(),
        train.loc[train["season"] == 2022].copy(),
    )
    selected_k = float(selection.selected_k)
    k_path = output / "k_selection.json"
    _atomic_json(k_path, _selection_payload(selection))

    jobs = build_jobs(contract)
    results: dict[str, TrainingJobResult] = {}
    job_directories: dict[str, Path] = {}
    for job in jobs[:2]:
        _deadline(absolute_deadline, guard_seconds=0.0)
        directory = output / "jobs" / job.job_id
        result = runtime.run_oof(
            job, output_dir=directory, selected_k=selected_k,
            absolute_deadline=absolute_deadline,
        )
        if result.status != "completed" or result.predictions_path is None:
            raise HierarchicalRunnerError(f"OOF job did not complete: {job.job_id}")
        _ensure_worker_result(result, directory)
        results[job.job_id] = result
        job_directories[job.job_id] = directory
    oof = {
        "2022->2023": pd.read_csv(results[jobs[0].job_id].predictions_path),
        "2023->2024": pd.read_csv(results[jobs[1].job_id].predictions_path),
    }
    calibration_paths: dict[str, Path] = {}
    selections: dict[str, object] = {}
    for candidate in ("H2", "H3"):
        path = output / "calibration" / f"{candidate}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        selections[candidate] = runtime.calibrate(
            candidate, oof_by_fold=oof, output_path=path
        )
        calibration_paths[candidate] = path
    decisions: dict[str, CandidateDecision] = {}
    decision_paths: dict[str, Path] = {}
    for candidate in ("H1", "H2", "H3"):
        decision = runtime.decide(
            candidate, oof_by_fold=oof, selections=selections,
            anchor_predictions=getattr(verified, "anchor_predictions", {}),
        )
        decisions[candidate] = decision
        path = output / "decisions" / f"{candidate}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, candidate_decision_payload(decision))
        decision_paths[candidate] = path
    delivery_ids = tuple(
        candidate for candidate in ("H1", "H2", "H3")
        if decisions[candidate].delivery_role is not None
    )
    final_epochs = None
    final_completed = False
    if delivery_ids:
        final_epochs = choose_final_epochs(
            [
                (int(results[jobs[0].job_id].valid_rows), int(results[jobs[0].job_id].best_epoch) + 1),
                (int(results[jobs[1].job_id].valid_rows), int(results[jobs[1].job_id].best_epoch) + 1),
            ],
            minimum=contract.final_min_epochs,
            maximum=contract.final_max_epochs,
        )
        _deadline(absolute_deadline)
        final_job = jobs[2]
        directory = output / "jobs" / final_job.job_id
        final_result = runtime.run_full_fit(
            final_job, output_dir=directory, selected_k=selected_k,
            absolute_deadline=absolute_deadline, final_epochs=final_epochs,
        )
        if final_result.status != "completed" or final_result.final_model_path is None:
            raise HierarchicalRunnerError("full fit did not complete")
        _ensure_worker_result(final_result, directory)
        results[final_job.job_id] = final_result
        job_directories[final_job.job_id] = directory
        final_completed = True
        status = "delivery_pending_validation"
    else:
        status = "completed_no_candidate"
    state = CampaignState(
        1, status, bindings, selected_k, tuple(job_directories), None,
        MappingProxyType(decisions), delivery_ids, final_epochs, final_completed,
    )
    state_path = output / "stage_state.json"
    _atomic_json(state_path, _state_payload(state))
    bundles = write_campaign_bundles(
        CampaignEvidence(
            bindings, DEFAULT_CONTRACT, log_path, state_path, k_path,
            MappingProxyType(job_directories), MappingProxyType(calibration_paths),
            MappingProxyType(decision_paths), None,
        ),
        output / "bundles",
    )
    return CampaignRun(state, bundles, None)
