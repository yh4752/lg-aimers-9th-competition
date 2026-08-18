from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time
from types import MappingProxyType, SimpleNamespace
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from .artifacts import (
    CampaignBundles,
    CampaignEvidence,
    DeliveryEvidence,
    restore_resume,
    write_campaign_bundles,
    write_candidate_delivery,
)
from .calibration import (
    CalibrationSelection,
    CalibrationState,
    canonical_state_json,
    fit_h2,
    fit_h3,
)
from .contracts import DEFAULT_CONTRACT, HierarchicalJob, build_jobs, load_contract
from .metrics import CandidateDecision, candidate_decision_payload
from .inference import CandidateValidation, validate_candidate_artifacts
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


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _restored_path(
    directory: Path,
    payload: Mapping[str, object],
    *,
    name_key: str,
    sha_key: str,
) -> Path | None:
    name = payload[name_key]
    digest = payload[sha_key]
    if name is None or digest is None:
        if name is not None or digest is not None:
            raise HierarchicalRunnerError("restored worker artifact binding is incomplete")
        return None
    if (
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or not isinstance(digest, str)
        or len(digest) != 64
    ):
        raise HierarchicalRunnerError("restored worker artifact binding is invalid")
    path = directory / name
    if path.is_symlink() or not path.is_file() or _file_sha256(path) != digest:
        raise HierarchicalRunnerError("restored worker artifact hash differs")
    return path


def _restored_result(directory: Path, expected_job: HierarchicalJob) -> TrainingJobResult:
    result_path = directory / "worker_result.json"
    if result_path.is_symlink() or not result_path.is_file():
        raise HierarchicalRunnerError("restored worker result is missing")
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise HierarchicalRunnerError("restored worker result is unreadable") from error
    expected_keys = {
        "schema_version", "job_id", "kind", "status", "train_rows", "valid_rows",
        "best_epoch", "best_brier", "completed_epochs", "predictions", "checkpoint",
        "final_model", "feature_state", "failure", "predictions_sha256",
        "checkpoint_sha256", "final_model_sha256", "feature_state_sha256",
    }
    if (
        type(payload) is not dict
        or set(payload) != expected_keys
        or payload["schema_version"] != 1
        or payload["job_id"] != expected_job.job_id
        or payload["kind"] != expected_job.kind
        or payload["status"] != "completed"
        or payload["failure"] is not None
    ):
        raise HierarchicalRunnerError("restored worker result identity differs")
    integer_keys = ("train_rows", "completed_epochs")
    if any(
        isinstance(payload[key], bool)
        or not isinstance(payload[key], int)
        or payload[key] <= 0
        for key in integer_keys
    ):
        raise HierarchicalRunnerError("restored worker result counters are invalid")
    if expected_job.kind == "oof":
        if (
            isinstance(payload["valid_rows"], bool)
            or not isinstance(payload["valid_rows"], int)
            or payload["valid_rows"] <= 0
            or isinstance(payload["best_epoch"], bool)
            or not isinstance(payload["best_epoch"], int)
            or payload["best_epoch"] < 0
            or isinstance(payload["best_brier"], bool)
            or not isinstance(payload["best_brier"], (int, float))
            or not math.isfinite(float(payload["best_brier"]))
        ):
            raise HierarchicalRunnerError("restored OOF metrics are invalid")
    elif any(payload[key] is not None for key in ("valid_rows", "best_epoch", "best_brier")):
        raise HierarchicalRunnerError("restored full-fit metrics differ")
    predictions = _restored_path(
        directory, payload, name_key="predictions", sha_key="predictions_sha256"
    )
    checkpoint = _restored_path(
        directory, payload, name_key="checkpoint", sha_key="checkpoint_sha256"
    )
    final_model = _restored_path(
        directory, payload, name_key="final_model", sha_key="final_model_sha256"
    )
    feature_state = _restored_path(
        directory, payload, name_key="feature_state", sha_key="feature_state_sha256"
    )
    if checkpoint is None or feature_state is None:
        raise HierarchicalRunnerError("restored worker state is incomplete")
    if expected_job.kind == "oof" and (predictions is None or final_model is not None):
        raise HierarchicalRunnerError("restored OOF artifacts differ")
    if expected_job.kind == "full_fit" and (predictions is not None or final_model is None):
        raise HierarchicalRunnerError("restored full-fit artifacts differ")
    return TrainingJobResult(
        expected_job.job_id,
        expected_job.kind,
        "completed",
        int(payload["train_rows"]),
        None if payload["valid_rows"] is None else int(payload["valid_rows"]),
        None if payload["best_epoch"] is None else int(payload["best_epoch"]),
        None if payload["best_brier"] is None else float(payload["best_brier"]),
        int(payload["completed_epochs"]),
        predictions,
        checkpoint,
        final_model,
        feature_state,
        None,
    )


def _restored_selection(path: Path, selected_k: object) -> object:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise HierarchicalRunnerError("restored K selection is unreadable") from error
    if (
        type(payload) is not dict
        or set(payload) != {"fold", "selected_k", "scores", "row_count"}
        or payload["fold"] != "2021->2022"
        or isinstance(payload["selected_k"], bool)
        or not isinstance(payload["selected_k"], (int, float))
        or not math.isfinite(float(payload["selected_k"]))
        or float(payload["selected_k"]) <= 0
        or selected_k != payload["selected_k"]
        or type(payload["scores"]) is not dict
        or isinstance(payload["row_count"], bool)
        or not isinstance(payload["row_count"], int)
        or payload["row_count"] <= 0
    ):
        raise HierarchicalRunnerError("restored K selection differs")
    return SimpleNamespace(**payload)


def _restored_active_checkpoint(
    directory: Path,
    *,
    job_id: str,
    bindings: Mapping[str, str],
) -> Path:
    checkpoint = directory / "checkpoint.pt"
    metadata_path = directory / "checkpoint.pt.meta.json"
    if (
        checkpoint.is_symlink()
        or metadata_path.is_symlink()
        or not checkpoint.is_file()
        or not metadata_path.is_file()
    ):
        raise HierarchicalRunnerError("restored active checkpoint is incomplete")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise HierarchicalRunnerError("restored active checkpoint metadata is unreadable") from error
    if (
        type(metadata) is not dict
        or set(metadata) != {
            "schema_version", "job_id", "identity", "checkpoint_sha256",
            "completed_epochs",
        }
        or metadata["schema_version"] != 1
        or metadata["job_id"] != job_id
        or metadata["identity"] != dict(bindings)
        or metadata["checkpoint_sha256"] != _file_sha256(checkpoint)
        or isinstance(metadata["completed_epochs"], bool)
        or not isinstance(metadata["completed_epochs"], int)
        or metadata["completed_epochs"] <= 0
    ):
        raise HierarchicalRunnerError("restored active checkpoint binding differs")
    return checkpoint


def run_campaign(
    verified: object,
    output_dir: Path,
    *,
    resume_bundle: Path | None,
    absolute_deadline: float,
    runtime: CampaignRuntime,
    on_verified_resume,
    on_progress_state: Callable[[Path], None] | None = None,
) -> CampaignRun:
    contract = load_contract()
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise HierarchicalRunnerError("campaign output already exists")
    output.mkdir(parents=True)
    log_path = output / "campaign.log"
    log_path.write_text("CAMPAIGN_START\n", encoding="utf-8")
    bindings = MappingProxyType(dict(runtime.bindings))
    train = pd.read_csv(verified.training.data_dir / "train.csv")
    restored = None
    restored_results: dict[str, TrainingJobResult] = {}
    restored_directories: dict[str, Path] = {}
    active_resume_job_id: str | None = None
    active_resume_checkpoint: Path | None = None
    jobs = build_jobs(contract)
    jobs_by_id = {job.job_id: job for job in jobs}
    if resume_bundle is not None:
        _deadline(absolute_deadline)
        restored = restore_resume(
            Path(resume_bundle),
            output / "restored_resume",
            expected_bindings=bindings,
            check_deadline=lambda: _deadline(absolute_deadline),
        )
        if restored.state.get("schema_version") != 1 or restored.state.get("bindings") != dict(bindings):
            raise HierarchicalRunnerError("restored campaign identity differs")
        completed = restored.state.get("completed_job_ids")
        if (
            type(completed) is not list
            or len(completed) != len(set(completed))
            or any(not isinstance(job_id, str) or job_id not in jobs_by_id for job_id in completed)
        ):
            raise HierarchicalRunnerError("restored completed job set differs")
        for job_id in completed:
            directory = restored.completed_job_directories[job_id]
            restored_results[job_id] = _restored_result(directory, jobs_by_id[job_id])
            restored_directories[job_id] = directory
        active_value = restored.state.get("active_job_id")
        if active_value is not None:
            if (
                not isinstance(active_value, str)
                or active_value not in jobs_by_id
                or active_value in completed
                or restored.active_job_directory is None
            ):
                raise HierarchicalRunnerError("restored active job identity differs")
            active_resume_job_id = active_value
            active_resume_checkpoint = _restored_active_checkpoint(
                restored.active_job_directory,
                job_id=active_value,
                bindings=bindings,
            )
        selection_path = restored.root / "k_selection.json"
        if not selection_path.is_file():
            raise HierarchicalRunnerError("restored K selection is missing")
        selection = _restored_selection(selection_path, restored.state.get("selected_k"))
        on_verified_resume(Path(resume_bundle))
    else:
        _deadline(absolute_deadline)
        selection = runtime.select_k(
            train.loc[train["season"] <= 2021].copy(),
            train.loc[train["season"] == 2022].copy(),
        )
    selected_k = float(selection.selected_k)
    k_path = output / "k_selection.json"
    _atomic_json(k_path, _selection_payload(selection))
    state_path = output / "stage_state.json"

    results: dict[str, TrainingJobResult] = dict(restored_results)
    job_directories: dict[str, Path] = dict(restored_directories)

    def publish_progress(
        *,
        active_job_id: str | None,
        decisions: Mapping[str, CandidateDecision] | None = None,
        delivery_candidate_ids: tuple[str, ...] = (),
        final_epochs: int | None = None,
        final_fit_completed: bool = False,
    ) -> None:
        state = CampaignState(
            1,
            "running",
            bindings,
            selected_k,
            tuple(job_directories),
            active_job_id,
            MappingProxyType(dict(decisions or {})),
            delivery_candidate_ids,
            final_epochs,
            final_fit_completed,
        )
        _atomic_json(state_path, _state_payload(state))
        if on_progress_state is not None:
            on_progress_state(state_path)

    publish_progress(active_job_id=None)
    for job in jobs[:2]:
        if job.job_id in results:
            continue
        _deadline(
            absolute_deadline,
            guard_seconds=float(contract.new_job_guard_seconds),
        )
        directory = output / "jobs" / job.job_id
        publish_progress(active_job_id=job.job_id)
        result = runtime.run_oof(
            job, output_dir=directory, selected_k=selected_k,
            absolute_deadline=absolute_deadline,
            resume_checkpoint=(
                active_resume_checkpoint if job.job_id == active_resume_job_id else None
            ),
        )
        if result.status != "completed" or result.predictions_path is None:
            raise HierarchicalRunnerError(f"OOF job did not complete: {job.job_id}")
        _ensure_worker_result(result, directory)
        results[job.job_id] = result
        job_directories[job.job_id] = directory
        publish_progress(active_job_id=None)
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
    candidate_delivery = None
    final_calibration_paths: dict[str, Path] = {}
    if delivery_ids:
        final_epochs = choose_final_epochs(
            [
                (int(results[jobs[0].job_id].valid_rows), int(results[jobs[0].job_id].best_epoch) + 1),
                (int(results[jobs[1].job_id].valid_rows), int(results[jobs[1].job_id].best_epoch) + 1),
            ],
            minimum=contract.final_min_epochs,
            maximum=contract.final_max_epochs,
        )
        _deadline(
            absolute_deadline,
            guard_seconds=float(contract.new_job_guard_seconds),
        )
        final_job = jobs[2]
        if final_job.job_id in results:
            final_result = results[final_job.job_id]
            directory = job_directories[final_job.job_id]
        else:
            directory = output / "jobs" / final_job.job_id
            publish_progress(
                active_job_id=final_job.job_id,
                decisions=decisions,
                delivery_candidate_ids=delivery_ids,
                final_epochs=final_epochs,
            )
            final_result = runtime.run_full_fit(
                final_job, output_dir=directory, selected_k=selected_k,
                absolute_deadline=absolute_deadline, final_epochs=final_epochs,
                resume_checkpoint=(
                    active_resume_checkpoint
                    if final_job.job_id == active_resume_job_id else None
                ),
            )
        if final_result.status != "completed" or final_result.final_model_path is None:
            raise HierarchicalRunnerError("full fit did not complete")
        if final_job.job_id not in results:
            _ensure_worker_result(final_result, directory)
            results[final_job.job_id] = final_result
            job_directories[final_job.job_id] = directory
            publish_progress(
                active_job_id=None,
                decisions=decisions,
                delivery_candidate_ids=delivery_ids,
                final_epochs=final_epochs,
                final_fit_completed=True,
            )
        final_completed = True
        for candidate_id in delivery_ids:
            if candidate_id == "H1":
                continue
            final_state = fit_final_calibration(selections[candidate_id], oof)
            path = output / "final_calibration" / f"{candidate_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(canonical_state_json(final_state))
            final_calibration_paths[candidate_id] = path
        delivery_roles = MappingProxyType(
            {
                candidate_id: str(decisions[candidate_id].delivery_role)
                for candidate_id in delivery_ids
            }
        )
        validation_method = getattr(runtime, "validate_candidates", None)
        if validation_method is None:
            validation = validate_candidate_artifacts(
                bindings=bindings,
                delivery_roles=delivery_roles,
                final_checkpoint_path=final_result.final_model_path,
                feature_state_path=final_result.feature_state_path,
                calibration_paths=MappingProxyType(final_calibration_paths),
                audit_frame=train,
                output_dir=output / "candidate_validation",
                device=str(getattr(runtime, "inference_device", "cuda")),
                contract=contract,
            )
        else:
            validation = validation_method(
                bindings=bindings,
                delivery_roles=delivery_roles,
                final_checkpoint_path=final_result.final_model_path,
                feature_state_path=final_result.feature_state_path,
                calibration_paths=MappingProxyType(final_calibration_paths),
                audit_frame=train,
                output_dir=output / "candidate_validation",
                absolute_deadline=absolute_deadline,
            )
        if not isinstance(validation, CandidateValidation):
            raise HierarchicalRunnerError("candidate validation result differs")
        if (
            set(validation.report_paths) != set(delivery_roles)
            or not set(validation.accepted_roles).issubset(delivery_roles)
            or any(
                validation.accepted_roles[candidate_id] != delivery_roles[candidate_id]
                for candidate_id in validation.accepted_roles
            )
        ):
            raise HierarchicalRunnerError("candidate validation evidence differs")
        for candidate_id, report_path in validation.report_paths.items():
            decision_paths[f"inference_{candidate_id}"] = report_path
        accepted_roles = dict(validation.accepted_roles)
        if accepted_roles:
            accepted_calibration = {
                candidate_id: final_calibration_paths[candidate_id]
                for candidate_id in accepted_roles
                if candidate_id != "H1"
            }
            accepted_reports = {
                candidate_id: validation.report_paths[candidate_id]
                for candidate_id in accepted_roles
            }
            candidate_delivery = write_candidate_delivery(
                DeliveryEvidence(
                    bindings,
                    MappingProxyType(accepted_roles),
                    final_result.final_model_path,
                    final_result.feature_state_path,
                    MappingProxyType(accepted_calibration),
                    MappingProxyType(accepted_reports),
                ),
                output / "candidate_delivery",
            )
            delivery_ids = tuple(accepted_roles)
            status = "completed_candidate_delivery"
        else:
            delivery_ids = ()
            status = "completed_validation_rejected"
    else:
        status = "completed_no_candidate"
    state = CampaignState(
        1, status, bindings, selected_k, tuple(job_directories), None,
        MappingProxyType(decisions), delivery_ids, final_epochs, final_completed,
    )
    _atomic_json(state_path, _state_payload(state))
    if on_progress_state is not None:
        on_progress_state(state_path)
    bundles = write_campaign_bundles(
        CampaignEvidence(
            bindings, DEFAULT_CONTRACT, log_path, state_path, k_path,
            MappingProxyType(job_directories), MappingProxyType(calibration_paths),
            MappingProxyType(decision_paths), None,
        ),
        output / "bundles",
    )
    return CampaignRun(state, bundles, candidate_delivery)
