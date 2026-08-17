from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
from typing import Callable
from zipfile import ZipFile

import pandas as pd

from .artifacts import (
    DeploymentBundlePaths,
    verify_deployment_resume,
    write_deployment_bundles,
)
from .contracts import DEFAULT_CONTRACT, build_jobs, contract_sha256, load_contract
from .inputs import VerifiedDeploymentInputs, deployment_bindings
from .metrics import (
    DeploymentDecision,
    PrefixCandidate,
    decision_payload,
    evaluate_prefixes,
)
from .training import DeploymentJobResult, run_alignment_job, run_full_fit_job


class DeploymentRunnerError(RuntimeError):
    """Raised when the deployment campaign cannot advance safely."""


@dataclass(frozen=True)
class DeploymentRun:
    status: str
    bundles: DeploymentBundlePaths
    decision: DeploymentDecision | None
    completed_job_ids: tuple[str, ...]
    active_job_id: str | None


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def code_sha256() -> str:
    from .runtime_inventory import code_identity_sha256

    return code_identity_sha256(_root())


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json(value))
    os.replace(temporary, path)


def _state_payload(
    bindings: dict[str, str],
    *,
    status: str,
    completed: list[str],
    active: str | None,
    decision_sha256: str | None,
    selected_tree_count: int | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "campaign_id": "catboost_deployment_v1",
        "status": status,
        "completed_job_ids": list(completed),
        "active_job_id": active,
        "decision_sha256": decision_sha256,
        "selected_tree_count": selected_tree_count,
        "bindings": dict(bindings),
    }


def _decision_from_payload(value: object) -> DeploymentDecision:
    if type(value) is not dict or type(value.get("candidates")) is not list:
        raise DeploymentRunnerError("saved alignment decision is invalid")
    candidates: list[PrefixCandidate] = []
    for item in value["candidates"]:
        if type(item) is not dict:
            raise DeploymentRunnerError("saved prefix candidate is invalid")
        candidates.append(
            PrefixCandidate(
                tree_count=int(item["tree_count"]),
                fold_brier=dict(item["fold_brier"]),
                fold_regression=dict(item["fold_regression"]),
                weighted_brier=float(item["weighted_brier"]),
                weighted_gain=float(item["weighted_gain"]),
                passed=item["passed"] is True,
            )
        )
    return DeploymentDecision(
        status=str(value["status"]),
        selected_tree_count=value["selected_tree_count"],
        baseline_weighted_brier=float(value["baseline_weighted_brier"]),
        candidates=tuple(candidates),
        reason=str(value["reason"]),
    )


def _log(path: Path, message: str) -> None:
    print(message, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def _job_directories(jobs_root: Path, completed: list[str], active: str | None) -> dict[str, Path]:
    names = [*completed, *([] if active is None else [active])]
    return {name: jobs_root / name for name in names}


def _publish(
    *,
    output_dir: Path,
    bindings: dict[str, str],
    state_path: Path,
    log_path: Path,
    jobs_root: Path,
    completed: list[str],
    status: str,
    active: str | None,
    decision_path: Path | None,
    decision: DeploymentDecision | None,
    callback: Callable[[Path], None] | None,
) -> DeploymentBundlePaths:
    decision_sha = None if decision_path is None else sha256(decision_path.read_bytes()).hexdigest()
    selected = None if decision is None else decision.selected_tree_count
    _atomic_json(
        state_path,
        _state_payload(
            bindings,
            status=status,
            completed=completed,
            active=active,
            decision_sha256=decision_sha,
            selected_tree_count=selected,
        ),
    )
    bundles = write_deployment_bundles(
        output_dir=output_dir / "bundles",
        bindings=bindings,
        contract_path=DEFAULT_CONTRACT,
        stage_state_path=state_path,
        campaign_log_path=log_path,
        job_directories=_job_directories(jobs_root, completed, active),
        decision_path=decision_path,
    )
    verify_deployment_resume(bundles.resume, expected_bindings=bindings)
    if callback is not None:
        callback(bundles.resume)
    return bundles


def _restore_resume(
    resume: Path,
    *,
    bindings: dict[str, str],
    jobs_root: Path,
    state_path: Path,
    decision_path: Path,
) -> tuple[list[str], str | None, DeploymentDecision | None]:
    verified = verify_deployment_resume(resume, expected_bindings=bindings)
    temporary = Path(tempfile.mkdtemp(prefix="catboost-deployment-restore-", dir=jobs_root.parent))
    try:
        with ZipFile(resume) as archive:
            for name in archive.namelist():
                pure = PurePosixPath(name)
                if pure.parts[:1] == ("jobs",) and len(pure.parts) == 3:
                    target = temporary.joinpath(*pure.parts[1:])
                elif name == "state/stage_state.json":
                    target = temporary / "stage_state.json"
                elif name == "decision/alignment_decision.json":
                    target = temporary / "alignment_decision.json"
                else:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        for directory in sorted(path for path in temporary.iterdir() if path.is_dir()):
            os.replace(directory, jobs_root / directory.name)
        os.replace(temporary / "stage_state.json", state_path)
        decision: DeploymentDecision | None = None
        saved_decision = temporary / "alignment_decision.json"
        if saved_decision.is_file():
            os.replace(saved_decision, decision_path)
            decision = _decision_from_payload(json.loads(decision_path.read_text(encoding="utf-8")))
        return list(verified.completed_job_ids), verified.active_job_id, decision
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _load_alignment_frames(
    verified: VerifiedDeploymentInputs, jobs_root: Path
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    tabm = {
        fold: pd.read_csv(path) for fold, path in verified.stage_c.prediction_paths.items()
    }
    catboost = {
        "2022->2023": pd.read_csv(jobs_root / "align_2022_2023/predictions.csv"),
        "2023->2024": pd.read_csv(jobs_root / "align_2023_2024/predictions.csv"),
    }
    return tabm, catboost


def run_deployment_campaign(
    *,
    verified: VerifiedDeploymentInputs,
    output_dir: Path,
    resume_bundle: Path | None,
    absolute_deadline: float,
    on_verified_resume: Callable[[Path], None] | None = None,
    alignment_runtime: Callable[..., DeploymentJobResult] = run_alignment_job,
    full_runtime: Callable[..., DeploymentJobResult] = run_full_fit_job,
) -> DeploymentRun:
    contract = load_contract()
    current_code_sha = code_sha256()
    bindings = deployment_bindings(verified, expected_code_sha256=current_code_sha)
    if bindings["contract_sha256"] != contract_sha256():
        raise DeploymentRunnerError("contract identity differs")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    jobs_root = output_dir / "jobs"
    jobs_root.mkdir()
    state_path = output_dir / "stage_state.json"
    decision_path = output_dir / "alignment_decision.json"
    log_path = output_dir / "catboost_deployment.log"
    log_path.write_text("", encoding="utf-8")
    completed: list[str] = []
    active: str | None = None
    restored_decision: DeploymentDecision | None = None
    if resume_bundle is not None:
        completed, active, restored_decision = _restore_resume(
            Path(resume_bundle),
            bindings=bindings,
            jobs_root=jobs_root,
            state_path=state_path,
            decision_path=decision_path,
        )
        _log(log_path, f"DEPLOY_RESUME_READY completed={len(completed)} active={active or 'none'}")

    jobs = build_jobs(contract)
    alignment_jobs = jobs[:2]
    latest: DeploymentBundlePaths | None = None
    for job in alignment_jobs:
        if job.job_id in completed:
            _log(log_path, f"DEPLOY_JOB_REUSED job={job.job_id}")
            continue
        if time.time() + contract.new_job_guard_seconds >= absolute_deadline:
            latest = _publish(
                output_dir=output_dir,
                bindings=bindings,
                state_path=state_path,
                log_path=log_path,
                jobs_root=jobs_root,
                completed=completed,
                status="fresh" if not completed else "alignment_incomplete",
                active=None,
                decision_path=None,
                decision=None,
                callback=on_verified_resume,
            )
            return DeploymentRun("deployment_incomplete", latest, None, tuple(completed), None)
        _log(log_path, f"DEPLOY_JOB_START job={job.job_id}")
        result = alignment_runtime(
            job=job,
            contract=contract,
            data_dir=verified.training.data_dir,
            output_dir=jobs_root / job.job_id,
            contract_sha256=bindings["contract_sha256"],
            input_manifest_sha256=bindings["input_manifest_sha256"],
            code_sha256=bindings["code_sha256"],
            absolute_deadline=absolute_deadline,
        )
        if result.status != "completed":
            if result.status != "budget_inconclusive":
                raise DeploymentRunnerError(f"alignment job failed: {job.job_id}: {result.failure}")
            active = job.job_id if result.snapshot_path is not None else None
            latest = _publish(
                output_dir=output_dir,
                bindings=bindings,
                state_path=state_path,
                log_path=log_path,
                jobs_root=jobs_root,
                completed=completed,
                status="alignment_active" if active is not None else ("fresh" if not completed else "alignment_incomplete"),
                active=active,
                decision_path=None,
                decision=None,
                callback=on_verified_resume,
            )
            return DeploymentRun("deployment_incomplete", latest, None, tuple(completed), active)
        completed.append(job.job_id)
        active = None
        if len(completed) < 2:
            latest = _publish(
                output_dir=output_dir,
                bindings=bindings,
                state_path=state_path,
                log_path=log_path,
                jobs_root=jobs_root,
                completed=completed,
                status="alignment_incomplete",
                active=None,
                decision_path=None,
                decision=None,
                callback=on_verified_resume,
            )

    tabm_frames, catboost_frames = _load_alignment_frames(verified, jobs_root)
    decision = evaluate_prefixes(tabm_frames, catboost_frames, contract)
    encoded_decision = _canonical_json(decision_payload(decision))
    if restored_decision is not None and _canonical_json(decision_payload(restored_decision)) != encoded_decision:
        raise DeploymentRunnerError("restored alignment decision differs")
    decision_path.write_bytes(encoded_decision)
    _log(
        log_path,
        f"DEPLOY_ALIGNMENT_DECISION status={decision.status} selected_tree_count={decision.selected_tree_count}",
    )
    full_job = jobs[2]
    if full_job.job_id in completed:
        latest = _publish(
            output_dir=output_dir,
            bindings=bindings,
            state_path=state_path,
            log_path=log_path,
            jobs_root=jobs_root,
            completed=completed,
            status="full_training_complete",
            active=None,
            decision_path=decision_path,
            decision=decision,
            callback=on_verified_resume,
        )
        return DeploymentRun(
            "full_training_complete", latest, decision, tuple(completed), None
        )
    latest = _publish(
        output_dir=output_dir,
        bindings=bindings,
        state_path=state_path,
        log_path=log_path,
        jobs_root=jobs_root,
        completed=completed,
        status=decision.status,
        active=None,
        decision_path=decision_path,
        decision=decision,
        callback=on_verified_resume,
    )
    if decision.status == "deployment_blocked":
        return DeploymentRun("deployment_blocked", latest, decision, tuple(completed), None)
    if decision.selected_tree_count is None:
        raise DeploymentRunnerError("aligned decision is missing tree count")

    if full_job.job_id not in completed:
        if time.time() + contract.new_job_guard_seconds >= absolute_deadline:
            return DeploymentRun("deployment_incomplete", latest, decision, tuple(completed), None)
        _log(log_path, f"DEPLOY_JOB_START job={full_job.job_id}")
        result = full_runtime(
            job=full_job,
            contract=contract,
            selected_tree_count=decision.selected_tree_count,
            alignment_decision_sha256=sha256(encoded_decision).hexdigest(),
            data_dir=verified.training.data_dir,
            output_dir=jobs_root / full_job.job_id,
            contract_sha256=bindings["contract_sha256"],
            input_manifest_sha256=bindings["input_manifest_sha256"],
            code_sha256=bindings["code_sha256"],
            absolute_deadline=absolute_deadline,
        )
        if result.status != "completed":
            if result.status != "budget_inconclusive":
                raise DeploymentRunnerError(f"full fit failed: {result.failure}")
            active = full_job.job_id if result.snapshot_path is not None else None
            if active is not None:
                latest = _publish(
                    output_dir=output_dir,
                    bindings=bindings,
                    state_path=state_path,
                    log_path=log_path,
                    jobs_root=jobs_root,
                    completed=completed,
                    status="full_fit_active",
                    active=active,
                    decision_path=decision_path,
                    decision=decision,
                    callback=on_verified_resume,
                )
            return DeploymentRun("deployment_incomplete", latest, decision, tuple(completed), active)
        completed.append(full_job.job_id)
    latest = _publish(
        output_dir=output_dir,
        bindings=bindings,
        state_path=state_path,
        log_path=log_path,
        jobs_root=jobs_root,
        completed=completed,
        status="full_training_complete",
        active=None,
        decision_path=decision_path,
        decision=decision,
        callback=on_verified_resume,
    )
    return DeploymentRun(
        "full_training_complete", latest, decision, tuple(completed), None
    )
