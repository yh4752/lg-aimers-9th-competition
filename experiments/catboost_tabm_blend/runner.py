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

from .artifacts import BundlePaths, verify_resume_bundle, write_bundles
from .contracts import DEFAULT_CONTRACT, BlendContract, BlendJob, build_jobs, contract_sha256, load_contract
from .inputs import VerifiedStageC, VerifiedTrainingInput
from .metrics import BlendDecision, canonical_json, decision_payload, evaluate_blends
from .training import FoldResult, run_fold


class BlendRunnerError(RuntimeError):
    """Raised when the two-fold campaign cannot advance safely."""


@dataclass(frozen=True)
class BlendRun:
    stage_complete: bool
    bundles: BundlePaths
    decision: BlendDecision | None
    completed_job_ids: tuple[str, ...]
    active_job_id: str | None


_CODE_MEMBERS = (
    "experiments/catboost_preprocessing/features.py",
    "experiments/independent_dl/preprocessing.py",
    "experiments/independent_dl/row_features.py",
    "experiments/catboost_tabm_blend/contract.json",
    "experiments/catboost_tabm_blend/contracts.py",
    "experiments/catboost_tabm_blend/metrics.py",
    "experiments/catboost_tabm_blend/inputs.py",
    "experiments/catboost_tabm_blend/training.py",
    "experiments/catboost_tabm_blend/artifacts.py",
    "experiments/catboost_tabm_blend/runner.py",
    "experiments/catboost_tabm_blend/requirements-colab.txt",
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def code_sha256() -> str:
    digest = sha256()
    root = _root()
    for name in _CODE_MEMBERS:
        path = root / name
        if not path.is_file():
            raise BlendRunnerError(f"code identity member is missing: {name}")
        value = path.read_bytes()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def _bindings(
    verified_input: VerifiedTrainingInput,
    verified_stage_c: VerifiedStageC,
) -> dict[str, str]:
    return {
        "contract_sha256": contract_sha256(),
        "code_sha256": code_sha256(),
        "input_manifest_sha256": verified_input.manifest_sha256,
        "train_sha256": verified_input.train_sha256,
        "history_sha256": verified_input.history_sha256,
        "stage_c_delivery_sha256": verified_stage_c.delivery_sha256,
        "stage_c_review_sha256": verified_stage_c.review_sha256,
        "stage_c_state_sha256": verified_stage_c.stage_state_sha256,
    }


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(payload))
    os.replace(temporary, path)


def _state_payload(
    bindings: dict[str, str],
    completed: list[str],
    active: str | None,
    *,
    complete: bool,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "campaign_id": "catboost_tabm_blend_v1",
        "stage_complete": complete,
        "completed_job_ids": list(completed),
        "active_job_id": active,
        "bindings": dict(bindings),
    }


def _log(path: Path, message: str) -> None:
    print(message, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def _restore_resume(
    resume: Path,
    *,
    expected_bindings: dict[str, str],
    jobs_root: Path,
    state_path: Path,
) -> tuple[list[str], str | None]:
    verified = verify_resume_bundle(resume, expected_bindings=expected_bindings)
    temporary = Path(tempfile.mkdtemp(prefix="blend-restore-", dir=jobs_root.parent))
    try:
        with ZipFile(resume) as archive:
            state_path.write_bytes(archive.read("state/stage_state.json"))
            for name in archive.namelist():
                pure = PurePosixPath(name)
                if len(pure.parts) != 3 or pure.parts[0] != "jobs":
                    continue
                destination = temporary.joinpath(*pure.parts[1:])
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        for directory in sorted(temporary.iterdir()):
            os.replace(directory, jobs_root / directory.name)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return list(verified.completed_job_ids), verified.active_job_id


def _job_map(
    completed: list[str], active: str | None, jobs_root: Path
) -> dict[str, Path]:
    names = [*completed, *([] if active is None else [active])]
    return {name: jobs_root / name for name in names}


def _publish_state(
    *,
    output_dir: Path,
    bindings: dict[str, str],
    state_path: Path,
    log_path: Path,
    jobs_root: Path,
    completed: list[str],
    active: str | None,
    complete: bool,
    decision_path: Path | None,
) -> BundlePaths:
    _atomic_json(
        state_path,
        _state_payload(bindings, completed, active, complete=complete),
    )
    return write_bundles(
        output_dir=output_dir / "bundles",
        contract_path=DEFAULT_CONTRACT,
        bindings=bindings,
        stage_state_path=state_path,
        campaign_log_path=log_path,
        job_directories=_job_map(completed, active, jobs_root),
        decision_path=decision_path,
    )


def run_campaign(
    *,
    verified_input: VerifiedTrainingInput,
    verified_stage_c: VerifiedStageC,
    output_dir: Path,
    resume_bundle: Path | None,
    absolute_deadline: float,
    on_verified_resume: Callable[[Path], None] | None = None,
    runtime: Callable[..., FoldResult] = run_fold,
) -> BlendRun:
    contract: BlendContract = load_contract()
    bindings = _bindings(verified_input, verified_stage_c)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    jobs_root = output_dir / "jobs"
    jobs_root.mkdir()
    state_path = output_dir / "stage_state.json"
    log_path = output_dir / "blend_campaign.log"
    log_path.write_text("", encoding="utf-8")
    completed: list[str] = []
    active: str | None = None
    if resume_bundle is not None:
        completed, active = _restore_resume(
            Path(resume_bundle),
            expected_bindings=bindings,
            jobs_root=jobs_root,
            state_path=state_path,
        )
        _log(
            log_path,
            f"BLEND_RESUME_READY completed={len(completed)} active={active or 'none'}",
        )

    jobs = build_jobs(contract)
    valid_job_ids = {job.job_id for job in jobs}
    if not set(completed) <= valid_job_ids or (active is not None and active not in valid_job_ids):
        raise BlendRunnerError("restored job identity differs")

    latest_bundles: BundlePaths | None = None
    for job in jobs:
        if job.job_id in completed:
            _log(log_path, f"CATBOOST_JOB_REUSED job={job.job_id}")
            continue
        if time.time() + contract.new_job_guard_seconds >= absolute_deadline:
            active = None
            latest_bundles = _publish_state(
                output_dir=output_dir,
                bindings=bindings,
                state_path=state_path,
                log_path=log_path,
                jobs_root=jobs_root,
                completed=completed,
                active=None,
                complete=False,
                decision_path=None,
            )
            if on_verified_resume is not None:
                on_verified_resume(latest_bundles.resume)
            return BlendRun(
                False, latest_bundles, None, tuple(completed), None
            )
        active = job.job_id
        _log(log_path, f"CATBOOST_JOB_START job={job.job_id}")
        result = runtime(
            job=job,
            contract=contract,
            data_dir=verified_input.data_dir,
            output_dir=jobs_root / job.job_id,
            contract_sha256=bindings["contract_sha256"],
            input_manifest_sha256=bindings["input_manifest_sha256"],
            code_sha256=bindings["code_sha256"],
            absolute_deadline=absolute_deadline,
        )
        _log(log_path, f"CATBOOST_JOB_END job={job.job_id} status={result.status}")
        if result.status == "completed":
            completed.append(job.job_id)
            active = None
            _log(
                log_path,
                f"CATBOOST_FOLD_RESULT job={job.job_id} brier={result.brier:.12f}",
            )
        elif result.snapshot_path is not None and result.snapshot_path.is_file():
            active = job.job_id
        else:
            active = None
        latest_bundles = _publish_state(
            output_dir=output_dir,
            bindings=bindings,
            state_path=state_path,
            log_path=log_path,
            jobs_root=jobs_root,
            completed=completed,
            active=active,
            complete=False,
            decision_path=None,
        )
        if on_verified_resume is not None:
            on_verified_resume(latest_bundles.resume)
        if result.status != "completed":
            return BlendRun(
                False, latest_bundles, None, tuple(completed), active
            )

    if len(completed) != 2:
        raise BlendRunnerError("two completed folds are required for a decision")
    tabm = {
        fold: pd.read_csv(path)
        for fold, path in verified_stage_c.prediction_paths.items()
    }
    catboost = {
        f"{job.train_end_year}->{job.valid_year}": pd.read_csv(
            jobs_root / job.job_id / "predictions.csv"
        )
        for job in jobs
    }
    decision = evaluate_blends(tabm, catboost, contract)
    decision_path = output_dir / "blend_decision.json"
    decision_path.write_bytes(canonical_json(decision_payload(decision)))
    for candidate in decision.candidates:
        _log(
            log_path,
            f"BLEND_WEIGHT_RESULT tabm_weight={candidate.tabm_weight:.1f} "
            f"weighted_brier={candidate.weighted_brier:.12f} passed={str(candidate.passed).lower()}",
        )
    _log(
        log_path,
        f"BLEND_DECISION selected_tabm_weight="
        f"{decision.selected_tabm_weight if decision.selected_tabm_weight is not None else 'none'} "
        f"reason={decision.reason}",
    )
    latest_bundles = _publish_state(
        output_dir=output_dir,
        bindings=bindings,
        state_path=state_path,
        log_path=log_path,
        jobs_root=jobs_root,
        completed=completed,
        active=None,
        complete=True,
        decision_path=decision_path,
    )
    if on_verified_resume is not None:
        on_verified_resume(latest_bundles.resume)
    return BlendRun(
        True, latest_bundles, decision, tuple(completed), None
    )
