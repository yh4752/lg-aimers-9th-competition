from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import pandas as pd

from .hc_base import (
    ensemble_source_predictions,
    run_source_baseline,
    run_source_residual_job,
    source_e2_jobs,
)
from .hc_contracts import HCContract, HCJob, build_hc_jobs
from .hc_decisions import select_profile
from .hc_runner import HCStageOutcome
from .hc_state import (
    CampaignState,
    advance_stage,
    complete_job,
    fail_job,
    record_artifact,
    record_decision,
    skip_job,
    start_job,
    state_payload,
)
from .hc_training import (
    build_c1_training_rows,
    materialize_oof_feature_rows,
    run_c1_job,
)


class HCProductionError(RuntimeError):
    """Raised when production evidence is incomplete or changes identity."""


@dataclass(frozen=True)
class HCProductionDependencies:
    source_baseline: Callable[..., object] = run_source_baseline
    source_residual: Callable[..., object] = run_source_residual_job
    materialize: Callable[..., object] = materialize_oof_feature_rows
    c1_job: Callable[..., object] = run_c1_job


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_json(path: Path, value: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _read_prediction(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        raise HCProductionError(f"prediction evidence is unreadable: {path}") from error
    if frame.empty or "row_id" not in frame or not frame["row_id"].is_unique:
        raise HCProductionError(f"prediction evidence differs: {path}")
    return frame


class HCProductionRuntime:
    def __init__(
        self,
        *,
        data: object,
        evidence: object,
        output: Path,
        contract: HCContract,
        dependencies: HCProductionDependencies | None = None,
    ) -> None:
        self.data = data
        self.evidence = evidence
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.contract = contract
        self.dependencies = dependencies or HCProductionDependencies()
        self.log_path = self.output / "tree_hierarchical.log"

    def _log(self, message: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")
            handle.flush()
        print(message, flush=True)

    def _persist(self, state: CampaignState) -> None:
        _write_json(self.output / "states/state.json", state_payload(state))

    def _complete(self, state: CampaignState, job_id: str) -> CampaignState:
        if job_id in state.completed_jobs:
            return state
        updated = complete_job(start_job(state, job_id), job_id)
        self._persist(updated)
        return updated

    def _failed(self, state: CampaignState, job_id: str) -> CampaignState:
        updated = fail_job(start_job(state, job_id), job_id)
        self._persist(updated)
        return updated

    def _enough_time(self, deadline: float) -> bool:
        return time.time() + self.contract.runtime.new_job_guard_seconds < deadline

    def _source_baseline(
        self, state: CampaignState, deadline: float
    ) -> tuple[CampaignState, Path] | None:
        job_id = "hc__source_tabm__tr2020__va2021__s3407"
        output = self.output / "jobs" / job_id
        prediction = output / "predictions.csv"
        if job_id in state.completed_jobs:
            if not prediction.is_file():
                raise HCProductionError("completed source baseline is absent")
            return state, prediction
        if not self._enough_time(deadline):
            return None
        self._log(f"TREE_HC_JOB_START job={job_id} gpu=0")
        try:
            result = self.dependencies.source_baseline(
                data=self.data,
                output_dir=output,
                cache_root=self.output / "cache/source_2021",
                absolute_deadline=deadline,
            )
        except Exception:
            self._failed(state, job_id)
            raise
        if result.status != "completed" or not result.ready_for_residual:
            self._log(f"TREE_HC_JOB_END job={job_id} status={result.status}")
            return None
        state = self._complete(state, job_id)
        self._log(f"TREE_HC_JOB_END job={job_id} status=completed")
        return state, Path(result.predictions_path)

    def _source_residuals(
        self,
        state: CampaignState,
        baseline_path: Path,
        deadline: float,
        gpu_count: int,
    ) -> tuple[CampaignState, Mapping[int, Path]] | None:
        baseline = _read_prediction(baseline_path)
        jobs = source_e2_jobs()
        paths: dict[int, Path] = {}
        pending = []
        for job in jobs:
            prediction = self.output / "jobs" / job.job_id / "predictions.csv"
            if job.job_id in state.completed_jobs:
                if not prediction.is_file():
                    raise HCProductionError(f"completed source job is absent: {job.job_id}")
                paths[job.seed] = prediction
            else:
                pending.append(job)
        for start in range(0, len(pending), gpu_count):
            if not self._enough_time(deadline):
                return None
            chunk = pending[start : start + gpu_count]
            with ThreadPoolExecutor(max_workers=len(chunk)) as pool:
                active = {}
                for gpu_id, job in enumerate(chunk):
                    self._log(f"TREE_HC_JOB_START job={job.job_id} gpu={gpu_id}")
                    future = pool.submit(
                        self.dependencies.source_residual,
                        job=job,
                        data=self.data,
                        baseline=baseline,
                        output_dir=self.output / "jobs" / job.job_id,
                        absolute_deadline=deadline,
                        gpu_id=gpu_id,
                        input_manifest_sha256=self.evidence.manifest_sha256,
                    )
                    active[future] = job
                for future in as_completed(active):
                    job = active[future]
                    try:
                        result = future.result()
                    except Exception:
                        state = self._failed(state, job.job_id)
                        self._log(f"TREE_HC_JOB_END job={job.job_id} status=failed")
                        raise
                    if result.status != "completed":
                        self._log(
                            f"TREE_HC_JOB_END job={job.job_id} status={result.status}"
                        )
                        return None
                    state = self._complete(state, job.job_id)
                    paths[job.seed] = Path(result.predictions_path)
                    self._log(f"TREE_HC_JOB_END job={job.job_id} status=completed")
        if set(paths) != set(self.contract.seeds):
            raise HCProductionError("source seed evidence differs")
        return state, paths

    def _baseline_predictions(self, source_paths: Mapping[int, Path]) -> Mapping[int, pd.DataFrame]:
        source = ensemble_source_predictions(source_paths)
        source_path = self.output / "evidence/source_e2_2021.csv"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source.to_csv(source_path, index=False)
        predictions = {2021: source}
        for fold, path in self.evidence.fold_predictions.items():
            predictions[int(fold[1])] = _read_prediction(Path(path))
        if set(predictions) != {2021, 2022, 2023, 2024}:
            raise HCProductionError("baseline OOF year set differs")
        return predictions

    def _materialized(
        self, profile_name: str, p0_by_year: Mapping[int, pd.DataFrame]
    ) -> object:
        return self.dependencies.materialize(
            pd.read_csv(self.data.train),
            history=pd.read_csv(self.data.history),
            p0_by_year=p0_by_year,
            profile_name=profile_name,
            profile=self.contract.profiles[profile_name],
            minimum_group_rows=self.contract.minimum_group_rows,
        )

    def _c1_jobs(
        self,
        state: CampaignState,
        jobs: Sequence[HCJob],
        materialized_by_profile: Mapping[str, object],
        deadline: float,
        gpu_count: int,
    ) -> tuple[CampaignState, bool]:
        pending = [job for job in jobs if job.job_id not in state.completed_jobs]
        for start in range(0, len(pending), gpu_count):
            if not self._enough_time(deadline):
                return state, False
            chunk = pending[start : start + gpu_count]
            with ThreadPoolExecutor(max_workers=len(chunk)) as pool:
                active = {}
                for gpu_id, job in enumerate(chunk):
                    profile_name = job.profile
                    if profile_name is None:
                        raise HCProductionError("H1 C1 profile is absent")
                    materialized = materialized_by_profile[profile_name]
                    train = build_c1_training_rows(
                        materialized.frame, valid_year=int(job.valid_year)
                    )
                    valid = materialized.frame.loc[
                        materialized.frame["oof_year"].eq(job.valid_year)
                    ].copy(deep=True)
                    self._log(f"TREE_HC_JOB_START job={job.job_id} gpu={gpu_id}")
                    future = pool.submit(
                        self.dependencies.c1_job,
                        job_id=job.job_id,
                        seed=job.seed,
                        profile_name=profile_name,
                        train_rows=train,
                        valid_rows=valid,
                        feature_columns=materialized.feature_columns,
                        categorical_columns=materialized.categorical_columns,
                        output_dir=self.output / "jobs" / job.job_id,
                        absolute_deadline=deadline,
                        gpu_id=gpu_id,
                        contract=self.contract,
                    )
                    active[future] = job
                for future in as_completed(active):
                    job = active[future]
                    try:
                        result = future.result()
                    except Exception:
                        state = self._failed(state, job.job_id)
                        self._log(f"TREE_HC_JOB_END job={job.job_id} status=failed")
                        raise
                    if result.status != "completed":
                        self._log(
                            f"TREE_HC_JOB_END job={job.job_id} status={result.status}"
                        )
                        return state, False
                    state = self._complete(state, job.job_id)
                    self._log(f"TREE_HC_JOB_END job={job.job_id} status=completed")
        return state, True

    def _run_h1(
        self, state: CampaignState, deadline: float, gpu_count: int
    ) -> HCStageOutcome:
        source = self._source_baseline(state, deadline)
        if source is None:
            return HCStageOutcome(state, "incomplete", {}, None, None)
        state, baseline_path = source
        residuals = self._source_residuals(
            state, baseline_path, deadline, gpu_count
        )
        if residuals is None:
            return HCStageOutcome(state, "incomplete", {}, None, None)
        state, source_paths = residuals
        p0_by_year = self._baseline_predictions(source_paths)
        materialized = {
            profile: self._materialized(profile, p0_by_year)
            for profile in self.contract.profile_tie_order
        }
        registered = build_hc_jobs(self.contract)
        structure = tuple(
            job
            for job in registered
            if job.stage == "H1"
            and job.kind == "c1_residual"
            and (job.train_end_year, job.valid_year) in self.contract.structure_folds
        )
        state, complete = self._c1_jobs(
            state, structure, materialized, deadline, gpu_count
        )
        if not complete:
            return HCStageOutcome(state, "incomplete", {}, None, None)
        profile_frames = {
            profile: {
                year: _read_prediction(
                    self.output
                    / "jobs"
                    / f"hc__c1_{profile}__tr{year - 1}__va{year}__s3407"
                    / "predictions.csv"
                )
                for year in (2022, 2023)
            }
            for profile in self.contract.profile_tie_order
        }
        decision = select_profile(profile_frames, self.contract)
        decision_path = _write_json(self.output / "decisions/profile.json", decision)
        state = record_decision(state, "profile", _jsonable(decision))
        self._persist(state)
        self._log(f"TREE_HC_DECISION profile={decision.selected}")
        confirmation_jobs = tuple(
            job
            for job in registered
            if job.stage == "H1"
            and job.kind == "c1_residual"
            and (job.train_end_year, job.valid_year) == self.contract.confirmation_fold
        )
        selected_jobs = tuple(job for job in confirmation_jobs if job.profile == decision.selected)
        for job in confirmation_jobs:
            if job.profile != decision.selected and job.job_id not in dict(state.skipped_jobs):
                state = skip_job(state, job.job_id, "skipped_not_selected")
                self._persist(state)
        state, complete = self._c1_jobs(
            state,
            selected_jobs,
            {decision.selected: materialized[decision.selected]},
            deadline,
            gpu_count,
        )
        if not complete:
            return HCStageOutcome(
                state, "incomplete", {"decisions/profile.json": decision_path}, None, None
            )
        state = record_artifact(state, "profile_decision", "decisions/profile.json")
        state = advance_stage(state, "H2")
        self._persist(state)
        return HCStageOutcome(
            state=state,
            status="completed",
            review_sources={
                "decisions/profile.json": decision_path,
                "evidence/source_e2_2021.csv": self.output
                / "evidence/source_e2_2021.csv",
            },
            acceptance=None,
            delivery=None,
        )

    def run_stage(
        self,
        stage: str,
        state: CampaignState,
        deadline: float,
        gpu_count: int,
    ) -> HCStageOutcome:
        if stage != state.stage:
            raise HCProductionError("requested stage and state differ")
        if stage == "H1":
            return self._run_h1(state, deadline, gpu_count)
        raise HCProductionError(f"production stage is not available yet: {stage}")
