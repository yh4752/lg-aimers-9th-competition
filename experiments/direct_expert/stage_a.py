from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

import pandas as pd

from .artifacts import DirectExpertBindings, create_bundle, extract_bundle
from .contracts import ExpertJob, load_contract, screening_jobs
from .inputs import canonical_json
from .selection import LockedSelection, select_structure_experts
from .state import complete_job, fail_job, initial_state, save_state, transition


class StageARuntime(Protocol):
    bindings: DirectExpertBindings

    def now(self) -> float: ...
    def run_job(self, job: ExpertJob, gpu: int, output: Path): ...
    def e2_oof(self, year: int) -> pd.DataFrame: ...


@dataclass(frozen=True)
class StageAResult:
    status: str
    handoff: Path
    completed_jobs: tuple[str, ...]
    failed_jobs: tuple[str, ...]
    selection: LockedSelection | None


def _selection_streams(runtime: StageARuntime, results: dict[str, object]) -> dict[str, pd.DataFrame]:
    streams = {}
    d0_by_year = {}
    for year in (2022, 2023):
        job_id = f"screen__D0__{year - 1}_{year}__s3407"
        d0_by_year[year] = results[job_id].predictions
    for expert_id in (f"D{i}" for i in range(8)):
        frames = []
        for year in (2022, 2023):
            job_id = f"screen__{expert_id}__{year - 1}_{year}__s3407"
            candidate = results[job_id].predictions.copy(deep=True)
            if expert_id in {"D5", "D6"}:
                active_type = "R" if expert_id == "D5" else "F"
                fallback = d0_by_year[year].set_index("row_id")["probability"]
                inactive = candidate["game_type"].ne(active_type)
                candidate.loc[inactive, "probability"] = candidate.loc[inactive, "row_id"].map(fallback)
            baseline = runtime.e2_oof(year).loc[:, ["row_id", "target", "p_anchor"]]
            merged = candidate.merge(
                baseline,
                on=["row_id", "target"],
                how="inner",
                validate="one_to_one",
            )
            if len(merged) != len(candidate):
                raise ValueError("E2 OOF alignment differs")
            frames.append(merged)
        streams[expert_id] = pd.concat(frames, ignore_index=True)
    return streams


def run_stage_a(
    runtime: StageARuntime,
    output_dir: Path,
    *,
    absolute_deadline: float,
    previous_handoff: Path | None = None,
) -> StageAResult:
    contract = load_contract()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = initial_state("stage_a")
    jobs = screening_jobs(contract)
    results: dict[str, object] = {}
    failures: dict[str, str] = {}
    guard = int(contract.runtime["new_job_guard_seconds"])
    if previous_handoff is not None:
        restored = output / "restored_stage_a"
        extract_bundle(previous_handoff, restored, "stage_a", runtime.bindings)
        for job in jobs:
            modern = restored / "jobs" / job.job_id / "predictions.csv"
            legacy_prediction = restored / f"predictions/{job.job_id}.csv"
            if not modern.is_file() and not legacy_prediction.is_file():
                continue
            job_output = output / "jobs" / job.job_id
            job_output.mkdir(parents=True, exist_ok=True)
            for member in ("predictions.csv", "metrics.json", "job_identity.json", "worker.log"):
                candidate = restored / "jobs" / job.job_id / member
                legacy = legacy_prediction if member == "predictions.csv" else None
                source_member = candidate if candidate.is_file() else legacy
                if source_member is not None and source_member.is_file():
                    (job_output / member).write_bytes(source_member.read_bytes())
            target = job_output / "predictions.csv"
            results[job.job_id] = SimpleNamespace(
                status="completed",
                job_id=job.job_id,
                predictions=pd.read_csv(target),
                output_dir=job_output,
            )
            state = complete_job(state, job.job_id)
    pending = [job for job in jobs if job.job_id not in results]
    runnable = [] if runtime.now() + guard >= absolute_deadline else pending

    def execute_gpu(gpu: int, assigned: list[ExpertJob]):
        observed = []
        for job in assigned:
            if runtime.now() + guard >= absolute_deadline:
                break
            try:
                result = runtime.run_job(job, gpu, output / "jobs" / job.job_id)
                if result.status != "completed" or result.job_id != job.job_id:
                    raise ValueError("worker result identity differs")
                observed.append((job, result, None))
            except Exception as error:
                observed.append((job, None, error))
        return observed

    if runnable:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(execute_gpu, gpu, runnable[gpu::2]) for gpu in (0, 1)]
            for future in futures:
                for job, result, error in future.result():
                    if error is None:
                        results[job.job_id] = result
                        state = complete_job(state, job.job_id)
                    else:
                        failures[job.job_id] = f"{type(error).__name__}: {error}"
                        state = fail_job(state, job.job_id, failures[job.job_id])
    selection = None
    selection_error = None
    if len(results) == len(jobs):
        state = transition(state, "selection")
        try:
            selection = select_structure_experts(_selection_streams(runtime, results))
        except Exception as error:
            selection_error = f"{type(error).__name__}: {error}"
        state = transition(state, "completed")
        status = "completed" if selection is not None else "completed_with_selection_failure"
    elif failures:
        status = "completed_with_candidate_failure"
    else:
        status = "incomplete"
    state_path = save_state(output / "state.json", state)
    status_path = output / "stage_a_status.json"
    status_path.write_bytes(
        canonical_json(
            {
                "status": status,
                "completed_jobs": list(state.completed_jobs),
                "failed_jobs": failures,
                "selection_error": selection_error,
                "selection_sha256": selection.selection_sha256 if selection else None,
            }
        )
    )
    payloads = {"state/state.json": state_path, "stage_a_status.json": status_path}
    if selection is not None:
        selection_path = output / "locked_selection.json"
        selection_path.write_bytes(
            canonical_json(
                {
                    "expert_ids": list(selection.expert_ids),
                    "roles": dict(selection.roles),
                    "locked_on_years": list(selection.locked_on_years),
                    "selection_sha256": selection.selection_sha256,
                }
            )
        )
        payloads["selection/locked_selection.json"] = selection_path
    for job_id, result in sorted(results.items()):
        for member in ("predictions.csv", "metrics.json", "job_identity.json", "worker.log"):
            source = result.output_dir / member
            if source.is_file():
                payloads[f"jobs/{job_id}/{member}"] = source
    handoff = create_bundle("stage_a", payloads, output / "direct_expert_stage_A_handoff.zip", runtime.bindings)
    return StageAResult(
        status=status,
        handoff=handoff,
        completed_jobs=state.completed_jobs,
        failed_jobs=tuple(sorted(failures)),
        selection=selection,
    )
