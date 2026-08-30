from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from .artifacts import DirectExpertBindings, create_bundle
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
) -> StageAResult:
    contract = load_contract()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = initial_state("stage_a")
    jobs = screening_jobs(contract)
    results: dict[str, object] = {}
    failures: dict[str, str] = {}
    guard = int(contract.runtime["new_job_guard_seconds"])
    runnable = [] if runtime.now() + guard >= absolute_deadline else list(jobs)

    def execute(index_job: tuple[int, ExpertJob]):
        index, job = index_job
        gpu = index % 2
        return job, runtime.run_job(job, gpu, output / "jobs" / job.job_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(execute, item): item[1] for item in enumerate(runnable)}
        for future in as_completed(futures):
            job = futures[future]
            try:
                observed_job, result = future.result()
                if observed_job != job or result.status != "completed" or result.job_id != job.job_id:
                    raise ValueError("worker result identity differs")
                results[job.job_id] = result
                state = complete_job(state, job.job_id)
            except Exception as error:
                failures[job.job_id] = f"{type(error).__name__}: {error}"
                state = fail_job(state, job.job_id, failures[job.job_id])
    selection = None
    if len(results) == len(jobs):
        state = transition(state, "selection")
        selection = select_structure_experts(_selection_streams(runtime, results))
        state = transition(state, "completed")
        status = "completed"
    elif runnable:
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
        payloads[f"predictions/{job_id}.csv"] = result.output_dir / "predictions.csv"
    handoff = create_bundle("stage_a", payloads, output / "direct_expert_stage_A_handoff.zip", runtime.bindings)
    return StageAResult(
        status=status,
        handoff=handoff,
        completed_jobs=state.completed_jobs,
        failed_jobs=tuple(sorted(failures)),
        selection=selection,
    )
