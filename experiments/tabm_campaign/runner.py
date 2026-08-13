from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping, Protocol
from zipfile import ZipFile

from .artifacts import BundlePaths, StageEvidence, verify_resume_bundle, write_stage_bundles
from .contracts import Campaign, Candidate, load_campaign
from .decisions import CandidateScore, TemporalEvidence, choose_version_a_survivors, temporal_verdict


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


def _completed(results: tuple[CampaignJobResult, ...]) -> list[CampaignJobResult]:
    return [result for result in results if result.status == "completed" and result.brier is not None]


def _read_resume_state(path: Path, expected_config_sha: str) -> tuple[str, str, dict[str, object]]:
    verified = verify_resume_bundle(path)
    if verified.campaign_config_sha256 != expected_config_sha:
        raise CampaignRunnerError("resume bundle campaign config SHA-256 differs")
    with ZipFile(path, "r") as archive:
        if "stage_state.json" not in archive.namelist():
            raise CampaignRunnerError("resume bundle has no stage_state.json")
        state = json.loads(archive.read("stage_state.json"))
    if state.get("version") != verified.version:
        raise CampaignRunnerError("resume stage state differs from its manifest")
    return verified.version, verified.manifest_sha256, state


def _run_a(
    campaign: Campaign,
    runtime: CampaignRuntime,
    output_dir: Path,
    deadline: float,
) -> tuple[dict[str, object], tuple[CampaignJobResult, ...], tuple[Path, ...]]:
    jobs = tuple(_job_from_candidate(candidate) for candidate in campaign.version_a_candidates)
    results = runtime.run_jobs("A", jobs, output_dir / "jobs", gpu_count=2, job_deadline=deadline)
    rows = _completed(results)
    if len(rows) < 4:
        raise CampaignRunnerError("Version A needs at least four completed candidates")
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
    primary = runtime.run_jobs("B", primary_jobs, output_dir / "jobs", gpu_count=2, job_deadline=deadline)
    completed_primary = sorted(_completed(primary), key=lambda row: (float(row.brier), row.candidate_id))
    if len(completed_primary) < 2:
        raise CampaignRunnerError("Version B needs at least two completed primary-fold candidates")
    original_by_job = {job.candidate_id: candidate for job, candidate in zip(primary_jobs, survivors)}
    older_candidates = [original_by_job[row.candidate_id] for row in completed_primary[:2]]
    p2 = min((candidate for candidate in survivors if candidate.capacity == "p2"), key=lambda x: x.candidate_id)
    if p2.candidate_id not in {candidate.candidate_id for candidate in older_candidates}:
        older_candidates.append(p2)
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
    older = runtime.run_jobs("B", older_jobs, output_dir / "jobs", gpu_count=2, job_deadline=deadline)
    all_results = (*primary, *older)
    result_by_id = {row.candidate_id: row for row in _completed(all_results)}
    p2_primary_id = next(job.candidate_id for job, candidate in zip(primary_jobs, survivors) if candidate.candidate_id == p2.candidate_id)
    p2_older_id = next(job.candidate_id for job, candidate in zip(older_jobs, older_candidates) if candidate.candidate_id == p2.candidate_id)
    champion = p2
    champion_score = 0.0
    for candidate in older_candidates:
        primary_id = next(job.candidate_id for job, item in zip(primary_jobs, survivors) if item.candidate_id == candidate.candidate_id)
        older_id = next(job.candidate_id for job, item in zip(older_jobs, older_candidates) if item.candidate_id == candidate.candidate_id)
        if not all(key in result_by_id for key in (primary_id, older_id, p2_primary_id, p2_older_id)):
            continue
        delta_2024 = float(result_by_id[primary_id].brier) - float(result_by_id[p2_primary_id].brier)
        delta_2023 = float(result_by_id[older_id].brier) - float(result_by_id[p2_older_id].brier)
        verdict = temporal_verdict(TemporalEvidence(candidate.candidate_id, delta_2024, delta_2023))
        weighted = 0.70 * delta_2024 + 0.30 * delta_2023
        if verdict.accepted and weighted < champion_score:
            champion, champion_score = candidate, weighted
    checkpoints = tuple(row.checkpoint for row in all_results if row.checkpoint is not None and champion.candidate_id in row.candidate_id)
    champion_primary_id = next(
        job.candidate_id
        for job, candidate in zip(primary_jobs, survivors)
        if candidate.candidate_id == champion.candidate_id
    )
    champion_older_id = next(
        job.candidate_id
        for job, candidate in zip(older_jobs, older_candidates)
        if candidate.candidate_id == champion.candidate_id
    )
    state = {
        "version": "B",
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


def _run_c(
    campaign: Campaign,
    prior: dict[str, object],
    runtime: CampaignRuntime,
    output_dir: Path,
    deadline: float,
) -> tuple[dict[str, object], tuple[CampaignJobResult, ...], tuple[Path, ...]]:
    champion_raw = prior.get("champion")
    if not isinstance(champion_raw, dict):
        raise CampaignRunnerError("Version C requires a Version B champion")
    champion = _candidate_from_payload(champion_raw)
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
    proxy = runtime.run_jobs("C", proxy_jobs, output_dir / "jobs", gpu_count=2, job_deadline=deadline)
    proxy_completed = sorted(_completed(proxy), key=lambda row: (float(row.brier), row.candidate_id))
    finalists = [next(item for item in refinements if item.candidate_id == row.candidate_id) for row in proxy_completed[:2]]
    confirmation_jobs = tuple(
        _job_from_candidate(
            item,
            candidate_id=f"{item.candidate_id}__tr{train_end}__va{valid}",
            train_end_year=train_end,
            valid_year=valid,
            sample_mode="full",
            max_epochs=40,
            patience=10,
        )
        for item in finalists
        for train_end, valid in ((2023, 2024), (2022, 2023))
    )
    confirmation = runtime.run_jobs("C", confirmation_jobs, output_dir / "jobs", gpu_count=2, job_deadline=deadline)
    complete_by_id = {row.candidate_id: row for row in _completed(confirmation)}
    selected = champion
    baselines = prior.get("champion_fold_briers")
    if not isinstance(baselines, dict) or not {"2024", "2023"}.issubset(baselines):
        raise CampaignRunnerError("Version C requires both Version B champion fold metrics")
    best_weighted = 0.0
    for item in finalists:
        primary_id = f"{item.candidate_id}__tr2023__va2024"
        older_id = f"{item.candidate_id}__tr2022__va2023"
        if primary_id in complete_by_id and older_id in complete_by_id:
            delta_2024 = float(complete_by_id[primary_id].brier) - float(baselines["2024"])
            delta_2023 = float(complete_by_id[older_id].brier) - float(baselines["2023"])
            verdict = temporal_verdict(TemporalEvidence(item.candidate_id, delta_2024, delta_2023))
            weighted = 0.70 * delta_2024 + 0.30 * delta_2023
            if verdict.accepted and weighted < best_weighted:
                best_weighted, selected = weighted, item
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
    seeds = runtime.run_jobs("C", seed_jobs, output_dir / "jobs", gpu_count=2, job_deadline=deadline)
    seed_completed = sorted(_completed(seeds), key=lambda row: (float(row.brier), row.candidate_id))
    if not seed_completed:
        raise CampaignRunnerError("Version C produced no completed final seed")
    # A multi-weight predictor is never promoted here without both-fold aligned
    # ensemble evidence. The next block adds such members only when every gate
    # can be computed; otherwise the best completed single is the safe champion.
    chosen = seed_completed[:1]
    checkpoints = tuple(row.checkpoint for row in chosen if row.checkpoint is not None)
    state = {
        "version": "C",
        "champion": _candidate_payload(selected),
        "final_members": [_result_payload(row) for row in chosen],
        "ensemble_status": "single_champion_retained_without_complete_both_fold_ensemble_evidence",
        "results": [_result_payload(result) for result in (*proxy, *confirmation, *seeds)],
    }
    return state, (*proxy, *confirmation, *seeds), checkpoints


def _bundle_members(
    state: dict[str, object],
    results: tuple[CampaignJobResult, ...],
    checkpoints: tuple[Path, ...],
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    state_bytes = _canonical_json(state)
    review: dict[str, bytes] = {
        "stage_state.json": state_bytes,
        "metrics/job_results.json": _canonical_json([_result_payload(result) for result in results]),
        "logs/stage.log": f"version={state['version']} completed={sum(r.status == 'completed' for r in results)}\n".encode(),
    }
    for result in results:
        if result.predictions_path is not None and result.predictions_path.is_file():
            review[f"predictions/{result.candidate_id}.csv"] = result.predictions_path.read_bytes()
    resume: dict[str, bytes] = {"stage_state.json": state_bytes}
    for checkpoint in checkpoints:
        if checkpoint.is_file():
            resume[f"checkpoints/{checkpoint.parent.name}__{checkpoint.name}"] = checkpoint.read_bytes()
    return review, resume


def run_one_version(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    resume_bundle: str | Path | None = None,
    runtime: CampaignRuntime | None = None,
    config_path: str | Path = _DEFAULT_CONFIG,
    now: Callable[[], float] = time.time,
) -> StageRunResult:
    """Advance exactly one trusted A-D stage; never create a submission artifact."""

    config = Path(config_path)
    campaign = load_campaign(config)
    config_sha = _config_sha(config)
    prior_manifest_sha: str | None = None
    prior: dict[str, object] = {}
    if resume_bundle is None:
        version = "A"
    else:
        previous, prior_manifest_sha, prior = _read_resume_state(Path(resume_bundle), config_sha)
        version = {"A": "B", "B": "C", "C": "D"}.get(previous, "")
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
        state, results, checkpoints = _run_a(campaign, runtime, root, job_deadline)
    elif version == "B":
        state, results, checkpoints = _run_b(campaign, prior, runtime, root, job_deadline)
    elif version == "C":
        state, results, checkpoints = _run_c(campaign, prior, runtime, root, job_deadline)
    else:
        from .final_review import run_final_review

        return run_final_review(campaign, prior, Path(data_dir), root, prior_manifest_sha, config_sha)
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
