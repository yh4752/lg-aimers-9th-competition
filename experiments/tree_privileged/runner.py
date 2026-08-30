"""Restartable two-GPU privileged campaign runner."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
from types import MappingProxyType
from typing import Callable, Mapping, Sequence
from zipfile import ZipFile

import numpy as np
import pandas as pd

from experiments.tree_expert.e2_full_fit import AcceptedForFullFit
from experiments.tree_expert.e2_inference import load_inference_runtime
from experiments.tree_expert.inputs import VerifiedOfficialData

from .artifacts import CampaignArtifactState, CampaignBundles, file_sha256, write_campaign_bundles
from .contracts import contract_sha256, load_contract
from .decisions import (
    CandidateDecision, decide_candidate, select_confirmation_candidates, select_ensemble,
    select_rf_blend, summarize_screen,
)
from .full_fit import FullFitSources, accepted_full_fit_token, fit_accepted_candidate
from .inference import PrivilegedPredictor, StudentRuntime, audit_independence
from .inputs import VerifiedPrivilegedInput
from .profiles import ProfileStrengths, select_strengths
from .teacher import build_teacher_oof, load_teacher_evidence, save_teacher_evidence
from .training import CandidateJobResult, PrivilegedJob, run_candidate_job


class PrivilegedRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerifiedCampaignInputs:
    official: VerifiedOfficialData
    experiment: VerifiedPrivilegedInput


@dataclass(frozen=True)
class CampaignRunResult:
    status: str
    phase: str
    completed_jobs: tuple[str, ...]
    bundles: CampaignBundles
    accepted_candidates: tuple[str, ...]


def _code_sha256() -> str:
    root = Path(__file__).resolve().parents[2]
    names = sorted(path for path in (root / "experiments/tree_privileged").glob("*.py"))
    digest = sha256()
    for path in names:
        digest.update(path.relative_to(root).as_posix().encode()); digest.update(b"\0"); digest.update(path.read_bytes())
    return digest.hexdigest()


def _bindings(verified: VerifiedCampaignInputs) -> dict[str, str]:
    return {
        "contract_sha256": contract_sha256(), "code_sha256": _code_sha256(),
        "official_train_sha256": verified.official.train_sha256,
        "official_history_sha256": verified.official.history_sha256,
        "e2_handoff_sha256": verified.experiment.e2_handoff_sha256,
        "input_manifest_sha256": verified.experiment.manifest_sha256,
    }


def _log(path: Path, event: str, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = " ".join(f"{key}={value}" for key, value in values.items())
    line = f"{event}{(' ' + suffix) if suffix else ''}\n"
    with path.open("a", encoding="utf-8") as handle: handle.write(line); handle.flush()
    print(line, end="", flush=True)


def _atomic_json(path: Path, value: object) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    temporary = path.with_name(f".{path.name}.tmp"); temporary.write_bytes(payload); os.replace(temporary, path)


def _teacher_cache(all_rows: pd.DataFrame, history: pd.DataFrame, fold: tuple[int, int], root: Path, log: Path) -> Path:
    train_end, valid_year = fold; cache = root / str(valid_year)
    if (cache / "metadata.json").is_file():
        load_teacher_evidence(cache); _log(log, "TREE_PRIV_TEACHER_JOB_END", valid_year=valid_year, status="reused"); return cache
    _log(log, "TREE_PRIV_TEACHER_JOB_START", train_end=train_end, valid_year=valid_year)
    evidence = build_teacher_oof(all_rows.loc[all_rows["season"].le(train_end)].copy(), history,
                                 cutoff_year=train_end)
    save_teacher_evidence(evidence, cache)
    _log(log, "TREE_PRIV_MATCH_AUDIT", valid_year=valid_year, total_coverage=f"{evidence.coverage:.6f}",
         latest_coverage=f"{evidence.latest_coverage:.6f}", segments=json.dumps(dict(evidence.coverage_by_segment), sort_keys=True))
    _log(log, "TREE_PRIV_TEACHER_JOB_END", valid_year=valid_year, status=evidence.status,
         coverage=f"{evidence.coverage:.6f}", latest_coverage=f"{evidence.latest_coverage:.6f}")
    return cache


def _worker(
    job: PrivilegedJob,
    data: VerifiedOfficialData,
    baseline_path: Path,
    teacher_cache: Path,
    strengths: ProfileStrengths,
    output_dir: Path,
    deadline: float,
    gpu_id: int,
) -> CandidateJobResult:
    evidence = load_teacher_evidence(teacher_cache)
    baseline = pd.read_csv(baseline_path)
    return run_candidate_job(job=job, data=data, baseline=baseline, output_dir=output_dir,
                             absolute_deadline=deadline, gpu_id=gpu_id, teacher_evidence=evidence,
                             strengths=strengths)


def _reuse(job: PrivilegedJob, output: Path) -> CandidateJobResult | None:
    result_path = output / "worker_result.json"
    if not result_path.is_file(): return None
    try: value = json.loads(result_path.read_text())
    except Exception: return None
    if value.get("job_id") != job.job_id or value.get("candidate_id") != job.candidate_id: return None
    status = value.get("status")
    model = output / "model.cbm" if status == "completed" else None
    predictions = output / "predictions.csv" if status == "completed" else None
    if status == "completed" and not predictions.is_file(): return None
    if status not in {"completed", "skipped"}: return None
    return CandidateJobResult(job.job_id, job.candidate_id, status, value.get("brier"),
                              model if model is not None and model.is_file() else None,
                              predictions, value.get("failure"), value.get("best_iteration"))


def _run_jobs(
    jobs: Sequence[PrivilegedJob], *, data: VerifiedOfficialData, baseline_root: Path,
    teacher_root: Path, strengths: ProfileStrengths, jobs_root: Path, deadline: float,
    gpu_ids: tuple[int, ...], log: Path, clock: Callable[[], float] = time.time,
    launcher: Callable[..., CandidateJobResult] = _worker,
) -> tuple[dict[str, CandidateJobResult], bool]:
    contract = load_contract(); results: dict[str, CandidateJobResult] = {}; pending = []
    for job in jobs:
        reused = _reuse(job, jobs_root / job.job_id)
        if reused is not None:
            results[job.job_id] = reused; _log(log, "TREE_PRIV_JOB_REUSED", job=job.job_id, status=reused.status)
        else: pending.append(job)
    paused = False
    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as pool:
        active = {}; cursor = 0
        while pending or active:
            while pending and len(active) < len(gpu_ids):
                if clock() >= deadline - contract.runtime.new_job_guard_seconds:
                    paused = True; pending.clear(); break
                job = pending.pop(0); gpu = gpu_ids[cursor % len(gpu_ids)]; cursor += 1
                baseline = baseline_root / f"{job.valid_year}.csv"
                cache = teacher_root / str(job.valid_year)
                _log(log, "TREE_PRIV_CANDIDATE_JOB_START", job=job.job_id, gpu=gpu)
                future = pool.submit(launcher, job, data, baseline, cache, strengths,
                                     jobs_root / job.job_id, deadline - contract.runtime.artifact_reserve_seconds, gpu)
                active[future] = job
            if not active: break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                try: result = future.result()
                except Exception as error:
                    result = CandidateJobResult(job.job_id, job.candidate_id, "failed", None, None, None,
                                                f"{type(error).__name__}: {error}", None)
                results[job.job_id] = result
                _log(log, "TREE_PRIV_CANDIDATE_JOB_END", job=job.job_id, status=result.status,
                     brier=result.brier, failure=result.failure)
    return results, paused


def _job(candidate: str, fold: tuple[int, int], seed: int) -> PrivilegedJob:
    return PrivilegedJob(f"{candidate.lower()}__tr{fold[0]}__va{fold[1]}__s{seed}", candidate, fold[0], fold[1], seed)


def _evidence(results: Mapping[str, CandidateJobResult], baseline_root: Path) -> pd.DataFrame:
    parts = []
    for result in results.values():
        if result.status != "completed" or result.predictions_path is None: continue
        prediction = pd.read_csv(result.predictions_path)
        baseline = pd.read_csv(baseline_root / f"{next(j.valid_year for j in _ALL_JOBS if j.job_id == result.job_id)}.csv")
        if prediction["row_id"].astype(str).tolist() != baseline["row_id"].astype(str).tolist():
            raise PrivilegedRunnerError("OOF row alignment differs")
        job = next(j for j in _ALL_JOBS if j.job_id == result.job_id)
        part = prediction.rename(columns={"probability": "probability"}).copy()
        part["baseline_probability"] = baseline["probability"].to_numpy()
        part["candidate_id"], part["valid_year"], part["seed"] = job.candidate_id, job.valid_year, job.seed
        parts.append(part.loc[:, ["candidate_id", "valid_year", "seed", "row_id", "target",
                                  "baseline_probability", "probability", "game_type"]])
    if not parts: raise PrivilegedRunnerError("no completed candidate evidence")
    return pd.concat(parts, ignore_index=True)


_CONTRACT = load_contract()
_ALL_JOBS = tuple(
    _job(candidate, fold, seed)
    for candidate in _CONTRACT.candidates
    for fold in _CONTRACT.folds
    for seed in (_CONTRACT.screen_seed, *_CONTRACT.confirm_seeds)
)


def _sources(root: Path, *, models: bool) -> dict[str, Path]:
    output = {}
    if not root.exists(): return output
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink(): continue
        if not models and (path.suffix in {".cbm", ".cbsnapshot"} or "catboost_info" in path.parts): continue
        output[path.relative_to(root).as_posix()] = path
    return output


def _load_e2(delivery: Path, root: Path):
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True)
    with ZipFile(delivery) as archive:
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or any(part in {"", ".", ".."} for part in name.parts) or info.is_dir():
                raise PrivilegedRunnerError("unsafe E2 delivery member")
            target = root.joinpath(*name.parts); target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(archive.read(info))
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("predictor") != "catboost" or manifest.get("seeds") != [42, 2026, 3407]:
        raise PrivilegedRunnerError("nested E2 predictor differs")
    token = AcceptedForFullFit(manifest["candidate_id"], "catboost", (42, 2026, 3407),
                               MappingProxyType({int(k): int(v) for k, v in manifest["iterations"].items()}),
                               manifest["decision_sha256"])
    return load_inference_runtime(token=token, frozen_state_dir=root / "frozen_state",
                                  catboost_model_dir=root / "models", blend_method="catboost", catboost_weight=1.0)


def _bundle(
    root: Path, status: str, bindings: Mapping[str, str], log: Path,
    *, delivery_root: Path | None = None,
) -> CampaignBundles:
    state_path = root / "campaign_state.json"
    review_sources = {name: path for name, path in _sources(root / "review", models=False).items()}
    review_sources["campaign_state.json"] = state_path
    resume_sources = _sources(root / "jobs", models=False)
    resume_sources.update({f"cache/{name}": path for name, path in _sources(root / "cache", models=False).items()})
    resume_sources["campaign_state.json"] = state_path
    return write_campaign_bundles(CampaignArtifactState(
        status, MappingProxyType(dict(bindings)), MappingProxyType(review_sources),
        MappingProxyType(resume_sources), delivery_root, log,
    ), root / "bundles")


def run_campaign(
    verified: VerifiedCampaignInputs,
    output_dir: Path,
    *,
    wall_deadline: float | None = None,
    gpu_ids: tuple[int, ...] = (0, 1),
    clock: Callable[[], float] = time.time,
) -> CampaignRunResult:
    if type(verified) is not VerifiedCampaignInputs or gpu_ids != (0, 1):
        raise PrivilegedRunnerError("campaign inputs or T4x2 assignment differ")
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    for name in ("jobs", "cache", "review", "delivery"): (root / name).mkdir(exist_ok=True)
    log = root / "tree_privileged.log"; bindings = _bindings(verified)
    expected = load_contract().inputs
    if (verified.official.train_sha256 != expected["official_train_sha256"]
            or verified.official.history_sha256 != expected["official_history_sha256"]
            or verified.experiment.e2_handoff_sha256 != expected["e2_handoff_sha256"]):
        raise PrivilegedRunnerError("campaign source bindings differ")
    deadline = wall_deadline or clock() + load_contract().runtime.wall_seconds
    all_rows = pd.read_csv(verified.official.train); history = pd.read_csv(verified.official.history)
    strength_path = root / "cache/selected_strengths.json"
    if strength_path.is_file():
        strength_value = json.loads(strength_path.read_text()); strengths = ProfileStrengths(**strength_value["selected"])
    else:
        selection = select_strengths(all_rows, load_contract().folds); strengths = selection.selected
        _atomic_json(strength_path, {"selected": {"identity": strengths.identity, "interaction": strengths.interaction,
                                                   "matchup": strengths.matchup}, "scores": dict(selection.scores)})
    teacher_root = root / "cache/teacher"; teacher_root.mkdir(exist_ok=True)
    for fold in load_contract().folds:
        if clock() >= deadline - load_contract().runtime.new_job_guard_seconds: break
        _teacher_cache(all_rows, history, fold, teacher_root, log)
    if any(not (teacher_root / str(fold[1]) / "metadata.json").is_file() for fold in load_contract().folds):
        _atomic_json(root / "campaign_state.json", {"status": "paused", "phase": "teacher", "bindings": bindings,
                                                     "completed_jobs": []})
        bundles = _bundle(root, "paused", bindings, log)
        return CampaignRunResult("paused", "teacher", (), bundles, ())
    baseline_root = verified.experiment.e2_oof_root
    screen_jobs = [_job(candidate, fold, load_contract().screen_seed) for candidate in load_contract().candidates for fold in load_contract().folds]
    screen_results, paused = _run_jobs(screen_jobs, data=verified.official, baseline_root=baseline_root,
                                       teacher_root=teacher_root, strengths=strengths, jobs_root=root / "jobs",
                                       deadline=deadline, gpu_ids=gpu_ids, log=log, clock=clock)
    completed = dict(screen_results)
    if paused or any(result.status == "failed" for result in screen_results.values()):
        status = "paused" if paused else "failed"; phase = "screen"
        _atomic_json(root / "campaign_state.json", {"status": status, "phase": phase, "bindings": bindings,
                                                     "completed_jobs": sorted(completed)})
        bundles = _bundle(root, status, bindings, log)
        return CampaignRunResult(status, phase, tuple(sorted(completed)), bundles, ())
    screen_evidence = _evidence(completed, baseline_root)
    selected = select_confirmation_candidates(summarize_screen(screen_evidence))
    rf_decisions = {
        candidate: select_rf_blend(screen_evidence.loc[screen_evidence["candidate_id"].eq(candidate)])
        for candidate in selected if candidate.startswith("PD")
    }
    _atomic_json(root / "review/screen_decision.json", {
        "selected": list(selected),
        "rf": {candidate: {"alpha_r": value.alpha_r, "alpha_f": value.alpha_f}
               for candidate, value in rf_decisions.items()},
    })
    _log(log, "TREE_PRIV_DECISION", phase="screen", selected=",".join(selected) or "none")
    confirm_jobs = [_job(candidate, fold, seed) for candidate in selected for fold in load_contract().folds for seed in load_contract().confirm_seeds]
    confirm_results, paused = _run_jobs(confirm_jobs, data=verified.official, baseline_root=baseline_root,
                                        teacher_root=teacher_root, strengths=strengths, jobs_root=root / "jobs",
                                        deadline=deadline, gpu_ids=gpu_ids, log=log, clock=clock)
    completed.update(confirm_results)
    if paused or any(result.status == "failed" for result in confirm_results.values()):
        status = "paused" if paused else "failed"; phase = "confirm"
        _atomic_json(root / "campaign_state.json", {"status": status, "phase": phase, "bindings": bindings,
                                                     "completed_jobs": sorted(completed)})
        bundles = _bundle(root, status, bindings, log)
        return CampaignRunResult(status, phase, tuple(sorted(completed)), bundles, ())
    evidence = _evidence(completed, baseline_root)
    decisions = []
    for candidate in selected:
        part = evidence.loc[evidence["candidate_id"].eq(candidate)].copy()
        rf = rf_decisions.get(candidate)
        decisions.append(decide_candidate(part, rf_blend=rf))
    accepted = tuple(decision.candidate_id for decision in decisions if decision.status == "accepted")
    accepted = select_ensemble(evidence, accepted)
    decision_payload = {decision.candidate_id: {"status": decision.status, "failed_gates": list(decision.failed_gates),
                                                "observed": dict(decision.observed), "rf": None if decision.rf_blend is None else {
                                                    "alpha_r": decision.rf_blend.alpha_r, "alpha_f": decision.rf_blend.alpha_f}}
                        for decision in decisions}
    _atomic_json(root / "review/acceptance_decisions.json", decision_payload)
    _log(log, "TREE_PRIV_DECISION", phase="acceptance", accepted=",".join(accepted) or "none")
    if not accepted:
        _atomic_json(root / "campaign_state.json", {"status": "rejected", "phase": "complete", "bindings": bindings,
                                                     "completed_jobs": sorted(completed)})
        bundles = _bundle(root, "rejected", bindings, log)
        return CampaignRunResult("rejected", "complete", tuple(sorted(completed)), bundles, ())
    full_results = []
    full_bindings = {key: expected[key] for key in ("official_train_sha256", "official_history_sha256", "e2_handoff_sha256")}
    for candidate in accepted:
        decision = next(item for item in decisions if item.candidate_id == candidate)
        best = {seed: tuple(int(next(result.best_iteration for result in completed.values()
                                    if result.candidate_id == candidate and next(job.valid_year for job in _ALL_JOBS if job.job_id == result.job_id) == year
                                    and next(job.seed for job in _ALL_JOBS if job.job_id == result.job_id) == seed))
                                  for year in (2022, 2023, 2024)) for seed in (42, 2026, 3407)}
        token = accepted_full_fit_token(decision, bindings=full_bindings, best_iterations=best, strengths=strengths)
        full_results.append(fit_accepted_candidate(token, FullFitSources(verified.official,
                            verified.experiment.e2_delivery, verified.experiment.e2_handoff_sha256),
                            root / f"delivery/{candidate}", gpu_ids=gpu_ids))
    e2 = _load_e2(verified.experiment.e2_delivery, root / "cache/e2_runtime")
    runtimes = []
    for result in full_results:
        decision = next(item for item in decisions if item.candidate_id == result.candidate_id)
        alpha_r = 1.0 if decision.rf_blend is None else decision.rf_blend.alpha_r
        alpha_f = 1.0 if decision.rf_blend is None else decision.rf_blend.alpha_f
        runtimes.append(StudentRuntime(result.candidate_id, result.state, result.model_objects, alpha_r, alpha_f))
    predictor = PrivilegedPredictor(e2_predictor=e2, students=runtimes)
    audit_rows = all_rows.loc[all_rows["season"].eq(2024)].drop(columns="control_success").head(37).copy()
    audit = audit_independence(predictor, audit_rows)
    _atomic_json(root / "review/inference_audit.json", {"status": audit.status, "checks": dict(audit.checks),
                                                         "maximum_absolute_difference": audit.maximum_absolute_difference})
    status = "accepted" if audit.status == "passed" else "failed"
    if status == "accepted":
        evidence_root = root / "delivery/evidence"; evidence_root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "review/acceptance_decisions.json", evidence_root / "acceptance_decisions.json")
        shutil.copyfile(root / "review/inference_audit.json", evidence_root / "inference_audit.json")
    _atomic_json(root / "campaign_state.json", {"status": status, "phase": "complete", "bindings": bindings,
                                                 "completed_jobs": sorted(completed), "accepted_candidates": list(accepted)})
    bundles = _bundle(root, status, bindings, log, delivery_root=root / "delivery" if status == "accepted" else None)
    return CampaignRunResult(status, "complete", tuple(sorted(completed)), bundles, accepted if status == "accepted" else ())
