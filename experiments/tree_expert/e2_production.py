from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
import time
from types import MappingProxyType
from typing import Mapping, Sequence
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .e2_artifacts import E2DeliverySources, file_sha256, verify_e2_resume, write_e2_bundles
from .e2_baseline import load_reused_baselines, run_f1_baseline
from .e2_contracts import E2Contract, E2Job, build_structure_jobs, load_e2_contract
from .e2_decisions import (
    FOLD_ORDER,
    SeedEvidence,
    StructureEvidence,
    causal_blend_decision,
    decide_acceptance,
    decide_seeds,
    decide_structure,
    evaluate_fold,
)
from .e2_full_fit import (
    AcceptedForFullFit,
    accepted_full_fit_token,
    export_frozen_tree_state,
    fit_full_seed,
    prepare_full_fit_features,
)
from .e2_inference import audit_inference, load_inference_runtime
from .e2_inputs import VerifiedE2Input, verify_and_extract_e2_input, verify_official_data
from .e2_kaggle import runtime_identity_sha256
from .e2_runner import CampaignResult, CampaignState, PhaseOutcome, run_e2_campaign
from .e2_training import expected_job_identity, reusable_completed_job, run_e2_fold_job


class E2ProductionError(RuntimeError):
    """Raised when full E2 evidence cannot be produced without guessing."""


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(destination)
    return destination


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise E2ProductionError(f"JSON object differs: {path}")
    return value


def _append(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()
    print(message, flush=True)


def _metric(path: Path) -> dict[str, object]:
    value = _read_json(path)
    if type(value.get("best_iteration")) is not int or value["best_iteration"] < 0:
        raise E2ProductionError(f"best iteration differs: {path}")
    return value


def _prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty or not frame["row_id"].is_unique:
        raise E2ProductionError(f"prediction evidence differs: {path}")
    return frame


def _average_predictions(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    values = tuple(frames)
    if not values:
        raise E2ProductionError("prediction ensemble is empty")
    output = values[0].copy(deep=True)
    probabilities = []
    for frame in values:
        if (
            frame["row_id"].astype(str).tolist() != output["row_id"].astype(str).tolist()
            or not np.array_equal(frame["target"].to_numpy(), output["target"].to_numpy())
        ):
            raise E2ProductionError("ensemble prediction alignment differs")
        probabilities.append(frame["probability"].to_numpy(dtype="float64"))
    output["probability"] = np.mean(np.stack(probabilities), axis=0)
    return output


def build_inference_audit_rows(train_path: Path, *, row_count: int) -> pd.DataFrame:
    if type(row_count) is not int or row_count <= 0:
        raise E2ProductionError("inference audit row count differs")
    rows = pd.read_csv(train_path)
    seasons = pd.to_numeric(rows["season"], errors="raise")
    eligible = rows.loc[seasons.eq(2024)].iloc[:row_count].copy(deep=True)
    if len(eligible) != row_count or "control_success" not in eligible:
        raise E2ProductionError("inference audit source rows differ")
    return eligible.drop(columns="control_success").reset_index(drop=True)


def _token_from_path(path: Path) -> AcceptedForFullFit:
    value = _read_json(path)
    return AcceptedForFullFit(
        candidate_id=str(value["candidate_id"]),
        predictor=str(value["predictor"]),
        seeds=tuple(int(seed) for seed in value["seeds"]),
        iterations=MappingProxyType(
            {int(seed): int(count) for seed, count in value["iterations"].items()}
        ),
        decision_sha256=str(value["decision_sha256"]),
    )


class ProductionRuntime:
    def __init__(
        self,
        *,
        data: object,
        evidence: VerifiedE2Input,
        output: Path,
        contract: E2Contract,
    ) -> None:
        self.data = data
        self.evidence = evidence
        self.output = Path(output)
        self.contract = contract
        self.log = self.output / "tree_expert_e2.log"

    def _baselines(self) -> dict[tuple[int, int], pd.DataFrame]:
        paths = {
            (2021, 2022): self.output / "baselines/F1/predictions.csv",
            (2022, 2023): self.output / "baselines/F2/predictions.csv",
            (2023, 2024): self.output / "baselines/F3/predictions.csv",
        }
        return {fold: _prediction(path) for fold, path in paths.items()}

    def _job_path(self, job: E2Job) -> Path:
        return self.output / "jobs" / job.job_id

    def _f3_path(self, candidate: str) -> Path:
        return Path(self.evidence.e1_predictions[candidate])

    def _candidate_path(self, candidate: str, seed: int, fold: tuple[int, int]) -> Path:
        if seed == 3407 and fold == (2023, 2024):
            return self._f3_path(candidate)
        job = next(
            item
            for item in build_structure_jobs(self.contract, seed=seed, folds=(fold,))
            if item.candidate_id == candidate
        )
        return self._job_path(job) / "predictions.csv"

    def _best_iteration(self, candidate: str, seed: int, fold: tuple[int, int]) -> int:
        if seed == 3407 and fold == (2023, 2024):
            return int(_metric(self.evidence.e1_metrics[candidate])["best_iteration"])
        job = next(
            item
            for item in build_structure_jobs(self.contract, seed=seed, folds=(fold,))
            if item.candidate_id == candidate
        )
        return int(_metric(self._job_path(job) / "metrics.json")["best_iteration"])

    def _run_jobs(
        self,
        jobs: Sequence[E2Job],
        baselines: Mapping[tuple[int, int], pd.DataFrame],
        wall_deadline: float,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        completed: list[str] = []
        skipped: list[str] = []
        failed: list[str] = []
        pending: list[E2Job] = []
        for job in jobs:
            identity = expected_job_identity(
                job, self.data, self.evidence.manifest_sha256, contract=self.contract
            )
            if reusable_completed_job(self._job_path(job), job, identity) is not None:
                completed.append(job.job_id)
            else:
                pending.append(job)
        with ThreadPoolExecutor(max_workers=2) as pool:
            for start in range(0, len(pending), 2):
                active = {}
                for gpu_id, job in enumerate(pending[start : start + 2]):
                    _append(self.log, f"TREE_E2_JOB_START job={job.job_id} gpu={gpu_id}")
                    future = pool.submit(
                        run_e2_fold_job,
                        job=job,
                        data=self.data,
                        baseline=baselines[(job.train_end_year, job.valid_year)],
                        output_dir=self._job_path(job),
                        absolute_deadline=wall_deadline,
                        gpu_id=gpu_id,
                        input_manifest_sha256=self.evidence.manifest_sha256,
                        contract=self.contract,
                    )
                    active[future] = job
                for future in as_completed(active):
                    job = active[future]
                    try:
                        result = future.result()
                        if result.status == "completed":
                            completed.append(job.job_id)
                        elif result.status == "skipped":
                            skipped.append(job.job_id)
                        else:
                            failed.append(job.job_id)
                        status = result.status
                    except Exception as error:
                        failed.append(job.job_id)
                        status = f"failed:{type(error).__name__}"
                    _append(self.log, f"TREE_E2_JOB_END job={job.job_id} status={status}")
        return tuple(sorted(completed)), tuple(sorted(skipped)), tuple(sorted(failed))

    def _structure_decision(self) -> object:
        baselines = self._baselines()
        evidence = []
        for candidate in self.contract.structures:
            scores = []
            try:
                for fold in FOLD_ORDER:
                    scores.append(
                        evaluate_fold(
                            baselines[fold],
                            _prediction(self._candidate_path(candidate, 3407, fold)),
                            fold=fold,
                            best_iteration=self._best_iteration(candidate, 3407, fold),
                        )
                    )
            except (OSError, E2ProductionError):
                continue
            evidence.append(StructureEvidence(candidate, tuple(scores)))
        if not evidence:
            raise E2ProductionError("no complete structure evidence")
        return decide_structure(evidence, self.contract)

    def _selected(self) -> str:
        value = _read_json(self.output / "decisions/structure.json")
        selected = value.get("selected")
        if selected not in self.contract.structures:
            raise E2ProductionError("selected structure differs")
        return str(selected)

    def _ensemble_predictions(self, candidate: str) -> dict[tuple[int, int], pd.DataFrame]:
        output: dict[tuple[int, int], pd.DataFrame] = {}
        for fold in FOLD_ORDER:
            frames = [
                _prediction(self._candidate_path(candidate, seed, fold))
                for seed in self.contract.seeds
            ]
            averaged = _average_predictions(frames)
            path = self.output / f"ensembles/{fold[0]}_{fold[1]}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            averaged.to_csv(path, index=False)
            output[fold] = averaged
        return output

    def run_phase(
        self,
        phase: str,
        state: CampaignState,
        *,
        gpu_ids: tuple[int, int],
        wall_deadline: float,
    ) -> PhaseOutcome:
        del state
        if gpu_ids != (0, 1):
            raise E2ProductionError("production GPU identities differ")
        _append(self.log, f"TREE_E2_PHASE_START phase={phase}")
        if phase == "B0":
            reused = load_reused_baselines(self.evidence, self.data)
            for fold, label in (((2022, 2023), "F2"), ((2023, 2024), "F3")):
                path = self.output / f"baselines/{label}/predictions.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                reused[fold].to_csv(path, index=False)
            result = run_f1_baseline(
                data=self.data,
                output_dir=self.output / "baselines/F1",
                cache_root=self.output / "cache/F1",
                absolute_deadline=wall_deadline,
                contract=self.contract,
            )
            status = "passed" if result.ready_for_structure else "budget_inconclusive"
            decision = {"status": result.status, "best_epoch": result.best_epoch, "brier": result.brier}
            _write_json(self.output / "decisions/baseline.json", decision)
            return PhaseOutcome(status, ("baseline_F1",) if status == "passed" else (), (), decision, {})

        baselines = self._baselines()
        if phase == "B1":
            jobs = build_structure_jobs(
                self.contract, seed=3407, folds=FOLD_ORDER[:2]
            )
            completed, skipped, failed = self._run_jobs(jobs, baselines, wall_deadline)
            if failed:
                return PhaseOutcome("failed", completed, failed, {"status": "failed"}, {}, skipped)
            decision = self._structure_decision()
            _write_json(self.output / "decisions/structure.json", decision)
            return PhaseOutcome(
                "passed" if decision.status == "passed" else "rejected",
                completed,
                (),
                _jsonable(decision),
                {"structure_decision": "decisions/structure.json"},
                skipped,
            )

        candidate = self._selected()
        if phase == "B2":
            jobs = tuple(
                job
                for seed in (42, 2026)
                for job in build_structure_jobs(self.contract, seed=seed, folds=FOLD_ORDER)
                if job.candidate_id == candidate
            )
            completed, skipped, failed = self._run_jobs(jobs, baselines, wall_deadline)
            if failed or skipped:
                return PhaseOutcome("failed", completed, failed, {"status": "failed"}, {}, skipped)
            seed_evidence = []
            for seed in self.contract.seeds:
                scores = tuple(
                    evaluate_fold(
                        baselines[fold],
                        _prediction(self._candidate_path(candidate, seed, fold)),
                        fold=fold,
                        best_iteration=self._best_iteration(candidate, seed, fold),
                    )
                    for fold in FOLD_ORDER
                )
                seed_evidence.append(SeedEvidence(seed, scores))
            ensembles = self._ensemble_predictions(candidate)
            ensemble_scores = tuple(
                evaluate_fold(
                    baselines[fold],
                    ensembles[fold],
                    fold=fold,
                    best_iteration=max(
                        self._best_iteration(candidate, seed, fold)
                        for seed in self.contract.seeds
                    ),
                )
                for fold in FOLD_ORDER
            )
            decision = decide_seeds(candidate, seed_evidence, ensemble_scores, self.contract)
            _write_json(self.output / "decisions/seeds.json", decision)
            return PhaseOutcome(
                "passed" if decision.status == "passed" else "rejected",
                completed,
                (),
                _jsonable(decision),
                {"seed_decision": "decisions/seeds.json"},
            )

        ensembles = {
            fold: _prediction(self.output / f"ensembles/{fold[0]}_{fold[1]}.csv")
            for fold in FOLD_ORDER
        }
        if phase == "B3":
            decision = causal_blend_decision(baselines, ensembles, self.contract)
            _write_json(self.output / "decisions/blend.json", decision)
            return PhaseOutcome(
                "passed" if decision.status == "passed" else "rejected",
                (), (), _jsonable(decision), {"blend_decision": "decisions/blend.json"}
            )

        blend_payload = _read_json(self.output / "decisions/blend.json")
        if phase == "ACCEPTANCE":
            scores = tuple(
                evaluate_fold(
                    baselines[fold],
                    ensembles[fold],
                    fold=fold,
                    best_iteration=max(
                        self._best_iteration(candidate, seed, fold)
                        for seed in self.contract.seeds
                    ),
                )
                for fold in FOLD_ORDER
            )
            rows = pd.read_csv(self.data.train)
            groups = {
                fold: rows.loc[rows["season"].eq(fold[1]), "pitcher_id"].tolist()
                for fold in FOLD_ORDER
            }
            from .e2_decisions import BlendDecision
            blend = BlendDecision(**blend_payload)
            decision = decide_acceptance(
                scores, baselines, ensembles, groups, self.contract, blend=blend
            )
            decision_path = _write_json(self.output / "decisions/acceptance.json", decision)
            if decision.status == "accepted":
                best = {
                    seed: tuple(
                        self._best_iteration(candidate, seed, fold) for fold in FOLD_ORDER
                    )
                    for seed in self.contract.seeds
                }
                token = accepted_full_fit_token(
                    decision,
                    candidate_id=candidate,
                    best_iterations=best,
                    decision_sha256=file_sha256(decision_path),
                )
                _write_json(self.output / "decisions/accepted_token.json", token)
            return PhaseOutcome(
                "passed" if decision.status == "accepted" else "rejected",
                (), (), _jsonable(decision), {"acceptance_decision": "decisions/acceptance.json"}
            )

        token = _token_from_path(self.output / "decisions/accepted_token.json")
        if phase == "FULL_FIT":
            feature_state, batch = prepare_full_fit_features(token, self.data)
            frozen = self.output / "full_fit/frozen_state"
            if not frozen.exists():
                export_frozen_tree_state(feature_state, frozen, candidate_id=token.candidate_id)
            models = []
            with ThreadPoolExecutor(max_workers=2) as pool:
                for start in range(0, len(token.seeds), 2):
                    active = {
                        pool.submit(
                            fit_full_seed,
                            token=token,
                            seed=seed,
                            state=feature_state,
                            batch=batch,
                            output_dir=self.output / "full_fit/models",
                            gpu_id=gpu_id,
                            contract=self.contract,
                        ): seed
                        for gpu_id, seed in enumerate(token.seeds[start : start + 2])
                    }
                    for future in as_completed(active):
                        models.append(future.result())
            manifest = {
                "candidate_id": token.candidate_id,
                "predictor": token.predictor,
                "model_sha256": {str(model.seed): model.model_sha256 for model in models},
            }
            _write_json(self.output / "full_fit/full_fit_manifest.json", manifest)
            return PhaseOutcome("passed", tuple(f"full_fit_s{seed}" for seed in token.seeds), (), manifest, {"full_fit": "full_fit/full_fit_manifest.json"})

        if phase == "AUDIT":
            method = str(blend_payload["method"]) if token.predictor == "blend" else "catboost"
            weight = float(blend_payload["catboost_weight"]) if token.predictor == "blend" else 1.0
            runtime = load_inference_runtime(
                token=token,
                frozen_state_dir=self.output / "full_fit/frozen_state",
                catboost_model_dir=self.output / "full_fit/models",
                blend_method=method,
                catboost_weight=weight,
                tabm_dir=self.evidence.tabm_runtime_root if token.predictor == "blend" else None,
            )
            audit_rows = build_inference_audit_rows(
                self.data.train,
                row_count=int(self.contract.runtime["inference_rows"]),
            )
            audit = audit_inference(runtime, audit_rows, self.contract)
            _write_json(self.output / "audits/inference.json", audit)
            return PhaseOutcome(
                "passed" if audit.status == "passed" else "failed",
                ("inference_audit",) if audit.status == "passed" else (),
                () if audit.status == "passed" else ("inference_audit",),
                _jsonable(audit),
                {"inference_audit": "audits/inference.json"},
            )
        raise E2ProductionError(f"unsupported phase: {phase}")

    def _all_sources(self, *, include_models: bool) -> dict[str, Path]:
        excluded = {"stage_state.json"}
        output: dict[str, Path] = {}
        for path in sorted(self.output.rglob("*")):
            relative_parts = path.relative_to(self.output).parts
            if (
                not path.is_file()
                or path.is_symlink()
                or any(
                    part in {"bundles", "cache", "verified_input", "runtime", "materialized"}
                    for part in relative_parts
                )
                or path.name in excluded
                or path == self.log
            ):
                continue
            relative = path.relative_to(self.output).as_posix()
            if not include_models and path.suffix in {".cbm", ".cbsnapshot", ".pt"}:
                continue
            output[relative] = path
        return output

    def _delivery(self) -> E2DeliverySources:
        token = _token_from_path(self.output / "decisions/accepted_token.json")
        models = {
            seed: self.output / f"full_fit/models/catboost_seed_{seed}.cbm"
            for seed in token.seeds
        }
        return E2DeliverySources(
            token=token,
            models=MappingProxyType(models),
            frozen_state=self.output / "full_fit/frozen_state",
            full_fit_manifest=self.output / "full_fit/full_fit_manifest.json",
            acceptance_decision=self.output / "decisions/acceptance.json",
            inference_audit=self.output / "audits/inference.json",
            tabm_root=self.evidence.tabm_runtime_root if token.predictor == "blend" else None,
        )

    def publish(self, state: CampaignState, output_dir: Path) -> object:
        if not self.log.is_file():
            _append(self.log, "TREE_E2_LOG_READY")
        return write_e2_bundles(
            output_dir=output_dir,
            bindings=state.bindings,
            state_path=self.output / "stage_state.json",
            log_path=self.log,
            review_sources=self._all_sources(include_models=False),
            resume_sources=self._all_sources(include_models=True),
            delivery_sources=self._delivery() if state.status == "accepted" else None,
        )


def _restore_resume(path: Path, output: Path, bindings: Mapping[str, str]) -> Path:
    verify_e2_resume(path, bindings)
    with ZipFile(path) as archive:
        for name in archive.namelist():
            if name == "manifest.json":
                continue
            target_name = "stage_state.json" if name == "state/stage_state.json" else name
            target = output / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    return output / "stage_state.json"


def run_production_campaign(
    *,
    official_root: Path,
    compact_input: Path,
    resume_bundle: Path | None,
    output_dir: Path,
    absolute_deadline: float,
) -> CampaignResult:
    contract = load_e2_contract()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = verify_official_data(official_root, contract)
    verified_root = output / "verified_input"
    if verified_root.is_dir():
        shutil.rmtree(verified_root)
    evidence = verify_and_extract_e2_input(compact_input, verified_root, contract=contract)
    repository = Path(__file__).resolve().parents[2]
    bindings = {
        "contract_sha256": file_sha256(contract.source_path),
        "code_sha256": runtime_identity_sha256(repository),
        "input_sha256": evidence.archive_sha256,
        "train_sha256": data.train_sha256,
        "history_sha256": data.history_sha256,
    }
    state_path = output / "stage_state.json"
    if resume_bundle is not None and not state_path.is_file():
        _restore_resume(Path(resume_bundle), output, bindings)
    runtime = ProductionRuntime(data=data, evidence=evidence, output=output, contract=contract)
    return run_e2_campaign(
        runtime=runtime,
        output_dir=output,
        bindings=bindings,
        wall_deadline=absolute_deadline,
        state_path=state_path,
        new_job_guard_seconds=contract.new_job_guard_seconds,
        snapshot_interval_seconds=contract.snapshot_interval_seconds,
        clock=time.time,
    )
