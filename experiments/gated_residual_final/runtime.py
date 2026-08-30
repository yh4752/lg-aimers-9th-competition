from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
from statistics import median
import time
from typing import Callable, Mapping
from zipfile import ZipFile

import numpy as np
import pandas as pd

from experiments.direct_expert.compliance import audit_source
from experiments.direct_expert.contracts import expert_spec, load_contract as load_direct_contract
from experiments.direct_expert.features import feature_profile, fit_direct_features
from experiments.direct_expert.kaggle import ProductionStageARuntime
from experiments.direct_expert.training import catboost_parameters, season_weights, training_mask

from .artifacts import ArtifactBindings
from .calibration import fit_temporal_calibrator
from .compliance import audit_row_independence, audit_temporal_sources
from .contracts import load_contract
from .evidence import average_seed_frames, build_seed_frame, candidate_probability
from .inference import FinalPredictor
from .inputs import VerifiedFinalInput, canonical_json, file_sha256
from .runner import AuditOutcome, FullFitOutcome, SelectionOutcome
from .runtime_inventory import code_identity_sha256, runtime_members
from .selection import (
    CandidateDecision,
    FrozenCandidate,
    decide,
    evidence_from_predictions,
    freeze_archetypes,
)


class FinalRuntimeError(ValueError):
    pass


def _prediction_frame(source: pd.DataFrame, probability) -> pd.DataFrame:
    required = {
        "row_id", "target", "p_anchor", "oof_year", "pitcher_id", "game_type", "hand_matchup",
    }
    if not required.issubset(source.columns):
        raise FinalRuntimeError("candidate source columns differ")
    output = source.loc[:, sorted(required)].copy(deep=True)
    output["probability"] = probability
    return output


def _evidence_payload(evidence) -> dict[str, object]:
    return {
        "candidate_id": evidence.candidate_id,
        "fold_gains": {str(year): value for year, value in evidence.fold_gains.items()},
        "weighted_gain": evidence.weighted_gain,
        "latest_gain": evidence.latest_gain,
        "minimum_fold_gain": evidence.minimum_fold_gain,
        "bootstrap_lower": evidence.bootstrap_lower,
        "non_worse_seed_count": evidence.non_worse_seed_count,
        "latest_non_worse_seed_count": evidence.latest_non_worse_seed_count,
        "maximum_segment_regression": evidence.maximum_segment_regression,
        "finite_probabilities": evidence.finite_probabilities,
    }


def evaluate_candidates(
    seed_frames: Mapping[int, pd.DataFrame],
    output_dir: Path,
    *,
    bootstrap_repetitions: int = 1000,
) -> tuple[SelectionOutcome, tuple[FrozenCandidate, ...]]:
    averaged = average_seed_frames(seed_frames)
    frozen = freeze_archetypes(
        averaged,
        predictor=lambda frame, config: candidate_probability(frame, config)[0],
    )
    records = []
    evaluated = []
    for candidate in frozen:
        probability, temporal_sources = candidate_probability(averaged, candidate.config)
        candidate_frame = _prediction_frame(averaged, probability)
        per_seed = {}
        for seed, frame in sorted(seed_frames.items()):
            seed_probability, seed_sources = candidate_probability(frame, candidate.config)
            if seed_sources != temporal_sources:
                raise FinalRuntimeError(f"temporal source evidence differs: seed={seed}")
            per_seed[seed] = _prediction_frame(frame, seed_probability)
        evidence = evidence_from_predictions(
            candidate.config.candidate_id,
            candidate_frame,
            seed_predictions=per_seed,
            bootstrap_repetitions=bootstrap_repetitions,
        )
        decision = decide(evidence)
        evaluated.append((decision, evidence, candidate))
        records.append({
            "candidate_id": candidate.config.candidate_id,
            "config": asdict(candidate.config),
            "locked_years": list(candidate.locked_years),
            "tuning_objective": candidate.tuning_objective,
            "temporal_sources": {str(year): list(values) for year, values in temporal_sources.items()},
            "decision": asdict(decision),
            "evidence": _evidence_payload(evidence),
        })
    accepted = [item for item in evaluated if item[0].status == "accepted"]
    pool = accepted or evaluated
    chosen = max(
        pool,
        key=lambda item: (
            item[1].weighted_gain,
            item[1].latest_gain,
            item[1].bootstrap_lower,
            item[0].candidate_id,
        ),
    )
    chosen_decision = chosen[0]
    status = "accepted" if accepted else "completed_no_candidate"
    payload = {
        "schema_version": 1,
        "status": status,
        "selection_years": [2022, 2023],
        "confirmation_year": 2024,
        "chosen_candidate_id": chosen_decision.candidate_id,
        "candidates": records,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evidence_path = output / "selection_evidence.json"
    evidence_path.write_bytes(canonical_json(payload))
    return SelectionOutcome(chosen_decision, evidence_path), frozen


def _best_iterations(stage_a: Path, stage_b: Path, role: str) -> int:
    observed: list[int] = []
    for source in (Path(stage_a), Path(stage_b)):
        with ZipFile(source) as archive:
            for name in archive.namelist():
                if f"__{role}__" not in name or not name.endswith("/metrics.json"):
                    continue
                payload = json.loads(archive.read(name))
                value = payload.get("best_iteration")
                if type(value) is int and value >= 0:
                    observed.append(value + 1)
    if not observed:
        raise FinalRuntimeError(f"best iteration evidence is absent: {role}")
    return min(load_contract().maximum_iterations, int(median(observed)))


class ProductionFinalRuntime:
    def __init__(
        self,
        *,
        official_root: Path,
        verified_input: VerifiedFinalInput,
        work_root: Path,
    ):
        self.root = Path(work_root)
        self.root.mkdir(parents=True, exist_ok=True)
        contract = load_contract()
        expected_sources = {
            "e2_input": contract.expected_hashes["direct_expert_input"],
            "stage_a": contract.expected_hashes["stage_a_handoff"],
            "stage_b": contract.expected_hashes["stage_b_handoff"],
        }
        if dict(verified_input.source_hashes) != expected_sources:
            raise FinalRuntimeError("final input source bindings differ")
        self.verified = verified_input
        self.base = ProductionStageARuntime(
            Path(official_root), verified_input.e2_input, self.root / "direct_base",
        )
        repository = Path(__file__).resolve().parents[2]
        self.bindings = ArtifactBindings(
            code_sha256=code_identity_sha256(repository),
            contract_sha256=file_sha256(Path(__file__).with_name("contract.json")),
            input_manifest_sha256=verified_input.manifest_sha256,
            train_sha256=contract.expected_hashes["train_csv"],
            history_sha256=contract.expected_hashes["trackman_history"],
        )
        self.seed_frames: dict[int, pd.DataFrame] | None = None
        self.selected_config = None
        self._feature_state = None
        self._calibrator = None
        self._pitcher_counts: dict[str, int] = {}
        self._batter_counts: dict[str, int] = {}
        self._model_paths: dict[str, Path] = {}

    def now(self) -> float:
        return time.time()

    def _frames(self) -> Mapping[int, pd.DataFrame]:
        if self.seed_frames is None:
            e2 = {year: self.base.e2_oof(year) for year in (2022, 2023, 2024)}
            self.seed_frames = {
                seed: build_seed_frame(
                    e2_by_year=e2,
                    stage_a=self.verified.stage_a,
                    stage_b=self.verified.stage_b,
                    seed=seed,
                    train_for_counts=self.base.train,
                )
                for seed in load_contract().full_fit_seeds
            }
        return self.seed_frames

    def select(self, output: Path) -> SelectionOutcome:
        outcome, frozen = evaluate_candidates(self._frames(), output)
        configs = {item.config.candidate_id: item.config for item in frozen}
        if outcome.decision.candidate_id not in configs:
            raise FinalRuntimeError("selected candidate config is absent")
        self.selected_config = configs[outcome.decision.candidate_id]
        return outcome

    def _prepare_production_state(self, output: Path) -> dict[str, Path]:
        if self.selected_config is None:
            raise FinalRuntimeError("selected candidate is absent")
        self.base.clear_fold_cache()
        state, batch = fit_direct_features(
            self.base.train.copy(deep=True), self.base.history, valid_year=2025,
        )
        self._feature_state = state
        self._full_batch = batch
        pitchers = self.base.train["pitcher_id"].astype(str).value_counts(sort=False).astype(int).to_dict()
        batters = self.base.train["batter_id"].astype(str).value_counts(sort=False).astype(int).to_dict()
        self._pitcher_counts = pitchers
        self._batter_counts = batters
        averaged = average_seed_frames(self._frames())
        base_probability, temporal_sources = candidate_probability(averaged, self.selected_config)
        if self.selected_config.archetype in {"G2", "G3"}:
            audit_temporal_sources({year: values for year, values in temporal_sources.items() if values})
            work = averaged.copy(deep=True)
            work["probability"] = base_probability
            hierarchy = "global_game" if self.selected_config.archetype == "G2" else "full"
            self._calibrator = fit_temporal_calibrator(
                work, validation_year=2025, hierarchy=hierarchy,
                ridge=int(self.selected_config.ridge),
            )
        state_root = Path(output) / "state"
        state_root.mkdir(parents=True, exist_ok=True)
        paths = {
            "state/feature_state.pkl": state_root / "feature_state.pkl",
            "state/calibration.pkl": state_root / "calibration.pkl",
            "state/config.json": state_root / "config.json",
            "state/counts.json": state_root / "counts.json",
        }
        paths["state/feature_state.pkl"].write_bytes(pickle.dumps(state, protocol=5))
        paths["state/calibration.pkl"].write_bytes(pickle.dumps(self._calibrator, protocol=5))
        paths["state/config.json"].write_bytes(canonical_json(asdict(self.selected_config)))
        paths["state/counts.json"].write_bytes(canonical_json({
            "pitcher": pitchers, "batter": batters,
        }))
        return paths

    def _train_model(self, role: str, seed: int, iterations: int, gpu: int, output: Path) -> Path:
        from catboost import CatBoostClassifier

        spec = expert_spec(role)
        parameters = catboost_parameters(
            spec, load_direct_contract(), seed=seed, gpu=gpu, phase="full_fit",
        )
        parameters["iterations"] = int(iterations)
        parameters["snapshot_file"] = str(output.with_suffix(".snapshot"))
        parameters["train_dir"] = str(output.parent / "catboost_info")
        model = CatBoostClassifier(**parameters)
        weights = season_weights(spec, self._full_batch.season, 2025)
        mask = training_mask(spec, self._full_batch) & (weights > 0)
        columns, categorical = feature_profile(
            self._full_batch.frame,
            self._feature_state.categorical_columns,
            spec.interaction_profile,
        )
        categories = [columns.index(name) for name in categorical]
        output.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"FINAL_CANDIDATE_JOB_START job={role}_s{seed} gpu={gpu} iterations={iterations}",
            flush=True,
        )
        model.fit(
            self._full_batch.frame.loc[mask, columns], self._full_batch.target[mask],
            sample_weight=weights[mask], cat_features=categories,
        )
        model.save_model(str(output))
        return output

    def full_fit(
        self,
        decision: CandidateDecision,
        output: Path,
        *,
        completed_jobs: tuple[str, ...],
        restored_root: Path | None,
        on_job_complete: Callable[[str], None],
        absolute_deadline: float | None,
    ) -> FullFitOutcome:
        if decision.status != "accepted":
            raise FinalRuntimeError("full fit requires acceptance")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        payloads = self._prepare_production_state(output)
        iterations = {
            role: _best_iterations(self.verified.stage_a, self.verified.stage_b, role)
            for role in ("D0", "D5")
        }
        iteration_path = output / "state" / "iterations.json"
        iteration_path.write_bytes(canonical_json(iterations))
        payloads["state/iterations.json"] = iteration_path
        completed = list(completed_jobs)
        tasks = [(role, seed) for role in ("D0", "D5") for seed in load_contract().full_fit_seeds]
        pending = []
        for role, seed in tasks:
            job_id = f"{role}_s{seed}"
            target = output / "jobs" / job_id / "model.cbm"
            if job_id in completed:
                if restored_root is None:
                    raise FinalRuntimeError(f"completed job has no handoff: {job_id}")
                source = restored_root / "jobs" / job_id / "model.cbm"
                if not source.is_file():
                    raise FinalRuntimeError(f"completed model is absent: {job_id}")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                self._model_paths[job_id] = target
                payloads[f"jobs/{job_id}/model.cbm"] = target
            else:
                pending.append((role, seed, target))
        for start in range(0, len(pending), 2):
            if absolute_deadline is not None and self.now() + 1800 >= absolute_deadline:
                return FullFitOutcome(False, tuple(completed), payloads)
            pair = pending[start : start + 2]
            with ThreadPoolExecutor(max_workers=len(pair)) as executor:
                futures = [
                    executor.submit(
                        self._train_model, role, seed, iterations[role], gpu, target,
                    )
                    for gpu, (role, seed, target) in enumerate(pair)
                ]
                for (role, seed, target), future in zip(pair, futures):
                    path = future.result()
                    if path != target or not path.is_file():
                        raise FinalRuntimeError(f"trained model differs: {role}_s{seed}")
                    job_id = f"{role}_s{seed}"
                    completed.append(job_id)
                    self._model_paths[job_id] = path
                    payloads[f"jobs/{job_id}/model.cbm"] = path
                    on_job_complete(job_id)
        payloads["e2/catboost_3seed_v1.zip"] = self.base.verified.e2_submission_path
        requirements = output / "requirements.txt"
        requirements.write_text(
            "catboost==1.2.10\npandas==2.0.3\nnumpy==1.26.4\n", encoding="utf-8",
        )
        payloads["requirements.txt"] = requirements
        repository = Path(__file__).resolve().parents[2]
        for name in runtime_members(repository):
            payloads[f"runtime/{name}"] = repository / name
        return FullFitOutcome(True, tuple(completed), payloads)

    def _load_e2_predictor(self, output: Path):
        e2_root = Path(output) / "e2_runtime"
        if e2_root.exists():
            shutil.rmtree(e2_root)
        e2_root.mkdir(parents=True)
        with ZipFile(self.base.verified.e2_submission_path) as archive:
            archive.extractall(e2_root)
        spec = importlib.util.spec_from_file_location("gated_residual_e2_runtime", e2_root / "script.py")
        if spec is None or spec.loader is None:
            raise FinalRuntimeError("E2 inference module differs")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.load_frozen_predictor(e2_root / "model")

    def _load_models(self):
        from catboost import CatBoostClassifier

        output = {}
        for role in ("D0", "D5"):
            members = []
            for seed in load_contract().full_fit_seeds:
                model = CatBoostClassifier()
                model.load_model(str(self._model_paths[f"{role}_s{seed}"]))
                members.append(model)
            output[role] = tuple(members)
        return output

    def audit(self, full_fit: FullFitOutcome, output: Path) -> AuditOutcome:
        if not full_fit.completed or self._feature_state is None or self.selected_config is None:
            raise FinalRuntimeError("audit production state differs")
        audit_source(Path(__file__).with_name("inference.py").read_text(encoding="utf-8"))
        predictor = FinalPredictor(
            feature_state=self._feature_state,
            models=self._load_models(),
            e2_predictor=self._load_e2_predictor(output),
            pitcher_counts=self._pitcher_counts,
            batter_counts=self._batter_counts,
            config=self.selected_config,
            calibrator=self._calibrator,
        )
        rows = self.base.train.drop(columns=["control_success"], errors="ignore").head(64).copy()
        report = audit_row_independence(predictor, rows)
        path = Path(output) / "audit_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json({
            "schema_version": 1,
            "status": "passed" if report.passed else "failed",
            "row_independence": "passed" if report.passed else "failed",
            "maximum_absolute_difference": report.maximum_absolute_difference,
            "tolerance": report.tolerance,
            "checks": dict(report.checks),
            "code_sha256": self.bindings.code_sha256,
        }))
        return AuditOutcome(report.passed, path)
