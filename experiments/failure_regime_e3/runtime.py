from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import pickle
import shutil
import threading
import time
from types import MappingProxyType

import numpy as np
import pandas as pd

from experiments.direct_expert.features import feature_profile, fit_direct_features, transform_direct_features
from experiments.direct_expert.inputs import EXPECTED_E2_SUBMISSION_SHA256, file_sha256
from experiments.direct_expert.kaggle import ProductionStageARuntime

from .artifacts import E3Bindings, canonical_json
from .contracts import RoleSpec, load_contract, role_spec
from .labels import FailureTargets, recover_failure_targets
from .meta import build_meta_frame
from .selection import (
    CandidateEvidence,
    GateRecipe,
    apply_recipe,
    candidate_evidence,
    crossfit_gate_probability,
    decide as decide_candidate,
    fit_confirmation_gate,
    fit_frozen_recipe,
    gate_features,
)
from .training import (
    E3Job,
    FoldData,
    catboost_parameters,
    role_training_target,
    run_fold_job,
    season_weights,
)


class E3RuntimeError(RuntimeError):
    pass


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class _GateModel:
    def __init__(self, seed: int, *, gpu: int = 0):
        from catboost import CatBoostClassifier

        parameters = dict(load_contract().gate_parameters)
        parameters.update(
            loss_function="Logloss",
            eval_metric="BrierScore",
            random_seed=int(seed),
            task_type="GPU",
            devices=str(gpu),
            allow_writing_files=False,
            verbose=False,
        )
        self.model = CatBoostClassifier(**parameters)
        self.columns: tuple[str, ...] | None = None
        self.categorical: tuple[int, ...] = ()

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> None:
        self.columns = tuple(frame.columns)
        self.categorical = tuple(
            index for index, name in enumerate(self.columns)
            if pd.api.types.is_object_dtype(frame[name].dtype) or pd.api.types.is_string_dtype(frame[name].dtype)
        )
        self.model.fit(frame, target, cat_features=list(self.categorical))

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.columns is None or tuple(frame.columns) != self.columns:
            raise E3RuntimeError("gate feature schema differs")
        return np.asarray(self.model.predict_proba(frame), dtype="float64")

    def save(self, model_path: Path, schema_path: Path) -> None:
        if self.columns is None:
            raise E3RuntimeError("gate is not fitted")
        self.model.save_model(str(model_path))
        _atomic(schema_path, canonical_json({
            "columns": list(self.columns),
            "categorical_indices": list(self.categorical),
        }))


def _evidence_payload(evidence: CandidateEvidence) -> dict[str, object]:
    return {
        **asdict(evidence),
        "fold_gains": {str(year): gain for year, gain in evidence.fold_gains.items()},
    }


class ProductionE3Runtime:
    """Production runtime using only official training/TrackMan and verified E2 artifacts."""

    def __init__(self, official_root: Path, campaign_input: Path, work_root: Path, *, code_sha256: str):
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.base = ProductionStageARuntime(official_root, campaign_input, self.work_root / "direct_base")
        contract_path = Path(__file__).with_name("contract.json")
        self.bindings = E3Bindings(
            contract_sha256=file_sha256(contract_path),
            code_sha256=code_sha256,
            input_manifest_sha256=self.base.verified.manifest_sha256,
            train_sha256=self.base.bindings.train_sha256,
            history_sha256=self.base.bindings.history_sha256,
            e2_submission_sha256=EXPECTED_E2_SUBMISSION_SHA256,
        )
        self.train = self.base.train
        self.history = self.base.history
        self._folds: OrderedDict[int, FoldData] = OrderedDict()
        self._fold_lock = threading.Lock()
        self._full_lock = threading.Lock()
        self._full_cache = None
        self._model_cache: dict[str, object] = {}

    def now(self) -> float:
        return time.time()

    def can_start_job(self) -> bool:
        return shutil.disk_usage(self.work_root).free >= int(load_contract().runtime["minimum_free_bytes"])

    def compact_oof_job(self, job: E3Job, output: Path) -> None:
        del job
        (Path(output) / "model.cbm").unlink(missing_ok=True)
        (Path(output) / "training.snapshot").unlink(missing_ok=True)
        shutil.rmtree(Path(output) / "catboost_info", ignore_errors=True)

    def _store_fold(self, valid_year: int, fold: FoldData) -> None:
        self._folds[valid_year] = fold
        self._folds.move_to_end(valid_year)
        while len(self._folds) > 2:
            self._folds.popitem(last=False)

    def _prepare_fold_slot(self, valid_year: int) -> None:
        if valid_year in self._folds:
            self._folds.move_to_end(valid_year)
            return
        while len(self._folds) >= 2:
            self._folds.popitem(last=False)

    def release_training_cache(self) -> None:
        with self._fold_lock:
            self._folds.clear()
        with self._full_lock:
            self._full_cache = None
        gc.collect()

    def _labels(self, rows: pd.DataFrame, valid_year: int) -> FailureTargets:
        contract = load_contract()
        result = recover_failure_targets(
            rows,
            valid_year=valid_year,
            tolerance=float(contract.label_gate["delta_tolerance"]),
        )
        gates = contract.label_gate
        if (
            result.coverage < gates["minimum_coverage"]
            or result.binary_fraction < gates["minimum_binary_fraction"]
            or result.success_agreement < gates["minimum_success_agreement"]
            or any(value < gates["minimum_positive_rows"] for value in result.positive_counts.values())
        ):
            raise E3RuntimeError(
                "failure label gate failed: "
                f"coverage={result.coverage:.6f} binary={result.binary_fraction:.6f} "
                f"agreement={result.success_agreement:.6f} positives={dict(result.positive_counts)}"
            )
        return result

    def _fold(self, valid_year: int) -> FoldData:
        with self._fold_lock:
            if valid_year not in self._folds:
                self._prepare_fold_slot(valid_year)
                gc.collect()
                seasons = pd.to_numeric(self.train["season"], errors="raise")
                prefix = self.train.loc[seasons.lt(valid_year)].copy()
                valid_source = self.train.loc[seasons.eq(valid_year)].copy()
                state, train_batch = fit_direct_features(prefix, self.history, valid_year=valid_year)
                labels = self._labels(prefix, valid_year)
                target = pd.to_numeric(valid_source.pop("control_success"), errors="raise").to_numpy(dtype="int8")
                valid_batch = transform_direct_features(valid_source, state)
                metadata = valid_source.loc[:, ["row_id", "game_type", "pitcher_id"]].copy()
                metadata["oof_year"] = valid_year
                self._store_fold(valid_year, FoldData(
                    train=train_batch,
                    valid=valid_batch,
                    valid_target=target,
                    valid_metadata=metadata,
                    categorical_columns=state.categorical_columns,
                    subtype_targets=labels.frame,
                    bindings=MappingProxyType({
                        "contract": self.bindings.contract_sha256,
                        "code": self.bindings.code_sha256,
                        "train": self.bindings.train_sha256,
                        "history": self.bindings.history_sha256,
                        "input": self.bindings.input_manifest_sha256,
                    }),
                ))
            self._folds.move_to_end(valid_year)
            return self._folds[valid_year]

    def run_oof_job(self, job: E3Job, gpu: int, output: Path) -> None:
        print(
            f"E3_JOB_START phase={job.phase} role={job.role_id} fold={job.fold[0]}->{job.fold[1]} "
            f"seed={job.seed} gpu={gpu}",
            flush=True,
        )
        result = run_fold_job(job, self._fold(job.fold[1]), output, gpu=gpu)
        print(f"E3_JOB_END job={job.job_id} status={result.status} brier={result.brier}", flush=True)

    @staticmethod
    def _prediction_path(root: Path, phase: str, role: str, year: int, seed: int) -> Path:
        fold_start = year - 1
        job_id = f"{phase}__{role}__tr{fold_start}__va{year}__s{seed}"
        return root / "jobs" / job_id / "predictions.csv"

    def _role_stream(self, campaign_root: Path, role: str, year: int, seed: int) -> pd.DataFrame:
        if seed == 3407:
            phase = "screening" if year in load_contract().selection_years else "confirmation"
        else:
            phase = "extra_seeds"
        path = self._prediction_path(campaign_root, phase, role, year, seed)
        if not path.is_file():
            raise E3RuntimeError(f"OOF stream is absent: {path.name}")
        frame = pd.read_csv(path, usecols=["row_id", "probability", "game_type"])
        if role in {"S_R", "S_F"}:
            global_frame = self._role_stream(campaign_root, "S_GLOBAL", year, seed)
            active = "R" if role == "S_R" else "F"
            fallback = global_frame.set_index(global_frame["row_id"].astype(str))["probability"]
            inactive = frame["game_type"].ne(active)
            frame.loc[inactive, "probability"] = frame.loc[inactive, "row_id"].astype(str).map(fallback)
        return frame.loc[:, ["row_id", "probability"]]

    def _meta(self, campaign_root: Path, seed: int, years: tuple[int, ...] = (2022, 2023, 2024)) -> pd.DataFrame:
        pieces = []
        seasons = pd.to_numeric(self.train["season"], errors="raise")
        names = {
            "S_GLOBAL": "p_s_global", "S_FAST": "p_s_fast", "S_R": "p_s_r", "S_F": "p_s_f",
            "MIDDLE": "p_middle", "WILD": "p_wild", "REVERSE": "p_reverse",
        }
        for year in years:
            rows = self.train.loc[seasons.eq(year)].copy()
            rows["target"] = pd.to_numeric(rows.pop("control_success"), errors="raise").astype("int8")
            rows["oof_year"] = year
            e2 = self.base.e2_oof(year).loc[:, ["row_id", "p_anchor"]].rename(columns={"p_anchor": "probability"})
            streams = {"p_e2": e2}
            streams.update({name: self._role_stream(campaign_root, role, year, seed) for role, name in names.items()})
            pieces.append(build_meta_frame(rows, streams, include_target=True))
        return pd.concat(pieces, ignore_index=True)

    @staticmethod
    def _factory(seed: int, *, gpu: int = 0):
        return _GateModel(seed, gpu=gpu)

    def select_recipe(self, output: Path) -> str:
        meta = self._meta(Path(output).parents[1], 3407, load_contract().selection_years)
        frozen = fit_frozen_recipe(meta, estimator_factory=lambda seed: self._factory(seed, gpu=0))
        payload = {
            "schema_version": 1,
            "recipe_id": frozen.recipe.identity_sha256,
            "e2_weight": frozen.recipe.e2_weight,
            "gate_strength": frozen.recipe.gate_strength,
            "selection_years": list(frozen.recipe.selection_years),
            "selection_brier": frozen.selection_brier,
        }
        _atomic(output, canonical_json(payload))
        print(
            f"E3_RECIPE_SELECTED id={frozen.recipe.identity_sha256[:12]} "
            f"e2_weight={frozen.recipe.e2_weight} strength={frozen.recipe.gate_strength}",
            flush=True,
        )
        return frozen.recipe.identity_sha256

    @staticmethod
    def _read_recipe(path: Path, recipe_id: str) -> GateRecipe:
        payload = json.loads(path.read_text(encoding="utf-8"))
        recipe = GateRecipe(float(payload["e2_weight"]), float(payload["gate_strength"]), tuple(payload["selection_years"]))
        if recipe.identity_sha256 != recipe_id or payload["recipe_id"] != recipe_id:
            raise E3RuntimeError("recipe identity differs")
        return recipe

    def _seed_probability(self, meta: pd.DataFrame, recipe: GateRecipe, seed: int) -> np.ndarray:
        contract = load_contract()
        selection_mask = meta["oof_year"].isin(contract.selection_years).to_numpy()
        confirmation_mask = meta["oof_year"].eq(contract.confirmation_year).to_numpy()
        gate = np.full(len(meta), np.nan, dtype="float64")
        selection = meta.loc[selection_mask].reset_index(drop=True)
        gate[selection_mask] = crossfit_gate_probability(
            selection,
            estimator_factory=lambda offset: self._factory(seed + offset % 1000, gpu=seed % 2),
        )
        confirmation_model = fit_confirmation_gate(
            meta,
            estimator_factory=lambda _: self._factory(seed, gpu=seed % 2),
            seed=seed,
        )
        gate[confirmation_mask] = confirmation_model.predict_proba(
            gate_features(meta.loc[confirmation_mask])
        )[:, 1]
        if not np.isfinite(gate).all():
            raise E3RuntimeError("seed gate probability is incomplete")
        return apply_recipe(recipe, meta["p_e2"].to_numpy(dtype="float64"), gate)

    def decide(self, output: Path, recipe_id: str) -> str:
        campaign_root = Path(output).parents[1]
        recipe = self._read_recipe(output.with_name("recipe.json"), recipe_id)
        seeds = load_contract().seeds
        metas = {seed: self._meta(campaign_root, seed) for seed in seeds}
        reference_ids = metas[3407]["row_id"].tolist()
        if any(frame["row_id"].tolist() != reference_ids for frame in metas.values()):
            raise E3RuntimeError("seed OOF alignment differs")
        probabilities = {seed: self._seed_probability(metas[seed], recipe, seed) for seed in seeds}
        ensemble = np.mean(list(probabilities.values()), axis=0)
        evidence = candidate_evidence(metas[3407], ensemble, seed_probabilities=probabilities)
        decision = decide_candidate(evidence)
        for seed, probability in probabilities.items():
            pd.DataFrame({"row_id": reference_ids, "probability": probability}).to_csv(
                output.parent / f"candidate_seed_{seed}.csv", index=False
            )
        pd.DataFrame({"row_id": reference_ids, "probability": ensemble}).to_csv(
            output.parent / "candidate_ensemble.csv", index=False
        )
        _atomic(output, canonical_json({
            "schema_version": 1,
            "status": decision.status,
            "failed_gates": list(decision.failed_gates),
            "recipe_id": recipe_id,
            "evidence": _evidence_payload(evidence),
        }))
        print(
            f"E3_DECISION status={decision.status} weighted_gain={evidence.weighted_gain:.10f} "
            f"latest_gain={evidence.latest_gain:.10f} failed={','.join(decision.failed_gates) or 'none'}",
            flush=True,
        )
        return decision.status

    def _full_data(self):
        with self._full_lock:
            if self._full_cache is None:
                with self._fold_lock:
                    self._folds.clear()
                gc.collect()
                valid_year = int(pd.to_numeric(self.train["season"], errors="raise").max()) + 1
                state, batch = fit_direct_features(self.train, self.history, valid_year=valid_year)
                labels = self._labels(self.train, valid_year)
                shared = self.work_root / "full_shared"
                shared.mkdir(parents=True, exist_ok=True)
                _atomic(shared / "feature_state.pkl", pickle.dumps(state, protocol=5))
                _atomic(shared / "label_audit.json", canonical_json({
                    "coverage": labels.coverage,
                    "binary_fraction": labels.binary_fraction,
                    "success_agreement": labels.success_agreement,
                    "overlap_rate": labels.overlap_rate,
                    "positive_counts": dict(labels.positive_counts),
                }))
                self._full_cache = (valid_year, state, batch, labels.frame)
            return self._full_cache

    def run_full_fit(self, role_id: str, seed: int, gpu: int, output: Path) -> None:
        output.mkdir(parents=True, exist_ok=True)
        if role_id == "GATE":
            meta = self._meta(output.parents[1], seed)
            model = self._factory(seed, gpu=gpu)
            model.fit(gate_features(meta), meta["target"].to_numpy(dtype="int8"))
            model.save(output / "model.cbm", output / "schema.json")
            _atomic(output / "metrics.json", canonical_json({"role_id": role_id, "seed": seed, "rows": len(meta)}))
            return
        from catboost import CatBoostClassifier

        valid_year, _, batch, subtypes = self._full_data()
        shared = output.parents[1] / "full_fit" / "shared"
        with self._full_lock:
            shared.mkdir(parents=True, exist_ok=True)
            state_payload = pickle.dumps(self._full_cache[1], protocol=5)
            if not (shared / "feature_state.pkl").is_file():
                _atomic(shared / "feature_state.pkl", state_payload)
        spec: RoleSpec = role_spec(role_id)
        target, mask = role_training_target(spec, batch, subtypes)
        weights = season_weights(spec, batch.season, valid_year)
        mask &= weights > 0
        # The fitted state owns the authoritative categorical schema.
        categorical_columns = self._full_cache[1].categorical_columns
        columns, categorical = feature_profile(batch.frame, categorical_columns, spec.profile)
        parameters = catboost_parameters(spec, seed=seed, gpu=gpu, output=output)
        model = CatBoostClassifier(**parameters)
        model.fit(
            batch.frame.loc[mask, columns],
            target[mask],
            sample_weight=weights[mask],
            cat_features=[columns.index(name) for name in categorical],
            use_best_model=False,
        )
        model.save_model(str(output / "model.cbm"))
        _atomic(output / "schema.json", canonical_json({
            "role_id": role_id,
            "seed": seed,
            "columns": list(columns),
            "categorical_columns": list(categorical),
            "rows": int(mask.sum()),
        }))
        print(f"E3_FULL_FIT_END role={role_id} seed={seed} gpu={gpu}", flush=True)

    def _load_catboost(self, path: Path):
        from catboost import CatBoostClassifier

        key = str(Path(path).resolve())
        if key not in self._model_cache:
            model = CatBoostClassifier()
            model.load_model(key)
            self._model_cache[key] = model
        return self._model_cache[key]

    def _full_predict(self, campaign_root: Path, rows: pd.DataFrame, p_e2: pd.DataFrame) -> np.ndarray:
        state_path = campaign_root / "full_fit" / "shared" / "feature_state.pkl"
        if not state_path.is_file():
            raise E3RuntimeError("full-fit feature state is absent")
        state = pickle.loads(state_path.read_bytes())
        evaluation = rows.drop(columns=["control_success"], errors="ignore")
        batch = transform_direct_features(evaluation, state)
        recipe = self._read_recipe(campaign_root / "selection" / "recipe.json", json.loads(
            (campaign_root / "selection" / "recipe.json").read_text(encoding="utf-8")
        )["recipe_id"])
        seed_predictions = []
        for seed in load_contract().seeds:
            role_values: dict[str, np.ndarray] = {}
            for spec in load_contract().roles:
                model_root = campaign_root / "full_fit" / f"full_fit__{spec.role_id}__s{seed}"
                schema = json.loads((model_root / "schema.json").read_text(encoding="utf-8"))
                columns = tuple(schema["columns"])
                model = self._load_catboost(model_root / "model.cbm")
                probability = np.asarray(model.predict_proba(batch.frame.loc[:, columns]), dtype="float64")[:, 1]
                role_values[spec.role_id] = probability
            is_r = np.asarray(batch.game_type == "R")
            is_f = np.asarray(batch.game_type == "F")
            role_values["S_R"] = np.where(is_r, role_values["S_R"], role_values["S_GLOBAL"])
            role_values["S_F"] = np.where(is_f, role_values["S_F"], role_values["S_GLOBAL"])
            stream_names = {
                "S_GLOBAL": "p_s_global", "S_FAST": "p_s_fast", "S_R": "p_s_r", "S_F": "p_s_f",
                "MIDDLE": "p_middle", "WILD": "p_wild", "REVERSE": "p_reverse",
            }
            streams = {
                "p_e2": p_e2.loc[:, ["row_id", "probability"]],
                **{
                    name: pd.DataFrame({"row_id": evaluation["row_id"].astype(str), "probability": role_values[role]})
                    for role, name in stream_names.items()
                },
            }
            meta = build_meta_frame(evaluation, streams, include_target=False)
            gate_root = campaign_root / "full_fit" / f"full_fit__GATE__s{seed}"
            schema = json.loads((gate_root / "schema.json").read_text(encoding="utf-8"))
            features = gate_features(meta).loc[:, schema["columns"]]
            gate = self._load_catboost(gate_root / "model.cbm")
            gate_probability = np.asarray(gate.predict_proba(features), dtype="float64")[:, 1]
            seed_predictions.append(apply_recipe(recipe, meta["p_e2"].to_numpy(dtype="float64"), gate_probability))
        output = np.mean(seed_predictions, axis=0)
        if output.shape != (len(rows),) or not np.isfinite(output).all():
            raise E3RuntimeError("full-fit probability differs")
        return output

    def audit(self, output: Path) -> bool:
        self.release_training_cache()
        started = time.monotonic()
        campaign_root = Path(output).parents[1]
        seasons = pd.to_numeric(self.train["season"], errors="raise")
        source = self.train.loc[seasons.eq(load_contract().confirmation_year)].head(12).copy()
        e2 = self.base.e2_oof(load_contract().confirmation_year).loc[:, ["row_id", "p_anchor"]]
        e2 = e2.rename(columns={"p_anchor": "probability"})
        e2_key = e2.set_index(e2["row_id"].astype(str))["probability"]
        anchor = pd.DataFrame({
            "row_id": source["row_id"].astype(str),
            "probability": source["row_id"].astype(str).map(e2_key),
        })
        baseline = pd.Series(self._full_predict(campaign_root, source, anchor), index=source["row_id"].astype(str))
        reversed_rows = source.iloc[::-1].reset_index(drop=True)
        reversed_anchor = anchor.iloc[::-1].reset_index(drop=True)
        reversed_prediction = pd.Series(
            self._full_predict(campaign_root, reversed_rows, reversed_anchor),
            index=reversed_rows["row_id"].astype(str),
        )
        singleton = {}
        for index in range(len(source)):
            one_row = source.iloc[[index]].reset_index(drop=True)
            one_anchor = anchor.iloc[[index]].reset_index(drop=True)
            singleton[str(one_row.loc[0, "row_id"])] = float(self._full_predict(campaign_root, one_row, one_anchor)[0])
        singleton_series = pd.Series(singleton)
        reorder_max = float((baseline.sort_index() - reversed_prediction.sort_index()).abs().max())
        singleton_max = float((baseline.sort_index() - singleton_series.sort_index()).abs().max())
        elapsed = time.monotonic() - started
        passed = bool(
            np.isfinite(baseline.to_numpy()).all()
            and reorder_max <= 1e-12
            and singleton_max <= 1e-12
            and elapsed <= load_contract().inference_max_seconds
        )
        payload = {
            "schema_version": 1,
            "passed": passed,
            "scope": "full_fit_row_independence",
            "policy": load_contract().policy_version,
            "rows": len(source),
            "reorder_max_abs": reorder_max,
            "singleton_max_abs": singleton_max,
            "elapsed_seconds": elapsed,
            "submission_package_created": False,
            "note": "Deployable submission-script canary remains mandatory before packaging.",
        }
        _atomic(output, canonical_json(payload))
        print(
            f"E3_AUDIT status={'passed' if passed else 'failed'} reorder={reorder_max:.3e} "
            f"singleton={singleton_max:.3e} elapsed={elapsed:.1f}",
            flush=True,
        )
        return passed
