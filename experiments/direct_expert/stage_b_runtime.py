from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import threading
from types import MappingProxyType
from typing import Mapping
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .artifacts import extract_bundle
from .compliance import audit_inference, audit_source
from .contracts import ExpertJob, expert_spec, load_contract
from .features import feature_profile, fit_direct_features, transform_direct_features
from .full_fit import accepted_token, full_fit_iterations
from .inference import predict_fixed_recipe
from .kaggle import ProductionStageARuntime
from .runtime_inventory import code_identity_sha256, runtime_members
from .selection import CandidateEvidence, decide, evidence_from_predictions
from .stacking import StackRecipe, apply_stack, fit_e2_safety_blend, fit_stack, required_experts
from .training import catboost_parameters, season_weights, training_mask
from .training import FoldJobResult


class ProductionStageBRuntime:
    def __init__(self, official_root: Path, campaign_input: Path, stage_a_handoff: Path, work_root: Path):
        self.root = Path(work_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.base = ProductionStageARuntime(official_root, campaign_input, self.root / "base")
        self.bindings = self.base.bindings
        self.stage_a_root = self.root / "stage_a"
        extract_bundle(stage_a_handoff, self.stage_a_root, "stage_a", self.bindings)
        selection = json.loads((self.stage_a_root / "selection/locked_selection.json").read_text())
        self._experts = tuple(selection["expert_ids"])
        if len(self._experts) != 4 or len(set(self._experts)) != 4:
            raise ValueError("locked expert count differs")
        self.results = {}
        self._lock = threading.Lock()
        self._chosen_recipe: StackRecipe | None = None
        self._decision: dict[str, object] | None = None
        self._full_state = None
        self._full_rows = None
        self._frame_cache: dict[tuple[str, int, int], pd.DataFrame] = {}
        self._averaged_cache: dict[str, pd.DataFrame] = {}

    def now(self) -> float:
        return self.base.now()

    def locked_experts(self, _handoff: Path) -> tuple[str, ...]:
        return self._experts

    def run_job(self, job: ExpertJob, gpu: int, output: Path) -> Path:
        result = self.base.run_job(job, gpu, output)
        with self._lock:
            self.results[(job.expert_id, job.fold[1], job.seed)] = result
        return result.output_dir / "metrics.json"

    def restore_job(self, job: ExpertJob, folder: Path) -> Path:
        predictions = pd.read_csv(Path(folder) / "predictions.csv")
        metrics = json.loads((Path(folder) / "metrics.json").read_text())
        result = FoldJobResult(
            "completed",
            job.job_id,
            Path(folder),
            predictions,
            float(metrics["brier"]),
            int(metrics["best_iteration"]),
        )
        self.results[(job.expert_id, job.fold[1], job.seed)] = result
        return Path(folder) / "metrics.json"

    def _raw(self, expert: str, year: int, seed: int) -> pd.DataFrame:
        if seed == 3407 and year in (2022, 2023):
            name = f"screen__{expert}__{year - 1}_{year}__s3407"
            return pd.read_csv(self.stage_a_root / f"jobs/{name}/predictions.csv")
        return self.results[(expert, year, seed)].predictions

    def _frame(self, expert: str, year: int, seed: int) -> pd.DataFrame:
        key = (expert, year, seed)
        if key in self._frame_cache:
            return self._frame_cache[key]
        frame = self._raw(expert, year, seed).copy(deep=True)
        if expert in {"D5", "D6"}:
            active = "R" if expert == "D5" else "F"
            fallback = self._raw("D0", year, seed).set_index("row_id")["probability"]
            mask = frame["game_type"].ne(active)
            frame.loc[mask, "probability"] = frame.loc[mask, "row_id"].map(fallback)
        baseline = self.base.e2_oof(year).loc[:, ["row_id", "target", "p_anchor"]]
        merged = frame.merge(baseline, on=["row_id", "target"], validate="one_to_one")
        if len(merged) != len(frame) or len(baseline) != len(frame):
            raise ValueError("E2 OOF alignment differs")
        self._frame_cache[key] = merged
        return merged

    def _averaged(self, expert: str) -> pd.DataFrame:
        if expert in self._averaged_cache:
            return self._averaged_cache[expert]
        frames = []
        for year in (2022, 2023, 2024):
            seeds = [self._frame(expert, year, seed).sort_values("row_id") for seed in (42, 2026, 3407)]
            base = seeds[0].copy(deep=True)
            if any(tuple(item["row_id"]) != tuple(base["row_id"]) for item in seeds[1:]):
                raise ValueError("seed row alignment differs")
            for item in seeds[1:]:
                for column in ("target", "game_type", "pitcher_id", "oof_year", "p_anchor"):
                    if not item[column].reset_index(drop=True).equals(base[column].reset_index(drop=True)):
                        raise ValueError("seed metadata alignment differs")
            base["probability"] = np.mean([item["probability"].to_numpy() for item in seeds], axis=0)
            frames.append(base)
        combined = pd.concat(frames, ignore_index=True)
        self._averaged_cache[expert] = combined
        return combined

    def _recipe_frame(self, recipe: StackRecipe, *, seed: int | None = None) -> pd.DataFrame:
        frames = []
        for year in (2022, 2023, 2024):
            sources = {
                expert: (self._frame(expert, year, seed) if seed else self._averaged(expert).query("oof_year == @year"))
                for expert in required_experts(tuple(recipe.weights))
            }
            template = next(iter(sources.values())).sort_values("row_id").reset_index(drop=True)
            for source in sources.values():
                aligned = source.sort_values("row_id").reset_index(drop=True)
                for column in ("row_id", "target", "game_type", "pitcher_id", "oof_year", "p_anchor"):
                    if not aligned[column].equals(template[column]):
                        raise ValueError("recipe source alignment differs")
            arrays = {
                expert: sources[expert].sort_values("row_id")["probability"].to_numpy()
                for expert in recipe.weights
            }
            e2 = template["p_anchor"].to_numpy() if recipe.e2_weight else None
            template["probability"] = apply_stack(recipe, arrays, e2=e2)
            frames.append(template)
        return pd.concat(frames, ignore_index=True)

    def _evidence(self, candidate_id: str, frame: pd.DataFrame, seed_frames: list[pd.DataFrame]) -> CandidateEvidence:
        seed_evidence = [evidence_from_predictions(f"{candidate_id}_{index}", item) for index, item in enumerate(seed_frames)]
        return evidence_from_predictions(
            candidate_id,
            frame,
            non_worse_seed_count=sum(item.weighted_gain >= 0 for item in seed_evidence),
            improving_latest_seed_count=sum(item.latest_gain > 0 for item in seed_evidence),
        )

    def decide(self, _jobs: Mapping[str, Path]) -> Mapping[str, object]:
        self.base.clear_fold_cache()
        structure = {
            expert: pd.concat([self._frame(expert, year, 3407) for year in (2022, 2023)], ignore_index=True)
            for expert in self._experts
        }
        recipes = [StackRecipe("probability", {expert: 1.0}, (2022, 2023)) for expert in self._experts]
        for method in ("probability", "logit"):
            recipes.append(fit_stack({key: value.loc[:, ["row_id", "target", "probability", "oof_year"]] for key, value in structure.items()}, method=method))
        for direct in tuple(recipes[-2:]):
            if len(required_experts(tuple(direct.weights))) <= 2:
                template = next(iter(structure.values())).sort_values("row_id")
                compact = {
                    key: value.loc[:, ["row_id", "target", "probability", "oof_year"]]
                    for key, value in structure.items()
                }
                recipes.append(fit_e2_safety_blend(direct, compact, template["p_anchor"].to_numpy()))
        evaluated = []
        records = []
        for index, recipe in enumerate(recipes):
            candidate_id = f"recipe_{index:02d}_{recipe.recipe_sha256[:8]}"
            frame = self._recipe_frame(recipe)
            seed_frames = [self._recipe_frame(recipe, seed=seed) for seed in (42, 2026, 3407)]
            evidence = self._evidence(candidate_id, frame, seed_frames)
            decision = decide(evidence)
            evaluated.append((decision, evidence, recipe))
            records.append(
                {
                    "candidate_id": candidate_id,
                    "status": decision.status,
                    "gate": decision.gate,
                    "failed_gates": list(decision.failed_gates),
                    "recipe": {
                        "method": recipe.method,
                        "weights": dict(recipe.weights),
                        "e2_weight": recipe.e2_weight,
                        "recipe_sha256": recipe.recipe_sha256,
                    },
                    "evidence": {
                        **{key: value for key, value in vars(evidence).items() if key != "fold_gains"},
                        "fold_gains": dict(evidence.fold_gains),
                    },
                }
            )
        accepted = [item for item in evaluated if item[0].status.startswith("accepted_")]
        if not accepted:
            self._decision = {
                "candidate_id": None,
                "status": "rejected",
                "evaluated": records,
            }
            return self._decision
        chosen = max(accepted, key=lambda item: (item[1].weighted_gain, item[1].latest_gain))
        self._chosen_recipe = chosen[2]
        self._decision = {
            "candidate_id": chosen[0].candidate_id,
            "status": chosen[0].status,
            "recipe": {
                "method": chosen[2].method,
                "weights": dict(chosen[2].weights),
                "e2_weight": chosen[2].e2_weight,
            },
            "evidence": {
                **{key: value for key, value in vars(chosen[1]).items() if key != "fold_gains"},
                "fold_gains": dict(chosen[1].fold_gains),
            },
            "evaluated": records,
        }
        return self._decision

    def full_fit(self, decision: Mapping[str, object], output: Path) -> Mapping[str, Path]:
        if self._chosen_recipe is None:
            raise ValueError("accepted recipe is absent")
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        self.base.clear_fold_cache()
        state, batch = fit_direct_features(self.base.train.copy(deep=True), self.base.history, valid_year=2025)
        dependencies = required_experts(tuple(self._chosen_recipe.weights))
        iterations = {}
        for expert in dependencies:
            observed = [result.best_iteration + 1 for (name, _, _), result in self.results.items() if name == expert and result.best_iteration >= 0]
            iterations[expert] = full_fit_iterations(observed or [1800], maximum=2400)
        models = {}
        def train(item):
            expert, seed, gpu = item
            from catboost import CatBoostClassifier, CatBoostRegressor
            spec = expert_spec(expert)
            params = catboost_parameters(spec, load_contract(), seed=seed, gpu=gpu, phase="full_fit")
            params["iterations"] = iterations[expert]
            params.pop("save_snapshot", None); params.pop("snapshot_interval", None)
            params.pop("snapshot_file", None); params["allow_writing_files"] = False
            model = CatBoostClassifier(**params) if spec.objective == "Logloss" else CatBoostRegressor(**params)
            weights = season_weights(spec, batch.season, 2025); mask = training_mask(spec, batch) & (weights > 0)
            columns, categorical = feature_profile(batch.frame, state.categorical_columns, spec.interaction_profile)
            categories = [columns.index(name) for name in categorical]
            model.fit(batch.frame.loc[mask, columns], batch.target[mask], sample_weight=weights[mask], cat_features=categories)
            path = output / "models" / f"{expert}_seed_{seed}.cbm"; path.parent.mkdir(parents=True, exist_ok=True); model.save_model(str(path))
            return expert, seed, path
        tasks = [(expert, seed, index % 2) for index, (expert, seed) in enumerate((e, s) for e in dependencies for s in (42, 2026, 3407))]
        def train_gpu(gpu: int, assigned):
            return [train((expert, seed, gpu)) for expert, seed, _ in assigned]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(train_gpu, gpu, tasks[gpu::2]) for gpu in (0, 1)]
            for future in futures:
                for expert, seed, path in future.result():
                    models[(expert, seed)] = path
        state_path = output / "feature_state.pkl"; state_path.write_bytes(pickle.dumps(state, protocol=5))
        recipe_path = output / "recipe.json"; recipe_path.write_text(json.dumps(self._decision, sort_keys=True), encoding="utf-8")
        token = accepted_token(decision, self._chosen_recipe, asdict(self.bindings), seeds=(42, 2026, 3407), iterations=iterations)
        token_path = output / "token.json"; token_path.write_text(json.dumps(token, sort_keys=True), encoding="utf-8")
        audit_path = output / "audit_report.json"
        audit_path.write_text(json.dumps({"status": "pending"}, sort_keys=True), encoding="utf-8")
        requirements = output / "requirements.txt"
        requirements.write_text("catboost==1.2.10\npandas==2.0.3\nnumpy==1.26.4\n", encoding="utf-8")
        payloads = {
            "accepted/token.json": token_path,
            "state/feature_state.pkl": state_path,
            "recipe.json": recipe_path,
            "audit/report.json": audit_path,
            "requirements.txt": requirements,
        }
        payloads.update({f"models/{path.name}": path for path in models.values()})
        repository = Path(__file__).resolve().parents[2]
        payloads.update({f"runtime/{name}": repository / name for name in runtime_members(repository)})
        if self._chosen_recipe.e2_weight:
            payloads["e2/catboost_3seed_v1.zip"] = self.base.verified.e2_submission_path
        self._full_state, self._full_rows = state, self.base.train.drop(columns="control_success").head(128).copy()
        return payloads

    def audit(self, payloads: Mapping[str, Path]) -> bool:
        audit_source(Path(__file__).with_name("inference.py").read_text(encoding="utf-8"))
        if self._chosen_recipe is None or self._full_state is None or self._full_rows is None:
            return False
        from catboost import CatBoostClassifier, CatBoostRegressor
        models = {}
        for expert in required_experts(tuple(self._chosen_recipe.weights)):
            spec = expert_spec(expert)
            loaded = []
            for seed in (42, 2026, 3407):
                model = CatBoostClassifier() if spec.objective == "Logloss" else CatBoostRegressor()
                model.load_model(str(payloads[f"models/{expert}_seed_{seed}.cbm"])); loaded.append(model)
            models[expert] = loaded
        state = self._full_state; recipe = self._chosen_recipe
        e2_predictor = None
        if recipe.e2_weight:
            e2_root = self.root / "e2_audit"
            if e2_root.exists():
                shutil.rmtree(e2_root)
            e2_root.mkdir(parents=True)
            with ZipFile(payloads["e2/catboost_3seed_v1.zip"]) as archive:
                archive.extractall(e2_root)
            spec = importlib.util.spec_from_file_location("direct_expert_e2_runtime", e2_root / "script.py")
            if spec is None or spec.loader is None:
                return False
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            e2_predictor = module.load_frozen_predictor(e2_root / "model")
        class Runtime:
            def predict(_, rows):
                batch = transform_direct_features(rows, state)
                predictions = {}
                for expert, members in models.items():
                    spec = expert_spec(expert)
                    columns, _ = feature_profile(batch.frame, state.categorical_columns, spec.interaction_profile)
                    values = []
                    for model in members:
                        raw = model.predict_proba(batch.frame.loc[:, columns])[:, 1] if spec.objective == "Logloss" else model.predict(batch.frame.loc[:, columns])
                        values.append(np.clip(raw, 1e-6, 1 - 1e-6))
                    predictions[expert] = np.mean(values, axis=0)
                e2 = (
                    np.asarray(e2_predictor.predict_batch(rows, batch_size=4096), dtype="float64")
                    if e2_predictor is not None
                    else None
                )
                return predict_fixed_recipe(rows, predictions, recipe, e2=e2)
        report = audit_inference(Runtime(), self._full_rows)
        maximum = max(
            report.singleton_max_abs,
            report.reverse_max_abs,
            report.shuffle_max_abs,
            report.rebatch_max_abs,
            report.companion_max_abs,
            report.same_feature_audit_row_max_abs,
        )
        passed = maximum <= 1e-6
        Path(payloads["audit/report.json"]).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "passed" if passed else "failed",
                    "tolerance": 1e-6,
                    "maximum_abs_difference": maximum,
                    "checks": vars(report),
                    "runtime_code_sha256": code_identity_sha256(Path(__file__).resolve().parents[2]),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return passed
