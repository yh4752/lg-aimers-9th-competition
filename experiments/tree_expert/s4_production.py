from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import shutil
import threading
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from .features import fit_tree_features, transform_tree_features
from .s4_calibration import apply_s4_calibrator, fit_s4_calibrator
from .s4_contracts import AnchorSpec, anchor_specs, load_s4_contract
from .s4_decisions import (
    AnchorEvidence,
    FullChainArchetype,
    decide_submission_eligibility,
    decision_payload,
    evaluate_full_chain,
    full_chain_archetypes,
    select_anchor_coverage,
)
from .s4_inputs import VerifiedS4Input
from .s4_runner import S4Job
from .s4_state import S4State, record_s4_decision
from .s4_temporal import blend_anchor, season_decay_weights
from .s4_training import fit_residual_estimator
from .t3_inputs import (
    VerifiedOfficialData,
    VerifiedT3Input,
    prepare_t3_input,
    verify_and_extract_t3_input,
    verify_official_data,
)


class S4ProductionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PruneReport:
    removed_files: int
    removed_bytes: int


def _path_usage(path: Path) -> tuple[int, int]:
    source = Path(path)
    if source.is_symlink():
        raise S4ProductionError("prune target is a symlink")
    if source.is_file():
        return 1, source.stat().st_size
    if not source.exists():
        return 0, 0
    files = 0
    size = 0
    for item in source.rglob("*"):
        if item.is_symlink():
            raise S4ProductionError("prune source contains a symlink")
        if item.is_file():
            files += 1
            size += item.stat().st_size
    return files, size


def _remove_campaign_path(path: Path) -> tuple[int, int]:
    files, size = _path_usage(path)
    if not Path(path).exists():
        return files, size
    if Path(path).is_file():
        Path(path).unlink()
    else:
        shutil.rmtree(path)
    return files, size


def prune_after_full_chain_selection(
    root: Path,
    selected: tuple[int, ...],
    archetypes: tuple[FullChainArchetype, ...],
) -> PruneReport:
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(index < 0 or index >= len(archetypes) for index in selected)
    ):
        raise S4ProductionError("prune selection differs")
    source = Path(root)
    if source.is_symlink() or not source.is_dir():
        raise S4ProductionError("prune campaign root differs")
    selected_indices = set(selected)
    selected_anchors = {
        _candidate_token(archetypes[index].anchor_id) for index in selected
    }
    targets = [
        source / "anchor_basis",
        source / "verified_e2",
        source / "verified_e2_input.zip",
        source / "jobs",
    ]
    anchors = source / "anchors"
    if anchors.is_dir():
        targets.extend(
            path for path in anchors.iterdir() if path.name not in selected_anchors
        )
    residuals = source / "residual_predictions"
    if residuals.is_dir():
        targets.extend(
            path for path in residuals.iterdir()
            if path.name[1:].isdigit() and path.name.startswith("c")
            and int(path.name[1:]) not in selected_indices
        )
    chains = source / "full_chains"
    if chains.is_dir():
        targets.extend(
            path for path in chains.iterdir()
            if path.name[1:].isdigit() and path.name.startswith("c")
            and int(path.name[1:]) not in selected_indices
        )
    removed_files = 0
    removed_bytes = 0
    for target in targets:
        files, size = _remove_campaign_path(target)
        removed_files += files
        removed_bytes += size
    return PruneReport(removed_files, removed_bytes)


_SOURCE_FOLD = (2020, 2021)


@dataclass(frozen=True)
class FoldCache:
    fold: tuple[int, int]
    prefix: pd.DataFrame
    valid: pd.DataFrame
    train_frame: pd.DataFrame
    valid_frame: pd.DataFrame
    train_matrix: np.ndarray
    valid_matrix: np.ndarray
    target: np.ndarray
    valid_target: np.ndarray
    train_anchor: np.ndarray
    valid_anchor: np.ndarray


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n", float_format="%.17g")
    os.replace(temporary, path)


def _encode(train: pd.DataFrame, valid: pd.DataFrame, categories: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    left, right = [], []
    category_set = set(categories)
    for column in train.columns:
        if column in category_set:
            values = train[column].astype("string").fillna("__MISSING__").astype(str)
            mapping = {value: index for index, value in enumerate(sorted(values.unique()))}
            left.append(values.map(mapping).to_numpy(dtype="float32"))
            right.append(valid[column].astype("string").fillna("__MISSING__").astype(str).map(mapping).fillna(-1).to_numpy(dtype="float32"))
        else:
            left.append(pd.to_numeric(train[column], errors="coerce").to_numpy(dtype="float32"))
            right.append(pd.to_numeric(valid[column], errors="coerce").to_numpy(dtype="float32"))
    return np.column_stack(left), np.column_stack(right)


def _positive(value: object, rows: int) -> np.ndarray:
    raw = np.asarray(value, dtype="float64")
    if raw.shape == (rows, 2):
        raw = raw[:, 1]
    if raw.shape != (rows,) or not np.isfinite(raw).all():
        raise S4ProductionError("probability prediction differs")
    return np.clip(raw, 1e-5, 1 - 1e-5)


def _save_model(model: object, family: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    if family in {"catboost", "catboost_rf", "dual_temporal", "anchor"}:
        model.save_model(str(temporary))
    elif family == "xgboost":
        model.save_model(str(temporary))
    elif family == "lightgbm":
        getattr(model, "booster_", model).save_model(str(temporary))
    else:
        raise S4ProductionError("model family differs")
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise S4ProductionError("model output is empty")
    os.replace(temporary, path)


def _candidate_token(candidate_id: str) -> str:
    return candidate_id.replace("s4__anchor__", "")


class ProductionS4Runtime:
    def __init__(
        self,
        verified: VerifiedS4Input,
        official: VerifiedOfficialData,
        work_root: Path,
    ) -> None:
        self.contract = load_s4_contract()
        self.verified = verified
        self.official = official
        self.root = Path(work_root)
        self.train = pd.read_csv(official.train)
        seasons = pd.to_numeric(self.train["season"], errors="raise")
        if not seasons.isin(range(2019, 2025)).all():
            raise S4ProductionError("official training seasons differ")
        self.e2: VerifiedT3Input | None = None
        self.baselines: dict[tuple[int, int], pd.DataFrame] = {}
        self._cache: dict[tuple[int, int], FoldCache] = {}
        self._cache_lock = threading.Lock()

    def _ensure_e2(self) -> VerifiedT3Input:
        if self.e2 is not None:
            return self.e2
        archive = self.root / "verified_e2_input.zip"
        extracted = self.root / "verified_e2"
        if not archive.exists():
            prepare_t3_input(
                e2_handoff=self.verified.e2_handoff,
                output=archive,
                expected_e2_sha256=self.verified.e2_handoff_sha256,
            )
        if extracted.exists():
            shutil.rmtree(extracted)
        self.e2 = verify_and_extract_t3_input(
            archive,
            extracted,
            expected_e2_sha256=self.verified.e2_handoff_sha256,
        )
        self.baselines = {fold: pd.read_csv(path) for fold, path in self.e2.fold_predictions.items()}
        return self.e2

    def _fold_cache(self, fold: tuple[int, int]) -> FoldCache:
        with self._cache_lock:
            cached = self._cache.get(fold)
            if cached is not None:
                return cached
            train_end, valid_year = fold
            seasons = pd.to_numeric(self.train["season"], errors="raise")
            prefix = self.train.loc[seasons.le(train_end)].reset_index(drop=True)
            valid = self.train.loc[seasons.eq(valid_year)].reset_index(drop=True)
            if prefix.empty or valid.empty:
                raise S4ProductionError(f"fold rows are empty: {fold}")
            state, train_batch = fit_tree_features(prefix, None, valid_year=valid_year, use_trackman=False)
            valid_batch = transform_tree_features(valid.drop(columns="control_success"), state)
            train_matrix, valid_matrix = _encode(train_batch.frame, valid_batch.frame, state.categorical_columns)
            cached = FoldCache(
                fold,
                prefix.loc[:, ["season", "game_type"]].copy(deep=True),
                valid.copy(deep=True),
                train_batch.frame, valid_batch.frame,
                train_matrix, valid_matrix,
                np.asarray(train_batch.target, dtype="float64"),
                valid["control_success"].to_numpy(dtype="float64"),
                np.asarray(train_batch.anchor, dtype="float64"),
                np.asarray(valid_batch.anchor, dtype="float64"),
            )
            self._cache[fold] = cached
            return cached

    def jobs_for_phase(self, phase: str, state: S4State, root: Path) -> tuple[S4Job, ...]:
        if phase == "anchors":
            jobs = []
            for fold in (_SOURCE_FOLD, *self.contract.folds):
                jobs.append(S4Job(f"anchors__basis__{fold[0]}_{fold[1]}__recent", phase, {"fold": fold, "head": "recent", "decay": None}))
                for decay in self.contract.decays:
                    jobs.append(S4Job(f"anchors__basis__{fold[0]}_{fold[1]}__d{int(decay*100):02d}", phase, {"fold": fold, "head": "multi", "decay": decay}))
            return tuple(jobs)
        if phase == "residuals":
            return tuple(
                S4Job(
                    f"residuals__{index:02d}__{fold[0]}_{fold[1]}__s{self.contract.structure_seed}",
                    phase,
                    {"index": index, "fold": fold, "seed": self.contract.structure_seed},
                )
                for index, _ in enumerate(self._archetypes())
                for fold in self.contract.folds
            )
        if phase == "full_chains":
            return tuple(
                S4Job(f"full_chains__{index:02d}", phase, {"index": index})
                for index, _ in enumerate(self._archetypes())
            )
        if phase == "confirmation":
            selection = self._selection()
            return tuple(
                S4Job(
                    f"confirmation__{index:02d}__{fold[0]}_{fold[1]}__s{seed}",
                    phase,
                    {"index": index, "fold": fold, "seed": seed},
                )
                for index in selection
                for seed in self.contract.confirmation_seeds
                for fold in self.contract.folds
            )
        if phase == "full_fit":
            return ()
        raise S4ProductionError("campaign phase differs")

    def _basis_path(self, fold: tuple[int, int], head: str, decay: float | None) -> Path:
        token = "recent" if head == "recent" else f"d{int(float(decay)*100):02d}"
        return self.root / "anchor_basis" / f"{fold[0]}_{fold[1]}" / token / "predictions.csv"

    def _run_anchor_basis(self, job: S4Job, job_dir: Path, gpu_id: int) -> str:
        fold = tuple(job.payload["fold"])
        head = str(job.payload["head"])
        decay = job.payload["decay"]
        cache = self._fold_cache(fold)
        age = fold[0] - pd.to_numeric(cache.prefix["season"], errors="raise").to_numpy(dtype="int64")
        weight = (age == 0).astype("float64") if head == "recent" else np.power(float(decay), age)
        selected = weight > 0
        from catboost import CatBoostRegressor

        parameters = dict(self.contract.parameters["catboost"])
        parameters.update(
            loss_function="RMSE", eval_metric="RMSE", random_seed=self.contract.structure_seed,
            task_type="GPU", devices=str(gpu_id), allow_writing_files=False, verbose=100,
        )
        model = CatBoostRegressor(**parameters)
        categories = [cache.train_frame.columns.get_loc(name) for name in cache.train_frame.select_dtypes(include=["object", "string"]).columns]
        model.fit(
            cache.train_frame.loc[selected],
            (cache.target - cache.train_anchor)[selected],
            sample_weight=weight[selected],
            cat_features=categories,
            eval_set=(cache.valid_frame, cache.valid_target - cache.valid_anchor),
            use_best_model=True,
            early_stopping_rounds=100,
        )
        probability = np.clip(cache.valid_anchor + np.asarray(model.predict(cache.valid_frame), dtype="float64"), 1e-5, 1 - 1e-5)
        output = self._diagnostic_frame(cache.valid, probability, "p_basis")
        path = self._basis_path(fold, head, None if decay is None else float(decay))
        _atomic_frame(path, output)
        _atomic_json(job_dir / "result.json", {"status": "completed", "best_iteration": max(0, int(model.get_best_iteration())), "predictions": str(path.relative_to(self.root))})
        return "completed"

    @staticmethod
    def _diagnostic_frame(rows: pd.DataFrame, probability: np.ndarray, name: str) -> pd.DataFrame:
        columns = (
            "row_id", "game_type", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
            "balls_before", "strikes_before", "outs_before", "base_state",
        )
        output = rows.loc[:, list(columns)].copy(deep=True)
        output["target"] = rows["control_success"].to_numpy(dtype="int64")
        output[name] = probability
        return output

    def _candidate_path(self, candidate_id: str, year: int) -> Path:
        return self.root / "anchors" / _candidate_token(candidate_id) / f"{year}.csv"

    def _anchor_frame(self, spec: AnchorSpec, fold: tuple[int, int]) -> pd.DataFrame:
        self._ensure_e2()
        if spec.recent_weight is None:
            if fold in self.baselines:
                baseline = self.baselines[fold]
                cache = self._fold_cache(fold)
                if baseline["row_id"].astype(str).tolist() != cache.valid["row_id"].astype(str).tolist():
                    raise S4ProductionError("E2 baseline row alignment differs")
                output = self._diagnostic_frame(cache.valid, baseline["probability"].to_numpy(dtype="float64"), "p_anchor")
                output["oof_year"] = fold[1]
                return output
            spec = next(item for item in anchor_specs(self.contract) if item.mandatory_role == "external_template")
        recent = pd.read_csv(self._basis_path(fold, "recent", None))
        multi = pd.read_csv(self._basis_path(fold, "multi", spec.decay))
        if recent["row_id"].astype(str).tolist() != multi["row_id"].astype(str).tolist():
            raise S4ProductionError("anchor basis row alignment differs")
        output = recent.drop(columns="p_basis")
        output["p_anchor"] = blend_anchor(recent["p_basis"], multi["p_basis"], recent_weight=float(spec.recent_weight))
        output["oof_year"] = fold[1]
        return output

    def _finish_anchors(self, state: S4State) -> S4State:
        self._ensure_e2()
        specs = anchor_specs(self.contract)
        evidence = []
        weights = np.asarray((0.60, 0.75), dtype="float64")
        for spec in specs:
            fold_gains = {}
            signatures = []
            rf_gain = []
            for fold in (_SOURCE_FOLD, *self.contract.folds):
                frame = self._anchor_frame(spec, fold)
                _atomic_frame(self._candidate_path(spec.candidate_id, fold[1]), frame)
                if fold not in self.contract.folds[:2]:
                    continue
                base = self.baselines[fold]["probability"].to_numpy(dtype="float64")
                target = frame["target"].to_numpy(dtype="float64")
                candidate = frame["p_anchor"].to_numpy(dtype="float64")
                gain = np.square(base - target) - np.square(candidate - target)
                fold_gains[fold] = float(gain.mean())
                signatures.extend((target - candidate).tolist())
                mask = frame["game_type"].astype(str).eq("F").to_numpy()
                rf_gain.append(float(gain[mask].mean()) if mask.any() else float(gain.mean()))
            evidence.append(AnchorEvidence(
                spec.candidate_id, spec.mandatory_role,
                float(np.average(list(fold_gains.values()), weights=weights)),
                MappingProxyType(fold_gains), float(np.mean(rf_gain)), tuple(signatures),
            ))
        coverage = select_anchor_coverage(tuple(evidence))
        _atomic_json(self.root / "decisions/anchor_coverage.json", {
            "coverage": [{"role": item.role, "candidate_id": item.candidate_id} for item in coverage],
        })
        with self._cache_lock:
            self._cache.clear()
        gc.collect()
        return state

    def _coverage(self):
        payload = json.loads((self.root / "decisions/anchor_coverage.json").read_text())
        from .s4_decisions import SelectedAnchor
        return tuple(SelectedAnchor(str(item["role"]), str(item["candidate_id"])) for item in payload["coverage"])

    def _archetypes(self) -> tuple[FullChainArchetype, ...]:
        return full_chain_archetypes(self.contract, self._coverage())

    def _residual_path(self, index: int, fold: tuple[int, int], seed: int) -> Path:
        return self.root / "residual_predictions" / f"c{index:02d}" / f"s{seed}" / f"{fold[1]}.csv"

    def _fit_one_residual(
        self, family: str, cache: FoldCache, gpu_id: int, seed: int,
        sample_weight: np.ndarray, train_mask: np.ndarray | None = None,
    ):
        mask = np.ones(len(cache.target), dtype=bool) if train_mask is None else train_mask
        return fit_residual_estimator(
            family=family,
            train_matrix=cache.train_matrix[mask],
            train_target=(cache.target - cache.train_anchor)[mask],
            valid_matrix=cache.valid_matrix,
            gpu_id=gpu_id,
            seed=seed,
            sample_weight=sample_weight[mask],
            contract=self.contract,
        )

    def _run_residual(self, job: S4Job, job_dir: Path, gpu_id: int) -> str:
        self._ensure_e2()
        index = int(job.payload["index"])
        fold = tuple(job.payload["fold"])
        seed = int(job.payload["seed"])
        archetype = self._archetypes()[index]
        cache = self._fold_cache(fold)
        anchor = pd.read_csv(self._candidate_path(archetype.anchor_id, fold[1]))
        if anchor["row_id"].astype(str).tolist() != cache.valid["row_id"].astype(str).tolist():
            raise S4ProductionError("residual anchor alignment differs")
        seasons = pd.to_numeric(cache.prefix["season"], errors="raise").to_numpy(dtype="int64")
        decay = 0.55
        for spec in anchor_specs(self.contract):
            if spec.candidate_id == archetype.anchor_id and spec.decay is not None:
                decay = spec.decay
        multi_weight = season_decay_weights(seasons, cutoff_year=fold[0], decay=decay)
        family = archetype.residual_family
        if family == "catboost_rf":
            correction = np.zeros(len(cache.valid), dtype="float64")
            for game_type in ("R", "F"):
                mask = cache.prefix["game_type"].astype(str).eq(game_type).to_numpy()
                result = self._fit_one_residual(family, cache, gpu_id, seed, multi_weight, mask)
                valid_mask = cache.valid["game_type"].astype(str).eq(game_type).to_numpy()
                correction[valid_mask] = result.prediction[valid_mask]
        elif family == "dual_temporal":
            recent_mask = pd.to_numeric(cache.prefix["season"], errors="raise").eq(fold[0]).to_numpy()
            recent = self._fit_one_residual(family, cache, gpu_id, seed, np.ones(len(cache.target)), recent_mask)
            multi = self._fit_one_residual(family, cache, gpu_id, seed + 10000, multi_weight)
            correction = 0.5 * (recent.prediction + multi.prediction)
        else:
            result = self._fit_one_residual(family, cache, gpu_id, seed, multi_weight)
            correction = result.prediction
        probability = np.clip(anchor["p_anchor"].to_numpy(dtype="float64") + archetype.residual_alpha * correction, 1e-5, 1 - 1e-5)
        output = anchor.rename(columns={"p_anchor": "p_anchor"}).copy(deep=True)
        output["oof_year"] = fold[1]
        output["raw_correction"] = correction
        output["p_chain"] = probability
        baseline = self.baselines[fold]
        output["p_base"] = baseline["probability"].to_numpy(dtype="float64")
        path = self._residual_path(index, fold, seed)
        _atomic_frame(path, output)
        _atomic_json(job_dir / "result.json", {"status": "completed", "candidate": archetype.candidate_id, "seed": seed, "predictions": str(path.relative_to(self.root))})
        return "completed"

    def _chain_frames(self, index: int, seed: int) -> dict[tuple[int, int], pd.DataFrame]:
        return {fold: pd.read_csv(self._residual_path(index, fold, seed)) for fold in self.contract.folds}

    def _calibrate(
        self,
        index: int,
        frames: Mapping[tuple[int, int], pd.DataFrame],
        *,
        alpha: float,
        beta: float,
        target_folds: tuple[tuple[int, int], ...] | None = None,
    ) -> dict[tuple[int, int], pd.DataFrame]:
        archetype = self._archetypes()[index]
        source = pd.read_csv(self._candidate_path(archetype.anchor_id, 2021))
        adjusted = {}
        for fold, frame in frames.items():
            if "raw_correction" not in frame:
                raise S4ProductionError("raw residual correction is absent")
            current = frame.copy(deep=True)
            current["p_chain"] = np.clip(
                current["p_anchor"].to_numpy(dtype="float64")
                + float(alpha) * current["raw_correction"].to_numpy(dtype="float64"),
                1e-5,
                1 - 1e-5,
            )
            adjusted[fold] = current
        selected_folds = self.contract.folds if target_folds is None else target_folds
        output = {}
        for fold in selected_folds:
            if fold[1] == 2022:
                calibration_rows = adjusted[fold]
            else:
                calibration_rows = pd.concat(
                    [adjusted[item] for item in self.contract.folds if item[1] < fold[1]],
                    ignore_index=True,
                )
            state = fit_s4_calibrator(
                fold[1], source, calibration_rows,
                profile_name=archetype.calibration_profile,
            )
            calibrated = apply_s4_calibrator(adjusted[fold], state, beta=beta)
            calibrated["p_candidate"] = calibrated["p_final"]
            output[fold] = calibrated
        return output

    @staticmethod
    def _structure_gain(frames: Mapping[tuple[int, int], pd.DataFrame]) -> tuple[float, float]:
        gains = []
        for fold in ((2021, 2022), (2022, 2023)):
            frame = frames[fold]
            target = frame["target"].to_numpy(dtype="float64")
            gains.append(float(np.mean(
                np.square(frame["p_base"].to_numpy(dtype="float64") - target)
                - np.square(frame["p_candidate"].to_numpy(dtype="float64") - target)
            )))
        return float(np.average(gains, weights=(0.60, 0.75))), min(gains)

    def _run_full_chain(self, job: S4Job, job_dir: Path) -> str:
        index = int(job.payload["index"])
        raw = self._chain_frames(index, self.contract.structure_seed)
        trials = []
        for alpha in self.contract.residual_alphas:
            for beta in self.contract.calibration_betas:
                structure = self._calibrate(
                    index,
                    {fold: raw[fold] for fold in self.contract.folds[:2]},
                    alpha=alpha,
                    beta=beta,
                    target_folds=self.contract.folds[:2],
                )
                weighted, worst = self._structure_gain(structure)
                trials.append((weighted, worst, -alpha, -beta, alpha, beta))
        _, _, _, _, alpha, beta = max(trials)
        config = {
            "candidate_id": self._archetypes()[index].candidate_id,
            "residual_alpha": alpha,
            "calibration_beta": beta,
            "selection_folds": ["2021->2022", "2022->2023"],
        }
        _atomic_json(self.root / "full_chains" / f"c{index:02d}" / "config.json", config)
        frames = self._calibrate(index, raw, alpha=alpha, beta=beta)
        for fold, frame in frames.items():
            _atomic_frame(self.root / "full_chains" / f"c{index:02d}" / f"{fold[1]}.csv", frame)
        evidence = evaluate_full_chain(self._archetypes()[index].candidate_id, frames, confirmed=False, non_worse_seed_count=0)
        decision = decide_submission_eligibility(evidence)
        _atomic_json(job_dir / "result.json", {"status": "completed", "config": config, "decision": decision_payload(decision)})
        _atomic_json(self.root / "decisions" / f"structure_c{index:02d}.json", decision_payload(decision))
        return "completed"

    def _finish_full_chains(self, state: S4State) -> S4State:
        scores = []
        external = []
        for index, archetype in enumerate(self._archetypes()):
            folds = {
                fold: pd.read_csv(self.root / "full_chains" / f"c{index:02d}" / f"{fold[1]}.csv")
                for fold in self.contract.folds
            }
            structure_gains = []
            for fold in self.contract.folds[:2]:
                frame = folds[fold]
                target = frame["target"].to_numpy(dtype="float64")
                structure_gains.append(float(np.mean(np.square(frame["p_base"] - target) - np.square(frame["p_candidate"] - target))))
            score = float(np.average(structure_gains, weights=(0.60, 0.75)))
            scores.append((score, min(structure_gains), index))
            if archetype.anchor_role == "external_template":
                external.append((score, index))
            state = record_s4_decision(state, archetype.candidate_id, "research_only")
        selected = [index for _, _, index in sorted(scores, reverse=True)[: self.contract.maximum_confirmation_candidates]]
        if external:
            external_index = max(external)[1]
            if external_index not in selected:
                selected[-1] = external_index
        selected = sorted(set(selected))
        _atomic_json(self.root / "decisions/confirmation_selection.json", {"indices": selected, "selection_folds": ["2021->2022", "2022->2023"]})
        report = prune_after_full_chain_selection(
            self.root, tuple(selected), self._archetypes()
        )
        print(
            f"S4_PRUNE_COMPLETE removed_files={report.removed_files} "
            f"removed_bytes={report.removed_bytes}",
            flush=True,
        )
        return state

    def _selection(self) -> tuple[int, ...]:
        payload = json.loads((self.root / "decisions/confirmation_selection.json").read_text())
        return tuple(int(value) for value in payload["indices"])

    def _finish_confirmation(self, state: S4State) -> S4State:
        for index in self._selection():
            config = json.loads((self.root / "full_chains" / f"c{index:02d}" / "config.json").read_text())
            alpha = float(config["residual_alpha"])
            beta = float(config["calibration_beta"])
            seeds = (self.contract.structure_seed, *self.contract.confirmation_seeds)
            seed_frames = {seed: self._chain_frames(index, seed) for seed in seeds}
            averaged = {}
            non_worse = 0
            for seed, frames in seed_frames.items():
                gains = []
                for fold in self.contract.folds:
                    frame = frames[fold]
                    target = frame["target"].to_numpy(dtype="float64")
                    probability = np.clip(
                        frame["p_anchor"].to_numpy(dtype="float64")
                        + alpha * frame["raw_correction"].to_numpy(dtype="float64"),
                        1e-5,
                        1 - 1e-5,
                    )
                    gains.append(float(np.mean(np.square(frame["p_base"] - target) - np.square(probability - target))))
                if min(gains) >= -self.contract.gates.maximum_fold_regression:
                    non_worse += 1
            for fold in self.contract.folds:
                reference = seed_frames[seeds[0]][fold].copy(deep=True)
                for seed in seeds[1:]:
                    if reference["row_id"].astype(str).tolist() != seed_frames[seed][fold]["row_id"].astype(str).tolist():
                        raise S4ProductionError("confirmation seed row alignment differs")
                reference["raw_correction"] = np.mean(
                    [seed_frames[seed][fold]["raw_correction"].to_numpy(dtype="float64") for seed in seeds],
                    axis=0,
                )
                averaged[fold] = reference
            calibrated = self._calibrate(index, averaged, alpha=alpha, beta=beta)
            evidence = evaluate_full_chain(self._archetypes()[index].candidate_id, calibrated, confirmed=True, non_worse_seed_count=non_worse)
            decision = decide_submission_eligibility(evidence)
            _atomic_json(self.root / "decisions" / f"acceptance_c{index:02d}.json", decision_payload(decision))
            for fold, frame in calibrated.items():
                _atomic_frame(self.root / "confirmation" / f"c{index:02d}" / f"{fold[1]}.csv", frame)
            state = record_s4_decision(state, decision.candidate_id, decision.status)
            print(f"S4_DECISION candidate={decision.candidate_id} status={decision.status} weighted_gain={decision.weighted_gain:.12f} recent_gain={decision.recent_gain:.12f}", flush=True)
        return state

    def run_job(self, job: S4Job, job_dir: Path, gpu_id: int, deadline: float) -> str:
        if job.phase == "anchors":
            return self._run_anchor_basis(job, job_dir, gpu_id)
        if job.phase in {"residuals", "confirmation"}:
            return self._run_residual(job, job_dir, gpu_id)
        if job.phase == "full_chains":
            return self._run_full_chain(job, job_dir)
        raise S4ProductionError("unsupported production job")

    def finalize_phase(self, phase: str, state: S4State, root: Path) -> S4State:
        if phase == "anchors":
            return self._finish_anchors(state)
        if phase == "full_chains":
            return self._finish_full_chains(state)
        if phase == "confirmation":
            return self._finish_confirmation(state)
        return state


def prepare_production_runtime(
    verified: VerifiedS4Input,
    official_root: Path,
    work_root: Path,
) -> tuple[ProductionS4Runtime, VerifiedOfficialData]:
    official = verify_official_data(Path(official_root))
    return ProductionS4Runtime(verified, official, Path(work_root)), official
