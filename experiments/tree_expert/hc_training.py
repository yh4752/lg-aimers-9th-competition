from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .hc_contracts import HCContract, load_hc_contract
from .hc_contracts import HCProfile
from .hc_features import fit_hierarchy, transform_hierarchy
from .features import fit_tree_features, transform_tree_features


class HCTrainingError(ValueError):
    """Raised when C1 rolling residual evidence differs."""


@dataclass(frozen=True)
class C1JobResult:
    job_id: str
    status: str
    seed: int
    profile_name: str
    best_iteration: int
    brier: float
    model_path: Path
    predictions_path: Path


@dataclass(frozen=True)
class OOFFeatureMaterialization:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _validate_oof_rows(rows: object) -> pd.DataFrame:
    required = {
        "row_id",
        "oof_year",
        "target",
        "p0",
        "game_type",
        "pitcher_id",
        "batter_id",
    }
    if type(rows) is not pd.DataFrame or rows.columns.has_duplicates:
        raise HCTrainingError("OOF rows must be a DataFrame with unique columns")
    if not required.issubset(rows.columns) or rows.empty:
        raise HCTrainingError("OOF row schema differs")
    if rows["row_id"].isna().any() or not rows["row_id"].is_unique:
        raise HCTrainingError("OOF row identity differs")
    target = pd.to_numeric(rows["target"], errors="coerce")
    probability = pd.to_numeric(rows["p0"], errors="coerce")
    years = pd.to_numeric(rows["oof_year"], errors="coerce")
    if (
        not target.isin([0, 1]).all()
        or probability.isna().any()
        or not probability.between(0, 1).all()
        or years.isna().any()
        or not np.isfinite(probability.to_numpy(dtype="float64")).all()
    ):
        raise HCTrainingError("OOF row values differ")
    return rows.copy(deep=True)


def build_c1_training_rows(rows: pd.DataFrame, *, valid_year: int) -> pd.DataFrame:
    frame = _validate_oof_rows(rows)
    if type(valid_year) is not int or valid_year not in {2022, 2023, 2024, 2025}:
        raise HCTrainingError("C1 validation year differs")
    years = pd.to_numeric(frame["oof_year"], errors="raise").astype("int64")
    output = frame.loc[years.ge(2021) & years.lt(valid_year)].copy(deep=True)
    if output.empty or set(output["oof_year"].astype(int)) != set(range(2021, valid_year)):
        raise HCTrainingError("C1 rolling OOF window differs")
    output["residual_target"] = (
        pd.to_numeric(output["target"], errors="raise").astype("float64")
        - pd.to_numeric(output["p0"], errors="raise").astype("float64")
    )
    return output


def materialize_oof_feature_rows(
    official_rows: pd.DataFrame,
    *,
    history: pd.DataFrame,
    p0_by_year: Mapping[int, pd.DataFrame],
    profile_name: str,
    profile: HCProfile,
    minimum_group_rows: Mapping[str, int],
    tree_feature_builder: Callable[..., tuple[object, object]] = fit_tree_features,
    tree_feature_transformer: Callable[[pd.DataFrame, object], object] = transform_tree_features,
) -> OOFFeatureMaterialization:
    if type(official_rows) is not pd.DataFrame or "season" not in official_rows:
        raise HCTrainingError("official OOF rows differ")
    if set(p0_by_year) != {2021, 2022, 2023, 2024}:
        raise HCTrainingError("OOF baseline year set differs")
    rows = official_rows.copy(deep=True)
    seasons = pd.to_numeric(rows["season"], errors="coerce")
    if seasons.isna().any():
        raise HCTrainingError("official OOF seasons differ")
    outputs: list[pd.DataFrame] = []
    feature_columns: tuple[str, ...] | None = None
    categorical_columns: tuple[str, ...] | None = None
    for year in (2021, 2022, 2023, 2024):
        fit_rows = rows.loc[seasons.lt(year)].reset_index(drop=True)
        valid_labeled = rows.loc[seasons.eq(year)].reset_index(drop=True)
        if fit_rows.empty or valid_labeled.empty:
            raise HCTrainingError(f"OOF feature fold {year} is empty")
        state, _ = tree_feature_builder(
            fit_rows,
            history,
            valid_year=year,
            use_trackman=False,
            minimum_trackman_coverage=0.0,
        )
        valid_unlabeled = valid_labeled.drop(columns="control_success")
        batch = tree_feature_transformer(valid_unlabeled, state)
        if list(map(str, batch.row_id)) != valid_labeled["row_id"].astype(str).tolist():
            raise HCTrainingError(f"OOF tree row alignment differs: {year}")
        tree_frame = batch.frame.reset_index(drop=True).add_prefix("tree__")
        tree_features = tuple(tree_frame.columns)
        tree_categorical = tuple(
            f"tree__{name}" for name in state.categorical_columns
        )
        hierarchy_state = fit_hierarchy(
            fit_rows,
            cutoff_year=year - 1,
            profile_name=profile_name,
            profile=profile,
            minimum_group_rows=minimum_group_rows,
        )
        hierarchy = transform_hierarchy(valid_unlabeled, hierarchy_state).reset_index(drop=True)
        baseline = p0_by_year[year].reset_index(drop=True)
        required_baseline = {"row_id", "target", "probability"}
        if (
            not required_baseline.issubset(baseline.columns)
            or baseline["row_id"].astype(str).tolist()
            != valid_labeled["row_id"].astype(str).tolist()
            or not np.array_equal(
                pd.to_numeric(baseline["target"], errors="coerce").to_numpy(),
                pd.to_numeric(valid_labeled["control_success"], errors="coerce").to_numpy(),
            )
        ):
            raise HCTrainingError(f"OOF baseline row alignment differs: {year}")
        probability = pd.to_numeric(baseline["probability"], errors="coerce")
        if probability.isna().any() or not probability.between(0, 1).all():
            raise HCTrainingError(f"OOF baseline probability differs: {year}")
        diagnostics = valid_labeled.drop(columns="control_success").reset_index(drop=True)
        materialized = pd.concat([diagnostics, tree_frame, hierarchy], axis=1)
        materialized["target"] = valid_labeled["control_success"].to_numpy(dtype="float64")
        materialized["p0"] = probability.to_numpy(dtype="float64")
        materialized["oof_year"] = year
        current_features = (*tree_features, *tuple(hierarchy.columns), "p0")
        current_categorical = tree_categorical
        if feature_columns is None:
            feature_columns = current_features
            categorical_columns = current_categorical
        elif feature_columns != current_features or categorical_columns != current_categorical:
            raise HCTrainingError("OOF feature schema changes between years")
        outputs.append(materialized)
    combined = pd.concat(outputs, ignore_index=True)
    if combined["row_id"].isna().any() or not combined["row_id"].is_unique:
        raise HCTrainingError("OOF materialized row identity differs")
    return OOFFeatureMaterialization(
        frame=combined,
        feature_columns=feature_columns or (),
        categorical_columns=categorical_columns or (),
    )


def _default_model_factory(parameters: dict[str, object]) -> object:
    try:
        from catboost import CatBoostRegressor
    except ImportError as error:
        raise HCTrainingError("catboost==1.2.10 is required") from error
    return CatBoostRegressor(**parameters)


def _parameters(
    contract: HCContract, seed: int, gpu_id: int, output_dir: Path
) -> dict[str, object]:
    if seed not in contract.seeds:
        raise HCTrainingError("C1 seed differs")
    parameters = dict(contract.residual_catboost)
    parameters.update(
        random_seed=seed,
        devices=str(gpu_id),
        train_dir=str(output_dir / "catboost_info"),
        eval_metric="RMSE",
    )
    return parameters


def _save_model(model: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        model.save_model(str(temporary))
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise HCTrainingError("C1 model output is empty")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_c1_job(
    *,
    job_id: str,
    seed: int,
    profile_name: str,
    train_rows: pd.DataFrame,
    valid_rows: pd.DataFrame,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
    output_dir: Path,
    absolute_deadline: float,
    gpu_id: int,
    contract: HCContract | None = None,
    model_factory: Callable[[dict[str, object]], object] | None = None,
) -> C1JobResult:
    if time.time() >= absolute_deadline:
        raise TimeoutError("C1 job deadline reached")
    active = load_hc_contract() if contract is None else contract
    train = _validate_oof_rows(train_rows)
    valid = _validate_oof_rows(valid_rows)
    features = tuple(feature_columns)
    categorical = tuple(categorical_columns)
    if (
        not features
        or len(features) != len(set(features))
        or not set(features).issubset(train.columns)
        or not set(features).issubset(valid.columns)
        or not set(categorical).issubset(features)
    ):
        raise HCTrainingError("C1 feature schema differs")
    if "residual_target" not in train:
        train = train.copy(deep=True)
        train["residual_target"] = train["target"] - train["p0"]
    residual_target = pd.to_numeric(train["residual_target"], errors="coerce")
    valid_residual = pd.to_numeric(valid["target"] - valid["p0"], errors="coerce")
    if residual_target.isna().any() or valid_residual.isna().any():
        raise HCTrainingError("C1 residual target differs")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    parameters = _parameters(active, seed, gpu_id, output)
    factory = _default_model_factory if model_factory is None else model_factory
    model = factory(parameters)
    model.fit(
        train.loc[:, features],
        residual_target.to_numpy(dtype="float64"),
        eval_set=(valid.loc[:, features], valid_residual.to_numpy(dtype="float64")),
        cat_features=list(categorical),
        use_best_model=True,
        early_stopping_rounds=int(active.residual_catboost["od_wait"]),
        verbose=50,
    )
    if time.time() >= absolute_deadline:
        raise TimeoutError("C1 job deadline reached")
    residual = np.asarray(model.predict(valid.loc[:, features]), dtype="float64")
    if residual.shape != (len(valid),) or not np.isfinite(residual).all():
        raise HCTrainingError("C1 residual prediction differs")
    p1 = np.clip(valid["p0"].to_numpy(dtype="float64") + residual, 1e-5, 1 - 1e-5)
    brier = float(np.mean(np.square(p1 - valid["target"].to_numpy(dtype="float64"))))
    model_path = output / "model.cbm"
    _save_model(model, model_path)
    diagnostics = (
        "row_id",
        "target",
        "oof_year",
        "game_type",
        "pitcher_id",
        "batter_id",
        "pitcher_hand",
        "batter_hand",
        "balls_before",
        "strikes_before",
        "outs_before",
        "base_state",
        "pitcher_id_known",
        "batter_id_known",
        "p0",
    )
    prediction = valid.loc[:, [name for name in diagnostics if name in valid]].copy(deep=True)
    prediction["p1"] = p1
    predictions_path = output / "predictions.csv"
    _atomic_bytes(predictions_path, prediction.to_csv(index=False).encode())
    best_iteration = int(model.get_best_iteration())
    _atomic_bytes(
        output / "feature_schema.json",
        _canonical(
            {
                "feature_columns": list(features),
                "categorical_columns": list(categorical),
            }
        ),
    )
    _atomic_bytes(
        output / "job.json",
        _canonical(
            {
                "job_id": job_id,
                "seed": seed,
                "profile_name": profile_name,
                "gpu_id": gpu_id,
                "parameters": parameters,
            }
        ),
    )
    _atomic_bytes(
        output / "metrics.json",
        _canonical({"status": "completed", "brier": brier, "best_iteration": best_iteration}),
    )
    return C1JobResult(
        job_id=job_id,
        status="completed",
        seed=seed,
        profile_name=profile_name,
        best_iteration=best_iteration,
        brier=brier,
        model_path=model_path,
        predictions_path=predictions_path,
    )


def ensemble_c1_predictions(paths: Mapping[int, Path]) -> pd.DataFrame:
    if set(paths) != {3407, 42, 2026}:
        raise HCTrainingError("C1 seed set differs")
    frames: dict[int, pd.DataFrame] = {}
    for seed in (3407, 42, 2026):
        try:
            frame = pd.read_csv(paths[seed])
        except Exception as error:
            raise HCTrainingError(f"cannot read C1 seed {seed} predictions") from error
        if "p1" not in frame or frame.empty or frame["row_id"].isna().any():
            raise HCTrainingError(f"C1 seed {seed} prediction schema differs")
        frames[seed] = frame
    reference = frames[3407]
    for seed, frame in frames.items():
        if (
            frame["row_id"].astype(str).tolist()
            != reference["row_id"].astype(str).tolist()
            or not np.array_equal(frame["target"].to_numpy(), reference["target"].to_numpy())
            or not np.allclose(frame["p0"], reference["p0"], rtol=0, atol=0)
        ):
            raise HCTrainingError(f"C1 seed {seed} row alignment differs")
    output = reference.copy(deep=True)
    output["p1"] = np.mean(
        np.column_stack([frames[seed]["p1"].to_numpy(dtype="float64") for seed in (3407, 42, 2026)]),
        axis=1,
    )
    if not np.isfinite(output["p1"]).all() or not output["p1"].between(0, 1).all():
        raise HCTrainingError("C1 ensemble probabilities differ")
    return output
