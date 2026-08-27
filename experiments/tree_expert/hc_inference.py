from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .features import TreeFeatureBatch, transform_tree_features
from .hc_calibration import CalibrationState, apply_calibrator
from .hc_features import HierarchyState, transform_hierarchy


class HCInferenceError(ValueError):
    """Raised when the accepted hierarchical predictor changes identity."""


TreeTransformer = Callable[[pd.DataFrame, object], TreeFeatureBatch]
HierarchyTransformer = Callable[[pd.DataFrame, object], pd.DataFrame]


class HCInferenceRuntime:
    def __init__(
        self,
        *,
        baseline_predictor: object,
        tree_state: object,
        hierarchy_state: object,
        c1_models: Sequence[object],
        feature_columns: Sequence[str],
        categorical_columns: Sequence[str],
        calibration_state: CalibrationState | None = None,
        calibration_alpha: float | None = None,
        tree_transformer: TreeTransformer = transform_tree_features,
        hierarchy_transformer: HierarchyTransformer = transform_hierarchy,
    ) -> None:
        models = tuple(c1_models)
        features = tuple(feature_columns)
        categorical = tuple(categorical_columns)
        if not hasattr(baseline_predictor, "predict"):
            raise HCInferenceError("baseline predictor differs")
        if len(models) != 3 or any(not hasattr(model, "predict") for model in models):
            raise HCInferenceError("exactly three C1 models are required")
        if (
            not features
            or len(features) != len(set(features))
            or not set(categorical).issubset(features)
        ):
            raise HCInferenceError("feature schema differs")
        if (calibration_state is None) != (calibration_alpha is None):
            raise HCInferenceError("C2 calibration identity differs")
        if not callable(tree_transformer) or not callable(hierarchy_transformer):
            raise HCInferenceError("feature transformer differs")
        self.baseline_predictor = baseline_predictor
        self.tree_state = tree_state
        self.hierarchy_state = hierarchy_state
        self.c1_models = models
        self.feature_columns = features
        self.categorical_columns = categorical
        self.calibration_state = calibration_state
        self.calibration_alpha = calibration_alpha
        self.tree_transformer = tree_transformer
        self.hierarchy_transformer = hierarchy_transformer

    def _feature_frame(self, rows: pd.DataFrame, p0: np.ndarray) -> pd.DataFrame:
        tree = self.tree_transformer(rows.copy(deep=True), self.tree_state)
        if type(tree) is not TreeFeatureBatch or tree.target is not None:
            raise HCInferenceError("evaluation tree feature batch differs")
        expected_ids = rows["row_id"].astype(str).to_numpy()
        if not np.array_equal(tree.row_id.astype(str), expected_ids):
            raise HCInferenceError("transformed row order differs")
        tree_frame = tree.frame.reset_index(drop=True).add_prefix("tree__")
        hierarchy = self.hierarchy_transformer(
            rows.copy(deep=True), self.hierarchy_state
        ).reset_index(drop=True)
        if len(tree_frame) != len(rows) or len(hierarchy) != len(rows):
            raise HCInferenceError("transformed row count differs")
        frame = pd.concat([tree_frame, hierarchy], axis=1)
        frame["p0"] = p0
        if frame.columns.has_duplicates or tuple(frame.columns) != self.feature_columns:
            raise HCInferenceError("feature schema differs")
        return frame

    def predict(self, rows: pd.DataFrame, *, batch_size: int = 4096) -> np.ndarray:
        if type(rows) is not pd.DataFrame or rows.empty:
            raise HCInferenceError("evaluation rows must be a non-empty DataFrame")
        if "control_success" in rows or "target" in rows:
            raise HCInferenceError("evaluation rows contain target")
        if (
            "row_id" not in rows
            or rows["row_id"].isna().any()
            or not rows["row_id"].is_unique
        ):
            raise HCInferenceError("row identity differs")
        if type(batch_size) is not int or type(batch_size) is bool or batch_size <= 0:
            raise HCInferenceError("batch size differs")
        p0 = np.asarray(
            self.baseline_predictor.predict(rows.copy(deep=True), batch_size=batch_size),
            dtype="float64",
        )
        if (
            p0.shape != (len(rows),)
            or not np.isfinite(p0).all()
            or np.any((p0 < 0) | (p0 > 1))
        ):
            raise HCInferenceError("baseline probability differs")
        frame = self._feature_frame(rows, p0)
        residuals = [
            np.asarray(model.predict(frame), dtype="float64") for model in self.c1_models
        ]
        if any(item.shape != p0.shape or not np.isfinite(item).all() for item in residuals):
            raise HCInferenceError("C1 residual prediction differs")
        p1 = np.clip(p0 + np.mean(np.stack(residuals), axis=0), 1e-5, 1 - 1e-5)
        if self.calibration_state is None:
            return p1
        calibration_rows = rows.copy(deep=True)
        calibration_rows["p1"] = p1
        calibrated = apply_calibrator(
            calibration_rows,
            self.calibration_state,
            alpha=float(self.calibration_alpha),
        )
        p2 = calibrated["p2"].to_numpy(dtype="float64")
        if p2.shape != p0.shape or not np.isfinite(p2).all():
            raise HCInferenceError("C2 probability differs")
        return p2


def _aligned_difference(
    baseline_ids: np.ndarray,
    baseline: np.ndarray,
    rows: pd.DataFrame,
    values: np.ndarray,
) -> float:
    positions = {str(row_id): index for index, row_id in enumerate(rows["row_id"])}
    if len(positions) != len(rows) or set(positions) != set(baseline_ids):
        raise HCInferenceError("audit row identity differs")
    aligned = np.asarray([values[positions[str(row_id)]] for row_id in baseline_ids])
    return float(np.max(np.abs(aligned - baseline)))


def audit_row_independence(
    rows: pd.DataFrame,
    predictor: HCInferenceRuntime,
    *,
    tolerance: float,
) -> Mapping[str, object]:
    if type(tolerance) not in {int, float} or not np.isfinite(tolerance) or tolerance < 0:
        raise HCInferenceError("audit tolerance differs")
    baseline = predictor.predict(rows, batch_size=4096)
    baseline_ids = rows["row_id"].astype(str).to_numpy()
    checks: dict[str, float] = {}
    reversed_rows = rows.iloc[::-1].reset_index(drop=True)
    checks["reverse"] = _aligned_difference(
        baseline_ids,
        baseline,
        reversed_rows,
        predictor.predict(reversed_rows, batch_size=1),
    )
    shuffled = rows.sample(frac=1.0, random_state=3407).reset_index(drop=True)
    checks["shuffle"] = _aligned_difference(
        baseline_ids,
        baseline,
        shuffled,
        predictor.predict(shuffled, batch_size=7),
    )
    for size in (1, 257, 4096):
        checks[f"batch_{size}"] = float(
            np.max(np.abs(predictor.predict(rows, batch_size=size) - baseline))
        )
    indices = np.unique(np.linspace(0, len(rows) - 1, min(7, len(rows))).astype(int))
    singleton = 0.0
    companion = 0.0
    twin = 0.0
    for index in indices:
        single = predictor.predict(rows.iloc[[int(index)]].copy(), batch_size=1)[0]
        singleton = max(singleton, abs(float(single) - float(baseline[index])))
        other = int((index + 1) % len(rows))
        pair = rows.iloc[[int(index), other]].copy()
        paired = predictor.predict(pair, batch_size=2)[0]
        companion = max(companion, abs(float(paired) - float(baseline[index])))
        duplicated = pd.concat(
            [rows.iloc[[int(index)]], rows.iloc[[int(index)]]], ignore_index=True
        )
        duplicated.loc[1, "row_id"] = f"{duplicated.loc[0, 'row_id']}__audit_twin"
        duplicate_values = predictor.predict(duplicated, batch_size=2)
        twin = max(twin, abs(float(duplicate_values[0] - duplicate_values[1])))
    checks.update(singleton=singleton, companion=companion, feature_twin=twin)
    maximum = max(checks.values(), default=0.0)
    return {
        "status": "passed" if maximum <= float(tolerance) else "failed",
        "row_count": len(rows),
        "checks": checks,
        "maximum_absolute_difference": maximum,
        "tolerance": float(tolerance),
    }
