from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .features import TreeFeatureBatch, transform_tree_features
from .rf_decisions import RFDecisionError, route_probability


class RFInferenceError(ValueError):
    pass


Transformer = Callable[[pd.DataFrame, object], TreeFeatureBatch]


class RFInferenceRuntime:
    def __init__(
        self,
        *,
        baseline_predictor: object,
        f_state: object,
        f_models: Sequence[object],
        alpha_f: float,
        r_state: object | None = None,
        r_models: Sequence[object] = (),
        alpha_r: float = 0.0,
        transformer: Transformer = transform_tree_features,
    ) -> None:
        f_members = tuple(f_models)
        r_members = tuple(r_models)
        if not hasattr(baseline_predictor, "predict"):
            raise RFInferenceError("baseline predictor differs")
        if len(f_members) != 3 or any(not hasattr(model, "predict") for model in f_members):
            raise RFInferenceError("exactly three F models are required")
        if alpha_r > 0 and (
            r_state is None
            or len(r_members) != 3
            or any(not hasattr(model, "predict") for model in r_members)
        ):
            raise RFInferenceError("R expert identity differs")
        if alpha_r == 0 and (r_state is not None or r_members):
            raise RFInferenceError("F-only runtime contains an R expert")
        if not callable(transformer):
            raise RFInferenceError("feature transformer differs")
        self.baseline_predictor = baseline_predictor
        self.f_state = f_state
        self.f_models = f_members
        self.alpha_f = float(alpha_f)
        self.r_state = r_state
        self.r_models = r_members
        self.alpha_r = float(alpha_r)
        self.transformer = transformer

    def _expert_probability(
        self,
        rows: pd.DataFrame,
        state: object,
        models: tuple[object, ...],
    ) -> np.ndarray:
        batch = self.transformer(rows.copy(deep=True), state)
        if type(batch) is not TreeFeatureBatch or batch.target is not None:
            raise RFInferenceError("evaluation feature batch differs")
        expected = rows["row_id"].astype(str).to_numpy()
        if not np.array_equal(batch.row_id.astype(str), expected):
            raise RFInferenceError("transformed row order differs")
        anchor = np.asarray(batch.anchor, dtype="float64")
        if anchor.shape != (len(rows),) or not np.isfinite(anchor).all():
            raise RFInferenceError("anchor values differ")
        members: list[np.ndarray] = []
        for model in models:
            residual = np.asarray(model.predict(batch.frame), dtype="float64")
            if residual.shape != anchor.shape or not np.isfinite(residual).all():
                raise RFInferenceError("expert residual differs")
            members.append(np.clip(anchor + residual, 1e-5, 1 - 1e-5))
        return np.mean(np.stack(members), axis=0)

    def predict(self, rows: pd.DataFrame, *, batch_size: int = 4096) -> np.ndarray:
        if type(rows) is not pd.DataFrame or rows.empty:
            raise RFInferenceError("evaluation rows must be a non-empty DataFrame")
        if "control_success" in rows:
            raise RFInferenceError("evaluation rows contain target")
        required = {"row_id", "game_type"}
        if not required.issubset(rows.columns) or rows["row_id"].isna().any() or not rows["row_id"].is_unique:
            raise RFInferenceError("row identity differs")
        game_type = rows["game_type"].astype(str).to_numpy()
        if not np.isin(game_type, ["R", "F"]).all():
            raise RFInferenceError("game_type differs")
        if type(batch_size) is not int or batch_size <= 0:
            raise RFInferenceError("batch size differs")
        baseline = np.asarray(
            self.baseline_predictor.predict(rows.copy(deep=True), batch_size=batch_size),
            dtype="float64",
        )
        if baseline.shape != (len(rows),) or not np.isfinite(baseline).all():
            raise RFInferenceError("baseline probability differs")
        f_probability = self._expert_probability(rows, self.f_state, self.f_models)
        r_probability = baseline
        if self.alpha_r > 0:
            r_probability = self._expert_probability(rows, self.r_state, self.r_models)
        try:
            return route_probability(
                baseline,
                r_probability,
                f_probability,
                game_type,
                self.alpha_r,
                self.alpha_f,
            )
        except RFDecisionError as error:
            raise RFInferenceError(str(error)) from error


def _aligned_difference(
    baseline_ids: np.ndarray,
    baseline: np.ndarray,
    rows: pd.DataFrame,
    values: np.ndarray,
) -> float:
    positions = {str(row_id): index for index, row_id in enumerate(rows["row_id"])}
    if len(positions) != len(rows) or set(positions) != set(baseline_ids):
        raise RFInferenceError("audit row identity differs")
    aligned = np.asarray([values[positions[str(row_id)]] for row_id in baseline_ids])
    return float(np.max(np.abs(aligned - baseline)))


def audit_row_independence(
    rows: pd.DataFrame,
    predictor: RFInferenceRuntime,
    *,
    tolerance: float,
) -> Mapping[str, object]:
    if type(tolerance) not in {int, float} or tolerance < 0:
        raise RFInferenceError("audit tolerance differs")
    baseline = predictor.predict(rows, batch_size=4096)
    baseline_ids = rows["row_id"].astype(str).to_numpy()
    checks: dict[str, float] = {}

    singleton = np.asarray([
        predictor.predict(rows.iloc[[index]].copy(), batch_size=1)[0]
        for index in range(len(rows))
    ])
    checks["singleton"] = float(np.max(np.abs(singleton - baseline)))

    shuffled_rows = rows.sample(frac=1, random_state=3407).reset_index(drop=True)
    checks["shuffle"] = _aligned_difference(
        baseline_ids,
        baseline,
        shuffled_rows,
        predictor.predict(shuffled_rows, batch_size=7),
    )
    reversed_rows = rows.iloc[::-1].reset_index(drop=True)
    checks["reverse"] = _aligned_difference(
        baseline_ids,
        baseline,
        reversed_rows,
        predictor.predict(reversed_rows, batch_size=64),
    )

    companion_difference = 0.0
    for index in range(min(len(rows), 32)):
        other = (index + 1) % len(rows)
        pair = rows.iloc[[index, other]].copy()
        value = predictor.predict(pair, batch_size=2)[0]
        companion_difference = max(companion_difference, abs(float(value - baseline[index])))
    checks["companion"] = companion_difference

    for size in (1, 7, 64, 4096):
        values = predictor.predict(rows, batch_size=size)
        checks[f"batch_{size}"] = float(np.max(np.abs(values - baseline)))
    maximum = max(checks.values(), default=0.0)
    return {
        "status": "passed" if maximum <= float(tolerance) else "failed",
        "row_count": len(rows),
        "checks": checks,
        "maximum_absolute_difference": maximum,
        "tolerance": float(tolerance),
    }
