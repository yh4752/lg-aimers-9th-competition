from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


class T3DiagnosticError(ValueError):
    pass


SEGMENTS = ("game_type", "control_success", "pitcher_n_bucket", "hand_matchup", "count_state")


def _arrays(target: np.ndarray, candidate: np.ndarray, baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = tuple(np.asarray(item, dtype="float64") for item in (target, candidate, baseline))
    if values[0].ndim != 1 or any(item.shape != values[0].shape for item in values):
        raise T3DiagnosticError("diagnostic arrays must be aligned vectors")
    if any(not np.isfinite(item).all() for item in values):
        raise T3DiagnosticError("diagnostic arrays must be finite")
    return values


def segment_values(rows: pd.DataFrame, column: str) -> pd.Series:
    if column in rows:
        return rows[column].astype("string").fillna("__MISSING__")
    if column == "pitcher_n_bucket":
        return pd.cut(
            pd.to_numeric(rows["asof_pitcher_n"], errors="coerce"),
            [-np.inf, 0, 25, 100, 500, np.inf],
            labels=["zero", "1_25", "26_100", "101_500", "over_500"],
        ).astype("string").fillna("__MISSING__")
    if column == "hand_matchup":
        return rows["pitcher_hand"].astype("string").fillna("__MISSING__").str.cat(
            rows["batter_hand"].astype("string").fillna("__MISSING__"), sep="|",
        )
    if column == "count_state":
        return rows["balls_before"].astype("string").fillna("__MISSING__").str.cat(
            rows["strikes_before"].astype("string").fillna("__MISSING__"), sep="|",
        )
    raise T3DiagnosticError(f"unsupported segment: {column}")


def segment_diagnostics(
    rows: pd.DataFrame,
    target: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    minimum_rows: int,
) -> pd.DataFrame:
    if type(rows) is not pd.DataFrame or "control_success" not in rows:
        raise T3DiagnosticError("labeled OOF rows are required")
    if type(minimum_rows) is not int or minimum_rows < 1:
        raise T3DiagnosticError("minimum segment rows differ")
    target, candidate, baseline = _arrays(target, candidate, baseline)
    if len(rows) != len(target):
        raise T3DiagnosticError("labeled OOF rows are required")
    source = rows.reset_index(drop=True)
    records: list[dict[str, object]] = []
    for column in SEGMENTS:
        values = segment_values(source, column)
        for value in sorted(values.unique().tolist()):
            positions = np.flatnonzero(values.eq(value).to_numpy())
            if len(positions) < minimum_rows:
                continue
            base_brier = float(np.mean(np.square(baseline[positions] - target[positions])))
            candidate_brier = float(np.mean(np.square(candidate[positions] - target[positions])))
            records.append({
                "segment": column, "value": str(value), "rows": len(positions),
                "baseline_brier": base_brier, "candidate_brier": candidate_brier,
                "gain": base_brier - candidate_brier,
            })
    columns = ["segment", "value", "rows", "baseline_brier", "candidate_brier", "gain"]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records(records, columns=columns).sort_values(
        ["segment", "value"], kind="stable", ignore_index=True,
    )


def calibration_diagnostics(target: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    target, prediction, _ = _arrays(target, prediction, prediction)
    bucket = np.clip(np.digitize(prediction, np.linspace(0.0, 1.0, 11)[1:-1]), 0, 9)
    return pd.DataFrame({"target": target, "prediction": prediction, "bin": bucket}).groupby(
        "bin", sort=True, observed=False,
    ).agg(
        rows=("target", "size"),
        mean_target=("target", "mean"),
        mean_prediction=("prediction", "mean"),
    ).reset_index()


def residual_correlation(target: np.ndarray, predictions: Mapping[str, np.ndarray]) -> pd.DataFrame:
    if type(predictions) not in {dict, Mapping} and not isinstance(predictions, Mapping):
        raise T3DiagnosticError("predictions must be a mapping")
    names = tuple(sorted(predictions))
    if len(names) < 2:
        raise T3DiagnosticError("at least two predictions are required")
    target = np.asarray(target, dtype="float64")
    columns: list[np.ndarray] = []
    for name in names:
        prediction = np.asarray(predictions[name], dtype="float64")
        if prediction.shape != target.shape:
            raise T3DiagnosticError("residual arrays differ")
        columns.append(target - prediction)
    matrix = np.corrcoef(np.column_stack(columns), rowvar=False)
    return pd.DataFrame(matrix, index=names, columns=names)
