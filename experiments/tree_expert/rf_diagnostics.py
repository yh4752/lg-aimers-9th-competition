from __future__ import annotations

import math

import numpy as np
import pandas as pd


class RFDiagnosticError(ValueError):
    pass


def _vector(value: object, length: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype="float64")
    if (
        result.shape != (length,)
        or not np.isfinite(result).all()
        or np.any((result < 0) | (result > 1))
    ):
        raise RFDiagnosticError(f"{label} probability differs")
    return result


def _brier(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.square(prediction - target)))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 1.0 if np.array_equal(left, right) else 0.0
    result = float(np.corrcoef(left, right)[0, 1])
    return result if math.isfinite(result) else 0.0


def build_rf_diagnostics(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    minimum_rows: int,
) -> dict[str, object]:
    required = {"row_id", "target", "game_type"}
    if type(frame) is not pd.DataFrame or not required.issubset(frame.columns) or frame.empty:
        raise RFDiagnosticError("diagnostic frame differs")
    if type(minimum_rows) is not int or minimum_rows <= 0:
        raise RFDiagnosticError("minimum rows differs")
    target = pd.to_numeric(frame["target"], errors="raise").to_numpy(dtype="float64")
    if not np.isin(target, [0.0, 1.0]).all():
        raise RFDiagnosticError("diagnostic target differs")
    base = _vector(baseline, len(frame), "baseline")
    pred = _vector(candidate, len(frame), "candidate")
    game_type = frame["game_type"].astype(str).to_numpy()
    if not np.isin(game_type, ["R", "F"]).all():
        raise RFDiagnosticError("diagnostic game_type differs")

    base_brier = _brier(target, base)
    candidate_brier = _brier(target, pred)
    segments: list[dict[str, object]] = []
    for name in ("R", "F"):
        mask = game_type == name
        count = int(mask.sum())
        if count < minimum_rows:
            continue
        segment_base = _brier(target[mask], base[mask])
        segment_candidate = _brier(target[mask], pred[mask])
        segments.append(
            {
                "segment": name,
                "rows": count,
                "baseline_brier": segment_base,
                "candidate_brier": segment_candidate,
                "gain": segment_base - segment_candidate,
            }
        )

    bins = np.minimum(np.floor(pred * 10).astype("int64"), 9)
    calibration: list[dict[str, object]] = []
    for index in range(10):
        mask = bins == index
        calibration.append(
            {
                "bin": index,
                "lower": index / 10,
                "upper": (index + 1) / 10,
                "rows": int(mask.sum()),
                "mean_probability": float(np.mean(pred[mask])) if mask.any() else None,
                "observed_rate": float(np.mean(target[mask])) if mask.any() else None,
            }
        )
    return {
        "overall": {
            "rows": len(frame),
            "baseline_brier": base_brier,
            "candidate_brier": candidate_brier,
            "gain": base_brier - candidate_brier,
        },
        "game_type": segments,
        "calibration": calibration,
        "prediction_correlation": _correlation(base, pred),
        "residual_correlation": _correlation(target - base, target - pred),
    }
