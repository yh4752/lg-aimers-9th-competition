from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd


class OOFError(ValueError):
    pass


_REQUIRED = {"row_id", "target", "probability", "oof_year"}


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame or frame.empty or not _REQUIRED.issubset(frame.columns):
        raise OOFError("prediction columns differ")
    output = frame.copy(deep=True)
    row_id = pd.to_numeric(output["row_id"], errors="coerce")
    target = pd.to_numeric(output["target"], errors="coerce")
    probability = pd.to_numeric(output["probability"], errors="coerce")
    year = pd.to_numeric(output["oof_year"], errors="coerce")
    if row_id.isna().any() or not row_id.is_unique:
        raise OOFError("row identity differs")
    if not target.isin((0, 1)).all():
        raise OOFError("target values differ")
    if probability.isna().any() or not np.isfinite(probability).all() or not probability.between(0, 1).all():
        raise OOFError("probability values differ")
    if year.isna().any() or not np.equal(year, np.floor(year)).all():
        raise OOFError("OOF year values differ")
    output["row_id"] = row_id.astype("int64")
    output["target"] = target.astype("int8")
    output["probability"] = probability.astype("float64")
    output["oof_year"] = year.astype("int16")
    return output.sort_values("row_id", kind="stable").reset_index(drop=True)


def align_predictions(frames: Mapping[str, pd.DataFrame]) -> Mapping[str, pd.DataFrame]:
    if not frames:
        raise OOFError("prediction frames are absent")
    normalized = {str(name): normalize_frame(frame) for name, frame in frames.items()}
    anchor = next(iter(normalized.values()))
    for name, frame in normalized.items():
        if not np.array_equal(anchor["row_id"].to_numpy(), frame["row_id"].to_numpy()):
            raise OOFError(f"row identity differs: {name}")
        if not np.array_equal(anchor["target"].to_numpy(), frame["target"].to_numpy()):
            raise OOFError(f"target values differ: {name}")
        if not np.array_equal(anchor["oof_year"].to_numpy(), frame["oof_year"].to_numpy()):
            raise OOFError(f"OOF year values differ: {name}")
    return MappingProxyType(normalized)


def direct_probability(*, game_type: np.ndarray, d0: np.ndarray, d5: np.ndarray) -> np.ndarray:
    groups = np.asarray(game_type).astype(str)
    global_probability = np.asarray(d0, dtype="float64")
    regular_probability = np.asarray(d5, dtype="float64")
    if groups.ndim != 1 or global_probability.shape != groups.shape or regular_probability.shape != groups.shape:
        raise OOFError("direct probability shapes differ")
    if (
        not np.isfinite(global_probability).all()
        or not np.isfinite(regular_probability).all()
        or np.any((global_probability < 0) | (global_probability > 1))
        or np.any((regular_probability < 0) | (regular_probability > 1))
    ):
        raise OOFError("direct probability values differ")
    return np.where(groups == "R", regular_probability, global_probability)
