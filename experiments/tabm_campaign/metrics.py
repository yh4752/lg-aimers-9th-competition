from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


class MetricError(ValueError):
    """Raised when prediction evidence cannot be aligned one-to-one."""


@dataclass(frozen=True)
class EnsembleDiagnostics:
    member_count: int
    brier: float
    probability_correlations: tuple[float | None, ...]
    squared_error_correlations: tuple[float | None, ...]
    component_briers: tuple[float, ...]
    inference_seconds_sum: float


@dataclass(frozen=True)
class SegmentMetric:
    dimension: str
    value: str
    rows: int
    brier: float
    reference_delta: float | None
    hard_gate_eligible: bool


def _target_frame(target: pd.DataFrame) -> pd.DataFrame:
    required = {"row_id", "target"}
    if not required.issubset(target):
        raise MetricError(f"target evidence is missing columns: {sorted(required - set(target))}")
    output = target.loc[:, ["row_id", "target"]].copy()
    output["row_id"] = output["row_id"].astype("string")
    output["target"] = pd.to_numeric(output["target"], errors="coerce")
    if output["row_id"].isna().any() or output["row_id"].duplicated().any():
        raise MetricError("target row_id must be non-null and unique")
    if output["target"].isna().any() or not output["target"].isin([0, 1]).all():
        raise MetricError("targets must be finite binary values")
    return output


def _prediction_frame(prediction: pd.DataFrame) -> pd.DataFrame:
    required = {"row_id", "probability"}
    if not required.issubset(prediction):
        raise MetricError(f"prediction evidence is missing columns: {sorted(required - set(prediction))}")
    output = prediction.loc[:, ["row_id", "probability"]].copy()
    output["row_id"] = output["row_id"].astype("string")
    output["probability"] = pd.to_numeric(output["probability"], errors="coerce")
    if output["row_id"].isna().any() or output["row_id"].duplicated().any():
        raise MetricError("prediction row_id must be non-null and unique")
    values = output["probability"].to_numpy(dtype="float64")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise MetricError("probabilities must be finite values in [0, 1]")
    return output


def align_predictions(target: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    truth = _target_frame(target)
    predicted = _prediction_frame(prediction)
    aligned = truth.merge(predicted, on="row_id", how="outer", validate="one_to_one", indicator=True)
    if not aligned["_merge"].eq("both").all() or len(aligned) != len(truth):
        raise MetricError("prediction row IDs do not match target row IDs one-to-one")
    return aligned.drop(columns="_merge")


def aligned_brier(target: pd.DataFrame, prediction: pd.DataFrame) -> float:
    aligned = align_predictions(target, prediction)
    return float(np.mean(np.square(aligned["probability"] - aligned["target"])))


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def ensemble_diagnostics(
    target: pd.DataFrame,
    predictions: Iterable[pd.DataFrame],
    *,
    inference_seconds: Iterable[float] | None = None,
) -> EnsembleDiagnostics:
    truth = _target_frame(target)
    members = list(predictions)
    if not 1 <= len(members) <= 3:
        raise MetricError("an equal-mean ensemble must contain one to three members")
    arrays: list[np.ndarray] = []
    component_briers: list[float] = []
    targets = truth["target"].to_numpy(dtype="float64")
    for member in members:
        aligned = align_predictions(truth, member)
        values = aligned["probability"].to_numpy(dtype="float64")
        arrays.append(values)
        component_briers.append(float(np.mean(np.square(values - targets))))
    mean_probability = np.mean(np.vstack(arrays), axis=0)
    probability_correlations: list[float | None] = []
    squared_error_correlations: list[float | None] = []
    for left in range(len(arrays)):
        for right in range(left + 1, len(arrays)):
            probability_correlations.append(_correlation(arrays[left], arrays[right]))
            squared_error_correlations.append(
                _correlation(np.square(arrays[left] - targets), np.square(arrays[right] - targets))
            )
    seconds = tuple(float(value) for value in (inference_seconds or (0.0,) * len(members)))
    if len(seconds) != len(members) or any(value < 0 or not math.isfinite(value) for value in seconds):
        raise MetricError("inference times must be finite, non-negative, and member-aligned")
    return EnsembleDiagnostics(
        member_count=len(members),
        brier=float(np.mean(np.square(mean_probability - targets))),
        probability_correlations=tuple(probability_correlations),
        squared_error_correlations=tuple(squared_error_correlations),
        component_briers=tuple(component_briers),
        inference_seconds_sum=sum(seconds),
    )


def segment_metrics(
    frame: pd.DataFrame,
    *,
    probability: np.ndarray,
    target: np.ndarray,
    reference_probability: np.ndarray | None = None,
    category_maps: Mapping[str, Mapping[str, int]] | None = None,
    min_hard_gate_rows: int = 1000,
) -> tuple[SegmentMetric, ...]:
    """Report predeclared row-local segments; OOV uses only frozen train maps."""

    if len(frame) != len(probability) or len(frame) != len(target):
        raise MetricError("segment arrays must be row-aligned")
    working = frame.reset_index(drop=True).copy()
    working["__probability"] = np.asarray(probability, dtype="float64")
    working["__target"] = np.asarray(target, dtype="float64")
    if reference_probability is not None:
        working["__reference"] = np.asarray(reference_probability, dtype="float64")
    dimensions: list[str] = []
    for column in ("game_type", "game_month"):
        if column in working:
            dimensions.append(column)
    for entity in ("pitcher_id", "batter_id"):
        if entity in working and category_maps is not None and entity in category_maps:
            known = set(str(key) for key in category_maps[entity])
            name = f"{entity}_known"
            working[name] = working[entity].astype("string").isin(known).map({True: "known", False: "oov"})
            dimensions.append(name)
    rows: list[SegmentMetric] = []
    for dimension in dimensions:
        for value, group in working.groupby(dimension, dropna=False, sort=True):
            brier = float(np.mean(np.square(group["__probability"] - group["__target"])))
            delta = None
            if "__reference" in group:
                reference = float(np.mean(np.square(group["__reference"] - group["__target"])))
                delta = brier - reference
            rows.append(
                SegmentMetric(
                    dimension,
                    str(value),
                    len(group),
                    brier,
                    delta,
                    len(group) >= min_hard_gate_rows,
                )
            )
    return tuple(rows)
