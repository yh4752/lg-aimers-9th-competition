from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClusterBootstrapResult:
    cluster_column: str
    repeats: int
    seed: int
    mean_gain: float
    lower_95: float
    upper_95: float


def _arrays(target: object, probability: object) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(target, dtype="float64")
    predicted = np.asarray(probability, dtype="float64")
    if (
        truth.ndim != 1
        or predicted.shape != truth.shape
        or truth.size == 0
        or not np.isfinite(truth).all()
        or not np.isfinite(predicted).all()
        or not np.isin(truth, (0.0, 1.0)).all()
        or np.any((predicted < 0) | (predicted > 1))
    ):
        raise ValueError("target or probability values differ")
    return truth, predicted


def brier_score(target: object, probability: object) -> float:
    truth, predicted = _arrays(target, probability)
    return float(np.mean(np.square(predicted - truth)))


def calibration_gap(target: object, probability: object) -> float:
    truth, predicted = _arrays(target, probability)
    return float(abs(np.mean(truth) - np.mean(predicted)))


def expected_calibration_error(
    target: object, probability: object, *, bins: int = 10
) -> float:
    truth, predicted = _arrays(target, probability)
    if type(bins) is not int or type(bins) is bool or bins <= 1:
        raise ValueError("ECE bin count differs")
    indices = np.minimum((predicted * bins).astype("int64"), bins - 1)
    result = 0.0
    for index in range(bins):
        selected = indices == index
        if selected.any():
            result += float(selected.mean()) * abs(
                float(predicted[selected].mean()) - float(truth[selected].mean())
            )
    return result


def fit_c0_decile_edges(probability: object) -> tuple[float, ...]:
    values = np.asarray(probability, dtype="float64")
    if values.ndim != 1 or values.size < 10 or not np.isfinite(values).all():
        raise ValueError("C0 decile evidence differs")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("C0 decile probability differs")
    edges = np.quantile(values, np.linspace(0.0, 1.0, 11))
    edges[0] = min(0.0, edges[0])
    edges[-1] = max(1.0, edges[-1])
    return tuple(float(value) for value in edges)


def apply_c0_deciles(probability: object, edges: Sequence[float]) -> np.ndarray:
    values = np.asarray(probability, dtype="float64")
    boundaries = np.asarray(tuple(edges), dtype="float64")
    if boundaries.shape != (11,) or np.any(np.diff(boundaries) < 0):
        raise ValueError("C0 decile edges differ")
    return np.searchsorted(boundaries[1:-1], values, side="right").astype("int8")


def paired_cluster_bootstrap(
    frame: pd.DataFrame,
    base_column: str,
    candidate_column: str,
    *,
    repeats: int,
    seed: int,
    cluster_column: str = "pitcher_id",
) -> ClusterBootstrapResult:
    required = {"target", base_column, candidate_column, cluster_column}
    if type(frame) is not pd.DataFrame or not required.issubset(frame) or frame.empty:
        raise ValueError("bootstrap frame differs")
    if type(repeats) is not int or repeats <= 0 or type(seed) is not int:
        raise ValueError("bootstrap registry differs")
    if frame[cluster_column].isna().any():
        raise ValueError("bootstrap cluster values differ")
    truth, base = _arrays(frame["target"], frame[base_column])
    _, candidate = _arrays(frame["target"], frame[candidate_column])
    row_gain = np.square(base - truth) - np.square(candidate - truth)
    clusters = frame[cluster_column].astype(str).to_numpy()
    names = np.asarray(sorted(set(clusters)), dtype=object)
    positions = {name: np.flatnonzero(clusters == name) for name in names}
    generator = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype="float64")
    for repeat in range(repeats):
        selected = generator.choice(names, size=len(names), replace=True)
        indices = np.concatenate([positions[name] for name in selected])
        samples[repeat] = float(row_gain[indices].mean())
    return ClusterBootstrapResult(
        cluster_column=cluster_column,
        repeats=repeats,
        seed=seed,
        mean_gain=float(row_gain.mean()),
        lower_95=float(np.quantile(samples, 0.025)),
        upper_95=float(np.quantile(samples, 0.975)),
    )


def _segment_values(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "validation_year": frame["oof_year"].astype(str),
        "game_type": frame["game_type"].astype("string").fillna("__MISSING__"),
        "pitcher_known": frame["pitcher_id_known"].astype("string").fillna("__MISSING__"),
        "batter_known": frame["batter_id_known"].astype("string").fillna("__MISSING__"),
        "hand_matchup": frame["pitcher_hand"].astype(str).str.cat(
            frame["batter_hand"].astype(str), sep="|"
        ),
        "count_state": frame["balls_before"].astype(str).str.cat(
            frame["strikes_before"].astype(str), sep="|"
        ),
        "base_state": frame["base_state"].astype("string").fillna("__MISSING__"),
        "c0_decile": frame["c0_decile"].astype(str),
    }


def segment_diagnostics(
    frame: pd.DataFrame, candidate_column: str, *, minimum_rows: int
) -> pd.DataFrame:
    required = {
        "target",
        "p0",
        candidate_column,
        "oof_year",
        "game_type",
        "pitcher_id_known",
        "batter_id_known",
        "pitcher_hand",
        "batter_hand",
        "balls_before",
        "strikes_before",
        "base_state",
        "c0_decile",
    }
    if type(frame) is not pd.DataFrame or not required.issubset(frame) or frame.empty:
        raise ValueError("segment frame differs")
    if type(minimum_rows) is not int or minimum_rows <= 0:
        raise ValueError("minimum segment rows differ")
    records: list[dict[str, object]] = []
    for family, values in _segment_values(frame).items():
        for value in sorted(values.astype(str).unique()):
            selected = values.astype(str).eq(value).to_numpy()
            base = brier_score(frame.loc[selected, "target"], frame.loc[selected, "p0"])
            candidate = brier_score(
                frame.loc[selected, "target"], frame.loc[selected, candidate_column]
            )
            records.append(
                {
                    "family": family,
                    "value": value,
                    "rows": int(selected.sum()),
                    "base_brier": base,
                    "candidate_brier": candidate,
                    "regression": candidate - base,
                    "eligible": bool(selected.sum() >= minimum_rows),
                }
            )
    return pd.DataFrame(records)


def maximum_segment_regression(report: pd.DataFrame) -> float:
    if type(report) is not pd.DataFrame or not {"eligible", "regression"}.issubset(report):
        raise ValueError("segment report differs")
    eligible = report.loc[report["eligible"].astype(bool), "regression"]
    return 0.0 if eligible.empty else float(eligible.max())
