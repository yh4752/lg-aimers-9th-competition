"""Paired metrics and deterministic promotion for preprocessing candidates."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Iterable

import numpy as np
import pandas as pd


COMPONENT_ORDER = (
    "pitcher_smooth",
    "batter_smooth",
    "dl_selective_transform",
    "asof_count_log1p",
    "entity_frequency_log1p",
    "grouped_missing_indicators",
    "hand_matchup",
    "count_state",
    "pitcher_team_win_expectancy",
)
METRIC_COLUMNS = (
    "anchor_id",
    "preprocessing_id",
    "seed",
    "fold",
    "valid_rows",
    "brier",
    "pitcher_oov_brier",
    "batter_oov_brier",
)
PREDICTION_COLUMNS = (
    "row_id",
    "fold",
    "season",
    "game_type",
    "target",
    "probability",
    "anchor_id",
    "preprocessing_id",
    "seed",
    "pitcher_oov",
    "batter_oov",
)


@dataclass(frozen=True)
class CombinationSetting:
    preprocessing_id: str
    components: tuple[str, ...]
    parent_id: str | None = None


def build_metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate aligned prediction rows into fold and OOV metrics."""

    if not isinstance(predictions, pd.DataFrame):
        raise ValueError("prediction evidence must be a DataFrame")
    missing = set(PREDICTION_COLUMNS) - set(predictions)
    if missing:
        raise ValueError(f"prediction columns are missing: {sorted(missing)}")
    frame = predictions.loc[:, PREDICTION_COLUMNS].copy()
    keys = ["row_id", "anchor_id", "preprocessing_id", "seed", "fold"]
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("prediction row bindings must be non-null and unique")
    target = pd.to_numeric(frame["target"], errors="coerce").to_numpy(
        dtype="float64"
    )
    probability = pd.to_numeric(frame["probability"], errors="coerce").to_numpy(
        dtype="float64"
    )
    if not np.isfinite(target).all() or not np.isin(target, (0.0, 1.0)).all():
        raise ValueError("prediction targets must be finite binary values")
    if not np.isfinite(probability).all() or (
        (probability < 0.0) | (probability > 1.0)
    ).any():
        raise ValueError("prediction probabilities must be inside [0, 1]")
    frame["target"] = target
    frame["probability"] = probability
    frame["squared_error"] = np.square(probability - target)
    for column in ("pitcher_oov", "batter_oov"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.isin((0, 1)).all():
            raise ValueError(f"{column} must contain only 0 and 1")
        frame[column] = values.astype(bool)
    rows: list[dict[str, object]] = []
    group_columns = ["anchor_id", "preprocessing_id", "seed", "fold"]
    for binding, group in frame.groupby(group_columns, sort=True, observed=True):
        pitcher = group.loc[group["pitcher_oov"]]
        batter = group.loc[group["batter_oov"]]
        rows.append(
            {
                **dict(zip(group_columns, binding)),
                "valid_rows": len(group),
                "brier": float(group["squared_error"].mean()),
                "pitcher_oov_brier": (
                    float(pitcher["squared_error"].mean())
                    if not pitcher.empty
                    else float(group["squared_error"].mean())
                ),
                "batter_oov_brier": (
                    float(batter["squared_error"].mean())
                    if not batter.empty
                    else float(group["squared_error"].mean())
                ),
            }
        )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def _component_rank(component: str) -> tuple[int, int | str]:
    if component.startswith("pitcher_smooth_k"):
        return 0, int(component.removeprefix("pitcher_smooth_k"))
    if component.startswith("batter_smooth_k"):
        return 1, int(component.removeprefix("batter_smooth_k"))
    try:
        return COMPONENT_ORDER.index(component), component
    except ValueError as error:
        raise ValueError(f"unknown combination component: {component}") from error


def _canonical(components: Iterable[str]) -> tuple[str, ...]:
    values = tuple(sorted(set(components), key=_component_rank))
    if len(values) != len(tuple(components)):
        raise ValueError("combination components must be unique")
    return values


def _combination(components: Iterable[str], parent_id: str | None = None) -> CombinationSetting:
    canonical = _canonical(tuple(components))
    return CombinationSetting("combo__" + "__".join(canonical), canonical, parent_id)


def generate_pairwise_settings(
    *, pitcher_component: str, batter_component: str
) -> tuple[CombinationSetting, ...]:
    """Generate every unordered pair from the fixed nine-component set."""

    components = (
        pitcher_component,
        batter_component,
        "dl_selective_transform",
        "asof_count_log1p",
        "entity_frequency_log1p",
        "grouped_missing_indicators",
        "hand_matchup",
        "count_state",
        "pitcher_team_win_expectancy",
    )
    if len(set(components)) != 9:
        raise ValueError("pairwise search requires nine unique components")
    return tuple(_combination(pair) for pair in itertools.combinations(components, 2))


def next_beam(
    ranked: pd.DataFrame,
    *,
    beam_width: int,
    max_components: int,
) -> tuple[CombinationSetting, ...]:
    """Expand the fixed best parents by one component, then deduplicate."""

    required = {
        "preprocessing_id",
        "components",
        "weighted_delta",
        "improved_folds",
        "worst_fold_delta",
        "max_oov_delta",
    }
    if not required.issubset(ranked):
        raise ValueError("beam input columns are incomplete")
    if isinstance(beam_width, bool) or beam_width < 1:
        raise ValueError("beam_width must be positive")
    if isinstance(max_components, bool) or max_components < 3:
        raise ValueError("max_components must be at least three")
    ordered = ranked.sort_values(
        [
            "weighted_delta",
            "improved_folds",
            "worst_fold_delta",
            "max_oov_delta",
            "preprocessing_id",
        ],
        ascending=[True, False, True, True, True],
        kind="stable",
    ).head(beam_width)
    universe: set[str] = set()
    for components in ranked["components"]:
        universe.update(tuple(components))
    output: dict[tuple[str, ...], CombinationSetting] = {}
    for row in ordered.itertuples(index=False):
        parent = _canonical(tuple(row.components))
        if len(parent) >= max_components:
            continue
        for component in sorted(universe - set(parent), key=_component_rank):
            child = _combination(parent + (component,), str(row.preprocessing_id))
            output.setdefault(child.components, child)
    return tuple(output.values())


def _validated_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("paired metrics must be a DataFrame")
    missing = set(METRIC_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"paired metric columns are missing: {sorted(missing)}")
    output = frame.loc[:, METRIC_COLUMNS].copy()
    for column in (
        "valid_rows",
        "brier",
        "pitcher_oov_brier",
        "batter_oov_brier",
    ):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    values = output[
        ["valid_rows", "brier", "pitcher_oov_brier", "batter_oov_brier"]
    ].to_numpy(dtype="float64")
    if not np.isfinite(values).all() or (output["valid_rows"] <= 0).any():
        raise ValueError("paired metrics must be finite with positive row counts")
    keys = ["anchor_id", "preprocessing_id", "seed", "fold"]
    if output.duplicated(keys).any():
        raise ValueError("paired metric keys must be unique")
    return output


def evaluate_paired_setting(
    frame: pd.DataFrame,
    *,
    candidate_id: str,
    baseline_id: str,
) -> dict[str, object]:
    """Evaluate one setting only against matching anchor/fold/seed baselines."""

    metrics = _validated_metrics(frame)
    candidate = metrics.loc[metrics["preprocessing_id"].eq(candidate_id)].copy()
    baseline = metrics.loc[metrics["preprocessing_id"].eq(baseline_id)].copy()
    join = ["anchor_id", "seed", "fold"]
    paired = candidate.merge(
        baseline,
        on=join,
        how="outer",
        suffixes=("_candidate", "_baseline"),
        indicator=True,
        validate="one_to_one",
    )
    if paired.empty or not paired["_merge"].eq("both").all():
        raise ValueError("paired baseline is missing for a candidate anchor/fold/seed")
    if not np.array_equal(
        paired["valid_rows_candidate"].to_numpy(),
        paired["valid_rows_baseline"].to_numpy(),
    ):
        raise ValueError("paired baseline row counts differ")
    paired["delta"] = paired["brier_candidate"] - paired["brier_baseline"]
    paired["pitcher_oov_delta"] = (
        paired["pitcher_oov_brier_candidate"]
        - paired["pitcher_oov_brier_baseline"]
    )
    paired["batter_oov_delta"] = (
        paired["batter_oov_brier_candidate"]
        - paired["batter_oov_brier_baseline"]
    )
    weights = paired["valid_rows_candidate"].to_numpy(dtype="float64")
    weighted_delta = float(np.average(paired["delta"], weights=weights))
    fold_delta = paired.groupby("fold", sort=True, observed=True)["delta"].mean()
    seed_delta = paired.groupby("seed", sort=True, observed=True)["delta"].mean()
    worst_fold = float(fold_delta.max())
    pitcher_oov_delta = float(
        np.average(paired["pitcher_oov_delta"], weights=weights)
    )
    batter_oov_delta = float(
        np.average(paired["batter_oov_delta"], weights=weights)
    )
    gates = {
        "weighted_mean_improved": weighted_delta < 0.0,
        "four_of_five_folds_improved": int((fold_delta < 0.0).sum()) >= 4,
        "two_of_three_seeds_improved": int((seed_delta < 0.0).sum()) >= 2,
        "worst_fold_delta_lte_0_0001": worst_fold <= 0.0001,
        "pitcher_oov_delta_lte_0_0002": pitcher_oov_delta <= 0.0002,
        "batter_oov_delta_lte_0_0002": batter_oov_delta <= 0.0002,
    }
    return {
        "candidate_id": candidate_id,
        "baseline_id": baseline_id,
        "weighted_delta": weighted_delta,
        "improved_folds": int((fold_delta < 0.0).sum()),
        "improved_seeds": int((seed_delta < 0.0).sum()),
        "seed_delta_std": float(seed_delta.std(ddof=0)),
        "worst_fold_delta": worst_fold,
        "pitcher_oov_delta": pitcher_oov_delta,
        "batter_oov_delta": batter_oov_delta,
        "max_oov_delta": max(pitcher_oov_delta, batter_oov_delta),
        "gates": gates,
        "adopt_global": all(gates.values()),
    }
