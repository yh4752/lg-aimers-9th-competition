"""Pitcher-only, cutoff-bound Trackman lookup construction.

Source commit: ``9454d68b93971627e3d3f613ce30be690cb5dce2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np
import pandas as pd


PITCH_GROUPS = ("fastball", "breaking", "offspeed", "other")
PHYSICAL_COLUMNS = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
)
MAIN_RATE_COLUMNS = (
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)
_GROUP_METRICS = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "zone_speed",
)
_MAPPING_COLUMNS = (
    "pitcher_id",
    "tm_match_cost",
    "tm_match_margin",
    "tm_match_confidence",
    "tm_match_accepted",
)
_AGGREGATE_COLUMNS = (
    *(f"tm_career_{metric}_{stat}" for metric in PHYSICAL_COLUMNS for stat in ("mean", "std", "median")),
    "tm_history_n",
    *(f"tm_recent_{metric}_{stat}" for metric in PHYSICAL_COLUMNS for stat in ("mean", "std")),
    *(f"tm_{group}_{metric}_mean" for group in PITCH_GROUPS for metric in _GROUP_METRICS),
    *(f"tm_history_{group}_rate" for group in PITCH_GROUPS),
    *(f"tm_trend_{metric}" for metric in ("rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension")),
    "tm_fastball_breaking_speed_gap",
    "tm_fastball_offspeed_speed_gap",
)
PITCHER_LOOKUP_COLUMNS = (*_MAPPING_COLUMNS, *_AGGREGATE_COLUMNS)


@dataclass(frozen=True)
class TrackmanBuildResult:
    """Pitcher lookup and diagnostics resulting from a cutoff-bound match.

    ``cutoff_year``, ``lookup_schema``, and the construction-time fingerprint
    bind a reusable lookup to the exact temporal state from which it was
    created.  The fingerprint detects ordinary in-memory mutation after
    construction; it is an integrity guard, not a security boundary against a
    caller deliberately forging a result object.
    """

    cutoff_year: int
    lookup: pd.DataFrame
    lookup_schema: tuple[str, ...]
    mapping: pd.DataFrame
    team_mapping: pd.DataFrame
    lookup_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lookup_schema", tuple(self.lookup_schema))
        object.__setattr__(self, "lookup_fingerprint", _lookup_fingerprint(self.lookup))


def _lookup_fingerprint(lookup: pd.DataFrame) -> str:
    """Return a deterministic digest for the lookup's schema, index, and values."""

    if not isinstance(lookup, pd.DataFrame):
        raise ValueError("Trackman lookup must be a DataFrame")
    digest = sha256()
    digest.update(repr(tuple(lookup.columns)).encode("utf-8"))
    digest.update(repr(tuple(str(dtype) for dtype in lookup.dtypes)).encode("utf-8"))
    digest.update(
        pd.util.hash_pandas_object(lookup, index=True, categorize=False)
        .to_numpy(dtype="uint64", copy=False)
        .tobytes()
    )
    return digest.hexdigest()


def validate_trackman_build_result(
    result: TrackmanBuildResult,
    *,
    expected_cutoff_year: int,
) -> pd.DataFrame:
    """Validate a cutoff-bound lookup before a feature merge.

    This verifies the regular replay contract (type, cutoff, fixed schema,
    pitcher-key uniqueness, and construction-time fingerprint).  It detects
    accidental cache mutation and wrong-cutoff reuse, not an adversarial caller
    that intentionally manufactures matching Python objects.
    """

    if not isinstance(result, TrackmanBuildResult):
        raise ValueError("Trackman result must be a TrackmanBuildResult")
    if result.cutoff_year != expected_cutoff_year:
        raise ValueError(
            "Trackman result cutoff does not match the requested cutoff "
            f"{expected_cutoff_year}"
        )
    if result.lookup_schema != PITCHER_LOOKUP_COLUMNS:
        raise ValueError("Trackman result schema evidence is invalid")
    lookup = result.lookup
    if not isinstance(lookup, pd.DataFrame):
        raise ValueError("Trackman result lookup must be a DataFrame")
    if "pitcher_id" not in lookup.columns:
        raise ValueError("Trackman result lookup is missing pitcher_id")
    if tuple(lookup.columns) != PITCHER_LOOKUP_COLUMNS:
        raise ValueError("Trackman result lookup schema has changed")
    if lookup["pitcher_id"].isna().any():
        raise ValueError("Trackman result lookup has null pitcher_id")
    if lookup["pitcher_id"].duplicated().any():
        raise ValueError("Trackman result lookup has duplicate pitcher_id")
    if result.lookup_fingerprint != _lookup_fingerprint(lookup):
        raise ValueError("Trackman result lookup fingerprint does not match")
    return lookup


def _require_linear_sum_assignment():
    """Load the optional matching dependency at an explicit action boundary."""

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:  # pragma: no cover - exercised in SciPy-free runtime
        raise RuntimeError(
            "Trackman matching requires scipy; install scipy==1.16.3 "
            "before building an uncached Trackman lookup."
        ) from error
    return linear_sum_assignment


def _mode_or_first(series: pd.Series):
    mode = series.mode(dropna=True)
    if len(mode):
        return mode.iat[0]
    non_null = series.dropna()
    return non_null.iat[0] if len(non_null) else np.nan


def _empty_mapping() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "pitcher_id",
            "pitcher_trackman_id",
            "main_n",
            "tm_match_cost",
            "tm_match_margin",
            "tm_match_confidence",
            "tm_match_accepted",
            "main_last_year",
        ]
    )


def _main_signatures(
    train: pd.DataFrame,
    cutoff_year: int,
    *,
    pre_sliced: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = (
        "season",
        "pitcher_id",
        "pitcher_hand",
        "pitcher_team_id",
        "asof_pitcher_n",
        *MAIN_RATE_COLUMNS,
    )
    data = train.loc[:, columns] if pre_sliced else train.loc[
        train["season"].le(cutoff_year), columns
    ].copy()
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    index = data.groupby(["pitcher_id", "season"], sort=False)["asof_pitcher_n"].idxmax()
    yearly = data.loc[index].copy()
    yearly["main_n"] = yearly["asof_pitcher_n"].astype("int64") + 1
    yearly["main_hand"] = yearly["pitcher_hand"].map({1: "Left", 2: "Right"})
    yearly = yearly.sort_values(["pitcher_id", "season"])
    final = yearly.groupby("pitcher_id", sort=False).tail(1).copy()
    first_year = yearly.groupby("pitcher_id")["season"].min().rename("main_first_year")
    last_year = yearly.groupby("pitcher_id")["season"].max().rename("main_last_year")
    final = final.merge(first_year, on="pitcher_id").merge(last_year, on="pitcher_id")
    return final.reset_index(drop=True), yearly.reset_index(drop=True)


def _trackman_signatures(
    history: pd.DataFrame,
    cutoff_year: int,
    *,
    pre_sliced: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = history if pre_sliced else _slice_through_cutoff(history, cutoff_year)
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    counts = (
        data.groupby(
            ["pitcher_trackman_id", "season", "pitch_type_group"], observed=True
        )
        .size()
        .unstack(fill_value=0)
    )
    for group in PITCH_GROUPS:
        if group not in counts.columns:
            counts[group] = 0
    counts = counts[list(PITCH_GROUPS)].sort_index()
    cumulative = counts.groupby(level=0).cumsum().reset_index()
    cumulative["tm_n"] = cumulative[list(PITCH_GROUPS)].sum(axis=1)
    for group in PITCH_GROUPS[:3]:
        cumulative[f"tm_{group}_rate"] = cumulative[group] / cumulative["tm_n"].clip(
            lower=1
        )
    meta = (
        data.groupby(["pitcher_trackman_id", "season"], observed=True)
        .agg(
            tm_hand=("pitcher_hand", _mode_or_first),
            tm_team=("pitcher_team", _mode_or_first),
        )
        .reset_index()
    )
    yearly = cumulative.merge(meta, on=["pitcher_trackman_id", "season"], how="left")
    yearly = yearly.sort_values(["pitcher_trackman_id", "season"])
    final = yearly.groupby("pitcher_trackman_id", sort=False).tail(1).copy()
    first_year = yearly.groupby("pitcher_trackman_id")["season"].min().rename(
        "tm_first_year"
    )
    last_year = yearly.groupby("pitcher_trackman_id")["season"].max().rename(
        "tm_last_year"
    )
    final = final.merge(first_year, on="pitcher_trackman_id").merge(
        last_year, on="pitcher_trackman_id"
    )
    return final.reset_index(drop=True), yearly.reset_index(drop=True)


def _signature_cost(main: pd.DataFrame, trackman: pd.DataFrame) -> np.ndarray:
    main_values = np.column_stack(
        [
            np.log1p(main["main_n"].to_numpy(dtype="float64")),
            *[
                main[column].fillna(0.0).to_numpy(dtype="float64")
                for column in MAIN_RATE_COLUMNS
            ],
        ]
    )
    trackman_values = np.column_stack(
        [
            np.log1p(trackman["tm_n"].to_numpy(dtype="float64")),
            trackman["tm_fastball_rate"].to_numpy(dtype="float64"),
            trackman["tm_breaking_rate"].to_numpy(dtype="float64"),
            trackman["tm_offspeed_rate"].to_numpy(dtype="float64"),
        ]
    )
    cost = np.square(
        (main_values[:, None, :] - trackman_values[None, :, :])
        / np.array([0.45, 0.06, 0.06, 0.05], dtype="float64")
    ).sum(axis=2)
    hand_mismatch = (
        main["main_hand"].to_numpy()[:, None] != trackman["tm_hand"].to_numpy()[None, :]
    )
    cost += hand_mismatch.astype("float64") * 25.0
    first_gap = np.abs(
        main["main_first_year"].to_numpy(dtype="float64")[:, None]
        - trackman["tm_first_year"].to_numpy(dtype="float64")[None, :]
    )
    last_gap = np.abs(
        main["main_last_year"].to_numpy(dtype="float64")[:, None]
        - trackman["tm_last_year"].to_numpy(dtype="float64")[None, :]
    )
    return cost + 0.35 * first_gap + 0.15 * last_gap


def _nearest_summary(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cost.shape[1] == 1:
        return (
            np.zeros(cost.shape[0], dtype="int64"),
            cost[:, 0],
            np.full(cost.shape[0], np.inf, dtype="float64"),
        )
    partition = np.argpartition(cost, kth=1, axis=1)[:, :2]
    values = np.take_along_axis(cost, partition, axis=1)
    order = np.argsort(values, axis=1)
    candidates = np.take_along_axis(partition, order, axis=1)
    distances = np.take_along_axis(values, order, axis=1)
    return candidates[:, 0], distances[:, 0], distances[:, 1] - distances[:, 0]


def _infer_team_mapping(
    main_yearly: pd.DataFrame,
    tm_yearly: pd.DataFrame,
    main_final: pd.DataFrame,
    tm_final: pd.DataFrame,
    nearest: np.ndarray,
    distance: np.ndarray,
    margin: np.ndarray,
) -> pd.DataFrame:
    seed = main_final[["pitcher_id", "main_n"]].copy()
    seed["pitcher_trackman_id"] = tm_final.iloc[nearest][
        "pitcher_trackman_id"
    ].to_numpy()
    seed["distance"] = distance
    seed["margin"] = margin
    seed = seed[
        (seed["main_n"] >= 100)
        & (seed["distance"] < 1.2)
        & (seed["margin"] > 0.08)
    ]
    if seed.empty:
        return pd.DataFrame(
            columns=["season", "pitcher_team_id", "tm_team", "votes", "share"]
        )
    paired = main_yearly.merge(
        seed[["pitcher_id", "pitcher_trackman_id"]], on="pitcher_id", how="inner"
    ).merge(
        tm_yearly[["pitcher_trackman_id", "season", "tm_team"]],
        on=["pitcher_trackman_id", "season"],
        how="inner",
    )
    votes = (
        paired.groupby(["season", "pitcher_team_id", "tm_team"], observed=True)
        .size()
        .rename("votes")
        .reset_index()
    )
    votes["share"] = votes["votes"] / votes.groupby(
        ["season", "pitcher_team_id"]
    )["votes"].transform("sum")
    votes = votes.sort_values(
        ["season", "pitcher_team_id", "votes"], ascending=[True, True, False]
    )
    return votes.groupby(["season", "pitcher_team_id"], sort=False).head(1).reset_index(
        drop=True
    )


def _add_team_cost(
    cost: np.ndarray,
    main_final: pd.DataFrame,
    tm_final: pd.DataFrame,
    team_mapping: pd.DataFrame,
) -> np.ndarray:
    if team_mapping.empty:
        return cost
    mapped = team_mapping.set_index(["season", "pitcher_team_id"])["tm_team"]
    expected = [
        mapped.get((row.main_last_year, row.pitcher_team_id), None)
        for row in main_final.itertuples(index=False)
    ]
    mismatch = (
        np.asarray([value is not None for value in expected], dtype=bool)[:, None]
        & (np.asarray(expected, dtype=object)[:, None] != tm_final["tm_team"].to_numpy(dtype=object)[None, :])
    )
    return cost + mismatch.astype("float64") * 4.0


def build_pitcher_mapping(
    train: pd.DataFrame,
    history: pd.DataFrame,
    cutoff_year: int,
    *,
    pre_sliced: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match main and Trackman pitchers using only rows through ``cutoff_year``.

    SciPy is deliberately imported here, rather than at module import time, so
    safety checks can load the replay package without the optional dependency.
    """

    linear_sum_assignment = _require_linear_sum_assignment()

    if not pre_sliced:
        train = _slice_through_cutoff(train, cutoff_year)
        history = _slice_through_cutoff(history, cutoff_year)
    main_final, main_yearly = _main_signatures(train, cutoff_year, pre_sliced=True)
    tm_final, tm_yearly = _trackman_signatures(history, cutoff_year, pre_sliced=True)
    if main_final.empty or tm_final.empty:
        return _empty_mapping(), pd.DataFrame(
            columns=["season", "pitcher_team_id", "tm_team", "votes", "share"]
        )
    base_cost = _signature_cost(main_final, tm_final)
    nearest, distance, margin = _nearest_summary(base_cost)
    team_mapping = _infer_team_mapping(
        main_yearly, tm_yearly, main_final, tm_final, nearest, distance, margin
    )
    cost = _add_team_cost(base_cost, main_final, tm_final, team_mapping)
    unmatched_cost = 12.0
    augmented = np.concatenate(
        [
            cost,
            np.full((len(main_final), len(main_final)), unmatched_cost, dtype="float64"),
        ],
        axis=1,
    )
    row_index, column_index = linear_sum_assignment(augmented)
    assigned = np.full(len(main_final), -1, dtype="int64")
    assigned[row_index] = column_index
    _, distance2, margin2 = _nearest_summary(cost)
    rows: list[dict[str, object]] = []
    for index, main_row in main_final.iterrows():
        selected = int(assigned[index])
        is_real = selected < len(tm_final)
        selected_cost = float(cost[index, selected]) if is_real else unmatched_cost
        main_n = int(main_row["main_n"])
        rows.append(
            {
                "pitcher_id": int(main_row["pitcher_id"]),
                "pitcher_trackman_id": (
                    int(tm_final.iloc[selected]["pitcher_trackman_id"])
                    if is_real
                    else np.nan
                ),
                "main_n": main_n,
                "tm_match_cost": selected_cost,
                "tm_match_margin": float(margin2[index]),
                "tm_match_confidence": float(
                    np.exp(-min(selected_cost, 30.0) / 4.0)
                    * min(1.0, np.log1p(main_n) / 7.0)
                ),
                "tm_match_accepted": int(
                    is_real
                    and main_n >= 50
                    and selected_cost < unmatched_cost
                    and (margin2[index] > 0.04 or distance2[index] < 0.15)
                ),
                "main_last_year": int(main_row["main_last_year"]),
            }
        )
    return pd.DataFrame(rows), team_mapping


def _flatten_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    output = frame.copy()
    output.columns = [
        f"{prefix}_{column}_{stat}" if stat else f"{prefix}_{column}"
        for column, stat in output.columns.to_flat_index()
    ]
    return output


def aggregate_trackman_features(
    history: pd.DataFrame,
    cutoff_year: int,
    *,
    pre_sliced: bool = False,
) -> pd.DataFrame:
    """Aggregate pitcher Trackman measurements after pre-slicing by cutoff."""

    data = history if pre_sliced else _slice_through_cutoff(history, cutoff_year)
    if data.empty:
        return pd.DataFrame(columns=["pitcher_trackman_id", *_AGGREGATE_COLUMNS])
    data["speed_loss"] = data["rel_speed"] - data["zone_speed"]
    data["break_magnitude"] = np.sqrt(
        np.square(data["induced_vert_break"]) + np.square(data["horz_break"])
    )
    numeric = [*PHYSICAL_COLUMNS, "speed_loss", "break_magnitude"]
    career = _flatten_columns(
        data.groupby("pitcher_trackman_id", observed=True)[numeric].agg(
            ["mean", "std", "median"]
        ),
        "tm_career",
    ).reset_index()
    count = (
        data.groupby("pitcher_trackman_id", observed=True)
        .size()
        .rename("tm_history_n")
        .reset_index()
    )
    recent = data.loc[data["season"].eq(cutoff_year)]
    recent_agg = _flatten_columns(
        recent.groupby("pitcher_trackman_id", observed=True)[numeric].agg(
            ["mean", "std"]
        ),
        "tm_recent",
    ).reset_index()
    by_group = data.groupby(
        ["pitcher_trackman_id", "pitch_type_group"], observed=True
    )[list(_GROUP_METRICS)].mean().unstack("pitch_type_group")
    by_group.columns = [
        f"tm_{group}_{metric}_mean" for metric, group in by_group.columns
    ]
    by_group = by_group.reset_index()
    usage = data.groupby(
        ["pitcher_trackman_id", "pitch_type_group"], observed=True
    ).size().unstack(fill_value=0)
    for group in PITCH_GROUPS:
        if group not in usage.columns:
            usage[group] = 0
    usage = usage[list(PITCH_GROUPS)]
    usage = usage.div(usage.sum(axis=1).clip(lower=1), axis=0)
    usage.columns = [f"tm_history_{group}_rate" for group in usage.columns]
    usage = usage.reset_index()
    features = career.merge(count, on="pitcher_trackman_id", how="outer")
    features = features.merge(recent_agg, on="pitcher_trackman_id", how="left")
    features = features.merge(by_group, on="pitcher_trackman_id", how="left")
    features = features.merge(usage, on="pitcher_trackman_id", how="left")
    # The source aggregates can omit a pitch group entirely.  The R9 model
    # views still require a stable lookup schema across adjacent cutoffs.
    features = features.reindex(columns=["pitcher_trackman_id", *_AGGREGATE_COLUMNS])
    for metric in ("rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension"):
        features[f"tm_trend_{metric}"] = (
            features[f"tm_recent_{metric}_mean"] - features[f"tm_career_{metric}_mean"]
        )
    features["tm_fastball_breaking_speed_gap"] = (
        features["tm_fastball_rel_speed_mean"]
        - features["tm_breaking_rel_speed_mean"]
    )
    features["tm_fastball_offspeed_speed_gap"] = (
        features["tm_fastball_rel_speed_mean"]
        - features["tm_offspeed_rel_speed_mean"]
    )
    return features.reindex(columns=["pitcher_trackman_id", *_AGGREGATE_COLUMNS])


def normalize_pitcher_lookup(lookup: pd.DataFrame) -> pd.DataFrame:
    """Return the fixed Trackman schema expected by every final R9 view."""

    if "pitcher_id" not in lookup.columns:
        raise ValueError("trackman lookup must contain pitcher_id")
    output = lookup.copy()
    for column in PITCHER_LOOKUP_COLUMNS:
        if column not in output:
            output[column] = np.nan
    return output.loc[:, list(PITCHER_LOOKUP_COLUMNS)]


def _slice_through_cutoff(frame: pd.DataFrame, cutoff_year: int) -> pd.DataFrame:
    """Copy the legal temporal prefix once before Trackman transformations."""

    return frame.loc[frame["season"].le(cutoff_year)].copy()


def build_trackman_lookup(
    train: pd.DataFrame, history: pd.DataFrame, cutoff_year: int
) -> TrackmanBuildResult:
    """Build the fixed-schema pitcher lookup with defensive double pre-slicing."""

    _require_linear_sum_assignment()
    train_limited = _slice_through_cutoff(train, cutoff_year)
    history_limited = _slice_through_cutoff(history, cutoff_year)
    mapping, team_mapping = build_pitcher_mapping(
        train_limited,
        history_limited,
        cutoff_year,
        pre_sliced=True,
    )
    aggregates = aggregate_trackman_features(
        history_limited,
        cutoff_year,
        pre_sliced=True,
    )
    accepted = mapping.loc[mapping["tm_match_accepted"].eq(1)].copy()
    lookup = accepted.merge(
        aggregates,
        on="pitcher_trackman_id",
        how="left",
        validate="many_to_one",
    ).drop(columns=["pitcher_trackman_id", "main_n", "main_last_year"])
    return TrackmanBuildResult(
        cutoff_year=cutoff_year,
        lookup=normalize_pitcher_lookup(lookup),
        lookup_schema=PITCHER_LOOKUP_COLUMNS,
        mapping=mapping,
        team_mapping=team_mapping,
    )
