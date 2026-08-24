"""Cutoff-bound batter TrackMan exposure and pitcher-matchup features."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from hashlib import sha256
import math
from numbers import Integral, Real
import numpy as np
import pandas as pd
from pandas.api.types import is_scalar

from experiments.independent_dl.feature_sources.trackman import PITCH_GROUPS
from experiments.temporal_portfolio.trackman_pitcher import PitcherTrackmanState


class BatterTrackmanError(ValueError):
    """Raised when batter TrackMan features cannot be built or audited safely."""


_METRICS = ("rel_speed", "spin_rate", "induced_vert_break", "horz_break")
_MAIN_COLUMNS = (
    "season",
    "batter_id",
    "batter_hand",
    "batter_team_id",
    "asof_batter_n",
)
_HISTORY_COLUMNS = (
    "season",
    "batter_trackman_id",
    "pitch_type_group",
    "batter_hand",
    "batter_team",
    *_METRICS,
)

BATTER_MAPPING_COLUMNS = (
    "batter_id",
    "batter_trackman_id",
    "main_n",
    "tm_batter_match_cost",
    "tm_batter_match_margin",
    "tm_batter_match_confidence",
    "tm_batter_match_accepted",
    "main_last_year",
)
BATTER_EXPOSURE_COLUMNS = (
    "batter_id",
    "tm_batter_match_confidence",
    "tm_batter_match_missing",
    "tm_batter_seen_history_n",
    "tm_batter_seen_recent_n",
    *(f"tm_batter_seen_{group}_rate" for group in PITCH_GROUPS),
    *(f"tm_batter_seen_{metric}_{stat}" for metric in _METRICS for stat in ("mean", "std")),
    *(
        f"tm_batter_seen_{group}_{metric}_mean"
        for group in PITCH_GROUPS
        for metric in _METRICS
    ),
)


@dataclass(frozen=True, init=False)
class BatterTrackmanState:
    """Immutable scalar payload for a fitted batter mapping and exposure lookup."""

    cutoff_year: int
    coverage: float
    status: str
    _mapping_columns: tuple[str, ...] = field(repr=False)
    _mapping_dtypes: tuple[str, ...] = field(repr=False)
    _mapping_rows: tuple[tuple[object, ...], ...] = field(repr=False)
    _exposure_columns: tuple[str, ...] = field(repr=False)
    _exposure_dtypes: tuple[str, ...] = field(repr=False)
    _exposure_rows: tuple[tuple[object, ...], ...] = field(repr=False)
    _mapping_sha256: str = field(repr=False)
    _exposure_sha256: str = field(repr=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "BatterTrackmanState instances must be created with fit_batter_trackman()"
        )

    @classmethod
    def _from_frames(
        cls,
        *,
        cutoff_year: int,
        mapping: pd.DataFrame,
        exposure: pd.DataFrame,
    ) -> BatterTrackmanState:
        mapping = _canonical_mapping(mapping)
        exposure = _canonical_exposure(exposure, mapping=mapping)
        coverage = (
            float(mapping["tm_batter_match_accepted"].mean())
            if len(mapping)
            else 0.0
        )
        state = object.__new__(cls)
        object.__setattr__(state, "cutoff_year", cutoff_year)
        object.__setattr__(state, "coverage", coverage)
        object.__setattr__(state, "status", coverage_status(coverage))
        object.__setattr__(state, "_mapping_columns", tuple(mapping.columns))
        object.__setattr__(state, "_mapping_dtypes", tuple(map(str, mapping.dtypes)))
        object.__setattr__(state, "_mapping_rows", _freeze_rows(mapping))
        object.__setattr__(state, "_exposure_columns", tuple(exposure.columns))
        object.__setattr__(state, "_exposure_dtypes", tuple(map(str, exposure.dtypes)))
        object.__setattr__(state, "_exposure_rows", _freeze_rows(exposure))
        object.__setattr__(
            state,
            "_mapping_sha256",
            _frame_sha256(mapping, cutoff_year=cutoff_year, identity="mapping"),
        )
        object.__setattr__(
            state,
            "_exposure_sha256",
            _frame_sha256(exposure, cutoff_year=cutoff_year, identity="exposure"),
        )
        _validate_state(state)
        return state

    @property
    def mapping(self) -> pd.DataFrame:
        _validate_state(self)
        return _thaw_frame(
            self._mapping_columns, self._mapping_dtypes, self._mapping_rows
        ).copy(deep=True)

    @property
    def exposure(self) -> pd.DataFrame:
        _validate_state(self)
        return _thaw_frame(
            self._exposure_columns, self._exposure_dtypes, self._exposure_rows
        ).copy(deep=True)

    @property
    def mapping_sha256(self) -> str:
        _validate_state(self)
        return self._mapping_sha256

    @property
    def exposure_sha256(self) -> str:
        _validate_state(self)
        return self._exposure_sha256


def coverage_status(coverage: float) -> str:
    """Return the explicit experiment gate associated with mapping coverage."""

    if isinstance(coverage, (bool, np.bool_)) or not isinstance(
        coverage, (Integral, Real, Decimal)
    ):
        raise BatterTrackmanError("coverage must be a finite value in [0, 1]")
    value = float(coverage)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise BatterTrackmanError("coverage must be a finite value in [0, 1]")
    if value < 0.30:
        return "insufficient_mapping"
    if value < 0.60:
        return "exploratory"
    return "eligible"


def fit_batter_trackman(
    train: pd.DataFrame,
    history: pd.DataFrame,
    *,
    cutoff_year: int,
) -> BatterTrackmanState:
    """Fit a one-to-one batter mapping using official rows through the cutoff."""

    _validate_cutoff(cutoff_year)
    train_prefix = _validated_prefix(train, _MAIN_COLUMNS, cutoff_year, "train")
    history_prefix = _validated_prefix(
        history, _HISTORY_COLUMNS, cutoff_year, "history"
    )
    _validate_main_values(train_prefix)
    _validate_history_values(history_prefix)
    try:
        main_final, main_yearly = _main_signatures(train_prefix)
        tm_final, tm_yearly = _trackman_signatures(history_prefix)
        mapping = _solve_assignment(main_final, main_yearly, tm_final, tm_yearly)
        exposure = _aggregate_exposure(
            history_prefix, mapping, cutoff_year=cutoff_year
        )
        return BatterTrackmanState._from_frames(
            cutoff_year=cutoff_year, mapping=mapping, exposure=exposure
        )
    except BatterTrackmanError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError, ImportError) as error:
        raise BatterTrackmanError("batter TrackMan construction failed") from error


def attach_batter_exposure(
    rows: pd.DataFrame, state: BatterTrackmanState
) -> pd.DataFrame:
    """Attach only the frozen B1 lookup, preserving row count, order, and index."""

    exposure = _validated_state_frames(state)[1]
    result, ids = _validated_join_rows(rows, "batter_id")
    feature_columns = tuple(column for column in exposure if column != "batter_id")
    _reject_collisions(result, feature_columns)
    indexed = exposure.set_index("batter_id")
    for column in feature_columns:
        result[column] = ids.map(indexed[column])
    for column in (
        "tm_batter_match_confidence",
        "tm_batter_seen_history_n",
        "tm_batter_seen_recent_n",
    ):
        result[column] = result[column].fillna(0.0)
    result["tm_batter_match_missing"] = result[
        "tm_batter_match_missing"
    ].fillna(1).astype("int8")
    return result


def build_matchup_features(
    rows: pd.DataFrame,
    *,
    pitcher_state: PitcherTrackmanState,
    batter_state: BatterTrackmanState,
) -> pd.DataFrame:
    """Build M1 from frozen pitcher/batter states and same-row handedness only."""

    _validate_state(batter_state)
    if type(pitcher_state) is not PitcherTrackmanState:
        raise BatterTrackmanError("pitcher_state must be a PitcherTrackmanState")
    if pitcher_state.cutoff_year != batter_state.cutoff_year:
        raise BatterTrackmanError("pitcher and batter TrackMan cutoffs differ")
    result, pitcher_ids = _validated_join_rows(rows, "pitcher_id")
    _, batter_ids = _validated_join_rows(rows, "batter_id")
    _validate_row_hands(result)
    try:
        pitcher_lookup = pd.concat(
            [
                pitcher_state.bundles["P0"].set_index("pitcher_id"),
                pitcher_state.bundles["P2"].set_index("pitcher_id"),
            ],
            axis=1,
        )
    except Exception as error:
        raise BatterTrackmanError("pitcher TrackMan state is invalid") from error
    pitcher_features = pd.DataFrame(index=result.index)
    for column in pitcher_lookup.columns:
        pitcher_features[column] = pitcher_ids.map(pitcher_lookup[column])
    batter_lookup = _validated_state_frames(batter_state)[1].set_index("batter_id")
    batter_features = pd.DataFrame(index=result.index)
    for column in batter_lookup.columns:
        batter_features[column] = batter_ids.map(batter_lookup[column])
    batter_features["tm_batter_match_confidence"] = batter_features[
        "tm_batter_match_confidence"
    ].fillna(0.0)
    batter_features["tm_batter_match_missing"] = batter_features[
        "tm_batter_match_missing"
    ].fillna(1)

    matchup: dict[str, pd.Series] = {}
    for group in PITCH_GROUPS:
        matchup[f"tm_matchup_{group}_rate_gap"] = (
            pitcher_features[f"tm_history_{group}_rate"]
            - batter_features[f"tm_batter_seen_{group}_rate"]
        )
        for metric in _METRICS:
            matchup[f"tm_matchup_{group}_{metric}_gap"] = (
                pitcher_features[f"tm_{group}_{metric}_mean"]
                - batter_features[f"tm_batter_seen_{group}_{metric}_mean"]
            )
    pitcher_confidence = pitcher_features["tm_match_confidence"].fillna(0.0)
    matchup["tm_matchup_mapping_confidence_min"] = pd.Series(
        np.minimum(
            pitcher_confidence.to_numpy(dtype="float64"),
            batter_features["tm_batter_match_confidence"].to_numpy(dtype="float64"),
        ),
        index=result.index,
    )
    matchup["tm_matchup_mapping_missing"] = (
        pitcher_features["tm_match_accepted"].fillna(0).ne(1)
        | batter_features["tm_batter_match_missing"].eq(1)
    ).astype("int8")
    pitcher_hand = result["pitcher_hand"].astype("int64")
    batter_hand = result["batter_hand"].astype("int64")
    for pitcher_code, pitcher_name in ((1, "left"), (2, "right")):
        for batter_code, batter_name in ((1, "left"), (2, "right")):
            matchup[f"tm_matchup_{pitcher_name}_{batter_name}"] = (
                pitcher_hand.eq(pitcher_code) & batter_hand.eq(batter_code)
            ).astype("int8")
    matchup["tm_matchup_hand_missing"] = pd.Series(
        np.zeros(len(result), dtype="int8"), index=result.index
    )
    _reject_collisions(result, tuple(matchup))
    for column, values in matchup.items():
        result[column] = values
    if not result.index.equals(rows.index) or len(result) != len(rows):
        raise BatterTrackmanError("matchup transform changed evaluation rows")
    return result


def _main_signatures(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical = train.copy(deep=True)
    canonical["season"] = canonical["season"].map(int)
    canonical["batter_id"] = canonical["batter_id"].map(int)
    canonical["asof_batter_n"] = canonical["asof_batter_n"].map(int)
    keys = ["batter_id", "season"]
    maximum = canonical.groupby(keys, sort=False)["asof_batter_n"].transform("max")
    maximum_rows = canonical.loc[canonical["asof_batter_n"].eq(maximum)]
    signature_columns = ["batter_hand", "batter_team_id", "asof_batter_n"]
    distinct = maximum_rows.groupby(keys, sort=False)[signature_columns].nunique(
        dropna=False
    )
    if distinct.gt(1).any(axis=None):
        raise BatterTrackmanError(
            "batter train has conflicting maximum asof batter signatures"
        )
    yearly = (
        maximum_rows.sort_values(
            ["batter_id", "season", "batter_hand", "batter_team_id"], kind="stable"
        )
        .drop_duplicates(keys, keep="first")
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )
    yearly["main_n"] = yearly["asof_batter_n"] + 1
    yearly["main_hand"] = yearly["batter_hand"].map({1: "Left", 2: "Right"})
    spans = yearly.groupby("batter_id", sort=True)["season"].agg(
        main_first_year="min", main_last_year="max"
    )
    final = yearly.groupby("batter_id", sort=True).tail(1).merge(
        spans, on="batter_id", how="left", validate="one_to_one"
    )
    return final.sort_values("batter_id").reset_index(drop=True), yearly


def _trackman_signatures(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical = history.copy(deep=True)
    canonical["season"] = canonical["season"].map(int)
    canonical["batter_trackman_id"] = canonical["batter_trackman_id"].map(int)
    keys = ["batter_trackman_id", "season"]
    counts = canonical.groupby(keys, sort=True).size().rename("season_n").reset_index()
    hands = canonical.groupby("batter_trackman_id", sort=True)["batter_hand"].agg(
        lambda values: _unique_or_none(values)
    )
    teams = canonical.groupby(keys, sort=True)["batter_team"].agg(
        lambda values: tuple(sorted(set(values.tolist()), key=_stable_scalar_key))
    )
    yearly = counts.merge(teams.rename("tm_teams"), on=keys, validate="one_to_one")
    yearly["tm_n"] = yearly.groupby("batter_trackman_id", sort=False)[
        "season_n"
    ].cumsum()
    yearly = yearly.merge(hands.rename("tm_hand"), on="batter_trackman_id")
    spans = yearly.groupby("batter_trackman_id", sort=True)["season"].agg(
        tm_first_year="min", tm_last_year="max"
    )
    final = yearly.groupby("batter_trackman_id", sort=True).tail(1).merge(
        spans, on="batter_trackman_id", how="left", validate="one_to_one"
    )
    return final.sort_values("batter_trackman_id").reset_index(drop=True), yearly


def _solve_assignment(
    main: pd.DataFrame,
    main_yearly: pd.DataFrame,
    tm: pd.DataFrame,
    tm_yearly: pd.DataFrame,
) -> pd.DataFrame:
    base = _base_cost(main, tm)
    base_margin = _row_margin(base)
    team_map = _infer_team_map(main, main_yearly, tm, tm_yearly, base, base_margin)
    cost = base + _team_overlap_penalty(main, main_yearly, tm, tm_yearly, team_map)
    margin = _row_margin(cost)
    assignment = _hungarian(cost)
    rows: list[dict[str, object]] = []
    for index, main_row in main.iterrows():
        selected = int(assignment[index])
        real = selected < len(tm)
        selected_cost = float(cost[index, selected]) if real else 1.5
        selected_margin = _selected_margin(cost[index], selected) if real else 0.0
        hand_equal = bool(real and main_row["main_hand"] == tm.iloc[selected]["tm_hand"])
        accepted = bool(
            real
            and hand_equal
            and selected_cost <= 1.5
            and selected_margin >= 0.15
        )
        sample_factor = min(1.0, math.log1p(float(main_row["main_n"])) / 7.0)
        confidence = (
            math.exp(-min(selected_cost, 30.0))
            * min(1.0, max(0.0, selected_margin))
            * sample_factor
            if real
            else 0.0
        )
        rows.append(
            {
                "batter_id": int(main_row["batter_id"]),
                "batter_trackman_id": (
                    float(tm.iloc[selected]["batter_trackman_id"])
                    if real
                    else np.nan
                ),
                "main_n": int(main_row["main_n"]),
                "tm_batter_match_cost": selected_cost,
                "tm_batter_match_margin": selected_margin,
                "tm_batter_match_confidence": confidence,
                "tm_batter_match_accepted": int(accepted),
                "main_last_year": int(main_row["main_last_year"]),
            }
        )
    return pd.DataFrame(rows, columns=BATTER_MAPPING_COLUMNS)


def _base_cost(main: pd.DataFrame, tm: pd.DataFrame) -> np.ndarray:
    count_gap = np.abs(
        np.log1p(main["main_n"].to_numpy(dtype="float64"))[:, None]
        - np.log1p(tm["tm_n"].to_numpy(dtype="float64"))[None, :]
    ) / 0.45
    first_gap = np.abs(
        main["main_first_year"].to_numpy(dtype="float64")[:, None]
        - tm["tm_first_year"].to_numpy(dtype="float64")[None, :]
    )
    last_gap = np.abs(
        main["main_last_year"].to_numpy(dtype="float64")[:, None]
        - tm["tm_last_year"].to_numpy(dtype="float64")[None, :]
    )
    hand_equal = (
        main["main_hand"].to_numpy(dtype=object)[:, None]
        == tm["tm_hand"].to_numpy(dtype=object)[None, :]
    )
    return count_gap + 0.15 * first_gap + 0.15 * last_gap + (~hand_equal) * 1.0e6


def _row_margin(cost: np.ndarray) -> np.ndarray:
    if cost.shape[1] == 1:
        return np.full(cost.shape[0], np.inf, dtype="float64")
    ordered = np.sort(cost, axis=1, kind="stable")
    return ordered[:, 1] - ordered[:, 0]


def _selected_margin(cost: np.ndarray, selected: int) -> float:
    alternatives = np.delete(cost, selected)
    if not len(alternatives):
        return math.inf
    return float(np.min(alternatives) - cost[selected])


def _infer_team_map(
    main: pd.DataFrame,
    main_yearly: pd.DataFrame,
    tm: pd.DataFrame,
    tm_yearly: pd.DataFrame,
    cost: np.ndarray,
    margin: np.ndarray,
) -> dict[tuple[int, object], object]:
    nearest = np.argmin(cost, axis=1)
    seeds = []
    for index, selected in enumerate(nearest):
        if cost[index, selected] <= 1.5 and margin[index] >= 0.15:
            seeds.append(
                (int(main.iloc[index]["batter_id"]), int(tm.iloc[selected]["batter_trackman_id"]))
            )
    if not seeds:
        return {}
    seed = pd.DataFrame(seeds, columns=["batter_id", "batter_trackman_id"])
    paired = main_yearly.merge(seed, on="batter_id", how="inner").merge(
        tm_yearly[["batter_trackman_id", "season", "tm_teams"]],
        on=["batter_trackman_id", "season"],
        how="inner",
    )
    votes: dict[tuple[int, object], dict[object, int]] = {}
    for row in paired.itertuples(index=False):
        key = (int(row.season), row.batter_team_id)
        counts = votes.setdefault(key, {})
        for team in row.tm_teams:
            counts[team] = counts.get(team, 0) + 1
    result: dict[tuple[int, object], object] = {}
    for key, counts in votes.items():
        maximum = max(counts.values())
        winners = [team for team, value in counts.items() if value == maximum]
        if len(winners) == 1:
            result[key] = winners[0]
    return result


def _team_overlap_penalty(
    main: pd.DataFrame,
    main_yearly: pd.DataFrame,
    tm: pd.DataFrame,
    tm_yearly: pd.DataFrame,
    team_map: dict[tuple[int, object], object],
) -> np.ndarray:
    penalty = np.zeros((len(main), len(tm)), dtype="float64")
    main_groups = {key: frame for key, frame in main_yearly.groupby("batter_id")}
    tm_groups = {key: frame for key, frame in tm_yearly.groupby("batter_trackman_id")}
    for left, main_row in main.iterrows():
        main_seasons = main_groups[main_row["batter_id"]]
        for right, tm_row in tm.iterrows():
            tm_seasons = tm_groups[tm_row["batter_trackman_id"]].set_index("season")
            comparable = 0
            agreements = 0
            for row in main_seasons.itertuples(index=False):
                expected = team_map.get((int(row.season), row.batter_team_id))
                if expected is None or int(row.season) not in tm_seasons.index:
                    continue
                comparable += 1
                teams = tm_seasons.loc[int(row.season), "tm_teams"]
                agreements += int(expected in teams)
            if comparable:
                penalty[left, right] = 0.75 * (1.0 - agreements / comparable)
    return penalty


def _hungarian(cost: np.ndarray) -> np.ndarray:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:  # pragma: no cover - environment boundary
        raise BatterTrackmanError("batter TrackMan matching requires scipy") from error
    augmented = np.concatenate(
        [cost, np.full((len(cost), len(cost)), 1.5, dtype="float64")], axis=1
    )
    row_index, column_index = linear_sum_assignment(augmented)
    assigned = np.full(len(cost), len(cost[0]) + len(cost), dtype="int64")
    assigned[row_index] = column_index
    return assigned


def _aggregate_exposure(
    history: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    cutoff_year: int,
) -> pd.DataFrame:
    accepted = mapping.loc[mapping["tm_batter_match_accepted"].eq(1)].set_index(
        "batter_id"
    )
    rows: list[dict[str, object]] = []
    for mapped in mapping.itertuples(index=False):
        row = {column: np.nan for column in BATTER_EXPOSURE_COLUMNS}
        row["batter_id"] = int(mapped.batter_id)
        row["tm_batter_match_confidence"] = float(mapped.tm_batter_match_confidence)
        row["tm_batter_match_missing"] = int(not mapped.tm_batter_match_accepted)
        row["tm_batter_seen_history_n"] = 0.0
        row["tm_batter_seen_recent_n"] = 0.0
        if mapped.batter_id in accepted.index:
            tm_id = int(accepted.loc[mapped.batter_id, "batter_trackman_id"])
            seen = history.loc[history["batter_trackman_id"].eq(tm_id)]
            row["tm_batter_seen_history_n"] = float(len(seen))
            row["tm_batter_seen_recent_n"] = float(
                seen["season"].eq(cutoff_year).sum()
            )
            counts = seen["pitch_type_group"].value_counts()
            for group in PITCH_GROUPS:
                row[f"tm_batter_seen_{group}_rate"] = float(
                    counts.get(group, 0) / len(seen)
                )
            for metric in _METRICS:
                values = seen[metric]
                row[f"tm_batter_seen_{metric}_mean"] = float(values.mean())
                row[f"tm_batter_seen_{metric}_std"] = float(values.std())
            grouped = seen.groupby("pitch_type_group", observed=True)
            for group in PITCH_GROUPS:
                group_frame = grouped.get_group(group) if group in grouped.groups else None
                for metric in _METRICS:
                    row[f"tm_batter_seen_{group}_{metric}_mean"] = (
                        float(group_frame[metric].mean())
                        if group_frame is not None
                        else np.nan
                    )
        rows.append(row)
    return pd.DataFrame(rows, columns=BATTER_EXPOSURE_COLUMNS)


def _validated_prefix(
    frame: object,
    required: tuple[str, ...],
    cutoff_year: int,
    label: str,
) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame:
        raise BatterTrackmanError(f"{label} must be an actual DataFrame")
    if frame.columns.has_duplicates:
        raise BatterTrackmanError(f"{label} has duplicate columns")
    if any(type(column) is not str for column in frame.columns):
        raise BatterTrackmanError(f"{label} column names must be strings")
    missing = [column for column in required if column not in frame]
    if missing:
        raise BatterTrackmanError(f"{label} required columns are missing: {missing}")
    _validate_integral(frame["season"], f"{label} season", minimum=1000, maximum=9999)
    prefix = frame.loc[frame["season"].le(cutoff_year), list(required)].copy(deep=True)
    if prefix.empty:
        raise BatterTrackmanError(f"{label} has no rows through cutoff")
    return prefix


def _validate_main_values(frame: pd.DataFrame) -> None:
    _validate_integral(frame["batter_id"], "batter_id", minimum=0)
    _validate_integral(frame["batter_hand"], "batter_hand", minimum=1, maximum=2)
    _validate_scalar(frame["batter_team_id"], "batter_team_id")
    _validate_integral(frame["asof_batter_n"], "asof_batter_n", minimum=0)


def _validate_history_values(frame: pd.DataFrame) -> None:
    _validate_integral(frame["batter_trackman_id"], "batter_trackman_id", minimum=0)
    _validate_scalar(frame["batter_hand"], "batter_hand")
    if not set(frame["batter_hand"]).issubset({"Left", "Right"}):
        raise BatterTrackmanError("history batter_hand must be Left or Right")
    _validate_scalar(frame["batter_team"], "batter_team")
    _validate_scalar(frame["pitch_type_group"], "pitch_type_group")
    unexpected = sorted(set(frame["pitch_type_group"]) - set(PITCH_GROUPS), key=str)
    if unexpected:
        raise BatterTrackmanError(f"pitch_type_group contains unsupported values: {unexpected}")
    for metric in _METRICS:
        _validate_numeric(frame[metric], metric, allow_nan=True)


def _validated_join_rows(
    rows: object, id_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    if type(rows) is not pd.DataFrame:
        raise BatterTrackmanError("transform rows must be an actual DataFrame")
    if rows.columns.has_duplicates or any(type(column) is not str for column in rows.columns):
        raise BatterTrackmanError("transform rows must have unique string columns")
    if id_column not in rows:
        raise BatterTrackmanError(f"transform rows are missing {id_column}")
    _validate_integral(rows[id_column], id_column, minimum=0)
    ids = rows[id_column].map(int)
    return rows.copy(deep=True), ids


def _validate_row_hands(rows: pd.DataFrame) -> None:
    for column in ("pitcher_hand", "batter_hand"):
        if column not in rows:
            raise BatterTrackmanError(f"matchup rows are missing {column}")
        _validate_integral(rows[column], column, minimum=1, maximum=2)


def _reject_collisions(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    collisions = sorted(set(frame.columns).intersection(columns))
    if collisions:
        raise BatterTrackmanError(f"feature columns already exist: {collisions}")


def _validate_cutoff(value: object) -> None:
    if type(value) is not int or not 1000 <= value <= 9999:
        raise BatterTrackmanError("cutoff_year must be an exact four-digit int")


def _validate_scalar(values: pd.Series, label: str) -> None:
    for value in values.tolist():
        try:
            valid = is_scalar(value) and not bool(pd.isna(value))
            hash(value)
        except Exception as error:
            raise BatterTrackmanError(f"{label} must contain scalar values") from error
        if not valid:
            raise BatterTrackmanError(f"{label} must contain scalar values")


def _validate_integral(
    values: pd.Series,
    label: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> None:
    for value in values.tolist():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (Integral, Real, Decimal)
        ):
            raise BatterTrackmanError(f"{label} must contain finite integers")
        numeric = float(value)
        if (
            not math.isfinite(numeric)
            or not numeric.is_integer()
            or numeric < minimum
            or (maximum is not None and numeric > maximum)
        ):
            raise BatterTrackmanError(f"{label} must contain finite integers")


def _validate_numeric(values: pd.Series, label: str, *, allow_nan: bool) -> None:
    for value in values.tolist():
        if value is None or value is pd.NA:
            if allow_nan:
                continue
            raise BatterTrackmanError(f"{label} must contain numeric values")
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (Integral, Real, Decimal)
        ):
            raise BatterTrackmanError(f"{label} must contain numeric values")
        numeric = float(value)
        if math.isnan(numeric) and allow_nan:
            continue
        if not math.isfinite(numeric):
            raise BatterTrackmanError(f"{label} must not contain infinity")


def _unique_or_none(values: pd.Series) -> object:
    unique = sorted(set(values.tolist()), key=_stable_scalar_key)
    return unique[0] if len(unique) == 1 else None


def _stable_scalar_key(value: object) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _canonical_mapping(frame: pd.DataFrame) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame or tuple(frame.columns) != BATTER_MAPPING_COLUMNS:
        raise BatterTrackmanError("batter mapping schema is invalid")
    canonical = frame.copy(deep=True)
    _validate_integral(canonical["batter_id"], "batter_id", minimum=0)
    _validate_integral(canonical["main_n"], "main_n", minimum=1)
    _validate_integral(canonical["main_last_year"], "main_last_year", minimum=1000, maximum=9999)
    _validate_integral(
        canonical["tm_batter_match_accepted"],
        "tm_batter_match_accepted",
        minimum=0,
        maximum=1,
    )
    for column in BATTER_MAPPING_COLUMNS[1:]:
        if column not in {"main_n", "tm_batter_match_accepted", "main_last_year"}:
            _validate_numeric(canonical[column], column, allow_nan=True)
    canonical["batter_id"] = canonical["batter_id"].map(int).astype("int64")
    canonical["main_n"] = canonical["main_n"].map(int).astype("int64")
    canonical["main_last_year"] = canonical["main_last_year"].map(int).astype("int64")
    canonical["tm_batter_match_accepted"] = canonical[
        "tm_batter_match_accepted"
    ].map(int).astype("int8")
    if canonical["batter_id"].duplicated().any():
        raise BatterTrackmanError("batter mapping ids must be unique")
    accepted = canonical["tm_batter_match_accepted"].eq(1)
    accepted_tm = canonical.loc[accepted, "batter_trackman_id"]
    if accepted_tm.isna().any() or accepted_tm.duplicated().any():
        raise BatterTrackmanError("accepted batter mapping must be one-to-one")
    numeric = [column for column in BATTER_MAPPING_COLUMNS if column not in {"batter_id", "main_n", "tm_batter_match_accepted", "main_last_year"}]
    canonical.loc[:, numeric] = canonical.loc[:, numeric].round(12)
    return canonical.sort_values("batter_id", kind="stable").reset_index(drop=True)


def _canonical_exposure(
    frame: pd.DataFrame, *, mapping: pd.DataFrame
) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame or tuple(frame.columns) != BATTER_EXPOSURE_COLUMNS:
        raise BatterTrackmanError("batter exposure schema is invalid")
    canonical = frame.copy(deep=True)
    _validate_integral(canonical["batter_id"], "batter_id", minimum=0)
    _validate_integral(
        canonical["tm_batter_match_missing"],
        "tm_batter_match_missing",
        minimum=0,
        maximum=1,
    )
    for column in BATTER_EXPOSURE_COLUMNS[1:]:
        if column != "tm_batter_match_missing":
            _validate_numeric(canonical[column], column, allow_nan=True)
    canonical["batter_id"] = canonical["batter_id"].map(int).astype("int64")
    canonical["tm_batter_match_missing"] = canonical[
        "tm_batter_match_missing"
    ].map(int).astype("int8")
    canonical.loc[:, list(BATTER_EXPOSURE_COLUMNS[1:])] = canonical.loc[
        :, list(BATTER_EXPOSURE_COLUMNS[1:])
    ].round(12)
    canonical = canonical.sort_values("batter_id", kind="stable").reset_index(drop=True)
    if canonical["batter_id"].tolist() != mapping["batter_id"].tolist():
        raise BatterTrackmanError("batter exposure ids differ from mapping")
    expected_missing = 1 - mapping["tm_batter_match_accepted"].to_numpy(dtype="int8")
    if not np.array_equal(canonical["tm_batter_match_missing"], expected_missing):
        raise BatterTrackmanError("batter exposure missing flags differ from mapping")
    return canonical


def _freeze_rows(frame: pd.DataFrame) -> tuple[tuple[object, ...], ...]:
    rows = tuple(tuple(row) for row in frame.itertuples(index=False, name=None))
    if any(not all(is_scalar(value) for value in row) for row in rows):
        raise BatterTrackmanError("batter TrackMan state contains non-scalar values")
    return rows


def _thaw_frame(
    columns: tuple[str, ...],
    dtypes: tuple[str, ...],
    rows: tuple[tuple[object, ...], ...],
) -> pd.DataFrame:
    try:
        frame = pd.DataFrame.from_records(rows, columns=columns)
        for column, dtype in zip(columns, dtypes, strict=True):
            frame[column] = frame[column].astype(dtype)
        return frame
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise BatterTrackmanError("batter TrackMan state payload is invalid") from error


def _frame_sha256(frame: pd.DataFrame, *, cutoff_year: int, identity: str) -> str:
    digest = sha256()
    digest.update(b"batter-trackman-v1\0")
    digest.update(str(cutoff_year).encode("ascii"))
    digest.update(b"\0" + identity.encode("ascii") + b"\0")
    digest.update(repr(tuple(frame.columns)).encode("utf-8"))
    digest.update(repr(tuple(map(str, frame.dtypes))).encode("utf-8"))
    try:
        hashed = pd.util.hash_pandas_object(
            frame, index=False, categorize=False
        ).to_numpy(dtype="uint64", copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise BatterTrackmanError("batter TrackMan identity cannot be computed") from error
    digest.update(hashed.tobytes(order="C"))
    return digest.hexdigest()


def _validated_state_frames(
    state: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_state(state)
    return (
        _thaw_frame(state._mapping_columns, state._mapping_dtypes, state._mapping_rows),
        _thaw_frame(state._exposure_columns, state._exposure_dtypes, state._exposure_rows),
    )


def _validate_state(state: object) -> None:
    if type(state) is not BatterTrackmanState:
        raise BatterTrackmanError("state type must be exactly BatterTrackmanState")
    try:
        cutoff = object.__getattribute__(state, "cutoff_year")
        coverage = object.__getattribute__(state, "coverage")
        status = object.__getattribute__(state, "status")
        mapping_columns = object.__getattribute__(state, "_mapping_columns")
        mapping_dtypes = object.__getattribute__(state, "_mapping_dtypes")
        mapping_rows = object.__getattribute__(state, "_mapping_rows")
        exposure_columns = object.__getattribute__(state, "_exposure_columns")
        exposure_dtypes = object.__getattribute__(state, "_exposure_dtypes")
        exposure_rows = object.__getattribute__(state, "_exposure_rows")
        mapping_digest = object.__getattribute__(state, "_mapping_sha256")
        exposure_digest = object.__getattribute__(state, "_exposure_sha256")
    except (AttributeError, TypeError) as error:
        raise BatterTrackmanError("batter TrackMan state is missing fields") from error
    _validate_cutoff(cutoff)
    if mapping_columns != BATTER_MAPPING_COLUMNS or exposure_columns != BATTER_EXPOSURE_COLUMNS:
        raise BatterTrackmanError("batter TrackMan state schema is invalid")
    if not all(
        type(value) is tuple
        for value in (mapping_dtypes, mapping_rows, exposure_dtypes, exposure_rows)
    ):
        raise BatterTrackmanError("batter TrackMan state payload is invalid")
    if len(mapping_dtypes) != len(mapping_columns) or len(exposure_dtypes) != len(exposure_columns):
        raise BatterTrackmanError("batter TrackMan state payload is invalid")
    stored_mapping = _thaw_frame(mapping_columns, mapping_dtypes, mapping_rows)
    mapping = _canonical_mapping(stored_mapping)
    if not stored_mapping.equals(mapping):
        raise BatterTrackmanError("batter TrackMan state mapping is not canonical")
    stored_exposure = _thaw_frame(exposure_columns, exposure_dtypes, exposure_rows)
    exposure = _canonical_exposure(stored_exposure, mapping=mapping)
    if not stored_exposure.equals(exposure):
        raise BatterTrackmanError("batter TrackMan state exposure is not canonical")
    observed_coverage = float(mapping["tm_batter_match_accepted"].mean()) if len(mapping) else 0.0
    if coverage != observed_coverage or status != coverage_status(observed_coverage):
        raise BatterTrackmanError("batter TrackMan state coverage differs")
    if not _is_sha256(mapping_digest) or not _is_sha256(exposure_digest):
        raise BatterTrackmanError("batter TrackMan state identity is invalid")
    if _frame_sha256(mapping, cutoff_year=cutoff, identity="mapping") != mapping_digest:
        raise BatterTrackmanError("batter TrackMan state mapping identity differs")
    if _frame_sha256(exposure, cutoff_year=cutoff, identity="exposure") != exposure_digest:
        raise BatterTrackmanError("batter TrackMan state exposure identity differs")


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
