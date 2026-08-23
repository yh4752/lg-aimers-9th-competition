"""Conservative, cutoff-bound current-pitch TrackMan matching for LUPI."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np
import pandas as pd


class LupiMatchingError(ValueError):
    """Raised when LUPI matching inputs or results are not safely auditable."""


MAIN_TOKEN = (
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_trackman_id",
    "batter_trackman_id",
)
TM_TOKEN = MAIN_TOKEN
LUPI_MATCH_COLUMNS = (
    "row_id",
    "trackman_id",
    "lupi_match_accepted",
    "lupi_match_coverage",
    "lupi_match_mean_cost",
    "lupi_match_candidate_margin",
    "lupi_match_exact_token_agreement",
    "lupi_match_exact_token_evidence",
)

_MAIN_REQUIRED = (
    "row_id",
    "season",
    "game_month",
    "game_dayofweek",
    "pitcher_team_id",
    "batter_team_id",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_id",
    "batter_id",
    "control_success",
)
_HISTORY_REQUIRED = (
    "trackman_id",
    "trackman_game_id",
    "pitch_no",
    "season",
    "game_month",
    "game_dayofweek",
    "pitcher_team",
    "batter_team",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_trackman_id",
    "batter_trackman_id",
)
_MAIN_BOUNDARY = (
    "season",
    "game_month",
    "game_dayofweek",
    "pitcher_team_id",
    "batter_team_id",
)
_MIN_COVERAGE = 0.85
_MAX_MEAN_COST = 0.10
_MIN_CANDIDATE_MARGIN = 0.02


@dataclass(frozen=True, init=False)
class EntityMaps:
    """Immutable, validated one-to-one MAIN-to-TrackMan entity maps."""

    _pitcher_items: tuple[tuple[object, object], ...] = field(repr=False)
    _batter_items: tuple[tuple[object, object], ...] = field(repr=False)
    _team_items: tuple[tuple[object, object], ...] = field(repr=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("EntityMaps instances must be created with from_mappings()")

    @classmethod
    def from_mappings(
        cls,
        *,
        pitchers: Mapping[object, object],
        batters: Mapping[object, object],
        teams: Mapping[object, object] | None = None,
    ) -> EntityMaps:
        state = object.__new__(cls)
        object.__setattr__(state, "_pitcher_items", _seal_mapping(pitchers, "pitcher"))
        object.__setattr__(state, "_batter_items", _seal_mapping(batters, "batter"))
        object.__setattr__(
            state,
            "_team_items",
            _seal_mapping({} if teams is None else teams, "team"),
        )
        _validated_maps(state)
        return state

    @property
    def pitchers(self) -> Mapping[object, object]:
        return MappingProxyType(dict(_validated_maps(self)[0]))

    @property
    def batters(self) -> Mapping[object, object]:
        return MappingProxyType(dict(_validated_maps(self)[1]))

    @property
    def teams(self) -> Mapping[object, object]:
        return MappingProxyType(dict(_validated_maps(self)[2]))


@dataclass(frozen=True)
class _Alignment:
    mean_cost: float
    coverage: float
    exact_agreement: float
    exact_evidence: int
    pairs: tuple[tuple[int, int], ...]


def fit_lupi_matches(
    main: pd.DataFrame,
    history: pd.DataFrame,
    *,
    cutoff_year: int,
    id_maps: EntityMaps,
) -> pd.DataFrame:
    """Match labeled training rows using only TrackMan history through cutoff."""

    _validate_cutoff(cutoff_year)
    _require_frame(main, _MAIN_REQUIRED, "main")
    _require_frame(history, _HISTORY_REQUIRED, "history")
    main_years = _validated_integer_series(main["season"], "main season", 1000, 9999)
    history_years = _validated_integer_series(
        history["season"], "history season", 1000, 9999
    )
    if any(year > cutoff_year for year in main_years):
        raise LupiMatchingError("main contains seasons beyond cutoff")
    history_prefix = history.loc[
        np.asarray(history_years, dtype="int64") <= cutoff_year
    ].copy(deep=True)
    return _match_validated(main, history_prefix, id_maps=id_maps)


def match_current_pitch_rows(
    main: pd.DataFrame,
    history: pd.DataFrame,
    *,
    id_maps: EntityMaps,
) -> pd.DataFrame:
    """Match labeled MAIN rows to current-pitch TrackMan IDs in source order."""

    return _match_validated(main, history, id_maps=id_maps)


def _match_validated(
    main: pd.DataFrame, history: pd.DataFrame, *, id_maps: EntityMaps
) -> pd.DataFrame:
    maps = _validated_maps(id_maps)
    canonical_main = _canonical_main(main)
    canonical_history = _canonical_history(history)
    mapped_main = _attach_entity_maps(canonical_main, maps)
    history_games = _stable_history_games(canonical_history)
    rows: list[dict[str, object]] = []
    for pseudo_id, pseudo_game in _split_pseudo_games(mapped_main):
        candidates = _candidate_games(pseudo_game, history_games)
        scored = [
            (_sequence_alignment(pseudo_game, candidate), candidate)
            for candidate in candidates
        ]
        scored.sort(key=lambda item: item[0].mean_cost)
        best = scored[0] if scored else None
        margin = (
            float("nan")
            if best is None
            else float("inf")
            if len(scored) == 1
            else scored[1][0].mean_cost - best[0].mean_cost
        )
        accepted = bool(
            best is not None
            and best[0].coverage >= _MIN_COVERAGE
            and best[0].mean_cost <= _MAX_MEAN_COST
            and (len(scored) == 1 or margin > _MIN_CANDIDATE_MARGIN)
        )
        rows.extend(
            _alignment_rows(
                pseudo_game,
                pseudo_id=pseudo_id,
                best=best,
                margin=margin,
                accepted=accepted,
            )
        )
    _reject_globally_reused_matches(rows)
    return _canonical_result(rows, expected_row_ids=canonical_main["row_id"].tolist())


def _canonical_main(frame: pd.DataFrame) -> pd.DataFrame:
    _require_frame(frame, _MAIN_REQUIRED, "main")
    if frame.empty:
        raise LupiMatchingError("main labeled training rows must not be empty")
    result = frame.loc[:, _MAIN_REQUIRED].copy(deep=True).reset_index(drop=True)
    _validate_unique_ids(result["row_id"], "main row_id")
    for column, low, high in (
        ("season", 1000, 9999),
        ("game_month", 1, 12),
        ("game_dayofweek", 0, 6),
        ("inning", 1, None),
        ("balls_before", 0, 3),
        ("strikes_before", 0, 2),
        ("outs_before", 0, 2),
        ("control_success", 0, 1),
    ):
        result[column] = _validated_integer_series(result[column], column, low, high)
    for column in (
        "pitcher_team_id",
        "batter_team_id",
        "pitcher_id",
        "batter_id",
    ):
        result[column] = [_entity_scalar(value, column) for value in result[column]]
    result["top_bottom"] = _canonical_sides(
        result["top_bottom"], mapping={"T": "T", "B": "B"}, label="main"
    )
    return result


def _canonical_history(frame: pd.DataFrame) -> pd.DataFrame:
    _require_frame(frame, _HISTORY_REQUIRED, "history")
    result = frame.loc[:, _HISTORY_REQUIRED].copy(deep=True).reset_index(drop=True)
    if result.empty:
        return result
    _validate_unique_ids(result["trackman_id"], "history trackman_id")
    for column, low, high in (
        ("season", 1000, 9999),
        ("game_month", 1, 12),
        ("game_dayofweek", 0, 6),
        ("pitch_no", 1, None),
        ("inning", 1, None),
        ("balls_before", 0, 3),
        ("strikes_before", 0, 2),
        ("outs_before", 0, 2),
    ):
        result[column] = _validated_integer_series(result[column], column, low, high)
    for column in (
        "trackman_game_id",
        "pitcher_team",
        "batter_team",
        "pitcher_trackman_id",
        "batter_trackman_id",
    ):
        result[column] = [_entity_scalar(value, column) for value in result[column]]
    result["top_bottom"] = _canonical_sides(
        result["top_bottom"],
        mapping={"Top": "T", "Bottom": "B"},
        label="history",
    )
    return result


def _attach_entity_maps(
    main: pd.DataFrame,
    maps: tuple[
        tuple[tuple[object, object], ...],
        tuple[tuple[object, object], ...],
        tuple[tuple[object, object], ...],
    ],
) -> pd.DataFrame:
    result = main.copy(deep=True)
    pitchers, batters, teams = (dict(items) for items in maps)
    result["pitcher_trackman_id"] = result["pitcher_id"].map(pitchers)
    result["batter_trackman_id"] = result["batter_id"].map(batters)
    if teams:
        result["pitcher_team_trackman"] = result["pitcher_team_id"].map(teams)
        result["batter_team_trackman"] = result["batter_team_id"].map(teams)
    else:
        result["pitcher_team_trackman"] = result["pitcher_team_id"]
        result["batter_team_trackman"] = result["batter_team_id"]
    return result


def _split_pseudo_games(frame: pd.DataFrame) -> list[tuple[int, pd.DataFrame]]:
    boundaries = [True]
    for position in range(1, len(frame)):
        previous = frame.iloc[position - 1]
        current = frame.iloc[position]
        metadata_changed = any(
            previous[column] != current[column] for column in _MAIN_BOUNDARY
        )
        boundaries.append(metadata_changed or current["inning"] < previous["inning"])
    group_ids = np.cumsum(np.asarray(boundaries, dtype="int64")) - 1
    games: list[tuple[int, pd.DataFrame]] = []
    for pseudo_id in range(int(group_ids[-1]) + 1):
        game = frame.loc[group_ids == pseudo_id].copy(deep=True)
        if game["top_bottom"].nunique(dropna=False) != 1:
            raise LupiMatchingError("main source order is ambiguous within a pseudo-game")
        games.append((pseudo_id, game))
    return games


def _stable_history_games(frame: pd.DataFrame) -> list[pd.DataFrame]:
    if frame.empty:
        return []
    game_order = tuple(dict.fromkeys(frame["trackman_game_id"].tolist()))
    games: list[pd.DataFrame] = []
    for game_id in game_order:
        game = frame.loc[frame["trackman_game_id"].eq(game_id)].copy(deep=True)
        for column in ("season", "game_month", "game_dayofweek"):
            if game[column].nunique(dropna=False) != 1:
                raise LupiMatchingError(f"TrackMan game has ambiguous {column}")
        if not game["pitch_no"].is_unique:
            raise LupiMatchingError(
                "pitch_no must be unique within each trackman_game_id"
            )
        games.append(game.sort_values("pitch_no", kind="stable").reset_index(drop=True))
    return games


def _candidate_games(
    main_game: pd.DataFrame, history_games: list[pd.DataFrame]
) -> list[pd.DataFrame]:
    first = main_game.iloc[0]
    mapped = (
        first["pitcher_team_trackman"],
        first["batter_team_trackman"],
    )
    if any(_is_missing(value) for value in mapped):
        return []
    candidates: list[pd.DataFrame] = []
    for game in history_games:
        metadata_matches = all(
            game.iloc[0][column] == first[column]
            for column in ("season", "game_month", "game_dayofweek")
        )
        if not metadata_matches:
            continue
        selected = game.loc[
            game["pitcher_team"].eq(mapped[0])
            & game["batter_team"].eq(mapped[1])
            & game["top_bottom"].eq(first["top_bottom"])
        ].copy(deep=True)
        if not selected.empty:
            candidates.append(selected)
    return candidates


def _sequence_alignment(main: pd.DataFrame, history: pd.DataFrame) -> _Alignment:
    main_tokens = [
        tuple(row[column] for column in MAIN_TOKEN) for _, row in main.iterrows()
    ]
    tm_tokens = [
        tuple(row[column] for column in TM_TOKEN) for _, row in history.iterrows()
    ]
    n, m = len(main_tokens), len(tm_tokens)
    costs = np.zeros((n + 1, m + 1), dtype="int64")
    costs[:, 0] = np.arange(n + 1)
    costs[0, :] = np.arange(m + 1)
    path = np.empty((n + 1, m + 1), dtype="U1")
    path[1:, 0] = "U"
    path[0, 1:] = "L"
    for left in range(1, n + 1):
        for right in range(1, m + 1):
            diagonal = costs[left - 1, right - 1] + int(
                main_tokens[left - 1] != tm_tokens[right - 1]
            )
            upward = costs[left - 1, right] + 1
            lateral = costs[left, right - 1] + 1
            best = min(diagonal, upward, lateral)
            costs[left, right] = best
            path[left, right] = (
                "D" if diagonal == best else "U" if upward == best else "L"
            )
    pairs: list[tuple[int, int]] = []
    left, right = n, m
    while left or right:
        operation = path[left, right]
        if operation == "D":
            pairs.append((left - 1, right - 1))
            left -= 1
            right -= 1
        elif operation == "U":
            left -= 1
        else:
            right -= 1
    pairs.reverse()
    exact = sum(main_tokens[i] == tm_tokens[j] for i, j in pairs)
    return _Alignment(
        mean_cost=float(costs[n, m]) / float(max(n, m)),
        coverage=float(len(pairs)) / float(n),
        exact_agreement=float(exact) / float(len(pairs)) if pairs else 0.0,
        exact_evidence=int(exact),
        pairs=tuple(pairs),
    )


def _alignment_rows(
    main: pd.DataFrame,
    *,
    pseudo_id: int,
    best: tuple[_Alignment, pd.DataFrame] | None,
    margin: float,
    accepted: bool,
) -> list[dict[str, object]]:
    alignment = None if best is None else best[0]
    candidate = None if best is None else best[1]
    paired = {} if alignment is None else dict(alignment.pairs)
    rows: list[dict[str, object]] = []
    for position, row_id in enumerate(main["row_id"].tolist()):
        history_position = paired.get(position)
        exact_token = bool(
            history_position is not None
            and candidate is not None
            and tuple(main.iloc[position][column] for column in MAIN_TOKEN)
            == tuple(candidate.iloc[history_position][column] for column in TM_TOKEN)
        )
        row_accepted = bool(accepted and exact_token)
        rows.append(
            {
                "row_id": row_id,
                "trackman_id": (
                    candidate.iloc[history_position]["trackman_id"]
                    if row_accepted and candidate is not None
                    else pd.NA
                ),
                "lupi_match_accepted": int(row_accepted),
                "lupi_match_coverage": 0.0 if alignment is None else alignment.coverage,
                "lupi_match_mean_cost": (
                    float("nan") if alignment is None else alignment.mean_cost
                ),
                "lupi_match_candidate_margin": margin,
                "lupi_match_exact_token_agreement": (
                    0.0 if alignment is None else alignment.exact_agreement
                ),
                "lupi_match_exact_token_evidence": (
                    0 if alignment is None else alignment.exact_evidence
                ),
                "_pseudo_game": pseudo_id,
            }
        )
    return rows


def _reject_globally_reused_matches(rows: list[dict[str, object]]) -> None:
    accepted = [row for row in rows if row["lupi_match_accepted"] == 1]
    owners: dict[object, set[int]] = {}
    for row in accepted:
        owners.setdefault(row["trackman_id"], set()).add(int(row["_pseudo_game"]))
    conflicts = (
        set().union(*(groups for groups in owners.values() if len(groups) > 1))
        if owners
        else set()
    )
    for row in rows:
        if row["_pseudo_game"] in conflicts:
            row["trackman_id"] = pd.NA
            row["lupi_match_accepted"] = 0


def _canonical_result(
    rows: list[dict[str, object]], *, expected_row_ids: list[object]
) -> pd.DataFrame:
    result = pd.DataFrame(rows).loc[:, LUPI_MATCH_COLUMNS].copy(deep=True)
    result["lupi_match_accepted"] = result["lupi_match_accepted"].astype("int8")
    result["lupi_match_exact_token_evidence"] = result[
        "lupi_match_exact_token_evidence"
    ].astype("int64")
    if result["row_id"].tolist() != expected_row_ids:
        raise LupiMatchingError("LUPI result changed main row order or count")
    accepted = result.loc[result["lupi_match_accepted"].eq(1), "trackman_id"]
    if accepted.isna().any() or not accepted.is_unique:
        raise LupiMatchingError("accepted TrackMan pitch IDs must be globally one-to-one")
    if result.loc[result["lupi_match_accepted"].eq(0), "trackman_id"].notna().any():
        raise LupiMatchingError("unmatched rows cannot contain fabricated TrackMan IDs")
    return result.reset_index(drop=True).copy(deep=True)


def _validated_maps(
    value: EntityMaps,
) -> tuple[
    tuple[tuple[object, object], ...],
    tuple[tuple[object, object], ...],
    tuple[tuple[object, object], ...],
]:
    if type(value) is not EntityMaps:
        raise LupiMatchingError("id_maps must be a sealed EntityMaps")
    try:
        items = (value._pitcher_items, value._batter_items, value._team_items)
    except AttributeError as error:
        raise LupiMatchingError("EntityMaps state is incomplete") from error
    labels = ("pitcher", "batter", "team")
    for item_set, label in zip(items, labels):
        if type(item_set) is not tuple or any(
            type(item) is not tuple or len(item) != 2 for item in item_set
        ):
            raise LupiMatchingError(f"{label} map state is invalid")
        if _seal_mapping(dict(item_set), label) != item_set:
            raise LupiMatchingError(f"{label} map state is not canonical")
    return items


def _seal_mapping(
    mapping: Mapping[object, object], label: str
) -> tuple[tuple[object, object], ...]:
    if not isinstance(mapping, Mapping):
        raise LupiMatchingError(f"{label} map must be a mapping")
    snapshot: dict[object, object] = {}
    try:
        for item in mapping.items():
            if type(item) is not tuple or len(item) != 2:
                raise LupiMatchingError(f"{label} map has a malformed item")
            key = _entity_scalar(item[0], f"{label} MAIN ID")
            target = _entity_scalar(item[1], f"{label} TrackMan ID")
            if key in snapshot:
                raise LupiMatchingError(f"{label} map has a duplicate MAIN ID")
            snapshot[key] = target
    except LupiMatchingError:
        raise
    except Exception as error:
        raise LupiMatchingError(f"{label} map snapshot failed") from error
    if len(set(snapshot.values())) != len(snapshot):
        raise LupiMatchingError(f"{label} map must be one-to-one")
    return tuple(sorted(snapshot.items(), key=lambda item: _scalar_sort_key(item[0])))


def _require_frame(frame: object, required: tuple[str, ...], label: str) -> None:
    if type(frame) is not pd.DataFrame:
        raise LupiMatchingError(f"{label} must be a pandas DataFrame")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        suffix = (
            " for labeled training"
            if label == "main" and "control_success" in missing
            else ""
        )
        raise LupiMatchingError(f"{label} schema is missing {missing}{suffix}")
    if not frame.columns.is_unique:
        raise LupiMatchingError(f"{label} schema has duplicate columns")


def _validate_unique_ids(values: pd.Series, label: str) -> None:
    canonical = [_entity_scalar(value, label) for value in values]
    if len(set(canonical)) != len(canonical):
        raise LupiMatchingError(f"{label} values must be unique")


def _validated_integer_series(
    values: pd.Series, label: str, minimum: int, maximum: int | None
) -> list[int]:
    output: list[int] = []
    for value in values.tolist():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Integral, Real)):
            raise LupiMatchingError(f"{label} must contain finite integer values")
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise LupiMatchingError(f"{label} must contain finite integer values")
        integer = int(numeric)
        if integer < minimum or (maximum is not None and integer > maximum):
            raise LupiMatchingError(f"{label} contains an out-of-range value")
        output.append(integer)
    return output


def _validate_cutoff(value: object) -> None:
    if type(value) is not int or not 1000 <= value <= 9999:
        raise LupiMatchingError("cutoff_year must be an exact four-digit integer")


def _canonical_sides(
    values: pd.Series, *, mapping: Mapping[str, str], label: str
) -> list[str]:
    canonical: list[str] = []
    for value in values.tolist():
        if type(value) is not str or value not in mapping:
            allowed = tuple(mapping)
            raise LupiMatchingError(
                f"{label} top_bottom must contain only {allowed}"
            )
        canonical.append(mapping[value])
    return canonical


def _entity_scalar(value: object, label: str) -> object:
    if isinstance(value, (bool, np.bool_)) or _is_missing(value):
        raise LupiMatchingError(f"{label} must contain non-null scalar IDs")
    if isinstance(value, Integral):
        return int(value)
    if type(value) is str and value:
        return value
    raise LupiMatchingError(f"{label} must contain integer or non-empty string IDs")


def _is_missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _scalar_sort_key(value: object) -> tuple[str, object]:
    return ("int", value) if type(value) is int else ("str", value)
