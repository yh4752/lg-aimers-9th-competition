"""Training-only, R/F-aware matching of official rows to current TrackMan pitches."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterator, Mapping

import numpy as np
import pandas as pd

from experiments.independent_dl.feature_sources.trackman import build_pitcher_mapping
from experiments.temporal_portfolio.lupi_matching import EntityMaps
from experiments.temporal_portfolio.trackman_batter import fit_batter_trackman


class PrivilegedMatchingError(ValueError):
    pass


MATCH_COLUMNS = (
    "row_id", "trackman_id", "lupi_match_accepted", "matched_game_type",
    "lupi_match_coverage", "lupi_match_mean_cost", "lupi_match_candidate_margin",
    "lupi_match_exact_token_agreement", "lupi_match_exact_token_evidence",
)
_MAIN_REQUIRED = (
    "row_id", "season", "game_month", "game_dayofweek", "game_type", "pitcher_team_id",
    "batter_team_id", "inning", "top_bottom", "balls_before", "strikes_before", "outs_before",
    "pitcher_id", "batter_id", "control_success",
)
_HISTORY_REQUIRED = (
    "trackman_id", "trackman_game_id", "pitch_no", "season", "game_month", "game_dayofweek",
    "pitcher_team", "batter_team", "inning", "top_bottom", "balls_before", "strikes_before",
    "outs_before", "pitcher_trackman_id", "batter_trackman_id",
)
_TOKEN = (
    "inning", "top_bottom", "balls_before", "strikes_before", "outs_before",
    "pitcher_trackman_id", "batter_trackman_id",
)
_BOUNDARY = ("season", "game_month", "game_dayofweek", "game_type", "pitcher_team_id", "batter_team_id")


@dataclass(frozen=True)
class _Alignment:
    mean_cost: float
    coverage: float
    exact_agreement: float
    exact_evidence: int
    pairs: tuple[tuple[int, int], ...]


def _require(frame: object, columns: tuple[str, ...], label: str) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame or not frame.columns.is_unique:
        raise PrivilegedMatchingError(f"{label} must be a DataFrame with unique columns")
    missing = [name for name in columns if name not in frame]
    if missing:
        raise PrivilegedMatchingError(f"{label} schema is missing {missing}")
    return frame.loc[:, columns].copy(deep=True)


def _integer(value: object, label: str, low: int, high: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Integral, Real)):
        raise PrivilegedMatchingError(f"{label} must be an integer")
    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise PrivilegedMatchingError(f"{label} must be an integer")
    result = int(numeric)
    if result < low or (high is not None and result > high):
        raise PrivilegedMatchingError(f"{label} is out of range")
    return result


def _canonical_main(frame: pd.DataFrame, cutoff_year: int) -> pd.DataFrame:
    result = _require(frame, _MAIN_REQUIRED, "main").reset_index(drop=True)
    if result.empty or result["row_id"].isna().any() or not result["row_id"].is_unique:
        raise PrivilegedMatchingError("main row_id must be nonempty and unique")
    ranges = {
        "season": (1000, cutoff_year), "game_month": (1, 12), "game_dayofweek": (0, 6),
        "inning": (1, None), "balls_before": (0, 3), "strikes_before": (0, 2),
        "outs_before": (0, 2), "control_success": (0, 1),
    }
    for column, (low, high) in ranges.items():
        result[column] = [_integer(value, column, low, high) for value in result[column]]
    if not result["game_type"].isin(["R", "F"]).all():
        raise PrivilegedMatchingError("main game_type must contain only R/F")
    if not result["top_bottom"].isin(["T", "B"]).all():
        raise PrivilegedMatchingError("main top_bottom must contain only T/B")
    return result


def _canonical_history(frame: pd.DataFrame, cutoff_year: int) -> pd.DataFrame:
    result = _require(frame, _HISTORY_REQUIRED, "history").reset_index(drop=True)
    result = result.loc[pd.to_numeric(result["season"], errors="coerce").le(cutoff_year)].copy()
    ranges = (
        ("season", 1000, cutoff_year), ("game_month", 1, 12), ("game_dayofweek", 0, 6),
        ("pitch_no", 1, None), ("inning", 1, None), ("balls_before", 0, 3),
        ("strikes_before", 0, 2), ("outs_before", 0, 2),
    )
    valid = pd.Series(True, index=result.index)
    numeric: dict[str, pd.Series] = {}
    for column, low, high in ranges:
        values = pd.to_numeric(result[column], errors="coerce")
        keep = values.notna() & np.isfinite(values) & values.eq(np.floor(values)) & values.ge(low)
        if high is not None:
            keep &= values.le(high)
        valid &= keep
        numeric[column] = values
    result = result.loc[valid].copy()
    for column, _, _ in ranges:
        result[column] = numeric[column].loc[valid].astype("int64").to_numpy()
    if result["trackman_id"].isna().any() or not result["trackman_id"].is_unique:
        raise PrivilegedMatchingError("history trackman_id must be unique")
    mapping = {"Top": "T", "Bottom": "B", "T": "T", "B": "B"}
    result["top_bottom"] = result["top_bottom"].map(mapping)
    if result["top_bottom"].isna().any():
        raise PrivilegedMatchingError("history top_bottom differs")
    result["matched_game_type"] = np.where(
        result["pitcher_team"].astype(str).str.startswith("MIN_")
        | result["batter_team"].astype(str).str.startswith("MIN_"),
        "F", "R",
    )
    return result.reset_index(drop=True)


def _fit_maps(main: pd.DataFrame, history: pd.DataFrame, cutoff_year: int) -> EntityMaps:
    pitcher, _ = build_pitcher_mapping(main, history, cutoff_year)
    batter = fit_batter_trackman(main, history, cutoff_year=cutoff_year).mapping
    pitcher = pitcher.loc[pitcher["tm_match_accepted"].eq(1), ["pitcher_id", "pitcher_trackman_id"]]
    batter = batter.loc[batter["tm_batter_match_accepted"].eq(1), ["batter_id", "batter_trackman_id"]]
    return EntityMaps.from_mappings(
        pitchers={
            _integer(main_id, "pitcher_id", 0): _integer(trackman_id, "pitcher_trackman_id", 0)
            for main_id, trackman_id in pitcher.itertuples(index=False, name=None)
        },
        batters={
            _integer(main_id, "batter_id", 0): _integer(trackman_id, "batter_trackman_id", 0)
            for main_id, trackman_id in batter.itertuples(index=False, name=None)
        },
    )


def _split_games(frame: pd.DataFrame) -> Iterator[tuple[int, pd.DataFrame]]:
    starts = frame.loc[:, _BOUNDARY].ne(frame.loc[:, _BOUNDARY].shift()).any(axis=1)
    starts |= frame["inning"].lt(frame["inning"].shift())
    starts.iloc[0] = True
    groups = starts.astype("int64").cumsum().to_numpy() - 1
    grouped = frame.assign(_pseudo_game=groups).groupby("_pseudo_game", sort=False, observed=True)
    for group, part in grouped:
        part = part.drop(columns="_pseudo_game").copy(deep=True)
        if part["top_bottom"].nunique(dropna=False) != 1:
            raise PrivilegedMatchingError("main pseudo-game has mixed top_bottom")
        yield int(group), part


def _history_games(frame: pd.DataFrame) -> list[pd.DataFrame]:
    output = []
    for _, part in frame.groupby("trackman_game_id", sort=False, observed=True):
        part = part.sort_values("pitch_no", kind="stable")
        if not part["pitch_no"].is_unique or any(part[name].nunique(dropna=False) != 1 for name in (
            "season", "game_month", "game_dayofweek", "matched_game_type",
        )):
            continue
        output.append(part.reset_index(drop=True))
    return output


def _candidate_games(main: pd.DataFrame, games: list[pd.DataFrame]) -> list[pd.DataFrame]:
    first = main.iloc[0]
    pitcher_ids = set(main["pitcher_trackman_id"].dropna())
    batter_ids = set(main["batter_trackman_id"].dropna())
    if not pitcher_ids or not batter_ids:
        return []
    innings = set(main["inning"])
    output = []
    for game in games:
        head = game.iloc[0]
        if any(head[name] != first[name] for name in ("season", "game_month", "game_dayofweek")):
            continue
        if head["matched_game_type"] != first["game_type"]:
            continue
        selected = game.loc[
            game["top_bottom"].eq(first["top_bottom"]) & game["inning"].isin(innings)
        ].copy(deep=True)
        pair_overlap = (
            selected["pitcher_trackman_id"].isin(pitcher_ids)
            & selected["batter_trackman_id"].isin(batter_ids)
        ).any()
        if pair_overlap:
            output.append(selected.reset_index(drop=True))
    return output


def _alignment(main: pd.DataFrame, history: pd.DataFrame) -> _Alignment:
    left_tokens = [tuple(row[name] for name in _TOKEN) for _, row in main.iterrows()]
    right_tokens = [tuple(row[name] for name in _TOKEN) for _, row in history.iterrows()]
    n, m = len(left_tokens), len(right_tokens)
    costs = np.zeros((n + 1, m + 1), dtype="int64")
    costs[:, 0], costs[0, :] = np.arange(n + 1), np.arange(m + 1)
    path = np.empty((n + 1, m + 1), dtype="U1")
    path[1:, 0], path[0, 1:] = "U", "L"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = costs[i - 1, j - 1] + int(left_tokens[i - 1] != right_tokens[j - 1])
            upward, lateral = costs[i - 1, j] + 1, costs[i, j - 1] + 1
            best = min(diagonal, upward, lateral)
            costs[i, j] = best
            path[i, j] = "D" if diagonal == best else "U" if upward == best else "L"
    pairs = []
    i, j = n, m
    while i or j:
        operation = path[i, j]
        if operation == "D":
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif operation == "U":
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    exact = sum(left_tokens[i] == right_tokens[j] for i, j in pairs)
    return _Alignment(float(costs[n, m]) / max(n, m), len(pairs) / n,
                      exact / len(pairs) if pairs else 0.0, exact, tuple(pairs))


def match_training_pitches(
    main: pd.DataFrame,
    history: pd.DataFrame,
    *,
    cutoff_year: int,
    entity_maps: EntityMaps | None = None,
) -> pd.DataFrame:
    """Return conservative matches. This function is forbidden in final inference."""

    if type(cutoff_year) is not int or not 1000 <= cutoff_year <= 9999:
        raise PrivilegedMatchingError("cutoff_year must be a four-digit integer")
    canonical_main = _canonical_main(main, cutoff_year)
    canonical_history = _canonical_history(history, cutoff_year)
    maps = entity_maps or _fit_maps(main, history, cutoff_year)
    if type(maps) is not EntityMaps:
        raise PrivilegedMatchingError("entity_maps must be sealed EntityMaps")
    mapped = canonical_main.copy(deep=True)
    mapped["pitcher_trackman_id"] = mapped["pitcher_id"].map(dict(maps.pitchers))
    mapped["batter_trackman_id"] = mapped["batter_id"].map(dict(maps.batters))
    history_games = _history_games(canonical_history)
    history_index: dict[tuple[object, ...], list[pd.DataFrame]] = {}
    for game in history_games:
        head = game.iloc[0]
        key = tuple(head[name] for name in ("season", "game_month", "game_dayofweek", "matched_game_type"))
        history_index.setdefault(key, []).append(game)
    rows: list[dict[str, object]] = []
    for pseudo_id, game in _split_games(mapped):
        first = game.iloc[0]
        key = tuple(first[name] for name in ("season", "game_month", "game_dayofweek", "game_type"))
        scored = sorted(
            ((_alignment(game, candidate), candidate)
             for candidate in _candidate_games(game, history_index.get(key, []))),
            key=lambda item: item[0].mean_cost,
        )
        best = scored[0] if scored else None
        margin = np.nan if best is None else np.inf if len(scored) == 1 else scored[1][0].mean_cost - best[0].mean_cost
        accepted_game = bool(best and best[0].coverage >= 0.85 and best[0].mean_cost <= 0.10
                             and (len(scored) == 1 or margin > 0.02))
        alignment, candidate = (None, None) if best is None else best
        paired = {} if alignment is None else dict(alignment.pairs)
        for position, row_id in enumerate(game["row_id"]):
            target_position = paired.get(position)
            exact = bool(target_position is not None and candidate is not None
                         and tuple(game.iloc[position][name] for name in _TOKEN)
                         == tuple(candidate.iloc[target_position][name] for name in _TOKEN))
            accepted = accepted_game and exact
            rows.append({
                "row_id": row_id,
                "trackman_id": candidate.iloc[target_position]["trackman_id"] if accepted else pd.NA,
                "lupi_match_accepted": int(accepted),
                "matched_game_type": str(candidate.iloc[0]["matched_game_type"]) if accepted else pd.NA,
                "lupi_match_coverage": 0.0 if alignment is None else alignment.coverage,
                "lupi_match_mean_cost": np.nan if alignment is None else alignment.mean_cost,
                "lupi_match_candidate_margin": margin,
                "lupi_match_exact_token_agreement": 0.0 if alignment is None else alignment.exact_agreement,
                "lupi_match_exact_token_evidence": 0 if alignment is None else alignment.exact_evidence,
                "_pseudo_game": pseudo_id,
            })
    accepted_rows = [row for row in rows if row["lupi_match_accepted"]]
    owners: dict[object, set[int]] = {}
    for row in accepted_rows:
        owners.setdefault(row["trackman_id"], set()).add(int(row["_pseudo_game"]))
    conflicting_games = set().union(*(owners_set for owners_set in owners.values() if len(owners_set) > 1)) if owners else set()
    for row in rows:
        if row["_pseudo_game"] in conflicting_games:
            row["trackman_id"], row["lupi_match_accepted"], row["matched_game_type"] = pd.NA, 0, pd.NA
    result = pd.DataFrame(rows).loc[:, MATCH_COLUMNS]
    result["lupi_match_accepted"] = result["lupi_match_accepted"].astype("int8")
    result["lupi_match_exact_token_evidence"] = result["lupi_match_exact_token_evidence"].astype("int64")
    if result["row_id"].tolist() != canonical_main["row_id"].tolist():
        raise PrivilegedMatchingError("matching changed row order")
    accepted_ids = result.loc[result["lupi_match_accepted"].eq(1), "trackman_id"]
    if accepted_ids.isna().any() or not accepted_ids.is_unique:
        raise PrivilegedMatchingError("accepted TrackMan IDs are not one-to-one")
    return result.reset_index(drop=True)
