"""Standalone evaluator for the accepted Tree Expert E2 CatBoost ensemble."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


EMBEDDED_METADATA = None
MISSING_CATEGORY = "__MISSING__"
TARGET = "control_success"
ROW_ID = "row_id"
RAW_CATEGORICAL = (
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_hand",
    "batter_hand",
    "top_bottom",
    "game_type",
    "base_state",
    "balls_before",
    "strikes_before",
    "outs_before",
)
CROSS_CATEGORICAL = (
    "count_state",
    "hand_matchup",
    "pitcher_batter_hand",
    "pitcher_count",
    "pitcher_base",
    "pitcher_game_type",
    "batter_pitcher_hand",
    "team_count",
    "count_base",
    "hand_count",
    "inning_bin",
    "score_bin",
    "leverage_bin",
    "inning_score",
    "leverage_score",
    "leverage_runners_count",
)
SUCCESS_RECENT = tuple(
    f"asof_pitcher_prev{games}_game_success_rate" for games in (1, 3, 5)
)
MIDDLE_RECENT = tuple(
    f"asof_pitcher_prev{games}_game_middle_rate" for games in (1, 3, 5)
)
PITCHER_RATES = (
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
)
PITCHMIX_RATES = (
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)
REQUIRED = {
    ROW_ID,
    "season",
    "inning",
    "score_diff_pitcher_team",
    "li",
    "num_runners_on",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_pitcher_pitchmix_n",
    *PITCHER_RATES,
    *PITCHMIX_RATES,
    "asof_batter_middle_rate",
    *RAW_CATEGORICAL,
    *SUCCESS_RECENT,
    *MIDDLE_RECENT,
}
MODEL_MEMBERS = {
    "frozen_state/feature_state.json",
    "frozen_state/s1_batter.csv",
    "frozen_state/s1_pitcher.csv",
    "models/catboost_seed_42.cbm",
    "models/catboost_seed_2026.cbm",
    "models/catboost_seed_3407.cbm",
}


class EvaluatorError(ValueError):
    """Raised before a changed model or invalid prediction can be published."""


class FrozenState:
    def __init__(
        self,
        *,
        valid_year: int,
        prior_rate: float,
        categorical_columns: tuple[str, ...],
        feature_columns: tuple[str, ...],
        pitcher: pd.DataFrame,
        batter: pd.DataFrame,
        state_sha256: str,
    ) -> None:
        self.valid_year = valid_year
        self.prior_rate = prior_rate
        self.categorical_columns = categorical_columns
        self.feature_columns = feature_columns
        self.pitcher = pitcher
        self.batter = batter
        self.state_sha256 = state_sha256


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    def unique(items: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in items:
            if key in output:
                raise EvaluatorError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    def finite(value: str) -> object:
        raise EvaluatorError(f"non-finite JSON value: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique,
            parse_constant=finite,
        )
    except EvaluatorError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluatorError(f"cannot read JSON: {path}") from error
    if type(value) is not dict:
        raise EvaluatorError(f"JSON must be an object: {path}")
    return value


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _cast(frame: pd.DataFrame, dtypes: object, label: str) -> pd.DataFrame:
    if type(dtypes) is not dict or set(dtypes) != set(frame.columns):
        raise EvaluatorError(f"{label} dtype metadata differs")
    output = frame.copy(deep=True)
    for column in output:
        dtype = dtypes[column]
        if not isinstance(dtype, str):
            raise EvaluatorError(f"{label} dtype differs: {column}")
        try:
            output[column] = output[column].astype(object if dtype == "object" else dtype)
        except (TypeError, ValueError) as error:
            raise EvaluatorError(f"{label} dtype differs: {column}") from error
    return output


def load_frozen_state(root: Path) -> FrozenState:
    source = Path(root)
    manifest_path = source / "feature_state.json"
    if source.is_symlink() or not source.is_dir() or not manifest_path.is_file():
        raise EvaluatorError("frozen state directory is missing or unsafe")
    manifest = _load_json(manifest_path)
    expected_keys = {
        "schema_version",
        "artifact_kind",
        "candidate_id",
        "valid_year",
        "prior_rate",
        "categorical_columns",
        "feature_columns",
        "source_hashes",
        "anchor_formula",
        "probability_clip",
        "s1",
        "trackman",
        "members",
    }
    if (
        set(manifest) != expected_keys
        or manifest["schema_version"] != 1
        or manifest["artifact_kind"] != "tree_expert_e2_frozen_state_v1"
        or manifest["candidate_id"] != "c1_anchor_residual"
        or manifest["valid_year"] != 2025 and manifest["valid_year"] != 2024
        or manifest["anchor_formula"] != "e1_anchor_v1"
        or manifest["probability_clip"] != [0.00001, 0.99999]
        or manifest["trackman"] is not None
    ):
        raise EvaluatorError("frozen state identity differs")
    members = manifest["members"]
    if type(members) is not dict or set(members) != {"s1_pitcher.csv", "s1_batter.csv"}:
        raise EvaluatorError("frozen state member set differs")
    values: dict[str, bytes] = {}
    for name in sorted(members):
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise EvaluatorError(f"frozen state member is missing: {name}")
        value = path.read_bytes()
        entry = members[name]
        if (
            type(entry) is not dict
            or set(entry) != {"sha256", "size"}
            or entry["size"] != len(value)
            or entry["sha256"] != sha256(value).hexdigest()
        ):
            raise EvaluatorError(f"frozen state member differs: {name}")
        values[name] = value
    s1 = manifest["s1"]
    if type(s1) is not dict or set(s1) != {"cutoff_year", "pitcher_dtypes", "batter_dtypes"}:
        raise EvaluatorError("frozen S1 metadata differs")
    from io import BytesIO

    pitcher = _cast(
        pd.read_csv(BytesIO(values["s1_pitcher.csv"])), s1["pitcher_dtypes"], "pitcher"
    )
    batter = _cast(
        pd.read_csv(BytesIO(values["s1_batter.csv"])), s1["batter_dtypes"], "batter"
    )
    valid_year = int(manifest["valid_year"])
    prior = float(manifest["prior_rate"])
    if int(s1["cutoff_year"]) != valid_year - 1 or not math.isfinite(prior):
        raise EvaluatorError("frozen S1 cutoff or prior differs")
    if not isinstance(manifest["categorical_columns"], list) or not isinstance(
        manifest["feature_columns"], list
    ):
        raise EvaluatorError("frozen feature schema differs")
    return FrozenState(
        valid_year=valid_year,
        prior_rate=prior,
        categorical_columns=tuple(manifest["categorical_columns"]),
        feature_columns=tuple(manifest["feature_columns"]),
        pitcher=pitcher,
        batter=batter,
        state_sha256=_sha256_file(manifest_path),
    )


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(rows[column], errors="coerce").astype("float64")
    if np.isinf(values.to_numpy()).any():
        raise EvaluatorError(f"{column} contains infinity")
    return values


def _category(rows: pd.DataFrame, column: str) -> pd.Series:
    return rows[column].astype("string").fillna(MISSING_CATEGORY).astype(object)


def _join(*parts: pd.Series) -> pd.Series:
    result = parts[0].astype("string").fillna(MISSING_CATEGORY)
    for part in parts[1:]:
        result = result.str.cat(part.astype("string").fillna(MISSING_CATEGORY), sep="|")
    return result.astype(object)


def _validate_rows(rows: object) -> pd.DataFrame:
    if type(rows) is not pd.DataFrame or rows.empty:
        raise EvaluatorError("evaluation rows must be a non-empty pandas DataFrame")
    if rows.columns.has_duplicates or any(type(column) is not str for column in rows.columns):
        raise EvaluatorError("evaluation columns must be unique strings")
    if TARGET in rows:
        raise EvaluatorError("evaluation rows contain target")
    missing = sorted(REQUIRED.difference(rows.columns))
    if missing:
        raise EvaluatorError(f"evaluation rows are missing required columns: {missing}")
    ids = rows[ROW_ID].astype("string")
    if ids.isna().any() or ids.astype(str).duplicated().any():
        raise EvaluatorError("row_id values must be non-null and unique")
    return rows.copy(deep=True)


def _season_delta(
    output: pd.DataFrame,
    *,
    current_n: pd.Series,
    current_rate: pd.Series,
    snapshot_n_column: str,
    snapshot_count_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    cumulative_n = pd.to_numeric(current_n).fillna(0).to_numpy(dtype="float64")
    cumulative_count = cumulative_n * pd.to_numeric(current_rate).fillna(0).to_numpy(
        dtype="float64"
    )
    snapshot_n = output[snapshot_n_column].fillna(0).to_numpy(dtype="float64")
    snapshot_count = output[snapshot_count_column].fillna(0).to_numpy(dtype="float64")
    delta_n = np.maximum(cumulative_n - snapshot_n, 0.0)
    return delta_n, np.clip(cumulative_count - snapshot_count, 0.0, delta_n)


def _smoothed_rate(
    count: np.ndarray, n: np.ndarray, prior: np.ndarray | float, strength: float
) -> np.ndarray:
    return ((count + np.asarray(prior) * strength) / (n + strength)).astype("float32")


def _seasonal_features(rows: pd.DataFrame, state: FrozenState) -> pd.DataFrame:
    selected_columns = (
        "season",
        "pitcher_id",
        "batter_id",
        "asof_pitcher_n",
        *PITCHER_RATES,
        "asof_pitcher_pitchmix_n",
        *PITCHMIX_RATES,
        "asof_batter_n",
        "asof_batter_success_rate",
        "asof_batter_middle_rate",
        *SUCCESS_RECENT,
    )
    working = rows.loc[:, selected_columns].copy(deep=True)
    seasons = pd.to_numeric(working["season"], errors="raise")
    if not seasons.eq(state.valid_year).all():
        raise EvaluatorError("evaluation season differs from frozen valid_year")
    working["__position__"] = np.arange(len(working), dtype="int64")
    output = working.merge(
        state.pitcher, on="pitcher_id", how="left", validate="many_to_one", sort=False
    ).merge(
        state.batter, on="batter_id", how="left", validate="many_to_one", sort=False
    )
    output = output.sort_values("__position__", kind="stable").reset_index(drop=True)
    pitcher_n, pitcher_success = _season_delta(
        output,
        current_n=output["asof_pitcher_n"],
        current_rate=output["asof_pitcher_success_rate"],
        snapshot_n_column="snapshot_pitcher_success_n",
        snapshot_count_column="snapshot_pitcher_success_count",
    )
    output["season_pitcher_n"] = pitcher_n.astype("float32")
    output["season_pitcher_log1p_n"] = np.log1p(pitcher_n).astype("float32")
    output["season_pitcher_reliability_100"] = (pitcher_n / (pitcher_n + 100.0)).astype(
        "float32"
    )
    for strength in (10.0, 25.0, 50.0, 100.0, 200.0, 500.0):
        output[f"season_pitcher_success_smooth_{int(strength)}"] = _smoothed_rate(
            pitcher_success, pitcher_n, state.prior_rate, strength
        )
    season_success = _smoothed_rate(pitcher_success, pitcher_n, state.prior_rate, 25.0)
    output["season_pitcher_success_rate"] = np.divide(
        pitcher_success,
        pitcher_n,
        out=np.full(len(output), state.prior_rate, dtype="float64"),
        where=pitcher_n > 0,
    ).astype("float32")
    career_success = pd.to_numeric(output["asof_pitcher_success_rate"]).fillna(
        state.prior_rate
    ).to_numpy()
    output["season_vs_career_success"] = (season_success - career_success).astype(
        "float32"
    )
    for source_column in PITCHER_RATES[1:]:
        suffix = source_column.removeprefix("asof_pitcher_").removesuffix("_rate")
        component_n, component_count = _season_delta(
            output,
            current_n=output["asof_pitcher_n"],
            current_rate=output[source_column],
            snapshot_n_column=f"snapshot_pitcher_{suffix}_n",
            snapshot_count_column=f"snapshot_pitcher_{suffix}_count",
        )
        career = pd.to_numeric(output[source_column]).fillna(0).to_numpy(dtype="float64")
        output[f"season_pitcher_{suffix}_smooth_50"] = _smoothed_rate(
            component_count, component_n, career, 50.0
        )
    for column in SUCCESS_RECENT:
        suffix = column.removeprefix("asof_pitcher_")
        recent_raw = pd.to_numeric(output[column]).to_numpy(dtype="float64")
        recent = np.where(np.isfinite(recent_raw), recent_raw, season_success)
        output[f"season_vs_{suffix}"] = (season_success - recent).astype("float32")
    mix_total = pd.to_numeric(output["asof_pitcher_pitchmix_n"]).fillna(0).to_numpy(
        dtype="float64"
    )
    mix_delta_candidates = []
    for source_column in PITCHMIX_RATES:
        suffix = source_column.removeprefix("asof_pitcher_").removesuffix("_rate")
        component_n, component_count = _season_delta(
            output,
            current_n=pd.Series(mix_total, index=output.index),
            current_rate=output[source_column],
            snapshot_n_column=f"snapshot_pitchmix_{suffix}_n",
            snapshot_count_column=f"snapshot_pitchmix_{suffix}_count",
        )
        career = pd.to_numeric(output[source_column]).fillna(0).to_numpy(dtype="float64")
        output[f"season_pitchmix_{suffix}_smooth_50"] = _smoothed_rate(
            component_count, component_n, career, 50.0
        )
        mix_delta_candidates.append(component_n)
    output["season_pitchmix_n"] = np.maximum.reduce(mix_delta_candidates).astype(
        "float32"
    )
    batter_n, batter_success = _season_delta(
        output,
        current_n=output["asof_batter_n"],
        current_rate=output["asof_batter_success_rate"],
        snapshot_n_column="snapshot_batter_success_n",
        snapshot_count_column="snapshot_batter_success_count",
    )
    output["season_batter_n"] = batter_n.astype("float32")
    output["season_batter_log1p_n"] = np.log1p(batter_n).astype("float32")
    for strength in (50.0, 100.0, 200.0, 500.0):
        output[f"season_batter_success_smooth_{int(strength)}"] = _smoothed_rate(
            batter_success, batter_n, state.prior_rate, strength
        )
    output["season_pitcher_batter_success_gap"] = (
        season_success
        - _smoothed_rate(batter_success, batter_n, state.prior_rate, 200.0)
    ).astype("float32")
    labels = ["0", "1_10", "11_50", "51_200", "201_500", "501_1000", "1000p"]
    bins = [-1, 0, 10, 50, 200, 500, 1000, np.inf]
    output["season_pitcher_n_bucket"] = pd.cut(
        pitcher_n, bins=bins, labels=labels
    ).astype(str)
    output["season_batter_n_bucket"] = pd.cut(
        batter_n, bins=bins, labels=labels
    ).astype(str)
    original = set(working.columns)
    added = [
        column
        for column in output.columns
        if column not in original and not column.startswith("snapshot_")
    ]
    result = output.loc[:, added].copy(deep=True)
    result.index = rows.index
    result.attrs["categorical_columns"] = (
        "season_pitcher_n_bucket",
        "season_batter_n_bucket",
    )
    return result


def _row_local_features(rows: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    frame = rows.drop(columns=[ROW_ID, TARGET], errors="ignore").copy(deep=True)
    for column in RAW_CATEGORICAL:
        frame[column] = _category(rows, column)
    frame["count_state"] = _join(
        _category(rows, "balls_before"), _category(rows, "strikes_before")
    )
    frame["hand_matchup"] = _join(
        _category(rows, "pitcher_hand"), _category(rows, "batter_hand")
    )
    frame["pitcher_batter_hand"] = _join(
        _category(rows, "pitcher_id"), _category(rows, "batter_hand")
    )
    frame["pitcher_count"] = _join(_category(rows, "pitcher_id"), frame["count_state"])
    frame["pitcher_base"] = _join(
        _category(rows, "pitcher_id"), _category(rows, "base_state")
    )
    frame["pitcher_game_type"] = _join(
        _category(rows, "pitcher_id"), _category(rows, "game_type")
    )
    frame["batter_pitcher_hand"] = _join(
        _category(rows, "batter_id"), _category(rows, "pitcher_hand")
    )
    frame["team_count"] = _join(
        _category(rows, "pitcher_team_id"), frame["count_state"]
    )
    frame["count_base"] = _join(frame["count_state"], _category(rows, "base_state"))
    frame["hand_count"] = _join(frame["hand_matchup"], frame["count_state"])
    inning = _numeric(rows, "inning")
    score = _numeric(rows, "score_diff_pitcher_team")
    leverage = _numeric(rows, "li")
    frame["inning_bin"] = pd.cut(
        inning, [-np.inf, 3, 6, 9, np.inf], labels=["early", "middle", "late", "extra"]
    ).astype("string").fillna(MISSING_CATEGORY).astype(object)
    frame["score_bin"] = pd.cut(
        score,
        [-np.inf, -3, -1, 1, 3, np.inf],
        labels=["far_behind", "behind", "close", "ahead", "far_ahead"],
    ).astype("string").fillna(MISSING_CATEGORY).astype(object)
    frame["leverage_bin"] = pd.cut(
        leverage,
        [-np.inf, 0.75, 1.5, 3, np.inf],
        labels=["low", "normal", "high", "extreme"],
    ).astype("string").fillna(MISSING_CATEGORY).astype(object)
    frame["inning_score"] = _join(frame["inning_bin"], frame["score_bin"])
    frame["leverage_score"] = _join(frame["leverage_bin"], frame["score_bin"])
    frame["leverage_runners_count"] = _join(
        frame["leverage_bin"], _category(rows, "num_runners_on"), frame["count_state"]
    )
    recent_success = pd.concat([_numeric(rows, column) for column in SUCCESS_RECENT], axis=1)
    recent_middle = pd.concat([_numeric(rows, column) for column in MIDDLE_RECENT], axis=1)
    frame["recent_success_mean"] = recent_success.mean(axis=1)
    frame["recent_success_std"] = recent_success.std(axis=1, ddof=0)
    frame["recent_success_slope"] = (recent_success.iloc[:, 0] - recent_success.iloc[:, 2]) / 4.0
    frame["recent_middle_mean"] = recent_middle.mean(axis=1)
    frame["recent_middle_slope"] = (recent_middle.iloc[:, 0] - recent_middle.iloc[:, 2]) / 4.0
    pitcher_n = _numeric(rows, "asof_pitcher_n").clip(lower=0)
    batter_n = _numeric(rows, "asof_batter_n").clip(lower=0)
    pitcher_rate = _numeric(rows, "asof_pitcher_success_rate")
    batter_rate = _numeric(rows, "asof_batter_success_rate")
    frame["pitcher_reliability_100"] = pitcher_n / (pitcher_n + 100.0)
    frame["batter_reliability_100"] = batter_n / (batter_n + 100.0)
    frame["pitcher_batter_success_gap"] = pitcher_rate - batter_rate
    clipped = pitcher_rate.clip(1e-5, 1 - 1e-5)
    frame["pitcher_success_logit"] = np.log(clipped / (1.0 - clipped))
    pitchmix = pd.concat([_numeric(rows, column) for column in PITCHMIX_RATES], axis=1).clip(lower=0)
    frame["pitchmix_entropy"] = -(pitchmix * np.log(pitchmix.clip(lower=1e-12))).sum(axis=1)
    runners = _numeric(rows, "num_runners_on").clip(lower=0)
    frame["pressure_index"] = (
        np.log1p(leverage.clip(lower=0)) * (1.0 + score.abs()) * (1.0 + runners / 3.0)
    )
    return frame, (*RAW_CATEGORICAL, *CROSS_CATEGORICAL)


def _anchor(rows: pd.DataFrame, s1: pd.DataFrame, prior: float) -> np.ndarray:
    n = _numeric(rows, "asof_pitcher_n").clip(lower=0)
    career = _numeric(rows, "asof_pitcher_success_rate").fillna(prior)
    recent = pd.concat([_numeric(rows, column) for column in SUCCESS_RECENT], axis=1).mean(
        axis=1
    ).fillna(prior)
    season_n = pd.to_numeric(s1["season_pitcher_n"], errors="coerce").clip(lower=0)
    season_rate = pd.to_numeric(
        s1["season_pitcher_success_smooth_100"], errors="coerce"
    ).fillna(prior)
    career_weight = n / (n + 100.0)
    season_weight = 0.15 + 0.30 * season_n / (season_n + 80.0)
    career_anchor = prior + career_weight * (career - prior)
    values = np.clip(
        career_anchor
        + season_weight * (season_rate - career_anchor)
        + 0.10 * (recent - career_anchor),
        1e-5,
        1 - 1e-5,
    ).to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise EvaluatorError("anchor contains non-finite values")
    return values


def transform_features(
    rows: pd.DataFrame, state: FrozenState
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    source = _validate_rows(rows)
    s1 = _seasonal_features(source, state)
    frame, categorical = _row_local_features(source)
    for column in s1:
        if column in frame:
            raise EvaluatorError(f"S1 feature collision: {column}")
        frame[column] = s1[column].to_numpy(copy=True)
    all_categories = tuple(
        dict.fromkeys((*categorical, *tuple(s1.attrs.get("categorical_columns", ()))))
    )
    for column in all_categories:
        frame[column] = frame[column].astype("string").fillna(MISSING_CATEGORY).astype(object)
    for column in frame:
        if column in all_categories:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").astype("float32")
        if np.isinf(values.to_numpy()).any():
            raise EvaluatorError(f"feature contains infinity: {column}")
        frame[column] = values
    if tuple(frame.columns) != state.feature_columns:
        raise EvaluatorError("evaluation feature schema differs")
    if all_categories != state.categorical_columns:
        raise EvaluatorError("evaluation categorical schema differs")
    return (
        frame,
        _anchor(source, s1, state.prior_rate),
        source[ROW_ID].astype(str).to_numpy(copy=True),
    )


class TreeE2Predictor:
    def __init__(
        self,
        state: FrozenState,
        models: Sequence[object],
        *,
        state_sha256: str,
    ) -> None:
        values = tuple(models)
        if len(values) != 3 or any(not callable(getattr(model, "predict", None)) for model in values):
            raise EvaluatorError("exactly three CatBoost-compatible models are required")
        if not _valid_sha(state_sha256):
            raise EvaluatorError("predictor state SHA-256 differs")
        self.state = state
        self.models = values
        self._state_sha256 = state_sha256

    def state_digest(self) -> str:
        return self._state_sha256

    def encode(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        frame, _, _ = transform_features(rows, self.state)
        categorical = set(self.state.categorical_columns)
        numeric_columns = [column for column in frame if column not in categorical]
        numeric = frame.loc[:, numeric_columns].to_numpy(dtype="float32", copy=True)
        text = frame.loc[:, self.state.categorical_columns].astype("string").fillna(
            MISSING_CATEGORY
        )
        width = max(1, max((len(value) for value in text.to_numpy().ravel()), default=1))
        categories = text.to_numpy(dtype=f"<U{width}", copy=True)
        return numeric, categories

    def predict_batch(self, rows: pd.DataFrame, *, batch_size: int = 4096) -> np.ndarray:
        if type(batch_size) is not int or batch_size <= 0:
            raise EvaluatorError("batch size must be positive")
        frame, anchor, _ = transform_features(rows, self.state)
        members = []
        for model in self.models:
            residual = np.asarray(model.predict(frame), dtype="float64")
            if residual.shape != anchor.shape or not np.isfinite(residual).all():
                raise EvaluatorError("CatBoost prediction values differ")
            members.append(np.clip(anchor + residual, 1e-5, 1 - 1e-5))
        return np.mean(np.stack(members), axis=0)


def _combined_model_digest(model_dir: Path, members: Mapping[str, str]) -> str:
    digest = sha256()
    for name in sorted(members):
        value = (model_dir / name).read_bytes()
        digest.update(name.encode("utf-8") + b"\0" + sha256(value).digest())
    return digest.hexdigest()


def load_frozen_predictor(
    model_dir: Path, *, metadata: Mapping[str, object] | None = None
) -> TreeE2Predictor:
    selected = EMBEDDED_METADATA if metadata is None else metadata
    if not isinstance(selected, Mapping):
        raise EvaluatorError("embedded candidate metadata is missing")
    expected = selected.get("members")
    if not isinstance(expected, Mapping) or set(expected) != MODEL_MEMBERS:
        raise EvaluatorError("embedded model member manifest differs")
    root = Path(model_dir)
    if root.is_symlink() or not root.is_dir():
        raise EvaluatorError("model directory is missing or unsafe")
    paths = [path for path in root.rglob("*") if path.is_file()]
    names = {path.relative_to(root).as_posix() for path in paths}
    if names != MODEL_MEMBERS or any(path.is_symlink() for path in paths):
        raise EvaluatorError("model directory member set differs")
    for name, expected_sha in expected.items():
        if not _valid_sha(expected_sha) or _sha256_file(root / name) != expected_sha:
            raise EvaluatorError(f"model member SHA-256 differs: {name}")
    if _combined_model_digest(root, expected) != selected.get("model_sha256"):
        raise EvaluatorError("combined model SHA-256 differs")
    try:
        from catboost import CatBoostRegressor
    except ImportError as error:
        raise EvaluatorError("catboost==1.2.10 is required") from error
    models = []
    for seed in (42, 2026, 3407):
        model = CatBoostRegressor(thread_count=6)
        model.load_model(str(root / f"models/catboost_seed_{seed}.cbm"))
        models.append(model)
    state = load_frozen_state(root / "frozen_state")
    return TreeE2Predictor(state, models, state_sha256=str(selected["model_sha256"]))


def validate_inputs(
    test: pd.DataFrame, sample: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if type(test) is not pd.DataFrame or type(sample) is not pd.DataFrame:
        raise EvaluatorError("test and sample submission must be pandas DataFrames")
    if sample.columns.tolist() != [ROW_ID, TARGET]:
        raise EvaluatorError("sample submission columns differ")
    test_copy = _validate_rows(test)
    sample_copy = sample.copy(deep=True)
    ids = sample_copy[ROW_ID].astype("string")
    if ids.isna().any() or ids.astype(str).duplicated().any():
        raise EvaluatorError("sample row_id values must be non-null and unique")
    sample_copy[ROW_ID] = ids.astype(str)
    test_copy[ROW_ID] = test_copy[ROW_ID].astype(str)
    if len(test_copy) != len(sample_copy) or set(test_copy[ROW_ID]) != set(sample_copy[ROW_ID]):
        raise EvaluatorError("test row IDs must exactly match sample submission")
    return test_copy, sample_copy


def _canonical_probability(value: float) -> str:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise EvaluatorError("probability must be finite and inside [0, 1]")
    return f"{number:.8f}"


def _publish(sample: pd.DataFrame, predictions: Mapping[str, str], output: Path) -> None:
    if output.exists():
        raise EvaluatorError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.partial")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow([ROW_ID, TARGET])
            writer.writerows((row_id, predictions[row_id]) for row_id in sample[ROW_ID])
            handle.flush()
            os.fsync(handle.fileno())
        with output.open("xb") as target, temporary.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
    except Exception:
        output.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def _input_directory() -> Path:
    for name in ("data", "open"):
        directory = Path(name)
        if (directory / "test.csv").is_file() and (
            directory / "sample_submission.csv"
        ).is_file():
            return directory
    raise EvaluatorError("test.csv and sample_submission.csv were not found")


def main() -> int:
    if not isinstance(EMBEDDED_METADATA, dict):
        raise EvaluatorError("script is not bound to a candidate")
    data_dir = _input_directory()
    test = pd.read_csv(data_dir / "test.csv", dtype={ROW_ID: "string"})
    sample = pd.read_csv(data_dir / "sample_submission.csv", dtype={ROW_ID: "string"})
    test, sample = validate_inputs(test, sample)
    predictor = load_frozen_predictor(Path("model"))
    before = predictor.state_digest()
    values = np.asarray(predictor.predict_batch(test, batch_size=4096), dtype="float64")
    if values.shape != (len(test),) or predictor.state_digest() != before:
        raise EvaluatorError("prediction shape or predictor state differs")
    by_id = dict(
        zip(test[ROW_ID], (_canonical_probability(value) for value in values), strict=True)
    )
    canary_order = sorted(
        range(len(test)),
        key=lambda index: sha256(
            ("tree-expert-e2-c1-v1\0" + test.iloc[index][ROW_ID]).encode("utf-8")
        ).hexdigest(),
    )[: min(8, len(test))]
    for index in canary_order:
        single = test.iloc[[index]].reset_index(drop=True)
        candidate = predictor.predict_batch(single, batch_size=1)[0]
        if abs(candidate - values[index]) > 1e-12:
            raise EvaluatorError("row-independence canary mismatch")
    _publish(sample, by_id, Path("output/submission.csv"))
    print(
        f"SUBMISSION_SUCCESS rows={len(sample)} output=output/submission.csv "
        f"candidate={EMBEDDED_METADATA['candidate_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
