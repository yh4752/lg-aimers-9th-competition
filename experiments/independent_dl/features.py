"""Fold-fitted feature views and memory-mapped cache artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from .feature_sources.trackman import (
    PITCHER_LOOKUP_COLUMNS,
    TrackmanBuildResult,
    build_trackman_lookup,
)
from .preprocessing import (
    PreprocessingSpec,
    PreprocessingState,
    fit_preprocessor,
    transform_preprocessor,
)


TARGET_COLUMN = "control_success"
ROW_ID_COLUMN = "row_id"
MISSING_CATEGORY = "__MISSING__"
VALID_VIEWS = (
    "raw_typed",
    "engineered",
    "entity_context",
    "trackman_augmented",
)
BASE_CATEGORICAL_COLUMNS = (
    "game_type",
    "top_bottom",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
)


class FeatureContractError(ValueError):
    """Raised when a fold feature view cannot be reproduced safely."""


@dataclass(frozen=True)
class FeatureState:
    view: str
    cutoff_year: int
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    category_maps: Mapping[str, Mapping[str, int]]
    numeric_mean: tuple[float, ...]
    numeric_std: tuple[float, ...]
    engineered_payload: Mapping[str, object]
    trackman_result: TrackmanBuildResult | None
    trackman_lookup_sha256: str | None


@dataclass(frozen=True)
class FeatureBatch:
    row_id: np.ndarray
    season: np.ndarray
    game_type: np.ndarray
    x_num: np.ndarray
    x_cat: np.ndarray
    y: np.ndarray | None


@dataclass(frozen=True)
class FeatureMetadata:
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    categorical_cardinalities: tuple[int, ...]


@dataclass(frozen=True)
class FoldCache:
    root: Path
    train: FeatureBatch
    valid: FeatureBatch
    state: FeatureState
    reused: bool


@dataclass(frozen=True)
class PreprocessedFeatureState:
    view: str
    cutoff_year: int
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    category_maps: Mapping[str, Mapping[str, int]]
    preprocessing: PreprocessingState
    trackman_result: TrackmanBuildResult | None
    trackman_lookup_sha256: str | None


def feature_metadata(state: FeatureState) -> FeatureMetadata:
    return FeatureMetadata(
        numeric_columns=state.numeric_columns,
        categorical_columns=state.categorical_columns,
        categorical_cardinalities=tuple(
            len(state.category_maps[column]) + 1
            for column in state.categorical_columns
        ),
    )


def _category_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna(MISSING_CATEGORY)


def _numeric(values: pd.Series, column: str) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & result.isna()
    if invalid.any():
        raise FeatureContractError(f"{column} contains a non-numeric value")
    return result.astype("float64")


def _frequency(values: pd.Series) -> dict[str, int]:
    return {
        str(key): int(count)
        for key, count in _category_text(values).value_counts(dropna=False).items()
    }


def _engineered_payload(train: pd.DataFrame) -> dict[str, object]:
    payload: dict[str, object] = {}
    for column in ("pitcher_id", "batter_id"):
        if column in train:
            payload[f"{column}_frequency"] = _frequency(train[column])
    return payload


def _add_engineered(
    frame: pd.DataFrame, payload: Mapping[str, object]
) -> pd.DataFrame:
    result = frame.copy()
    if {"balls_before", "strikes_before"}.issubset(result):
        result["count_state_numeric"] = (
            _numeric(result["balls_before"], "balls_before") * 3.0
            + _numeric(result["strikes_before"], "strikes_before")
        )
    if {"top_bottom", "home_win_expectancy", "away_win_expectancy"}.issubset(
        result
    ):
        result["we_pitcher_team"] = np.where(
            result["top_bottom"].eq("T"),
            _numeric(result["home_win_expectancy"], "home_win_expectancy"),
            _numeric(result["away_win_expectancy"], "away_win_expectancy"),
        )
    if {"pitcher_hand", "batter_hand"}.issubset(result):
        result["hand_matchup"] = (
            _category_text(result["pitcher_hand"])
            + "_"
            + _category_text(result["batter_hand"])
        )
    if {"asof_pitcher_n", "asof_pitcher_success_rate"}.issubset(result):
        count = _numeric(result["asof_pitcher_n"], "asof_pitcher_n").fillna(0).clip(
            lower=0
        )
        rate = (
            _numeric(result["asof_pitcher_success_rate"], "asof_pitcher_success_rate")
            .fillna(0.5)
            .clip(0, 1)
        )
        result["pitcher_success_smooth_150"] = (
            count * rate + 150.0 * 0.5
        ) / (count + 150.0)
    if {"run_top_before", "run_bot_before"}.issubset(result):
        result["runs_before_total"] = _numeric(
            result["run_top_before"], "run_top_before"
        ).fillna(0) + _numeric(result["run_bot_before"], "run_bot_before").fillna(0)
    if {"score_diff_home", "num_runners_on"}.issubset(result):
        result["score_runner_pressure"] = _numeric(
            result["score_diff_home"], "score_diff_home"
        ).fillna(0) * (1.0 + _numeric(result["num_runners_on"], "num_runners_on").fillna(0))
    for entity in ("pitcher_id", "batter_id"):
        key = f"{entity}_frequency"
        mapping_value = payload.get(key, {})
        if not isinstance(mapping_value, Mapping) or entity not in result:
            continue
        counts = _category_text(result[entity]).map(mapping_value).fillna(0).astype(float)
        result[key] = counts
        result[f"{entity}_frequency_log1p"] = np.log1p(counts)
        result[f"{entity}_unseen"] = counts.eq(0).astype(float)
    return result


def _add_entity_context(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    interactions = {
        "pitcher_batter": ("pitcher_id", "batter_id"),
        "pitcher_hand_matchup": ("pitcher_id", "batter_hand"),
        "inning_base_state": ("inning", "base_state"),
        "game_type_count": ("game_type", "balls_before", "strikes_before"),
    }
    for name, columns in interactions.items():
        if set(columns).issubset(result):
            token = _category_text(result[columns[0]])
            for column in columns[1:]:
                token = token + "|" + _category_text(result[column])
            result[f"ctx_{name}"] = token
    return result


def _lookup_sha256(lookup: pd.DataFrame) -> str:
    digest = sha256()
    digest.update(repr(tuple(lookup.columns)).encode("utf-8"))
    digest.update(
        pd.util.hash_pandas_object(lookup, index=True, categorize=False)
        .to_numpy(dtype="uint64", copy=False)
        .tobytes()
    )
    return digest.hexdigest()


def _attach_trackman(
    frame: pd.DataFrame, result: TrackmanBuildResult
) -> pd.DataFrame:
    if "pitcher_id" not in frame:
        raise FeatureContractError("trackman view requires pitcher_id")
    marker = "__independent_dl_row_order__"
    output = frame.copy()
    output[marker] = np.arange(len(output), dtype="int64")
    output = output.merge(
        result.lookup,
        on="pitcher_id",
        how="left",
        sort=False,
        validate="many_to_one",
    ).sort_values(marker, kind="stable")
    output = output.drop(columns=[marker]).reset_index(drop=True)
    output["tm_lookup_missing"] = output["tm_match_accepted"].isna().astype(float)
    return output


def _prepare_frame(
    frame: pd.DataFrame,
    *,
    view: str,
    payload: Mapping[str, object],
    trackman_result: TrackmanBuildResult | None,
) -> pd.DataFrame:
    output = frame.drop(
        columns=[column for column in (ROW_ID_COLUMN, TARGET_COLUMN) if column in frame]
    ).copy()
    if view != "raw_typed":
        output = _add_engineered(output, payload)
    if view in {"entity_context", "trackman_augmented"}:
        output = _add_entity_context(output)
    if view == "trackman_augmented":
        if trackman_result is None:
            raise FeatureContractError("trackman view is missing its cutoff lookup")
        output = _attach_trackman(output, trackman_result)
    if output.columns.duplicated().any():
        raise FeatureContractError("feature columns must be unique")
    return output


def _fit_categories(
    frame: pd.DataFrame, columns: tuple[str, ...]
) -> Mapping[str, Mapping[str, int]]:
    result: dict[str, Mapping[str, int]] = {}
    for column in columns:
        levels = sorted(set(_category_text(frame[column]).tolist()))
        result[column] = MappingProxyType(
            {level: index + 1 for index, level in enumerate(levels)}
        )
    return MappingProxyType(result)


def _build_batch(frame: pd.DataFrame, prepared: pd.DataFrame, state: FeatureState) -> FeatureBatch:
    if ROW_ID_COLUMN not in frame or frame[ROW_ID_COLUMN].isna().any():
        raise FeatureContractError("row_id must be present and non-null")
    row_id = frame[ROW_ID_COLUMN].astype(str).to_numpy(dtype=str)
    if len(set(row_id.tolist())) != len(row_id):
        raise FeatureContractError("row_id must be unique")
    numeric = prepared.loc[:, state.numeric_columns].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype="float64")
    values[~np.isfinite(values)] = np.nan
    means = np.asarray(state.numeric_mean, dtype="float64")
    scales = np.asarray(state.numeric_std, dtype="float64")
    values = np.where(np.isnan(values), means, values)
    x_num = ((values - means) / scales).astype("float32")
    if state.categorical_columns:
        encoded = [
            _category_text(prepared[column])
            .map(state.category_maps[column])
            .fillna(0)
            .to_numpy(dtype="int64")
            for column in state.categorical_columns
        ]
        x_cat = np.column_stack(encoded).astype("int64", copy=False)
    else:
        x_cat = np.empty((len(frame), 0), dtype="int64")
    target: np.ndarray | None = None
    if TARGET_COLUMN in frame:
        target = _numeric(frame[TARGET_COLUMN], TARGET_COLUMN).to_numpy(dtype="float32")
        if not np.isin(target, [0.0, 1.0]).all():
            raise FeatureContractError("target must contain only 0 and 1")
    season = _numeric(frame["season"], "season").to_numpy(dtype="int64")
    game_type = (
        _category_text(frame["game_type"]).to_numpy(dtype=str)
        if "game_type" in frame
        else np.full(len(frame), MISSING_CATEGORY, dtype=str)
    )
    return FeatureBatch(
        row_id=row_id,
        season=season,
        game_type=game_type,
        x_num=x_num,
        x_cat=x_cat,
        y=target,
    )


def fit_feature_view(
    train: pd.DataFrame,
    history: pd.DataFrame,
    *,
    view: str,
    cutoff_year: int,
) -> tuple[FeatureState, FeatureBatch]:
    """Fit one feature view exclusively on a fold's training rows."""

    if view not in VALID_VIEWS:
        raise FeatureContractError(f"unknown feature view: {view}")
    if "season" not in train or train["season"].gt(cutoff_year).any():
        raise FeatureContractError("training rows exceed the fold cutoff")
    payload = _engineered_payload(train) if view != "raw_typed" else {}
    trackman_result = (
        build_trackman_lookup(train, history, cutoff_year)
        if view == "trackman_augmented"
        else None
    )
    prepared = _prepare_frame(
        train,
        view=view,
        payload=payload,
        trackman_result=trackman_result,
    )
    categorical_columns = tuple(
        column
        for column in prepared.columns
        if column in BASE_CATEGORICAL_COLUMNS or column.startswith("ctx_") or column == "hand_matchup"
    )
    numeric_columns = tuple(
        column for column in prepared.columns if column not in categorical_columns
    )
    numeric = prepared.loc[:, numeric_columns].apply(
        lambda values: _numeric(values, str(values.name))
    )
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    means = numeric.mean(axis=0, skipna=True).fillna(0.0).to_numpy(dtype="float64")
    scales = numeric.std(axis=0, ddof=0, skipna=True).to_numpy(dtype="float64")
    scales[~np.isfinite(scales) | (scales <= 0)] = 1.0
    category_maps = _fit_categories(prepared, categorical_columns)
    lookup_hash = (
        _lookup_sha256(trackman_result.lookup) if trackman_result is not None else None
    )
    state = FeatureState(
        view=view,
        cutoff_year=cutoff_year,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        category_maps=category_maps,
        numeric_mean=tuple(float(item) for item in means),
        numeric_std=tuple(float(item) for item in scales),
        engineered_payload=MappingProxyType(dict(payload)),
        trackman_result=trackman_result,
        trackman_lookup_sha256=lookup_hash,
    )
    return state, _build_batch(train, prepared, state)


def transform_feature_view(frame: pd.DataFrame, state: FeatureState) -> FeatureBatch:
    """Transform rows using one already-fitted feature state."""

    prepared = _prepare_frame(
        frame,
        view=state.view,
        payload=state.engineered_payload,
        trackman_result=state.trackman_result,
    )
    expected = state.numeric_columns + state.categorical_columns
    if set(prepared.columns) != set(expected):
        missing = sorted(set(expected) - set(prepared.columns))
        extra = sorted(set(prepared.columns) - set(expected))
        raise FeatureContractError(
            f"feature schema differs from fitted state: missing={missing}, extra={extra}"
        )
    return _build_batch(frame, prepared, state)


def _frame_sha256(frame: pd.DataFrame) -> str:
    digest = sha256()
    digest.update(repr(tuple(frame.columns)).encode("utf-8"))
    digest.update(repr(tuple(str(dtype) for dtype in frame.dtypes)).encode("utf-8"))
    digest.update(
        pd.util.hash_pandas_object(frame, index=True, categorize=False)
        .to_numpy(dtype="uint64", copy=False)
        .tobytes()
    )
    return digest.hexdigest()


def _row_sha256(frame: pd.DataFrame) -> str:
    if ROW_ID_COLUMN not in frame:
        raise FeatureContractError("row_id is missing")
    return sha256("\n".join(frame[ROW_ID_COLUMN].astype(str)).encode("utf-8")).hexdigest()


def _code_sha256() -> str:
    return sha256(Path(__file__).read_bytes()).hexdigest()


def _state_payload(state: FeatureState) -> dict[str, object]:
    return {
        "view": state.view,
        "cutoff_year": state.cutoff_year,
        "numeric_columns": list(state.numeric_columns),
        "categorical_columns": list(state.categorical_columns),
        "category_maps": {
            column: dict(mapping) for column, mapping in state.category_maps.items()
        },
        "numeric_mean": list(state.numeric_mean),
        "numeric_std": list(state.numeric_std),
        "engineered_payload": {
            key: dict(value) if isinstance(value, Mapping) else value
            for key, value in state.engineered_payload.items()
        },
        "trackman": (
            None
            if state.trackman_result is None
            else {
                "cutoff_year": state.trackman_result.cutoff_year,
                "lookup_schema": list(state.trackman_result.lookup_schema),
                "lookup_sha256": state.trackman_lookup_sha256,
            }
        ),
    }


def _state_from_payload(root: Path, payload: Mapping[str, object]) -> FeatureState:
    trackman_payload = payload.get("trackman")
    trackman_result: TrackmanBuildResult | None = None
    trackman_hash: str | None = None
    if isinstance(trackman_payload, Mapping):
        lookup = pd.read_csv(root / "trackman_lookup.csv")
        trackman_hash = _lookup_sha256(lookup)
        if trackman_hash != trackman_payload.get("lookup_sha256"):
            raise FeatureContractError("trackman cache hash differs")
        trackman_result = TrackmanBuildResult(
            cutoff_year=int(trackman_payload["cutoff_year"]),
            lookup=lookup,
            lookup_schema=tuple(str(item) for item in trackman_payload["lookup_schema"]),
            mapping=pd.DataFrame(),
            team_mapping=pd.DataFrame(),
        )
    category_maps_payload = payload["category_maps"]
    if not isinstance(category_maps_payload, Mapping):
        raise FeatureContractError("cached category maps are invalid")
    category_maps = MappingProxyType(
        {
            str(column): MappingProxyType(
                {str(key): int(value) for key, value in mapping.items()}
            )
            for column, mapping in category_maps_payload.items()
            if isinstance(mapping, Mapping)
        }
    )
    engineered_payload = payload.get("engineered_payload", {})
    if not isinstance(engineered_payload, Mapping):
        raise FeatureContractError("cached engineered payload is invalid")
    return FeatureState(
        view=str(payload["view"]),
        cutoff_year=int(payload["cutoff_year"]),
        numeric_columns=tuple(str(item) for item in payload["numeric_columns"]),
        categorical_columns=tuple(str(item) for item in payload["categorical_columns"]),
        category_maps=category_maps,
        numeric_mean=tuple(float(item) for item in payload["numeric_mean"]),
        numeric_std=tuple(float(item) for item in payload["numeric_std"]),
        engineered_payload=MappingProxyType(dict(engineered_payload)),
        trackman_result=trackman_result,
        trackman_lookup_sha256=trackman_hash,
    )


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_batch(root: Path, batch: FeatureBatch) -> None:
    root.mkdir()
    np.save(root / "row_id.npy", np.asarray(batch.row_id, dtype=str), allow_pickle=False)
    np.save(root / "season.npy", batch.season, allow_pickle=False)
    np.save(root / "game_type.npy", np.asarray(batch.game_type, dtype=str), allow_pickle=False)
    np.save(root / "x_num.npy", batch.x_num, allow_pickle=False)
    np.save(root / "x_cat.npy", batch.x_cat, allow_pickle=False)
    if batch.y is not None:
        np.save(root / "y.npy", batch.y, allow_pickle=False)


def _load_batch(root: Path) -> FeatureBatch:
    target = root / "y.npy"
    return FeatureBatch(
        row_id=np.load(root / "row_id.npy", mmap_mode="r", allow_pickle=False),
        season=np.load(root / "season.npy", mmap_mode="r", allow_pickle=False),
        game_type=np.load(root / "game_type.npy", mmap_mode="r", allow_pickle=False),
        x_num=np.load(root / "x_num.npy", mmap_mode="r", allow_pickle=False),
        x_cat=np.load(root / "x_cat.npy", mmap_mode="r", allow_pickle=False),
        y=(np.load(target, mmap_mode="r", allow_pickle=False) if target.is_file() else None),
    )


def materialize_fold_cache(
    cache_root: str | Path,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    history: pd.DataFrame,
    view: str,
    train_end_year: int,
    valid_year: int,
) -> FoldCache:
    """Build or safely reuse one fold/view cache."""

    if valid_year != train_end_year + 1:
        raise FeatureContractError("fold must be a yearly transition")
    if train["season"].gt(train_end_year).any():
        raise FeatureContractError("training rows exceed train_end_year")
    if not valid["season"].eq(valid_year).all():
        raise FeatureContractError("validation rows do not match valid_year")
    identity = {
        "view": view,
        "train_end_year": train_end_year,
        "valid_year": valid_year,
        "train_row_sha256": _row_sha256(train),
        "valid_row_sha256": _row_sha256(valid),
        "train_frame_sha256": _frame_sha256(train),
        "valid_frame_sha256": _frame_sha256(valid),
        "history_frame_sha256": _frame_sha256(history),
        "feature_code_sha256": _code_sha256(),
    }
    root = Path(cache_root).expanduser().resolve()
    fold_root = root / f"train_{train_end_year}_valid_{valid_year}"
    target = fold_root / view
    if target.exists():
        try:
            saved = json.loads((target / "identity.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FeatureContractError("existing cache is incomplete") from error
        if saved != identity:
            if saved.get("valid_row_sha256") != identity["valid_row_sha256"]:
                raise FeatureContractError("validation row identity differs from cache")
            if saved.get("train_row_sha256") != identity["train_row_sha256"]:
                raise FeatureContractError("training row identity differs from cache")
            raise FeatureContractError("cache identity differs from current inputs")
        state_payload = json.loads((target / "state.json").read_text(encoding="utf-8"))
        state = _state_from_payload(target, state_payload)
        return FoldCache(
            root=target,
            train=_load_batch(target / "train"),
            valid=_load_batch(target / "valid"),
            state=state,
            reused=True,
        )

    fold_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{view}-", dir=fold_root))
    try:
        state, train_batch = fit_feature_view(
            train, history, view=view, cutoff_year=train_end_year
        )
        valid_batch = transform_feature_view(valid, state)
        _write_batch(temporary / "train", train_batch)
        _write_batch(temporary / "valid", valid_batch)
        if state.trackman_result is not None:
            state.trackman_result.lookup.to_csv(
                temporary / "trackman_lookup.csv", index=False
            )
        _write_json(temporary / "state.json", _state_payload(state))
        _write_json(temporary / "identity.json", identity)
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return FoldCache(
        root=target,
        train=_load_batch(target / "train"),
        valid=_load_batch(target / "valid"),
        state=_state_from_payload(
            target,
            json.loads((target / "state.json").read_text(encoding="utf-8")),
        ),
        reused=False,
    )


def _preprocessing_id(spec: PreprocessingSpec) -> str:
    suffix = "__".join(spec.components) if spec.components else "baseline"
    return f"{spec.profile}__{suffix}"


def _preprocessing_code_sha256() -> str:
    from . import preprocessing

    return sha256(Path(preprocessing.__file__).read_bytes()).hexdigest()


def _preprocessing_payload(state: PreprocessingState) -> dict[str, object]:
    return {
        "spec": {
            "profile": state.spec.profile,
            "components": list(state.spec.components),
        },
        "source_columns": list(state.source_columns),
        "output_columns": list(state.output_columns),
        "categorical_columns": list(state.categorical_columns),
        "numeric_columns": list(state.numeric_columns),
        "numeric_median": dict(state.numeric_median),
        "numeric_mean": dict(state.numeric_mean),
        "numeric_std": dict(state.numeric_std),
        "yeo_johnson_lambda": dict(state.yeo_johnson_lambda),
        "entity_frequency": {
            entity: dict(mapping)
            for entity, mapping in state.entity_frequency.items()
        },
        "target_prior": state.target_prior,
    }


def _preprocessing_from_payload(payload: Mapping[str, object]) -> PreprocessingState:
    spec_payload = payload["spec"]
    if not isinstance(spec_payload, Mapping):
        raise FeatureContractError("cached preprocessing spec is invalid")
    frequencies = payload["entity_frequency"]
    if not isinstance(frequencies, Mapping):
        raise FeatureContractError("cached entity frequencies are invalid")
    return PreprocessingState(
        spec=PreprocessingSpec(
            str(spec_payload["profile"]),
            tuple(str(item) for item in spec_payload["components"]),
        ),
        source_columns=tuple(str(item) for item in payload["source_columns"]),
        output_columns=tuple(str(item) for item in payload["output_columns"]),
        categorical_columns=tuple(
            str(item) for item in payload["categorical_columns"]
        ),
        numeric_columns=tuple(str(item) for item in payload["numeric_columns"]),
        numeric_median=MappingProxyType(
            {str(key): float(value) for key, value in payload["numeric_median"].items()}
        ),
        numeric_mean=MappingProxyType(
            {str(key): float(value) for key, value in payload["numeric_mean"].items()}
        ),
        numeric_std=MappingProxyType(
            {str(key): float(value) for key, value in payload["numeric_std"].items()}
        ),
        yeo_johnson_lambda=MappingProxyType(
            {
                str(key): float(value)
                for key, value in payload["yeo_johnson_lambda"].items()
            }
        ),
        entity_frequency=MappingProxyType(
            {
                str(entity): MappingProxyType(
                    {str(key): int(value) for key, value in mapping.items()}
                )
                for entity, mapping in frequencies.items()
            }
        ),
        target_prior=float(payload["target_prior"]),
    )


def _preprocessed_state_payload(state: PreprocessedFeatureState) -> dict[str, object]:
    return {
        "view": state.view,
        "cutoff_year": state.cutoff_year,
        "numeric_columns": list(state.numeric_columns),
        "categorical_columns": list(state.categorical_columns),
        "category_maps": {
            column: dict(mapping) for column, mapping in state.category_maps.items()
        },
        "preprocessing": _preprocessing_payload(state.preprocessing),
        "trackman": (
            None
            if state.trackman_result is None
            else {
                "cutoff_year": state.trackman_result.cutoff_year,
                "lookup_schema": list(state.trackman_result.lookup_schema),
                "lookup_sha256": state.trackman_lookup_sha256,
            }
        ),
    }


def _preprocessed_state_from_payload(
    root: Path, payload: Mapping[str, object]
) -> PreprocessedFeatureState:
    trackman_payload = payload.get("trackman")
    trackman_result: TrackmanBuildResult | None = None
    trackman_hash: str | None = None
    if isinstance(trackman_payload, Mapping):
        lookup = pd.read_csv(root / "trackman_lookup.csv")
        trackman_hash = _lookup_sha256(lookup)
        if trackman_hash != trackman_payload.get("lookup_sha256"):
            raise FeatureContractError("trackman cache hash differs")
        trackman_result = TrackmanBuildResult(
            cutoff_year=int(trackman_payload["cutoff_year"]),
            lookup=lookup,
            lookup_schema=tuple(str(item) for item in trackman_payload["lookup_schema"]),
            mapping=pd.DataFrame(),
            team_mapping=pd.DataFrame(),
        )
    maps = payload["category_maps"]
    if not isinstance(maps, Mapping):
        raise FeatureContractError("cached category maps are invalid")
    preprocessing_payload = payload["preprocessing"]
    if not isinstance(preprocessing_payload, Mapping):
        raise FeatureContractError("cached preprocessing state is invalid")
    return PreprocessedFeatureState(
        view=str(payload["view"]),
        cutoff_year=int(payload["cutoff_year"]),
        numeric_columns=tuple(str(item) for item in payload["numeric_columns"]),
        categorical_columns=tuple(
            str(item) for item in payload["categorical_columns"]
        ),
        category_maps=MappingProxyType(
            {
                str(column): MappingProxyType(
                    {str(key): int(value) for key, value in mapping.items()}
                )
                for column, mapping in maps.items()
            }
        ),
        preprocessing=_preprocessing_from_payload(preprocessing_payload),
        trackman_result=trackman_result,
        trackman_lookup_sha256=trackman_hash,
    )


def _preprocessed_source(
    frame: pd.DataFrame,
    *,
    view: str,
    trackman_result: TrackmanBuildResult | None,
) -> pd.DataFrame:
    if view == "raw_typed":
        return frame.copy()
    if view != "raw_plus_trackman":
        raise FeatureContractError(f"unknown preprocessing feature view: {view}")
    if trackman_result is None:
        raise FeatureContractError("raw_plus_trackman requires a cutoff lookup")
    target = frame[TARGET_COLUMN] if TARGET_COLUMN in frame else None
    source = _attach_trackman(
        frame.drop(columns=[TARGET_COLUMN], errors="ignore"), trackman_result
    )
    if target is not None:
        source[TARGET_COLUMN] = target.to_numpy(copy=False)
    source.index = frame.index
    return source


def _preprocessed_batch(
    original: pd.DataFrame,
    prepared: pd.DataFrame,
    state: PreprocessedFeatureState,
) -> FeatureBatch:
    if ROW_ID_COLUMN not in original or original[ROW_ID_COLUMN].isna().any():
        raise FeatureContractError("row_id must be present and non-null")
    row_id = original[ROW_ID_COLUMN].astype(str).to_numpy(dtype=str)
    if len(set(row_id.tolist())) != len(row_id):
        raise FeatureContractError("row_id must be unique")
    x_num = prepared.loc[:, state.numeric_columns].to_numpy(dtype="float32")
    if not np.isfinite(x_num).all():
        raise FeatureContractError("DL numeric features must be finite")
    encoded = [
        _category_text(prepared[column])
        .map(state.category_maps[column])
        .fillna(0)
        .to_numpy(dtype="int64")
        for column in state.categorical_columns
    ]
    x_cat = (
        np.column_stack(encoded).astype("int64", copy=False)
        if encoded
        else np.empty((len(original), 0), dtype="int64")
    )
    target = None
    if TARGET_COLUMN in original:
        target = _numeric(original[TARGET_COLUMN], TARGET_COLUMN).to_numpy(
            dtype="float32"
        )
    season = _numeric(original["season"], "season").to_numpy(dtype="int64")
    game_type = (
        _category_text(original["game_type"]).to_numpy(dtype=str)
        if "game_type" in original
        else np.full(len(original), MISSING_CATEGORY, dtype=str)
    )
    return FeatureBatch(row_id, season, game_type, x_num, x_cat, target)


def materialize_preprocessed_fold_cache(
    cache_root: str | Path,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    history: pd.DataFrame,
    view: str,
    spec: PreprocessingSpec,
    train_end_year: int,
    valid_year: int,
) -> FoldCache:
    """Build or reuse a fold cache bound to an explicit preprocessing contract."""

    if spec.profile == "tree_native":
        raise FeatureContractError("tree_native does not produce a DL feature cache")
    if valid_year != train_end_year + 1:
        raise FeatureContractError("fold must be a yearly transition")
    if train["season"].gt(train_end_year).any():
        raise FeatureContractError("training rows exceed train_end_year")
    if not valid["season"].eq(valid_year).all():
        raise FeatureContractError("validation rows do not match valid_year")
    identifier = _preprocessing_id(spec)
    identity = {
        "view": view,
        "preprocessing_id": identifier,
        "train_end_year": train_end_year,
        "valid_year": valid_year,
        "train_row_sha256": _row_sha256(train),
        "valid_row_sha256": _row_sha256(valid),
        "train_frame_sha256": _frame_sha256(train),
        "valid_frame_sha256": _frame_sha256(valid),
        "history_frame_sha256": _frame_sha256(history),
        "feature_code_sha256": _code_sha256(),
        "preprocessing_code_sha256": _preprocessing_code_sha256(),
    }
    fold_root = (
        Path(cache_root).expanduser().resolve()
        / f"train_{train_end_year}_valid_{valid_year}"
        / view
    )
    target = fold_root / identifier
    if target.exists():
        saved = json.loads((target / "identity.json").read_text(encoding="utf-8"))
        if saved != identity:
            raise FeatureContractError("preprocessed cache identity differs")
        payload = json.loads((target / "state.json").read_text(encoding="utf-8"))
        state = _preprocessed_state_from_payload(target, payload)
        return FoldCache(
            target,
            _load_batch(target / "train"),
            _load_batch(target / "valid"),
            state,
            True,
        )

    fold_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{identifier}-", dir=fold_root))
    try:
        trackman_result = (
            build_trackman_lookup(train, history, train_end_year)
            if view == "raw_plus_trackman"
            else None
        )
        train_source = _preprocessed_source(
            train, view=view, trackman_result=trackman_result
        )
        valid_source = _preprocessed_source(
            valid, view=view, trackman_result=trackman_result
        )
        preprocessing_state, train_prepared = fit_preprocessor(train_source, spec)
        valid_prepared = transform_preprocessor(valid_source, preprocessing_state)
        categorical = preprocessing_state.categorical_columns
        numeric = preprocessing_state.numeric_columns
        category_maps = _fit_categories(train_prepared, categorical)
        lookup_hash = (
            _lookup_sha256(trackman_result.lookup)
            if trackman_result is not None
            else None
        )
        state = PreprocessedFeatureState(
            view,
            train_end_year,
            numeric,
            categorical,
            category_maps,
            preprocessing_state,
            trackman_result,
            lookup_hash,
        )
        _write_batch(
            temporary / "train", _preprocessed_batch(train, train_prepared, state)
        )
        _write_batch(
            temporary / "valid", _preprocessed_batch(valid, valid_prepared, state)
        )
        if trackman_result is not None:
            trackman_result.lookup.to_csv(temporary / "trackman_lookup.csv", index=False)
        _write_json(temporary / "state.json", _preprocessed_state_payload(state))
        _write_json(temporary / "identity.json", identity)
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    payload = json.loads((target / "state.json").read_text(encoding="utf-8"))
    return FoldCache(
        target,
        _load_batch(target / "train"),
        _load_batch(target / "valid"),
        _preprocessed_state_from_payload(target, payload),
        False,
    )
