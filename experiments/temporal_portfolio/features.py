"""Cutoff-safe composition of temporal portfolio feature bundles."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np
import pandas as pd

from experiments.independent_dl.features import FeatureBatch, MISSING_CATEGORY
from experiments.independent_dl.preprocessing import (
    PreprocessingError,
    PreprocessingSpec,
    PreprocessingState,
    fit_preprocessor,
    transform_preprocessor,
)
from experiments.temporal_portfolio.seasonal_features import (
    S1State,
    SeasonalFeatureError,
    build_training_s1,
    transform_s1,
)
from experiments.temporal_portfolio.trackman_batter import (
    BatterTrackmanError,
    BatterTrackmanState,
    attach_batter_exposure,
    build_matchup_features,
    fit_batter_trackman,
)
from experiments.temporal_portfolio.trackman_pitcher import (
    PitcherTrackmanError,
    PitcherTrackmanState,
    fit_pitcher_trackman,
)


class PortfolioFeatureError(ValueError):
    """Raised when a feature state would cross a cutoff or be ambiguous."""


VALID_BUNDLES = ("base", "S1", "P0", "P1", "P2", "P3", "B1", "M1")
VALID_PROFILES = ("dl_standard", "dl_selective_transform")
_TARGET = "control_success"
_PITCHER_BUNDLES = frozenset(("P0", "P1", "P2", "P3"))


@dataclass(frozen=True)
class PortfolioFeatureSpec:
    bundles: tuple[str, ...]
    profile: str


@dataclass(frozen=True, init=False)
class PortfolioFeatureState:
    spec: PortfolioFeatureSpec
    valid_year: int
    history_cutoff_year: int
    preprocessing_state: PreprocessingState
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    schema: tuple[str, ...]
    inference_mode: bool
    _category_items: tuple[tuple[str, tuple[tuple[str, int], ...]], ...] = field(
        repr=False
    )
    _source_items: tuple[tuple[str, object], ...] = field(repr=False)
    _source_hash_items: tuple[tuple[str, str], ...] = field(repr=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("PortfolioFeatureState must be created by fit_portfolio_features")

    @classmethod
    def _create(
        cls,
        *,
        spec: PortfolioFeatureSpec,
        valid_year: int,
        preprocessing_state: PreprocessingState,
        category_maps: Mapping[str, Mapping[str, int]],
        sources: Mapping[str, object],
        source_hashes: Mapping[str, str],
        inference_mode: bool,
    ) -> "PortfolioFeatureState":
        if type(inference_mode) is not bool:
            raise PortfolioFeatureError("inference_mode must be an exact bool")
        state = object.__new__(cls)
        object.__setattr__(state, "spec", spec)
        object.__setattr__(state, "valid_year", valid_year)
        object.__setattr__(state, "history_cutoff_year", valid_year - 1)
        object.__setattr__(state, "preprocessing_state", preprocessing_state)
        object.__setattr__(state, "numeric_columns", preprocessing_state.numeric_columns)
        object.__setattr__(state, "categorical_columns", preprocessing_state.categorical_columns)
        object.__setattr__(state, "schema", preprocessing_state.output_columns)
        object.__setattr__(state, "inference_mode", inference_mode)
        object.__setattr__(
            state,
            "_category_items",
            tuple(
                (column, tuple(sorted(mapping.items())))
                for column, mapping in category_maps.items()
            ),
        )
        object.__setattr__(state, "_source_items", tuple(sources.items()))
        object.__setattr__(state, "_source_hash_items", tuple(source_hashes.items()))
        _validate_state(state)
        return state

    @property
    def category_maps(self) -> Mapping[str, Mapping[str, int]]:
        _validate_state(self)
        return MappingProxyType(
            {
                column: MappingProxyType(dict(items))
                for column, items in self._category_items
            }
        )

    @property
    def fitted_sources(self) -> Mapping[str, object]:
        _validate_state(self)
        return MappingProxyType(dict(self._source_items))

    @property
    def source_hashes(self) -> Mapping[str, str]:
        _validate_state(self)
        return MappingProxyType(dict(self._source_hash_items))


def normalize_feature_spec(spec: PortfolioFeatureSpec) -> PortfolioFeatureSpec:
    if type(spec) is not PortfolioFeatureSpec:
        raise PortfolioFeatureError("spec must be a PortfolioFeatureSpec")
    if type(spec.bundles) is not tuple or any(type(item) is not str for item in spec.bundles):
        raise PortfolioFeatureError("feature bundles must be a tuple of names")
    if not spec.bundles or spec.bundles[0] != "base":
        raise PortfolioFeatureError("base bundle must be present and first")
    if len(spec.bundles) != len(set(spec.bundles)):
        raise PortfolioFeatureError("feature bundles must be unique")
    unknown = sorted(set(spec.bundles).difference(VALID_BUNDLES))
    if unknown:
        raise PortfolioFeatureError(f"unknown feature bundles: {unknown}")
    if spec.profile == "tree_native":
        raise PortfolioFeatureError("tree_native cannot produce a FeatureBatch")
    if spec.profile not in VALID_PROFILES:
        raise PortfolioFeatureError(f"unsupported DL preprocessing profile: {spec.profile}")
    ordered = tuple(name for name in VALID_BUNDLES if name in spec.bundles)
    return PortfolioFeatureSpec(ordered, spec.profile)


def fit_portfolio_features(
    train: pd.DataFrame,
    history: pd.DataFrame,
    *,
    spec: PortfolioFeatureSpec,
    valid_year: int,
    inference_mode: bool = False,
    feature_fit_rows: pd.DataFrame | None = None,
) -> tuple[PortfolioFeatureState, FeatureBatch]:
    """Fit all state on training rows and return their encoded FeatureBatch."""

    normalized = normalize_feature_spec(spec)
    if type(inference_mode) is not bool:
        raise PortfolioFeatureError("inference_mode must be an exact bool")
    source = _validate_training_rows(train, valid_year=valid_year)
    context = (
        source
        if feature_fit_rows is None
        else _validate_training_rows(feature_fit_rows, valid_year=valid_year)
    )
    context_positions = _context_positions(source, context)
    if type(history) is not pd.DataFrame or history.columns.has_duplicates:
        raise PortfolioFeatureError("history must be a DataFrame with unique columns")
    sources: dict[str, object] = {}
    hashes: dict[str, str] = {}
    raw = source.copy(deep=True)
    try:
        if "S1" in normalized.bundles:
            s1_state, context_s1 = build_training_s1(
                context, valid_year=valid_year
            )
            training_s1 = context_s1.iloc[context_positions].copy(deep=True)
            training_s1.index = source.index
            raw = _append_new_columns(raw, training_s1, bundle="S1")
            sources["S1"] = s1_state
            hashes["S1"] = _s1_sha256(s1_state)

        pitcher_state: PitcherTrackmanState | None = None
        if _PITCHER_BUNDLES.intersection(normalized.bundles) or "M1" in normalized.bundles:
            pitcher_state = fit_pitcher_trackman(
                context, history, cutoff_year=valid_year - 1
            )
            sources["pitcher"] = pitcher_state
            hashes["pitcher"] = pitcher_state.lookup_sha256
            for bundle in normalized.bundles:
                if bundle in _PITCHER_BUNDLES:
                    raw = _attach_pitcher_bundle(raw, pitcher_state, bundle)

        batter_state: BatterTrackmanState | None = None
        if "B1" in normalized.bundles or "M1" in normalized.bundles:
            batter_state = fit_batter_trackman(
                context, history, cutoff_year=valid_year - 1
            )
            if batter_state.status == "insufficient_mapping":
                raise PortfolioFeatureError(
                    "skippable insufficient_mapping: B1/M1 coverage is below 0.30"
                )
            sources["batter"] = batter_state
            hashes["batter_mapping"] = batter_state.mapping_sha256
            hashes["batter_exposure"] = batter_state.exposure_sha256
            if "B1" in normalized.bundles:
                raw = attach_batter_exposure(raw, batter_state)
            if "M1" in normalized.bundles:
                if pitcher_state is None:
                    raise PortfolioFeatureError("M1 requires a fitted pitcher state")
                raw = _attach_matchup_only(raw, pitcher_state, batter_state)
    except PortfolioFeatureError:
        raise
    except (SeasonalFeatureError, PitcherTrackmanError, BatterTrackmanError) as error:
        raise PortfolioFeatureError(str(error)) from error

    try:
        preprocessing, prepared = fit_preprocessor(
            raw, PreprocessingSpec(normalized.profile, ("hand_matchup",))
        )
    except PreprocessingError as error:
        raise PortfolioFeatureError(str(error)) from error
    category_maps = _fit_categories(prepared, preprocessing.categorical_columns)
    state = PortfolioFeatureState._create(
        spec=normalized,
        valid_year=valid_year,
        preprocessing_state=preprocessing,
        category_maps=category_maps,
        sources=sources,
        source_hashes=hashes,
        inference_mode=inference_mode,
    )
    return state, _to_batch(source, prepared, state)


def transform_portfolio_features(
    rows: pd.DataFrame, state: PortfolioFeatureState
) -> FeatureBatch:
    """Transform validation/inference rows using frozen, row-local state only."""

    _validate_state(state)
    source = _validate_evaluation_rows(rows, state)
    raw = source.copy(deep=True)
    sources = state.fitted_sources
    try:
        if "S1" in state.spec.bundles:
            unlabeled = raw.drop(columns=_TARGET, errors="ignore")
            s1 = transform_s1(unlabeled, sources["S1"])
            raw = _append_new_columns(raw, s1, bundle="S1")
        pitcher = sources.get("pitcher")
        for bundle in state.spec.bundles:
            if bundle in _PITCHER_BUNDLES:
                raw = _attach_pitcher_bundle(raw, pitcher, bundle)
        batter = sources.get("batter")
        if "B1" in state.spec.bundles:
            raw = attach_batter_exposure(raw, batter)
        if "M1" in state.spec.bundles:
            raw = _attach_matchup_only(raw, pitcher, batter)
        prepared = transform_preprocessor(raw, state.preprocessing_state)
    except (
        KeyError,
        TypeError,
        PreprocessingError,
        SeasonalFeatureError,
        PitcherTrackmanError,
        BatterTrackmanError,
    ) as error:
        raise PortfolioFeatureError("frozen feature transform failed") from error
    return _to_batch(source, prepared, state)


def _append_new_columns(base: pd.DataFrame, added: pd.DataFrame, *, bundle: str) -> pd.DataFrame:
    if len(base) != len(added) or not base.index.equals(added.index):
        raise PortfolioFeatureError(f"{bundle} changed row count or index")
    collisions = sorted(set(base.columns).intersection(added.columns))
    if collisions:
        raise PortfolioFeatureError(f"{bundle} feature-column collisions: {collisions}")
    result = base.copy(deep=True)
    for column in added:
        result[column] = added[column].to_numpy(copy=True)
    return result


def _attach_pitcher_bundle(
    rows: pd.DataFrame, state: object, bundle: str
) -> pd.DataFrame:
    if type(state) is not PitcherTrackmanState:
        raise PortfolioFeatureError("pitcher source state is invalid")
    if "pitcher_id" not in rows:
        raise PortfolioFeatureError("pitcher bundle requires pitcher_id")
    lookup = state.bundles[bundle].set_index("pitcher_id")
    features = pd.DataFrame(index=rows.index)
    ids = rows["pitcher_id"]
    for column in lookup:
        features[column] = ids.map(lookup[column])
    if bundle == "P0":
        features["tm_history_n"] = features["tm_history_n"].fillna(0.0)
        features["tm_match_confidence"] = features["tm_match_confidence"].fillna(0.0)
        features["tm_match_accepted"] = features["tm_match_accepted"].fillna(0).astype("int8")
        features["tm_pitcher_mapping_missing"] = features["tm_match_accepted"].ne(1).astype("int8")
        recent = state.bundles["P1"].set_index("pitcher_id")
        recent_columns = [column for column in recent if column.startswith("tm_recent_")]
        if recent_columns:
            present = pd.DataFrame(
                {column: ids.map(recent[column]) for column in recent_columns},
                index=rows.index,
            ).notna().any(axis=1)
        else:
            present = pd.Series(False, index=rows.index)
        features["tm_pitcher_recent_history_missing"] = (~present).astype("int8")
    return _append_new_columns(rows, features, bundle=bundle)


def _attach_matchup_only(
    rows: pd.DataFrame, pitcher: object, batter: object
) -> pd.DataFrame:
    if type(pitcher) is not PitcherTrackmanState or type(batter) is not BatterTrackmanState:
        raise PortfolioFeatureError("M1 source states are invalid")
    full = build_matchup_features(rows, pitcher_state=pitcher, batter_state=batter)
    columns = tuple(column for column in full if column.startswith("tm_matchup_"))
    return _append_new_columns(rows, full.loc[:, columns], bundle="M1")


def _fit_categories(
    prepared: pd.DataFrame, columns: tuple[str, ...]
) -> Mapping[str, Mapping[str, int]]:
    result: dict[str, Mapping[str, int]] = {}
    for column in columns:
        values = prepared[column].astype("string").fillna(MISSING_CATEGORY)
        unique = sorted(set(values.astype(str).tolist()))
        result[column] = MappingProxyType(
            {value: index + 1 for index, value in enumerate(unique)}
        )
    return MappingProxyType(result)


def _to_batch(
    original: pd.DataFrame, prepared: pd.DataFrame, state: PortfolioFeatureState
) -> FeatureBatch:
    if len(original) != len(prepared) or not original.index.equals(prepared.index):
        raise PortfolioFeatureError("preprocessing changed row identity")
    x_num = prepared.loc[:, state.numeric_columns].to_numpy(dtype="float32")
    if not np.isfinite(x_num).all():
        raise PortfolioFeatureError("DL numeric features must be finite")
    encoded = []
    for column in state.categorical_columns:
        values = prepared[column].astype("string").fillna(MISSING_CATEGORY).astype(str)
        encoded.append(
            values.map(state.category_maps[column]).fillna(0).to_numpy(dtype="int64")
        )
    x_cat = (
        np.column_stack(encoded).astype("int64", copy=False)
        if encoded
        else np.empty((len(original), 0), dtype="int64")
    )
    y = None
    if _TARGET in original:
        y = pd.to_numeric(original[_TARGET], errors="raise").to_numpy(dtype="float32")
        if not np.isin(y, (0.0, 1.0)).all():
            raise PortfolioFeatureError("target must be binary")
    game_type = (
        original["game_type"]
        .astype("string")
        .fillna(MISSING_CATEGORY)
        .astype(str)
        .to_numpy(dtype=str)
        if "game_type" in original
        else np.full(len(original), MISSING_CATEGORY, dtype=str)
    )
    return FeatureBatch(
        original["row_id"].astype(str).to_numpy(dtype=str),
        original["season"].to_numpy(dtype="int64"),
        game_type,
        x_num,
        x_cat,
        y,
    )


def _validate_training_rows(frame: object, *, valid_year: int) -> pd.DataFrame:
    result = _validate_common_rows(frame, label="training")
    if type(valid_year) is not int or not 1000 <= valid_year <= 9999:
        raise PortfolioFeatureError("valid_year must be an exact four-digit int")
    seasons = result["season"].to_numpy(dtype="int64")
    if np.any(seasons >= valid_year):
        raise PortfolioFeatureError("training seasons must be strictly before valid_year")
    if int(seasons.max()) != valid_year - 1:
        raise PortfolioFeatureError("training must include the immediate previous season")
    if _TARGET not in result:
        raise PortfolioFeatureError("training requires target")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (Integral, Real, np.integer, np.floating))
        for value in result[_TARGET].tolist()
    ):
        raise PortfolioFeatureError("target must be finite numeric binary")
    target = pd.to_numeric(result[_TARGET], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(target).all() or not np.isin(target, (0.0, 1.0)).all():
        raise PortfolioFeatureError("target must be finite numeric binary")
    return result


def _validate_evaluation_rows(
    frame: object, state: PortfolioFeatureState
) -> pd.DataFrame:
    result = _validate_common_rows(frame, label="evaluation")
    if not result["season"].eq(state.valid_year).all():
        raise PortfolioFeatureError("evaluation season differs from state valid_year")
    if state.inference_mode and _TARGET in result:
        raise PortfolioFeatureError("inference evaluation rows contain target")
    if _TARGET in result:
        target = pd.to_numeric(result[_TARGET], errors="coerce").to_numpy(dtype="float64")
        if not np.isfinite(target).all() or not np.isin(target, (0.0, 1.0)).all():
            raise PortfolioFeatureError("evaluation target must be finite numeric binary")
    return result


def _validate_common_rows(frame: object, *, label: str) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame or frame.columns.has_duplicates:
        raise PortfolioFeatureError(f"{label} rows must be a DataFrame with unique columns")
    if frame.empty:
        raise PortfolioFeatureError(f"{label} rows are empty")
    missing = sorted({"row_id", "season"}.difference(frame.columns))
    if missing:
        raise PortfolioFeatureError(f"{label} rows missing required columns: {missing}")
    ids = frame["row_id"]
    if ids.isna().any() or ids.astype(str).duplicated().any():
        raise PortfolioFeatureError(f"{label} row_id must be non-null and unique")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (Integral, Real, np.integer, np.floating))
        for value in frame["season"].tolist()
    ):
        raise PortfolioFeatureError(f"{label} season must contain finite integer years")
    numeric = pd.to_numeric(frame["season"], errors="coerce")
    if (
        numeric.isna().any()
        or not np.isfinite(numeric).all()
        or not np.equal(numeric, np.floor(numeric)).all()
    ):
        raise PortfolioFeatureError(f"{label} season must contain finite integer years")
    result = frame.copy(deep=True)
    result["season"] = numeric.astype("int64")
    return result


def _context_positions(train: pd.DataFrame, context: pd.DataFrame) -> np.ndarray:
    if tuple(train.columns) != tuple(context.columns):
        raise PortfolioFeatureError(
            "expert train schema differs from the feature-fit context"
        )
    context_ids = context["row_id"].astype(str).tolist()
    positions = {row_id: position for position, row_id in enumerate(context_ids)}
    train_ids = train["row_id"].astype(str).tolist()
    missing = [row_id for row_id in train_ids if row_id not in positions]
    if missing:
        raise PortfolioFeatureError(
            "expert train rows are not an exact subset of feature-fit context"
        )
    selected_positions = np.asarray(
        [positions[row_id] for row_id in train_ids], dtype="int64"
    )
    if not context.iloc[selected_positions].reset_index(drop=True).equals(
        train.reset_index(drop=True)
    ):
        raise PortfolioFeatureError(
            "expert train rows differ from feature-fit context values"
        )
    return selected_positions


def _s1_sha256(state: S1State) -> str:
    payload = {
        "valid_year": state.valid_year,
        "prior_rate": state.prior_rate,
        "pitcher": state.snapshot.pitcher.to_dict("split"),
        "batter": state.snapshot.batter.to_dict("split"),
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _validate_state(state: object) -> None:
    if type(state) is not PortfolioFeatureState:
        raise PortfolioFeatureError("portfolio feature state type is invalid")
    if type(state.inference_mode) is not bool:
        raise PortfolioFeatureError("portfolio inference_mode is invalid")
    if state.history_cutoff_year != state.valid_year - 1:
        raise PortfolioFeatureError("portfolio feature state cutoff is invalid")
    if normalize_feature_spec(state.spec) != state.spec:
        raise PortfolioFeatureError("portfolio feature state spec is not normalized")
    if state.schema != state.preprocessing_state.output_columns:
        raise PortfolioFeatureError("portfolio feature state schema differs")
    expected_preprocessing = PreprocessingSpec(
        state.spec.profile, ("hand_matchup",)
    )
    if state.preprocessing_state.spec != expected_preprocessing:
        raise PortfolioFeatureError("portfolio preprocessing contract differs")
    if (
        "hand_matchup" not in state.schema
        or "hand_matchup" not in state.categorical_columns
    ):
        raise PortfolioFeatureError("portfolio hand_matchup schema is missing")
    if (
        state.numeric_columns != state.preprocessing_state.numeric_columns
        or state.categorical_columns != state.preprocessing_state.categorical_columns
    ):
        raise PortfolioFeatureError("portfolio feature state metadata differs")
    categories = dict(state._category_items)
    if tuple(categories) != state.categorical_columns:
        raise PortfolioFeatureError("portfolio category maps differ from schema")
    for column, items in categories.items():
        values = [value for _, value in items]
        if any(type(key) is not str for key, _ in items) or values != list(
            range(1, len(values) + 1)
        ):
            raise PortfolioFeatureError(f"portfolio category map is invalid: {column}")
    if (
        type(state._source_items) is not tuple
        or type(state._source_hash_items) is not tuple
        or len(dict(state._source_items)) != len(state._source_items)
        or len(dict(state._source_hash_items)) != len(state._source_hash_items)
    ):
        raise PortfolioFeatureError("portfolio source hashes are invalid")
    sources = dict(state._source_items)
    expected_sources: set[str] = set()
    if "S1" in state.spec.bundles:
        expected_sources.add("S1")
    if _PITCHER_BUNDLES.intersection(state.spec.bundles) or "M1" in state.spec.bundles:
        expected_sources.add("pitcher")
    if "B1" in state.spec.bundles or "M1" in state.spec.bundles:
        expected_sources.add("batter")
    if set(sources) != expected_sources:
        raise PortfolioFeatureError("portfolio source state keys differ from spec")
    try:
        observed = _recomputed_source_hashes(sources)
    except (SeasonalFeatureError, PitcherTrackmanError, BatterTrackmanError) as error:
        raise PortfolioFeatureError("portfolio source states are invalid") from error
    if dict(state._source_hash_items) != observed:
        raise PortfolioFeatureError("portfolio source hashes differ from fitted states")


def _recomputed_source_hashes(sources: Mapping[str, object]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    if "S1" in sources:
        if type(sources["S1"]) is not S1State:
            raise PortfolioFeatureError("S1 source state is invalid")
        hashes["S1"] = _s1_sha256(sources["S1"])
    if "pitcher" in sources:
        if type(sources["pitcher"]) is not PitcherTrackmanState:
            raise PortfolioFeatureError("pitcher source state is invalid")
        hashes["pitcher"] = sources["pitcher"].lookup_sha256
    if "batter" in sources:
        if type(sources["batter"]) is not BatterTrackmanState:
            raise PortfolioFeatureError("batter source state is invalid")
        hashes["batter_mapping"] = sources["batter"].mapping_sha256
        hashes["batter_exposure"] = sources["batter"].exposure_sha256
    if set(sources).difference(("S1", "pitcher", "batter")):
        raise PortfolioFeatureError("portfolio source state keys are invalid")
    return hashes
