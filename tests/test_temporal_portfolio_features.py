from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from experiments.temporal_portfolio.seasonal_features import (
    S1State,
    SeasonalFeatureError,
    fit_s1_state,
    transform_s1,
)


def test_training_s1_is_strictly_past_fitted_per_season() -> None:
    from experiments.temporal_portfolio.seasonal_features import build_training_s1

    train, _ = _fit_and_valid()
    state, first = build_training_s1(train, valid_year=2024)
    mutated = train.copy(deep=True)
    later = mutated["season"].eq(2023)
    mutated.loc[later, "control_success"] = 1 - mutated.loc[later, "control_success"]
    mutated.loc[later, "asof_pitcher_success_rate"] = 0.99
    replay_state, second = build_training_s1(mutated, valid_year=2024)

    early = train["season"].eq(2022).to_numpy()
    pd.testing.assert_frame_equal(first.loc[early], second.loc[early])
    assert state.valid_year == replay_state.valid_year == 2024
    assert state.snapshot.cutoff_year == replay_state.snapshot.cutoff_year == 2023


def test_training_s1_preserves_interleaved_rows_with_duplicate_index() -> None:
    from experiments.temporal_portfolio.seasonal_features import build_training_s1

    train, _ = _fit_and_valid()
    reordered = train.iloc[[1, 0, 2]].copy(deep=True)
    reordered.index = [7, 7, 2]

    _, result = build_training_s1(reordered, valid_year=2024)

    assert result.index.tolist() == [7, 7, 2]
    assert len(result) == len(reordered)
    assert result.iloc[0]["season_pitcher_n"] != result.iloc[1]["season_pitcher_n"]


def test_training_s1_rejects_single_season_without_full_prefix() -> None:
    from experiments.temporal_portfolio.seasonal_features import build_training_s1

    train, _ = _fit_and_valid()
    recent = train.loc[train["season"].eq(2023)]

    with pytest.raises(SeasonalFeatureError, match="full contiguous feature-fit prefix"):
        build_training_s1(recent, valid_year=2024)


def test_portfolio_composition_and_transform_are_row_stable(tmp_path) -> None:
    from experiments.temporal_portfolio.feature_cache import materialize_fold_cache
    from experiments.temporal_portfolio.features import (
        PortfolioFeatureSpec,
        fit_portfolio_features,
        transform_portfolio_features,
    )

    train, valid = _portfolio_fit_and_valid()
    valid = valid.assign(row_id=["v0", "v1", "v2"])
    spec = PortfolioFeatureSpec(("base", "S1"), "dl_standard")
    state, batch = fit_portfolio_features(
        train, pd.DataFrame(), spec=spec, valid_year=2024
    )
    assert state.spec == spec
    assert state.history_cutoff_year == 2023
    assert batch.row_id.tolist() == train["row_id"].astype(str).tolist()
    assert batch.x_num.dtype == np.float32
    assert batch.x_cat.dtype == np.int64
    assert np.isfinite(batch.x_num).all()
    assert state.preprocessing_state.spec.components == ("hand_matchup",)
    assert "hand_matchup" in state.schema
    assert "hand_matchup" in state.category_maps

    whole = transform_portfolio_features(valid, state)
    shuffled_rows = valid.iloc[[2, 0, 1]]
    shuffled = transform_portfolio_features(shuffled_rows, state)
    by_id = {str(row_id): row for row_id, row in zip(whole.row_id, whole.x_num)}
    for row_id, row in zip(shuffled.row_id, shuffled.x_num):
        np.testing.assert_array_equal(row, by_id[str(row_id)])

    cached = materialize_fold_cache(
        tmp_path / "cache",
        train=train,
        valid=valid,
        history=pd.DataFrame(),
        spec=spec,
        valid_year=2024,
    )
    assert cached.reused is False
    replay = materialize_fold_cache(
        tmp_path / "cache",
        train=train,
        valid=valid,
        history=pd.DataFrame(),
        spec=spec,
        valid_year=2024,
    )
    assert replay.reused is True
    assert replay.train.x_num.flags.writeable is False
    assert replay.valid.x_num.flags.writeable is False


def test_portfolio_rejects_invalid_spec_chronology_and_inference_target() -> None:
    from experiments.temporal_portfolio.features import (
        PortfolioFeatureError,
        PortfolioFeatureSpec,
        fit_portfolio_features,
        transform_portfolio_features,
    )

    train, valid = _portfolio_fit_and_valid()
    valid = valid.assign(row_id=["v0", "v1", "v2"])
    with pytest.raises(PortfolioFeatureError, match="base"):
        fit_portfolio_features(
            train,
            pd.DataFrame(),
            spec=PortfolioFeatureSpec(("S1",), "dl_standard"),
            valid_year=2024,
        )
    with pytest.raises(PortfolioFeatureError, match="tree_native"):
        fit_portfolio_features(
            train,
            pd.DataFrame(),
            spec=PortfolioFeatureSpec(("base",), "tree_native"),
            valid_year=2024,
        )
    with pytest.raises(PortfolioFeatureError, match="immediate previous season"):
        fit_portfolio_features(
            train.loc[train["season"].eq(2022)],
            pd.DataFrame(),
            spec=PortfolioFeatureSpec(("base",), "dl_standard"),
            valid_year=2024,
        )

    state, _ = fit_portfolio_features(
        train,
        pd.DataFrame(),
        spec=PortfolioFeatureSpec(("base",), "dl_standard"),
        valid_year=2024,
        inference_mode=True,
    )
    with pytest.raises(PortfolioFeatureError, match="target"):
        transform_portfolio_features(valid.assign(control_success=0), state)


def test_portfolio_spec_normalizes_exact_bundle_order_and_rejects_duplicates() -> None:
    from experiments.temporal_portfolio.features import (
        PortfolioFeatureError,
        PortfolioFeatureSpec,
        normalize_feature_spec,
    )

    normalized = normalize_feature_spec(
        PortfolioFeatureSpec(("base", "M1", "P2", "S1"), "dl_selective_transform")
    )
    assert normalized == PortfolioFeatureSpec(
        ("base", "S1", "P2", "M1"), "dl_selective_transform"
    )
    with pytest.raises(PortfolioFeatureError, match="unique"):
        normalize_feature_spec(
            PortfolioFeatureSpec(("base", "S1", "S1"), "dl_standard")
        )


def test_portfolio_evaluation_row_is_unchanged_by_other_row_mutation() -> None:
    from experiments.temporal_portfolio.features import (
        PortfolioFeatureSpec,
        fit_portfolio_features,
        transform_portfolio_features,
    )

    train, valid = _portfolio_fit_and_valid()
    valid = valid.assign(row_id=["v0", "v1", "v2"])
    state, _ = fit_portfolio_features(
        train,
        pd.DataFrame(),
        spec=PortfolioFeatureSpec(("base", "S1"), "dl_standard"),
        valid_year=2024,
    )
    expected = transform_portfolio_features(valid, state)
    changed = valid.copy(deep=True)
    column = changed.columns.get_loc("asof_pitcher_success_rate")
    changed.iloc[1:, column] = [0.01, 0.99]
    actual = transform_portfolio_features(changed, state)

    np.testing.assert_array_equal(actual.x_num[0], expected.x_num[0])
    np.testing.assert_array_equal(actual.x_cat[0], expected.x_cat[0])


def test_portfolio_b1_insufficient_mapping_is_explicitly_skippable(monkeypatch) -> None:
    import experiments.temporal_portfolio.features as feature_module
    from experiments.temporal_portfolio.features import (
        PortfolioFeatureError,
        PortfolioFeatureSpec,
        fit_portfolio_features,
    )
    from experiments.temporal_portfolio.trackman_batter import BatterTrackmanState

    train, _ = _portfolio_fit_and_valid()
    fake = object.__new__(BatterTrackmanState)
    object.__setattr__(fake, "status", "insufficient_mapping")
    monkeypatch.setattr(feature_module, "fit_batter_trackman", lambda *_a, **_k: fake)

    with pytest.raises(PortfolioFeatureError, match="skippable insufficient_mapping"):
        fit_portfolio_features(
            train,
            pd.DataFrame(),
            spec=PortfolioFeatureSpec(("base", "B1"), "dl_standard"),
            valid_year=2024,
        )


def test_recent_expert_s1_requires_and_uses_full_feature_fit_prefix(tmp_path) -> None:
    from experiments.temporal_portfolio.feature_cache import (
        PortfolioFeatureCacheError,
        materialize_fold_cache,
    )
    from experiments.temporal_portfolio.features import (
        PortfolioFeatureError,
        PortfolioFeatureSpec,
        fit_portfolio_features,
    )

    context, _ = _portfolio_fit_and_valid()
    recent = context.loc[context["season"].eq(2023)].copy(deep=True)
    spec = PortfolioFeatureSpec(("base", "S1"), "dl_standard")

    with pytest.raises(PortfolioFeatureError, match="full contiguous feature-fit prefix"):
        fit_portfolio_features(
            recent, pd.DataFrame(), spec=spec, valid_year=2024
        )

    state, batch = fit_portfolio_features(
        recent,
        pd.DataFrame(),
        spec=spec,
        valid_year=2024,
        feature_fit_rows=context,
    )
    assert batch.row_id.tolist() == recent["row_id"].astype(str).tolist()
    assert state.fitted_sources["S1"].snapshot.cutoff_year == 2023

    _, valid = _portfolio_fit_and_valid()
    valid = valid.assign(row_id=["v0", "v1", "v2"])
    cache = materialize_fold_cache(
        tmp_path / "cache",
        train=recent,
        valid=valid,
        history=pd.DataFrame(),
        spec=spec,
        valid_year=2024,
        feature_fit_rows=context,
    )
    assert cache.train.row_id.tolist() == recent["row_id"].astype(str).tolist()
    changed_context = context.copy(deep=True)
    changed_context.iloc[0, changed_context.columns.get_loc("control_success")] = 0
    with pytest.raises(PortfolioFeatureCacheError, match="identity"):
        materialize_fold_cache(
            tmp_path / "cache",
            train=recent,
            valid=valid,
            history=pd.DataFrame(),
            spec=spec,
            valid_year=2024,
            feature_fit_rows=changed_context,
        )


def test_portfolio_state_recomputes_and_binds_source_hashes() -> None:
    from experiments.temporal_portfolio.features import (
        PortfolioFeatureError,
        PortfolioFeatureSpec,
        PortfolioFeatureState,
        fit_portfolio_features,
        transform_portfolio_features,
    )

    train, valid = _portfolio_fit_and_valid()
    valid = valid.assign(row_id=["v0", "v1", "v2"])
    state, _ = fit_portfolio_features(
        train,
        pd.DataFrame(),
        spec=PortfolioFeatureSpec(("base", "S1"), "dl_standard"),
        valid_year=2024,
    )
    forged = object.__new__(PortfolioFeatureState)
    for name in PortfolioFeatureState.__dataclass_fields__:
        object.__setattr__(forged, name, object.__getattribute__(state, name))
    object.__setattr__(forged, "_source_hash_items", (("S1", "0" * 64),))

    with pytest.raises(PortfolioFeatureError, match="source hashes"):
        transform_portfolio_features(valid, forged)


@pytest.mark.parametrize("bad_season", [True, "2022", 2022.5])
def test_portfolio_rejects_non_numeric_or_non_integral_training_season(
    bad_season,
) -> None:
    from experiments.temporal_portfolio.features import (
        PortfolioFeatureError,
        PortfolioFeatureSpec,
        fit_portfolio_features,
    )

    train, _ = _portfolio_fit_and_valid()
    train["season"] = train["season"].astype(object)
    train.iloc[0, train.columns.get_loc("season")] = bad_season
    with pytest.raises(PortfolioFeatureError, match="season"):
        fit_portfolio_features(
            train,
            pd.DataFrame(),
            spec=PortfolioFeatureSpec(("base",), "dl_standard"),
            valid_year=2024,
        )


def test_portfolio_rejects_string_target_and_non_boolean_inference_flag() -> None:
    from experiments.temporal_portfolio.features import (
        PortfolioFeatureError,
        PortfolioFeatureSpec,
        fit_portfolio_features,
    )

    train, _ = _portfolio_fit_and_valid()
    string_target = train.copy(deep=True)
    string_target["control_success"] = string_target["control_success"].astype(str)
    with pytest.raises(PortfolioFeatureError, match="target"):
        fit_portfolio_features(
            string_target,
            pd.DataFrame(),
            spec=PortfolioFeatureSpec(("base",), "dl_standard"),
            valid_year=2024,
        )
    with pytest.raises(PortfolioFeatureError, match="inference_mode"):
        fit_portfolio_features(
            train,
            pd.DataFrame(),
            spec=PortfolioFeatureSpec(("base",), "dl_standard"),
            valid_year=2024,
            inference_mode="yes",  # type: ignore[arg-type]
        )


def test_portfolio_cache_detects_tampering_and_input_mismatch(tmp_path) -> None:
    from experiments.temporal_portfolio.feature_cache import (
        PortfolioFeatureCacheError,
        materialize_fold_cache,
    )
    from experiments.temporal_portfolio.features import PortfolioFeatureSpec

    train, valid = _portfolio_fit_and_valid()
    valid = valid.assign(row_id=["v0", "v1", "v2"])
    kwargs = dict(
        train=train,
        valid=valid,
        history=pd.DataFrame(),
        spec=PortfolioFeatureSpec(("base",), "dl_standard"),
        valid_year=2024,
    )
    cached = materialize_fold_cache(tmp_path / "cache", **kwargs)
    target = cached.root / "train" / "x_num.npy"
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(PortfolioFeatureCacheError, match="hash"):
        materialize_fold_cache(tmp_path / "cache", **kwargs)

    other = tmp_path / "other"
    materialize_fold_cache(other, **kwargs)
    changed = train.copy(deep=True)
    changed.loc[changed.index[0], "asof_pitcher_n"] += 1
    with pytest.raises(PortfolioFeatureCacheError, match="identity"):
        materialize_fold_cache(other, **{**kwargs, "train": changed})


def test_portfolio_cache_hashes_arrays_without_path_read_bytes(
    tmp_path, monkeypatch
) -> None:
    from pathlib import Path

    from experiments.temporal_portfolio.feature_cache import _file_sha256

    path = tmp_path / "array.npy"
    np.save(path, np.arange(1000, dtype="float32"), allow_pickle=False)

    def forbidden(_self):
        raise AssertionError("array hashing must stream")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert len(_file_sha256(path)) == 64


def test_portfolio_cache_rejects_hash_consistent_invalid_batch_shape(tmp_path) -> None:
    import hashlib
    import json

    from experiments.temporal_portfolio.feature_cache import (
        PortfolioFeatureCacheError,
        materialize_fold_cache,
    )
    from experiments.temporal_portfolio.features import PortfolioFeatureSpec

    train, valid = _portfolio_fit_and_valid()
    valid = valid.assign(row_id=["v0", "v1", "v2"])
    kwargs = dict(
        train=train,
        valid=valid,
        history=pd.DataFrame(),
        spec=PortfolioFeatureSpec(("base",), "dl_standard"),
        valid_year=2024,
    )
    cached = materialize_fold_cache(tmp_path / "cache", **kwargs)
    path = cached.root / "train" / "x_num.npy"
    np.save(path, np.zeros((1, 1), dtype="float32"), allow_pickle=False)
    manifest_path = cached.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["members"]["train/x_num.npy"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(PortfolioFeatureCacheError, match="shape|row count"):
        materialize_fold_cache(tmp_path / "cache", **kwargs)


def _s1_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["h0", "h1", "h2", "v0", "v1", "v1"],
            "season": [2022, 2023, 2023, 2024, 2024, 2024],
            "pitcher_id": [11, 11, 12, 11, 12, 999],
            "batter_id": [21, 21, 22, 21, 22, 999],
            "asof_pitcher_n": [8, 9, 4, 10, 5, 0],
            "asof_pitcher_success_rate": [0.50, 5 / 9, 0.50, 0.60, 0.60, 0.50],
            "asof_pitcher_reverse_rate": [0.10, 1 / 9, 0.25, 0.20, 0.20, 0.0],
            "asof_pitcher_middle_rate": [0.25, 2 / 9, 0.25, 0.30, 0.20, 0.0],
            "asof_pitcher_ball_rate": [0.40, 4 / 9, 0.50, 0.40, 0.40, 0.0],
            "asof_pitcher_strike_rate": [0.60, 5 / 9, 0.50, 0.60, 0.60, 0.0],
            "asof_pitcher_pitchmix_n": [8, 9, 4, 10, 5, 0],
            "asof_pitcher_fastball_rate": [0.50, 5 / 9, 0.50, 0.60, 0.40, 0.0],
            "asof_pitcher_breaking_rate": [0.25, 2 / 9, 0.25, 0.20, 0.40, 0.0],
            "asof_pitcher_offspeed_rate": [0.25, 2 / 9, 0.25, 0.20, 0.20, 0.0],
            "asof_pitcher_prev1_game_success_rate": [
                0.5,
                0.5,
                0.5,
                0.7,
                np.nan,
                np.nan,
            ],
            "asof_pitcher_prev3_game_success_rate": [0.5, 0.5, 0.5, 0.6, 0.5, np.nan],
            "asof_pitcher_prev5_game_success_rate": [0.5, 0.5, 0.5, 0.5, 0.4, np.nan],
            "asof_batter_n": [6, 7, 3, 8, 4, 0],
            "asof_batter_success_rate": [0.50, 4 / 7, 1 / 3, 0.625, 0.50, 0.50],
            "asof_batter_middle_rate": [1 / 6, 1 / 7, 1 / 3, 0.25, 0.25, 0.0],
            "control_success": [1, 0, 1, 1, 0, 1],
        },
        index=[30, 10, 20, 8, 8, 3],
    )


def _fit_and_valid() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _s1_frame()
    train = frame.loc[frame["season"].lt(2024)].copy(deep=True)
    valid = frame.loc[frame["season"].eq(2024)].drop(columns="control_success")
    return train, valid


def _portfolio_fit_and_valid() -> tuple[pd.DataFrame, pd.DataFrame]:
    train, valid = _fit_and_valid()
    train = train.assign(
        pitcher_hand=[1, 1, 2],
        batter_hand=[1, 1, 2],
        pitcher_team_id=[7, 7, 8],
        batter_team_id=[9, 9, 10],
    )
    valid = valid.assign(
        pitcher_hand=[1, 2, 1],
        batter_hand=[1, 1, 2],
        pitcher_team_id=[7, 8, 99],
        batter_team_id=[9, 10, 99],
    )
    return train, valid


def _by_row_position(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.reset_index(drop=True)


def _forge_state(state: S1State, **changes: object) -> S1State:
    forged = object.__new__(S1State)
    for name in S1State.__dataclass_fields__:
        object.__setattr__(forged, name, object.__getattribute__(state, name))
    for name, value in changes.items():
        object.__setattr__(forged, name, value)
    return forged


def test_s1_uses_only_the_previous_season_snapshot() -> None:
    train, valid = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)

    first = transform_s1(valid, state)
    mutated_source = _s1_frame()
    mutated_source.loc[mutated_source["season"].eq(2024), "control_success"] = [0, 1, 0]
    replay_train = mutated_source.loc[mutated_source["season"].lt(2024)]
    second = transform_s1(valid, fit_s1_state(replay_train, valid_year=2024))

    pd.testing.assert_frame_equal(first, second)
    assert state.snapshot.cutoff_year == 2023 == state.valid_year - 1
    assert {
        "season_pitcher_n",
        "season_batter_n",
        "season_vs_career_success",
    }.issubset(first)


def test_s1_transform_is_separable_for_single_shuffle_and_subset() -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)
    whole = transform_s1(rows, state)

    single = transform_s1(rows.iloc[[0]], state)
    pd.testing.assert_frame_equal(
        _by_row_position(whole.iloc[[0]]), _by_row_position(single)
    )

    order = [2, 0, 1]
    shuffled = transform_s1(rows.iloc[order], state)
    pd.testing.assert_frame_equal(
        _by_row_position(whole.iloc[order]), _by_row_position(shuffled)
    )

    subset = transform_s1(rows.iloc[[2, 1]], state)
    pd.testing.assert_frame_equal(
        _by_row_position(whole.iloc[[2, 1]]), _by_row_position(subset)
    )


def test_s1_preserves_duplicate_index_row_ids_order_count_and_caller() -> None:
    train, rows = _fit_and_valid()
    original = rows.copy(deep=True)

    result = transform_s1(rows, fit_s1_state(train, valid_year=2024))

    pd.testing.assert_frame_equal(rows, original)
    assert result.index.tolist() == [8, 8, 3]
    assert len(result) == len(rows)
    assert result.columns.is_unique
    assert not set(rows.columns) & set(result.columns)
    assert not any(column.startswith("snapshot_") for column in result)


def test_s1_output_has_only_finite_features_and_exact_categorical_metadata() -> None:
    train, rows = _fit_and_valid()

    result = transform_s1(rows, fit_s1_state(train, valid_year=2024))

    numeric = result.select_dtypes(include="number")
    assert np.isfinite(numeric.to_numpy()).all()
    assert result.attrs == {
        "categorical_columns": (
            "season_pitcher_n_bucket",
            "season_batter_n_bucket",
        )
    }
    assert type(result.attrs["categorical_columns"]) is tuple
    assert result.iloc[-1]["season_pitcher_n"] == pytest.approx(0.0)
    assert result.iloc[-1]["season_batter_n"] == pytest.approx(0.0)


def test_s1_fit_is_defensive_against_later_history_mutation() -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)
    expected = transform_s1(rows, state)

    train.loc[:, "control_success"] = 1 - train["control_success"]
    train.loc[:, "asof_pitcher_success_rate"] = 0.0
    exposed = state.snapshot
    exposed.pitcher.loc[:, "snapshot_pitcher_success_count"] = 9999.0
    exposed.batter.loc[:, "snapshot_batter_success_count"] = 9999.0

    pd.testing.assert_frame_equal(expected, transform_s1(rows, state))
    assert not state.snapshot.pitcher["snapshot_pitcher_success_count"].eq(9999.0).any()
    assert not state.snapshot.batter["snapshot_batter_success_count"].eq(9999.0).any()


def test_s1_state_is_frozen_and_transform_requires_exact_state_type() -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)

    with pytest.raises(FrozenInstanceError):
        state.valid_year = 2025  # type: ignore[misc]
    with pytest.raises(SeasonalFeatureError, match="state type"):
        transform_s1(rows, object())  # type: ignore[arg-type]

    class DerivedState(S1State):
        pass

    derived = object.__new__(DerivedState)
    with pytest.raises(SeasonalFeatureError, match="state type"):
        transform_s1(rows, derived)


@pytest.mark.parametrize(
    "valid_year", [True, 2024.0, np.int64(2024), 999, 10000, "2024"]
)
def test_s1_valid_year_must_be_an_exact_four_digit_int(valid_year: object) -> None:
    train, _ = _fit_and_valid()

    with pytest.raises(SeasonalFeatureError, match="valid_year"):
        fit_s1_state(train, valid_year=valid_year)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_season", [2023.5, np.nan, np.inf, True, "2023"])
def test_s1_fit_rejects_malformed_seasons(bad_season: object) -> None:
    train, _ = _fit_and_valid()
    train.iloc[0, train.columns.get_loc("season")] = bad_season

    with pytest.raises(SeasonalFeatureError, match="season"):
        fit_s1_state(train, valid_year=2024)


def test_s1_fit_rejects_validation_and_future_rows() -> None:
    frame = _s1_frame()

    with pytest.raises(SeasonalFeatureError, match="reach validation season"):
        fit_s1_state(frame, valid_year=2024)


def test_s1_fit_requires_rows_from_the_immediate_previous_season() -> None:
    train, _ = _fit_and_valid()
    earlier_only = train.loc[train["season"].eq(2022)]

    with pytest.raises(SeasonalFeatureError, match="immediate previous season"):
        fit_s1_state(earlier_only, valid_year=2024)


def test_s1_fit_does_not_require_each_entity_in_the_previous_season() -> None:
    train, _ = _fit_and_valid()
    train = train.copy(deep=True)
    train.loc[train["season"].eq(2022), ["pitcher_id", "batter_id"]] = [91, 92]

    state = fit_s1_state(train, valid_year=2024)

    assert state.snapshot.cutoff_year == 2023


@pytest.mark.parametrize("target", [np.nan, np.inf, -np.inf, -1, 0.5, 2, "1"])
def test_s1_fit_requires_a_finite_numeric_binary_target(target: object) -> None:
    train, _ = _fit_and_valid()
    train.iloc[0, train.columns.get_loc("control_success")] = target

    with pytest.raises(SeasonalFeatureError, match="target"):
        fit_s1_state(train, valid_year=2024)


def test_s1_transform_rejects_target_and_wrong_or_mixed_seasons() -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)

    with pytest.raises(SeasonalFeatureError, match="target"):
        transform_s1(rows.assign(control_success=0), state)
    with pytest.raises(SeasonalFeatureError, match="transform season"):
        transform_s1(rows.assign(season=[2024, 2023, 2024]), state)
    with pytest.raises(SeasonalFeatureError, match="season"):
        transform_s1(rows.assign(season=[2024, "2024", 2024]), state)


@pytest.mark.parametrize("operation", ["fit", "transform"])
def test_s1_requires_an_actual_dataframe_and_unique_columns(operation: str) -> None:
    train, rows = _fit_and_valid()
    if operation == "fit":
        function = lambda value: fit_s1_state(value, valid_year=2024)
        valid = train
    else:
        state = fit_s1_state(train, valid_year=2024)
        function = lambda value: transform_s1(value, state)
        valid = rows

    with pytest.raises(SeasonalFeatureError, match="DataFrame"):
        function(valid.to_dict("list"))
    duplicate = pd.concat([valid, valid.iloc[:, [-1]]], axis=1)
    duplicate.columns = [*valid.columns, valid.columns[-1]]
    with pytest.raises(SeasonalFeatureError, match="duplicate columns"):
        function(duplicate)


@pytest.mark.parametrize("operation", ["fit", "transform"])
def test_s1_reports_missing_required_schema(operation: str) -> None:
    train, rows = _fit_and_valid()
    if operation == "fit":
        with pytest.raises(SeasonalFeatureError, match="required columns.*pitcher_id"):
            fit_s1_state(train.drop(columns="pitcher_id"), valid_year=2024)
    else:
        state = fit_s1_state(train, valid_year=2024)
        with pytest.raises(
            SeasonalFeatureError, match="required columns.*asof_pitcher_n"
        ):
            transform_s1(rows.drop(columns="asof_pitcher_n"), state)


def test_s1_rejects_non_string_column_names_with_a_schema_error() -> None:
    train, rows = _fit_and_valid()
    rows = rows.copy(deep=True)
    rows[7] = "extra"

    with pytest.raises(SeasonalFeatureError, match="column names"):
        transform_s1(rows, fit_s1_state(train, valid_year=2024))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("pitcher_id", ["unhashable"]),
        ("asof_pitcher_n", "not-a-number"),
        ("asof_pitcher_success_rate", np.inf),
        ("asof_pitcher_success_rate", 0.5 + 0.1j),
    ],
)
def test_s1_transform_rejects_malformed_entity_and_numeric_values(
    column: str, value: object
) -> None:
    train, rows = _fit_and_valid()
    rows = rows.copy(deep=True)
    if column == "pitcher_id":
        rows[column] = rows[column].astype("object")
    rows.iat[0, rows.columns.get_loc(column)] = value

    with pytest.raises(SeasonalFeatureError, match=column):
        transform_s1(rows, fit_s1_state(train, valid_year=2024))


def test_s1_rejects_preexisting_generated_or_snapshot_columns() -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)

    with pytest.raises(SeasonalFeatureError, match="reserved"):
        transform_s1(rows.assign(season_pitcher_n=123), state)
    with pytest.raises(SeasonalFeatureError, match="reserved"):
        transform_s1(rows.assign(snapshot_pitcher_success_n=123), state)


def test_s1_ignores_unrelated_season_prefixed_columns() -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)

    expected = transform_s1(rows, state)
    actual = transform_s1(rows.assign(season_note=["a", "b", "c"]), state)

    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("valid_year", 2024.0),
        ("valid_year", 999),
        ("_cutoff_year", 2022),
        ("_cutoff_year", 2023.0),
        ("prior_rate", np.float64(0.5)),
        ("prior_rate", np.nan),
        ("prior_rate", -0.1),
        ("prior_rate", 1.1),
    ],
)
def test_s1_transform_rejects_forged_scalar_state_fields(
    field: str, value: object
) -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)

    with pytest.raises(SeasonalFeatureError, match="state"):
        transform_s1(rows, _forge_state(state, **{field: value}))


def test_s1_transform_normalizes_missing_state_fields() -> None:
    _, rows = _fit_and_valid()
    forged = object.__new__(S1State)

    with pytest.raises(SeasonalFeatureError, match="state"):
        transform_s1(rows, forged)


@pytest.mark.parametrize("side", ["pitcher", "batter"])
def test_s1_transform_rejects_forged_snapshot_schema(side: str) -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)
    field = f"_{side}_columns"
    columns = object.__getattribute__(state, field)

    with pytest.raises(SeasonalFeatureError, match="state"):
        transform_s1(rows, _forge_state(state, **{field: columns[:-1]}))


@pytest.mark.parametrize("side", ["pitcher", "batter"])
def test_s1_transform_rejects_forged_snapshot_row_shapes(side: str) -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)
    field = f"_{side}_rows"
    snapshot_rows = object.__getattribute__(state, field)
    malformed = (snapshot_rows[0][:-1], *snapshot_rows[1:])

    with pytest.raises(SeasonalFeatureError, match="state"):
        transform_s1(rows, _forge_state(state, **{field: malformed}))
    with pytest.raises(SeasonalFeatureError, match="state"):
        transform_s1(rows, _forge_state(state, **{field: list(snapshot_rows)}))
    with pytest.raises(SeasonalFeatureError, match="state"):
        transform_s1(
            rows,
            _forge_state(state, **{field: (list(snapshot_rows[0]),)}),
        )


@pytest.mark.parametrize("side", ["pitcher", "batter"])
def test_s1_transform_rejects_duplicate_or_malformed_snapshot_entities(
    side: str,
) -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)
    field = f"_{side}_rows"
    snapshot_rows = object.__getattribute__(state, field)

    with pytest.raises(SeasonalFeatureError, match="state"):
        transform_s1(
            rows,
            _forge_state(state, **{field: (*snapshot_rows, snapshot_rows[0])}),
        )
    for bad_entity in (None, ["unhashable"]):
        malformed = ((bad_entity, *snapshot_rows[0][1:]), *snapshot_rows[1:])
        with pytest.raises(SeasonalFeatureError, match="state"):
            transform_s1(rows, _forge_state(state, **{field: malformed}))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("snapshot_pitcher_success_n", -1.0),
        ("snapshot_pitcher_success_count", np.nan),
        ("snapshot_pitcher_success_count", "1"),
        ("snapshot_pitcher_success_count", 999.0),
    ],
)
def test_s1_transform_rejects_forged_snapshot_numeric_payloads(
    column: str, value: object
) -> None:
    train, rows = _fit_and_valid()
    state = fit_s1_state(train, valid_year=2024)
    snapshot_rows = object.__getattribute__(state, "_pitcher_rows")
    columns = object.__getattribute__(state, "_pitcher_columns")
    changed = list(snapshot_rows[0])
    changed[columns.index(column)] = value
    forged_rows = (tuple(changed), *snapshot_rows[1:])

    with pytest.raises(SeasonalFeatureError, match="state"):
        transform_s1(rows, _forge_state(state, _pitcher_rows=forged_rows))
