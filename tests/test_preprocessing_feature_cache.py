from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.independent_dl.features import (
    materialize_fold_cache,
    materialize_preprocessed_fold_cache,
)
from experiments.independent_dl.preprocessing import PreprocessingSpec


def test_legacy_feature_cache_still_uses_the_existing_api(
    preprocessing_train: pd.DataFrame,
    preprocessing_valid: pd.DataFrame,
    preprocessing_history: pd.DataFrame,
    tmp_path: Path,
) -> None:
    cache = materialize_fold_cache(
        tmp_path / "legacy",
        preprocessing_train,
        preprocessing_valid,
        preprocessing_history,
        "raw_typed",
        2023,
        2024,
    )

    assert cache.train.x_num.shape[0] == len(preprocessing_train)


def test_preprocessing_identity_changes_cache_namespace(
    preprocessing_train: pd.DataFrame,
    preprocessing_valid: pd.DataFrame,
    preprocessing_history: pd.DataFrame,
    tmp_path: Path,
) -> None:
    base = materialize_preprocessed_fold_cache(
        tmp_path,
        preprocessing_train,
        preprocessing_valid,
        preprocessing_history,
        "raw_typed",
        PreprocessingSpec("dl_standard", ()),
        2023,
        2024,
    )
    smooth = materialize_preprocessed_fold_cache(
        tmp_path,
        preprocessing_train,
        preprocessing_valid,
        preprocessing_history,
        "raw_typed",
        PreprocessingSpec("dl_standard", ("pitcher_smooth_k100",)),
        2023,
        2024,
    )

    assert base.root != smooth.root
    assert base.state.preprocessing.spec.components == ()
    assert smooth.state.preprocessing.spec.components == ("pitcher_smooth_k100",)


def test_raw_plus_trackman_adds_lookup_without_engineered_context(
    preprocessing_train: pd.DataFrame,
    preprocessing_valid: pd.DataFrame,
    preprocessing_history: pd.DataFrame,
    tmp_path: Path,
) -> None:
    cache = materialize_preprocessed_fold_cache(
        tmp_path,
        preprocessing_train,
        preprocessing_valid,
        preprocessing_history,
        "raw_plus_trackman",
        PreprocessingSpec("dl_standard", ()),
        2023,
        2024,
    )

    assert any(name.startswith("tm_") for name in cache.state.numeric_columns)
    assert not any(
        name.startswith("ctx_") for name in cache.state.categorical_columns
    )


def test_preprocessed_cache_reuses_only_exact_contract(
    preprocessing_train: pd.DataFrame,
    preprocessing_valid: pd.DataFrame,
    preprocessing_history: pd.DataFrame,
    tmp_path: Path,
) -> None:
    first = materialize_preprocessed_fold_cache(
        tmp_path,
        preprocessing_train,
        preprocessing_valid,
        preprocessing_history,
        "raw_typed",
        PreprocessingSpec("dl_standard", ("count_state",)),
        2023,
        2024,
    )
    second = materialize_preprocessed_fold_cache(
        tmp_path,
        preprocessing_train,
        preprocessing_valid,
        preprocessing_history,
        "raw_typed",
        PreprocessingSpec("dl_standard", ("count_state",)),
        2023,
        2024,
    )

    assert first.reused is False
    assert second.reused is True
    assert first.train.row_id.tolist() == preprocessing_train["row_id"].tolist()
    assert second.valid.row_id.tolist() == preprocessing_valid["row_id"].tolist()
