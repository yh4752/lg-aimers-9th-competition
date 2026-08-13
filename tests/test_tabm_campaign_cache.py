from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from experiments.independent_dl.preprocessing import PreprocessingSpec
from experiments.tabm_campaign.cache import CacheError, CacheIdentity, materialize_fixed_cache


def test_cache_key_changes_for_any_semantic_input(preprocessing_train: pd.DataFrame) -> None:
    identity = CacheIdentity.from_frames(
        preprocessing_train,
        preprocessing_train.iloc[:2],
        preprocessing_train,
        train_end_year=2023,
        valid_year=2024,
        spec=PreprocessingSpec("dl_standard", ("hand_matchup",)),
        sample_ids=(str(preprocessing_train.iloc[0]["row_id"]),),
    )
    changed = replace(identity, preprocessing_code_sha256="0" * 64)
    assert identity.digest() != changed.digest()


def test_cache_reuse_is_read_only_and_hash_checked(
    preprocessing_train: pd.DataFrame,
    preprocessing_valid: pd.DataFrame,
    preprocessing_history: pd.DataFrame,
    tmp_path: Path,
) -> None:
    sample_ids = tuple(preprocessing_train["row_id"].astype(str).iloc[::2])
    kwargs = dict(
        train=preprocessing_train,
        valid=preprocessing_valid,
        history=preprocessing_history,
        train_end_year=2023,
        valid_year=2024,
        spec=PreprocessingSpec("dl_standard", ("hand_matchup",)),
        sample_ids=sample_ids,
    )
    first = materialize_fixed_cache(tmp_path, **kwargs)
    second = materialize_fixed_cache(tmp_path, **kwargs)

    assert first.reused is False
    assert second.reused is True
    assert first.array_sha256 == second.array_sha256
    assert first.train.row_id.tolist() == list(sample_ids)
    assert len(first.model_metadata.train_x_num) == len(preprocessing_train)
    assert first.model_metadata.n_num_features == first.train.x_num.shape[1]

    identity_path = first.root / "campaign_identity.json"
    identity_path.write_text(identity_path.read_text().replace("sha256", "tampered", 1))
    with pytest.raises(CacheError):
        materialize_fixed_cache(tmp_path, **kwargs)
