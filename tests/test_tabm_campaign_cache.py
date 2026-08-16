from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.independent_dl.preprocessing import PreprocessingSpec
from experiments.independent_dl import row_features
from experiments.independent_dl.row_features import ROW_FEATURE_BUNDLES
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
    changed_row_features = replace(identity, row_feature_code_sha256="0" * 64)
    assert identity.digest() != changed.digest()
    assert identity.digest() != changed_row_features.digest()
    assert identity.row_feature_code_sha256 == sha256(
        Path(row_features.__file__).read_bytes()
    ).hexdigest()


@pytest.mark.parametrize("bundle", (None, *ROW_FEATURE_BUNDLES))
def test_cache_accepts_baseline_or_exactly_one_row_feature_bundle(
    preprocessing_train: pd.DataFrame,
    preprocessing_valid: pd.DataFrame,
    preprocessing_history: pd.DataFrame,
    tmp_path: Path,
    bundle: str | None,
) -> None:
    components = ("hand_matchup",) if bundle is None else ("hand_matchup", bundle)

    cache = materialize_fixed_cache(
        tmp_path,
        train=preprocessing_train,
        valid=preprocessing_valid,
        history=preprocessing_history,
        train_end_year=2023,
        valid_year=2024,
        spec=PreprocessingSpec("dl_standard", components),
        sample_ids=(str(preprocessing_train.iloc[0]["row_id"]),),
    )

    assert cache.identity.components == components
    manifest = json.loads(
        (cache.root / "campaign_identity.json").read_text(encoding="utf-8")
    )
    assert (
        manifest["identity"]["row_feature_code_sha256"]
        == cache.identity.row_feature_code_sha256
    )


@pytest.mark.parametrize(
    "spec",
    (
        PreprocessingSpec("dl_standard", ()),
        PreprocessingSpec("tree_native", ("hand_matchup",)),
        PreprocessingSpec("dl_standard", ("hand_matchup", "unknown")),
        PreprocessingSpec(
            "dl_standard", ("hand_matchup", "count_context", "pressure_context")
        ),
        PreprocessingSpec(
            "dl_standard", ("hand_matchup", "count_context", "count_context")
        ),
        PreprocessingSpec("dl_standard", ("hand_matchup", "count_state")),
        PreprocessingSpec("dl_standard", ("count_context", "hand_matchup")),
    ),
)
def test_cache_rejects_every_non_campaign_preprocessing_spec(
    preprocessing_train: pd.DataFrame,
    preprocessing_valid: pd.DataFrame,
    preprocessing_history: pd.DataFrame,
    tmp_path: Path,
    spec: PreprocessingSpec,
) -> None:
    with pytest.raises(CacheError, match="campaign cache requires"):
        materialize_fixed_cache(
            tmp_path,
            train=preprocessing_train,
            valid=preprocessing_valid,
            history=preprocessing_history,
            train_end_year=2023,
            valid_year=2024,
            spec=spec,
            sample_ids=(str(preprocessing_train.iloc[0]["row_id"]),),
        )


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
