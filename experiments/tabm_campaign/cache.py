from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.independent_dl.features import (
    FeatureBatch,
    PreprocessedFeatureState,
    materialize_preprocessed_fold_cache,
)
from experiments.independent_dl.preprocessing import PreprocessingSpec
from experiments.independent_dl.row_features import ROW_FEATURE_BUNDLES
from experiments.independent_dl.models.common import ModelMetadata


class CacheError(ValueError):
    """Raised when a campaign cache is incomplete or fails its hash binding."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _frame_sha256(frame: pd.DataFrame) -> str:
    digest = sha256()
    digest.update(_canonical_json([str(column) for column in frame.columns]))
    digest.update(
        pd.util.hash_pandas_object(frame, index=True, categorize=False)
        .to_numpy(dtype="uint64", copy=False)
        .tobytes()
    )
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _ids_sha256(values: Iterable[str]) -> str:
    return sha256(_canonical_json(list(values))).hexdigest()


@dataclass(frozen=True)
class CacheIdentity:
    train_frame_sha256: str
    valid_frame_sha256: str
    history_frame_sha256: str
    sample_ids_sha256: str
    train_end_year: int
    valid_year: int
    profile: str
    components: tuple[str, ...]
    preprocessing_code_sha256: str
    row_feature_code_sha256: str
    feature_code_sha256: str

    @classmethod
    def from_frames(
        cls,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        history: pd.DataFrame,
        *,
        train_end_year: int,
        valid_year: int,
        spec: PreprocessingSpec,
        sample_ids: Iterable[str],
    ) -> "CacheIdentity":
        package_root = Path(__file__).resolve().parents[1] / "independent_dl"
        ids = tuple(str(value) for value in sample_ids)
        return cls(
            train_frame_sha256=_frame_sha256(train),
            valid_frame_sha256=_frame_sha256(valid),
            history_frame_sha256=_frame_sha256(history),
            sample_ids_sha256=_ids_sha256(ids),
            train_end_year=int(train_end_year),
            valid_year=int(valid_year),
            profile=spec.profile,
            components=tuple(spec.components),
            preprocessing_code_sha256=_file_sha256(package_root / "preprocessing.py"),
            row_feature_code_sha256=_file_sha256(package_root / "row_features.py"),
            feature_code_sha256=_file_sha256(package_root / "features.py"),
        )

    def digest(self) -> str:
        return sha256(_canonical_json(asdict(self))).hexdigest()


@dataclass(frozen=True)
class FixedCache:
    root: Path
    train: FeatureBatch
    valid: FeatureBatch
    state: PreprocessedFeatureState
    reused: bool
    identity: CacheIdentity
    array_sha256: dict[str, str]
    model_metadata: ModelMetadata


def _batch_arrays(batch: FeatureBatch) -> dict[str, np.ndarray]:
    arrays = {
        "row_id": batch.row_id,
        "season": batch.season,
        "game_type": batch.game_type,
        "x_num": batch.x_num,
        "x_cat": batch.x_cat,
    }
    if batch.y is not None:
        arrays["y"] = batch.y
    return arrays


def _write_batch(root: Path, batch: FeatureBatch) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, values in _batch_arrays(batch).items():
        path = root / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        hashes[name] = _file_sha256(path)
    return hashes


def _load_batch(root: Path, expected: dict[str, str]) -> FeatureBatch:
    loaded: dict[str, np.ndarray] = {}
    for name, expected_hash in expected.items():
        path = root / f"{name}.npy"
        if not path.is_file() or _file_sha256(path) != expected_hash:
            raise CacheError(f"cached array SHA-256 differs: {name}")
        loaded[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    return FeatureBatch(
        loaded["row_id"],
        loaded["season"],
        loaded["game_type"],
        loaded["x_num"],
        loaded["x_cat"],
        loaded.get("y"),
    )


def _subset(batch: FeatureBatch, sample_ids: tuple[str, ...]) -> FeatureBatch:
    positions = {str(row_id): index for index, row_id in enumerate(batch.row_id.tolist())}
    if len(positions) != len(batch.row_id):
        raise CacheError("base cache row_id must be unique")
    missing = [row_id for row_id in sample_ids if row_id not in positions]
    if missing:
        raise CacheError(f"sample IDs are absent from the training fold: {missing[:3]}")
    index = np.asarray([positions[row_id] for row_id in sample_ids], dtype="int64")
    return FeatureBatch(
        batch.row_id[index],
        batch.season[index],
        batch.game_type[index],
        batch.x_num[index],
        batch.x_cat[index],
        None if batch.y is None else batch.y[index],
    )


def _category_maps_sha256(state: PreprocessedFeatureState) -> str:
    payload = {column: dict(mapping) for column, mapping in state.category_maps.items()}
    return sha256(_canonical_json(payload)).hexdigest()


def _model_metadata(batch: FeatureBatch, state: PreprocessedFeatureState) -> ModelMetadata:
    return ModelMetadata(
        n_num_features=batch.x_num.shape[1],
        categorical_cardinalities=tuple(
            len(state.category_maps[column]) + 1 for column in state.categorical_columns
        ),
        train_x_num=batch.x_num,
    )


def _validate_campaign_spec(spec: PreprocessingSpec) -> None:
    components = spec.components
    valid = (
        spec.profile == "dl_standard"
        and isinstance(components, tuple)
        and (
            components == ("hand_matchup",)
            or (
                len(components) == 2
                and components[0] == "hand_matchup"
                and components[1] in ROW_FEATURE_BUNDLES
            )
        )
    )
    if not valid:
        raise CacheError(
            "campaign cache requires dl_standard + hand_matchup and at most one "
            "sealed row feature bundle"
        )


def materialize_fixed_cache(
    cache_root: str | Path,
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    history: pd.DataFrame,
    train_end_year: int,
    valid_year: int,
    spec: PreprocessingSpec,
    sample_ids: Iterable[str],
) -> FixedCache:
    """Create a read-only sampled view over a full, fold-fitted preprocessing cache."""

    _validate_campaign_spec(spec)
    ids = tuple(str(value) for value in sample_ids)
    if not ids or len(set(ids)) != len(ids):
        raise CacheError("sample_ids must be non-empty and unique")
    identity = CacheIdentity.from_frames(
        train,
        valid,
        history,
        train_end_year=train_end_year,
        valid_year=valid_year,
        spec=spec,
        sample_ids=ids,
    )
    root = Path(cache_root).expanduser().resolve()
    base = materialize_preprocessed_fold_cache(
        root / "full_folds" / identity.row_feature_code_sha256,
        train,
        valid,
        history,
        "raw_typed",
        spec,
        train_end_year,
        valid_year,
    )
    target = root / "campaign_views" / identity.digest()
    identity_payload = json.loads(_canonical_json(asdict(identity)))
    state_sha = _category_maps_sha256(base.state)
    if target.exists():
        try:
            manifest = json.loads((target / "campaign_identity.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CacheError(f"cached campaign identity is invalid: {exc}") from exc
        if manifest.get("identity") != identity_payload or manifest.get("category_maps_sha256") != state_sha:
            raise CacheError("cached campaign identity differs")
        array_sha = manifest.get("array_sha256")
        if not isinstance(array_sha, dict) or set(array_sha) != {"train", "valid"}:
            raise CacheError("cached array SHA-256 manifest is invalid")
        train_batch = _load_batch(target / "train", array_sha["train"])
        valid_batch = _load_batch(target / "valid", array_sha["valid"])
        return FixedCache(
            target,
            train_batch,
            valid_batch,
            base.state,
            True,
            identity,
            array_sha,
            _model_metadata(base.train, base.state),
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{identity.digest()}-", dir=target.parent))
    try:
        train_batch = _subset(base.train, ids)
        array_sha = {
            "train": _write_batch(temporary / "train", train_batch),
            "valid": _write_batch(temporary / "valid", base.valid),
        }
        manifest = {
            "identity": identity_payload,
            "category_maps_sha256": state_sha,
            "array_sha256": array_sha,
        }
        (temporary / "campaign_identity.json").write_bytes(_canonical_json(manifest))
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return FixedCache(
        target,
        train_batch,
        base.valid,
        base.state,
        False,
        identity,
        array_sha,
        _model_metadata(base.train, base.state),
    )
