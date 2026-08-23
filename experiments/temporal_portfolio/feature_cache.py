"""Small, fail-closed fold cache for portfolio FeatureBatch objects."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from experiments.independent_dl.features import FeatureBatch
from experiments.independent_dl.preprocessing import PreprocessingSpec, PreprocessingState
from experiments.independent_dl.feature_sources.seasonal import SeasonalSnapshot
from experiments.temporal_portfolio.features import (
    PortfolioFeatureSpec,
    PortfolioFeatureState,
    fit_portfolio_features,
    normalize_feature_spec,
    transform_portfolio_features,
)
from experiments.temporal_portfolio.seasonal_features import S1State
from experiments.temporal_portfolio.trackman_batter import BatterTrackmanState
from experiments.temporal_portfolio.trackman_pitcher import PitcherTrackmanState


class PortfolioFeatureCacheError(ValueError):
    """Raised when an existing fold cache cannot be trusted or reused."""


@dataclass(frozen=True)
class PortfolioFoldCache:
    root: Path
    train: FeatureBatch
    valid: FeatureBatch
    state: PortfolioFeatureState
    reused: bool
    identity_sha256: str


_ARRAY_NAMES = ("row_id", "season", "game_type", "x_num", "x_cat", "y")


def materialize_fold_cache(
    cache_root: str | Path,
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    history: pd.DataFrame,
    spec: PortfolioFeatureSpec,
    valid_year: int,
    inference_mode: bool = False,
    feature_fit_rows: pd.DataFrame | None = None,
) -> PortfolioFoldCache:
    """Create or strictly verify one train/validation portfolio cache."""

    normalized = normalize_feature_spec(spec)
    if type(inference_mode) is not bool:
        raise PortfolioFeatureCacheError("inference_mode must be an exact bool")
    context = train if feature_fit_rows is None else feature_fit_rows
    raw_identity = _raw_identity(
        train,
        valid,
        history,
        context,
        normalized,
        valid_year,
        inference_mode,
    )
    root = _safe_cache_root(cache_root)
    identifier = "__".join((str(valid_year), normalized.profile, *normalized.bundles))
    target = root / identifier
    if target.exists() or target.is_symlink():
        return _load_cache(target, raw_identity=raw_identity, reused=True)

    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{identifier}-", dir=root))
    try:
        state, train_batch = fit_portfolio_features(
            train,
            history,
            spec=normalized,
            valid_year=valid_year,
            inference_mode=inference_mode,
            feature_fit_rows=feature_fit_rows,
        )
        valid_batch = transform_portfolio_features(valid, state)
        state_payload = _state_payload(state)
        state_bytes = _canonical_json(_json_safe(state_payload))
        state_sha = sha256(state_bytes).hexdigest()
        (temporary / "state.json").write_bytes(state_bytes)
        member_hashes: dict[str, str] = {"state.json": state_sha}
        member_hashes.update(_write_batch(temporary / "train", train_batch, "train"))
        member_hashes.update(_write_batch(temporary / "valid", valid_batch, "valid"))
        identity = {
            **raw_identity,
            "state_sha256": state_sha,
            "preprocessing_state_sha256": _payload_sha(state_payload["preprocessing"]),
            "category_maps_sha256": _payload_sha(state_payload["category_maps"]),
            "source_state_sha256": _payload_sha(state_payload["sources"]),
            "source_hashes": state_payload["source_hashes"],
        }
        identity_sha = _payload_sha(identity)
        manifest = {
            "schema_version": 1,
            "identity": identity,
            "identity_sha256": identity_sha,
            "members": member_hashes,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(_canonical_json(_json_safe(manifest)))
        _fsync_tree(temporary)
        os.replace(temporary, target)
        _fsync_directory(root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _load_cache(target, raw_identity=raw_identity, reused=False)


def _load_cache(
    target: Path, *, raw_identity: Mapping[str, object], reused: bool
) -> PortfolioFoldCache:
    _reject_unsafe_tree(target)
    try:
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PortfolioFeatureCacheError("existing cache is incomplete") from error
    if type(manifest) is not dict or manifest.get("schema_version") != 1:
        raise PortfolioFeatureCacheError("cache manifest is invalid")
    identity = manifest.get("identity")
    if type(identity) is not dict:
        raise PortfolioFeatureCacheError("cache identity is invalid")
    for key, expected in raw_identity.items():
        if identity.get(key) != expected:
            raise PortfolioFeatureCacheError("cache identity differs from current inputs")
    if manifest.get("identity_sha256") != _payload_sha(identity):
        raise PortfolioFeatureCacheError("cache identity hash differs")
    members = manifest.get("members")
    if type(members) is not dict:
        raise PortfolioFeatureCacheError("cache member hash manifest is invalid")
    actual = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file() and path.relative_to(target).as_posix() != "manifest.json"
    }
    if actual != set(members):
        raise PortfolioFeatureCacheError("cache members are incomplete or unexpected")
    for name, expected in members.items():
        path = target / name
        if type(expected) is not str or _file_sha256(path) != expected:
            raise PortfolioFeatureCacheError(f"cache member hash differs: {name}")
    state_bytes = (target / "state.json").read_bytes()
    if sha256(state_bytes).hexdigest() != identity.get("state_sha256"):
        raise PortfolioFeatureCacheError("cache state hash differs")
    try:
        payload = _json_restore(json.loads(state_bytes))
        state = _state_from_payload(payload)
    except Exception as error:
        raise PortfolioFeatureCacheError("cached feature state is invalid") from error
    if (
        _payload_sha(payload["preprocessing"]) != identity.get("preprocessing_state_sha256")
        or _payload_sha(payload["category_maps"]) != identity.get("category_maps_sha256")
        or _payload_sha(payload["sources"]) != identity.get("source_state_sha256")
        or payload["source_hashes"] != identity.get("source_hashes")
    ):
        raise PortfolioFeatureCacheError("cached state identity differs")
    train = _load_batch(target / "train", members, "train")
    valid = _load_batch(target / "valid", members, "valid")
    _validate_loaded_batch(train, state=state, label="train", target_required=True)
    _validate_loaded_batch(valid, state=state, label="valid", target_required=False)
    if _ids_sha256(train.row_id) != raw_identity["train_row_sha256"]:
        raise PortfolioFeatureCacheError("cached train row identity differs")
    if _ids_sha256(valid.row_id) != raw_identity["valid_row_sha256"]:
        raise PortfolioFeatureCacheError("cached valid row identity differs")
    return PortfolioFoldCache(
        target, train, valid, state, reused, str(manifest["identity_sha256"])
    )


def _raw_identity(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    history: pd.DataFrame,
    feature_fit_rows: pd.DataFrame,
    spec: PortfolioFeatureSpec,
    valid_year: int,
    inference_mode: bool,
) -> dict[str, object]:
    return {
        "train_row_sha256": _row_sha256(train),
        "valid_row_sha256": _row_sha256(valid),
        "train_frame_sha256": _frame_sha256(train),
        "valid_frame_sha256": _frame_sha256(valid),
        "history_frame_sha256": _frame_sha256(history),
        "feature_fit_row_sha256": _row_sha256(feature_fit_rows),
        "feature_fit_frame_sha256": _frame_sha256(feature_fit_rows),
        "spec": {"bundles": list(spec.bundles), "profile": spec.profile},
        "valid_year": valid_year,
        "history_cutoff_year": valid_year - 1,
        "inference_mode": inference_mode,
        "feature_code_sha256": _feature_code_sha256(),
    }


def _state_payload(state: PortfolioFeatureState) -> dict[str, object]:
    sources: dict[str, object] = {}
    for name, source in state.fitted_sources.items():
        if type(source) is S1State:
            sources[name] = {
                "type": "S1",
                "valid_year": source.valid_year,
                "prior_rate": source.prior_rate,
                "pitcher": _frame_payload(source.snapshot.pitcher),
                "batter": _frame_payload(source.snapshot.batter),
            }
        elif type(source) is PitcherTrackmanState:
            sources[name] = {
                "type": "pitcher",
                "cutoff_year": source.cutoff_year,
                "lookup": _frame_payload(source.lookup),
            }
        elif type(source) is BatterTrackmanState:
            sources[name] = {
                "type": "batter",
                "cutoff_year": source.cutoff_year,
                "mapping": _frame_payload(source.mapping),
                "exposure": _frame_payload(source.exposure),
            }
        else:
            raise PortfolioFeatureCacheError(f"unsupported source state: {name}")
    p = state.preprocessing_state
    return {
        "spec": {"bundles": list(state.spec.bundles), "profile": state.spec.profile},
        "valid_year": state.valid_year,
        "inference_mode": state.inference_mode,
        "preprocessing": {
            "spec": {"profile": p.spec.profile, "components": list(p.spec.components)},
            "source_columns": list(p.source_columns),
            "output_columns": list(p.output_columns),
            "categorical_columns": list(p.categorical_columns),
            "numeric_columns": list(p.numeric_columns),
            "numeric_median": dict(p.numeric_median),
            "numeric_mean": dict(p.numeric_mean),
            "numeric_std": dict(p.numeric_std),
            "yeo_johnson_lambda": dict(p.yeo_johnson_lambda),
            "entity_frequency": {name: dict(values) for name, values in p.entity_frequency.items()},
            "target_prior": p.target_prior,
        },
        "category_maps": {
            name: dict(values) for name, values in state.category_maps.items()
        },
        "sources": sources,
        "source_hashes": dict(state.source_hashes),
    }


def _state_from_payload(payload: Mapping[str, object]) -> PortfolioFeatureState:
    spec_payload = payload["spec"]
    spec = PortfolioFeatureSpec(tuple(spec_payload["bundles"]), spec_payload["profile"])
    p = payload["preprocessing"]
    preprocessing = PreprocessingState(
        PreprocessingSpec(p["spec"]["profile"], tuple(p["spec"]["components"])),
        tuple(p["source_columns"]),
        tuple(p["output_columns"]),
        tuple(p["categorical_columns"]),
        tuple(p["numeric_columns"]),
        MappingProxyType({str(k): float(v) for k, v in p["numeric_median"].items()}),
        MappingProxyType({str(k): float(v) for k, v in p["numeric_mean"].items()}),
        MappingProxyType({str(k): float(v) for k, v in p["numeric_std"].items()}),
        MappingProxyType({str(k): float(v) for k, v in p["yeo_johnson_lambda"].items()}),
        MappingProxyType(
            {
                str(name): MappingProxyType(
                    {str(k): int(v) for k, v in values.items()}
                )
                for name, values in p["entity_frequency"].items()
            }
        ),
        float(p["target_prior"]),
    )
    sources: dict[str, object] = {}
    for name, source in payload["sources"].items():
        kind = source["type"]
        if kind == "S1":
            snapshot = SeasonalSnapshot(
                int(source["valid_year"]) - 1,
                _frame_from_split(source["pitcher"]),
                _frame_from_split(source["batter"]),
            )
            sources[name] = S1State._from_snapshot(
                valid_year=int(source["valid_year"]),
                prior_rate=float(source["prior_rate"]),
                snapshot=snapshot,
            )
        elif kind == "pitcher":
            sources[name] = PitcherTrackmanState._from_lookup(
                cutoff_year=int(source["cutoff_year"]),
                lookup=_frame_from_split(source["lookup"]),
            )
        elif kind == "batter":
            sources[name] = BatterTrackmanState._from_frames(
                cutoff_year=int(source["cutoff_year"]),
                mapping=_frame_from_split(source["mapping"]),
                exposure=_frame_from_split(source["exposure"]),
            )
        else:
            raise PortfolioFeatureCacheError("cached source type is invalid")
    return PortfolioFeatureState._create(
        spec=spec,
        valid_year=int(payload["valid_year"]),
        preprocessing_state=preprocessing,
        category_maps={
            name: {
                str(k): int(v)
                for k, v in payload["category_maps"][name].items()
            }
            for name in preprocessing.categorical_columns
        },
        sources=sources,
        source_hashes={str(k): str(v) for k, v in payload["source_hashes"].items()},
        inference_mode=bool(payload["inference_mode"]),
    )


def _frame_payload(frame: pd.DataFrame) -> dict[str, object]:
    split = frame.to_dict("split")
    return {**split, "dtypes": [str(dtype) for dtype in frame.dtypes]}


def _frame_from_split(payload: Mapping[str, object]) -> pd.DataFrame:
    columns = payload["columns"]
    dtypes = payload.get("dtypes")
    if not isinstance(columns, list) or not isinstance(dtypes, list):
        raise PortfolioFeatureCacheError("cached frame schema is invalid")
    if len(columns) != len(dtypes):
        raise PortfolioFeatureCacheError("cached frame dtypes differ from schema")
    frame = pd.DataFrame(
        payload["data"], columns=columns, index=payload["index"]
    )
    try:
        for column, dtype in zip(columns, dtypes, strict=True):
            frame[column] = frame[column].astype(dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise PortfolioFeatureCacheError("cached frame dtype is invalid") from error
    return frame


def _write_batch(root: Path, batch: FeatureBatch, prefix: str) -> dict[str, str]:
    root.mkdir()
    result: dict[str, str] = {}
    arrays = {
        "row_id": np.asarray(batch.row_id, dtype=str),
        "season": batch.season,
        "game_type": np.asarray(batch.game_type, dtype=str),
        "x_num": batch.x_num,
        "x_cat": batch.x_cat,
    }
    if batch.y is not None:
        arrays["y"] = batch.y
    for name, values in arrays.items():
        path = root / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        result[f"{prefix}/{name}.npy"] = _file_sha256(path)
    return result


def _load_batch(root: Path, members: Mapping[str, str], prefix: str) -> FeatureBatch:
    loaded: dict[str, np.ndarray] = {}
    for name in _ARRAY_NAMES:
        key = f"{prefix}/{name}.npy"
        if key in members:
            loaded[name] = np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    required = {"row_id", "season", "game_type", "x_num", "x_cat"}
    if not required.issubset(loaded):
        raise PortfolioFeatureCacheError("cached batch arrays are incomplete")
    return FeatureBatch(
        loaded["row_id"], loaded["season"], loaded["game_type"],
        loaded["x_num"], loaded["x_cat"], loaded.get("y")
    )


def _validate_loaded_batch(
    batch: FeatureBatch,
    *,
    state: PortfolioFeatureState,
    label: str,
    target_required: bool,
) -> None:
    arrays = (batch.row_id, batch.season, batch.game_type)
    if any(values.ndim != 1 for values in arrays):
        raise PortfolioFeatureCacheError(f"cached {label} row arrays have invalid shape")
    count = len(batch.row_id)
    if count == 0 or any(len(values) != count for values in arrays[1:]):
        raise PortfolioFeatureCacheError(f"cached {label} row count differs")
    if batch.x_num.ndim != 2 or batch.x_num.shape != (count, len(state.numeric_columns)):
        raise PortfolioFeatureCacheError(f"cached {label} numeric shape differs")
    if batch.x_cat.ndim != 2 or batch.x_cat.shape != (count, len(state.categorical_columns)):
        raise PortfolioFeatureCacheError(f"cached {label} categorical shape differs")
    if batch.x_num.dtype != np.dtype("float32") or not np.isfinite(batch.x_num).all():
        raise PortfolioFeatureCacheError(f"cached {label} numeric dtype or values differ")
    if batch.x_cat.dtype != np.dtype("int64") or np.any(batch.x_cat < 0):
        raise PortfolioFeatureCacheError(f"cached {label} categorical dtype or values differ")
    if batch.season.dtype != np.dtype("int64"):
        raise PortfolioFeatureCacheError(f"cached {label} season dtype differs")
    if target_required and batch.y is None:
        raise PortfolioFeatureCacheError(f"cached {label} target is missing")
    if batch.y is not None:
        if batch.y.ndim != 1 or len(batch.y) != count or batch.y.dtype != np.dtype("float32"):
            raise PortfolioFeatureCacheError(f"cached {label} target shape or dtype differs")
        if not np.isin(batch.y, (0.0, 1.0)).all():
            raise PortfolioFeatureCacheError(f"cached {label} target values differ")
    ids = batch.row_id.astype(str).tolist()
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise PortfolioFeatureCacheError(f"cached {label} row_id values differ")


def _safe_cache_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    for parent in (current, *current.parents):
        if parent.is_symlink():
            raise PortfolioFeatureCacheError("cache path traverses a symlink")
    return path.absolute()


def _reject_unsafe_tree(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise PortfolioFeatureCacheError("cache target is not a regular directory")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PortfolioFeatureCacheError("cache contains a symlink")
        if path.is_file() and path.stat().st_nlink != 1:
            raise PortfolioFeatureCacheError("cache contains a hard-linked file")


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for path in sorted((item for item in root.rglob("*") if item.is_dir()), reverse=True):
        _fsync_directory(path)
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _row_sha256(frame: pd.DataFrame) -> str:
    if type(frame) is not pd.DataFrame or "row_id" not in frame:
        raise PortfolioFeatureCacheError("cache input requires row_id")
    return sha256(_canonical_json(frame["row_id"].astype(str).tolist())).hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    if type(frame) is not pd.DataFrame:
        raise PortfolioFeatureCacheError("cache input must be a DataFrame")
    digest = sha256(_canonical_json([str(column) for column in frame.columns]))
    digest.update(
        pd.util.hash_pandas_object(frame, index=True, categorize=False)
        .to_numpy(dtype="uint64", copy=False).tobytes()
    )
    return digest.hexdigest()


def _feature_code_sha256() -> str:
    root = Path(__file__).resolve().parent
    files = (
        root / "features.py",
        root / "feature_cache.py",
        root / "seasonal_features.py",
        root / "trackman_pitcher.py",
        root / "trackman_batter.py",
        Path(PreprocessingState.__module__.replace(".", "/") + ".py"),
        Path("experiments/independent_dl/feature_sources/seasonal.py"),
        Path("experiments/independent_dl/feature_sources/trackman.py"),
    )
    digest = sha256()
    project = Path(__file__).resolve().parents[2]
    for path in files:
        resolved = path if path.is_absolute() else project / path
        digest.update(resolved.relative_to(project).as_posix().encode())
        digest.update(resolved.read_bytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(values: np.ndarray) -> str:
    return sha256(_canonical_json(np.asarray(values).astype(str).tolist())).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha(value: object) -> str:
    return sha256(_canonical_json(_json_safe(value))).hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return {"__float__": "nan"}
        if math.isinf(number):
            return {"__float__": "inf" if number > 0 else "-inf"}
        return number
    if value is pd.NA:
        return {"__missing__": "pd.NA"}
    return value


def _json_restore(value: object) -> object:
    if isinstance(value, list):
        return [_json_restore(item) for item in value]
    if isinstance(value, dict):
        if value == {"__float__": "nan"}:
            return float("nan")
        if value == {"__float__": "inf"}:
            return float("inf")
        if value == {"__float__": "-inf"}:
            return float("-inf")
        if value == {"__missing__": "pd.NA"}:
            return pd.NA
        return {key: _json_restore(item) for key, item in value.items()}
    return value
