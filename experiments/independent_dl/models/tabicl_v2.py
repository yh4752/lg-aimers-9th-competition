"""Direct, research-only TabICLv2 inference for one temporal fold."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from ..features import FeatureBatch
from .common import import_runtime_module


class TabICLv2ContractError(ValueError):
    """Raised when TabICLv2 inputs or outputs cannot be trusted."""


@dataclass(frozen=True)
class TabICLv2Result:
    predictions: np.ndarray
    metadata_path: Path
    submission_eligibility: str = "research_only"


def _typed_frame(batch: FeatureBatch) -> object:
    pandas = import_runtime_module("pandas")
    numeric = np.asarray(batch.x_num)
    categorical = np.asarray(batch.x_cat)
    if numeric.ndim != 2 or categorical.ndim != 2:
        raise TabICLv2ContractError("TabICLv2 features must be two-dimensional")
    if len(numeric) != len(batch.row_id) or len(categorical) != len(batch.row_id):
        raise TabICLv2ContractError("TabICLv2 rows are not aligned")
    if not np.isfinite(numeric).all():
        raise TabICLv2ContractError("TabICLv2 numerical features must be finite")
    frame = pandas.DataFrame(
        numeric,
        columns=[f"num_{index}" for index in range(numeric.shape[1])],
        index=batch.row_id.astype(str),
    )
    for index in range(categorical.shape[1]):
        frame[f"cat_{index}"] = pandas.Categorical(categorical[:, index])
    return frame


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def fit_predict_tabicl_v2(
    train: FeatureBatch,
    valid: FeatureBatch,
    model_config: Mapping[str, object],
    output_dir: str | Path,
    *,
    seed: int,
    estimator_factory: Callable[..., object] | None = None,
    device: str | None = None,
) -> TabICLv2Result:
    if train.y is None or valid.y is None:
        raise TabICLv2ContractError("TabICLv2 train and validation targets are required")
    if estimator_factory is None:
        tabicl = import_runtime_module("tabicl")
        estimator_factory = tabicl.TabICLClassifier
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    estimator = estimator_factory(
        n_estimators=int(model_config["n_estimators"]),
        kv_cache=bool(model_config["kv_cache"]),
        offload_mode=str(model_config["offload_mode"]),
        checkpoint_version=str(model_config["checkpoint_version"]),
        device=device,
        use_amp="auto",
        random_state=seed,
        verbose=True,
    )
    train_frame = _typed_frame(train)
    valid_frame = _typed_frame(valid)
    estimator.fit(train_frame, np.asarray(train.y, dtype="int64"))
    classes = np.asarray(estimator.classes_)
    if classes.shape != (2,) or classes.tolist() != [0, 1]:
        raise TabICLv2ContractError("TabICLv2 classifier classes must be exactly [0, 1]")
    matrix = np.asarray(estimator.predict_proba(valid_frame), dtype="float64")
    if matrix.shape != (len(valid.row_id), 2) or not np.isfinite(matrix).all():
        raise TabICLv2ContractError(
            "TabICLv2 predict_proba must return a finite binary probability matrix"
        )
    predictions = matrix[:, 1]
    if ((predictions < 0) | (predictions > 1)).any():
        raise TabICLv2ContractError("TabICLv2 probabilities must be in [0, 1]")
    metadata_path = root / "tabicl_v2_metadata.json"
    _atomic_json(
        metadata_path,
        {
            "architecture": "tabicl_v2",
            "candidate_seed": seed,
            "checkpoint_version": str(model_config["checkpoint_version"]),
            "n_estimators": int(model_config["n_estimators"]),
            "submission_eligibility": "research_only",
        },
    )
    return TabICLv2Result(
        predictions=predictions,
        metadata_path=metadata_path,
    )
