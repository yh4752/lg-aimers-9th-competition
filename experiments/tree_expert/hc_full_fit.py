from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
import tempfile
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .hc_contracts import HCContract, load_hc_contract


class HCFullFitError(ValueError):
    """Raised when H3 full fitting is attempted without accepted evidence."""


@dataclass(frozen=True)
class HCFullFitToken:
    winner: str
    profile_name: str
    calibration_alpha: float | None
    seeds: tuple[int, ...]
    iterations: Mapping[int, int]


@dataclass(frozen=True)
class HCFullFitModel:
    seed: int
    iterations: int
    model_path: Path
    model_sha256: str


ModelFactory = Callable[[dict[str, object]], object]


def _atomic_bytes(path: Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def full_fit_iterations(
    best_iterations: Sequence[int], contract: HCContract | None = None
) -> int:
    active = load_hc_contract() if contract is None else contract
    values = tuple(best_iterations)
    if len(values) != 3 or any(
        type(value) is not int or type(value) is bool or value < 0 for value in values
    ):
        raise HCFullFitError("three non-negative best iterations are required")
    selected = int(statistics.median(values)) + 1
    return min(
        int(active.full_fit_iterations["maximum"]),
        max(int(active.full_fit_iterations["minimum"]), selected),
    )


def accepted_full_fit_token(
    winner: Mapping[str, object],
    *,
    profile_name: str,
    calibration_alpha: float | None,
    best_iterations: Mapping[int, Sequence[int]],
    contract: HCContract | None = None,
) -> HCFullFitToken:
    active = load_hc_contract() if contract is None else contract
    if (
        not isinstance(winner, Mapping)
        or winner.get("status") != "accepted"
        or winner.get("candidate") not in {"C1", "C2"}
    ):
        raise HCFullFitError("full fit requires an accepted C1 or C2 winner")
    candidate = str(winner["candidate"])
    if profile_name not in active.profiles:
        raise HCFullFitError("selected profile differs")
    if candidate == "C1" and calibration_alpha is not None:
        raise HCFullFitError("C1 token contains calibration")
    if candidate == "C2" and calibration_alpha not in active.calibration_alphas:
        raise HCFullFitError("C2 calibration alpha differs")
    if set(best_iterations) != set(active.seeds):
        raise HCFullFitError("full-fit seed evidence differs")
    iterations = {
        seed: full_fit_iterations(best_iterations[seed], active)
        for seed in active.seeds
    }
    return HCFullFitToken(
        winner=candidate,
        profile_name=profile_name,
        calibration_alpha=calibration_alpha,
        seeds=active.seeds,
        iterations=MappingProxyType(iterations),
    )


def _default_model_factory(parameters: dict[str, object]) -> object:
    try:
        from catboost import CatBoostRegressor
    except ImportError as error:
        raise HCFullFitError("catboost==1.2.10 is required") from error
    return CatBoostRegressor(**parameters)


def fit_full_c1_seed(
    *,
    token: HCFullFitToken,
    seed: int,
    oof_rows: pd.DataFrame,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
    output_dir: Path,
    gpu_id: int,
    contract: HCContract | None = None,
    model_factory: ModelFactory = _default_model_factory,
) -> HCFullFitModel:
    active = load_hc_contract() if contract is None else contract
    if type(token) is not HCFullFitToken or seed not in token.seeds:
        raise HCFullFitError("full-fit token or seed differs")
    features = tuple(feature_columns)
    categorical = tuple(categorical_columns)
    required = {"row_id", "target", "p0", "oof_year", *features}
    if (
        type(oof_rows) is not pd.DataFrame
        or oof_rows.empty
        or not required.issubset(oof_rows.columns)
        or oof_rows["row_id"].isna().any()
        or not oof_rows["row_id"].is_unique
        or len(features) != len(set(features))
        or not set(categorical).issubset(features)
        or set(pd.to_numeric(oof_rows["oof_year"], errors="coerce").astype(int))
        != {2021, 2022, 2023, 2024}
    ):
        raise HCFullFitError("full-fit OOF evidence differs")
    target = pd.to_numeric(oof_rows["target"], errors="coerce")
    p0 = pd.to_numeric(oof_rows["p0"], errors="coerce")
    if not target.isin([0, 1]).all() or not p0.between(0, 1).all():
        raise HCFullFitError("full-fit target or baseline differs")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    parameters = dict(active.residual_catboost)
    parameters.pop("od_type", None)
    parameters.pop("od_wait", None)
    parameters.update(
        iterations=int(token.iterations[seed]),
        random_seed=int(seed),
        devices=str(gpu_id),
        train_dir=str(output / "catboost_info"),
    )
    model = model_factory(parameters)
    model.fit(
        oof_rows.loc[:, features],
        (target - p0).to_numpy(dtype="float64"),
        cat_features=list(categorical),
        verbose=50,
    )
    model_path = output / "model.cbm"
    temporary = output / ".model.cbm.tmp"
    temporary.unlink(missing_ok=True)
    try:
        model.save_model(str(temporary))
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise HCFullFitError("full-fit model output is empty")
        os.replace(temporary, model_path)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha256(model_path.read_bytes()).hexdigest()
    _atomic_bytes(
        output / "full_fit.json",
        json.dumps(
            {
                "schema_version": 1,
                "artifact_kind": "tree_hierarchical_full_fit_seed_v1",
                "winner": token.winner,
                "profile_name": token.profile_name,
                "seed": seed,
                "iterations": token.iterations[seed],
                "feature_columns": list(features),
                "categorical_columns": list(categorical),
                "model_sha256": digest,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode(),
    )
    return HCFullFitModel(seed, int(token.iterations[seed]), model_path, digest)
