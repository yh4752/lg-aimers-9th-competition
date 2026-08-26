from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from experiments.independent_dl.feature_sources.seasonal import SeasonalSnapshot
from experiments.temporal_portfolio.seasonal_features import S1State
from experiments.temporal_portfolio.trackman_pitcher import PitcherTrackmanState

from .e2_contracts import E2Contract, load_e2_contract
from .e2_decisions import AcceptanceDecision
from .e2_inputs import file_sha256
from .features import (
    TreeFeatureBatch,
    TreeFeatureState,
    fit_tree_features,
)
from .inputs import VerifiedOfficialData


class E2FullFitError(ValueError):
    """Raised when full fitting is attempted without accepted E2 evidence."""


@dataclass(frozen=True)
class AcceptedForFullFit:
    candidate_id: str
    predictor: str
    seeds: tuple[int, ...]
    iterations: Mapping[int, int]
    decision_sha256: str


@dataclass(frozen=True)
class FullFitModel:
    seed: int
    iterations: int
    model_path: Path
    model_sha256: str


ModelFactory = Callable[[dict[str, object]], object]
_STATE_JSON = "feature_state.json"
_S1_PITCHER = "s1_pitcher.csv"
_S1_BATTER = "s1_batter.csv"
_TRACKMAN = "trackman_lookup.csv"


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise E2FullFitError(f"cannot encode full-fit state: {error}") from error


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


def _valid_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def full_fit_iterations(best_iterations: Sequence[int]) -> int:
    values = tuple(best_iterations)
    if (
        len(values) != 3
        or any(type(value) is not int or value < 0 for value in values)
    ):
        raise E2FullFitError("three non-negative best iterations are required")
    selected = int(statistics.median(values)) + 1
    return min(400, max(50, selected))


def accepted_full_fit_token(
    decision: AcceptanceDecision,
    *,
    candidate_id: str,
    best_iterations: Mapping[int, Sequence[int]],
    decision_sha256: str,
) -> AcceptedForFullFit:
    if decision.status != "accepted":
        raise E2FullFitError("acceptance decision is not accepted")
    if candidate_id not in {"c1_anchor_residual", "c2_trackman_residual"}:
        raise E2FullFitError("full-fit candidate differs")
    if set(best_iterations) != {42, 2026, 3407}:
        raise E2FullFitError("full-fit seed evidence differs")
    if not _valid_sha(decision_sha256):
        raise E2FullFitError("acceptance decision SHA-256 differs")
    iterations = {
        seed: full_fit_iterations(tuple(best_iterations[seed]))
        for seed in (42, 2026, 3407)
    }
    return AcceptedForFullFit(
        candidate_id=candidate_id,
        predictor=decision.predictor,
        seeds=(42, 2026, 3407),
        iterations=MappingProxyType(iterations),
        decision_sha256=decision_sha256,
    )


def _frame_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode(
        "utf-8"
    )


def _dtype_payload(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column): str(dtype) for column, dtype in frame.dtypes.items()}


def _cast_frame(frame: pd.DataFrame, dtypes: Mapping[str, str]) -> pd.DataFrame:
    if set(frame.columns) != set(dtypes):
        raise E2FullFitError("frozen frame columns differ")
    output = frame.copy(deep=True)
    for column in output.columns:
        dtype = dtypes[column]
        try:
            output[column] = output[column].astype(object if dtype == "object" else dtype)
        except (TypeError, ValueError) as error:
            raise E2FullFitError(f"frozen frame dtype differs: {column}") from error
    return output


def export_frozen_tree_state(
    state: TreeFeatureState,
    destination: Path,
    *,
    candidate_id: str,
) -> Path:
    if type(state) is not TreeFeatureState:
        raise E2FullFitError("tree feature state type differs")
    if candidate_id not in {"c1_anchor_residual", "c2_trackman_residual"}:
        raise E2FullFitError("frozen-state candidate differs")
    if candidate_id == "c1_anchor_residual" and state.trackman_state is not None:
        raise E2FullFitError("c1 must not contain a TrackMan state")
    if candidate_id == "c2_trackman_residual" and (
        state.trackman_state is None
        or state.trackman_state.cutoff_year != state.valid_year - 1
        or state.trackman_state.cutoff_year != 2024
    ):
        raise E2FullFitError("c2 requires the cutoff-2024 TrackMan state")
    if any(not _valid_sha(value) for value in state.source_hashes.values()):
        raise E2FullFitError("feature source SHA-256 differs")
    root = Path(destination)
    if root.exists() and (root.is_symlink() or any(root.iterdir())):
        raise E2FullFitError("frozen-state destination is not empty")
    root.mkdir(parents=True, exist_ok=True)
    snapshot = state.s1_state.snapshot
    payloads = {
        _S1_PITCHER: _frame_bytes(snapshot.pitcher),
        _S1_BATTER: _frame_bytes(snapshot.batter),
    }
    trackman_dtypes: dict[str, str] | None = None
    trackman_lookup_sha256: str | None = None
    trackman_bundle_sha256: dict[str, str] | None = None
    if state.trackman_state is not None:
        lookup = state.trackman_state.lookup
        payloads[_TRACKMAN] = _frame_bytes(lookup)
        trackman_dtypes = _dtype_payload(lookup)
        trackman_lookup_sha256 = state.trackman_state.lookup_sha256
        trackman_bundle_sha256 = dict(state.trackman_state.bundle_sha256)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "tree_expert_e2_frozen_state_v1",
        "candidate_id": candidate_id,
        "valid_year": state.valid_year,
        "prior_rate": state.prior_rate,
        "categorical_columns": list(state.categorical_columns),
        "feature_columns": list(state.feature_columns),
        "source_hashes": dict(state.source_hashes),
        "anchor_formula": "e1_anchor_v1",
        "probability_clip": [0.00001, 0.99999],
        "s1": {
            "cutoff_year": snapshot.cutoff_year,
            "pitcher_dtypes": _dtype_payload(snapshot.pitcher),
            "batter_dtypes": _dtype_payload(snapshot.batter),
        },
        "trackman": None
        if state.trackman_state is None
        else {
            "cutoff_year": state.trackman_state.cutoff_year,
            "dtypes": trackman_dtypes,
            "lookup_sha256": trackman_lookup_sha256,
            "bundle_sha256": trackman_bundle_sha256,
        },
        "members": {
            name: {"size": len(value), "sha256": sha256(value).hexdigest()}
            for name, value in sorted(payloads.items())
        },
    }
    for name, payload in payloads.items():
        _atomic_bytes(root / name, payload)
    _atomic_bytes(root / _STATE_JSON, _canonical_json(manifest))
    return root


def _verified_payload(root: Path, name: str, evidence: object) -> bytes:
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise E2FullFitError(f"frozen state member is absent: {name}")
    payload = path.read_bytes()
    if (
        type(evidence) is not dict
        or set(evidence) != {"size", "sha256"}
        or evidence["size"] != len(payload)
        or evidence["sha256"] != sha256(payload).hexdigest()
    ):
        raise E2FullFitError(f"frozen state member SHA-256 differs: {name}")
    return payload


def load_frozen_tree_state(root: Path) -> TreeFeatureState:
    source = Path(root)
    manifest_path = source / _STATE_JSON
    if source.is_symlink() or not source.is_dir() or not manifest_path.is_file():
        raise E2FullFitError("frozen state root differs")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise E2FullFitError(f"cannot read frozen state: {error}") from error
    expected_keys = {
        "schema_version",
        "artifact_kind",
        "candidate_id",
        "valid_year",
        "prior_rate",
        "categorical_columns",
        "feature_columns",
        "source_hashes",
        "anchor_formula",
        "probability_clip",
        "s1",
        "trackman",
        "members",
    }
    if type(manifest) is not dict or set(manifest) != expected_keys:
        raise E2FullFitError("frozen state manifest keys differ")
    candidate_id = manifest["candidate_id"]
    trackman_meta = manifest["trackman"]
    if (candidate_id == "c1_anchor_residual") != (trackman_meta is None):
        raise E2FullFitError("frozen candidate and TrackMan identity differ")
    expected_members = {_S1_PITCHER, _S1_BATTER}
    if trackman_meta is not None:
        expected_members.add(_TRACKMAN)
    if (
        manifest["schema_version"] != 1
        or manifest["artifact_kind"] != "tree_expert_e2_frozen_state_v1"
        or candidate_id not in {"c1_anchor_residual", "c2_trackman_residual"}
        or manifest["anchor_formula"] != "e1_anchor_v1"
        or manifest["probability_clip"] != [0.00001, 0.99999]
        or type(manifest["members"]) is not dict
        or set(manifest["members"]) != expected_members
    ):
        raise E2FullFitError("frozen state identity differs")
    payloads = {
        name: _verified_payload(source, name, manifest["members"][name])
        for name in expected_members
    }
    s1_meta = manifest["s1"]
    if type(s1_meta) is not dict or set(s1_meta) != {
        "cutoff_year",
        "pitcher_dtypes",
        "batter_dtypes",
    }:
        raise E2FullFitError("frozen S1 metadata differs")
    pitcher = pd.read_csv(BytesIO(payloads[_S1_PITCHER]))
    batter = pd.read_csv(BytesIO(payloads[_S1_BATTER]))
    pitcher = _cast_frame(pitcher, s1_meta["pitcher_dtypes"])
    batter = _cast_frame(batter, s1_meta["batter_dtypes"])
    valid_year = int(manifest["valid_year"])
    prior_rate = float(manifest["prior_rate"])
    if not math.isfinite(prior_rate):
        raise E2FullFitError("frozen prior rate differs")
    snapshot = SeasonalSnapshot(
        cutoff_year=int(s1_meta["cutoff_year"]),
        pitcher=pitcher,
        batter=batter,
    )
    try:
        s1_state = S1State._from_snapshot(
            valid_year=valid_year,
            prior_rate=prior_rate,
            snapshot=snapshot,
        )
    except Exception as error:
        raise E2FullFitError(f"cannot restore S1 state: {error}") from error
    trackman_state: PitcherTrackmanState | None = None
    if trackman_meta is not None:
        if type(trackman_meta) is not dict or set(trackman_meta) != {
            "cutoff_year",
            "dtypes",
            "lookup_sha256",
            "bundle_sha256",
        }:
            raise E2FullFitError("frozen TrackMan metadata differs")
        lookup = pd.read_csv(BytesIO(payloads[_TRACKMAN]))
        lookup = _cast_frame(lookup, trackman_meta["dtypes"])
        try:
            trackman_state = PitcherTrackmanState._from_lookup(
                cutoff_year=int(trackman_meta["cutoff_year"]),
                lookup=lookup,
            )
        except Exception as error:
            raise E2FullFitError(f"cannot restore TrackMan state: {error}") from error
        if (
            trackman_state.lookup_sha256 != trackman_meta["lookup_sha256"]
            or dict(trackman_state.bundle_sha256) != trackman_meta["bundle_sha256"]
        ):
            raise E2FullFitError("frozen TrackMan identity differs")
    source_hashes = manifest["source_hashes"]
    if type(source_hashes) is not dict or any(
        not _valid_sha(value) for value in source_hashes.values()
    ):
        raise E2FullFitError("frozen feature source hashes differ")
    return TreeFeatureState(
        valid_year=valid_year,
        prior_rate=prior_rate,
        categorical_columns=tuple(manifest["categorical_columns"]),
        feature_columns=tuple(manifest["feature_columns"]),
        s1_state=s1_state,
        trackman_state=trackman_state,
        source_hashes=MappingProxyType(dict(source_hashes)),
    )


def prepare_full_fit_features(
    token: AcceptedForFullFit,
    data: VerifiedOfficialData,
) -> tuple[TreeFeatureState, TreeFeatureBatch]:
    rows = pd.read_csv(data.train)
    seasons = pd.to_numeric(rows["season"], errors="raise").astype("int64")
    if rows.empty or seasons.min() < 2019 or seasons.max() != 2024:
        raise E2FullFitError("full-fit seasons differ")
    history = (
        pd.read_csv(data.history)
        if token.candidate_id == "c2_trackman_residual"
        else None
    )
    return fit_tree_features(
        rows,
        history,
        valid_year=2025,
        use_trackman=token.candidate_id == "c2_trackman_residual",
        minimum_trackman_coverage=0.30,
    )


def _default_model_factory(parameters: dict[str, object]) -> object:
    try:
        from catboost import CatBoostRegressor
    except ImportError as error:
        raise E2FullFitError("catboost==1.2.10 is required") from error
    return CatBoostRegressor(**parameters)


def fit_full_seed(
    *,
    token: AcceptedForFullFit,
    seed: int,
    state: TreeFeatureState,
    batch: TreeFeatureBatch,
    output_dir: Path,
    gpu_id: int,
    contract: E2Contract | None = None,
    model_factory: ModelFactory = _default_model_factory,
) -> FullFitModel:
    if seed not in token.seeds or seed not in token.iterations:
        raise E2FullFitError("full-fit seed differs")
    if batch.target is None or len(batch.target) != len(batch.frame):
        raise E2FullFitError("full-fit target differs")
    if tuple(batch.frame.columns) != state.feature_columns:
        raise E2FullFitError("full-fit feature schema differs")
    active = load_e2_contract() if contract is None else contract
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    parameters = dict(active.catboost)
    parameters.update(
        iterations=int(token.iterations[seed]),
        random_seed=seed,
        devices=str(gpu_id),
        train_dir=str(output / "catboost_info"),
        loss_function="RMSE",
        eval_metric="RMSE",
    )
    model = model_factory(parameters)
    residual = np.asarray(batch.target, dtype="float64") - np.asarray(
        batch.anchor, dtype="float64"
    )
    model.fit(
        batch.frame,
        residual,
        cat_features=list(state.categorical_columns),
        use_best_model=False,
        verbose=50,
    )
    model_path = output / f"catboost_seed_{seed}.cbm"
    temporary = model_path.with_name(f".{model_path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        model.save_model(str(temporary))
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise E2FullFitError("full-fit model output is empty")
        os.replace(temporary, model_path)
    finally:
        temporary.unlink(missing_ok=True)
    return FullFitModel(
        seed=seed,
        iterations=int(token.iterations[seed]),
        model_path=model_path,
        model_sha256=file_sha256(model_path),
    )
