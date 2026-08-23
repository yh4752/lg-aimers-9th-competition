"""Strict, selection-free CatBoost prefix evidence for temporal folds."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from numbers import Integral, Real
from types import MappingProxyType
from typing import Protocol

import numpy as np
import pandas as pd

from experiments.independent_dl.features import FeatureBatch
from experiments.independent_dl.models.common import ModelMetadata
from experiments.independent_dl.training import TrainRequest

from .identity import TrainingIdentity, audit_duplicate


class CatBoostTrainingError(ValueError):
    """Raised before malformed CatBoost evidence can be used or published."""


CATBOOST_PREFIXES = (16, 64, 192, 384)
_EXPERTS = frozenset({"catboost", "tabm", "lupi"})
_CANONICAL_FAMILY = {"catboost": "catboost", "tabm": "tabm", "lupi": "tabm"}
_BASE_AUDIT_COLUMNS = (
    "row_id",
    "target",
    "pitcher_id",
    "batter_id",
    "game_type",
    "pitcher_id_known",
    "batter_id_known",
    "trackman_available",
)


class CatBoostBackend(Protocol):
    def fit(
        self,
        frame: pd.DataFrame,
        target: np.ndarray,
        *,
        sample_weight: np.ndarray,
        iterations: int,
        seed: int,
    ) -> object: ...


@dataclass(frozen=True)
class _SealedBatch:
    row_id: np.ndarray = field(repr=False)
    season: np.ndarray = field(repr=False)
    game_type: np.ndarray = field(repr=False)
    x_num: np.ndarray = field(repr=False)
    x_cat: np.ndarray = field(repr=False)
    y: np.ndarray | None = field(repr=False)


@dataclass(frozen=True)
class _SealedMetadata:
    n_num_features: int
    categorical_cardinalities: tuple[int, ...]
    train_x_num: np.ndarray | None = field(repr=False)
    piecewise_bin_edges: tuple[np.ndarray, ...] | None = field(repr=False)


@dataclass(frozen=True)
class _SealedRequest:
    candidate_id: str
    family: str
    seed: int
    epochs: int
    min_epochs: int
    model_config_json: bytes = field(repr=False)
    training_config_json: bytes = field(repr=False)
    checkpoint_binding_json: bytes = field(repr=False)
    train: _SealedBatch = field(repr=False)
    valid: _SealedBatch = field(repr=False)
    model_metadata: _SealedMetadata | None = field(repr=False)


def temporal_train_request_sha256(request: TrainRequest) -> str:
    """Return the canonical digest used to bind a temporal TabM/LUPI request."""

    if type(request) is not TrainRequest:
        raise CatBoostTrainingError("temporal request digest requires an exact TrainRequest")
    return _sealed_request_sha256(_seal_request(request))


def temporal_sample_weight_sha256(sample_weight: np.ndarray) -> str:
    """Digest the exact positive float64 weights used by temporal trainers."""

    weight = _numeric_vector(sample_weight, "sample weight", nonempty=True)
    if np.any(weight <= 0):
        raise CatBoostTrainingError("sample weight must be strictly positive")
    return _array_digest(weight)["sha256"]


def temporal_audit_sha256(
    audit_frame: pd.DataFrame, *, segment_columns: tuple[str, ...]
) -> str:
    """Digest normalized, source-ordered worker audit evidence."""

    frame, _ = _validated_audit_frame(audit_frame, segment_columns)
    return _audit_frame_sha256(frame)


def temporal_catboost_request_sha256(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    target: np.ndarray,
    valid_row_id: np.ndarray,
) -> str:
    """Digest CatBoost frames, schema/dtypes, target, and validation row order."""

    if type(train_frame) is not pd.DataFrame or train_frame.empty:
        raise CatBoostTrainingError("CatBoost train frame must be nonempty")
    if type(valid_frame) is not pd.DataFrame or valid_frame.empty:
        raise CatBoostTrainingError("CatBoost valid frame must be nonempty")
    if not train_frame.columns.is_unique or not valid_frame.columns.is_unique:
        raise CatBoostTrainingError("CatBoost feature columns must be unique")
    if tuple(train_frame.columns) != tuple(valid_frame.columns):
        raise CatBoostTrainingError("CatBoost train and valid columns differ")
    train_target = _binary_vector(target, "training target", len(train_frame))
    if not isinstance(valid_row_id, np.ndarray):
        raise CatBoostTrainingError("valid row IDs must be a NumPy array")
    row_ids = _stable_ids(valid_row_id.tolist(), "valid row_id")
    if len(valid_frame) != len(row_ids):
        raise CatBoostTrainingError("CatBoost validation rows differ")
    payload = {
        "train_frame_sha256": _audit_frame_sha256(train_frame.reset_index(drop=True)),
        "valid_frame_sha256": _audit_frame_sha256(valid_frame.reset_index(drop=True)),
        "target": _array_digest(train_target),
        "valid_row_order_sha256": _row_order_sha256(row_ids),
    }
    return hashlib.sha256(_canonical_json_value(payload)).hexdigest()

    def predict(
        self,
        model: object,
        frame: pd.DataFrame,
        *,
        ntree_end: int,
    ) -> object: ...


@dataclass(frozen=True, init=False)
class TemporalTrainingJob:
    """One immutable-identity temporal training request.

    DataFrames and arrays are snapshotted on construction.  The existing
    ``TrainingIdentity`` remains the only semantic identity type.
    """

    job_id: str
    expert: str
    identity: TrainingIdentity
    seed: int
    segment_columns: tuple[str, ...]
    teacher_oof: object | None = field(repr=False)
    teacher_oof_sha256: str | None
    teacher_lambda: float
    _sample_weight_bytes: bytes = field(repr=False)
    _audit_frame: pd.DataFrame = field(repr=False)
    _train_frame: pd.DataFrame | None = field(repr=False)
    _valid_frame: pd.DataFrame | None = field(repr=False)
    _target_bytes: bytes | None = field(repr=False)
    _valid_row_ids: tuple[object, ...] = field(repr=False)
    _valid_row_order_sha256: str = field(repr=False)
    _audit_sha256: str = field(repr=False)
    _train_request: _SealedRequest | None = field(repr=False)
    _train_request_sha256: str | None = field(repr=False)

    def __init__(
        self,
        *,
        job_id: str,
        expert: str,
        identity: TrainingIdentity,
        sample_weight: np.ndarray,
        seed: int,
        audit_frame: pd.DataFrame,
        segment_columns: tuple[str, ...] = (),
        train_frame: pd.DataFrame | None = None,
        valid_frame: pd.DataFrame | None = None,
        target: np.ndarray | None = None,
        valid_row_id: np.ndarray | None = None,
        train_request: TrainRequest | None = None,
        teacher_oof: object | None = None,
        teacher_oof_sha256: str | None = None,
        teacher_lambda: float = 0.0,
    ) -> None:
        if type(job_id) is not str or not job_id or job_id != job_id.strip():
            raise CatBoostTrainingError("job_id must be a nonempty canonical string")
        if type(expert) is not str or expert not in _EXPERTS:
            raise CatBoostTrainingError("expert is not preregistered")
        if type(identity) is not TrainingIdentity:
            raise CatBoostTrainingError("identity must be an exact TrainingIdentity")
        try:
            audit_duplicate(identity, {})
        except ValueError as error:
            raise CatBoostTrainingError("training identity is invalid") from error
        if type(seed) is not int or seed < 0:
            raise CatBoostTrainingError("seed must be an exact non-negative integer")
        if expert != "catboost":
            if type(train_request) is not TrainRequest:
                raise CatBoostTrainingError("TabM jobs require an existing TrainRequest")
            if train_request.candidate_id != job_id:
                raise CatBoostTrainingError("TrainRequest candidate does not match job_id")
            if type(train_request.seed) is not int or train_request.seed != seed:
                raise CatBoostTrainingError("TrainRequest seed does not match job seed")
        _validate_identity_bindings(
            identity, job_id, expert, seed,
            None if expert == "catboost" else train_request,
        )

        weight = _numeric_vector(sample_weight, "sample weight", nonempty=True)
        if np.any(weight <= 0):
            raise CatBoostTrainingError("sample weight must be strictly positive")

        audit, row_ids = _validated_audit_frame(audit_frame, segment_columns)
        target_valid = audit["target"].to_numpy(dtype="int8", copy=True)

        if valid_row_id is None:
            supplied_row_ids = row_ids
        else:
            if not isinstance(valid_row_id, np.ndarray):
                raise CatBoostTrainingError("valid row IDs must be a NumPy array")
            supplied_row_ids = _stable_ids(valid_row_id.tolist(), "valid row_id")
        if supplied_row_ids != row_ids:
            raise CatBoostTrainingError("valid row order differs from the audit source")

        train_snapshot = valid_snapshot = None
        target_bytes = None
        sealed_request = None
        request_sha256 = None
        if expert == "catboost":
            if type(train_frame) is not pd.DataFrame or train_frame.empty:
                raise CatBoostTrainingError("CatBoost train frame must be nonempty")
            if type(valid_frame) is not pd.DataFrame or valid_frame.empty:
                raise CatBoostTrainingError("CatBoost valid frame must be nonempty")
            if not train_frame.columns.is_unique or not valid_frame.columns.is_unique:
                raise CatBoostTrainingError("CatBoost feature columns must be unique")
            if tuple(train_frame.columns) != tuple(valid_frame.columns):
                raise CatBoostTrainingError("CatBoost train and valid columns differ")
            if len(valid_frame) != len(row_ids):
                raise CatBoostTrainingError("CatBoost validation rows differ")
            train_target = _binary_vector(target, "training target", len(train_frame))
            if len(weight) != len(train_frame):
                raise CatBoostTrainingError("CatBoost sample weights are not row aligned")
            train_snapshot = train_frame.copy(deep=True).reset_index(drop=True)
            valid_snapshot = valid_frame.copy(deep=True).reset_index(drop=True)
            target_bytes = train_target.tobytes()
            if train_request is not None or teacher_oof is not None:
                raise CatBoostTrainingError("CatBoost job contains unrelated trainer state")
        else:
            if type(train_request) is not TrainRequest:
                raise CatBoostTrainingError("TabM jobs require an existing TrainRequest")
            if train_request.candidate_id != job_id:
                raise CatBoostTrainingError("TrainRequest candidate does not match job_id")
            if type(train_request.seed) is not int or train_request.seed != seed:
                raise CatBoostTrainingError("TrainRequest seed does not match job seed")
            sealed_request = _seal_request(train_request)
            request_sha256 = _sealed_request_sha256(sealed_request)
            identity_request_sha = identity.payload["model"]["train_request_sha256"]
            if identity_request_sha != request_sha256:
                raise CatBoostTrainingError("TrainRequest hash differs from training identity")
            if len(weight) != len(getattr(train_request, "train").row_id):
                raise CatBoostTrainingError("TabM sample weights are not row aligned")
            request_valid_ids = _stable_ids(
                list(getattr(train_request, "valid").row_id), "request valid row_id"
            )
            if request_valid_ids != row_ids:
                raise CatBoostTrainingError("TrainRequest validation order differs")
            if train_request.valid.y is None:
                raise CatBoostTrainingError("TrainRequest validation target is required")
            request_valid_target = _binary_vector(
                train_request.valid.y, "TrainRequest validation target", len(audit)
            )
            if not np.array_equal(target_valid, request_valid_target):
                raise CatBoostTrainingError(
                    "audit target differs from TrainRequest validation target"
                )
            if expert == "tabm" and (
                teacher_oof is not None
                or teacher_oof_sha256 is not None
                or teacher_lambda != 0.0
            ):
                raise CatBoostTrainingError("plain TabM job contains teacher state")
            if expert == "lupi":
                if teacher_oof is None or not _is_sha256(teacher_oof_sha256):
                    raise CatBoostTrainingError("LUPI job requires a sealed teacher OOF hash")
                if isinstance(teacher_lambda, bool) or not isinstance(teacher_lambda, Real):
                    raise CatBoostTrainingError("teacher lambda must be numeric")
                value = float(teacher_lambda)
                if not np.isfinite(value) or not 0.0 < value <= 1.0:
                    raise CatBoostTrainingError("teacher lambda must be in (0, 1]")
                teacher_lambda = value

        model_binding = identity.payload["model"]
        observed_weight_sha256 = temporal_sample_weight_sha256(weight)
        if model_binding["sample_weight_sha256"] != observed_weight_sha256:
            raise CatBoostTrainingError("training identity sample_weight_sha256 differs")
        observed_audit_sha256 = temporal_audit_sha256(
            audit, segment_columns=segment_columns
        )
        if model_binding["audit_sha256"] != observed_audit_sha256:
            raise CatBoostTrainingError("training identity audit_sha256 differs")
        if expert == "catboost":
            observed_catboost_sha256 = temporal_catboost_request_sha256(
                train_snapshot, valid_snapshot, train_target,
                np.asarray(row_ids, dtype=object),
            )
            if model_binding["catboost_request_sha256"] != observed_catboost_sha256:
                raise CatBoostTrainingError(
                    "training identity catboost_request_sha256 differs"
                )
        if expert == "lupi":
            if model_binding["teacher_oof_sha256"] != teacher_oof_sha256:
                raise CatBoostTrainingError("training identity teacher_oof_sha256 differs")
            identity_lambda = model_binding["teacher_lambda"]
            if type(identity_lambda) is not float or identity_lambda != teacher_lambda:
                raise CatBoostTrainingError("training identity teacher_lambda differs")

        object.__setattr__(self, "job_id", job_id)
        object.__setattr__(self, "expert", expert)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "segment_columns", segment_columns)
        object.__setattr__(self, "teacher_oof", teacher_oof)
        object.__setattr__(self, "teacher_oof_sha256", teacher_oof_sha256)
        object.__setattr__(self, "teacher_lambda", teacher_lambda)
        object.__setattr__(self, "_sample_weight_bytes", weight.tobytes())
        object.__setattr__(self, "_audit_frame", audit)
        object.__setattr__(self, "_train_frame", train_snapshot)
        object.__setattr__(self, "_valid_frame", valid_snapshot)
        object.__setattr__(self, "_target_bytes", target_bytes)
        object.__setattr__(self, "_valid_row_ids", row_ids)
        object.__setattr__(self, "_valid_row_order_sha256", _row_order_sha256(row_ids))
        object.__setattr__(self, "_audit_sha256", _audit_frame_sha256(audit))
        object.__setattr__(self, "_train_request", sealed_request)
        object.__setattr__(self, "_train_request_sha256", request_sha256)

    @property
    def sample_weight(self) -> np.ndarray:
        return np.frombuffer(self._sample_weight_bytes, dtype="float64")

    @property
    def audit_frame(self) -> pd.DataFrame:
        return self._audit_frame.copy(deep=True)

    @property
    def train_frame(self) -> pd.DataFrame | None:
        return None if self._train_frame is None else self._train_frame.copy(deep=True)

    @property
    def valid_frame(self) -> pd.DataFrame | None:
        return None if self._valid_frame is None else self._valid_frame.copy(deep=True)

    @property
    def target(self) -> np.ndarray | None:
        if self._target_bytes is None:
            return None
        return np.frombuffer(self._target_bytes, dtype="int8")

    @property
    def valid_row_id(self) -> np.ndarray:
        result = np.asarray(self._valid_row_ids, dtype=object)
        result.setflags(write=False)
        return result

    @property
    def train_request(self) -> TrainRequest | None:
        if self._train_request is None:
            return None
        self._validate_request_seal()
        return _restore_request(self._train_request)

    @property
    def train_request_sha256(self) -> str | None:
        self._validate_request_seal()
        return self._train_request_sha256

    @property
    def valid_rows(self) -> int:
        return len(self._valid_row_ids)

    @property
    def valid_row_ids(self) -> tuple[object, ...]:
        return self._valid_row_ids

    @property
    def valid_row_order_sha256(self) -> str:
        return self._valid_row_order_sha256

    def frozen_audit_frame(self) -> pd.DataFrame:
        frame = self._audit_frame.copy(deep=True)
        if _audit_frame_sha256(frame) != self._audit_sha256:
            raise CatBoostTrainingError("sealed audit frame integrity differs")
        if _row_order_sha256(_stable_ids(frame["row_id"].tolist(), "audit row_id")) != self._valid_row_order_sha256:
            raise CatBoostTrainingError("sealed audit row order differs")
        return frame

    def validate_seals(self) -> None:
        _validate_identity_bindings(
            self.identity, self.job_id, self.expert, self.seed,
            None if self._train_request is None else _restore_request(self._train_request),
        )
        self.frozen_audit_frame()
        self._validate_request_seal()
        model = self.identity.payload["model"]
        if model["sample_weight_sha256"] != temporal_sample_weight_sha256(
            self.sample_weight
        ):
            raise CatBoostTrainingError("sealed sample weight digest differs")
        if model["audit_sha256"] != temporal_audit_sha256(
            self._audit_frame, segment_columns=self.segment_columns
        ):
            raise CatBoostTrainingError("sealed audit identity digest differs")
        if self.expert == "catboost":
            observed = temporal_catboost_request_sha256(
                self._train_frame, self._valid_frame, self.target, self.valid_row_id
            )
            if model["catboost_request_sha256"] != observed:
                raise CatBoostTrainingError("sealed CatBoost request digest differs")
        if self.expert == "lupi":
            if model["teacher_oof_sha256"] != self.teacher_oof_sha256:
                raise CatBoostTrainingError("sealed teacher OOF digest differs")
            if type(model["teacher_lambda"]) is not float or model["teacher_lambda"] != self.teacher_lambda:
                raise CatBoostTrainingError("sealed teacher lambda differs")

    def _validate_request_seal(self) -> None:
        if self._train_request is None:
            if self._train_request_sha256 is not None:
                raise CatBoostTrainingError("unexpected TrainRequest digest")
            return
        observed = _sealed_request_sha256(self._train_request)
        if observed != self._train_request_sha256:
            raise CatBoostTrainingError("sealed TrainRequest integrity digest differs")
        identity_digest = self.identity.payload["model"]["train_request_sha256"]
        if identity_digest != observed:
            raise CatBoostTrainingError("sealed TrainRequest digest differs from identity")


@dataclass(frozen=True, init=False)
class CatBoostResult:
    model: object = field(repr=False)
    _prediction_bytes: Mapping[int, bytes] = field(repr=False)
    valid_row_ids: tuple[object, ...]
    valid_row_order_sha256: str
    backend_calls: Mapping[str, object]

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("CatBoostResult instances must be created by validation")

    @property
    def prefix_predictions(self) -> Mapping[int, np.ndarray]:
        return MappingProxyType(
            {
                prefix: np.frombuffer(self._prediction_bytes[prefix], dtype="float64")
                for prefix in CATBOOST_PREFIXES
            }
        )


def run_catboost_job(
    job: TemporalTrainingJob, *, backend: CatBoostBackend
) -> CatBoostResult:
    """Fit exactly 384 trees and record predictions at every sealed prefix."""

    if type(job) is not TemporalTrainingJob or job.expert != "catboost":
        raise CatBoostTrainingError("run_catboost_job requires a validated CatBoost job")
    job.validate_seals()
    fit = getattr(backend, "fit", None)
    predict = getattr(backend, "predict", None)
    if not callable(fit) or not callable(predict):
        raise CatBoostTrainingError("CatBoost backend must provide fit and predict")
    train = job.train_frame.copy(deep=True)
    valid = job.valid_frame.copy(deep=True)
    target = np.array(job.target, dtype="int8", copy=True)
    weight = np.array(job.sample_weight, dtype="float64", copy=True)
    model = fit(
        train,
        target,
        sample_weight=weight,
        iterations=max(CATBOOST_PREFIXES),
        seed=job.seed,
    )
    try:
        tree_count = model.tree_count_
    except AttributeError as error:
        raise CatBoostTrainingError("backend model tree count evidence is missing") from error
    if isinstance(tree_count, bool) or not isinstance(tree_count, Integral):
        raise CatBoostTrainingError("backend model tree count is invalid")
    if tree_count != max(CATBOOST_PREFIXES):
        raise CatBoostTrainingError(
            "backend did not fit the exact maximum prefix; early stopping is forbidden"
        )
    predictions = {
        prefix: predict(model, valid.copy(deep=True), ntree_end=prefix)
        for prefix in CATBOOST_PREFIXES
    }
    return validate_catboost_result(
        model,
        predictions,
        np.asarray(job.valid_row_ids, dtype=object),
        backend_calls={
            "fit_calls": 1,
            "iterations": max(CATBOOST_PREFIXES),
            "seed": job.seed,
            "sample_weight": True,
            "early_stopping": False,
            "predict_ntree_end": list(CATBOOST_PREFIXES),
        },
    )


def validate_catboost_result(
    model: object,
    predictions: Mapping[int, object],
    valid_row_id: np.ndarray,
    *,
    backend_calls: Mapping[str, object] | None = None,
) -> CatBoostResult:
    if not isinstance(predictions, Mapping):
        raise CatBoostTrainingError("prefix predictions must be a mapping")
    try:
        snapshot = dict(predictions.items())
    except Exception as error:
        raise CatBoostTrainingError("cannot snapshot prefix predictions") from error
    if tuple(sorted(snapshot)) != CATBOOST_PREFIXES:
        raise CatBoostTrainingError("prefix predictions differ from preregistration")
    if not isinstance(valid_row_id, np.ndarray):
        raise CatBoostTrainingError("valid row IDs must be a NumPy array")
    row_ids = _stable_ids(valid_row_id.tolist(), "valid row_id")
    stored: dict[int, bytes] = {}
    for prefix in CATBOOST_PREFIXES:
        try:
            probability = np.asarray(snapshot[prefix], dtype="float64")
        except (TypeError, ValueError, OverflowError) as error:
            raise CatBoostTrainingError("prefix probability must be numeric") from error
        if probability.shape != (len(row_ids),):
            raise CatBoostTrainingError("prefix probability shape differs")
        if not np.isfinite(probability).all():
            raise CatBoostTrainingError("prefix probability must be finite")
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise CatBoostTrainingError("prefix probability must be in [0, 1]")
        stored[prefix] = np.array(probability, dtype="float64", copy=True).tobytes()

    evidence = dict(backend_calls or {})
    if backend_calls is not None:
        expected = {
            "fit_calls": 1,
            "iterations": 384,
            "seed": evidence.get("seed"),
            "sample_weight": True,
            "early_stopping": False,
            "predict_ntree_end": list(CATBOOST_PREFIXES),
        }
        if evidence != expected or type(evidence["seed"]) is not int:
            raise CatBoostTrainingError("backend call evidence differs")
    instance = object.__new__(CatBoostResult)
    object.__setattr__(instance, "model", model)
    object.__setattr__(instance, "_prediction_bytes", MappingProxyType(stored))
    object.__setattr__(instance, "valid_row_ids", row_ids)
    object.__setattr__(instance, "valid_row_order_sha256", _row_order_sha256(row_ids))
    object.__setattr__(instance, "backend_calls", MappingProxyType(evidence))
    return instance


def _numeric_vector(value: object, label: str, *, nonempty: bool) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise CatBoostTrainingError(f"{label} must be a NumPy array")
    if value.ndim != 1 or (nonempty and len(value) == 0):
        raise CatBoostTrainingError(f"{label} must be a nonempty one-dimensional array")
    if np.issubdtype(value.dtype, np.bool_) or not (
        np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.floating)
    ):
        raise CatBoostTrainingError(f"{label} must have a real numeric dtype")
    result = np.array(value, dtype="float64", copy=True)
    if not np.isfinite(result).all():
        raise CatBoostTrainingError(f"{label} must be finite")
    return result


def _binary_vector(value: object, label: str, length: int) -> np.ndarray:
    try:
        snapshot = list(value)
    except (TypeError, ValueError) as error:
        raise CatBoostTrainingError(f"{label} must contain exact binary integers") from error
    if len(snapshot) != length or any(
        isinstance(item, (bool, np.bool_))
        or not isinstance(item, Integral)
        or int(item) not in (0, 1)
        for item in snapshot
    ):
        raise CatBoostTrainingError(f"{label} must contain exact binary integers")
    return np.asarray(snapshot, dtype="int8")


def _stable_ids(values: list[object], label: str) -> tuple[object, ...]:
    keys: list[str] = []
    normalized: list[object] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise CatBoostTrainingError(f"{label} contains an invalid ID")
        if isinstance(value, Integral):
            canonical = int(value)
            key = f"int:{canonical}"
        elif isinstance(value, (str, np.str_)) and str(value) and str(value) == str(value).strip():
            canonical = str(value)
            key = f"str:{len(canonical.encode('utf-8'))}:{canonical}"
        else:
            raise CatBoostTrainingError(f"{label} contains an invalid ID")
        keys.append(key)
        normalized.append(canonical)
    if not keys or len(keys) != len(set(keys)):
        raise CatBoostTrainingError(f"{label} values must be nonempty and unique")
    return tuple(normalized)


def _row_order_sha256(row_ids: tuple[object, ...]) -> str:
    digest = hashlib.sha256()
    for value in row_ids:
        encoded = (f"int:{int(value)}" if isinstance(value, Integral) else f"str:{value}").encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _validated_audit_frame(
    audit_frame: object, segment_columns: tuple[str, ...]
) -> tuple[pd.DataFrame, tuple[object, ...]]:
    if type(audit_frame) is not pd.DataFrame or audit_frame.empty:
        raise CatBoostTrainingError("audit frame must be a nonempty pandas DataFrame")
    if not audit_frame.columns.is_unique:
        raise CatBoostTrainingError("audit frame columns must be unique")
    if type(segment_columns) is not tuple or any(
        type(column) is not str or not column for column in segment_columns
    ):
        raise CatBoostTrainingError("segment columns must be canonical strings")
    if len(set(segment_columns)) != len(segment_columns):
        raise CatBoostTrainingError("segment columns must be unique")
    required = (*_BASE_AUDIT_COLUMNS, *segment_columns)
    if set(audit_frame.columns) != set(required):
        raise CatBoostTrainingError("audit frame schema differs")
    audit = audit_frame.loc[:, required].copy(deep=True).reset_index(drop=True)
    row_ids = _stable_ids(audit["row_id"].tolist(), "audit row_id")
    audit["target"] = _binary_vector(audit["target"], "audit target", len(audit))
    return audit, row_ids


def _audit_frame_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(repr(tuple(frame.columns)).encode("utf-8"))
    digest.update(repr(tuple(str(dtype) for dtype in frame.dtypes)).encode("utf-8"))
    digest.update(
        pd.util.hash_pandas_object(frame, index=True, categorize=False)
        .to_numpy(dtype="uint64", copy=False)
        .tobytes()
    )
    return digest.hexdigest()


def _validate_identity_bindings(
    identity: TrainingIdentity,
    job_id: str,
    expert: str,
    seed: int,
    request: TrainRequest | None,
) -> None:
    if identity.payload["seed"] != seed:
        raise CatBoostTrainingError("job seed differs from training identity seed")
    model = identity.payload["model"]
    required = {
        "job_id", "candidate_id", "expert", "family",
        "sample_weight_sha256", "audit_sha256",
    }
    if request is not None:
        required.add("train_request_sha256")
    if expert == "catboost":
        required.add("catboost_request_sha256")
    if expert == "lupi":
        required.update({"teacher_oof_sha256", "teacher_lambda"})
    missing = required.difference(model)
    if missing:
        raise CatBoostTrainingError(
            f"training identity model bindings are missing: {sorted(missing)}"
        )
    canonical_family = _CANONICAL_FAMILY[expert]
    if request is not None and request.family != canonical_family:
        raise CatBoostTrainingError("TrainRequest family is not canonical for expert")
    expected = {
        "job_id": job_id,
        "candidate_id": job_id,
        "expert": expert,
        "family": canonical_family,
    }
    if request is not None:
        expected["train_request_sha256"] = temporal_train_request_sha256(request)
    for key, value in expected.items():
        if model[key] != value:
            raise CatBoostTrainingError(f"training identity {key} differs")


def _seal_request(request: TrainRequest) -> _SealedRequest:
    metadata = request.model_metadata
    sealed_metadata = None
    if metadata is not None:
        if type(metadata) is not ModelMetadata:
            raise CatBoostTrainingError("TrainRequest model metadata type differs")
        sealed_metadata = _SealedMetadata(
            metadata.n_num_features,
            tuple(metadata.categorical_cardinalities),
            None if metadata.train_x_num is None else _sealed_array(metadata.train_x_num),
            None
            if metadata.piecewise_bin_edges is None
            else tuple(_sealed_array(edge) for edge in metadata.piecewise_bin_edges),
        )
    return _SealedRequest(
        candidate_id=request.candidate_id,
        family=request.family,
        seed=request.seed,
        epochs=request.epochs,
        min_epochs=request.min_epochs,
        model_config_json=_canonical_config(request.model_config, "model config"),
        training_config_json=_canonical_config(request.training_config, "training config"),
        checkpoint_binding_json=_canonical_config(
            request.checkpoint_binding, "checkpoint binding"
        ),
        train=_seal_batch(request.train, "training batch"),
        valid=_seal_batch(request.valid, "validation batch"),
        model_metadata=sealed_metadata,
    )


def _seal_batch(batch: object, label: str) -> _SealedBatch:
    if type(batch) is not FeatureBatch:
        raise CatBoostTrainingError(f"{label} type differs")
    return _SealedBatch(
        _sealed_array(batch.row_id),
        _sealed_array(batch.season),
        _sealed_array(batch.game_type),
        _sealed_array(batch.x_num),
        _sealed_array(batch.x_cat),
        None if batch.y is None else _sealed_array(batch.y),
    )


def _sealed_array(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise CatBoostTrainingError("TrainRequest arrays must be NumPy arrays")
    if value.dtype.hasobject:
        normalized = json.loads(_canonical_json_value(value.tolist()).decode("utf-8"))
        result = np.asarray(normalized, dtype=object).reshape(value.shape)
    else:
        result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


def _restore_request(sealed: _SealedRequest) -> TrainRequest:
    metadata = sealed.model_metadata
    restored_metadata = None
    if metadata is not None:
        restored_metadata = ModelMetadata(
            metadata.n_num_features,
            metadata.categorical_cardinalities,
            None if metadata.train_x_num is None else np.array(metadata.train_x_num, copy=True),
            None
            if metadata.piecewise_bin_edges is None
            else tuple(np.array(edge, copy=True) for edge in metadata.piecewise_bin_edges),
        )
    return TrainRequest(
        candidate_id=sealed.candidate_id,
        family=sealed.family,
        seed=sealed.seed,
        epochs=sealed.epochs,
        model_config=json.loads(sealed.model_config_json),
        training_config=json.loads(sealed.training_config_json),
        train=_restore_batch(sealed.train),
        valid=_restore_batch(sealed.valid),
        min_epochs=sealed.min_epochs,
        checkpoint_binding=json.loads(sealed.checkpoint_binding_json),
        model_metadata=restored_metadata,
    )


def _restore_batch(sealed: _SealedBatch) -> FeatureBatch:
    return FeatureBatch(
        *(np.array(value, copy=True) for value in (
            sealed.row_id, sealed.season, sealed.game_type,
            sealed.x_num, sealed.x_cat,
        )),
        None if sealed.y is None else np.array(sealed.y, copy=True),
    )


def _sealed_request_sha256(request: _SealedRequest) -> str:
    payload = {
        "candidate_id": request.candidate_id,
        "family": request.family,
        "seed": request.seed,
        "epochs": request.epochs,
        "min_epochs": request.min_epochs,
        "model_config": json.loads(request.model_config_json),
        "training_config": json.loads(request.training_config_json),
        "checkpoint_binding": json.loads(request.checkpoint_binding_json),
        "train": _batch_digest_payload(request.train),
        "valid": _batch_digest_payload(request.valid),
        "model_metadata": _metadata_digest_payload(request.model_metadata),
    }
    return hashlib.sha256(_canonical_json_value(payload)).hexdigest()


def _batch_digest_payload(batch: _SealedBatch) -> dict[str, object]:
    return {
        name: _array_digest(value)
        for name, value in (
            ("row_id", batch.row_id), ("season", batch.season),
            ("game_type", batch.game_type), ("x_num", batch.x_num),
            ("x_cat", batch.x_cat), ("y", batch.y),
        )
    }


def _metadata_digest_payload(metadata: _SealedMetadata | None) -> object:
    if metadata is None:
        return None
    return {
        "n_num_features": metadata.n_num_features,
        "categorical_cardinalities": list(metadata.categorical_cardinalities),
        "train_x_num": _array_digest(metadata.train_x_num),
        "piecewise_bin_edges": None
        if metadata.piecewise_bin_edges is None
        else [_array_digest(edge) for edge in metadata.piecewise_bin_edges],
    }


def _array_digest(value: np.ndarray | None) -> object:
    if value is None:
        return None
    payload = (
        _canonical_json_value(value.tolist())
        if value.dtype.hasobject
        else np.ascontiguousarray(value).tobytes()
    )
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _canonical_config(value: object, label: str) -> bytes:
    if not isinstance(value, Mapping):
        raise CatBoostTrainingError(f"TrainRequest {label} must be a mapping")
    return _canonical_json_value(value)


def _canonical_json_value(value: object) -> bytes:
    def plain(item: object) -> object:
        if item is None or type(item) in {bool, str, int}:
            return item
        if isinstance(item, np.generic):
            return plain(item.item())
        if type(item) is float:
            if not np.isfinite(item):
                raise CatBoostTrainingError("TrainRequest JSON contains non-finite values")
            return item
        if isinstance(item, Mapping):
            result: dict[str, object] = {}
            for key, nested in item.items():
                if type(key) is not str or key in result:
                    raise CatBoostTrainingError("TrainRequest JSON keys differ")
                result[key] = plain(nested)
            return result
        if type(item) in {list, tuple}:
            return [plain(nested) for nested in item]
        raise CatBoostTrainingError("TrainRequest JSON contains unsupported values")

    try:
        return json.dumps(
            plain(value), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise CatBoostTrainingError("TrainRequest JSON cannot be canonicalized") from error


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
