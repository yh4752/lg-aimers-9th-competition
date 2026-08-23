"""Strict, selection-free CatBoost prefix evidence for temporal folds."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
from numbers import Integral, Real
from types import MappingProxyType
from typing import Protocol

import numpy as np
import pandas as pd

from .identity import TrainingIdentity, audit_duplicate


class CatBoostTrainingError(ValueError):
    """Raised before malformed CatBoost evidence can be used or published."""


CATBOOST_PREFIXES = (16, 64, 192, 384)
_EXPERTS = frozenset({"catboost", "tabm", "lupi"})
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
    train_request: object | None = field(repr=False)
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
        train_request: object | None = None,
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

        weight = _numeric_vector(sample_weight, "sample weight", nonempty=True)
        if np.any(weight <= 0):
            raise CatBoostTrainingError("sample weight must be strictly positive")

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
        target_valid = _binary_vector(audit["target"], "audit target", len(audit))
        audit["target"] = target_valid

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
            if train_request is None:
                raise CatBoostTrainingError("TabM jobs require an existing TrainRequest")
            if len(weight) != len(getattr(train_request, "train").row_id):
                raise CatBoostTrainingError("TabM sample weights are not row aligned")
            request_valid_ids = _stable_ids(
                list(getattr(train_request, "valid").row_id), "request valid row_id"
            )
            if request_valid_ids != row_ids:
                raise CatBoostTrainingError("TrainRequest validation order differs")
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

        object.__setattr__(self, "job_id", job_id)
        object.__setattr__(self, "expert", expert)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "segment_columns", segment_columns)
        object.__setattr__(self, "train_request", train_request)
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


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
