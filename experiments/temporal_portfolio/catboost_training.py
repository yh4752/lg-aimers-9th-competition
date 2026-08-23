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


@dataclass(frozen=True)
class TemporalTrainingJob:
    """One immutable-identity temporal training request.

    DataFrames and arrays are snapshotted on construction.  The existing
    ``TrainingIdentity`` remains the only semantic identity type.
    """

    job_id: str
    expert: str
    identity: TrainingIdentity
    sample_weight: np.ndarray = field(repr=False)
    seed: int
    audit_frame: pd.DataFrame = field(repr=False)
    segment_columns: tuple[str, ...] = ()
    train_frame: pd.DataFrame | None = field(default=None, repr=False)
    valid_frame: pd.DataFrame | None = field(default=None, repr=False)
    target: np.ndarray | None = field(default=None, repr=False)
    valid_row_id: np.ndarray | None = field(default=None, repr=False)
    train_request: object | None = field(default=None, repr=False)
    teacher_oof: object | None = field(default=None, repr=False)
    teacher_oof_sha256: str | None = None
    teacher_lambda: float = 0.0

    def __post_init__(self) -> None:
        if type(self.job_id) is not str or not self.job_id or self.job_id != self.job_id.strip():
            raise CatBoostTrainingError("job_id must be a nonempty canonical string")
        if type(self.expert) is not str or self.expert not in _EXPERTS:
            raise CatBoostTrainingError("expert is not preregistered")
        if type(self.identity) is not TrainingIdentity:
            raise CatBoostTrainingError("identity must be an exact TrainingIdentity")
        try:
            audit_duplicate(self.identity, {})
        except ValueError as error:
            raise CatBoostTrainingError("training identity is invalid") from error
        if type(self.seed) is not int or self.seed < 0:
            raise CatBoostTrainingError("seed must be an exact non-negative integer")

        weight = _numeric_vector(self.sample_weight, "sample weight", nonempty=True)
        if np.any(weight <= 0):
            raise CatBoostTrainingError("sample weight must be strictly positive")
        object.__setattr__(self, "sample_weight", weight)

        if type(self.audit_frame) is not pd.DataFrame or self.audit_frame.empty:
            raise CatBoostTrainingError("audit frame must be a nonempty pandas DataFrame")
        if not self.audit_frame.columns.is_unique:
            raise CatBoostTrainingError("audit frame columns must be unique")
        if type(self.segment_columns) is not tuple or any(
            type(column) is not str or not column for column in self.segment_columns
        ):
            raise CatBoostTrainingError("segment columns must be canonical strings")
        if len(set(self.segment_columns)) != len(self.segment_columns):
            raise CatBoostTrainingError("segment columns must be unique")
        required = (*_BASE_AUDIT_COLUMNS, *self.segment_columns)
        if set(self.audit_frame.columns) != set(required):
            raise CatBoostTrainingError("audit frame schema differs")
        audit = self.audit_frame.loc[:, required].copy(deep=True).reset_index(drop=True)
        row_ids = _stable_ids(audit["row_id"].tolist(), "audit row_id")
        target_valid = _binary_vector(audit["target"], "audit target", len(audit))
        audit["target"] = target_valid
        object.__setattr__(self, "audit_frame", audit)

        if self.valid_row_id is None:
            supplied_row_ids = row_ids
        else:
            if not isinstance(self.valid_row_id, np.ndarray):
                raise CatBoostTrainingError("valid row IDs must be a NumPy array")
            supplied_row_ids = _stable_ids(self.valid_row_id.tolist(), "valid row_id")
        if supplied_row_ids != row_ids:
            raise CatBoostTrainingError("valid row order differs from the audit source")
        immutable_ids = np.asarray(supplied_row_ids, dtype=object)
        object.__setattr__(self, "valid_row_id", immutable_ids)

        if self.expert == "catboost":
            if type(self.train_frame) is not pd.DataFrame or self.train_frame.empty:
                raise CatBoostTrainingError("CatBoost train frame must be nonempty")
            if type(self.valid_frame) is not pd.DataFrame or self.valid_frame.empty:
                raise CatBoostTrainingError("CatBoost valid frame must be nonempty")
            if not self.train_frame.columns.is_unique or not self.valid_frame.columns.is_unique:
                raise CatBoostTrainingError("CatBoost feature columns must be unique")
            if tuple(self.train_frame.columns) != tuple(self.valid_frame.columns):
                raise CatBoostTrainingError("CatBoost train and valid columns differ")
            if len(self.valid_frame) != len(row_ids):
                raise CatBoostTrainingError("CatBoost validation rows differ")
            train_target = _binary_vector(self.target, "training target", len(self.train_frame))
            if len(weight) != len(self.train_frame):
                raise CatBoostTrainingError("CatBoost sample weights are not row aligned")
            object.__setattr__(self, "train_frame", self.train_frame.copy(deep=True).reset_index(drop=True))
            object.__setattr__(self, "valid_frame", self.valid_frame.copy(deep=True).reset_index(drop=True))
            object.__setattr__(self, "target", train_target)
            if self.train_request is not None or self.teacher_oof is not None:
                raise CatBoostTrainingError("CatBoost job contains unrelated trainer state")
        else:
            if self.train_request is None:
                raise CatBoostTrainingError("TabM jobs require an existing TrainRequest")
            if len(weight) != len(getattr(self.train_request, "train").row_id):
                raise CatBoostTrainingError("TabM sample weights are not row aligned")
            request_valid_ids = _stable_ids(
                list(getattr(self.train_request, "valid").row_id), "request valid row_id"
            )
            if request_valid_ids != row_ids:
                raise CatBoostTrainingError("TrainRequest validation order differs")
            if self.expert == "tabm" and (
                self.teacher_oof is not None
                or self.teacher_oof_sha256 is not None
                or self.teacher_lambda != 0.0
            ):
                raise CatBoostTrainingError("plain TabM job contains teacher state")
            if self.expert == "lupi":
                if self.teacher_oof is None or not _is_sha256(self.teacher_oof_sha256):
                    raise CatBoostTrainingError("LUPI job requires a sealed teacher OOF hash")
                if isinstance(self.teacher_lambda, bool) or not isinstance(self.teacher_lambda, Real):
                    raise CatBoostTrainingError("teacher lambda must be numeric")
                value = float(self.teacher_lambda)
                if not np.isfinite(value) or not 0.0 < value <= 1.0:
                    raise CatBoostTrainingError("teacher lambda must be in (0, 1]")
                object.__setattr__(self, "teacher_lambda", value)

    @property
    def valid_rows(self) -> int:
        return len(self.audit_frame)

    @property
    def valid_row_ids(self) -> tuple[object, ...]:
        return tuple(self.audit_frame["row_id"].tolist())


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
    tree_count = getattr(model, "tree_count_", max(CATBOOST_PREFIXES))
    if isinstance(tree_count, bool) or not isinstance(tree_count, Integral):
        raise CatBoostTrainingError("backend model tree count is invalid")
    if int(tree_count) != max(CATBOOST_PREFIXES):
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


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
