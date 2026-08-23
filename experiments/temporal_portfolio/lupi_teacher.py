"""Leakage-safe cross-fitted TrackMan teacher probabilities for LUPI."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
from numbers import Integral, Real
from types import MappingProxyType
from typing import Protocol

import numpy as np
import pandas as pd


class TeacherError(ValueError):
    """Raised when teacher probabilities cannot be produced or audited safely."""


TEACHER_FEATURES = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
)
TEACHER_INPUT_COLUMNS = (
    "row_id",
    "pitcher_id",
    "control_success",
    "lupi_match_accepted",
    *TEACHER_FEATURES,
)


class TeacherBackend(Protocol):
    """Minimal backend boundary used by the group cross-fit."""

    def fit(
        self, features: pd.DataFrame, target: pd.Series, *, seed: int
    ) -> object: ...

    def predict(self, model: object, features: pd.DataFrame) -> object: ...


@dataclass(frozen=True, init=False)
class TeacherOOF:
    """Sealed, row-aligned OOF probabilities with integrity evidence."""

    _row_ids: tuple[object, ...] = field(repr=False)
    _probability_bytes: bytes = field(repr=False)
    _fold_items: tuple[tuple[object, int], ...] = field(repr=False)
    _predicted_items: tuple[tuple[object, int], ...] = field(repr=False)
    _metadata_json: str = field(repr=False)
    _backend_evidence_json: str = field(repr=False)
    _sha256: str = field(repr=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("TeacherOOF instances must be created by crossfit_teacher()")

    @classmethod
    def _from_validated(
        cls,
        *,
        row_ids: tuple[object, ...],
        probabilities: np.ndarray,
        fold_items: tuple[tuple[object, int], ...],
        predicted_items: tuple[tuple[object, int], ...],
        metadata: Mapping[str, object],
        backend_evidence: Mapping[str, object],
    ) -> TeacherOOF:
        instance = object.__new__(cls)
        probability_bytes = np.asarray(probabilities, dtype="float64").tobytes()
        metadata_json = _canonical_json(metadata, "teacher metadata")
        backend_json = _canonical_json(backend_evidence, "backend evidence")
        object.__setattr__(instance, "_row_ids", row_ids)
        object.__setattr__(instance, "_probability_bytes", probability_bytes)
        object.__setattr__(instance, "_fold_items", fold_items)
        object.__setattr__(instance, "_predicted_items", predicted_items)
        object.__setattr__(instance, "_metadata_json", metadata_json)
        object.__setattr__(instance, "_backend_evidence_json", backend_json)
        object.__setattr__(instance, "_sha256", _oof_digest(instance))
        _validate_teacher_oof(instance)
        return instance

    @property
    def row_ids(self) -> tuple[object, ...]:
        _validate_teacher_oof(self)
        return tuple(self._row_ids)

    @property
    def probability(self) -> pd.Series:
        values = _validate_teacher_oof(self)
        return pd.Series(values.copy(), index=list(self._row_ids), name="teacher_probability")

    @property
    def probability_by_row_id(self) -> Mapping[object, float]:
        values = _validate_teacher_oof(self)
        return MappingProxyType(dict(zip(self._row_ids, values, strict=True)))

    @property
    def fold_by_row_id(self) -> Mapping[object, int]:
        _validate_teacher_oof(self)
        return MappingProxyType(dict(self._fold_items))

    @property
    def predicted_by_fold(self) -> Mapping[object, int]:
        _validate_teacher_oof(self)
        return MappingProxyType(dict(self._predicted_items))

    @property
    def metadata(self) -> Mapping[str, object]:
        _validate_teacher_oof(self)
        return _json_mapping(self._metadata_json)

    @property
    def backend_evidence(self) -> Mapping[str, object]:
        _validate_teacher_oof(self)
        return _json_mapping(self._backend_evidence_json)


def teacher_fold(pitcher_id: object, seed: int, folds: int) -> int:
    """Assign one canonical pitcher ID to a deterministic SHA-256 fold."""

    _validate_seed_and_folds(seed, folds)
    canonical_id = _canonical_id(pitcher_id, "pitcher_id")
    digest = sha256(f"{seed}|{canonical_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def crossfit_teacher(
    rows: pd.DataFrame,
    *,
    seed: int = 3407,
    folds: int = 5,
    backend: TeacherBackend,
) -> TeacherOOF:
    """Predict every accepted matched row from a disjoint pitcher-held-out model."""

    _validate_seed_and_folds(seed, folds)
    prepared = _validated_rows(rows)
    if not callable(getattr(backend, "fit", None)) or not callable(
        getattr(backend, "predict", None)
    ):
        raise TeacherError("backend must provide callable fit and predict methods")

    assignments = np.asarray(
        [teacher_fold(value, seed, folds) for value in prepared["pitcher_id"]],
        dtype="int64",
    )
    used_folds = tuple(sorted(set(assignments.tolist())))
    if len(used_folds) < 2:
        raise TeacherError(
            "teacher split requires disjoint nonempty train and validation groups"
        )

    probabilities = np.full(len(prepared), np.nan, dtype="float64")
    predicted = np.zeros(len(prepared), dtype="int8")
    fold_by_row_id = tuple(
        (row_id, int(fold))
        for row_id, fold in zip(prepared["row_id"], assignments, strict=True)
    )
    predicted_items: list[tuple[object, int]] = []
    for fold in used_folds:
        validation_mask = assignments == fold
        training_mask = ~validation_mask
        if not validation_mask.any() or not training_mask.any():
            raise TeacherError(
                "teacher split requires disjoint nonempty train and validation groups"
            )
        train_pitchers = {
            _canonical_id(value, "pitcher_id")
            for value in prepared.loc[training_mask, "pitcher_id"]
        }
        valid_pitchers = {
            _canonical_id(value, "pitcher_id")
            for value in prepared.loc[validation_mask, "pitcher_id"]
        }
        if not train_pitchers.isdisjoint(valid_pitchers):
            raise TeacherError("teacher train and validation pitcher groups overlap")

        train_features = prepared.loc[training_mask, TEACHER_FEATURES].copy(deep=True)
        train_target = prepared.loc[training_mask, "control_success"].copy(deep=True)
        valid_features = prepared.loc[validation_mask, TEACHER_FEATURES].copy(deep=True)
        model = backend.fit(train_features, train_target, seed=seed + int(fold))
        predicted_probability = _validated_probabilities(
            backend.predict(model, valid_features),
            expected_length=int(validation_mask.sum()),
        )
        probabilities[validation_mask] = predicted_probability
        predicted[validation_mask] += 1
        predicted_items.extend(
            (row_id, int(fold))
            for row_id in prepared.loc[validation_mask, "row_id"]
        )

    if not np.equal(predicted, 1).all() or not np.isfinite(probabilities).all():
        raise TeacherError("every teacher row must be predicted exactly once")
    if not np.logical_and(probabilities >= 0.0, probabilities <= 1.0).all():
        raise TeacherError("teacher probabilities must be within [0, 1]")

    evidence = _backend_evidence(backend)
    result = TeacherOOF._from_validated(
        row_ids=tuple(prepared["row_id"]),
        probabilities=probabilities,
        fold_items=fold_by_row_id,
        predicted_items=tuple(predicted_items),
        metadata={
            "seed": seed,
            "folds": folds,
            "used_folds": list(used_folds),
            "row_count": len(prepared),
            "features": list(TEACHER_FEATURES),
        },
        backend_evidence=evidence,
    )
    return result


def build_teacher_vector(
    all_row_ids: Iterable[object], teacher_oof: TeacherOOF
) -> np.ndarray:
    """Align teacher OOF values to student rows, leaving unmatched rows as NaN."""

    if type(all_row_ids) in (str, bytes):
        raise TeacherError("requested row IDs must be an iterable of stable scalar IDs")
    try:
        requested = tuple(all_row_ids)
    except TypeError as error:
        raise TeacherError("requested row IDs must be iterable") from error
    requested_keys = [_canonical_id(value, "requested row_id") for value in requested]
    if len(requested_keys) != len(set(requested_keys)):
        raise TeacherError("requested row IDs contain duplicates")
    if type(teacher_oof) is not TeacherOOF:
        raise TeacherError("teacher_oof must be a validated TeacherOOF")
    teacher_values = teacher_oof.probability_by_row_id
    teacher_keys = {
        _canonical_id(value, "teacher row_id"): value for value in teacher_values
    }
    if len(teacher_keys) != len(teacher_values):
        raise TeacherError("teacher row IDs contain duplicates")
    unknown = set(teacher_keys) - set(requested_keys)
    if unknown:
        raise TeacherError("teacher OOF contains unknown requested row IDs")
    result = np.full(len(requested), np.nan, dtype="float32")
    for index, key in enumerate(requested_keys):
        source_id = teacher_keys.get(key)
        if source_id is not None:
            result[index] = np.float32(teacher_values[source_id])
    return result


class CatBoostTeacherBackend:
    """Fixed CatBoost classifier adapter with auditable CPU/GPU selection."""

    _BASE_CONFIG = (
        ("iterations", 600),
        ("depth", 6),
        ("learning_rate", 0.05),
        ("l2_leaf_reg", 3.0),
        ("loss_function", "Logloss"),
        ("eval_metric", "Logloss"),
        ("verbose", False),
        ("allow_writing_files", False),
    )

    def __init__(self, *, device: str = "CPU") -> None:
        if type(device) is not str or device not in ("CPU", "GPU"):
            raise TeacherError("CatBoost teacher device must be exactly CPU or GPU")
        self._device = device
        self._fit_seeds: list[int] = []

    @property
    def config(self) -> Mapping[str, object]:
        return MappingProxyType(dict((*self._BASE_CONFIG, ("task_type", self._device))))

    @property
    def evidence(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "backend": "catboost",
                "device": self._device,
                "config": dict(self.config),
                "fit_seeds": tuple(self._fit_seeds),
                "early_stopping": False,
                "model_bytes_serialized": False,
            }
        )

    def fit(
        self, features: pd.DataFrame, target: pd.Series, *, seed: int
    ) -> object:
        _validate_seed(seed)
        _validate_backend_features(features)
        _validated_binary(target, "teacher target", expected_length=len(features))
        try:
            from catboost import CatBoostClassifier
        except ImportError as error:
            raise TeacherError(
                "CatBoost is required only when CatBoostTeacherBackend.fit() is used"
            ) from error
        kwargs = dict(self.config)
        kwargs["random_seed"] = seed
        model = CatBoostClassifier(**kwargs)
        model.fit(features, target)
        self._fit_seeds.append(seed)
        return model

    def predict(self, model: object, features: pd.DataFrame) -> np.ndarray:
        _validate_backend_features(features)
        predictor = getattr(model, "predict_proba", None)
        if not callable(predictor):
            raise TeacherError("CatBoost teacher model must provide predict_proba")
        raw = np.asarray(predictor(features))
        if raw.ndim != 2 or raw.shape != (len(features), 2):
            raise TeacherError("CatBoost predict_proba returned an invalid probability shape")
        return _validated_probabilities(raw[:, 1], expected_length=len(features))


def _validated_rows(rows: object) -> pd.DataFrame:
    if type(rows) is not pd.DataFrame:
        raise TeacherError("teacher rows must be exactly a pandas DataFrame")
    if rows.empty:
        raise TeacherError("teacher rows must not be empty")
    if not rows.columns.is_unique or set(rows.columns) != set(TEACHER_INPUT_COLUMNS):
        raise TeacherError("teacher row schema differs from the exact required schema")
    result = rows.loc[:, TEACHER_INPUT_COLUMNS].copy(deep=True)
    row_keys = [_canonical_id(value, "row_id") for value in result["row_id"]]
    if len(row_keys) != len(set(row_keys)):
        raise TeacherError("teacher row_id values must be unique")
    pitcher_keys = [
        _canonical_id(value, "pitcher_id") for value in result["pitcher_id"]
    ]
    if not pitcher_keys:
        raise TeacherError("teacher pitcher_id values must not be empty")
    _validated_binary(
        result["control_success"], "control_success", expected_length=len(result)
    )
    accepted = _validated_binary(
        result["lupi_match_accepted"],
        "lupi_match_accepted",
        expected_length=len(result),
    )
    if not np.equal(accepted, 1).all():
        raise TeacherError("teacher rows must all be accepted matched training rows")
    for column in TEACHER_FEATURES:
        try:
            values = pd.to_numeric(result[column], errors="raise").to_numpy(
                dtype="float64"
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise TeacherError(f"teacher feature {column} must be numeric") from error
        if np.isinf(values).any():
            raise TeacherError(f"teacher feature {column} must not contain infinity")
        result[column] = values
    return result


def _validated_binary(
    values: object, label: str, *, expected_length: int
) -> np.ndarray:
    if not isinstance(values, (pd.Series, np.ndarray, list, tuple)):
        raise TeacherError(f"{label} must contain exact binary integers")
    try:
        snapshot = list(values)
    except TypeError as error:
        raise TeacherError(f"{label} must contain exact binary integers") from error
    if len(snapshot) != expected_length or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) not in (0, 1)
        for value in snapshot
    ):
        raise TeacherError(f"{label} must contain exact binary integers")
    return np.asarray(snapshot, dtype="int8")


def _validated_probabilities(values: object, *, expected_length: int) -> np.ndarray:
    try:
        result = np.asarray(values, dtype="float64")
    except (TypeError, ValueError, OverflowError) as error:
        raise TeacherError("backend probabilities must be numeric") from error
    if result.ndim != 1 or len(result) != expected_length:
        raise TeacherError("backend probability length or shape is invalid")
    if not np.isfinite(result).all():
        raise TeacherError("backend probabilities must be finite")
    if not np.logical_and(result >= 0.0, result <= 1.0).all():
        raise TeacherError("backend probabilities must be within [0, 1]")
    return result.copy()


def _validate_backend_features(features: object) -> None:
    if type(features) is not pd.DataFrame or tuple(features.columns) != TEACHER_FEATURES:
        raise TeacherError("backend features must contain only TEACHER_FEATURES")
    if features.empty:
        raise TeacherError("backend features must not be empty")


def _canonical_id(value: object, label: str) -> str:
    if isinstance(value, (bool, np.bool_)):
        raise TeacherError(f"{label} must be a stable scalar ID, not bool")
    if isinstance(value, Integral):
        return f"int:{int(value)}"
    if type(value) is str:
        if not value or value != value.strip():
            raise TeacherError(f"{label} must be a nonempty canonical string")
        return f"str:{len(value.encode('utf-8'))}:{value}"
    if isinstance(value, Real) and not np.isfinite(value):
        raise TeacherError(f"{label} must not be NaN or infinite")
    raise TeacherError(
        f"{label} must be an integer or string without ambiguous coercion"
    )


def _validate_seed_and_folds(seed: object, folds: object) -> None:
    _validate_seed(seed)
    if type(folds) is not int or folds < 2:
        raise TeacherError("folds must be an exact integer of at least 2")


def _validate_seed(seed: object) -> None:
    if type(seed) is not int or seed < 0:
        raise TeacherError("seed must be an exact non-negative integer")


def _backend_evidence(backend: object) -> Mapping[str, object]:
    try:
        evidence = getattr(backend, "evidence")
    except AttributeError:
        return {"backend_type": type(backend).__qualname__}
    if not isinstance(evidence, Mapping):
        raise TeacherError("backend evidence must be a mapping")
    _canonical_json(evidence, "backend evidence")
    return dict(evidence)


def _canonical_json(value: object, label: str) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise TeacherError(f"{label} must contain JSON-compatible finite values") from error


def _json_mapping(value: str) -> Mapping[str, object]:
    decoded = json.loads(value)
    if type(decoded) is not dict:
        raise TeacherError("stored teacher evidence is not a mapping")
    return MappingProxyType(decoded)


def _oof_digest(value: TeacherOOF) -> str:
    digest = sha256()
    for row_id in value._row_ids:
        encoded = _canonical_id(row_id, "teacher row_id").encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    digest.update(value._probability_bytes)
    for items in (value._fold_items, value._predicted_items):
        for row_id, fold in items:
            encoded = _canonical_id(row_id, "teacher row_id").encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            digest.update(int(fold).to_bytes(8, "big", signed=True))
    digest.update(value._metadata_json.encode("utf-8"))
    digest.update(value._backend_evidence_json.encode("utf-8"))
    return digest.hexdigest()


def _validate_teacher_oof(value: object) -> np.ndarray:
    if type(value) is not TeacherOOF:
        raise TeacherError("teacher OOF has an invalid type")
    try:
        row_ids = value._row_ids
        probability_bytes = value._probability_bytes
        fold_items = value._fold_items
        predicted_items = value._predicted_items
        metadata_json = value._metadata_json
        backend_evidence_json = value._backend_evidence_json
        stored_digest = value._sha256
    except AttributeError as error:
        raise TeacherError("teacher OOF is incomplete") from error
    if type(row_ids) is not tuple or not row_ids:
        raise TeacherError("teacher OOF row IDs are invalid")
    keys = [_canonical_id(row_id, "teacher row_id") for row_id in row_ids]
    if len(keys) != len(set(keys)):
        raise TeacherError("teacher OOF contains duplicate row IDs")
    if type(probability_bytes) is not bytes or len(probability_bytes) != 8 * len(row_ids):
        raise TeacherError("teacher OOF probability storage is invalid")
    probabilities = np.frombuffer(probability_bytes, dtype="float64")
    _validated_probabilities(probabilities, expected_length=len(row_ids))
    for label, items in (
        ("fold", fold_items),
        ("predicted", predicted_items),
    ):
        if type(items) is not tuple or len(items) != len(row_ids):
            raise TeacherError(f"teacher OOF {label} evidence length differs")
        item_keys = [_canonical_id(row_id, "teacher row_id") for row_id, _ in items]
        if len(item_keys) != len(set(item_keys)) or set(item_keys) != set(keys):
            raise TeacherError(f"teacher OOF {label} evidence row IDs differ")
        if any(type(fold) is not int or fold < 0 for _, fold in items):
            raise TeacherError(f"teacher OOF {label} evidence has an invalid fold")
    if dict(fold_items) != dict(predicted_items):
        raise TeacherError("teacher OOF prediction and assignment evidence differ")
    if type(metadata_json) is not str or type(backend_evidence_json) is not str:
        raise TeacherError("teacher OOF evidence storage is invalid")
    if type(stored_digest) is not str or stored_digest != _oof_digest(value):
        raise TeacherError("teacher OOF integrity digest differs")
    _json_mapping(metadata_json)
    _json_mapping(backend_evidence_json)
    return probabilities
