"""Immutable loader for the sealed Stage P row-feature proxy contract."""

from __future__ import annotations

import json
import math
import re
import stat
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from experiments.independent_dl.row_features import ROW_FEATURE_BUNDLES


class RowFeatureContractError(ValueError):
    """Raised when a Stage P row-feature contract is not exactly sealed."""


@dataclass(frozen=True)
class RowFeatureBaseline:
    capacity: str
    k: int
    width: int
    blocks: int
    dropout: float
    num_embedding: str
    loss: str
    scheduler: str
    learning_rate: float
    weight_decay: float
    effective_batch_size: int
    micro_batch_size: int


@dataclass(frozen=True)
class RowFeatureFold:
    train_end_year: int
    valid_year: int


@dataclass(frozen=True)
class RowFeatureSample:
    mode: str
    max_rows: int


@dataclass(frozen=True)
class RowFeatureTraining:
    max_epochs: int
    min_epochs: int
    patience: int


@dataclass(frozen=True)
class RowFeatureBudget:
    wall_seconds: int
    new_job_guard_seconds: int


@dataclass(frozen=True)
class RowFeatureProxyGate:
    mean_delta_max: float
    worst_seed_delta_max: float


@dataclass(frozen=True)
class RowFeatureProxyContract:
    schema_version: int
    stage: str
    review_only: bool
    official_train_sha256: str
    baseline: RowFeatureBaseline
    seeds: tuple[int, ...]
    feature_bundles: tuple[str, ...]
    fold: RowFeatureFold
    sample: RowFeatureSample
    training: RowFeatureTraining
    budget: RowFeatureBudget
    proxy_gate: RowFeatureProxyGate


DEFAULT_ROW_FEATURE_PROXY_CONTRACT = (
    Path(__file__).resolve().parent / "configs" / "row_feature_proxy_v1.json"
)

_TOP_KEYS = {
    "schema_version",
    "stage",
    "review_only",
    "official_train_sha256",
    "baseline",
    "seeds",
    "feature_bundles",
    "fold",
    "sample",
    "training",
    "budget",
    "proxy_gate",
}
_BASELINE_KEYS = {
    "capacity",
    "k",
    "width",
    "blocks",
    "dropout",
    "num_embedding",
    "loss",
    "scheduler",
    "learning_rate",
    "weight_decay",
    "effective_batch_size",
    "micro_batch_size",
}
_FOLD_KEYS = {"train_end_year", "valid_year"}
_SAMPLE_KEYS = {"mode", "max_rows"}
_TRAINING_KEYS = {"max_epochs", "min_epochs", "patience"}
_BUDGET_KEYS = {"wall_seconds", "new_job_guard_seconds"}
_PROXY_GATE_KEYS = {"mean_delta_max", "worst_seed_delta_max"}

_OFFICIAL_TRAIN_SHA256 = (
    "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff"
)
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
_SEALED_BASELINE = RowFeatureBaseline(
    capacity="p2",
    k=32,
    width=512,
    blocks=4,
    dropout=0.1,
    num_embedding="piecewise_linear",
    loss="bce",
    scheduler="plateau",
    learning_rate=0.0006,
    weight_decay=0.0001,
    effective_batch_size=4096,
    micro_batch_size=512,
)
_SEALED_FOLD = RowFeatureFold(train_end_year=2023, valid_year=2024)
_SEALED_SAMPLE = RowFeatureSample(mode="proxy", max_rows=400_000)
_SEALED_TRAINING = RowFeatureTraining(max_epochs=8, min_epochs=3, patience=3)
_SEALED_BUDGET = RowFeatureBudget(wall_seconds=10_800, new_job_guard_seconds=900)
_SEALED_PROXY_GATE = RowFeatureProxyGate(
    mean_delta_max=-0.00003,
    worst_seed_delta_max=0.00005,
)


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RowFeatureContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise RowFeatureContractError(f"non-finite JSON number: {value}")


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise RowFeatureContractError(f"{label} must be an object")
    return value


def _require_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RowFeatureContractError(
            f"{label} keys are invalid; missing={missing}, extra={extra}"
        )


def _require_string(value: Any, label: str) -> str:
    if type(value) is not str:
        raise RowFeatureContractError(f"{label} must be a string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise RowFeatureContractError(f"{label} must be a boolean")
    return value


def _require_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise RowFeatureContractError(f"{label} must be an integer")
    return value


def _require_float(value: Any, label: str) -> float:
    if type(value) is not float:
        raise RowFeatureContractError(f"{label} must be a float")
    if not math.isfinite(value):
        raise RowFeatureContractError(f"{label} must be finite")
    return value


def _require_int_tuple(value: Any, label: str) -> tuple[int, ...]:
    if type(value) is not list:
        raise RowFeatureContractError(f"{label} must be a list")
    result = tuple(_require_int(item, f"{label}[{index}]") for index, item in enumerate(value))
    return result


def _require_string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise RowFeatureContractError(f"{label} must be a list")
    result = tuple(
        _require_string(item, f"{label}[{index}]") for index, item in enumerate(value)
    )
    return result


def _read_contract_bytes(path: str | Path) -> bytes:
    contract_path = Path(path)
    try:
        mode = contract_path.lstat().st_mode
    except FileNotFoundError as error:
        raise RowFeatureContractError(f"contract file is missing: {contract_path}") from error
    except OSError as error:
        raise RowFeatureContractError(f"cannot inspect contract file: {error}") from error
    if stat.S_ISLNK(mode):
        raise RowFeatureContractError("contract file must not be a symlink")
    if not stat.S_ISREG(mode):
        raise RowFeatureContractError("contract file must be a regular file")
    try:
        return contract_path.read_bytes()
    except OSError as error:
        raise RowFeatureContractError(f"cannot read contract file: {error}") from error


def _parse_contract(data: bytes) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RowFeatureContractError(f"contract is not valid UTF-8: {error}") from error
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_object_no_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except RowFeatureContractError:
        raise
    except json.JSONDecodeError as error:
        raise RowFeatureContractError(f"contract contains invalid JSON: {error}") from error
    return _require_object(raw, "contract root")


def _load_baseline(raw: Any) -> RowFeatureBaseline:
    value = _require_object(raw, "baseline")
    _require_keys(value, _BASELINE_KEYS, "baseline")
    baseline = RowFeatureBaseline(
        capacity=_require_string(value["capacity"], "baseline.capacity"),
        k=_require_int(value["k"], "baseline.k"),
        width=_require_int(value["width"], "baseline.width"),
        blocks=_require_int(value["blocks"], "baseline.blocks"),
        dropout=_require_float(value["dropout"], "baseline.dropout"),
        num_embedding=_require_string(
            value["num_embedding"], "baseline.num_embedding"
        ),
        loss=_require_string(value["loss"], "baseline.loss"),
        scheduler=_require_string(value["scheduler"], "baseline.scheduler"),
        learning_rate=_require_float(
            value["learning_rate"], "baseline.learning_rate"
        ),
        weight_decay=_require_float(value["weight_decay"], "baseline.weight_decay"),
        effective_batch_size=_require_int(
            value["effective_batch_size"], "baseline.effective_batch_size"
        ),
        micro_batch_size=_require_int(
            value["micro_batch_size"], "baseline.micro_batch_size"
        ),
    )
    if baseline.k <= 0 or baseline.width <= 0 or baseline.blocks <= 0:
        raise RowFeatureContractError("baseline dimensions must be positive")
    if not 0.0 <= baseline.dropout < 1.0:
        raise RowFeatureContractError("baseline.dropout must be in [0, 1)")
    if baseline.learning_rate <= 0.0 or baseline.weight_decay < 0.0:
        raise RowFeatureContractError("baseline optimizer values are out of range")
    if baseline.effective_batch_size <= 0 or baseline.micro_batch_size <= 0:
        raise RowFeatureContractError("baseline batch sizes must be positive")
    if baseline.effective_batch_size % baseline.micro_batch_size != 0:
        raise RowFeatureContractError(
            "baseline.effective_batch_size must be divisible by micro_batch_size"
        )
    if baseline != _SEALED_BASELINE:
        raise RowFeatureContractError("baseline differs from the sealed Stage P model")
    return baseline


def _load_fold(raw: Any) -> RowFeatureFold:
    value = _require_object(raw, "fold")
    _require_keys(value, _FOLD_KEYS, "fold")
    fold = RowFeatureFold(
        train_end_year=_require_int(value["train_end_year"], "fold.train_end_year"),
        valid_year=_require_int(value["valid_year"], "fold.valid_year"),
    )
    if fold.train_end_year <= 0 or fold.valid_year <= 0:
        raise RowFeatureContractError("fold years must be positive")
    if fold.valid_year != fold.train_end_year + 1:
        raise RowFeatureContractError("fold must be a yearly transition")
    if fold != _SEALED_FOLD:
        raise RowFeatureContractError("fold differs from the sealed Stage P fold")
    return fold


def _load_sample(raw: Any) -> RowFeatureSample:
    value = _require_object(raw, "sample")
    _require_keys(value, _SAMPLE_KEYS, "sample")
    sample = RowFeatureSample(
        mode=_require_string(value["mode"], "sample.mode"),
        max_rows=_require_int(value["max_rows"], "sample.max_rows"),
    )
    if sample.max_rows <= 0:
        raise RowFeatureContractError("sample.max_rows must be positive")
    if sample != _SEALED_SAMPLE:
        raise RowFeatureContractError("sample differs from the sealed Stage P proxy sample")
    return sample


def _load_training(raw: Any) -> RowFeatureTraining:
    value = _require_object(raw, "training")
    _require_keys(value, _TRAINING_KEYS, "training")
    training = RowFeatureTraining(
        max_epochs=_require_int(value["max_epochs"], "training.max_epochs"),
        min_epochs=_require_int(value["min_epochs"], "training.min_epochs"),
        patience=_require_int(value["patience"], "training.patience"),
    )
    if min(training.max_epochs, training.min_epochs, training.patience) <= 0:
        raise RowFeatureContractError("training values must be positive")
    if training.min_epochs > training.max_epochs:
        raise RowFeatureContractError("training.min_epochs must not exceed max_epochs")
    if training != _SEALED_TRAINING:
        raise RowFeatureContractError("training differs from the sealed Stage P schedule")
    return training


def _load_budget(raw: Any) -> RowFeatureBudget:
    value = _require_object(raw, "budget")
    _require_keys(value, _BUDGET_KEYS, "budget")
    budget = RowFeatureBudget(
        wall_seconds=_require_int(value["wall_seconds"], "budget.wall_seconds"),
        new_job_guard_seconds=_require_int(
            value["new_job_guard_seconds"], "budget.new_job_guard_seconds"
        ),
    )
    if budget.wall_seconds <= 0 or budget.new_job_guard_seconds <= 0:
        raise RowFeatureContractError("budget values must be positive")
    if budget.new_job_guard_seconds >= budget.wall_seconds:
        raise RowFeatureContractError(
            "budget.new_job_guard_seconds must be less than wall_seconds"
        )
    if budget != _SEALED_BUDGET:
        raise RowFeatureContractError("budget differs from the sealed Stage P budget")
    return budget


def _load_proxy_gate(raw: Any) -> RowFeatureProxyGate:
    value = _require_object(raw, "proxy_gate")
    _require_keys(value, _PROXY_GATE_KEYS, "proxy_gate")
    gate = RowFeatureProxyGate(
        mean_delta_max=_require_float(
            value["mean_delta_max"], "proxy_gate.mean_delta_max"
        ),
        worst_seed_delta_max=_require_float(
            value["worst_seed_delta_max"], "proxy_gate.worst_seed_delta_max"
        ),
    )
    if gate.mean_delta_max > 0.0 or gate.worst_seed_delta_max < 0.0:
        raise RowFeatureContractError("proxy_gate thresholds are out of range")
    if gate != _SEALED_PROXY_GATE:
        raise RowFeatureContractError("proxy_gate differs from the sealed Stage P gate")
    return gate


def _validate_contract(data: bytes) -> RowFeatureProxyContract:
    raw = _parse_contract(data)
    _require_keys(raw, _TOP_KEYS, "contract")

    schema_version = _require_int(raw["schema_version"], "schema_version")
    stage = _require_string(raw["stage"], "stage")
    review_only = _require_bool(raw["review_only"], "review_only")
    official_train_sha256 = _require_string(
        raw["official_train_sha256"], "official_train_sha256"
    )
    if _LOWER_SHA256.fullmatch(official_train_sha256) is None:
        raise RowFeatureContractError(
            "official_train_sha256 must be 64 lowercase hexadecimal characters"
        )

    baseline = _load_baseline(raw["baseline"])
    seeds = _require_int_tuple(raw["seeds"], "seeds")
    feature_bundles = _require_string_tuple(raw["feature_bundles"], "feature_bundles")
    fold = _load_fold(raw["fold"])
    sample = _load_sample(raw["sample"])
    training = _load_training(raw["training"])
    budget = _load_budget(raw["budget"])
    proxy_gate = _load_proxy_gate(raw["proxy_gate"])

    if schema_version != 1:
        raise RowFeatureContractError("schema_version must be 1")
    if stage != "P":
        raise RowFeatureContractError("stage must be P")
    if review_only is not True:
        raise RowFeatureContractError("review_only must be true")
    if official_train_sha256 != _OFFICIAL_TRAIN_SHA256:
        raise RowFeatureContractError("official_train_sha256 differs from the sealed source")
    if seeds != (42, 3407):
        raise RowFeatureContractError("seeds must equal the ordered pair (42, 3407)")
    if any(seed <= 0 for seed in seeds):
        raise RowFeatureContractError("seeds must be positive")
    if feature_bundles != ROW_FEATURE_BUNDLES:
        raise RowFeatureContractError(
            "feature_bundles must equal ROW_FEATURE_BUNDLES in order"
        )

    return RowFeatureProxyContract(
        schema_version=schema_version,
        stage=stage,
        review_only=review_only,
        official_train_sha256=official_train_sha256,
        baseline=baseline,
        seeds=seeds,
        feature_bundles=feature_bundles,
        fold=fold,
        sample=sample,
        training=training,
        budget=budget,
        proxy_gate=proxy_gate,
    )


def load_row_feature_proxy_contract(
    path: str | Path = DEFAULT_ROW_FEATURE_PROXY_CONTRACT,
) -> RowFeatureProxyContract:
    """Load and validate the immutable, sealed Stage P proxy contract."""

    return _validate_contract(_read_contract_bytes(path))


def row_feature_contract_sha256(
    path: str | Path = DEFAULT_ROW_FEATURE_PROXY_CONTRACT,
) -> str:
    """Validate a Stage P contract, then hash the exact validated file bytes."""

    data = _read_contract_bytes(path)
    _validate_contract(data)
    return sha256(data).hexdigest()
