from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping


class BlendContractError(ValueError):
    """Raised when the sealed blend contract differs."""


DEFAULT_CONTRACT = Path(__file__).with_name("contract.json")
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_FOLDS = ((2022, 2023), (2023, 2024))
_WEIGHTS = (0.9, 0.8, 0.7)
_CATBOOST = {
    "iterations": 400,
    "depth": 7,
    "learning_rate": 0.05,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "border_count": 128,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "l2_leaf_reg": 3.0,
    "model_size_reg": 0.5,
    "max_ctr_complexity": 1,
    "random_seed": 42,
    "task_type": "GPU",
    "od_type": "Iter",
    "od_wait": 50,
}
_STAGE_C_KEYS = {
    "delivery_sha256",
    "review_sha256",
    "resume_sha256",
    "stage_state_sha256",
    "selected_predictor",
    "prediction_members",
}
_TOP_KEYS = {
    "schema_version",
    "campaign_id",
    "review_only",
    "submission_package",
    "official_train_sha256",
    "official_history_sha256",
    "stage_c",
    "folds",
    "preprocessing",
    "catboost",
    "tabm_weights",
    "gates",
    "budget",
}


@dataclass(frozen=True)
class BlendJob:
    job_id: str
    train_end_year: int
    valid_year: int
    seed: int


@dataclass(frozen=True)
class BlendContract:
    campaign_id: str
    review_only: bool
    submission_package: bool
    official_train_sha256: str
    official_history_sha256: str
    stage_c: Mapping[str, object]
    folds: tuple[tuple[int, int], ...]
    preprocessing_components: tuple[str, ...]
    catboost_parameters: Mapping[str, object]
    tabm_weights: tuple[float, ...]
    minimum_weighted_gain: float
    maximum_fold_regression: float
    wall_seconds: int
    new_job_guard_seconds: int
    snapshot_interval_seconds: int
    download_interval_seconds: int


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise BlendContractError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise BlendContractError(f"{label} keys differ")


def _sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise BlendContractError(f"{label} must be a lowercase SHA-256")
    return value


def load_contract(path: str | Path | None = None) -> BlendContract:
    source = DEFAULT_CONTRACT if path is None else Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except Exception as error:
        raise BlendContractError(f"cannot read blend contract: {error}") from error
    root = _mapping(raw, "contract")
    _exact_keys(root, _TOP_KEYS, "contract")
    if (
        root["schema_version"] != 1
        or root["campaign_id"] != "catboost_tabm_blend_v1"
        or root["review_only"] is not True
        or root["submission_package"] is not False
    ):
        raise BlendContractError("contract policy differs")
    train_sha = _sha(root["official_train_sha256"], "train SHA-256")
    history_sha = _sha(root["official_history_sha256"], "history SHA-256")

    stage_c = _mapping(root["stage_c"], "Stage C")
    _exact_keys(stage_c, _STAGE_C_KEYS, "Stage C")
    for key in ("delivery_sha256", "review_sha256", "resume_sha256", "stage_state_sha256"):
        _sha(stage_c[key], f"Stage C {key}")
    members = _mapping(stage_c["prediction_members"], "Stage C prediction members")
    if (
        stage_c["selected_predictor"] != "single_s3407"
        or set(members) != {"2022->2023", "2023->2024"}
        or any(type(value) is not str or not value.startswith("predictions/") for value in members.values())
    ):
        raise BlendContractError("Stage C predictor identity differs")

    folds_raw = root["folds"]
    if type(folds_raw) is not list:
        raise BlendContractError("folds must be a list")
    folds: list[tuple[int, int]] = []
    for item in folds_raw:
        fold = _mapping(item, "fold")
        _exact_keys(fold, {"train_end_year", "valid_year"}, "fold")
        if type(fold["train_end_year"]) is not int or type(fold["valid_year"]) is not int:
            raise BlendContractError("fold years must be integers")
        folds.append((fold["train_end_year"], fold["valid_year"]))
    if tuple(folds) != _FOLDS:
        raise BlendContractError("fold schedule differs")

    preprocessing = _mapping(root["preprocessing"], "preprocessing")
    _exact_keys(preprocessing, {"profile", "components"}, "preprocessing")
    if preprocessing != {"profile": "tree_native", "components": ["hand_matchup"]}:
        raise BlendContractError("preprocessing differs")
    catboost = _mapping(root["catboost"], "CatBoost")
    if catboost != _CATBOOST:
        raise BlendContractError("CatBoost parameters differ")
    if root["tabm_weights"] != list(_WEIGHTS):
        raise BlendContractError("blend weights differ")
    gates = _mapping(root["gates"], "gates")
    _exact_keys(gates, {"minimum_weighted_gain", "maximum_fold_regression"}, "gates")
    if gates != {"minimum_weighted_gain": 0.00003, "maximum_fold_regression": 0.00003}:
        raise BlendContractError("promotion gates differ")
    budget = _mapping(root["budget"], "budget")
    _exact_keys(
        budget,
        {"wall_seconds", "new_job_guard_seconds", "snapshot_interval_seconds", "download_interval_seconds"},
        "budget",
    )
    if budget != {
        "wall_seconds": 10800,
        "new_job_guard_seconds": 900,
        "snapshot_interval_seconds": 300,
        "download_interval_seconds": 1200,
    }:
        raise BlendContractError("budget differs")

    frozen_stage_c = dict(stage_c)
    frozen_stage_c["prediction_members"] = MappingProxyType(dict(members))
    return BlendContract(
        campaign_id="catboost_tabm_blend_v1",
        review_only=True,
        submission_package=False,
        official_train_sha256=train_sha,
        official_history_sha256=history_sha,
        stage_c=MappingProxyType(frozen_stage_c),
        folds=_FOLDS,
        preprocessing_components=("hand_matchup",),
        catboost_parameters=MappingProxyType(dict(_CATBOOST)),
        tabm_weights=_WEIGHTS,
        minimum_weighted_gain=0.00003,
        maximum_fold_regression=0.00003,
        wall_seconds=10800,
        new_job_guard_seconds=900,
        snapshot_interval_seconds=300,
        download_interval_seconds=1200,
    )


def contract_sha256(path: str | Path | None = None) -> str:
    source = DEFAULT_CONTRACT if path is None else Path(path)
    load_contract(source)
    return sha256(source.read_bytes()).hexdigest()


def build_jobs(contract: BlendContract) -> tuple[BlendJob, ...]:
    jobs = tuple(
        BlendJob(
            job_id=f"catboost__hand_matchup__tr{train_year}__va{valid_year}__s42",
            train_end_year=train_year,
            valid_year=valid_year,
            seed=42,
        )
        for train_year, valid_year in contract.folds
    )
    if len(jobs) != 2 or len({job.job_id for job in jobs}) != 2:
        raise BlendContractError("job schedule differs")
    return jobs
