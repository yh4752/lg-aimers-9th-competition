from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping


class TreeExpertContractError(ValueError):
    """Raised when the preregistered E1 contract differs."""


DEFAULT_CONTRACT = Path(__file__).with_name("e1_contract.json")
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_TOP_KEYS = {
    "schema_version",
    "campaign_id",
    "review_only",
    "submission_package",
    "official_train_sha256",
    "official_history_sha256",
    "stage_c_delivery_sha256",
    "stage_c_review_sha256",
    "baseline_predictor",
    "fold",
    "seed",
    "candidates",
    "catboost",
    "failure_label_gate",
    "trackman_gate",
    "metrics",
    "gates",
    "budget",
}
_EXPECTED_HASHES = {
    "official_train_sha256": "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff",
    "official_history_sha256": "f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9",
    "stage_c_delivery_sha256": "f054e8e819d1053299fef4749a32e923db9136e20fc9c12cda79bbc7ef8c484a",
    "stage_c_review_sha256": "461c4422cebdc7383577ad6c53b044bfe3d9d15b667383e454591135e4242d2c",
}
_EXPECTED_CANDIDATES = (
    ("c0_native_ctr", "binary", False, False),
    ("c1_anchor_residual", "residual", False, False),
    ("c2_trackman_residual", "residual", True, False),
    ("c3_failure_aware", "multiclass", False, True),
)
_EXPECTED_CATBOOST = {
    "iterations": 800,
    "depth": 8,
    "learning_rate": 0.04,
    "l2_leaf_reg": 5.0,
    "random_strength": 0.5,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.5,
    "border_count": 128,
    "max_ctr_complexity": 2,
    "one_hot_max_size": 16,
    "od_type": "Iter",
    "od_wait": 60,
    "task_type": "GPU",
    "allow_writing_files": True,
}
_EXPECTED_FAILURE_GATE = {
    "minimum_coverage": 0.98,
    "minimum_binary_delta_fraction": 0.999,
    "minimum_success_agreement": 0.999,
    "maximum_middle_reverse_overlap": 0.001,
    "minimum_class_rows": 10000,
    "delta_tolerance": 0.02,
}
_EXPECTED_TRACKMAN_GATE = {"minimum_accepted_coverage": 0.3}
_EXPECTED_METRICS = {
    "bootstrap_repeats": 1000,
    "bootstrap_seed": 3407,
    "minimum_segment_rows": 5000,
}
_EXPECTED_GATES = {
    "stop_if_all_regress_more_than": 0.00005,
    "maximum_promoted": 2,
}
_EXPECTED_BUDGET = {
    "wall_seconds": 14400,
    "new_job_guard_seconds": 600,
    "snapshot_interval_seconds": 300,
}


@dataclass(frozen=True)
class E1Candidate:
    candidate_id: str
    objective: str
    use_trackman: bool
    use_failure_labels: bool


@dataclass(frozen=True)
class E1Job:
    job_id: str
    candidate_id: str
    train_end_year: int
    valid_year: int
    seed: int
    objective: str
    use_trackman: bool
    use_failure_labels: bool


@dataclass(frozen=True)
class E1Contract:
    campaign_id: str
    review_only: bool
    submission_package: bool
    official_train_sha256: str
    official_history_sha256: str
    stage_c_delivery_sha256: str
    stage_c_review_sha256: str
    baseline_predictor: str
    fold: tuple[int, int]
    seed: int
    candidates: tuple[E1Candidate, ...]
    catboost_parameters: Mapping[str, object]
    failure_label_gate: Mapping[str, object]
    trackman_gate: Mapping[str, object]
    bootstrap_repeats: int
    bootstrap_seed: int
    minimum_segment_rows: int
    stop_if_all_regress_more_than: float
    maximum_promoted: int
    wall_seconds: int
    new_job_guard_seconds: int
    snapshot_interval_seconds: int


def _object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TreeExpertContractError(f"{label} must be an object")
    return value


def _exact(value: dict[str, object], expected: dict[str, object], label: str) -> None:
    if set(value) != set(expected) or value != expected:
        raise TreeExpertContractError(f"{label} differ")


def _hash(value: object, label: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise TreeExpertContractError(f"{label} must be a lowercase SHA-256")
    return value


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise TreeExpertContractError(f"cannot read E1 contract: {error}") from error
    return _object(value, "contract")


def _candidate(value: object) -> E1Candidate:
    item = _object(value, "candidate")
    if set(item) != {
        "candidate_id",
        "objective",
        "use_trackman",
        "use_failure_labels",
    }:
        raise TreeExpertContractError("candidate keys differ")
    if type(item["candidate_id"]) is not str or type(item["objective"]) is not str:
        raise TreeExpertContractError("candidate identity differs")
    if type(item["use_trackman"]) is not bool or type(item["use_failure_labels"]) is not bool:
        raise TreeExpertContractError("candidate booleans differ")
    return E1Candidate(
        item["candidate_id"],
        item["objective"],
        item["use_trackman"],
        item["use_failure_labels"],
    )


def load_e1_contract(path: Path = DEFAULT_CONTRACT) -> E1Contract:
    root = _load_json(Path(path))
    if set(root) != _TOP_KEYS:
        raise TreeExpertContractError("contract keys differ")
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise TreeExpertContractError("schema version differs")
    if root["campaign_id"] != "tree_expert_e1_v1":
        raise TreeExpertContractError("campaign identity differs")
    if type(root["review_only"]) is not bool or root["review_only"] is not True:
        raise TreeExpertContractError("review policy differs")
    if type(root["submission_package"]) is not bool or root["submission_package"] is not False:
        raise TreeExpertContractError("submission policy differs")

    hashes = {key: _hash(root[key], key) for key in _EXPECTED_HASHES}
    if hashes != _EXPECTED_HASHES:
        raise TreeExpertContractError("input hashes differ")
    if root["baseline_predictor"] != "single_s3407":
        raise TreeExpertContractError("baseline predictor differs")

    fold = _object(root["fold"], "fold")
    _exact(fold, {"train_end_year": 2023, "valid_year": 2024}, "fold")
    if type(root["seed"]) is not int or root["seed"] != 3407:
        raise TreeExpertContractError("seed differs")

    raw_candidates = root["candidates"]
    if type(raw_candidates) is not list:
        raise TreeExpertContractError("candidates must be a list")
    candidates = tuple(_candidate(value) for value in raw_candidates)
    identity = tuple(
        (item.candidate_id, item.objective, item.use_trackman, item.use_failure_labels)
        for item in candidates
    )
    if identity != _EXPECTED_CANDIDATES:
        raise TreeExpertContractError("candidates differ")

    catboost = _object(root["catboost"], "CatBoost parameters")
    failure_gate = _object(root["failure_label_gate"], "failure label gate")
    trackman_gate = _object(root["trackman_gate"], "TrackMan gate")
    metrics = _object(root["metrics"], "metrics")
    gates = _object(root["gates"], "gates")
    budget = _object(root["budget"], "budget")
    _exact(catboost, _EXPECTED_CATBOOST, "CatBoost parameters")
    _exact(failure_gate, _EXPECTED_FAILURE_GATE, "failure label gate")
    _exact(trackman_gate, _EXPECTED_TRACKMAN_GATE, "TrackMan gate")
    _exact(metrics, _EXPECTED_METRICS, "metrics")
    _exact(gates, _EXPECTED_GATES, "gates")
    _exact(budget, _EXPECTED_BUDGET, "budget")

    return E1Contract(
        campaign_id="tree_expert_e1_v1",
        review_only=True,
        submission_package=False,
        official_train_sha256=hashes["official_train_sha256"],
        official_history_sha256=hashes["official_history_sha256"],
        stage_c_delivery_sha256=hashes["stage_c_delivery_sha256"],
        stage_c_review_sha256=hashes["stage_c_review_sha256"],
        baseline_predictor="single_s3407",
        fold=(2023, 2024),
        seed=3407,
        candidates=candidates,
        catboost_parameters=MappingProxyType(dict(catboost)),
        failure_label_gate=MappingProxyType(dict(failure_gate)),
        trackman_gate=MappingProxyType(dict(trackman_gate)),
        bootstrap_repeats=metrics["bootstrap_repeats"],
        bootstrap_seed=metrics["bootstrap_seed"],
        minimum_segment_rows=metrics["minimum_segment_rows"],
        stop_if_all_regress_more_than=gates["stop_if_all_regress_more_than"],
        maximum_promoted=gates["maximum_promoted"],
        wall_seconds=budget["wall_seconds"],
        new_job_guard_seconds=budget["new_job_guard_seconds"],
        snapshot_interval_seconds=budget["snapshot_interval_seconds"],
    )


def build_e1_jobs(contract: E1Contract) -> tuple[E1Job, ...]:
    if contract.campaign_id != "tree_expert_e1_v1":
        raise TreeExpertContractError("campaign identity differs")
    train_end, valid = contract.fold
    return tuple(
        E1Job(
            job_id=f"e1__{item.candidate_id}__tr{train_end}__va{valid}__s{contract.seed}",
            candidate_id=item.candidate_id,
            train_end_year=train_end,
            valid_year=valid,
            seed=contract.seed,
            objective=item.objective,
            use_trackman=item.use_trackman,
            use_failure_labels=item.use_failure_labels,
        )
        for item in contract.candidates
    )
