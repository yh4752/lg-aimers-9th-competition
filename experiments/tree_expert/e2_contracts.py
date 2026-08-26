from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Sequence


class E2ContractError(ValueError):
    """Raised when the preregistered E2 campaign contract differs."""


DEFAULT_E2_CONTRACT = Path(__file__).with_name("e2_contract.json")
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_TOP_KEYS = {
    "schema_version",
    "campaign_id",
    "review_only",
    "submission_package",
    "inputs",
    "promoted_structures",
    "folds",
    "seeds",
    "tabm",
    "catboost",
    "blend_candidates",
    "gates",
    "runtime",
}
_EXPECTED_INPUTS = {
    "official_train_sha256": "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff",
    "official_history_sha256": "f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9",
    "e1_handoff_sha256": "9e5b55d88715b7a76051d6e8497ffd19617a7a4b25fdafeb7f5f5af16b72fffc",
    "e1_review_sha256": "85a2dcf1a01a4c665c46d075571a4ffb8f463a981f49c22a2f3d73824ae4c2d2",
    "e1_resume_sha256": "66065feae86c0c252fdce1b33cfb0fb56c5f86811a7fc9a40b8c0e2a17af8d3c",
    "stage_c_delivery_sha256": "f054e8e819d1053299fef4749a32e923db9136e20fc9c12cda79bbc7ef8c484a",
    "stage_c_review_sha256": "461c4422cebdc7383577ad6c53b044bfe3d9d15b667383e454591135e4242d2c",
    "tabm_submission_sha256": "ee4d6324eb8d1afec08157526db473834620aabac9647a865e9ff6a6cd1345be",
    "tabm_weight_sha256": "940c358c7e4af258ffec957a8ea42a438f3e639ffc6b5b6f777f77f45db9c945",
}
_EXPECTED_STRUCTURES = ("c1_anchor_residual", "c2_trackman_residual")
_EXPECTED_FOLDS = ((2021, 2022), (2022, 2023), (2023, 2024))
_EXPECTED_SEEDS = (42, 2026, 3407)
_EXPECTED_TABM = {
    "capacity": "p2",
    "k": 32,
    "width": 512,
    "blocks": 4,
    "dropout": 0.1,
    "num_embedding": "piecewise_linear",
    "loss": "bce",
    "scheduler": "plateau",
    "learning_rate": 0.0006,
    "max_epochs": 40,
    "min_epochs": 3,
    "patience": 10,
    "effective_batch_size": 4096,
    "micro_batch_size": 512,
}
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
_EXPECTED_BLENDS = (
    ("catboost", 1.0),
    ("probability", 0.1),
    ("probability", 0.2),
    ("probability", 0.3),
    ("logit", 0.1),
    ("logit", 0.2),
    ("logit", 0.3),
)
_EXPECTED_GATES = {
    "structure_weighted_gain": 0.00010,
    "structure_f3_gain": 0.00010,
    "structure_max_fold_regression": 0.00010,
    "seed_max_weighted_regression": 0.00010,
    "blend_min_sequential_gain": 0.00001,
    "blend_max_fold_regression": 0.00005,
    "accept_weighted_gain": 0.00015,
    "accept_f3_gain": 0.00010,
    "accept_max_fold_regression": 0.00005,
    "accept_max_segment_regression": 0.00050,
    "bootstrap_repeats": 1000,
    "bootstrap_seed": 3407,
    "minimum_segment_rows": 5000,
}
_EXPECTED_RUNTIME = {
    "wall_seconds": 21600,
    "new_job_guard_seconds": 600,
    "snapshot_interval_seconds": 600,
    "inference_rows": 245789,
    "inference_max_seconds": 480,
    "rss_max_bytes": 23622320128,
    "gpu_max_bytes": 21474836480,
    "probability_tolerance": 0.000001,
}


@dataclass(frozen=True)
class BlendCandidate:
    method: str
    catboost_weight: float


@dataclass(frozen=True)
class E2Job:
    job_id: str
    candidate_id: str
    train_end_year: int
    valid_year: int
    seed: int
    objective: str
    use_trackman: bool


@dataclass(frozen=True)
class E2Contract:
    source_path: Path
    campaign_id: str
    review_only: bool
    submission_package: bool
    input_hashes: Mapping[str, str]
    structures: tuple[str, ...]
    folds: tuple[tuple[int, int], ...]
    seeds: tuple[int, ...]
    tabm: Mapping[str, object]
    catboost: Mapping[str, object]
    blend_candidates: tuple[BlendCandidate, ...]
    gates: Mapping[str, object]
    runtime: Mapping[str, object]

    @property
    def wall_seconds(self) -> int:
        return int(self.runtime["wall_seconds"])

    @property
    def new_job_guard_seconds(self) -> int:
        return int(self.runtime["new_job_guard_seconds"])

    @property
    def snapshot_interval_seconds(self) -> int:
        return int(self.runtime["snapshot_interval_seconds"])


def _object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise E2ContractError(f"{label} must be an object")
    return value


def _exact(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise E2ContractError(f"{label} differ")


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise E2ContractError(f"cannot read E2 contract: {error}") from error
    return _object(value, "contract")


def _folds(value: object) -> tuple[tuple[int, int], ...]:
    if type(value) is not list:
        raise E2ContractError("folds differ")
    output: list[tuple[int, int]] = []
    for item in value:
        if (
            type(item) is not list
            or len(item) != 2
            or any(type(year) is not int for year in item)
            or item[1] != item[0] + 1
        ):
            raise E2ContractError("folds differ")
        output.append((item[0], item[1]))
    return tuple(output)


def _blends(value: object) -> tuple[tuple[str, float], ...]:
    if type(value) is not list:
        raise E2ContractError("blend candidates differ")
    output: list[tuple[str, float]] = []
    for item in value:
        if (
            type(item) is not list
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) not in {int, float}
            or type(item[1]) is bool
            or not math.isfinite(float(item[1]))
        ):
            raise E2ContractError("blend candidates differ")
        output.append((item[0], float(item[1])))
    return tuple(output)


def load_e2_contract(path: Path = DEFAULT_E2_CONTRACT) -> E2Contract:
    source_path = Path(path)
    root = _load(source_path)
    if set(root) != _TOP_KEYS:
        raise E2ContractError("contract keys differ")
    if root["schema_version"] != 1 or type(root["schema_version"]) is not int:
        raise E2ContractError("schema version differs")
    if root["campaign_id"] != "tree_expert_e2_v1":
        raise E2ContractError("campaign identity differs")
    if root["review_only"] is not False or root["submission_package"] is not False:
        raise E2ContractError("artifact policy differs")

    inputs = _object(root["inputs"], "inputs")
    if inputs != _EXPECTED_INPUTS or any(
        _SHA_RE.fullmatch(str(value)) is None for value in inputs.values()
    ):
        raise E2ContractError("input hashes differ")
    structures = tuple(root["promoted_structures"]) if type(root["promoted_structures"]) is list else ()
    _exact(structures, _EXPECTED_STRUCTURES, "promoted structures")
    folds = _folds(root["folds"])
    _exact(folds, _EXPECTED_FOLDS, "folds")
    seeds = tuple(root["seeds"]) if type(root["seeds"]) is list else ()
    _exact(seeds, _EXPECTED_SEEDS, "seeds")
    tabm = _object(root["tabm"], "TabM parameters")
    catboost = _object(root["catboost"], "CatBoost parameters")
    gates = _object(root["gates"], "gates")
    runtime = _object(root["runtime"], "runtime")
    _exact(tabm, _EXPECTED_TABM, "TabM parameters")
    _exact(catboost, _EXPECTED_CATBOOST, "CatBoost parameters")
    blends = _blends(root["blend_candidates"])
    _exact(blends, _EXPECTED_BLENDS, "blend candidates")
    _exact(gates, _EXPECTED_GATES, "gates")
    if runtime != _EXPECTED_RUNTIME:
        raise E2ContractError("runtime differs")
    return E2Contract(
        source_path=source_path,
        campaign_id="tree_expert_e2_v1",
        review_only=False,
        submission_package=False,
        input_hashes=MappingProxyType(dict(inputs)),
        structures=structures,
        folds=folds,
        seeds=tuple(int(seed) for seed in seeds),
        tabm=MappingProxyType(dict(tabm)),
        catboost=MappingProxyType(dict(catboost)),
        blend_candidates=tuple(BlendCandidate(*item) for item in blends),
        gates=MappingProxyType(dict(gates)),
        runtime=MappingProxyType(dict(runtime)),
    )


def build_structure_jobs(
    contract: E2Contract,
    *,
    seed: int,
    folds: Sequence[tuple[int, int]],
) -> tuple[E2Job, ...]:
    if seed not in contract.seeds:
        raise E2ContractError("seed is not registered")
    selected_folds = tuple(folds)
    for fold in selected_folds:
        if fold not in contract.folds:
            raise E2ContractError("fold is not registered")
    jobs: list[E2Job] = []
    for train_end, valid_year in selected_folds:
        for candidate_id in contract.structures:
            jobs.append(
                E2Job(
                    job_id=(
                        f"e2__{candidate_id}__tr{train_end}"
                        f"__va{valid_year}__s{seed}"
                    ),
                    candidate_id=candidate_id,
                    train_end_year=train_end,
                    valid_year=valid_year,
                    seed=seed,
                    objective="residual",
                    use_trackman=candidate_id == "c2_trackman_residual",
                )
            )
    return tuple(jobs)
