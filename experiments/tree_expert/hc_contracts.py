from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping


class HCContractError(ValueError):
    """Raised when the hierarchical campaign registry differs."""


DEFAULT_HC_CONTRACT = Path(__file__).with_name("hc_contract.json")
_REGISTERED_SHA256 = "51749491dc40b85232dee4fbfc968732f5b1f556909ef1d2304761182423fa82"


@dataclass(frozen=True)
class HCProfile:
    identity_k: float
    context_k: float
    interaction_k: float


@dataclass(frozen=True)
class HCRuntime:
    h1_wall_seconds: int
    h2_wall_seconds: int
    h3_wall_seconds: int
    new_job_guard_seconds: int
    final_fit_guard_seconds: int
    snapshot_interval_seconds: int
    inference_rows: int
    inference_max_seconds: int
    model_state_max_bytes: int
    probability_tolerance: float


@dataclass(frozen=True)
class HCContract:
    source_path: Path
    campaign_id: str
    review_only: bool
    submission_package: bool
    inputs: Mapping[str, str]
    folds: tuple[tuple[int, int], ...]
    structure_folds: tuple[tuple[int, int], ...]
    confirmation_fold: tuple[int, int]
    seeds: tuple[int, ...]
    profiles: Mapping[str, HCProfile]
    minimum_group_rows: Mapping[str, int]
    profile_tie_order: tuple[str, ...]
    calibration_alphas: tuple[float, ...]
    seed_ensemble: str
    full_fit_iterations: Mapping[str, object]
    residual_catboost: Mapping[str, object]
    calibration: Mapping[str, object]
    gates: Mapping[str, object]
    runtime: HCRuntime


@dataclass(frozen=True)
class HCJob:
    job_id: str
    stage: Literal["H1", "H2", "H3"]
    kind: Literal[
        "source_tabm",
        "source_e2",
        "c1_residual",
        "c2_calibration",
        "full_fit",
        "c2_state",
    ]
    train_end_year: int | None
    valid_year: int | None
    seed: int
    profile: str | None
    condition: str


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: object, label: str) -> float:
    if type(value) not in {int, float} or type(value) is bool:
        raise HCContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise HCContractError(f"{label} must be finite")
    return result


def _integer(value: object, label: str) -> int:
    if type(value) is not int or type(value) is bool:
        raise HCContractError(f"{label} must be an integer")
    return value


def _fold(value: object, label: str) -> tuple[int, int]:
    if (
        type(value) is not list
        or len(value) != 2
        or any(type(year) is not int or type(year) is bool for year in value)
        or value[1] != value[0] + 1
    ):
        raise HCContractError(f"{label} differs")
    return int(value[0]), int(value[1])


def load_hc_contract(path: Path = DEFAULT_HC_CONTRACT) -> HCContract:
    source = Path(path)
    if not source.is_file() or source.is_symlink() or file_sha256(source) != _REGISTERED_SHA256:
        raise HCContractError("registered contract differs")
    try:
        root = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HCContractError("registered contract differs") from error
    if type(root) is not dict:
        raise HCContractError("registered contract differs")

    profiles = {
        name: HCProfile(
            identity_k=_number(values["identity_k"], f"{name} identity_k"),
            context_k=_number(values["context_k"], f"{name} context_k"),
            interaction_k=_number(values["interaction_k"], f"{name} interaction_k"),
        )
        for name, values in root["profiles"].items()
    }
    runtime = root["runtime"]
    return HCContract(
        source_path=source,
        campaign_id=str(root["campaign_id"]),
        review_only=bool(root["review_only"]),
        submission_package=bool(root["submission_package"]),
        inputs=MappingProxyType(dict(root["inputs"])),
        folds=tuple(_fold(item, "fold") for item in root["folds"]),
        structure_folds=tuple(
            _fold(item, "structure fold") for item in root["structure_folds"]
        ),
        confirmation_fold=_fold(root["confirmation_fold"], "confirmation fold"),
        seeds=tuple(_integer(seed, "seed") for seed in root["seeds"]),
        profiles=MappingProxyType(profiles),
        minimum_group_rows=MappingProxyType(
            {
                key: _integer(value, f"{key} minimum rows")
                for key, value in root["minimum_group_rows"].items()
            }
        ),
        profile_tie_order=tuple(root["profile_tie_order"]),
        calibration_alphas=tuple(
            _number(value, "calibration alpha") for value in root["calibration_alphas"]
        ),
        seed_ensemble=str(root["seed_ensemble"]),
        full_fit_iterations=MappingProxyType(dict(root["full_fit_iterations"])),
        residual_catboost=MappingProxyType(dict(root["residual_catboost"])),
        calibration=MappingProxyType(dict(root["calibration"])),
        gates=MappingProxyType(dict(root["gates"])),
        runtime=HCRuntime(
            h1_wall_seconds=_integer(runtime["h1_wall_seconds"], "H1 wall seconds"),
            h2_wall_seconds=_integer(runtime["h2_wall_seconds"], "H2 wall seconds"),
            h3_wall_seconds=_integer(runtime["h3_wall_seconds"], "H3 wall seconds"),
            new_job_guard_seconds=_integer(
                runtime["new_job_guard_seconds"], "new-job guard seconds"
            ),
            final_fit_guard_seconds=_integer(
                runtime["final_fit_guard_seconds"], "full-fit guard seconds"
            ),
            snapshot_interval_seconds=_integer(
                runtime["snapshot_interval_seconds"], "snapshot interval seconds"
            ),
            inference_rows=_integer(runtime["inference_rows"], "inference rows"),
            inference_max_seconds=_integer(
                runtime["inference_max_seconds"], "inference max seconds"
            ),
            model_state_max_bytes=_integer(
                runtime["model_state_max_bytes"], "model state max bytes"
            ),
            probability_tolerance=_number(
                runtime["probability_tolerance"], "probability tolerance"
            ),
        ),
    )


def contract_sha256(path: Path = DEFAULT_HC_CONTRACT) -> str:
    return file_sha256(path)


def _c1_job(
    stage: Literal["H1", "H2"],
    fold: tuple[int, int],
    seed: int,
    profile: str | None,
    condition: str,
) -> HCJob:
    train_end, valid = fold
    label = profile or "selected"
    return HCJob(
        job_id=f"hc__c1_{label}__tr{train_end}__va{valid}__s{seed}",
        stage=stage,
        kind="c1_residual",
        train_end_year=train_end,
        valid_year=valid,
        seed=seed,
        profile=profile,
        condition=condition,
    )


def build_hc_jobs(contract: HCContract) -> tuple[HCJob, ...]:
    jobs: list[HCJob] = [
        HCJob(
            "hc__source_tabm__tr2020__va2021__s3407",
            "H1",
            "source_tabm",
            2020,
            2021,
            3407,
            None,
            "always",
        )
    ]
    for seed in contract.seeds:
        jobs.append(
            HCJob(
                f"hc__source_e2__tr2020__va2021__s{seed}",
                "H1",
                "source_e2",
                2020,
                2021,
                seed,
                None,
                "source_tabm_complete",
            )
        )
    for profile in contract.profile_tie_order:
        for fold in contract.structure_folds:
            jobs.append(_c1_job("H1", fold, 3407, profile, "always"))
        jobs.append(
            _c1_job(
                "H1",
                contract.confirmation_fold,
                3407,
                profile,
                f"selected_profile:{profile}",
            )
        )
    for seed in (42, 2026):
        for fold in contract.folds[1:]:
            jobs.append(_c1_job("H2", fold, seed, None, "selected_profile"))
    for alpha in contract.calibration_alphas:
        label = str(alpha).replace(".", "p")
        jobs.append(
            HCJob(
                f"hc__c2_alpha_{label}__s3407",
                "H2",
                "c2_calibration",
                None,
                None,
                3407,
                None,
                "three_seed_c1_complete",
            )
        )
    for seed in contract.seeds:
        jobs.append(
            HCJob(
                f"hc__full_c1__s{seed}",
                "H3",
                "full_fit",
                None,
                None,
                seed,
                None,
                "c1_or_c2_accepted",
            )
        )
    jobs.append(
        HCJob(
            "hc__full_c2_state__s3407",
            "H3",
            "c2_state",
            None,
            None,
            3407,
            None,
            "c2_winner",
        )
    )
    if len({job.job_id for job in jobs}) != len(jobs):
        raise HCContractError("job identities differ")
    return tuple(jobs)
