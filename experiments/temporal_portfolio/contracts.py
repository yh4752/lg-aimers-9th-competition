from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class PortfolioContractError(ValueError):
    """Raised when the sealed temporal portfolio contract is invalid."""


DEFAULT_CONTRACT = Path(__file__).with_name("contract.json")
_SEALED_CONTRACT_SHA256 = (
    "394ed8f252ca6b3ce17af1df4dbd23859a2b8d345744ff27eebc2fb8f8b1f57f"
)

_ROOT_KEYS = {
    "schema_version",
    "campaign_id",
    "submission_package",
    "folds",
    "decays",
    "recent_weights",
    "anchor_betas",
    "feature_bundles",
    "tabm_profiles",
    "losses",
    "catboost_prefixes",
    "lupi_lambdas",
    "pair_primary_weights",
    "bootstrap",
    "gates",
    "stage_hours",
    "physical_stage_seconds",
    "seeds",
    "score_tiers",
}
_FOLDS = ((2021, 2019, 2021, 2022), (2022, 2019, 2022, 2023), (2023, 2020, 2023, 2024))
_DECAYS = ("0.40", "0.55", "0.70", "1.00")
_RECENT_WEIGHTS = ("0.50", "0.65", "0.75", "0.85", "1.00")
_ANCHOR_BETAS = ("0", "0.025", "0.05", "0.10")
_FEATURE_BUNDLES = ("S1", "P0", "P1", "P2", "P3", "B1", "M1")
_LOSSES = ("bce", "brier")
_CATBOOST_PREFIXES = (16, 64, 192, 384)
_LUPI_LAMBDAS = ("0.10", "0.25")
_PAIR_PRIMARY_WEIGHTS = ("0.90", "0.80", "0.70")
_TABM_PROFILES = {
    "p2": {"k": 32, "width": 512, "blocks": 4, "dropout": "0.10"},
    "p3_lite": {"k": 32, "width": 768, "blocks": 6, "dropout": "0.15"},
}
_BOOTSTRAP = {"repeats": 1000, "seed": 3407, "minimum_segment_rows": 5000}
_GATES = {
    "champion_weighted_gain": "0.00005",
    "champion_latest_gain": "0.00003",
    "champion_max_segment_regression": "0.00050",
    "exploratory_weighted_gain": "0.00003",
    "exploratory_worst_fold_regression": "0.00015",
    "exploratory_latest_regression": "0.00005",
    "exploratory_max_segment_regression": "0.00100",
}
_STAGE_HOURS = {
    "T1": "6",
    "T2A": "4",
    "T2B": "4",
    "T3": "6",
    "T4": "3",
    "T5": "5",
    "reserve": "2",
}
_PHYSICAL_STAGE_SECONDS = {
    "T1": 21600,
    "T2A": 14400,
    "T2B": 14400,
    "T3A": 10800,
    "T3BT4": 20700,
    "T5A": 9000,
    "T5B": 9000,
}
_SEEDS = {"screen": 3407, "confirm": 42}
_SCORE_TIERS = {
    "incremental": "0.00005",
    "competitive": "0.00025",
    "breakthrough": "0.00045",
}


@dataclass(frozen=True)
class TemporalFold:
    recent_year: int
    multi_start: int
    multi_end: int
    valid_year: int


@dataclass(frozen=True)
class PortfolioContract:
    schema_version: int
    campaign_id: str
    folds: tuple[TemporalFold, ...]
    decays: tuple[Decimal, ...]
    recent_weights: tuple[Decimal, ...]
    anchor_betas: tuple[Decimal, ...]
    feature_bundles: tuple[str, ...]
    tabm_profiles: Mapping[str, Mapping[str, object]]
    losses: tuple[str, ...]
    catboost_prefixes: tuple[int, ...]
    lupi_lambdas: tuple[Decimal, ...]
    pair_primary_weights: tuple[Decimal, ...]
    bootstrap_repeats: int
    minimum_segment_rows: int
    gates: Mapping[str, Decimal]
    stage_hours: Mapping[str, Decimal]
    physical_stage_seconds: Mapping[str, int]
    screen_seed: int
    confirm_seed: int
    score_tiers: Mapping[str, Decimal]


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    expert: str
    fold: TemporalFold
    decay: Decimal | None
    seed: int


def _contract_values_match_exactly(left: object, right: object) -> bool:
    """Compare contract values without numeric coercion or Decimal normalization."""
    if type(left) is not type(right):
        return False
    if isinstance(left, Decimal):
        return left.as_tuple() == right.as_tuple()
    if isinstance(left, tuple):
        return len(left) == len(right) and all(
            _contract_values_match_exactly(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    if isinstance(left, Mapping):
        if len(left) != len(right):
            return False
        unmatched = list(right.items())
        for left_key, left_value in left.items():
            for index, (right_key, right_value) in enumerate(unmatched):
                if _contract_values_match_exactly(left_key, right_key):
                    if not _contract_values_match_exactly(left_value, right_value):
                        return False
                    del unmatched[index]
                    break
            else:
                return False
        return not unmatched
    if isinstance(left, (TemporalFold, PortfolioContract)):
        return all(
            _contract_values_match_exactly(
                getattr(left, field_name), getattr(right, field_name)
            )
            for field_name in left.__dataclass_fields__
        )
    return left == right


def _object_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PortfolioContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise PortfolioContractError(f"non-finite JSON number: {value}")


def _read_sealed_json(path: str | Path) -> dict[str, object]:
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise PortfolioContractError(f"cannot read portfolio contract: {error}") from error
    if sha256(raw).hexdigest() != _SEALED_CONTRACT_SHA256:
        raise PortfolioContractError("contract bytes differ")
    try:
        root = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_no_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except PortfolioContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PortfolioContractError(f"cannot parse portfolio contract: {error}") from error
    if type(root) is not dict:
        raise PortfolioContractError("contract must be an object")
    return root


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise PortfolioContractError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise PortfolioContractError(f"{label} keys differ")


def _exact_int(value: object, expected: int, label: str) -> int:
    if type(value) is not int or value != expected:
        raise PortfolioContractError(f"{label} differs")
    return value


def _exact_string(value: object, expected: str, label: str) -> str:
    if type(value) is not str or value != expected:
        raise PortfolioContractError(f"{label} differs")
    return value


def _decimal(value: object, label: str) -> Decimal:
    if type(value) is not str:
        raise PortfolioContractError(f"{label} must be a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise PortfolioContractError(f"{label} must be a decimal string") from error
    if not parsed.is_finite():
        raise PortfolioContractError(f"{label} must be finite")
    return parsed


def _sealed_decimal_grid(value: object, expected: tuple[str, ...], label: str) -> tuple[Decimal, ...]:
    if type(value) is not list or len(value) != len(expected):
        raise PortfolioContractError(f"{label} grid differs")
    if any(type(item) is not str for item in value):
        raise PortfolioContractError(f"{label} grid differs")
    strings = tuple(value)
    if strings != expected:
        raise PortfolioContractError(f"{label} grid differs")
    return tuple(_decimal(item, label) for item in strings)


def _sealed_string_grid(value: object, expected: tuple[str, ...], label: str) -> tuple[str, ...]:
    if type(value) is not list or tuple(value) != expected or any(type(item) is not str for item in value):
        raise PortfolioContractError(f"{label} grid differs")
    return expected


def _sealed_int_grid(value: object, expected: tuple[int, ...], label: str) -> tuple[int, ...]:
    if type(value) is not list or tuple(value) != expected or any(type(item) is not int for item in value):
        raise PortfolioContractError(f"{label} grid differs")
    return expected


def _load_folds(value: object) -> tuple[TemporalFold, ...]:
    if type(value) is not list or len(value) != len(_FOLDS):
        raise PortfolioContractError("fold schedule differs")
    raw_folds: list[tuple[int, int, int, int]] = []
    for raw_fold in value:
        if type(raw_fold) is not list or len(raw_fold) != 4 or any(type(year) is not int for year in raw_fold):
            raise PortfolioContractError("fold schedule differs")
        raw_folds.append(tuple(raw_fold))
    if tuple(raw_folds) != _FOLDS:
        raise PortfolioContractError("fold schedule differs")
    return tuple(TemporalFold(*fold) for fold in _FOLDS)


def _load_profiles(value: object) -> Mapping[str, Mapping[str, object]]:
    profiles = _mapping(value, "tabm profiles")
    _exact_keys(profiles, set(_TABM_PROFILES), "tabm profiles")
    frozen: dict[str, Mapping[str, object]] = {}
    for name, expected in _TABM_PROFILES.items():
        profile = _mapping(profiles[name], f"tabm profile {name}")
        _exact_keys(profile, set(expected), f"tabm profile {name}")
        for key, expected_value in expected.items():
            if type(profile[key]) is not type(expected_value) or profile[key] != expected_value:
                raise PortfolioContractError(f"tabm profile {name} differs")
        frozen[name] = MappingProxyType(dict(profile))
    return MappingProxyType(frozen)


def _load_decimal_mapping(value: object, expected: Mapping[str, str], label: str) -> Mapping[str, Decimal]:
    raw = _mapping(value, label)
    _exact_keys(raw, set(expected), label)
    result: dict[str, Decimal] = {}
    for key, expected_value in expected.items():
        _exact_string(raw[key], expected_value, f"{label} {key}")
        result[key] = _decimal(raw[key], f"{label} {key}")
    return MappingProxyType(result)


def _load_exact_int_mapping(value: object, expected: Mapping[str, int], label: str) -> Mapping[str, int]:
    raw = _mapping(value, label)
    _exact_keys(raw, set(expected), label)
    return MappingProxyType(
        {
            key: _exact_int(raw[key], expected_value, f"{label} {key}")
            for key, expected_value in expected.items()
        }
    )


def load_contract(path: str | Path = DEFAULT_CONTRACT) -> PortfolioContract:
    root = _read_sealed_json(path)
    _exact_keys(root, _ROOT_KEYS, "contract")
    _exact_int(root["schema_version"], 1, "schema version")
    _exact_string(root["campaign_id"], "temporal_portfolio_v1", "campaign identity")
    if root["submission_package"] is not False:
        raise PortfolioContractError("submission package must be false")

    folds = _load_folds(root["folds"])
    decays = _sealed_decimal_grid(root["decays"], _DECAYS, "decays")
    recent_weights = _sealed_decimal_grid(root["recent_weights"], _RECENT_WEIGHTS, "recent weights")
    anchor_betas = _sealed_decimal_grid(root["anchor_betas"], _ANCHOR_BETAS, "anchor betas")
    feature_bundles = _sealed_string_grid(root["feature_bundles"], _FEATURE_BUNDLES, "feature bundles")
    tabm_profiles = _load_profiles(root["tabm_profiles"])
    losses = _sealed_string_grid(root["losses"], _LOSSES, "losses")
    catboost_prefixes = _sealed_int_grid(root["catboost_prefixes"], _CATBOOST_PREFIXES, "catboost prefixes")
    lupi_lambdas = _sealed_decimal_grid(root["lupi_lambdas"], _LUPI_LAMBDAS, "lupi lambdas")
    pair_primary_weights = _sealed_decimal_grid(root["pair_primary_weights"], _PAIR_PRIMARY_WEIGHTS, "pair primary weights")

    bootstrap = _load_exact_int_mapping(root["bootstrap"], _BOOTSTRAP, "bootstrap")
    gates = _load_decimal_mapping(root["gates"], _GATES, "gates")
    stage_hours = _load_decimal_mapping(root["stage_hours"], _STAGE_HOURS, "stage hours")
    if sum(stage_hours.values()) != Decimal("30"):
        raise PortfolioContractError("stage hour budget differs")
    physical_stage_seconds = _load_exact_int_mapping(
        root["physical_stage_seconds"], _PHYSICAL_STAGE_SECONDS, "physical stages"
    )
    seeds = _load_exact_int_mapping(root["seeds"], _SEEDS, "seeds")
    score_tiers = _load_decimal_mapping(root["score_tiers"], _SCORE_TIERS, "score tiers")

    return PortfolioContract(
        schema_version=1,
        campaign_id="temporal_portfolio_v1",
        folds=folds,
        decays=decays,
        recent_weights=recent_weights,
        anchor_betas=anchor_betas,
        feature_bundles=feature_bundles,
        tabm_profiles=tabm_profiles,
        losses=losses,
        catboost_prefixes=catboost_prefixes,
        lupi_lambdas=lupi_lambdas,
        pair_primary_weights=pair_primary_weights,
        bootstrap_repeats=bootstrap["repeats"],
        minimum_segment_rows=bootstrap["minimum_segment_rows"],
        gates=gates,
        stage_hours=stage_hours,
        physical_stage_seconds=physical_stage_seconds,
        screen_seed=seeds["screen"],
        confirm_seed=seeds["confirm"],
        score_tiers=score_tiers,
    )


def build_stage_jobs(contract: PortfolioContract, stage: str) -> tuple[JobSpec, ...]:
    if contract.campaign_id != "temporal_portfolio_v1":
        raise PortfolioContractError("campaign identity differs")
    if stage != "T1":
        raise PortfolioContractError("stage is not authorized")
    if not _contract_values_match_exactly(contract, load_contract()):
        raise PortfolioContractError("contract authorization differs")

    recent = tuple(
        JobSpec(
            job_id=f"t1__recent__va{fold.valid_year}__s{contract.screen_seed}",
            expert="recent",
            fold=fold,
            decay=None,
            seed=contract.screen_seed,
        )
        for fold in contract.folds
    )
    multi = tuple(
        JobSpec(
            job_id=f"t1__multi_d{str(decay).replace('.', 'p')}__va{fold.valid_year}__s{contract.screen_seed}",
            expert="multi",
            fold=fold,
            decay=decay,
            seed=contract.screen_seed,
        )
        for decay in contract.decays
        for fold in contract.folds
    )
    return recent + multi
