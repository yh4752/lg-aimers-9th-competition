from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path

import pytest

from experiments.temporal_portfolio import contracts as contract_module
from experiments.temporal_portfolio.contracts import (
    DEFAULT_CONTRACT,
    JobSpec,
    PortfolioContractError,
    TemporalFold,
    build_stage_jobs,
    load_contract,
)


def test_contract_pins_budget_folds_and_t1_grid() -> None:
    contract = load_contract()

    assert contract.campaign_id == "temporal_portfolio_v1"
    assert contract.bootstrap_seed == 3407
    assert sum(contract.stage_hours.values()) == Decimal("30")
    assert [(f.recent_year, f.multi_start, f.valid_year) for f in contract.folds] == [
        (2021, 2019, 2022),
        (2022, 2019, 2023),
        (2023, 2020, 2024),
    ]
    assert contract.decays == (
        Decimal("0.40"),
        Decimal("0.55"),
        Decimal("0.70"),
        Decimal("1.00"),
    )
    assert contract.recent_weights == (
        Decimal("0.50"),
        Decimal("0.65"),
        Decimal("0.75"),
        Decimal("0.85"),
        Decimal("1.00"),
    )
    assert len(build_stage_jobs(contract, "T1")) == 15


def test_contract_is_fail_closed() -> None:
    contract = load_contract()

    with pytest.raises(PortfolioContractError, match="campaign identity"):
        build_stage_jobs(replace(contract, campaign_id="other"), "T1")


@pytest.mark.parametrize(
    "changed",
    (
        {"folds": lambda contract: contract.folds[:1]},
        {"decays": lambda contract: (Decimal("0.99"),)},
        {"screen_seed": lambda contract: 99},
        {"screen_seed": lambda contract: 3407.0},
        {"bootstrap_seed": lambda contract: 99},
        {"schema_version": lambda contract: True},
        {"decays": lambda contract: (Decimal("0.4"), *contract.decays[1:])},
    ),
)
def test_stage_jobs_reject_mutated_contract_authorization(changed: dict[str, object]) -> None:
    contract = load_contract()
    field, factory = next(iter(changed.items()))

    with pytest.raises(PortfolioContractError, match="contract authorization"):
        build_stage_jobs(replace(contract, **{field: factory(contract)}), "T1")


def test_stage_jobs_match_the_complete_authorized_t1_schedule() -> None:
    contract = load_contract()

    assert build_stage_jobs(contract, "T1") == (
        JobSpec("t1__recent__va2022__s3407", "recent", TemporalFold(2021, 2019, 2021, 2022), None, 3407),
        JobSpec("t1__recent__va2023__s3407", "recent", TemporalFold(2022, 2019, 2022, 2023), None, 3407),
        JobSpec("t1__recent__va2024__s3407", "recent", TemporalFold(2023, 2020, 2023, 2024), None, 3407),
        JobSpec("t1__multi_d0p40__va2022__s3407", "multi", TemporalFold(2021, 2019, 2021, 2022), Decimal("0.40"), 3407),
        JobSpec("t1__multi_d0p40__va2023__s3407", "multi", TemporalFold(2022, 2019, 2022, 2023), Decimal("0.40"), 3407),
        JobSpec("t1__multi_d0p40__va2024__s3407", "multi", TemporalFold(2023, 2020, 2023, 2024), Decimal("0.40"), 3407),
        JobSpec("t1__multi_d0p55__va2022__s3407", "multi", TemporalFold(2021, 2019, 2021, 2022), Decimal("0.55"), 3407),
        JobSpec("t1__multi_d0p55__va2023__s3407", "multi", TemporalFold(2022, 2019, 2022, 2023), Decimal("0.55"), 3407),
        JobSpec("t1__multi_d0p55__va2024__s3407", "multi", TemporalFold(2023, 2020, 2023, 2024), Decimal("0.55"), 3407),
        JobSpec("t1__multi_d0p70__va2022__s3407", "multi", TemporalFold(2021, 2019, 2021, 2022), Decimal("0.70"), 3407),
        JobSpec("t1__multi_d0p70__va2023__s3407", "multi", TemporalFold(2022, 2019, 2022, 2023), Decimal("0.70"), 3407),
        JobSpec("t1__multi_d0p70__va2024__s3407", "multi", TemporalFold(2023, 2020, 2023, 2024), Decimal("0.70"), 3407),
        JobSpec("t1__multi_d1p00__va2022__s3407", "multi", TemporalFold(2021, 2019, 2021, 2022), Decimal("1.00"), 3407),
        JobSpec("t1__multi_d1p00__va2023__s3407", "multi", TemporalFold(2022, 2019, 2022, 2023), Decimal("1.00"), 3407),
        JobSpec("t1__multi_d1p00__va2024__s3407", "multi", TemporalFold(2023, 2020, 2023, 2024), Decimal("1.00"), 3407),
    )


def test_stage_jobs_reject_unauthorized_stage() -> None:
    with pytest.raises(PortfolioContractError, match="stage is not authorized"):
        build_stage_jobs(load_contract(), "T2A")


def _sealed_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: bytes) -> Path:
    path = tmp_path / "contract.json"
    path.write_bytes(raw)
    monkeypatch.setattr(contract_module, "_SEALED_CONTRACT_SHA256", sha256(raw).hexdigest())
    return path


def test_raw_tamper_is_rejected_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "tampered.json"
    path.write_bytes(DEFAULT_CONTRACT.read_bytes() + b"\x00")

    with pytest.raises(PortfolioContractError, match="contract bytes differ"):
        load_contract(path)


@pytest.mark.parametrize(
    ("raw", "message"),
    (
        (b'{"schema_version": 1, "schema_version": 1}', "duplicate JSON key"),
        (b'{"value": NaN}', "non-finite JSON number"),
    ),
)
def test_sealed_parser_rejects_duplicate_and_nonfinite_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: bytes, message: str
) -> None:
    with pytest.raises(PortfolioContractError, match=message):
        load_contract(_sealed_fixture(tmp_path, monkeypatch, raw))


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.__setitem__("unexpected", 1),
        lambda payload: payload.pop("losses"),
        lambda payload: payload["tabm_profiles"]["p2"].__setitem__("unexpected", 1),
        lambda payload: payload["bootstrap"].__setitem__("repeats", True),
    ),
)
def test_sealed_parser_rejects_invalid_structure_and_primitives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: object
) -> None:
    payload = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    mutate(payload)
    raw = json.dumps(payload).encode("utf-8")

    with pytest.raises(PortfolioContractError):
        load_contract(_sealed_fixture(tmp_path, monkeypatch, raw))


def test_loaded_mappings_are_immutable() -> None:
    contract = load_contract()

    for mapping in (
        contract.tabm_profiles,
        contract.gates,
        contract.stage_hours,
        contract.physical_stage_seconds,
        contract.score_tiers,
    ):
        with pytest.raises(TypeError):
            mapping["unexpected"] = 1
    with pytest.raises(TypeError):
        contract.tabm_profiles["p2"]["k"] = 64
