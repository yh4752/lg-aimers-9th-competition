from dataclasses import replace
from decimal import Decimal

import pytest

from experiments.temporal_portfolio.contracts import (
    PortfolioContractError,
    build_stage_jobs,
    load_contract,
)


def test_contract_pins_budget_folds_and_t1_grid() -> None:
    contract = load_contract()

    assert contract.campaign_id == "temporal_portfolio_v1"
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
