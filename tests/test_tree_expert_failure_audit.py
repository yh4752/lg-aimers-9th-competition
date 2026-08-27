from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from experiments.tree_expert.failure_audit import (
    FailureAuditError,
    run_failure_label_audit,
)
from experiments.tree_expert.failure_audit_contracts import (
    FailureAuditContract,
    load_failure_audit_contract,
)


def _season_rows(season: int, pitcher_id: int) -> pd.DataFrame:
    events = ("success", "middle", "middle", "reverse", "other_failure")
    success = middle = reverse = 0
    rows: list[dict[str, object]] = []
    for count in range(len(events) + 1):
        event = events[count] if count < len(events) else "success"
        rows.append(
            {
                "row_id": f"{season}_{count}",
                "season": season,
                "pitcher_id": pitcher_id,
                "asof_pitcher_n": count,
                "asof_pitcher_success_rate": success / count if count else 0.0,
                "asof_pitcher_middle_rate": middle / count if count else 0.0,
                "asof_pitcher_reverse_rate": reverse / count if count else 0.0,
                "control_success": int(event == "success"),
            }
        )
        if count < len(events):
            success += int(event == "success")
            middle += int(event == "middle")
            reverse += int(event == "reverse")
    return pd.DataFrame(rows)


@pytest.fixture
def audit_frame() -> pd.DataFrame:
    return pd.concat(
        [_season_rows(year, year) for year in range(2021, 2025)],
        ignore_index=True,
    )


def _small_contract() -> FailureAuditContract:
    return replace(
        load_failure_audit_contract(),
        minimum_coverage=0.8,
        minimum_binary_delta_fraction=1.0,
        minimum_success_agreement=1.0,
        maximum_middle_reverse_overlap=0.0,
        minimum_positive_rows=2,
        minimum_negative_rows=2,
    )


def test_audit_never_passes_rows_beyond_each_cutoff(audit_frame: pd.DataFrame) -> None:
    result = run_failure_label_audit(audit_frame, contract=_small_contract())

    assert result.cutoffs["A1"].maximum_source_season == 2021
    assert result.cutoffs["A2"].maximum_source_season == 2022
    assert result.cutoffs["A3"].maximum_source_season == 2023
    assert result.cutoffs["A4"].maximum_source_season == 2024


def test_type_eligibility_is_independent(audit_frame: pd.DataFrame) -> None:
    result = run_failure_label_audit(audit_frame, contract=_small_contract())

    assert result.type_decisions["middle"].status == "eligible"
    assert result.type_decisions["reverse"].status == "ineligible"
    assert "positive_rows" in result.type_decisions["reverse"].reason


def test_common_quality_failure_blocks_every_type(audit_frame: pd.DataFrame) -> None:
    contract = replace(_small_contract(), minimum_coverage=1.0)

    result = run_failure_label_audit(audit_frame, contract=contract)

    assert {item.status for item in result.type_decisions.values()} == {"ineligible"}


def test_audit_rejects_duplicate_row_identity(audit_frame: pd.DataFrame) -> None:
    changed = audit_frame.copy(deep=True)
    changed.loc[1, "row_id"] = changed.loc[0, "row_id"]

    with pytest.raises(FailureAuditError, match="row_id must be unique"):
        run_failure_label_audit(changed, contract=_small_contract())
