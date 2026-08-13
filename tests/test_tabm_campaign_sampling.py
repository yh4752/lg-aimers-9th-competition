from __future__ import annotations

import pandas as pd
import pytest

from experiments.tabm_campaign.sampling import SamplingError, proxy_row_ids


def _frame() -> pd.DataFrame:
    rows = []
    for season, count in ((2021, 30), (2022, 50), (2023, 20), (2024, 10)):
        for index in range(count):
            rows.append(
                {
                    "row_id": f"{season}-{index:03d}",
                    "season": season,
                    "control_success": index % 2,
                }
            )
    return pd.DataFrame(rows)


def test_proxy_sample_is_season_proportional_and_order_independent() -> None:
    frame = _frame()
    left = proxy_row_ids(frame, train_end_year=2023, max_rows=40, seed=42)
    right = proxy_row_ids(
        frame.sample(frac=1.0, random_state=7),
        train_end_year=2023,
        max_rows=40,
        seed=42,
    )

    assert left == right
    assert len(left) == 40
    sampled = frame.set_index("row_id").loc[list(left)]
    assert sampled.groupby("season").size().to_dict() == {2021: 12, 2022: 20, 2023: 8}


def test_proxy_sample_does_not_read_target() -> None:
    frame = _frame()
    changed = frame.copy()
    changed["control_success"] = 1 - changed["control_success"]
    assert proxy_row_ids(frame, train_end_year=2023, max_rows=40, seed=42) == proxy_row_ids(
        changed, train_end_year=2023, max_rows=40, seed=42
    )


def test_proxy_sample_rejects_duplicate_ids() -> None:
    frame = _frame()
    frame.loc[1, "row_id"] = frame.loc[0, "row_id"]
    with pytest.raises(SamplingError, match="unique"):
        proxy_row_ids(frame, train_end_year=2023, max_rows=40, seed=42)
