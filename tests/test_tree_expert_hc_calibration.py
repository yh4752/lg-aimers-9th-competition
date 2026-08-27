import numpy as np
import pandas as pd
import pytest

from experiments.tree_expert.hc_calibration import (
    HCCalibrationError,
    apply_calibrator,
    fit_rolling_calibrator,
    select_calibration_alpha,
)
from experiments.tree_expert.hc_contracts import load_hc_contract


def _frame(year: int, probability_column: str, offset: float = 0.0) -> pd.DataFrame:
    rows = []
    for index in range(20):
        rows.append(
            {
                "row_id": f"r{year}_{index}",
                "oof_year": year,
                "target": int(index % 3 != 0),
                probability_column: np.clip(0.55 + offset + (index % 2) * 0.05, 0.01, 0.99),
                "game_type": "R" if index % 4 else "F",
                "pitcher_id": f"p{index % 3}",
                "batter_id": f"b{index % 4}",
                "pitcher_hand": "R" if index % 2 else "L",
                "batter_hand": "L" if index % 3 else "R",
                "balls_before": index % 4,
                "strikes_before": index % 3,
                "outs_before": index % 3,
                "base_state": str(index % 4),
            }
        )
    return pd.DataFrame(rows)


def _sources():
    source = _frame(2021, "p0")
    c1 = pd.concat([_frame(year, "p1") for year in (2022, 2023, 2024)], ignore_index=True)
    return source, c1


def test_2022_calibration_uses_only_frozen_e2_2021():
    source, c1 = _sources()
    contract = load_hc_contract()
    state = fit_rolling_calibrator(
        2022,
        source,
        c1,
        profile_name="hc_balanced",
        profile=contract.profiles["hc_balanced"],
        minimum_group_rows={"identity": 1, "context": 1, "interaction": 1},
    )
    assert dict(state.source) == {2021: "e2"}


def test_2024_calibration_uses_only_prior_c1_errors():
    source, c1 = _sources()
    contract = load_hc_contract()
    state = fit_rolling_calibrator(
        2024,
        source,
        c1,
        profile_name="hc_balanced",
        profile=contract.profiles["hc_balanced"],
        minimum_group_rows={"identity": 1, "context": 1, "interaction": 1},
    )
    assert dict(state.source) == {2022: "c1", 2023: "c1"}


def test_unseen_groups_fall_back_without_nonfinite_probability():
    source, c1 = _sources()
    contract = load_hc_contract()
    state = fit_rolling_calibrator(
        2024,
        source,
        c1,
        profile_name="hc_balanced",
        profile=contract.profiles["hc_balanced"],
        minimum_group_rows={"identity": 1, "context": 1, "interaction": 1},
    )
    query = _frame(2024, "p1").head(1)
    query.loc[:, "pitcher_id"] = "unseen"
    result = apply_calibrator(query, state, alpha=1.0)
    assert np.isfinite(result["p2"]).all()
    assert result["p2"].between(1e-5, 1 - 1e-5).all()
    assert result.loc[0, "hc_calibration_effect"] == pytest.approx(
        np.clip(result.loc[0, "hc_calibration_effect"], -0.25, 0.25)
    )


def test_calibration_is_batch_order_and_duplicate_independent():
    source, c1 = _sources()
    contract = load_hc_contract()
    state = fit_rolling_calibrator(
        2024,
        source,
        c1,
        profile_name="hc_balanced",
        profile=contract.profiles["hc_balanced"],
        minimum_group_rows={"identity": 1, "context": 1, "interaction": 1},
    )
    query = _frame(2024, "p1").head(8)
    batch = apply_calibrator(query, state, alpha=0.5).set_index("row_id")
    shuffled = apply_calibrator(query.sample(frac=1, random_state=9), state, alpha=0.5).set_index("row_id")
    pd.testing.assert_series_equal(batch["p2"], shuffled.loc[batch.index, "p2"])
    singletons = pd.concat(
        [apply_calibrator(query.iloc[[index]], state, alpha=0.5) for index in range(len(query))]
    ).set_index("row_id")
    pd.testing.assert_series_equal(batch["p2"], singletons.loc[batch.index, "p2"])
    duplicate = apply_calibrator(pd.concat([query, query.iloc[[0]]], ignore_index=True), state, alpha=0.5)
    assert duplicate.iloc[0]["p2"] == pytest.approx(duplicate.iloc[-1]["p2"], abs=1e-12)


def test_alpha_selection_uses_structure_folds_and_smaller_tie():
    predictions = {}
    for alpha in (0.25, 0.5, 0.75, 1.0):
        frame = pd.concat(
            [
                pd.DataFrame({"oof_year": [2022], "target": [1], "p2": [0.6]}),
                pd.DataFrame({"oof_year": [2023], "target": [0], "p2": [0.4]}),
                pd.DataFrame(
                    {"oof_year": [2024], "target": [1], "p2": [0.99 if alpha == 1.0 else 0.01]}
                ),
            ],
            ignore_index=True,
        )
        predictions[alpha] = frame
    assert select_calibration_alpha(predictions) == 0.25


def test_calibrator_rejects_current_or_future_source_rows():
    source, c1 = _sources()
    source.loc[:, "oof_year"] = 2022
    with pytest.raises(HCCalibrationError, match="source window differs"):
        fit_rolling_calibrator(
            2022,
            source,
            c1,
            profile_name="hc_balanced",
            profile=load_hc_contract().profiles["hc_balanced"],
            minimum_group_rows={"identity": 1, "context": 1, "interaction": 1},
        )
