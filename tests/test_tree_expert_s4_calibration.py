import numpy as np
import pandas as pd
import pytest

from experiments.tree_expert.s4_calibration import (
    S4CalibrationError,
    apply_s4_calibrator,
    fit_s4_calibrator,
    s4_calibration_state_from_payload,
    s4_calibration_state_payload,
    select_s4_beta,
)


def _frame(year: int, probability: str) -> pd.DataFrame:
    rows = []
    for index in range(24):
        rows.append({
            "row_id": f"r{year}_{index}", "oof_year": year,
            "target": int(index % 3 != 0), probability: 0.52 + 0.03 * (index % 3),
            "game_type": "R" if index % 4 else "F",
            "pitcher_id": f"p{index % 3}", "batter_id": f"b{index % 4}",
            "pitcher_hand": "R" if index % 2 else "L",
            "batter_hand": "L" if index % 3 else "R",
            "balls_before": index % 4, "strikes_before": index % 3,
            "outs_before": index % 3, "base_state": str(index % 4),
        })
    return pd.DataFrame(rows)


def _sources():
    anchor = _frame(2021, "p_anchor")
    chain = pd.concat(
        [_frame(year, "p_chain") for year in (2022, 2023, 2024)],
        ignore_index=True,
    )
    return anchor, chain


def test_s4_calibration_uses_only_prior_oof_errors():
    anchor, chain = _sources()
    state = fit_s4_calibrator(
        2024, anchor, chain, profile_name="rf_matchup",
        minimum_group_rows={"identity": 1, "context": 1, "interaction": 1},
    )
    assert dict(state.source) == {2022: "s4_chain", 2023: "s4_chain"}


def test_s4_profiles_limit_the_hierarchy_that_can_act():
    anchor, chain = _sources()
    global_state = fit_s4_calibrator(
        2024, anchor, chain, profile_name="global_game",
        minimum_group_rows={"identity": 1, "context": 1, "interaction": 1},
    )
    pitcher_state = fit_s4_calibrator(
        2024, anchor, chain, profile_name="pitcher",
        minimum_group_rows={"identity": 1, "context": 1, "interaction": 1},
    )
    assert global_state.active_levels == ("game_type",)
    assert pitcher_state.active_levels == ("game_type", "pitcher", "pitcher_game")


def test_s4_beta_point_one_is_supported_and_row_independent():
    anchor, chain = _sources()
    state = fit_s4_calibrator(
        2024, anchor, chain, profile_name="rf_matchup",
        minimum_group_rows={"identity": 1, "context": 1, "interaction": 1},
    )
    query = _frame(2024, "p_chain").head(6).drop(columns=["target", "oof_year"])
    first = apply_s4_calibrator(query, state, beta=0.10).set_index("row_id")
    second = apply_s4_calibrator(query.sample(frac=1, random_state=4), state, beta=0.10).set_index("row_id")
    pd.testing.assert_series_equal(first["p_final"], second.loc[first.index, "p_final"])
    assert first["p_final"].between(1e-5, 1 - 1e-5).all()


def test_s4_state_round_trip_preserves_predictions():
    anchor, chain = _sources()
    state = fit_s4_calibrator(
        2024, anchor, chain, profile_name="matchup",
        minimum_group_rows={"identity": 1, "context": 1, "interaction": 1},
    )
    restored = s4_calibration_state_from_payload(s4_calibration_state_payload(state))
    query = _frame(2024, "p_chain").head(6).drop(columns=["target", "oof_year"])
    np.testing.assert_allclose(
        apply_s4_calibrator(query, state, beta=0.50)["p_final"],
        apply_s4_calibrator(query, restored, beta=0.50)["p_final"],
    )


def test_s4_beta_selection_ignores_recent_holdout_and_breaks_ties_small():
    predictions = {}
    for beta in (0.10, 0.25, 0.50, 0.75):
        predictions[beta] = pd.DataFrame({
            "oof_year": [2022, 2023, 2024], "target": [1, 0, 1],
            "p_final": [0.6, 0.4, 0.99 if beta == 0.75 else 0.01],
        })
    assert select_s4_beta(predictions) == 0.10


def test_s4_rejects_unknown_profile_or_beta():
    anchor, chain = _sources()
    with pytest.raises(S4CalibrationError):
        fit_s4_calibrator(2024, anchor, chain, profile_name="unknown")
    state = fit_s4_calibrator(2024, anchor, chain, profile_name="global_game")
    with pytest.raises(S4CalibrationError):
        apply_s4_calibrator(_frame(2024, "p_chain").head(1), state, beta=1.0)
