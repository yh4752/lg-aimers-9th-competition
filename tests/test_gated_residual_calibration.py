from __future__ import annotations

import numpy as np
import pandas as pd
import pickle

from experiments.gated_residual_final.calibration import (
    apply_calibration,
    fit_temporal_calibrator,
)


def _oof() -> pd.DataFrame:
    return pd.DataFrame({
        "row_id": range(1, 7),
        "target": [1, 0, 1, 0, 1, 0],
        "probability": [0.4, 0.4, 0.45, 0.45, 0.5, 0.5],
        "oof_year": [2022, 2022, 2023, 2023, 2024, 2024],
        "game_type": ["R"] * 6,
        "hand_matchup": ["RL", "RL", "RL", "RR", "RL", "RR"],
        "pitcher_id": ["p1", "p2", "p1", "p3", "p1", "p4"],
        "batter_id": ["b1", "b2", "b1", "b3", "b1", "b4"],
    })


def test_2024_calibrator_fits_only_2022_and_2023() -> None:
    fitted = fit_temporal_calibrator(_oof(), validation_year=2024, hierarchy="full", ridge=100)

    assert fitted.source_years == (2022, 2023)
    assert "p4" not in {key[1] for key in fitted.pitcher_effects}
    assert "b4" not in {key[1] for key in fitted.batter_effects}


def test_2022_calibrator_has_zero_effect_without_prior_oof() -> None:
    fitted = fit_temporal_calibrator(_oof(), validation_year=2022, hierarchy="full", ridge=100)

    assert fitted.source_years == ()
    assert fitted.global_effect == 0.0


def test_unknown_pitcher_and_batter_back_off_to_hand_effect() -> None:
    fitted = fit_temporal_calibrator(_oof(), validation_year=2024, hierarchy="full", ridge=100)

    effect = fitted.effect_for(
        game_type="R", hand_matchup="RL", pitcher_id="new", batter_id="new"
    )

    assert effect == fitted.hand_effects[("R", "RL")]


def test_other_evaluation_rows_do_not_change_one_row_prediction() -> None:
    fitted = fit_temporal_calibrator(_oof(), validation_year=2024, hierarchy="full", ridge=100)
    evaluation = _oof().loc[_oof()["oof_year"].eq(2024)].reset_index(drop=True)

    one = apply_calibration(fitted, evaluation.iloc[[0]], beta=0.1)
    many = apply_calibration(fitted, evaluation, beta=0.1)[[0]]

    np.testing.assert_allclose(one, many, atol=1e-12)


def test_global_game_hierarchy_does_not_create_player_tables() -> None:
    fitted = fit_temporal_calibrator(_oof(), validation_year=2024, hierarchy="global_game", ridge=100)

    assert fitted.hand_effects == {}
    assert fitted.pitcher_effects == {}
    assert fitted.batter_effects == {}


def test_frozen_calibrator_round_trips_through_delivery_pickle() -> None:
    fitted = fit_temporal_calibrator(_oof(), validation_year=2024, hierarchy="full", ridge=100)

    restored = pickle.loads(pickle.dumps(fitted, protocol=5))

    assert restored.source_years == fitted.source_years
    assert dict(restored.pitcher_effects) == dict(fitted.pitcher_effects)
