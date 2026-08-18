from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from experiments.hierarchical_tabm.calibration import (
    CALIBRATION_EFFECTS,
    CalibrationError,
    apply_calibration,
    calibration_state_from_payload,
    calibration_state_payload,
    calibration_state_sha256,
    canonical_state_json,
    fit_h2,
    fit_h3,
    select_calibration,
)


def _signal() -> tuple[np.ndarray, np.ndarray]:
    probability = np.linspace(0.05, 0.95, 200, dtype="float64")
    true_probability = 1.0 / (
        1.0 + np.exp(-(-0.7 + 1.8 * np.log(probability / (1 - probability))))
    )
    target = (np.arange(200) / 200.0 < true_probability).astype("float64")
    return probability, target


def _segments(length: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"r{i}" for i in range(length)],
            "game_type": np.resize(["R", "P"], length),
            "count_state": np.resize(["0_0", "1_1", "3_2"], length),
            "hand_matchup": np.resize(["R_L", "L_R"], length),
            "base_out_state": np.resize(["000_0", "100_1"], length),
        }
    )


def _brier(target: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean(np.square(target - probability), dtype=np.float64))


def test_affine_logit_improves_known_miscalibration() -> None:
    probability, target = _signal()
    state = fit_h2(probability, target, regularization=1e-4, clip=1e-6)
    calibrated = apply_calibration(probability, _segments(len(target)), state)

    assert state.kind == "H2"
    assert np.isfinite(calibrated).all()
    assert ((calibrated >= 1e-6) & (calibrated <= 1 - 1e-6)).all()
    assert _brier(target, calibrated) < _brier(target, probability)


def test_h2_penalty_shrinks_toward_identity() -> None:
    probability, target = _signal()
    weak = fit_h2(probability, target, regularization=1e-4, clip=1e-6)
    strong = fit_h2(probability, target, regularization=10.0, clip=1e-6)
    assert abs(strong.bias) <= abs(weak.bias)
    assert abs(strong.slope - 1.0) <= abs(weak.slope - 1.0)


def test_h3_uses_only_preregistered_main_effects() -> None:
    probability, target = _signal()
    segments = _segments(len(target))
    segments["pitcher_id"] = "must-not-enter"
    state = fit_h3(
        probability, target, segments, regularization=0.001, clip=1e-6
    )
    assert tuple(state.effects) == CALIBRATION_EFFECTS
    assert b"pitcher_id" not in canonical_state_json(state)


def test_h3_unseen_category_has_zero_offset() -> None:
    probability, target = _signal()
    state = fit_h3(
        probability, target, _segments(len(target)), regularization=0.01, clip=1e-6
    )
    unseen = _segments(1)
    for column in CALIBRATION_EFFECTS:
        unseen[column] = "never-seen"
    actual = apply_calibration(np.array([0.4]), unseen, state)[0]
    expected = 1 / (1 + np.exp(-(state.bias + state.slope * np.log(0.4 / 0.6))))
    assert actual == pytest.approx(expected)


def _oof() -> pd.DataFrame:
    probability, target = _signal()
    result = _segments(len(target))
    result["probability"] = probability
    result["control_success"] = target
    result["game_month"] = np.resize(np.arange(1, 13), len(target))
    return result


def test_regularization_selection_uses_early_fit_and_late_validation() -> None:
    oof = _oof()
    calls: list[int] = []

    def recording_optimizer(fun, x0, **kwargs):
        calls.append(len(x0))
        return minimize(fun, x0, **kwargs)

    selected = select_calibration(
        oof,
        kind="H3",
        grid=(1e-4, 1e-3),
        fit_month_max=7,
        validation_month_min=8,
        optimizer=recording_optimizer,
    )
    expected_ids = tuple(oof.loc[oof.game_month >= 8, "row_id"])
    assert selected.selection_row_ids == expected_ids
    assert tuple(selected.validation_brier) == (1e-4, 1e-3)
    assert selected.state.fit_row_ids_sha256 != ""
    assert len(calls) == 3  # two early fits, then one all-2023 refit


def test_regularization_tie_selects_larger_value() -> None:
    oof = _oof()

    def identity_optimizer(fun, x0, **kwargs):
        return SimpleNamespace(x=np.asarray(x0), success=True, fun=float(fun(x0)))

    selected = select_calibration(
        oof,
        kind="H2",
        grid=(1e-4, 1e-3, 1e-2),
        fit_month_max=7,
        validation_month_min=8,
        optimizer=identity_optimizer,
    )
    assert selected.selected_regularization == 1e-2


@pytest.mark.parametrize(
    "change",
    [
        lambda frame: frame.__setitem__("row_id", ["same"] * len(frame)),
        lambda frame: frame.__setitem__("game_month", 0),
        lambda frame: frame.__setitem__("probability", np.nan),
        lambda frame: frame.__setitem__("control_success", 2),
    ],
)
def test_invalid_selection_rows_are_rejected(change) -> None:
    oof = _oof()
    change(oof)
    with pytest.raises(CalibrationError):
        select_calibration(
            oof,
            kind="H2",
            grid=(1e-3,),
            fit_month_max=7,
            validation_month_min=8,
        )


def test_missing_h3_effect_and_target_length_mismatch_are_rejected() -> None:
    probability, target = _signal()
    segments = _segments(len(target)).drop(columns=["count_state"])
    with pytest.raises(CalibrationError, match="count_state"):
        fit_h3(probability, target, segments, regularization=0.01, clip=1e-6)
    with pytest.raises(CalibrationError, match="length"):
        fit_h2(probability[:-1], target, regularization=0.01, clip=1e-6)


def test_nonfinite_optimizer_output_is_rejected() -> None:
    oof = _oof()

    def bad_optimizer(fun, x0, **kwargs):
        return SimpleNamespace(x=np.full(len(x0), np.nan), success=True, fun=np.nan)

    with pytest.raises(CalibrationError, match="optimizer"):
        select_calibration(
            oof,
            kind="H2",
            grid=(1e-3,),
            fit_month_max=7,
            validation_month_min=8,
            optimizer=bad_optimizer,
        )


def test_calibration_state_round_trip_and_corruption_rejection() -> None:
    probability, target = _signal()
    state = fit_h3(
        probability, target, _segments(len(target)), regularization=0.01, clip=1e-6
    )
    payload = calibration_state_payload(state)
    restored = calibration_state_from_payload(payload)
    assert calibration_state_payload(restored) == payload
    assert calibration_state_sha256(restored) == calibration_state_sha256(state)

    corrupt = deepcopy(payload)
    corrupt["slope"] += 0.1
    with pytest.raises(CalibrationError, match="digest"):
        calibration_state_from_payload(corrupt)


@pytest.mark.parametrize(
    ("regularization", "clip"),
    [(0.0, 1e-6), (-1.0, 1e-6), (np.inf, 1e-6), (0.1, 0.0), (0.1, 0.5)],
)
def test_invalid_hyperparameters_are_rejected(regularization: float, clip: float) -> None:
    probability, target = _signal()
    with pytest.raises(CalibrationError):
        fit_h2(probability, target, regularization=regularization, clip=clip)
