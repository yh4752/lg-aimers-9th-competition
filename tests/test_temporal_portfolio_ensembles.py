from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.temporal_portfolio.ensembles import (
    AnchorSpec,
    EnsembleError,
    OOFStream,
    build_correction_recipes,
    build_raw_recipes,
    evaluate_forward_calibration,
)


def _calibration_frame() -> pd.DataFrame:
    rows = []
    for year in (2022, 2023, 2024):
        for index in range(40):
            target = index % 2
            probability = 0.72 if target else 0.28
            rows.append((f"{year}-{index}", year, target, probability))
    return pd.DataFrame(rows, columns=["row_id", "valid_year", "target", "probability"])


def _streams() -> dict[str, OOFStream]:
    base = _calibration_frame()
    output = {"C0": OOFStream.from_frame(base)}
    for name, shift in (("C1", -0.02), ("C2", 0.02), ("C3", 0.04)):
        frame = base.copy()
        direction = np.where(frame["target"].eq(1), shift, -shift)
        frame["probability"] = np.clip(frame["probability"] + direction, 0, 1)
        output[name] = OOFStream.from_frame(frame)
    return output


def test_forward_calibration_never_fits_the_scored_fold() -> None:
    result = evaluate_forward_calibration(_calibration_frame(), method="platt")
    assert result.fit_years_by_score_year == {2023: (2022,), 2024: (2022, 2023)}
    assert 2022 not in result.calibrated_probability_by_year


def test_recipe_grid_is_bounded_and_corrections_never_stack() -> None:
    recipes = build_raw_recipes(_streams(), primary="C0")
    assert len(recipes) <= 24
    corrected = build_correction_recipes(
        top_raw=recipes[:5], approved_anchor=AnchorSpec(beta=__import__("decimal").Decimal("0.05"))
    )
    assert len(corrected) <= 20
    assert all(not (item.anchor is not None and item.calibration is not None) for item in corrected)


def test_raw_recipe_grid_is_deterministic_aligned_and_bounded() -> None:
    streams = _streams()
    assert build_raw_recipes(streams, primary="C0") == build_raw_recipes(dict(reversed(list(streams.items()))), primary="C0")
    too_many = dict(streams)
    too_many["C4"] = streams["C1"]
    with pytest.raises(EnsembleError, match="exceeded"):
        build_raw_recipes(too_many, primary="C0")
    bad = dict(streams)
    frame = _calibration_frame().iloc[:-1]
    bad["C1"] = OOFStream.from_frame(frame)
    with pytest.raises(EnsembleError, match="aligned"):
        build_raw_recipes(bad, primary="C0")


def test_forward_calibration_requires_prior_two_class_evidence() -> None:
    frame = _calibration_frame()
    frame.loc[frame["valid_year"].eq(2022), "target"] = 1
    with pytest.raises(EnsembleError, match="both target classes"):
        evaluate_forward_calibration(frame, method="platt")


def test_calibration_outputs_are_detached_and_invalid_inputs_rejected() -> None:
    frame = _calibration_frame()
    result = evaluate_forward_calibration(frame, method="platt")
    before = result.calibrated_probability_by_year[2023].copy()
    frame["probability"] = 0.5
    np.testing.assert_array_equal(result.calibrated_probability_by_year[2023], before)
    assert not result.calibrated_probability_by_year[2023].flags.writeable
    with pytest.raises(EnsembleError, match="method"):
        evaluate_forward_calibration(_calibration_frame(), method="isotonic")
