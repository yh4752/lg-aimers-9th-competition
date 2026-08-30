from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.gated_residual_final.oof import OOFError, align_predictions, direct_probability


def _frame(row_dtype: str, target_dtype: str, probability=(0.2, 0.7)) -> pd.DataFrame:
    return pd.DataFrame({
        "row_id": pd.Series([2, 1], dtype=row_dtype),
        "target": pd.Series([1, 0], dtype=target_dtype),
        "probability": probability,
        "oof_year": [2024, 2024],
    })


def test_alignment_accepts_equivalent_numeric_dtypes() -> None:
    aligned = align_predictions({
        "e2": _frame("int64", "int8"),
        "d0": _frame("int32", "int64"),
    })

    assert aligned["e2"]["row_id"].tolist() == [1, 2]
    assert aligned["e2"]["row_id"].tolist() == aligned["d0"]["row_id"].tolist()


def test_alignment_rejects_different_target_values() -> None:
    changed = _frame("int64", "int8")
    changed.loc[0, "target"] = 0

    with pytest.raises(OOFError, match="target values differ"):
        align_predictions({"e2": _frame("int64", "int8"), "d0": changed})


def test_direct_probability_uses_d5_only_for_regular_season() -> None:
    result = direct_probability(
        game_type=np.array(["R", "F", "R"]),
        d0=np.array([0.2, 0.3, 0.4]),
        d5=np.array([0.8, 0.9, 0.6]),
    )

    np.testing.assert_allclose(result, [0.8, 0.3, 0.6])


def test_invalid_probability_is_rejected() -> None:
    bad = _frame("int64", "int8", probability=(0.2, 1.1))
    with pytest.raises(OOFError, match="probability values differ"):
        align_predictions({"bad": bad})
