from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

import experiments.hierarchical_tabm.context_features as context_module
from experiments.hierarchical_tabm.context_features import (
    ContextFeatureError,
    select_k_from_scores,
    select_smoothing_k,
)


def _group_rows(
    season: int,
    group: int,
    count: int,
    successes: int,
    *,
    prefix: str,
) -> list[dict[str, object]]:
    return [
        {
            "row_id": f"{prefix}-{group}-{position}",
            "season": season,
            "balls_before": group,
            "strikes_before": 0,
            "pitcher_hand": "L" if group == 0 else "R",
            "batter_hand": "R" if group == 0 else "L",
            "base_state": "000" if group == 0 else "100",
            "outs_before": group,
            "game_type": "R",
            "control_success": int(position < successes),
        }
        for position in range(count)
    ]


def _development_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        _group_rows(2021, 0, 10, 9, prefix="train")
        + _group_rows(2021, 1, 100, 45, prefix="train")
    )
    valid = pd.DataFrame(
        _group_rows(2022, 0, 1, 0, prefix="valid")
        + _group_rows(2022, 1, 20, 0, prefix="valid")
    )
    return train, valid


def test_selects_lowest_development_brier_and_keeps_order() -> None:
    train, valid = _development_rows()
    result = select_smoothing_k(
        train, valid, (32.0, 128.0, 512.0), tie_tolerance=1e-6
    )

    assert result.selected_k == 128.0
    assert result.fold == "2021->2022"
    assert result.row_count == 21
    assert tuple(result.scores) == (32.0, 128.0, 512.0)
    assert result.scores[128.0] == pytest.approx(0.21333975742907896)


def test_tie_within_tolerance_selects_larger_k() -> None:
    scores = OrderedDict(
        [(32.0, 0.2500000), (128.0, 0.2500008), (512.0, 0.2500020)]
    )
    assert select_k_from_scores(scores, tie_tolerance=1e-6) == 128.0


def test_brier_is_row_weighted_not_group_weighted() -> None:
    train, valid = _development_rows()
    result = select_smoothing_k(
        train, valid, (32.0,), tie_tolerance=0.0
    )
    expected = (0.7621449803427607**2 + 20 * 0.4501412895636345**2) / 21
    assert result.scores[32.0] == pytest.approx(expected)


def test_validation_transform_never_receives_target(monkeypatch) -> None:
    train, valid = _development_rows()
    original = context_module.transform_frozen
    observed_columns: list[tuple[str, ...]] = []

    def recording_transform(frame, state):
        observed_columns.append(tuple(frame.columns))
        assert "control_success" not in frame
        return original(frame, state)

    monkeypatch.setattr(context_module, "transform_frozen", recording_transform)
    select_smoothing_k(train, valid, (32.0, 128.0), tie_tolerance=1e-6)
    assert len(observed_columns) == 2


@pytest.mark.parametrize(
    ("train_season", "valid_season"),
    [(2022, 2022), (2021, 2023), (2023, 2022), (2021, 2024)],
)
def test_only_2021_to_2022_development_fold_is_accepted(
    train_season: int, valid_season: int
) -> None:
    train, valid = _development_rows()
    train["season"] = train_season
    valid["season"] = valid_season
    with pytest.raises(ContextFeatureError, match="development fold"):
        select_smoothing_k(train, valid, (32.0,), tie_tolerance=1e-6)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1, 2.0])
def test_invalid_validation_target_is_rejected(bad: float) -> None:
    train, valid = _development_rows()
    valid.loc[valid.index[0], "control_success"] = bad
    with pytest.raises(ContextFeatureError, match="control_success"):
        select_smoothing_k(train, valid, (32.0,), tie_tolerance=1e-6)


@pytest.mark.parametrize(
    "candidates",
    [(), (32.0, 32.0), (128.0, 32.0), (0.0,), (np.inf,), (True,)],
)
def test_candidate_grid_must_be_exact_ordered_positive_values(candidates) -> None:
    train, valid = _development_rows()
    with pytest.raises(ContextFeatureError, match="candidates"):
        select_smoothing_k(train, valid, candidates, tie_tolerance=1e-6)


@pytest.mark.parametrize("tolerance", [-1.0, np.nan, np.inf, True])
def test_invalid_tie_tolerance_is_rejected(tolerance: object) -> None:
    with pytest.raises(ContextFeatureError, match="tie_tolerance"):
        select_k_from_scores({32.0: 0.2}, tie_tolerance=tolerance)


@pytest.mark.parametrize(
    "scores",
    [{}, {32.0: np.nan}, {0.0: 0.2}, {32.0: -0.1}, {32.0: 1.1}],
)
def test_invalid_score_mapping_is_rejected(scores) -> None:
    with pytest.raises(ContextFeatureError, match="scores"):
        select_k_from_scores(scores, tie_tolerance=1e-6)
