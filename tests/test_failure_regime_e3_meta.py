from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.failure_regime_e3.meta import E3MetaError, build_meta_frame


STREAM_NAMES = (
    "p_e2", "p_s_global", "p_s_fast", "p_s_r", "p_s_f",
    "p_middle", "p_wild", "p_reverse",
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["r2", "r1", "r3"],
            "target": [1, 0, 1],
            "oof_year": [2022, 2023, 2024],
            "pitcher_id": ["p2", "p1", "p3"],
            "game_type": ["F", "R", "R"],
            "balls_before": [3, 1, 0],
            "strikes_before": [2, 1, 0],
            "inning": [9, 5, 1],
            "num_runners_on": [2, 1, 0],
            "li": [2.2, 1.0, 0.3],
            "asof_pitcher_n": [10, 100, 4],
            "asof_batter_n": [30, 80, 5],
            "asof_pitcher_success_rate": [0.4, 0.6, 0.5],
            "asof_batter_success_rate": [0.5, 0.45, 0.55],
            "asof_pitcher_prev1_game_success_rate": [0.3, 0.7, 0.5],
            "asof_pitcher_prev5_game_success_rate": [0.45, 0.55, 0.5],
        }
    )


def _streams(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    probabilities = {
        "p_e2": [0.55, 0.45, 0.50],
        "p_s_global": [0.60, 0.40, 0.55],
        "p_s_fast": [0.65, 0.35, 0.52],
        "p_s_r": [0.58, 0.38, 0.56],
        "p_s_f": [0.70, 0.48, 0.49],
        "p_middle": [0.20, 0.10, 0.15],
        "p_wild": [0.10, 0.30, 0.12],
        "p_reverse": [0.05, 0.15, 0.08],
    }
    return {
        name: pd.DataFrame({"row_id": rows["row_id"].iloc[::-1], "probability": values[::-1]})
        for name, values in probabilities.items()
    }


def test_meta_features_align_streams_by_row_id_and_use_active_regime() -> None:
    rows = _rows()
    meta = build_meta_frame(rows, _streams(rows), include_target=True)

    assert meta["row_id"].tolist() == rows["row_id"].tolist()
    assert meta["target"].tolist() == [1, 0, 1]
    assert meta["p_s_global"].tolist() == pytest.approx([0.60, 0.40, 0.55])
    assert meta["p_regime"].tolist() == pytest.approx([0.70, 0.38, 0.56])
    assert meta["d_regime_e2"].tolist() == pytest.approx([0.15, -0.07, 0.06])
    assert meta["middle_regime_interaction"].tolist() == pytest.approx([0.03, -0.007, 0.009])
    assert meta["pitcher_recent_gap"].tolist() == pytest.approx([-0.15, 0.15, 0.0])
    assert np.isfinite(meta.select_dtypes(include=[np.number]).to_numpy()).all()


def test_meta_features_are_row_local_under_shuffle_and_singleton() -> None:
    rows = _rows()
    streams = _streams(rows)
    full = build_meta_frame(rows, streams, include_target=True).set_index("row_id")
    shuffled_rows = rows.iloc[::-1].reset_index(drop=True)
    shuffled = build_meta_frame(shuffled_rows, streams, include_target=True).set_index("row_id")

    pd.testing.assert_frame_equal(full.sort_index(), shuffled.sort_index())
    for row_id in rows["row_id"]:
        singleton_rows = rows.loc[rows["row_id"].eq(row_id)].reset_index(drop=True)
        singleton = build_meta_frame(singleton_rows, streams, include_target=True).set_index("row_id")
        pd.testing.assert_frame_equal(full.loc[[row_id]], singleton)


def test_meta_rejects_duplicate_stream_identity() -> None:
    rows = _rows()
    streams = _streams(rows)
    streams["p_e2"] = pd.concat([streams["p_e2"], streams["p_e2"].iloc[[0]]], ignore_index=True)

    with pytest.raises(E3MetaError, match="stream row identity differs"):
        build_meta_frame(rows, streams, include_target=True)
