from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


class E3MetaError(ValueError):
    pass


STREAM_NAMES = (
    "p_e2", "p_s_global", "p_s_fast", "p_s_r", "p_s_f",
    "p_middle", "p_wild", "p_reverse",
)
_ROW_REQUIRED = {"row_id", "pitcher_id", "game_type"}
_NUMERIC_ROWS = (
    "balls_before", "strikes_before", "inning", "num_runners_on", "li",
    "asof_pitcher_n", "asof_batter_n", "asof_pitcher_success_rate",
    "asof_batter_success_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
)


def _probability_stream(
    name: str,
    source: pd.DataFrame,
    row_ids: pd.Series,
) -> np.ndarray:
    if type(source) is not pd.DataFrame or set(source) != {"row_id", "probability"}:
        raise E3MetaError(f"stream columns differ: {name}")
    if source["row_id"].isna().any() or not source["row_id"].is_unique:
        raise E3MetaError(f"stream row identity differs: {name}")
    keyed = source.assign(row_id=source["row_id"].astype(str)).set_index("row_id")["probability"]
    values = pd.to_numeric(row_ids.astype(str).map(keyed), errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise E3MetaError(f"stream probability differs: {name}")
    return values


def _numeric(rows: pd.DataFrame, name: str) -> np.ndarray:
    if name not in rows:
        return np.zeros(len(rows), dtype="float64")
    values = pd.to_numeric(rows[name], errors="coerce").fillna(0).to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise E3MetaError(f"row feature contains infinity: {name}")
    return values


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def build_meta_frame(
    rows: pd.DataFrame,
    streams: Mapping[str, pd.DataFrame],
    *,
    include_target: bool,
) -> pd.DataFrame:
    if type(rows) is not pd.DataFrame or rows.empty or rows.columns.has_duplicates:
        raise E3MetaError("rows differ")
    if not _ROW_REQUIRED.issubset(rows.columns):
        raise E3MetaError("row identity columns differ")
    if rows["row_id"].isna().any() or not rows["row_id"].astype(str).is_unique:
        raise E3MetaError("row identity differs")
    if set(streams) != set(STREAM_NAMES):
        raise E3MetaError("stream names differ")
    if include_target and not {"target", "oof_year"}.issubset(rows.columns):
        raise E3MetaError("target metadata differs")

    output = pd.DataFrame({
        "row_id": rows["row_id"].astype(str).to_numpy(copy=True),
        "pitcher_id": rows["pitcher_id"].astype(str).to_numpy(copy=True),
        "game_type": rows["game_type"].astype("string").fillna("__MISSING__").astype(str).to_numpy(),
    })
    if include_target:
        target = pd.to_numeric(rows["target"], errors="coerce")
        years = pd.to_numeric(rows["oof_year"], errors="coerce")
        if target.isna().any() or not target.isin((0, 1)).all() or years.isna().any():
            raise E3MetaError("target metadata values differ")
        output["target"] = target.to_numpy(dtype="int8")
        output["oof_year"] = years.to_numpy(dtype="int16")

    arrays = {
        name: _probability_stream(name, streams[name], rows["row_id"])
        for name in STREAM_NAMES
    }
    for name, values in arrays.items():
        output[name] = values
    is_f = output["game_type"].eq("F").to_numpy()
    regime = np.where(is_f, arrays["p_s_f"], arrays["p_s_r"])
    output["p_regime"] = regime
    output["logit_e2"] = _logit(arrays["p_e2"])
    for name in ("p_s_global", "p_s_fast", "p_s_r", "p_s_f"):
        output[f"d_{name[2:]}_e2"] = arrays[name] - arrays["p_e2"]
    output["d_regime_e2"] = regime - arrays["p_e2"]
    success_stack = np.column_stack([
        arrays["p_s_global"], arrays["p_s_fast"], arrays["p_s_r"], arrays["p_s_f"],
    ])
    output["success_spread"] = success_stack.max(axis=1) - success_stack.min(axis=1)
    output["subtype_sum"] = arrays["p_middle"] + arrays["p_wild"] + arrays["p_reverse"]
    output["subtype_max"] = np.max(
        np.column_stack([arrays["p_middle"], arrays["p_wild"], arrays["p_reverse"]]),
        axis=1,
    )
    output["middle_regime_interaction"] = arrays["p_middle"] * output["d_regime_e2"]
    output["wild_regime_interaction"] = arrays["p_wild"] * output["d_regime_e2"]
    output["reverse_regime_interaction"] = arrays["p_reverse"] * output["d_regime_e2"]

    for name in _NUMERIC_ROWS:
        output[name] = _numeric(rows, name)
    output["log_pitcher_n"] = np.log1p(np.clip(output["asof_pitcher_n"], 0, None))
    output["log_batter_n"] = np.log1p(np.clip(output["asof_batter_n"], 0, None))
    output["pitcher_recent_gap"] = (
        output["asof_pitcher_prev1_game_success_rate"]
        - output["asof_pitcher_prev5_game_success_rate"]
    )
    output["pitcher_batter_rate_gap"] = (
        output["asof_pitcher_success_rate"] - output["asof_batter_success_rate"]
    )
    numeric = output.select_dtypes(include=[np.number]).to_numpy(dtype="float64")
    if not np.isfinite(numeric).all():
        raise E3MetaError("meta frame contains non-finite values")
    return output
