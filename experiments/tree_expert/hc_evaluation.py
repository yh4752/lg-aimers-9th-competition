from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .hc_contracts import HCContract
from .hc_decisions import C1Evidence, C2Evidence
from .hc_metrics import (
    apply_c0_deciles,
    brier_score,
    calibration_gap,
    expected_calibration_error,
    fit_c0_decile_edges,
    maximum_segment_regression,
    paired_cluster_bootstrap,
    segment_diagnostics,
)


def _validated(frame: object, candidate: str) -> pd.DataFrame:
    required = {
        "row_id",
        "oof_year",
        "target",
        "p0",
        candidate,
        "pitcher_id",
        "game_type",
        "pitcher_id_known",
        "batter_id_known",
        "pitcher_hand",
        "batter_hand",
        "balls_before",
        "strikes_before",
        "base_state",
    }
    if (
        type(frame) is not pd.DataFrame
        or frame.empty
        or not required.issubset(frame.columns)
        or frame["row_id"].isna().any()
        or not frame["row_id"].is_unique
        or set(pd.to_numeric(frame["oof_year"], errors="coerce").astype(int))
        != {2022, 2023, 2024}
    ):
        raise ValueError("candidate evidence differs")
    output = frame.copy(deep=True)
    for name in ("p0", candidate):
        values = pd.to_numeric(output[name], errors="coerce")
        if values.isna().any() or not values.between(0, 1).all():
            raise ValueError("candidate probability differs")
    if not pd.to_numeric(output["target"], errors="coerce").isin([0, 1]).all():
        raise ValueError("candidate target differs")
    structure = output.loc[output["oof_year"].isin((2022, 2023))]
    output["c0_decile"] = apply_c0_deciles(
        output["p0"], fit_c0_decile_edges(structure["p0"])
    )
    return output


def _fold_gains(frame: pd.DataFrame, candidate: str) -> dict[int, float]:
    return {
        year: brier_score(part["target"], part["p0"])
        - brier_score(part["target"], part[candidate])
        for year, part in frame.groupby("oof_year", sort=True)
    }


def _maximum_segment(
    frame: pd.DataFrame, candidate: str, contract: HCContract
) -> float:
    report = segment_diagnostics(
        frame,
        candidate,
        minimum_rows=int(contract.gates["minimum_segment_rows"]),
    )
    return maximum_segment_regression(report)


def build_c1_evidence(
    frame: pd.DataFrame,
    seed_predictions: Mapping[int, pd.DataFrame],
    contract: HCContract,
) -> C1Evidence:
    rows = _validated(frame, "p1")
    if set(seed_predictions) != set(contract.seeds):
        raise ValueError("C1 seed evidence differs")
    tolerance = float(contract.gates["seed_fold_tolerance"])
    counts = {year: 0 for year in (2022, 2023, 2024)}
    for seed in contract.seeds:
        candidate = seed_predictions[seed]
        if (
            type(candidate) is not pd.DataFrame
            or not {"row_id", "oof_year", "target", "p0", "p1"}.issubset(candidate.columns)
            or candidate["row_id"].astype(str).tolist()
            != rows["row_id"].astype(str).tolist()
            or not np.array_equal(candidate["target"].to_numpy(), rows["target"].to_numpy())
        ):
            raise ValueError("C1 seed row alignment differs")
        for year in counts:
            selected = candidate["oof_year"].eq(year)
            base = brier_score(candidate.loc[selected, "target"], candidate.loc[selected, "p0"])
            score = brier_score(candidate.loc[selected, "target"], candidate.loc[selected, "p1"])
            counts[year] += int(score <= base + tolerance)
    gains = _fold_gains(rows, "p1")
    bootstrap = paired_cluster_bootstrap(
        rows,
        "p0",
        "p1",
        repeats=int(contract.gates["bootstrap_repeats"]),
        seed=int(contract.gates["bootstrap_seed"]),
    )
    base = brier_score(rows["target"], rows["p0"])
    score = brier_score(rows["target"], rows["p1"])
    return C1Evidence(
        weighted_brier=score,
        weighted_gain=base - score,
        confirmation_gain=gains[2024],
        minimum_fold_gain=min(gains.values()),
        maximum_segment_regression=_maximum_segment(rows, "p1", contract),
        bootstrap_lower_95=bootstrap.lower_95,
        non_worse_seed_counts=counts,
    )


def build_c2_evidence(frame: pd.DataFrame, contract: HCContract) -> C2Evidence:
    rows = _validated(frame, "p2")
    if "p1" not in rows:
        raise ValueError("C2 incremental evidence differs")
    gains = _fold_gains(rows, "p2")
    bootstrap = paired_cluster_bootstrap(
        rows,
        "p0",
        "p2",
        repeats=int(contract.gates["bootstrap_repeats"]),
        seed=int(contract.gates["bootstrap_seed"]),
    )
    base = brier_score(rows["target"], rows["p0"])
    c1 = brier_score(rows["target"], rows["p1"])
    score = brier_score(rows["target"], rows["p2"])
    bins = int(contract.calibration["ece_bins"])
    return C2Evidence(
        weighted_brier=score,
        weighted_gain=base - score,
        incremental_gain=c1 - score,
        minimum_fold_gain=min(gains.values()),
        c0_calibration_gap=calibration_gap(rows["target"], rows["p0"]),
        candidate_calibration_gap=calibration_gap(rows["target"], rows["p2"]),
        c0_ece=expected_calibration_error(rows["target"], rows["p0"], bins=bins),
        candidate_ece=expected_calibration_error(rows["target"], rows["p2"], bins=bins),
        maximum_segment_regression=_maximum_segment(rows, "p2", contract),
        bootstrap_lower_95=bootstrap.lower_95,
    )
