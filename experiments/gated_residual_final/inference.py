from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from experiments.direct_expert.contracts import expert_spec
from experiments.direct_expert.features import DirectFeatureState, feature_profile, transform_direct_features

from .calibration import TemporalCalibrator, apply_calibration
from .oof import direct_probability
from .residual import gated_residual, row_reliability
from .selection import CandidateConfig


class FinalInferenceError(ValueError):
    pass


def _metadata(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "row_id", "game_type", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    }
    if type(rows) is not pd.DataFrame or rows.empty or not required.issubset(rows.columns):
        raise FinalInferenceError("inference rows differ")
    if rows["row_id"].isna().any() or not rows["row_id"].is_unique:
        raise FinalInferenceError("inference row identity differs")
    output = rows.loc[:, sorted(required)].copy(deep=True)
    output["hand_matchup"] = output["pitcher_hand"].astype(str) + output["batter_hand"].astype(str)
    return output


def combine_final_probabilities(
    *,
    rows: pd.DataFrame,
    anchor: np.ndarray,
    d0: np.ndarray,
    d5: np.ndarray,
    pitcher_counts: Mapping[str, int],
    batter_counts: Mapping[str, int],
    config: CandidateConfig,
    calibrator: TemporalCalibrator | None,
) -> np.ndarray:
    metadata = _metadata(rows)
    anchor_probability = np.asarray(anchor, dtype="float64")
    if anchor_probability.shape != (len(metadata),):
        raise FinalInferenceError("anchor prediction shape differs")
    direct = direct_probability(
        game_type=metadata["game_type"].to_numpy(), d0=np.asarray(d0), d5=np.asarray(d5),
    )
    if config.archetype == "G0":
        strength = metadata["game_type"].astype(str).eq("R").to_numpy(dtype="float64")
    else:
        if config.k not in (25, 100, 400):
            raise FinalInferenceError("inference reliability differs")
        strength = row_reliability(
            metadata, pitcher_counts=pitcher_counts, batter_counts=batter_counts, k=config.k,
        )
    result = gated_residual(
        anchor=anchor_probability, direct=direct, row_reliability=strength, alpha=config.alpha,
    )
    if config.archetype in {"G2", "G3"}:
        if calibrator is None or config.beta is None:
            raise FinalInferenceError("inference calibrator is absent")
        work = metadata.copy(deep=True)
        work["probability"] = result
        result = apply_calibration(calibrator, work, beta=config.beta)
    if result.shape != (len(metadata),) or not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise FinalInferenceError("inference probability values differ")
    return result


class FinalPredictor:
    def __init__(
        self,
        *,
        feature_state: DirectFeatureState,
        models: Mapping[str, Sequence[object]],
        e2_predictor: object,
        pitcher_counts: Mapping[str, int],
        batter_counts: Mapping[str, int],
        config: CandidateConfig,
        calibrator: TemporalCalibrator | None,
    ):
        if set(models) != {"D0", "D5"} or any(len(tuple(value)) != 3 for value in models.values()):
            raise FinalInferenceError("inference model set differs")
        self.feature_state = feature_state
        self.models = {key: tuple(value) for key, value in models.items()}
        self.e2_predictor = e2_predictor
        self.pitcher_counts = dict(pitcher_counts)
        self.batter_counts = dict(batter_counts)
        self.config = config
        self.calibrator = calibrator

    def _direct(self, rows: pd.DataFrame) -> Mapping[str, np.ndarray]:
        batch = transform_direct_features(rows, self.feature_state)
        output = {}
        for role in ("D0", "D5"):
            spec = expert_spec(role)
            columns, _ = feature_profile(batch.frame, self.feature_state.categorical_columns, spec.interaction_profile)
            values = []
            for model in self.models[role]:
                probability = np.asarray(model.predict_proba(batch.frame.loc[:, columns])[:, 1], dtype="float64")
                values.append(np.clip(probability, 1e-6, 1.0 - 1e-6))
            output[role] = np.add.reduce(values) / len(values)
        return output

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        clean = rows.drop(columns=["control_success"], errors="ignore").copy(deep=True)
        direct = self._direct(clean)
        anchor = np.asarray(self.e2_predictor.predict_batch(clean, batch_size=4096), dtype="float64")
        return combine_final_probabilities(
            rows=clean, anchor=anchor, d0=direct["D0"], d5=direct["D5"],
            pitcher_counts=self.pitcher_counts, batter_counts=self.batter_counts,
            config=self.config, calibrator=self.calibrator,
        )
