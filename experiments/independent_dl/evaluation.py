"""Deterministic standalone, diversity, and aligned-blend diagnostics."""

from __future__ import annotations

import math
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OOF_COLUMNS = ("row_id", "fold", "season", "game_type", "target", "probability")


class EvaluationError(ValueError):
    """Raised before incomparable OOF predictions can enter a diagnostic."""


def _validated_oof(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise EvaluationError("OOF input must be a DataFrame")
    missing = set(OOF_COLUMNS) - set(frame.columns)
    if missing:
        raise EvaluationError(f"OOF input is missing columns: {sorted(missing)}")
    output = frame.loc[:, OOF_COLUMNS].copy()
    row_id = output["row_id"].astype("string")
    if row_id.isna().any() or row_id.duplicated().any():
        raise EvaluationError("row_id must be non-null and unique")
    output["row_id"] = row_id.astype(str)
    target = pd.to_numeric(output["target"], errors="coerce").to_numpy(
        dtype="float64"
    )
    probability = pd.to_numeric(
        output["probability"], errors="coerce"
    ).to_numpy(dtype="float64")
    if not np.isfinite(target).all() or not np.isin(target, [0.0, 1.0]).all():
        raise EvaluationError("target must contain only finite 0/1 values")
    if not np.isfinite(probability).all() or (
        (probability < 0.0) | (probability > 1.0)
    ).any():
        raise EvaluationError("probability must be finite and inside [0, 1]")
    season = pd.to_numeric(output["season"], errors="coerce").to_numpy(
        dtype="float64"
    )
    if not np.isfinite(season).all() or not np.equal(season, np.floor(season)).all():
        raise EvaluationError("season must contain finite integers")
    folds = output["fold"].astype("string")
    expected = np.array([f"valid_{int(item)}" for item in season], dtype=object)
    if folds.isna().any() or not np.array_equal(folds.astype(str).to_numpy(), expected):
        raise EvaluationError("fold and season are inconsistent")
    game_type = output["game_type"].astype("string")
    if game_type.isna().any() or game_type.str.len().eq(0).any():
        raise EvaluationError("game_type must contain non-empty values")
    output["fold"] = folds.astype(str)
    output["season"] = season.astype("int64")
    output["game_type"] = game_type.astype(str)
    output["target"] = target
    output["probability"] = probability
    return output


def _metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return {
        "n": int(len(target)),
        "brier": float(np.mean(np.square(probability - target))),
        "logloss": float(
            -np.mean(target * np.log(clipped) + (1.0 - target) * np.log1p(-clipped))
        ),
        "base_rate": float(np.mean(target)),
        "prediction_mean": float(np.mean(probability)),
        "calibration_gap": float(np.mean(probability) - np.mean(target)),
    }


def evaluate_predictions(frame: pd.DataFrame) -> dict[str, object]:
    oof = _validated_oof(frame)

    def metrics(rows: pd.DataFrame) -> dict[str, float | int]:
        return _metrics(
            rows["target"].to_numpy(dtype="float64"),
            rows["probability"].to_numpy(dtype="float64"),
        )

    return {
        "global": metrics(oof),
        "folds": {
            str(name): metrics(rows)
            for name, rows in oof.groupby("fold", sort=True, observed=True)
        },
        "game_type": {
            str(name): metrics(rows)
            for name, rows in oof.groupby("game_type", sort=True, observed=True)
        },
    }


def _aligned(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = _validated_oof(left)
    second = _validated_oof(right)
    binding = ("row_id", "fold", "season", "game_type", "target")
    if len(first) != len(second) or any(
        not np.array_equal(first[column].to_numpy(), second[column].to_numpy())
        for column in binding
    ):
        raise EvaluationError("row alignment differs between OOF inputs")
    return first, second


def evaluate_aligned_blends(
    dl_frame: pd.DataFrame,
    ml_frame: pd.DataFrame,
    *,
    weights: Iterable[float],
) -> pd.DataFrame:
    dl, ml = _aligned(dl_frame, ml_frame)
    target = dl["target"].to_numpy(dtype="float64")
    dl_probability = dl["probability"].to_numpy(dtype="float64")
    ml_probability = ml["probability"].to_numpy(dtype="float64")
    clip = 1e-6
    dl_logit = np.log(
        np.clip(dl_probability, clip, 1 - clip)
        / (1 - np.clip(dl_probability, clip, 1 - clip))
    )
    ml_logit = np.log(
        np.clip(ml_probability, clip, 1 - clip)
        / (1 - np.clip(ml_probability, clip, 1 - clip))
    )
    rows: list[dict[str, object]] = []
    for weight_value in weights:
        if isinstance(weight_value, bool) or not isinstance(weight_value, (int, float)):
            raise EvaluationError("blend weights must be numeric")
        weight = float(weight_value)
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise EvaluationError("blend weights must be finite and inside [0, 1]")
        probability_blend = weight * dl_probability + (1.0 - weight) * ml_probability
        logit = weight * dl_logit + (1.0 - weight) * ml_logit
        logit_blend = 1.0 / (1.0 + np.exp(-logit))
        for family, probability in (
            ("probability", probability_blend),
            ("logit", logit_blend),
        ):
            report = evaluate_predictions(dl.assign(probability=probability))
            row: dict[str, object] = {
                "blend_family": family,
                "dl_weight": weight,
                "global_brier": report["global"]["brier"],
                "global_logloss": report["global"]["logloss"],
            }
            for fold, metrics in report["folds"].items():
                row[f"{fold}_brier"] = metrics["brier"]
            for game_type, metrics in report["game_type"].items():
                row[f"game_type_{game_type}_brier"] = metrics["brier"]
            rows.append(row)
    return pd.DataFrame(rows)


def residual_correlation(dl_frame: pd.DataFrame, ml_frame: pd.DataFrame) -> float:
    dl, ml = _aligned(dl_frame, ml_frame)
    target = dl["target"].to_numpy(dtype="float64")
    first = dl["probability"].to_numpy(dtype="float64") - target
    second = ml["probability"].to_numpy(dtype="float64") - target
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        return 1.0 if np.array_equal(first, second) else 0.0
    return float(np.corrcoef(first, second)[0, 1])


def select_survivors(
    candidate_table: pd.DataFrame,
    *,
    ml_brier: float,
    proximity_delta: float,
    max_error_correlation: float,
    top_k_per_family: int,
) -> list[dict[str, object]]:
    required = {
        "candidate_id",
        "family",
        "brier",
        "blend_gain",
        "segment_gain",
        "error_correlation",
    }
    if not required.issubset(candidate_table.columns):
        raise EvaluationError("candidate table is missing survival columns")
    if candidate_table["candidate_id"].astype(str).duplicated().any():
        raise EvaluationError("candidate IDs must be unique")
    if isinstance(top_k_per_family, bool) or top_k_per_family < 1:
        raise EvaluationError("top_k_per_family must be positive")
    numeric_columns = (
        "brier",
        "blend_gain",
        "segment_gain",
        "error_correlation",
    )
    numeric = candidate_table.loc[:, numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype="float64")).all():
        raise EvaluationError("survival metrics must be finite")
    family_top = set(
        candidate_table.assign(_brier=numeric["brier"])
        .sort_values(["family", "_brier", "candidate_id"], kind="stable")
        .groupby("family", sort=False, observed=True)
        .head(top_k_per_family)["candidate_id"]
        .astype(str)
    )
    survivors: list[dict[str, object]] = []
    for index, row in candidate_table.reset_index(drop=True).iterrows():
        reasons: list[str] = []
        if float(numeric.iloc[index]["brier"]) <= ml_brier + proximity_delta:
            reasons.append("absolute")
        if float(numeric.iloc[index]["blend_gain"]) > 0.0:
            reasons.append("blend")
        if float(numeric.iloc[index]["segment_gain"]) > 0.0:
            reasons.append("segment")
        if float(numeric.iloc[index]["error_correlation"]) <= max_error_correlation:
            reasons.append("diversity")
        if str(row["candidate_id"]) in family_top:
            reasons.append("family_top")
        if reasons:
            survivors.append(
                {
                    "candidate_id": str(row["candidate_id"]),
                    "family": str(row["family"]),
                    "reasons": tuple(reasons),
                }
            )
    return survivors


def write_evaluation_artifacts(
    output_dir: str | Path,
    *,
    standalone: pd.DataFrame,
    diversity: pd.DataFrame,
    blend_contribution: pd.DataFrame,
    survivors: list[dict[str, object]],
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("standalone.csv", standalone),
        ("diversity.csv", diversity),
        ("blend_contribution.csv", blend_contribution),
    ):
        path = root / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    path = root / "survivors.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(survivors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
