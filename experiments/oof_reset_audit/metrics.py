from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .types import PredictionSet


SEGMENTS = (
    "game_month",
    "game_type",
    "count_state",
    "hand_matchup",
    "base_out_state",
    "pitcher_known",
    "batter_known",
)


def validate_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"row_id", "target", "probability"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("prediction columns differ")
    result = frame.copy()
    if result["row_id"].isna().any():
        raise ValueError("row_id is missing")
    result["row_id"] = result["row_id"].astype(str)
    if result["row_id"].duplicated().any():
        raise ValueError("row_id must be unique")
    target = pd.to_numeric(result["target"], errors="coerce").to_numpy("float64")
    probability = pd.to_numeric(
        result["probability"], errors="coerce"
    ).to_numpy("float64")
    if not np.isfinite(target).all() or not np.isin(target, (0.0, 1.0)).all():
        raise ValueError("target must contain only 0 and 1")
    if (
        not np.isfinite(probability).all()
        or ((probability < 0.0) | (probability > 1.0)).any()
    ):
        raise ValueError("probability must be finite and in [0, 1]")
    result["target"] = target.astype("int8")
    result["probability"] = probability
    return result


def _brier(target: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean(np.square(probability - target), dtype=np.float64))


def calibration_deciles(evidence: PredictionSet) -> pd.DataFrame:
    frame = validate_prediction_frame(evidence.frame)
    quantile = pd.qcut(frame["probability"], 10, labels=False, duplicates="drop")
    if quantile.isna().all():
        quantile = pd.Series(np.zeros(len(frame), dtype="int64"), index=frame.index)
    table = (
        frame.assign(decile=quantile)
        .groupby("decile", observed=True)
        .agg(
            rows=("row_id", "size"),
            target_mean=("target", "mean"),
            prediction_mean=("probability", "mean"),
        )
        .reset_index()
    )
    table["decile"] = table["decile"].astype("int64")
    table.insert(0, "fold", evidence.fold)
    table.insert(0, "model_id", evidence.model_id)
    return table


def single_model_metrics(evidence: PredictionSet) -> dict[str, object]:
    frame = validate_prediction_frame(evidence.frame)
    target = frame["target"].to_numpy("float64")
    probability = frame["probability"].to_numpy("float64")
    brier = _brier(target, probability)
    rate = float(target.mean())
    prior_brier = rate * (1.0 - rate)
    if prior_brier <= 0.0:
        raise ValueError("local BSS requires both target classes")
    deciles = calibration_deciles(evidence)
    ece = float(
        (
            deciles["rows"] / len(frame)
            * (deciles["target_mean"] - deciles["prediction_mean"]).abs()
        ).sum()
    )
    return {
        "model_id": evidence.model_id,
        "fold": evidence.fold,
        "trust": evidence.trust.value,
        "rows": len(frame),
        "target_mean": rate,
        "prediction_mean": float(probability.mean()),
        "prediction_std": float(probability.std()),
        "prediction_min": float(probability.min()),
        "prediction_max": float(probability.max()),
        "brier": brier,
        "local_bss": 100_000.0 * (1.0 - brier / prior_brier),
        "roc_auc": float(roc_auc_score(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "ece_10": ece,
    }


def align_pair(anchor: PredictionSet, candidate: PredictionSet) -> pd.DataFrame:
    if anchor.fold != candidate.fold:
        raise ValueError("fold differs")
    left = validate_prediction_frame(anchor.frame)
    right = validate_prediction_frame(candidate.frame)
    if set(left["row_id"]) != set(right["row_id"]):
        raise ValueError("row_id set differs")
    right = (
        right.set_index("row_id", drop=False)
        .loc[left["row_id"]]
        .reset_index(drop=True)
    )
    left = left.reset_index(drop=True)
    if not np.array_equal(left["target"].to_numpy(), right["target"].to_numpy()):
        raise ValueError("target differs")
    output = left.rename(columns={"probability": "anchor_probability"})
    output["candidate_probability"] = right["probability"].to_numpy("float64")
    for column in SEGMENTS:
        if column in left and column in right:
            matches = left[column].eq(right[column]) | (
                left[column].isna() & right[column].isna()
            )
            if not bool(matches.all()):
                raise ValueError(f"segment differs: {column}")
        elif column in right:
            output[column] = right[column].to_numpy()
    return output


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def paired_metrics(anchor: PredictionSet, candidate: PredictionSet) -> dict[str, object]:
    paired = align_pair(anchor, candidate)
    target = paired["target"].to_numpy("float64")
    anchor_probability = paired["anchor_probability"].to_numpy("float64")
    candidate_probability = paired["candidate_probability"].to_numpy("float64")
    anchor_loss = np.square(anchor_probability - target)
    candidate_loss = np.square(candidate_probability - target)
    return {
        "anchor_model_id": anchor.model_id,
        "candidate_model_id": candidate.model_id,
        "fold": anchor.fold,
        "rows": len(paired),
        "anchor_brier": float(anchor_loss.mean()),
        "candidate_brier": float(candidate_loss.mean()),
        "gain_vs_anchor": float((anchor_loss - candidate_loss).mean()),
        "prediction_correlation": _correlation(
            anchor_probability, candidate_probability
        ),
        "loss_correlation": _correlation(anchor_loss, candidate_loss),
        "residual_correlation": _correlation(
            target - anchor_probability, target - candidate_probability
        ),
    }


def segment_metrics(
    anchor: PredictionSet,
    candidate: PredictionSet,
    *,
    minimum_rows: int = 5_000,
) -> pd.DataFrame:
    if isinstance(minimum_rows, bool) or not isinstance(minimum_rows, int):
        raise ValueError("minimum_rows must be a positive integer")
    if minimum_rows <= 0:
        raise ValueError("minimum_rows must be a positive integer")
    paired = align_pair(anchor, candidate)
    rows: list[dict[str, object]] = []
    for column in SEGMENTS:
        if column not in paired:
            continue
        for level, group in paired.groupby(column, dropna=False, sort=True):
            target = group["target"].to_numpy("float64")
            anchor_probability = group["anchor_probability"].to_numpy("float64")
            candidate_probability = group["candidate_probability"].to_numpy("float64")
            anchor_brier = _brier(target, anchor_probability)
            candidate_brier = _brier(target, candidate_probability)
            rows.append(
                {
                    "anchor_model_id": anchor.model_id,
                    "candidate_model_id": candidate.model_id,
                    "fold": anchor.fold,
                    "segment": column,
                    "level": "__MISSING__" if pd.isna(level) else str(level),
                    "rows": len(group),
                    "anchor_brier": anchor_brier,
                    "candidate_brier": candidate_brier,
                    "regression": candidate_brier - anchor_brier,
                    "eligible": len(group) >= minimum_rows,
                }
            )
    return pd.DataFrame(rows)


def block_bootstrap_interval(
    losses: pd.DataFrame, *, repeats: int = 2_000, seed: int = 3407
) -> dict[str, object]:
    if set(losses) != {"block", "loss_delta"}:
        raise ValueError("bootstrap columns differ")
    if repeats != 2_000 or seed != 3407:
        raise ValueError("bootstrap contract differs")
    if losses.empty or losses["block"].isna().any():
        raise ValueError("bootstrap rows are invalid")
    values = pd.to_numeric(losses["loss_delta"], errors="coerce").to_numpy("float64")
    if not np.isfinite(values).all():
        raise ValueError("bootstrap rows are invalid")
    grouped = tuple(
        group["loss_delta"].to_numpy("float64")
        for _, group in losses.assign(loss_delta=values).groupby("block", sort=True)
    )
    if len(grouped) < 6:
        return {"status": "insufficient_blocks", "block_count": len(grouped)}
    block_sums = np.asarray([group.sum(dtype=np.float64) for group in grouped])
    block_counts = np.asarray([len(group) for group in grouped], dtype="int64")
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(grouped), size=(repeats, len(grouped)))
    samples = block_sums[selected].sum(axis=1) / block_counts[selected].sum(axis=1)
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return {
        "status": "completed",
        "block_count": len(grouped),
        "lower": float(lower),
        "upper": float(upper),
    }
