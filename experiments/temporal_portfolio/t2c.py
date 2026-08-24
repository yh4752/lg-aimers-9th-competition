"""Fixed T2-C safety candidate and confirmation jobs."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from experiments.independent_dl.models.common import metadata_from_train
from experiments.independent_dl.training import TrainRequest

from .catboost_training import (
    TemporalTrainingJob,
    temporal_audit_sha256,
    temporal_sample_weight_sha256,
    temporal_train_request_sha256,
)
from .contracts import load_contract
from .feature_cache import materialize_fold_cache
from .features import PortfolioFeatureError, PortfolioFeatureSpec
from .identity import TrainingIdentity
from .job_materialization import MaterializedJob, _AUDIT_SEGMENT_COLUMNS, _audit_frame
from .metrics import PortfolioMetricError, align_oof, blend_logit, brier
from .planner import PlannedJob
from .t2c_input import CANDIDATE_ID
from .uncertainty import SEGMENT_COLUMNS, pitcher_block_bootstrap, segment_regressions


class T2CError(ValueError):
    pass


@dataclass(frozen=True)
class T2CJobSpec:
    job_id: str
    seed: int
    valid_year: int


@dataclass(frozen=True)
class T2CDecision:
    candidate_id: str
    status: str
    reason: str
    new_seed_weighted_gains: Mapping[int, float]
    ensemble_weighted_gain: float | None
    ensemble_latest_gain: float | None
    ensemble_bootstrap_lower: float | None
    ensemble_max_segment_regression: float | None


NEW_SEEDS = (42, 2026)
ALL_SEEDS = (3407, 42, 2026)
VALID_YEARS = (2022, 2023, 2024)
MAXIMUM_JOB_COUNT = 6
_STABLE_COLUMNS = (
    "pitcher_id",
    "batter_id",
    "game_type",
    "trackman_available",
    "hand_matchup",
    "pitcher_id_known",
    "batter_id_known",
    "history_count_bucket",
    "runner_state",
    "leverage_bucket",
)
_EPS = np.finfo("float64").eps


def build_t2c_specs() -> tuple[T2CJobSpec, ...]:
    return tuple(
        T2CJobSpec(f"t2c__s1__va{year}__s{seed}", seed, year)
        for year in VALID_YEARS
        for seed in NEW_SEEDS
    )


def materialize_t2c_job(
    spec: T2CJobSpec,
    *,
    data_rows_sha256: str,
    parent_sha256: str,
    train: pd.DataFrame,
    history: pd.DataFrame,
    cache_root: str | Path,
) -> MaterializedJob:
    _validate_spec(spec)
    if not _is_sha256(data_rows_sha256) or not _is_sha256(parent_sha256):
        raise T2CError("T2-C input binding is invalid")
    if type(train) is not pd.DataFrame or train.empty or "season" not in train:
        raise T2CError("official training rows are invalid")
    if type(history) is not pd.DataFrame:
        raise T2CError("official history rows are invalid")

    fit_year = spec.valid_year - 1
    fit = train.loc[train["season"].eq(fit_year)].copy(deep=True)
    context = train.loc[train["season"].lt(spec.valid_year)].copy(deep=True)
    valid = train.loc[train["season"].eq(spec.valid_year)].copy(deep=True)
    if fit.empty or valid.empty or not set(fit["row_id"]).issubset(set(context["row_id"])):
        raise T2CError("T2-C fold rows are incomplete")
    try:
        cache = materialize_fold_cache(
            Path(cache_root) / spec.job_id,
            train=fit,
            valid=valid,
            history=history,
            spec=PortfolioFeatureSpec(("base", "S1"), "dl_standard"),
            valid_year=spec.valid_year,
            feature_fit_rows=context,
        )
    except PortfolioFeatureError as error:
        if "skippable insufficient_mapping" in str(error):
            raise T2CError("insufficient_mapping") from error
        raise

    sample_weight = np.ones(len(fit), dtype="float64")
    model_config = {
        "architecture": "tabm",
        "k": 32,
        "width": 512,
        "blocks": 4,
        "dropout": 0.10,
        "num_embedding": "piecewise_linear",
    }
    training_config = {
        "optimizer": "adamw",
        "scheduler": "plateau",
        "learning_rate": 0.0006,
        "weight_decay": 0.0001,
        "effective_batch_size": 4096,
        "micro_batch_size": 512,
        "amp": True,
        "patience": 3,
    }
    request = TrainRequest(
        candidate_id=spec.job_id,
        family="tabm",
        seed=spec.seed,
        epochs=12,
        min_epochs=3,
        model_config=model_config,
        training_config=training_config,
        train=cache.train,
        valid=cache.valid,
        checkpoint_binding={
            "data_rows_sha256": data_rows_sha256,
            "cache_identity_sha256": cache.identity_sha256,
            "parent_sha256": parent_sha256,
        },
        model_metadata=metadata_from_train(cache.train),
    )
    audit = _audit_frame(valid, fit)
    sample_sha = temporal_sample_weight_sha256(sample_weight)
    audit_sha = temporal_audit_sha256(audit, segment_columns=_AUDIT_SEGMENT_COLUMNS)
    request_sha = temporal_train_request_sha256(request)
    model_binding = {
        "job_id": spec.job_id,
        "candidate_id": spec.job_id,
        "expert": "tabm",
        "family": "tabm",
        "temporal_expert": "recent",
        "feature_bundle": "S1",
        "parent_sha256": parent_sha256,
        "sample_weight_sha256": sample_sha,
        "audit_sha256": audit_sha,
        "train_request_sha256": request_sha,
        "model_config": model_config,
    }
    identity = TrainingIdentity.from_payload(
        {
            "data_rows": data_rows_sha256,
            "train_seasons": [fit_year],
            "valid_year": spec.valid_year,
            "decay": None,
            "features": ["base", "S1"],
            "model": model_binding,
            "loss": "bce",
            "seed": spec.seed,
        }
    )
    training = TemporalTrainingJob(
        job_id=spec.job_id,
        expert="tabm",
        identity=identity,
        sample_weight=sample_weight,
        seed=spec.seed,
        audit_frame=audit,
        segment_columns=_AUDIT_SEGMENT_COLUMNS,
        train_request=request,
    )
    plan = PlannedJob(
        spec.job_id,
        3,
        "tabm",
        "temporal",
        "recent",
        "dl_standard",
        ("base", "S1"),
        MappingProxyType(model_binding),
        MappingProxyType(
            {
                "training_identity_sha256": identity.sha256,
                "cache_identity_sha256": cache.identity_sha256,
            }
        ),
        fit_year,
        spec.valid_year,
        spec.seed,
        2_400,
        "full",
        identity,
    )
    return MaterializedJob(plan, training, cache.root)


def build_gated_s1_oof(
    anchor: pd.DataFrame,
    fixed_multi: pd.DataFrame,
    recent_predictions: tuple[pd.DataFrame, ...],
    *,
    valid_year: int,
) -> pd.DataFrame:
    """Average recent logits, blend fixed multi, then fall back on exact F."""

    if valid_year not in VALID_YEARS:
        raise T2CError("T2-C validation year is not authorized")
    if type(recent_predictions) is not tuple or not 1 <= len(recent_predictions) <= 3:
        raise T2CError("T2-C recent prediction count differs")
    entries = [("baseline", _prediction_frame(anchor, valid_year, "anchor"))]
    entries.append(("multi", _prediction_frame(fixed_multi, valid_year, "multi")))
    entries.extend(
        (f"recent_{index}", _prediction_frame(frame, valid_year, f"recent_{index}"))
        for index, frame in enumerate(recent_predictions)
    )
    try:
        aligned = align_oof(tuple(entries), segment_columns=_STABLE_COLUMNS)
    except (PortfolioMetricError, KeyError) as error:
        raise T2CError("T2-C prediction alignment differs") from error
    if aligned["valid_year"].ne(valid_year).any():
        raise T2CError("T2-C prediction validation year differs")
    recent_columns = [f"recent_{index}" for index in range(len(recent_predictions))]
    recent = np.column_stack(
        [aligned[column].to_numpy(dtype="float64", copy=True) for column in recent_columns]
    )
    clipped = np.clip(recent, _EPS, 1.0 - _EPS)
    mean_logit = np.mean(np.log(clipped) - np.log1p(-clipped), axis=1)
    recent_mean = _expit(mean_logit)
    candidate = blend_logit(recent_mean, aligned["multi"], Decimal("0.50"))
    exact_f = aligned["game_type"].map(lambda value: type(value) is str and value == "F")
    baseline = aligned["baseline"].to_numpy(dtype="float64", copy=True)
    candidate[exact_f.to_numpy()] = baseline[exact_f.to_numpy()]
    oof = aligned.loc[
        :, ("row_id", "valid_year", "pitcher_id", "target", *SEGMENT_COLUMNS)
    ].copy(deep=True)
    oof["baseline"] = baseline
    oof["candidate"] = candidate
    return oof


def evaluate_gated_s1_oof(
    oof: pd.DataFrame,
    *,
    bootstrap_repeats: int = 1_000,
) -> Mapping[str, float | int | str]:
    """Return Brier, paired bootstrap, and preregistered segment diagnostics."""

    baseline_score = brier(oof["target"], oof["baseline"])
    candidate_score = brier(oof["target"], oof["candidate"])
    interval = pitcher_block_bootstrap(
        oof, repeats=bootstrap_repeats, seed=load_contract().bootstrap_seed
    )
    segments = segment_regressions(
        oof, minimum_rows=load_contract().minimum_segment_rows
    )
    maximum = max(
        (max(0.0, -item.brier_gain) for item in segments if item.eligible),
        default=0.0,
    )
    return MappingProxyType(
        {
            "status": "completed",
            "rows": len(oof),
            "baseline_brier": baseline_score,
            "candidate_brier": candidate_score,
            "gain": baseline_score - candidate_score,
            "bootstrap_lower": interval.lower,
            "bootstrap_median": interval.median,
            "bootstrap_upper": interval.upper,
            "max_segment_regression": float(maximum),
        }
    )


def decide_t2c(
    *,
    seed_evidence: Mapping[int, Mapping[int, Mapping[str, object]]],
    ensemble: Mapping[str, object],
) -> T2CDecision:
    """Return promoted, rejected, or budget_inconclusive from sealed evidence."""

    if not isinstance(seed_evidence, Mapping):
        return _decision("rejected", "seed evidence is invalid")
    keys = set(seed_evidence)
    if not keys.issubset(set(NEW_SEEDS)):
        return _decision("rejected", "seed evidence contains an unauthorized seed")
    incomplete = keys != set(NEW_SEEDS)
    weighted: dict[int, float] = {}
    row_counts: dict[int, int] = {}
    for seed in NEW_SEEDS:
        folds = seed_evidence.get(seed)
        if folds is None:
            incomplete = True
            continue
        if not isinstance(folds, Mapping):
            return _decision("rejected", f"seed {seed} evidence is invalid")
        fold_keys = set(folds)
        if not fold_keys.issubset(set(VALID_YEARS)):
            return _decision("rejected", f"seed {seed} evidence has an unauthorized fold")
        if fold_keys != set(VALID_YEARS):
            incomplete = True
        rows: list[int] = []
        gains: list[float] = []
        for year in VALID_YEARS:
            item = folds.get(year)
            if item is None or (
                isinstance(item, Mapping) and item.get("status") in {"pending", "failed"}
            ):
                incomplete = True
                continue
            parsed = _parse_fold(item)
            if parsed is None:
                return _decision("rejected", f"seed {seed} fold {year} evidence is invalid")
            row_count, gain, _ = parsed
            rows.append(row_count)
            gains.append(gain)
        if len(rows) == len(VALID_YEARS):
            weighted[seed] = float(np.average(gains, weights=rows))
            row_counts[seed] = sum(rows)
    if incomplete:
        return _decision(
            "budget_inconclusive",
            "one or more authorized jobs are incomplete",
            weighted=weighted,
        )
    if not isinstance(ensemble, Mapping) or set(ensemble) != {"folds", "combined"}:
        return _decision("rejected", "ensemble evidence is incomplete", weighted=weighted)
    folds = ensemble["folds"]
    combined = ensemble["combined"]
    if not isinstance(folds, Mapping) or set(folds) != set(VALID_YEARS):
        return _decision("rejected", "ensemble fold coverage differs", weighted=weighted)
    ensemble_rows: list[int] = []
    ensemble_gains: list[float] = []
    regressions: list[float] = []
    for year in VALID_YEARS:
        parsed = _parse_fold(folds[year])
        if parsed is None:
            return _decision(
                "rejected", f"ensemble fold {year} evidence is invalid", weighted=weighted
            )
        row_count, gain, regression = parsed
        if any(
            int(seed_evidence[seed][year]["rows"]) != row_count for seed in NEW_SEEDS
        ):
            return _decision(
                "rejected", f"ensemble fold {year} row count differs", weighted=weighted
            )
        ensemble_rows.append(row_count)
        ensemble_gains.append(gain)
        regressions.append(regression)
    parsed_combined = _parse_combined(combined)
    if parsed_combined is None:
        return _decision("rejected", "combined evidence is invalid", weighted=weighted)
    combined_rows, combined_gain, lower, combined_regression = parsed_combined
    ensemble_weighted = float(np.average(ensemble_gains, weights=ensemble_rows))
    if (
        combined_rows != sum(ensemble_rows)
        or not np.isclose(combined_gain, ensemble_weighted, rtol=0.0, atol=1e-12)
    ):
        return _decision("rejected", "combined evidence contradicts folds", weighted=weighted)
    latest = ensemble_gains[-1]
    maximum_regression = max(combined_regression, *regressions)
    gates = (
        all(weighted[seed] > 0.0 for seed in NEW_SEEDS),
        ensemble_weighted >= 0.00005,
        latest >= 0.00003,
        lower > 0.0,
        maximum_regression <= 0.00050,
        all(gain >= -0.00003 for gain in ensemble_gains),
    )
    status = "promoted" if all(gates) else "rejected"
    reason = "all preregistered gates passed" if status == "promoted" else "one or more preregistered gates failed"
    return _decision(
        status,
        reason,
        weighted=weighted,
        ensemble_weighted=ensemble_weighted,
        latest=latest,
        lower=lower,
        regression=maximum_regression,
    )


def _parse_fold(value: object) -> tuple[int, float, float] | None:
    if not isinstance(value, Mapping) or value.get("status") != "completed":
        return None
    rows = value.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, (int, np.integer)) or int(rows) <= 0:
        return None
    try:
        gain = float(value["gain"])
        regression = float(value["max_segment_regression"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite((gain, regression)).all() or regression < 0.0:
        return None
    return int(rows), gain, regression


def _parse_combined(value: object) -> tuple[int, float, float, float] | None:
    parsed = _parse_fold(value)
    if parsed is None or not isinstance(value, Mapping):
        return None
    try:
        lower = float(value["bootstrap_lower"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(lower):
        return None
    rows, gain, regression = parsed
    return rows, gain, lower, regression


def _decision(
    status: str,
    reason: str,
    *,
    weighted: Mapping[int, float] | None = None,
    ensemble_weighted: float | None = None,
    latest: float | None = None,
    lower: float | None = None,
    regression: float | None = None,
) -> T2CDecision:
    return T2CDecision(
        CANDIDATE_ID,
        status,
        reason,
        MappingProxyType(dict(weighted or {})),
        ensemble_weighted,
        latest,
        lower,
        regression,
    )


def _prediction_frame(frame: pd.DataFrame, valid_year: int, label: str) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame or frame.empty or not frame.columns.is_unique:
        raise T2CError(f"T2-C {label} prediction is invalid")
    work = frame if "valid_year" in frame else frame.assign(valid_year=valid_year)
    expected = {"row_id", "valid_year", "target", "probability", *_STABLE_COLUMNS}
    if set(work.columns) != expected:
        raise T2CError(f"T2-C {label} prediction schema differs")
    return work.loc[:, ("row_id", "valid_year", "target", "probability", *_STABLE_COLUMNS)]


def _validate_spec(spec: T2CJobSpec) -> None:
    if type(spec) is not T2CJobSpec or spec.seed not in NEW_SEEDS:
        raise T2CError("T2-C job is not authorized")
    if spec.valid_year not in VALID_YEARS:
        raise T2CError("T2-C job is not authorized")
    if spec.job_id != f"t2c__s1__va{spec.valid_year}__s{spec.seed}":
        raise T2CError("T2-C job is not authorized")


def _expit(value: np.ndarray) -> np.ndarray:
    output = np.empty_like(value, dtype="float64")
    positive = value >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
