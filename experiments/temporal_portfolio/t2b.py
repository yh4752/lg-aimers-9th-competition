"""Sealed T2-B feature stability and combination screening."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from numbers import Integral
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
from .decisions import CandidateDecision, CandidateMetrics, decide_candidate
from .feature_cache import materialize_fold_cache
from .features import PortfolioFeatureError, PortfolioFeatureSpec
from .identity import TrainingIdentity
from .job_materialization import MaterializedJob, _AUDIT_SEGMENT_COLUMNS, _audit_frame
from .metrics import align_oof, blend_logit, brier
from .planner import PlannedJob
from .uncertainty import SEGMENT_COLUMNS, pitcher_block_bootstrap, segment_regressions


class T2BError(ValueError):
    """Raised when a T2-B job or its evidence differs from the sealed design."""


@dataclass(frozen=True)
class T2BJobSpec:
    job_id: str
    phase: str
    bundles: tuple[str, ...]
    valid_year: int
    seed: int = 3407


PHASE_F_BUNDLES = ("S1", "P3", "P2")
PHASE_C_COMBINATIONS = (("S1", "P3"), ("S1", "P2"))
VALID_YEARS = (2022, 2023, 2024)
COMBINATION_MIN_GAIN = 0.00003
COMBINATION_MAX_SEGMENT_REGRESSION = 0.00050
_STABLE_COLUMNS = (
    "pitcher_id",
    "batter_id",
    "game_type",
    "trackman_available",
    "hand_matchup",
    "history_count_bucket",
    "runner_state",
    "leverage_bucket",
)


def build_phase_f_specs() -> tuple[T2BJobSpec, ...]:
    return tuple(
        _spec("F", (bundle,), year)
        for year in (2022, 2023)
        for bundle in PHASE_F_BUNDLES
    )


def build_phase_c_specs() -> tuple[T2BJobSpec, ...]:
    return tuple(_spec("C", bundles, 2024) for bundles in PHASE_C_COMBINATIONS)


def build_phase_h_specs(combination: str) -> tuple[T2BJobSpec, ...]:
    bundles = _parse_combination(combination)
    return tuple(_spec("H", bundles, year) for year in (2022, 2023))


def select_combination(
    evidence: Mapping[str, Mapping[str, object]],
) -> str | None:
    names = tuple("+".join(item) for item in PHASE_C_COMBINATIONS)
    if not isinstance(evidence, Mapping) or set(evidence) != set(names):
        raise T2BError("phase C evidence must cover both combinations")
    eligible: list[tuple[str, float]] = []
    for name in names:
        item = evidence[name]
        if not isinstance(item, Mapping):
            raise T2BError(f"phase C evidence is invalid: {name}")
        if item.get("status") != "completed":
            continue
        try:
            gain = float(item["gain"])
            lower = float(item["bootstrap_lower"])
            regression = float(item["max_segment_regression"])
        except (KeyError, TypeError, ValueError) as error:
            raise T2BError(f"phase C evidence is invalid: {name}") from error
        if not np.isfinite((gain, lower, regression)).all():
            raise T2BError(f"phase C evidence is non-finite: {name}")
        if (
            gain >= COMBINATION_MIN_GAIN
            and lower > 0.0
            and regression <= COMBINATION_MAX_SEGMENT_REGRESSION
        ):
            eligible.append((name, gain))
    eligible.sort(key=lambda item: (-item[1], names.index(item[0])))
    return eligible[0][0] if eligible else None


def evaluate_t2b_fold(
    anchor: pd.DataFrame,
    fixed_multi: pd.DataFrame,
    recent_prediction: pd.DataFrame,
    *,
    valid_year: int,
    bootstrap_repeats: int = 1_000,
) -> dict[str, float | int | str]:
    """Compare one recent-feature expert against the fixed T1 anchor."""

    if valid_year not in VALID_YEARS:
        raise T2BError("T2-B validation year is not authorized")
    frames = []
    for name, frame in (
        ("baseline", anchor),
        ("multi", fixed_multi),
        ("feature", recent_prediction),
    ):
        if type(frame) is not pd.DataFrame:
            raise T2BError(f"T2-B {name} prediction is invalid")
        work = frame if "valid_year" in frame else frame.assign(valid_year=valid_year)
        frames.append(
            (
                name,
                work.loc[
                    :,
                    ("row_id", "valid_year", "target", "probability", *_STABLE_COLUMNS),
                ],
            )
        )
    aligned = align_oof(tuple(frames), segment_columns=_STABLE_COLUMNS)
    if aligned["valid_year"].ne(valid_year).any():
        raise T2BError("T2-B prediction validation year differs")
    positions = {_row_token(value): index for index, value in enumerate(anchor["row_id"])}
    try:
        order = [positions[_row_token(value)] for value in aligned["row_id"]]
    except KeyError as error:
        raise T2BError("T2-B anchor row alignment differs") from error
    for column in ("pitcher_id_known", "batter_id_known"):
        if column not in anchor:
            raise T2BError(f"T2-B anchor lacks {column}")
        aligned[column] = anchor.iloc[order][column].to_numpy(copy=True)
    candidate = blend_logit(
        aligned["feature"], aligned["multi"], load_contract().recent_weights[0]
    )
    oof = aligned.loc[
        :, ("row_id", "valid_year", "pitcher_id", "target", *SEGMENT_COLUMNS)
    ].copy(deep=True)
    oof["baseline"] = aligned["baseline"].to_numpy(dtype="float64", copy=True)
    oof["candidate"] = candidate
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
    return {
        "status": "completed",
        "valid_year": valid_year,
        "rows": len(oof),
        "baseline_brier": baseline_score,
        "candidate_brier": candidate_score,
        "gain": baseline_score - candidate_score,
        "bootstrap_lower": interval.lower,
        "bootstrap_median": interval.median,
        "bootstrap_upper": interval.upper,
        "max_segment_regression": float(maximum),
    }


def decide_t2b_candidates(
    fold_evidence: Mapping[str, Mapping[int, Mapping[str, object]]],
    combined_evidence: Mapping[str, Mapping[str, object]],
) -> tuple[CandidateDecision, ...]:
    """Apply the shared temporal gates to complete three-fold candidates."""

    if not isinstance(fold_evidence, Mapping) or not isinstance(combined_evidence, Mapping):
        raise T2BError("T2-B candidate evidence is invalid")
    if set(fold_evidence) != set(combined_evidence):
        raise T2BError("T2-B combined evidence coverage differs")
    decisions = []
    for candidate_id, folds in fold_evidence.items():
        if not isinstance(folds, Mapping) or set(folds) != set(VALID_YEARS):
            raise T2BError(f"T2-B folds are incomplete: {candidate_id}")
        rows = []
        gains = []
        regressions = []
        for year in VALID_YEARS:
            item = folds[year]
            if not isinstance(item, Mapping) or item.get("status") != "completed":
                raise T2BError(f"T2-B fold is incomplete: {candidate_id}/{year}")
            try:
                row_count = int(item["rows"])
                gain = float(item["gain"])
                regression = float(item["max_segment_regression"])
            except (KeyError, TypeError, ValueError) as error:
                raise T2BError(f"T2-B fold evidence is invalid: {candidate_id}/{year}") from error
            if row_count <= 0 or not np.isfinite((gain, regression)).all():
                raise T2BError(f"T2-B fold evidence is invalid: {candidate_id}/{year}")
            rows.append(row_count)
            gains.append(gain)
            regressions.append(regression)
        combined = combined_evidence[candidate_id]
        try:
            lower = float(combined["bootstrap_lower"])
            combined_regression = float(combined["max_segment_regression"])
        except (KeyError, TypeError, ValueError) as error:
            raise T2BError(f"T2-B combined evidence is invalid: {candidate_id}") from error
        weighted = float(np.average(gains, weights=rows))
        fold_regressions = tuple(Decimal(str(max(0.0, -value))) for value in gains)
        metrics = CandidateMetrics(
            candidate_id=str(candidate_id),
            family=str(candidate_id),
            temporal_gain=Decimal(str(weighted)),
            weighted_gain=Decimal(str(weighted)),
            latest_gain=Decimal(str(gains[-1])),
            bootstrap_lower=Decimal(str(lower)),
            max_segment_regression=Decimal(str(max(combined_regression, *regressions))),
            fold_regressions=fold_regressions,
            improved_fold_count=sum(value > 0.0 for value in gains),
            worst_fold_regression=max(fold_regressions),
            latest_regression=fold_regressions[-1],
        )
        decisions.append(decide_candidate(metrics))
    return tuple(decisions)


def materialize_t2b_job(
    spec: T2BJobSpec,
    *,
    data_rows_sha256: str,
    parent_sha256: str,
    train: pd.DataFrame,
    history: pd.DataFrame,
    cache_root: str | Path,
) -> MaterializedJob:
    _validate_spec(spec)
    if not _is_sha256(data_rows_sha256) or not _is_sha256(parent_sha256):
        raise T2BError("T2-B input binding is invalid")
    if type(train) is not pd.DataFrame or train.empty or "season" not in train:
        raise T2BError("official training rows are invalid")
    if type(history) is not pd.DataFrame:
        raise T2BError("official history rows are invalid")

    fit_year = spec.valid_year - 1
    fit = train.loc[train["season"].eq(fit_year)].copy(deep=True)
    context = train.loc[train["season"].lt(spec.valid_year)].copy(deep=True)
    valid = train.loc[train["season"].eq(spec.valid_year)].copy(deep=True)
    if fit.empty or valid.empty or not set(fit["row_id"]).issubset(set(context["row_id"])):
        raise T2BError("T2-B fold rows are incomplete")
    try:
        cache = materialize_fold_cache(
            Path(cache_root) / spec.job_id,
            train=fit,
            valid=valid,
            history=history,
            spec=PortfolioFeatureSpec(("base", *spec.bundles), "dl_standard"),
            valid_year=spec.valid_year,
            feature_fit_rows=context,
        )
    except PortfolioFeatureError as error:
        if "skippable insufficient_mapping" in str(error):
            raise T2BError("insufficient_mapping") from error
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
    feature_name = "+".join(spec.bundles)
    model_binding = {
        "job_id": spec.job_id,
        "candidate_id": spec.job_id,
        "expert": "tabm",
        "family": "tabm",
        "temporal_expert": "recent",
        "feature_bundle": feature_name,
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
            "features": ["base", *spec.bundles],
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
        ("base", *spec.bundles),
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
        max(900, (load_contract().physical_stage_seconds["T2B"] - 600) // 5),
        "full",
        identity,
    )
    return MaterializedJob(plan, training, cache.root)


def _spec(phase: str, bundles: tuple[str, ...], valid_year: int) -> T2BJobSpec:
    slug = "_".join(bundle.casefold() for bundle in bundles)
    spec = T2BJobSpec(
        f"t2b__{phase.casefold()}__{slug}__va{valid_year}__s3407",
        phase,
        bundles,
        valid_year,
        3407,
    )
    _validate_spec(spec)
    return spec


def _parse_combination(value: str) -> tuple[str, ...]:
    if type(value) is not str:
        raise T2BError("T2-B combination is invalid")
    bundles = tuple(value.split("+"))
    if bundles not in PHASE_C_COMBINATIONS:
        raise T2BError("T2-B combination is invalid")
    return bundles


def _validate_spec(spec: T2BJobSpec) -> None:
    if type(spec) is not T2BJobSpec or spec.seed != 3407:
        raise T2BError("T2-B job is not authorized")
    allowed = {
        *(('F', (bundle,), year) for year in (2022, 2023) for bundle in PHASE_F_BUNDLES),
        *(('C', bundles, 2024) for bundles in PHASE_C_COMBINATIONS),
        *(('H', bundles, year) for year in (2022, 2023) for bundles in PHASE_C_COMBINATIONS),
    }
    if (spec.phase, spec.bundles, spec.valid_year) not in allowed:
        raise T2BError("T2-B job is not authorized")
    slug = "_".join(bundle.casefold() for bundle in spec.bundles)
    expected = f"t2b__{spec.phase.casefold()}__{slug}__va{spec.valid_year}__s3407"
    if spec.job_id != expected:
        raise T2BError("T2-B job is not authorized")


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _row_token(value: object) -> tuple[str, object]:
    if isinstance(value, bool):
        raise T2BError("T2-B row_id is invalid")
    if isinstance(value, Integral):
        return "int", int(value)
    if type(value) is str and value:
        return "str", value
    raise T2BError("T2-B row_id is invalid")
