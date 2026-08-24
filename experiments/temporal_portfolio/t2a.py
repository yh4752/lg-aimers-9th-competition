"""Sealed T2-A feature-ablation jobs and phase transition rules."""
from __future__ import annotations

from dataclasses import dataclass
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
from .feature_cache import materialize_fold_cache
from .features import PortfolioFeatureError, PortfolioFeatureSpec
from .identity import TrainingIdentity
from .job_materialization import (
    MaterializedJob,
    _AUDIT_SEGMENT_COLUMNS,
    _audit_frame,
)
from .metrics import align_oof, blend_logit, brier
from .planner import PlannedJob
from .uncertainty import (
    SEGMENT_COLUMNS,
    pitcher_block_bootstrap,
    segment_regressions,
)


class T2AError(ValueError):
    """Raised when T2-A authorization, evidence, or materialization differs."""


@dataclass(frozen=True)
class T2AJobSpec:
    job_id: str
    expert: str
    bundle: str
    valid_year: int
    seed: int


_BUNDLES = ("S1", "P0", "P1", "P2", "P3", "B1", "M1")
_FAMILY = {
    "S1": "seasonal",
    "P0": "pitcher_trackman",
    "P1": "pitcher_trackman",
    "P2": "pitcher_trackman",
    "P3": "pitcher_trackman",
    "B1": "batter_trackman",
    "M1": "matchup",
}
_MIN_GAIN = 0.00003
_MAX_SEGMENT_REGRESSION = 0.00050


def build_phase_r_specs() -> tuple[T2AJobSpec, ...]:
    return tuple(
        T2AJobSpec(
            f"t2a__r__{bundle.casefold()}__va2024__s3407",
            "recent",
            bundle,
            2024,
            3407,
        )
        for bundle in _BUNDLES
    )


def build_phase_m_specs(bundles: tuple[str, ...]) -> tuple[T2AJobSpec, ...]:
    if (
        type(bundles) is not tuple
        or not 1 <= len(bundles) <= 3
        or len(set(bundles)) != len(bundles)
        or any(bundle not in _BUNDLES for bundle in bundles)
    ):
        raise T2AError("phase M bundles are invalid")
    return tuple(
        T2AJobSpec(
            f"t2a__m__{bundle.casefold()}__va2024__s3407",
            "multi",
            bundle,
            2024,
            3407,
        )
        for bundle in bundles
    )


def select_phase_m_bundles(
    evidence: Mapping[str, Mapping[str, object]],
) -> tuple[str, ...]:
    if not isinstance(evidence, Mapping) or set(evidence) != set(_BUNDLES):
        raise T2AError("phase R evidence must cover all seven bundles")
    eligible: list[tuple[str, float]] = []
    for bundle in _BUNDLES:
        item = evidence[bundle]
        if not isinstance(item, Mapping):
            raise T2AError(f"phase R evidence is invalid: {bundle}")
        if item.get("status") != "completed":
            continue
        try:
            gain = float(item["gain"])
            lower = float(item["bootstrap_lower"])
            regression = float(item["max_segment_regression"])
        except (KeyError, TypeError, ValueError) as error:
            raise T2AError(f"phase R evidence is invalid: {bundle}") from error
        if not np.isfinite((gain, lower, regression)).all():
            raise T2AError(f"phase R evidence is non-finite: {bundle}")
        if gain >= _MIN_GAIN and lower > 0 and regression <= _MAX_SEGMENT_REGRESSION:
            eligible.append((bundle, gain))
    eligible.sort(key=lambda item: (-item[1], _BUNDLES.index(item[0])))
    selected = [bundle for bundle, _ in eligible[:2]]
    represented = {_FAMILY[bundle] for bundle in selected}
    wildcard = next(
        (
            bundle
            for bundle, _ in eligible[2:]
            if _FAMILY[bundle] not in represented
        ),
        None,
    )
    if wildcard is not None:
        selected.append(wildcard)
    elif len(selected) < 3 and len(eligible) > len(selected):
        selected.append(eligible[len(selected)][0])
    return tuple(selected)


def evaluate_feature_candidate(
    anchor: pd.DataFrame,
    fixed_multi: pd.DataFrame,
    feature_prediction: pd.DataFrame,
    *,
    bootstrap_repeats: int = 1_000,
) -> dict[str, float | str]:
    """Score one recent-feature expert under the fixed T1 logit blend."""

    stable = (
        "pitcher_id",
        "batter_id",
        "game_type",
        "trackman_available",
        "hand_matchup",
        "history_count_bucket",
        "runner_state",
        "leverage_bucket",
    )
    feature = feature_prediction.assign(valid_year=2024)
    multi = (
        fixed_multi
        if "valid_year" in fixed_multi.columns
        else fixed_multi.assign(valid_year=2024)
    )
    aligned = align_oof(
        (
            ("baseline", anchor.loc[:, ("row_id", "valid_year", "target", "probability", *stable)]),
            ("multi", multi.loc[:, ("row_id", "valid_year", "target", "probability", *stable)]),
            ("feature", feature.loc[:, ("row_id", "valid_year", "target", "probability", *stable)]),
        ),
        segment_columns=stable,
    )
    positions = {_row_token(value): index for index, value in enumerate(anchor["row_id"])}
    order = [positions[_row_token(value)] for value in aligned["row_id"]]
    for column in ("pitcher_id_known", "batter_id_known"):
        aligned[column] = anchor.iloc[order][column].to_numpy(copy=True)
    candidate = blend_logit(aligned["feature"], aligned["multi"], load_contract().recent_weights[0])
    oof = aligned.loc[
        :, ("row_id", "valid_year", "pitcher_id", "target", *SEGMENT_COLUMNS)
    ].copy(deep=True)
    oof["baseline"] = aligned["baseline"].to_numpy(dtype="float64", copy=True)
    oof["candidate"] = candidate
    gain = brier(oof["target"], oof["baseline"]) - brier(oof["target"], oof["candidate"])
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
        "gain": float(gain),
        "bootstrap_lower": interval.lower,
        "bootstrap_median": interval.median,
        "bootstrap_upper": interval.upper,
        "max_segment_regression": float(maximum),
    }


def materialize_t2a_job(
    spec: T2AJobSpec,
    *,
    data_rows_sha256: str,
    t1_decision_sha256: str,
    train: pd.DataFrame,
    history: pd.DataFrame,
    cache_root: str | Path,
) -> MaterializedJob:
    _validate_spec(spec)
    if not _is_sha256(data_rows_sha256) or not _is_sha256(t1_decision_sha256):
        raise T2AError("T2-A input binding is invalid")
    if type(train) is not pd.DataFrame or train.empty or "season" not in train:
        raise T2AError("official training rows are invalid")
    if type(history) is not pd.DataFrame:
        raise T2AError("official history rows are invalid")

    years = (2023,) if spec.expert == "recent" else (2020, 2021, 2022, 2023)
    fit = train.loc[train["season"].isin(years)].copy(deep=True)
    context = train.loc[train["season"].between(2020, 2023)].copy(deep=True)
    valid = train.loc[train["season"].eq(2024)].copy(deep=True)
    if fit.empty or valid.empty or not set(fit["row_id"]).issubset(set(context["row_id"])):
        raise T2AError("T2-A fold rows are incomplete")
    try:
        cache = materialize_fold_cache(
            Path(cache_root) / spec.job_id,
            train=fit,
            valid=valid,
            history=history,
            spec=PortfolioFeatureSpec(("base", spec.bundle), "dl_standard"),
            valid_year=2024,
            feature_fit_rows=context,
        )
    except PortfolioFeatureError as error:
        if "skippable insufficient_mapping" in str(error):
            raise T2AError("insufficient_mapping") from error
        raise

    sample_weight = (
        np.ones(len(fit), dtype="float64")
        if spec.expert == "recent"
        else np.power(0.55, 2023 - fit["season"].to_numpy(dtype="int64")).astype("float64")
    )
    batter_state = cache.state.fitted_sources.get("batter")
    mapping_status = getattr(batter_state, "status", "not_applicable")
    mapping_coverage = getattr(batter_state, "coverage", None)
    mapping_coverage_text = (
        "not_applicable" if mapping_coverage is None else format(float(mapping_coverage), ".12g")
    )
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
            "t1_decision_sha256": t1_decision_sha256,
            "mapping_status": mapping_status,
            "mapping_coverage": mapping_coverage_text,
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
        "temporal_expert": spec.expert,
        "feature_bundle": spec.bundle,
        "t1_decision_sha256": t1_decision_sha256,
        "mapping_status": mapping_status,
        "mapping_coverage": mapping_coverage_text,
        "sample_weight_sha256": sample_sha,
        "audit_sha256": audit_sha,
        "train_request_sha256": request_sha,
        "model_config": model_config,
    }
    identity = TrainingIdentity.from_payload(
        {
            "data_rows": data_rows_sha256,
            "train_seasons": list(years),
            "valid_year": 2024,
            "decay": None if spec.expert == "recent" else "0.55",
            "features": ["base", spec.bundle],
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
        2,
        "tabm",
        "temporal",
        spec.expert,
        "dl_standard",
        ("base", spec.bundle),
        MappingProxyType(model_binding),
        MappingProxyType(
            {
                "training_identity_sha256": identity.sha256,
                "cache_identity_sha256": cache.identity_sha256,
            }
        ),
        2023,
        2024,
        spec.seed,
        max(900, (load_contract().physical_stage_seconds["T2A"] - 600) // 5),
        "full",
        identity,
    )
    return MaterializedJob(plan, training, cache.root)


def _validate_spec(spec: T2AJobSpec) -> None:
    if type(spec) is not T2AJobSpec or spec.bundle not in _BUNDLES:
        raise T2AError("T2-A job is not authorized")
    if spec.expert not in {"recent", "multi"} or spec.valid_year != 2024 or spec.seed != 3407:
        raise T2AError("T2-A job is not authorized")
    marker = "r" if spec.expert == "recent" else "m"
    expected = f"t2a__{marker}__{spec.bundle.casefold()}__va2024__s3407"
    if spec.job_id != expected:
        raise T2AError("T2-A job is not authorized")


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _row_token(value: object) -> tuple[str, object]:
    if isinstance(value, bool):
        raise T2AError("T2-A row_id is invalid")
    if isinstance(value, Integral):
        return "int", int(value)
    if type(value) is str and value:
        return "str", value
    raise T2AError("T2-A row_id is invalid")
