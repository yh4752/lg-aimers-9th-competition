"""Turn a sealed T1 recipe into the exact arrays consumed by one worker."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

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
from .contracts import JobSpec, build_stage_jobs, load_contract
from .feature_cache import materialize_fold_cache
from .features import PortfolioFeatureSpec
from .identity import TrainingIdentity
from .planner import PlannedJob, _t1_job
from .uncertainty import SEGMENT_COLUMNS


class JobMaterializationError(ValueError):
    pass


_AUDIT_SEGMENT_COLUMNS = tuple(
    column
    for column in SEGMENT_COLUMNS
    if column
    not in {
        "game_type",
        "pitcher_id_known",
        "batter_id_known",
        "trackman_available",
    }
)


@dataclass(frozen=True)
class MaterializedJob:
    plan: PlannedJob
    training: TemporalTrainingJob
    cache_root: Path


def materialize_t1_job(
    spec: JobSpec,
    *,
    data_rows_sha256: str,
    train: pd.DataFrame,
    history: pd.DataFrame,
    cache_root: str | Path,
) -> MaterializedJob:
    contract = load_contract()
    if spec not in build_stage_jobs(contract, "T1"):
        raise JobMaterializationError("job is not authorized by the T1 contract")
    if type(train) is not pd.DataFrame or train.empty or "season" not in train:
        raise JobMaterializationError("official training rows are invalid")
    if type(history) is not pd.DataFrame:
        raise JobMaterializationError("official history rows are invalid")

    fold = spec.fold
    if spec.expert == "recent":
        years = (fold.recent_year,)
    else:
        years = tuple(range(fold.multi_start, fold.multi_end + 1))
    fit = train.loc[train["season"].isin(years)].copy(deep=True)
    valid = train.loc[train["season"].eq(fold.valid_year)].copy(deep=True)
    if fit.empty or valid.empty:
        raise JobMaterializationError("T1 fold rows are incomplete")

    scope = f"{spec.expert}__tr{'-'.join(map(str, years))}__va{fold.valid_year}"
    cache = materialize_fold_cache(
        Path(cache_root) / scope,
        train=fit,
        valid=valid,
        history=history,
        spec=PortfolioFeatureSpec(("base",), "dl_standard"),
        valid_year=fold.valid_year,
    )
    if spec.decay is None:
        sample_weight = np.ones(len(fit), dtype="float64")
    else:
        sample_weight = np.power(
            float(spec.decay), fold.multi_end - fit["season"].to_numpy(dtype="int64")
        ).astype("float64")

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
        "patience": 10,
    }
    request = TrainRequest(
        candidate_id=spec.job_id,
        family="tabm",
        seed=spec.seed,
        epochs=40,
        min_epochs=3,
        model_config=model_config,
        training_config=training_config,
        train=cache.train,
        valid=cache.valid,
        checkpoint_binding={
            "data_rows_sha256": data_rows_sha256,
            "cache_identity_sha256": cache.identity_sha256,
        },
        model_metadata=metadata_from_train(cache.train),
    )
    audit = _audit_frame(valid, fit)
    sample_sha = temporal_sample_weight_sha256(sample_weight)
    audit_sha = temporal_audit_sha256(
        audit, segment_columns=_AUDIT_SEGMENT_COLUMNS
    )
    request_sha = temporal_train_request_sha256(request)
    model_binding = {
        "job_id": spec.job_id,
        "candidate_id": spec.job_id,
        "expert": "tabm",
        "family": "tabm",
        "temporal_expert": spec.expert,
        "sample_weight_sha256": sample_sha,
        "audit_sha256": audit_sha,
        "train_request_sha256": request_sha,
        "model_config": model_config,
    }
    identity = TrainingIdentity.from_payload(
        {
            "data_rows": data_rows_sha256,
            "train_seasons": list(years),
            "valid_year": fold.valid_year,
            "decay": None if spec.decay is None else format(spec.decay, "f"),
            "features": ["base"],
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
    placeholder = _t1_job(spec, data_rows_sha256)
    plan = replace(
        placeholder,
        identity=identity,
        model=MappingProxyType(model_binding),
        training=MappingProxyType(
            {
                "training_identity_sha256": identity.sha256,
                "cache_identity_sha256": cache.identity_sha256,
            }
        ),
    )
    return MaterializedJob(plan, training, cache.root)


def _audit_frame(valid: pd.DataFrame, fit: pd.DataFrame) -> pd.DataFrame:
    required = {
        "row_id",
        "control_success",
        "pitcher_id",
        "batter_id",
        "game_type",
        "pitcher_hand",
        "batter_hand",
    }
    missing = sorted(required.difference(valid.columns))
    if missing:
        raise JobMaterializationError(f"validation audit columns are missing: {missing}")
    pitcher_known = set(fit["pitcher_id"].astype(str))
    batter_known = set(fit["batter_id"].astype(str))
    output = pd.DataFrame(
        {
            "row_id": valid["row_id"].astype(str),
            "target": valid["control_success"].astype("int8"),
            "pitcher_id": valid["pitcher_id"],
            "batter_id": valid["batter_id"],
            "game_type": valid["game_type"].fillna("missing").astype(str),
            "pitcher_id_known": np.where(
                valid["pitcher_id"].astype(str).isin(pitcher_known), "known", "oov"
            ),
            "batter_id_known": np.where(
                valid["batter_id"].astype(str).isin(batter_known), "known", "oov"
            ),
            "trackman_available": _trackman_available(valid),
            "hand_matchup": valid["pitcher_hand"].astype(str)
            + "_"
            + valid["batter_hand"].astype(str),
            "history_count_bucket": _history_bucket(valid),
            "runner_state": valid.get(
                "base_state", pd.Series("unknown", index=valid.index)
            ).fillna("unknown").astype(str),
            "leverage_bucket": _leverage_bucket(valid),
        }
    )
    return output.reset_index(drop=True)


def _trackman_available(frame: pd.DataFrame) -> np.ndarray:
    if "asof_pitcher_pitchmix_n" not in frame:
        return np.full(len(frame), "unavailable", dtype=object)
    count = pd.to_numeric(frame["asof_pitcher_pitchmix_n"], errors="coerce").fillna(0)
    return np.where(count > 0, "available", "unavailable")


def _history_bucket(frame: pd.DataFrame) -> np.ndarray:
    count = pd.to_numeric(
        frame.get("asof_pitcher_n", pd.Series(0, index=frame.index)), errors="coerce"
    ).fillna(0)
    return pd.cut(
        count,
        bins=[-np.inf, 0, 10, 50, np.inf],
        labels=["none", "low", "medium", "high"],
    ).astype(str).to_numpy()


def _leverage_bucket(frame: pd.DataFrame) -> np.ndarray:
    leverage = pd.to_numeric(
        frame.get("li", pd.Series(1.0, index=frame.index)), errors="coerce"
    ).fillna(1.0)
    return pd.cut(
        leverage,
        bins=[-np.inf, 0.7, 1.5, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str).to_numpy()
