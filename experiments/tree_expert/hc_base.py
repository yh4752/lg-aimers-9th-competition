from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from experiments.independent_dl.training import TrainRequest
from experiments.independent_dl.preprocessing import PreprocessingSpec
from experiments.tabm_campaign.cache import materialize_fixed_cache

from .e2_contracts import E2Contract, E2Job, load_e2_contract
from .e2_training import run_e2_fold_job
from .inputs import PREDICTION_COLUMNS, VerifiedOfficialData


class HCBaseError(ValueError):
    """Raised when the frozen earliest E2 source evidence differs."""


SOURCE_FOLD = (2020, 2021)
_SOURCE_ID = (
    "hc_source__tabm_p2_piecewise_linear_bce_plateau"
    "__tr2020__va2021__s3407"
)


@dataclass(frozen=True)
class SourceBaselineResult:
    status: str
    ready_for_residual: bool
    predictions_path: Path | None
    checkpoint: Path | None
    best_epoch: int | None
    completed_epochs: int
    brier: float | None
    failure: str | None


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _source_sha256() -> str:
    root = Path(__file__).resolve().parents[2]
    names = (
        "experiments/tree_expert/hc_base.py",
        "experiments/tree_expert/e2_baseline.py",
        "experiments/independent_dl/features.py",
        "experiments/independent_dl/preprocessing.py",
        "experiments/independent_dl/training.py",
        "experiments/independent_dl/models/tabm.py",
        "experiments/tabm_campaign/cache.py",
    )
    digest = sha256()
    for name in names:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def make_source_request(
    cache: object,
    contract: E2Contract,
    *,
    checkpoint_binding: Mapping[str, str],
) -> TrainRequest:
    tabm = contract.tabm
    return TrainRequest(
        candidate_id=_SOURCE_ID,
        family="tabm",
        seed=3407,
        epochs=int(tabm["max_epochs"]),
        min_epochs=int(tabm["min_epochs"]),
        model_config={
            "architecture": "tabm",
            "k": int(tabm["k"]),
            "width": int(tabm["width"]),
            "blocks": int(tabm["blocks"]),
            "dropout": float(tabm["dropout"]),
            "num_embedding": str(tabm["num_embedding"]),
        },
        training_config={
            "optimizer": "adamw",
            "scheduler": str(tabm["scheduler"]),
            "learning_rate": float(tabm["learning_rate"]),
            "weight_decay": 0.0001,
            "effective_batch_size": int(tabm["effective_batch_size"]),
            "micro_batch_size": int(tabm["micro_batch_size"]),
            "amp": True,
            "patience": int(tabm["patience"]),
        },
        train=cache.train,
        valid=cache.valid,
        checkpoint_binding=dict(checkpoint_binding),
        model_metadata=cache.model_metadata,
    )


def source_e2_jobs() -> tuple[E2Job, ...]:
    return tuple(
        E2Job(
            job_id=f"hc__source_e2__tr2020__va2021__s{seed}",
            candidate_id="c1_anchor_residual",
            train_end_year=2020,
            valid_year=2021,
            seed=seed,
            objective="residual",
            use_trackman=False,
        )
        for seed in (3407, 42, 2026)
    )


def _known(values: pd.Series, category_maps: Mapping[str, Mapping[object, object]], column: str):
    levels = {str(value) for value in category_maps.get(column, {})}
    normalized = values.astype("string").fillna("__MISSING__").astype(str)
    return np.where(normalized.isin(levels), "known", "oov")


def _baseline_predictions(
    valid_rows: pd.DataFrame, cache: object, probability: np.ndarray
) -> pd.DataFrame:
    if cache.valid.row_id.astype(str).tolist() != valid_rows["row_id"].astype(str).tolist():
        raise HCBaseError("source cache row alignment differs")
    target = pd.to_numeric(valid_rows["control_success"], errors="coerce").to_numpy()
    if not np.array_equal(target, np.asarray(cache.valid.y)):
        raise HCBaseError("source cache target differs")
    probability = np.asarray(probability, dtype="float64")
    if probability.shape != target.shape or not np.isfinite(probability).all():
        raise HCBaseError("source probability shape or values differ")
    if np.any((probability < 0) | (probability > 1)):
        raise HCBaseError("source probability is outside [0, 1]")
    category_maps = cache.state.category_maps
    return pd.DataFrame(
        {
            "row_id": valid_rows["row_id"].astype(str).to_numpy(),
            "target": target.astype("int64"),
            "probability": probability,
            "game_type": valid_rows["game_type"].astype(str).to_numpy(),
            "game_month": valid_rows["game_month"].to_numpy(),
            "pitcher_id_known": _known(valid_rows["pitcher_id"], category_maps, "pitcher_id"),
            "batter_id_known": _known(valid_rows["batter_id"], category_maps, "batter_id"),
        }
    ).loc[:, PREDICTION_COLUMNS]


def run_source_baseline(
    *,
    data: VerifiedOfficialData,
    output_dir: Path,
    cache_root: Path,
    absolute_deadline: float,
    contract: E2Contract | None = None,
    cache_builder: Callable[..., object] = materialize_fixed_cache,
    trainer: Callable[..., object] | None = None,
    adapter_factory: Callable[[], object] | None = None,
) -> SourceBaselineResult:
    if time.time() >= absolute_deadline:
        return SourceBaselineResult(
            "inconclusive", False, None, None, None, 0, None, "deadline_reached"
        )
    active = load_e2_contract() if contract is None else contract
    rows = pd.read_csv(data.train)
    history = pd.read_csv(data.history)
    seasons = pd.to_numeric(rows["season"], errors="raise").astype("int64")
    fit_rows = rows.loc[seasons.le(2020)].reset_index(drop=True)
    valid_rows = rows.loc[seasons.eq(2021)].reset_index(drop=True)
    if fit_rows.empty or valid_rows.empty:
        raise HCBaseError("source train or validation rows are empty")
    cache = cache_builder(
        Path(cache_root),
        train=fit_rows,
        valid=valid_rows,
        history=history,
        train_end_year=2020,
        valid_year=2021,
        spec=PreprocessingSpec("dl_standard", ("hand_matchup",)),
        sample_ids=tuple(fit_rows["row_id"].astype(str)),
    )
    binding = {
        "config_sha256": sha256(active.source_path.read_bytes()).hexdigest(),
        "cache_sha256": str(cache.identity.digest()),
        "training_source_sha256": _source_sha256(),
    }
    request = make_source_request(cache, active, checkpoint_binding=binding)
    if trainer is None:
        from experiments.independent_dl.training import fit_candidate

        trainer = fit_candidate
    if adapter_factory is None:
        from experiments.independent_dl.models.tabm import TabMAdapter

        adapter_factory = lambda: TabMAdapter(loss_name="bce")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["PREPROCESSING_SESSION_DEADLINE_UNIX"] = str(absolute_deadline)
    trained = trainer(request, adapter_factory(), output)
    predictions = _baseline_predictions(valid_rows, cache, trained.predictions)
    prediction_path = output / "predictions.csv"
    _atomic_bytes(prediction_path, predictions.to_csv(index=False).encode())
    brier = float(
        np.mean(
            np.square(
                predictions["probability"].to_numpy(dtype="float64")
                - predictions["target"].to_numpy(dtype="float64")
            )
        )
    )
    status = "inconclusive" if bool(trained.budget_reached) else "completed"
    result = SourceBaselineResult(
        status,
        status == "completed",
        prediction_path,
        Path(trained.checkpoint),
        int(trained.best_epoch),
        int(trained.completed_epochs),
        brier,
        "budget_reached" if status == "inconclusive" else None,
    )
    _atomic_bytes(
        output / "baseline_result.json",
        _canonical(
            {
                **asdict(result),
                "predictions_path": prediction_path.name,
                "checkpoint": Path(trained.checkpoint).name,
                "checkpoint_binding": binding,
                "hardware": dict(trained.hardware),
            }
        ),
    )
    return result


def run_source_residual_job(
    *,
    job: E2Job,
    data: VerifiedOfficialData,
    baseline: pd.DataFrame,
    output_dir: Path,
    absolute_deadline: float,
    gpu_id: int,
    input_manifest_sha256: str,
    runner: Callable[..., object] = run_e2_fold_job,
) -> object:
    if (
        (job.train_end_year, job.valid_year) != SOURCE_FOLD
        or job.candidate_id != "c1_anchor_residual"
        or job.seed not in {3407, 42, 2026}
        or job.objective != "residual"
        or job.use_trackman
    ):
        raise HCBaseError("source residual job differs")
    return runner(
        job=job,
        data=data,
        baseline=baseline,
        output_dir=output_dir,
        absolute_deadline=absolute_deadline,
        gpu_id=gpu_id,
        input_manifest_sha256=input_manifest_sha256,
    )


def _load_prediction(path: object, seed: int) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        raise HCBaseError(f"cannot read source seed {seed} predictions") from error
    if tuple(frame.columns) != PREDICTION_COLUMNS or frame.empty:
        raise HCBaseError(f"source seed {seed} prediction schema differs")
    if frame["row_id"].isna().any() or not frame["row_id"].is_unique:
        raise HCBaseError(f"source seed {seed} row identity differs")
    probability = pd.to_numeric(frame["probability"], errors="coerce").to_numpy(
        dtype="float64"
    )
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise HCBaseError(f"source seed {seed} probabilities differ")
    return frame


def ensemble_source_predictions(paths: Mapping[int, object]) -> pd.DataFrame:
    if set(paths) != {3407, 42, 2026}:
        raise HCBaseError("source seed set differs")
    frames = {seed: _load_prediction(paths[seed], seed) for seed in (3407, 42, 2026)}
    reference = frames[3407]
    for seed, frame in frames.items():
        if (
            frame["row_id"].astype(str).tolist()
            != reference["row_id"].astype(str).tolist()
            or not np.array_equal(
                pd.to_numeric(frame["target"], errors="coerce").to_numpy(),
                pd.to_numeric(reference["target"], errors="coerce").to_numpy(),
            )
        ):
            raise HCBaseError(f"source seed {seed} row alignment differs")
    output = reference.copy(deep=True)
    output["probability"] = np.mean(
        np.column_stack(
            [frames[seed]["probability"].to_numpy(dtype="float64") for seed in (3407, 42, 2026)]
        ),
        axis=1,
    )
    return output.loc[:, PREDICTION_COLUMNS]
