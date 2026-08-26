from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import time
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

import numpy as np
import pandas as pd

from experiments.independent_dl.preprocessing import PreprocessingSpec
from experiments.independent_dl.training import TrainRequest
from experiments.tabm_campaign.cache import FixedCache, materialize_fixed_cache

from .e2_contracts import E2Contract, load_e2_contract
from .e2_inputs import VerifiedE2Input, file_sha256
from .inputs import PREDICTION_COLUMNS, VerifiedOfficialData


class E2BaselineError(ValueError):
    """Raised when the temporal TabM baseline cannot be compared safely."""


_F1 = (2021, 2022)
_BASELINE_ID = (
    "e2_baseline__tabm_p2_piecewise_linear_bce_plateau"
    "__tr2021__va2022__s3407"
)


@dataclass(frozen=True)
class BaselinePlanItem:
    fold: tuple[int, int]
    action: str
    source: Path | None


@dataclass(frozen=True)
class F1BaselineResult:
    status: str
    ready_for_structure: bool
    predictions_path: Path | None
    checkpoint: Path | None
    best_epoch: int | None
    completed_epochs: int
    brier: float | None
    failure: str | None


class CacheBuilder(Protocol):
    def __call__(self, cache_root: Path, **kwargs: object) -> FixedCache: ...


class Trainer(Protocol):
    def __call__(self, request: TrainRequest, adapter: object, output_dir: Path) -> object: ...


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
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
    repository = Path(__file__).resolve().parents[2]
    names = (
        "experiments/tree_expert/e2_baseline.py",
        "experiments/independent_dl/features.py",
        "experiments/independent_dl/preprocessing.py",
        "experiments/independent_dl/training.py",
        "experiments/independent_dl/models/tabm.py",
        "experiments/tabm_campaign/cache.py",
    )
    digest = sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((repository / name).read_bytes())
    return digest.hexdigest()


def build_baseline_plan(
    contract: E2Contract,
    evidence: VerifiedE2Input | object,
) -> tuple[BaselinePlanItem, ...]:
    if contract.folds != (_F1, (2022, 2023), (2023, 2024)):
        raise E2BaselineError("E2 baseline folds differ")
    sources = getattr(evidence, "tabm_predictions", None)
    if not isinstance(sources, Mapping) or set(sources) != {
        "2022->2023",
        "2023->2024",
    }:
        raise E2BaselineError("reused TabM baseline evidence differs")
    return (
        BaselinePlanItem(_F1, "train", None),
        BaselinePlanItem((2022, 2023), "reuse", Path(sources["2022->2023"])),
        BaselinePlanItem((2023, 2024), "reuse", Path(sources["2023->2024"])),
    )


def align_baseline(
    baseline: pd.DataFrame,
    official_rows: pd.DataFrame,
    *,
    fold: tuple[int, int],
) -> pd.DataFrame:
    train_end, valid_year = fold
    if valid_year != train_end + 1:
        raise E2BaselineError("baseline fold is invalid")
    if type(official_rows) is not pd.DataFrame or "season" not in official_rows:
        raise E2BaselineError("official fold rows differ")
    seasons = pd.to_numeric(official_rows["season"], errors="coerce")
    valid = official_rows.loc[seasons.eq(valid_year)].copy(deep=True)
    if valid.empty:
        raise E2BaselineError("official validation fold is empty")
    if type(baseline) is not pd.DataFrame or tuple(baseline.columns) != PREDICTION_COLUMNS:
        raise E2BaselineError("baseline columns differ")
    if baseline.empty or baseline["row_id"].isna().any() or not baseline["row_id"].is_unique:
        raise E2BaselineError("baseline row_id values differ")
    if baseline["row_id"].astype(str).tolist() != valid["row_id"].astype(str).tolist():
        raise E2BaselineError("baseline row alignment differs")
    target = pd.to_numeric(valid["control_success"], errors="coerce").to_numpy()
    supplied_target = pd.to_numeric(baseline["target"], errors="coerce").to_numpy()
    if not np.array_equal(target, supplied_target):
        raise E2BaselineError("baseline target differs")
    probability = pd.to_numeric(baseline["probability"], errors="coerce")
    if probability.isna().any() or not probability.between(0, 1).all():
        raise E2BaselineError("baseline probability differs")
    for column in PREDICTION_COLUMNS[3:]:
        if baseline[column].isna().any():
            raise E2BaselineError(f"baseline diagnostic differs: {column}")
    return baseline.copy(deep=True)


def load_reused_baselines(
    evidence: VerifiedE2Input,
    data: VerifiedOfficialData,
) -> Mapping[tuple[int, int], pd.DataFrame]:
    rows = pd.read_csv(data.train)
    loaded: dict[tuple[int, int], pd.DataFrame] = {}
    for fold, label in (
        ((2022, 2023), "2022->2023"),
        ((2023, 2024), "2023->2024"),
    ):
        source = evidence.tabm_predictions.get(label)
        if source is None or Path(source).is_symlink() or not Path(source).is_file():
            raise E2BaselineError(f"reused baseline is absent: {label}")
        loaded[fold] = align_baseline(pd.read_csv(source), rows, fold=fold)
    return MappingProxyType(loaded)


def make_f1_request(
    cache: FixedCache | object,
    contract: E2Contract,
    *,
    checkpoint_binding: Mapping[str, str],
) -> TrainRequest:
    tabm = contract.tabm
    return TrainRequest(
        candidate_id=_BASELINE_ID,
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


def _known(
    values: pd.Series,
    category_maps: Mapping[str, Mapping[object, object]],
    column: str,
) -> np.ndarray:
    levels = {str(value) for value in category_maps.get(column, {})}
    return np.where(values.astype("string").fillna("__MISSING__").astype(str).isin(levels), "known", "oov")


def _prediction_frame(
    valid_rows: pd.DataFrame,
    cache: FixedCache | object,
    probability: np.ndarray,
) -> pd.DataFrame:
    if cache.valid.row_id.astype(str).tolist() != valid_rows["row_id"].astype(str).tolist():
        raise E2BaselineError("F1 cache row alignment differs")
    target = pd.to_numeric(valid_rows["control_success"], errors="coerce").to_numpy()
    cache_target = np.asarray(cache.valid.y)
    if not np.array_equal(target, cache_target):
        raise E2BaselineError("F1 cache target differs")
    probability = np.asarray(probability, dtype="float64")
    if probability.shape != target.shape or not np.isfinite(probability).all():
        raise E2BaselineError("F1 probability shape or values differ")
    if np.any((probability < 0) | (probability > 1)):
        raise E2BaselineError("F1 probability is outside [0, 1]")
    category_maps = cache.state.category_maps
    output = pd.DataFrame(
        {
            "row_id": valid_rows["row_id"].astype(str).to_numpy(),
            "target": target.astype("int64"),
            "probability": probability,
            "game_type": valid_rows["game_type"].astype(str).to_numpy(),
            "game_month": valid_rows["game_month"].to_numpy(),
            "pitcher_id_known": _known(valid_rows["pitcher_id"], category_maps, "pitcher_id"),
            "batter_id_known": _known(valid_rows["batter_id"], category_maps, "batter_id"),
        }
    )
    return output.loc[:, PREDICTION_COLUMNS]


def run_f1_baseline(
    *,
    data: VerifiedOfficialData,
    output_dir: Path,
    cache_root: Path,
    absolute_deadline: float,
    contract: E2Contract | None = None,
    cache_builder: CacheBuilder = materialize_fixed_cache,
    trainer: Trainer | None = None,
    adapter_factory: Callable[[], object] | None = None,
) -> F1BaselineResult:
    if time.time() >= absolute_deadline:
        return F1BaselineResult(
            "inconclusive", False, None, None, None, 0, None, "deadline_reached"
        )
    active = load_e2_contract() if contract is None else contract
    rows = pd.read_csv(data.train)
    history = pd.read_csv(data.history)
    seasons = pd.to_numeric(rows["season"], errors="raise").astype("int64")
    fit_rows = rows.loc[seasons.le(_F1[0])].reset_index(drop=True)
    valid_rows = rows.loc[seasons.eq(_F1[1])].reset_index(drop=True)
    if fit_rows.empty or valid_rows.empty:
        raise E2BaselineError("F1 train or validation rows are empty")
    cache = cache_builder(
        Path(cache_root),
        train=fit_rows,
        valid=valid_rows,
        history=history,
        train_end_year=_F1[0],
        valid_year=_F1[1],
        spec=PreprocessingSpec("dl_standard", ("hand_matchup",)),
        sample_ids=tuple(fit_rows["row_id"].astype(str)),
    )
    binding = {
        "config_sha256": file_sha256(active.source_path),
        "cache_sha256": str(cache.identity.digest()),
        "training_source_sha256": _source_sha256(),
    }
    request = make_f1_request(cache, active, checkpoint_binding=binding)
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
    predictions = _prediction_frame(valid_rows, cache, trained.predictions)
    predictions_path = output / "predictions.csv"
    _atomic_bytes(predictions_path, predictions.to_csv(index=False).encode("utf-8"))
    target = predictions["target"].to_numpy(dtype="float64")
    probability = predictions["probability"].to_numpy(dtype="float64")
    brier = float(np.mean(np.square(probability - target)))
    status = "inconclusive" if bool(trained.budget_reached) else "completed"
    result = F1BaselineResult(
        status=status,
        ready_for_structure=status == "completed",
        predictions_path=predictions_path,
        checkpoint=Path(trained.checkpoint),
        best_epoch=int(trained.best_epoch),
        completed_epochs=int(trained.completed_epochs),
        brier=brier,
        failure="budget_reached" if status == "inconclusive" else None,
    )
    _atomic_bytes(
        output / "baseline_result.json",
        _canonical_json(
            {
                **asdict(result),
                "predictions_path": predictions_path.name,
                "checkpoint": Path(trained.checkpoint).name,
                "checkpoint_binding": binding,
                "hardware": dict(trained.hardware),
            }
        ),
    )
    return result
