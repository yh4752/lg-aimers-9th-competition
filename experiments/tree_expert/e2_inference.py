from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import resource
import sys
import time
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .e2_contracts import E2Contract
from .e2_decisions import blend_probabilities
from .e2_full_fit import AcceptedForFullFit, load_frozen_tree_state
from .features import TreeFeatureBatch, TreeFeatureState, transform_tree_features


class E2InferenceError(ValueError):
    """Raised when the accepted E2 predictor cannot run exactly as reviewed."""


Transformer = Callable[[pd.DataFrame, object], TreeFeatureBatch]


@dataclass(frozen=True)
class InferenceAudit:
    status: str
    row_count: int
    checks: Mapping[str, float]
    maximum_absolute_difference: float
    elapsed_seconds: float
    peak_rss_bytes: int
    peak_gpu_bytes: int
    reason: str


def _validate_token(token: AcceptedForFullFit) -> None:
    if type(token) is not AcceptedForFullFit:
        raise E2InferenceError("accepted full-fit token differs")
    if token.seeds != (42, 2026, 3407) or set(token.iterations) != set(token.seeds):
        raise E2InferenceError("accepted seed identity differs")
    if token.predictor not in {"catboost", "blend"}:
        raise E2InferenceError("accepted predictor differs")


class E2InferenceRuntime:
    def __init__(
        self,
        *,
        token: AcceptedForFullFit,
        state: object,
        catboost_models: Sequence[object],
        transformer: Transformer = transform_tree_features,
        tabm_predictor: object | None = None,
        blend_method: str = "catboost",
        catboost_weight: float = 1.0,
    ) -> None:
        _validate_token(token)
        models = tuple(catboost_models)
        if len(models) != 3 or any(not hasattr(model, "predict") for model in models):
            raise E2InferenceError("exactly three CatBoost models are required")
        if not callable(transformer):
            raise E2InferenceError("tree feature transformer differs")
        if token.predictor == "catboost":
            if tabm_predictor is not None or blend_method != "catboost" or catboost_weight != 1.0:
                raise E2InferenceError("CatBoost-only inference identity differs")
        elif (
            tabm_predictor is None
            or not hasattr(tabm_predictor, "predict_batch")
            or blend_method not in {"probability", "logit"}
            or catboost_weight not in {0.1, 0.2, 0.3}
        ):
            raise E2InferenceError("accepted blend identity differs")
        self.token = token
        self.state = state
        self.catboost_models = models
        self.transformer = transformer
        self.tabm_predictor = tabm_predictor
        self.blend_method = blend_method
        self.catboost_weight = float(catboost_weight)

    def predict(self, rows: pd.DataFrame, *, batch_size: int = 4096) -> np.ndarray:
        if type(rows) is not pd.DataFrame or rows.empty:
            raise E2InferenceError("evaluation rows must be a non-empty pandas DataFrame")
        if "control_success" in rows:
            raise E2InferenceError("evaluation rows contain target")
        if "row_id" not in rows or rows["row_id"].isna().any() or not rows["row_id"].is_unique:
            raise E2InferenceError("row_id values must be present, non-null, and unique")
        if type(batch_size) is not int or batch_size <= 0:
            raise E2InferenceError("batch size differs")

        batch = self.transformer(rows.copy(deep=True), self.state)
        if type(batch) is not TreeFeatureBatch or batch.target is not None:
            raise E2InferenceError("evaluation feature batch differs")
        expected_ids = rows["row_id"].astype(str).to_numpy()
        if not np.array_equal(batch.row_id, expected_ids):
            raise E2InferenceError("transformed row_id order differs")
        anchor = np.asarray(batch.anchor, dtype="float64")
        if anchor.shape != (len(rows),) or not np.isfinite(anchor).all():
            raise E2InferenceError("anchor values differ")

        members: list[np.ndarray] = []
        for model in self.catboost_models:
            residual = np.asarray(model.predict(batch.frame), dtype="float64")
            if residual.shape != anchor.shape or not np.isfinite(residual).all():
                raise E2InferenceError("CatBoost prediction values differ")
            members.append(np.clip(anchor + residual, 1e-5, 1 - 1e-5))
        catboost = np.mean(np.stack(members, axis=0), axis=0)
        if self.token.predictor == "catboost":
            return catboost

        tabm = np.asarray(
            self.tabm_predictor.predict_batch(rows.copy(deep=True), batch_size=batch_size),
            dtype="float64",
        )
        if tabm.shape != catboost.shape or not np.isfinite(tabm).all():
            raise E2InferenceError("TabM prediction values differ")
        return blend_probabilities(
            tabm,
            catboost,
            self.blend_method,
            self.catboost_weight,
        )


def _load_catboost_models(model_dir: Path, token: AcceptedForFullFit) -> tuple[object, ...]:
    try:
        from catboost import CatBoostRegressor
    except ImportError as error:
        raise E2InferenceError("catboost==1.2.10 is required") from error
    models: list[object] = []
    for seed in token.seeds:
        path = Path(model_dir) / f"catboost_seed_{seed}.cbm"
        if path.is_symlink() or not path.is_file():
            raise E2InferenceError(f"CatBoost model is absent: seed={seed}")
        model = CatBoostRegressor()
        model.load_model(str(path))
        models.append(model)
    return tuple(models)


def _load_tabm_predictor(tabm_dir: Path, *, device: str) -> object:
    script = Path(tabm_dir) / "script.py"
    model_dir = Path(tabm_dir) / "model"
    if script.is_symlink() or not script.is_file() or not model_dir.is_dir():
        raise E2InferenceError("frozen TabM runtime differs")
    spec = importlib.util.spec_from_file_location("tree_expert_e2_frozen_tabm", script)
    if spec is None or spec.loader is None:
        raise E2InferenceError("cannot load frozen TabM runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader = getattr(module, "load_frozen_predictor", None)
    if not callable(loader):
        raise E2InferenceError("frozen TabM loader differs")
    return loader(model_dir, device=device)


def load_inference_runtime(
    *,
    token: AcceptedForFullFit,
    frozen_state_dir: Path,
    catboost_model_dir: Path,
    blend_method: str,
    catboost_weight: float,
    tabm_dir: Path | None = None,
    device: str = "cuda",
) -> E2InferenceRuntime:
    state: TreeFeatureState = load_frozen_tree_state(frozen_state_dir)
    models = _load_catboost_models(catboost_model_dir, token)
    tabm = None
    if token.predictor == "blend":
        if tabm_dir is None:
            raise E2InferenceError("accepted blend is missing frozen TabM")
        tabm = _load_tabm_predictor(tabm_dir, device=device)
    return E2InferenceRuntime(
        token=token,
        state=state,
        catboost_models=models,
        tabm_predictor=tabm,
        blend_method=blend_method,
        catboost_weight=catboost_weight,
    )


def _aligned_difference(
    baseline_ids: np.ndarray,
    baseline: np.ndarray,
    ids: Sequence[object],
    values: np.ndarray,
) -> float:
    positions = {str(row_id): index for index, row_id in enumerate(ids)}
    if len(positions) != len(baseline_ids) or set(positions) != set(baseline_ids):
        raise E2InferenceError("audit row identity differs")
    aligned = np.asarray([values[positions[str(row_id)]] for row_id in baseline_ids])
    return float(np.max(np.abs(aligned - baseline)))


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _peak_gpu_bytes() -> int:
    try:
        import torch
    except ImportError:
        return 0
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0


def audit_inference(
    runtime: E2InferenceRuntime,
    rows: pd.DataFrame,
    contract: E2Contract,
) -> InferenceAudit:
    expected_rows = int(contract.runtime["inference_rows"])
    if type(rows) is not pd.DataFrame or len(rows) != expected_rows:
        raise E2InferenceError("audit inference row count differs")
    tolerance = float(contract.runtime["probability_tolerance"])
    started = time.monotonic()
    baseline = runtime.predict(rows, batch_size=4096)
    baseline_ids = rows["row_id"].astype(str).to_numpy()
    checks: dict[str, float] = {}

    reverse_rows = rows.iloc[::-1].reset_index(drop=True)
    checks["reverse"] = _aligned_difference(
        baseline_ids,
        baseline,
        reverse_rows["row_id"],
        runtime.predict(reverse_rows, batch_size=4096),
    )
    shuffled = rows.sample(frac=1.0, random_state=3407).reset_index(drop=True)
    checks["shuffle"] = _aligned_difference(
        baseline_ids,
        baseline,
        shuffled["row_id"],
        runtime.predict(shuffled, batch_size=4096),
    )
    for size in (1, 257, 4096):
        checks[f"batch_{size}"] = float(
            np.max(np.abs(runtime.predict(rows, batch_size=size) - baseline))
        )

    sample_indices = np.unique(np.linspace(0, len(rows) - 1, min(7, len(rows))).astype(int))
    singleton_difference = 0.0
    companion_difference = 0.0
    companion_index = int((sample_indices[-1] + 1) % len(rows))
    for index in sample_indices:
        single = runtime.predict(rows.iloc[[int(index)]].copy(), batch_size=1)[0]
        singleton_difference = max(singleton_difference, abs(float(single) - float(baseline[index])))
        pair_indices = [int(index), companion_index]
        if pair_indices[0] == pair_indices[1]:
            pair_indices[1] = int((companion_index + 1) % len(rows))
        paired = runtime.predict(rows.iloc[pair_indices].copy(), batch_size=2)[0]
        companion_difference = max(companion_difference, abs(float(paired) - float(baseline[index])))
    checks["singleton"] = singleton_difference
    checks["companion"] = companion_difference

    elapsed = time.monotonic() - started
    rss = _peak_rss_bytes()
    gpu = _peak_gpu_bytes()
    maximum = max(checks.values())
    if maximum > tolerance:
        status, reason = "failed", "row_or_batch_invariance_differs"
    elif elapsed > float(contract.runtime["inference_max_seconds"]):
        status, reason = "failed", "inference_time_limit_exceeded"
    elif rss > int(contract.runtime["rss_max_bytes"]):
        status, reason = "failed", "inference_rss_limit_exceeded"
    elif gpu > int(contract.runtime["gpu_max_bytes"]):
        status, reason = "failed", "inference_gpu_limit_exceeded"
    else:
        status, reason = "passed", "inference_audit_passed"
    return InferenceAudit(
        status=status,
        row_count=len(rows),
        checks=MappingProxyType(dict(checks)),
        maximum_absolute_difference=maximum,
        elapsed_seconds=elapsed,
        peak_rss_bytes=rss,
        peak_gpu_bytes=gpu,
        reason=reason,
    )
