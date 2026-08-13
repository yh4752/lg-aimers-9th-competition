from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


class FrozenInferenceError(ValueError):
    """Raised when frozen prediction state or row-local behavior is invalid."""


class FrozenPredictor(Protocol):
    def state_digest(self) -> str: ...

    def encode(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]: ...

    def predict_batch(self, frame: pd.DataFrame, *, batch_size: int = 2048) -> np.ndarray: ...


@dataclass(frozen=True)
class IndependenceReport:
    row_count: int
    features_exact: bool
    max_abs_probability_delta: float
    state_digest_before: str
    state_digest_after: str
    checked_batch_sizes: tuple[int, ...]


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class FrozenTabMPredictor:
    """Load verified preprocessing state and one to three TabM members."""

    def __init__(self, artifact_root: str | Path, *, device: str = "cuda") -> None:
        from experiments.independent_dl.features import _preprocessed_state_from_payload
        from experiments.independent_dl.models.tabm import TabMAdapter
        from experiments.independent_dl.models.common import import_runtime_module
        from .model_state import build_inference_metadata, load_numeric_embedding_state

        self.root = Path(artifact_root).resolve()
        manifest_path = self.root / "inference_manifest.json"
        self._manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(self._manifest_bytes)
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise FrozenInferenceError("inference manifest has no file hashes")
        for name, expected in files.items():
            path = self.root / str(name)
            if not path.is_file() or _file_sha256(path) != expected:
                raise FrozenInferenceError(f"inference artifact SHA-256 differs: {name}")
        state_path = self.root / str(manifest["preprocessing_state"])
        state_payload = json.loads(state_path.read_text(encoding="utf-8"))
        self._state = _preprocessed_state_from_payload(self.root, state_payload)
        self._torch = import_runtime_module("torch")
        if device == "cuda" and not self._torch.cuda.is_available():
            raise FrozenInferenceError("CUDA is required for the frozen TabM predictor")
        self.device = device
        cardinalities = tuple(len(self._state.category_maps[name]) + 1 for name in self._state.categorical_columns)
        self._models: list[object] = []
        for member in manifest.get("members", []):
            numeric = load_numeric_embedding_state(self.root / str(member["numeric_state"]))
            metadata = build_inference_metadata(numeric, categorical_cardinalities=cardinalities)
            model = TabMAdapter().build(member["model_config"], metadata, device)
            state = self._torch.load(self.root / str(member["weights"]), map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            self._models.append(model)
        if not 1 <= len(self._models) <= 3:
            raise FrozenInferenceError("inference manifest must contain one to three members")

    def state_digest(self) -> str:
        return sha256(self._manifest_bytes).hexdigest()

    def encode(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        from experiments.independent_dl.features import _preprocessed_batch, _preprocessed_source
        from experiments.independent_dl.preprocessing import transform_preprocessor

        original = frame.reset_index(drop=True).copy()
        source = _preprocessed_source(original, view=self._state.view, trackman_result=self._state.trackman_result)
        prepared = transform_preprocessor(source, self._state.preprocessing)
        batch = _preprocessed_batch(original, prepared, self._state)
        return np.asarray(batch.x_num, dtype="float32"), np.asarray(batch.x_cat, dtype="int64")

    def predict_batch(self, frame: pd.DataFrame, *, batch_size: int = 2048) -> np.ndarray:
        if batch_size <= 0:
            raise FrozenInferenceError("batch_size must be positive")
        x_num, x_cat = self.encode(frame)
        outputs: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(frame), batch_size):
                stop = start + batch_size
                num = self._torch.as_tensor(x_num[start:stop], dtype=self._torch.float32, device=self.device)
                cat = self._torch.as_tensor(x_cat[start:stop], dtype=self._torch.long, device=self.device)
                members = [model(num, cat).squeeze(-1).sigmoid().mean(dim=1) for model in self._models]
                probability = self._torch.stack(members, dim=0).mean(dim=0)
                outputs.append(probability.to(dtype=self._torch.float32, device="cpu").numpy())
        result = np.concatenate(outputs).astype("float64", copy=False)
        if result.shape != (len(frame),) or not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
            raise FrozenInferenceError("prediction output is not a row-aligned probability vector")
        return result


def audit_frozen_predictor(
    predictor: FrozenPredictor,
    frame: pd.DataFrame,
    *,
    batch_sizes: tuple[int, ...] = (1, 257, 2048),
    tolerance: float = 1e-6,
) -> IndependenceReport:
    if frame.empty or "row_id" not in frame:
        raise FrozenInferenceError("audit frame must contain row_id")
    ids = frame["row_id"].astype("string")
    if ids.isna().any() or ids.duplicated().any():
        raise FrozenInferenceError("audit row_id must be non-null and unique")
    source = frame.reset_index(drop=True).copy()
    source["row_id"] = ids.astype(str).to_numpy()
    before = predictor.state_digest()
    base_num, base_cat = predictor.encode(source)
    baseline = predictor.predict_batch(source, batch_size=max(batch_sizes))
    base_features = {
        row_id: (base_num[index].copy(), base_cat[index].copy())
        for index, row_id in enumerate(source["row_id"])
    }
    base_probability = dict(zip(source["row_id"], baseline, strict=True))
    maximum = 0.0
    features_exact = True

    variants = [
        source.iloc[::-1].reset_index(drop=True),
        source.sample(frac=1.0, random_state=42).reset_index(drop=True),
    ]
    variants.extend(source.iloc[[index]].reset_index(drop=True) for index in range(len(source)))
    for variant in variants:
        num, cat = predictor.encode(variant)
        probability = predictor.predict_batch(variant, batch_size=max(batch_sizes))
        for index, row_id in enumerate(variant["row_id"].astype(str)):
            expected_num, expected_cat = base_features[row_id]
            features_exact = features_exact and np.array_equal(num[index], expected_num) and np.array_equal(cat[index], expected_cat)
            maximum = max(maximum, abs(float(probability[index]) - float(base_probability[row_id])))
    for size in batch_sizes:
        probability = predictor.predict_batch(source, batch_size=size)
        maximum = max(maximum, float(np.max(np.abs(probability - baseline))))
    after = predictor.state_digest()
    if not features_exact or maximum > tolerance or before != after:
        raise FrozenInferenceError(
            f"row-independence audit failed features_exact={features_exact} max_delta={maximum} state_unchanged={before == after}"
        )
    return IndependenceReport(len(source), features_exact, maximum, before, after, tuple(batch_sizes))
