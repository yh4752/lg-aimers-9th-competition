from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from experiments.independent_dl.models.common import ModelMetadata, quantile_bin_edges


class NumericEmbeddingStateError(ValueError):
    """Raised when frozen numerical-embedding state is malformed."""


@dataclass(frozen=True)
class NumericEmbeddingState:
    mode: str
    n_features: int
    embedding_dim: int
    piecewise_bin_edges: tuple[tuple[float, ...], ...] | None = None
    periodic_n_frequencies: int | None = None
    periodic_frequency_init_scale: float | None = None


def fit_numeric_embedding_state(mode: str, train_x_num: np.ndarray) -> NumericEmbeddingState:
    values = np.asarray(train_x_num, dtype="float32")
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise NumericEmbeddingStateError("training numeric matrix must be finite and two-dimensional")
    if mode == "piecewise_linear":
        edges = tuple(tuple(float(value) for value in edge) for edge in quantile_bin_edges(values))
        return NumericEmbeddingState(mode, values.shape[1], 32, piecewise_bin_edges=edges)
    if mode == "periodic":
        return NumericEmbeddingState(
            mode,
            values.shape[1],
            32,
            periodic_n_frequencies=48,
            periodic_frequency_init_scale=0.01,
        )
    raise NumericEmbeddingStateError(f"unsupported numerical embedding: {mode}")


def _payload(state: NumericEmbeddingState) -> dict[str, object]:
    value = asdict(state)
    if state.piecewise_bin_edges is not None:
        value["piecewise_bin_edges"] = [list(edge) for edge in state.piecewise_bin_edges]
    return value


def save_numeric_embedding_state(state: NumericEmbeddingState, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_payload(state), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_numeric_embedding_state(path: str | Path) -> NumericEmbeddingState:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        edges = raw.get("piecewise_bin_edges")
        state = NumericEmbeddingState(
            mode=str(raw["mode"]),
            n_features=int(raw["n_features"]),
            embedding_dim=int(raw["embedding_dim"]),
            piecewise_bin_edges=(
                None
                if edges is None
                else tuple(tuple(float(value) for value in edge) for edge in edges)
            ),
            periodic_n_frequencies=(
                None if raw.get("periodic_n_frequencies") is None else int(raw["periodic_n_frequencies"])
            ),
            periodic_frequency_init_scale=(
                None
                if raw.get("periodic_frequency_init_scale") is None
                else float(raw["periodic_frequency_init_scale"])
            ),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise NumericEmbeddingStateError(f"invalid numerical embedding state: {exc}") from exc
    if state.mode == "piecewise_linear":
        if state.piecewise_bin_edges is None or len(state.piecewise_bin_edges) != state.n_features:
            raise NumericEmbeddingStateError("piecewise state has invalid bin edges")
    elif state.mode == "periodic":
        if state.periodic_n_frequencies != 48 or state.periodic_frequency_init_scale != 0.01:
            raise NumericEmbeddingStateError("periodic state differs from the sealed configuration")
    else:
        raise NumericEmbeddingStateError(f"unsupported numerical embedding: {state.mode}")
    return state


def build_inference_metadata(
    state: NumericEmbeddingState,
    *,
    categorical_cardinalities: tuple[int, ...] = (),
) -> ModelMetadata:
    edges = (
        None
        if state.piecewise_bin_edges is None
        else tuple(np.asarray(edge, dtype="float32") for edge in state.piecewise_bin_edges)
    )
    return ModelMetadata(
        n_num_features=state.n_features,
        categorical_cardinalities=tuple(categorical_cardinalities),
        train_x_num=None,
        piecewise_bin_edges=edges,
    )
