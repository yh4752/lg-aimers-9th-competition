"""Closed registry for reviewed submission adapters."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

from .tabm_candidate import CANDIDATE_ID
from .tree_expert_e2_candidate import TREE_E2_ADAPTER_ID


class AdapterRegistryError(ValueError):
    """Raised when an adapter is not explicitly reviewed and registered."""


class SubmissionAdapter(Protocol):
    adapter_id: str

    def state_digest(self) -> str: ...

    def predict_batch(self, frame: "object") -> object: ...


AdapterFactory = Callable[[Path, Mapping[str, object]], SubmissionAdapter]

def _load_version_d(
    model_dir: Path, metadata: Mapping[str, object]
) -> SubmissionAdapter:
    del metadata
    from .tabm_version_d_script import load_frozen_predictor

    return load_frozen_predictor(model_dir)


def _load_tree_e2(
    model_dir: Path, metadata: Mapping[str, object]
) -> SubmissionAdapter:
    from .tree_expert_e2_script import load_frozen_predictor

    return load_frozen_predictor(model_dir, metadata=metadata)


ADAPTER_FACTORIES: Mapping[str, AdapterFactory] = MappingProxyType(
    {
        CANDIDATE_ID: _load_version_d,
        TREE_E2_ADAPTER_ID: _load_tree_e2,
    }
)


def resolve_adapter_factory(
    adapter_id: str,
    *,
    registry: Mapping[str, AdapterFactory] | None = None,
) -> AdapterFactory:
    selected = ADAPTER_FACTORIES if registry is None else registry
    if not isinstance(adapter_id, str) or adapter_id not in selected:
        raise AdapterRegistryError(f"adapter is not registered: {adapter_id!r}")
    factory = selected[adapter_id]
    if not callable(factory):
        raise AdapterRegistryError("registered adapter factory is not callable")
    return factory
