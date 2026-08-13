"""Closed registry for reviewed submission adapters."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Protocol


class AdapterRegistryError(ValueError):
    """Raised when an adapter is not explicitly reviewed and registered."""


class SubmissionAdapter(Protocol):
    adapter_id: str

    def state_digest(self) -> str: ...

    def predict_batch(self, frame: "object") -> object: ...


AdapterFactory = Callable[[Path, Mapping[str, object]], SubmissionAdapter]

# Real candidate adapters are added only in separately reviewed candidate work.
ADAPTER_FACTORIES: Mapping[str, AdapterFactory] = MappingProxyType({})


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
