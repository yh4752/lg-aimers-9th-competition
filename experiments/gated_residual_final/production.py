from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from .contracts import load_contract
from .selection import CandidateDecision


class ProductionError(ValueError):
    pass


FitModel = Callable[..., Path]


@dataclass(frozen=True)
class ProductionCandidate:
    candidate_id: str
    models: Mapping[str, Path]
    iterations: Mapping[str, int]
    direct_model_count: int
    deployed_model_count: int


def build_production_candidate(
    *,
    decision: CandidateDecision,
    fit_model: FitModel,
    iterations: Mapping[str, int],
    output_dir: Path,
) -> ProductionCandidate | None:
    if decision.status != "accepted":
        return None
    if set(iterations) != {"D0", "D5"} or any(
        type(value) is not int or not 1 <= value <= load_contract().maximum_iterations
        for value in iterations.values()
    ):
        raise ProductionError("iteration evidence differs")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    models: dict[str, Path] = {}
    for role in ("D0", "D5"):
        for seed in load_contract().full_fit_seeds:
            key = f"{role}_s{seed}"
            target = output / "models" / f"{key}.cbm"
            path = Path(fit_model(role=role, seed=seed, iterations=iterations[role], output=target))
            if path != target or path.is_symlink() or not path.is_file():
                raise ProductionError(f"trained model differs: {key}")
            models[key] = path
    direct_count = len(models)
    deployed_count = direct_count + 3
    if deployed_count > load_contract().maximum_deployed_models:
        raise ProductionError("model budget exceeded")
    return ProductionCandidate(
        candidate_id=decision.candidate_id,
        models=MappingProxyType(models),
        iterations=MappingProxyType(dict(iterations)),
        direct_model_count=direct_count,
        deployed_model_count=deployed_count,
    )
