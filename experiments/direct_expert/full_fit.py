from __future__ import annotations

from hashlib import sha256
from statistics import median
from typing import Mapping, Sequence

from .inputs import canonical_json
from .stacking import StackRecipe, required_experts


class DirectExpertFullFitError(ValueError):
    pass


def full_fit_iterations(best_iterations: Sequence[int], *, maximum: int) -> int:
    if not best_iterations or type(maximum) is not int or maximum <= 0:
        raise DirectExpertFullFitError("full-fit iteration evidence differs")
    if any(type(value) is not int or value <= 0 for value in best_iterations):
        raise DirectExpertFullFitError("best iterations differ")
    return min(maximum, int(median(best_iterations)))


def accepted_token(
    decision: Mapping[str, object],
    recipe: StackRecipe,
    bindings: Mapping[str, str],
    *,
    seeds: tuple[int, ...],
    iterations: Mapping[str, int],
) -> dict[str, object]:
    if decision.get("status") not in {"accepted_stable", "accepted_aggressive"}:
        raise DirectExpertFullFitError("candidate is not accepted")
    dependencies = required_experts(tuple(recipe.weights))
    if set(iterations) != set(dependencies) or seeds != (42, 2026, 3407):
        raise DirectExpertFullFitError("full-fit recipe differs")
    if any(type(value) is not int or not 1 <= value <= 2400 for value in iterations.values()):
        raise DirectExpertFullFitError("full-fit iterations differ")
    model_count = len(dependencies) * len(seeds) + (3 if recipe.e2_weight else 0)
    if model_count > 9:
        raise DirectExpertFullFitError("full-fit model budget exceeded")
    if any(type(value) is not str or len(value) != 64 for value in bindings.values()):
        raise DirectExpertFullFitError("full-fit bindings differ")
    token = {
        "schema_version": 1,
        "status": "accepted",
        "candidate_id": decision.get("candidate_id"),
        "gate": decision.get("status"),
        "recipe": {
            "method": recipe.method,
            "weights": dict(recipe.weights),
            "e2_weight": recipe.e2_weight,
            "recipe_sha256": recipe.recipe_sha256,
        },
        "experts": list(dependencies),
        "seeds": list(seeds),
        "iterations": dict(iterations),
        "model_count": model_count,
        "bindings": dict(sorted(bindings.items())),
    }
    token["token_sha256"] = sha256(canonical_json(token)).hexdigest()
    return token
