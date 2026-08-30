from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .stacking import StackRecipe, apply_stack


class DirectExpertInferenceError(ValueError):
    pass


def predict_fixed_recipe(
    rows: pd.DataFrame,
    expert_predictions: Mapping[str, np.ndarray],
    recipe: StackRecipe,
    *,
    e2: np.ndarray | None = None,
) -> np.ndarray:
    if type(rows) is not pd.DataFrame or set(("row_id", "game_type")) - set(rows):
        raise DirectExpertInferenceError("inference rows differ")
    if "control_success" in rows or rows["row_id"].isna().any() or not rows["row_id"].is_unique:
        raise DirectExpertInferenceError("inference row identity differs")
    game_type = rows["game_type"].astype(str).to_numpy()
    if not np.isin(game_type, ("R", "F")).all():
        raise DirectExpertInferenceError("game_type differs")
    arrays = {name: np.asarray(value, dtype="float64") for name, value in expert_predictions.items()}
    required = set(recipe.weights)
    if required.intersection({"D5", "D6"}):
        required.add("D0")
    if set(arrays) != required or any(value.shape != (len(rows),) for value in arrays.values()):
        raise DirectExpertInferenceError("expert prediction set differs")
    routed = {}
    for name in recipe.weights:
        if name == "D5":
            routed[name] = np.where(game_type == "R", arrays[name], arrays["D0"])
        elif name == "D6":
            routed[name] = np.where(game_type == "F", arrays[name], arrays["D0"])
        else:
            routed[name] = arrays[name]
    result = apply_stack(recipe, routed, e2=e2)
    if result.shape != (len(rows),) or not np.isfinite(result).all():
        raise DirectExpertInferenceError("inference output differs")
    return np.clip(result, 1e-6, 1 - 1e-6)
