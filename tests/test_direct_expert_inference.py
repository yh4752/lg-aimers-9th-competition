import numpy as np
import pandas as pd
import pytest

from experiments.direct_expert.inference import predict_fixed_recipe
from experiments.direct_expert.stacking import StackRecipe


def test_predictor_routes_only_by_current_row_game_type() -> None:
    rows = pd.DataFrame({"row_id": ["a", "b", "c"], "game_type": ["R", "F", "R"]})
    predictions = {
        "D0": np.asarray([0.4, 0.4, 0.4]),
        "D5": np.asarray([0.7, 0.7, 0.7]),
    }
    recipe = StackRecipe("probability", {"D5": 1.0}, (2022, 2023))
    predicted = predict_fixed_recipe(rows, predictions, recipe)
    changed = rows.copy()
    changed.loc[1:, "game_type"] = ["R", "F"]
    repeated = predict_fixed_recipe(changed, predictions, recipe)
    assert predicted[0] == pytest.approx(repeated[0])
    assert predicted.tolist() == pytest.approx([0.7, 0.4, 0.7])


def test_unknown_game_type_is_rejected() -> None:
    rows = pd.DataFrame({"row_id": ["a"], "game_type": ["X"]})
    with pytest.raises(ValueError, match="game_type"):
        predict_fixed_recipe(rows, {"D0": np.array([0.5])}, StackRecipe("probability", {"D0": 1.0}, (2022, 2023)))
