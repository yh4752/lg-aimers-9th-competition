from __future__ import annotations

import numpy as np
import pandas as pd


class T3TemporalError(ValueError):
    pass


_DECAYS = {0.35, 0.55, 0.75}


def temporal_training_weights(
    seasons: pd.Series,
    *,
    valid_year: int,
    head: str,
    decay: float | None = None,
) -> np.ndarray:
    if type(seasons) is not pd.Series:
        raise T3TemporalError("training seasons must be a pandas Series")
    if type(valid_year) is not int or type(valid_year) is bool or valid_year < 2000:
        raise T3TemporalError("validation year must be a four-digit integer")
    try:
        numeric = pd.to_numeric(seasons, errors="raise").to_numpy(dtype="float64")
    except (TypeError, ValueError) as error:
        raise T3TemporalError("training seasons must be finite integers") from error
    if not np.isfinite(numeric).all() or np.any(numeric != np.floor(numeric)):
        raise T3TemporalError("training seasons must be finite integers")
    values = numeric.astype("int64")
    if values.size == 0 or np.any(values >= valid_year):
        raise T3TemporalError("training seasons must precede validation")
    age = (valid_year - 1) - values
    if int(age.min()) != 0:
        raise T3TemporalError("immediate previous season is missing")
    if head == "recent" and decay is None:
        result = (age == 0).astype("float64")
    elif head == "multi" and type(decay) is float and decay in _DECAYS:
        result = np.power(float(decay), age).astype("float64")
    else:
        raise T3TemporalError("temporal head or decay differs")
    if not np.any(result > 0) or not np.isfinite(result).all():
        raise T3TemporalError("temporal weights are invalid")
    result.setflags(write=False)
    return result
