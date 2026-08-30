from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from experiments.tree_privileged.features import CandidateFeatureBatch
from experiments.tree_privileged.inference import PrivilegedPredictor, StudentRuntime, audit_independence


class E2:
    def predict(self, rows): return np.clip(.45 + rows["x"].to_numpy() * .001, 0, 1)


class Model:
    def __init__(self, scale): self.scale = scale
    def predict(self, frame): return frame["x"].to_numpy() * self.scale


def transformer(rows, state):
    x = rows["x"].to_numpy(dtype="float64")
    return CandidateFeatureBatch(pd.DataFrame({"x": x}), np.full(len(rows), .5),
                                 rows["row_id"].to_numpy(), None, None, None)


def _predictor() -> PrivilegedPredictor:
    runtime = StudentRuntime("PD15", object(), (Model(.001), Model(.002), Model(.003)), .6, .8)
    return PrivilegedPredictor(e2_predictor=E2(), students=(runtime,), transformer=transformer)


def _rows(count=37):
    return pd.DataFrame({"row_id": [f"r{i}" for i in range(count)], "game_type": ["R", "F"] * (count // 2) + (["R"] if count % 2 else []),
                         "x": np.arange(count, dtype="float64")})


def test_inference_is_identical_singleton_shuffle_reverse_and_batches() -> None:
    predictor = _predictor(); rows = _rows(); expected = predictor.predict(rows)
    audit = audit_independence(predictor, rows)
    assert audit.status == "passed"
    assert audit.maximum_absolute_difference <= 1e-6
    assert_allclose(np.asarray([predictor.predict(rows.iloc[[i]])[0] for i in range(len(rows))]), expected, atol=1e-6, rtol=0)


def test_inference_rejects_teacher_columns() -> None:
    rows = _rows(2); rows["rel_speed"] = 145.0
    try:
        _predictor().predict(rows)
    except ValueError as error:
        assert "forbidden" in str(error)
    else:
        raise AssertionError("teacher feature crossed inference boundary")
