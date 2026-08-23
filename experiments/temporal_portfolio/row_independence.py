"""Order, batch, and subset invariance checks for frozen inference."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from .inference import InferenceRuleError


@dataclass(frozen=True)
class RowIndependenceReport:
    accepted: bool
    rows: int
    maximum_absolute_difference: float
    prediction_sha256: str
    checks: Mapping[str, float]


def predict_by_row_id(
    predictor: object, rows: pd.DataFrame, *, batch_size: int
) -> dict[str, float]:
    if type(rows) is not pd.DataFrame or "row_id" not in rows:
        raise InferenceRuleError("row-independence rows are invalid")
    values = np.asarray(
        predictor.predict(rows, batch_size=batch_size), dtype="float64"
    )
    ids = rows["row_id"].astype(str).tolist()
    if len(set(ids)) != len(ids) or values.shape != (len(ids),):
        raise InferenceRuleError("row-independence result differs")
    return {row_id: float(value) for row_id, value in zip(ids, values, strict=True)}


def audit_row_independence(
    predictor: object, rows: pd.DataFrame
) -> RowIndependenceReport:
    reference = predict_by_row_id(predictor, rows, batch_size=32)
    variants = {
        "batch_1": predict_by_row_id(predictor, rows, batch_size=1),
        "batch_512": predict_by_row_id(predictor, rows, batch_size=512),
        "shuffled": predict_by_row_id(
            predictor, rows.sample(frac=1, random_state=3407), batch_size=32
        ),
        "subset": predict_by_row_id(predictor, rows.iloc[::2], batch_size=32),
    }
    differences: dict[str, float] = {}
    for name, actual in variants.items():
        expected = {key: reference[key] for key in actual}
        difference = max(abs(expected[key] - actual[key]) for key in actual)
        if difference > 1e-7:
            raise InferenceRuleError("frozen inference is not row-independent")
        differences[name] = difference
    encoded = json.dumps(reference, sort_keys=True, separators=(",", ":")).encode()
    return RowIndependenceReport(
        True,
        len(rows),
        max(differences.values()),
        sha256(encoded).hexdigest(),
        MappingProxyType(differences),
    )
