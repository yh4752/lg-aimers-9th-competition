from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class DirectExpertComplianceError(ValueError):
    pass


class Predictor(Protocol):
    def predict(self, rows: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class InferenceAuditReport:
    singleton_max_abs: float
    reverse_max_abs: float
    shuffle_max_abs: float
    rebatch_max_abs: float
    companion_max_abs: float
    same_feature_audit_row_max_abs: float
    rows: int


def _predict(runtime: Predictor, rows: pd.DataFrame) -> pd.Series:
    probability = np.asarray(runtime.predict(rows.copy(deep=True)), dtype="float64")
    if probability.shape != (len(rows),) or not np.isfinite(probability).all():
        raise DirectExpertComplianceError("audit prediction differs")
    return pd.Series(probability, index=rows["row_id"].astype(str).to_numpy())


def _difference(left: pd.Series, right: pd.Series) -> float:
    if set(left.index) != set(right.index):
        raise DirectExpertComplianceError("audit row identity differs")
    return float(np.max(np.abs(left.sort_index().to_numpy() - right.sort_index().to_numpy())))


def audit_inference(
    runtime: Predictor,
    rows: pd.DataFrame,
    *,
    batch_sizes: tuple[int, ...] = (1, 7, 64),
) -> InferenceAuditReport:
    if type(rows) is not pd.DataFrame or "row_id" not in rows or rows.empty or not rows["row_id"].is_unique:
        raise DirectExpertComplianceError("audit rows differ")
    baseline = _predict(runtime, rows)
    singleton = pd.concat([_predict(runtime, rows.iloc[[index]]) for index in range(len(rows))])
    reverse = _predict(runtime, rows.iloc[::-1])
    shuffled = _predict(runtime, rows.sample(frac=1, random_state=3407))
    rebatch_differences = []
    for size in batch_sizes:
        parts = [_predict(runtime, rows.iloc[start : start + size]) for start in range(0, len(rows), size)]
        rebatch_differences.append(_difference(baseline, pd.concat(parts)))
    companion_differences = []
    companion = rows.iloc[[-1]]
    for index in range(len(rows) - 1):
        observed = _predict(runtime, pd.concat([rows.iloc[[index]], companion], ignore_index=True))
        companion_differences.append(abs(float(observed.loc[str(rows.iloc[index]["row_id"])]) - float(baseline.loc[str(rows.iloc[index]["row_id"])])))
    copied = rows.iloc[[0]].copy(deep=True)
    original_id = str(copied.iloc[0]["row_id"])
    copied.loc[copied.index[0], "row_id"] = "__AUDIT_COPY__"
    copy_probability = float(_predict(runtime, copied).iloc[0])
    return InferenceAuditReport(
        singleton_max_abs=_difference(baseline, singleton),
        reverse_max_abs=_difference(baseline, reverse),
        shuffle_max_abs=_difference(baseline, shuffled),
        rebatch_max_abs=max(rebatch_differences, default=0.0),
        companion_max_abs=max(companion_differences, default=0.0),
        same_feature_audit_row_max_abs=abs(copy_probability - float(baseline.loc[original_id])),
        rows=len(rows),
    )


_FORBIDDEN_CALLS = {
    "groupby",
    "rolling",
    "expanding",
    "rank",
    "value_counts",
    "shift",
    "diff",
    "mean",
    "transform",
    "system",
    "popen",
    "run",
    "call",
    "check_call",
    "check_output",
}
_FORBIDDEN_IMPORTS = {"requests", "urllib", "http", "socket", "subprocess"}


def audit_source(source: str) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise DirectExpertComplianceError("inference source is not parseable") from error
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".")[0] for alias in node.names}
            if roots & _FORBIDDEN_IMPORTS:
                raise DirectExpertComplianceError("forbidden import in inference source")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in _FORBIDDEN_IMPORTS:
                raise DirectExpertComplianceError("forbidden import in inference source")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in _FORBIDDEN_CALLS:
                raise DirectExpertComplianceError("forbidden evaluation operation in inference source")
