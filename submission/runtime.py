"""Generic row-separable evaluator runtime."""

from __future__ import annotations

import csv
from hashlib import sha256
import os
from pathlib import Path
from typing import Mapping

from competition_rules.code_gate import RulesCodeGateError, canonical_probability


class SubmissionRuntimeError(ValueError):
    """Raised before an invalid evaluator output can be published."""


def _digest(adapter: object) -> str:
    function = getattr(adapter, "state_digest", None)
    if not callable(function):
        raise SubmissionRuntimeError("adapter must expose state_digest")
    value = function()
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SubmissionRuntimeError("adapter state digest must be a lowercase SHA-256")
    return value


def _predict(adapter: object, frame: "object") -> list[str]:
    import numpy as np

    function = getattr(adapter, "predict_batch", None)
    if not callable(function):
        raise SubmissionRuntimeError("adapter must expose predict_batch")
    try:
        values = np.asarray(function(frame.copy(deep=True)), dtype="float64")
    except SubmissionRuntimeError:
        raise
    except Exception as error:
        raise SubmissionRuntimeError("adapter prediction failed") from error
    if values.shape != (len(frame),):
        raise SubmissionRuntimeError("adapter predictions must be one-dimensional and row-aligned")
    output: list[str] = []
    for value in values:
        try:
            output.append(canonical_probability(float(value), decimal_places=8))
        except RulesCodeGateError as error:
            message = str(error).replace("probability must be ", "probability must be ")
            raise SubmissionRuntimeError(message) from error
    return output


def _validate_frames(test_frame: "object", sample_submission: "object") -> tuple["object", "object"]:
    import pandas as pd

    if not isinstance(test_frame, pd.DataFrame) or not isinstance(sample_submission, pd.DataFrame):
        raise SubmissionRuntimeError("test and sample submission must be DataFrames")
    if test_frame.empty or "row_id" not in test_frame:
        raise SubmissionRuntimeError("test rows must be non-empty with row_id")
    if sample_submission.columns.tolist() != ["row_id", "control_success"]:
        raise SubmissionRuntimeError("sample submission columns are invalid")
    test = test_frame.copy(deep=True)
    sample = sample_submission.copy(deep=True)
    for label, frame in (("test", test), ("sample", sample)):
        ids = frame["row_id"].astype("string")
        if ids.isna().any() or ids.astype(str).duplicated().any():
            raise SubmissionRuntimeError(f"{label} row_id must be non-null and unique")
        frame["row_id"] = ids.astype(str)
    if set(test["row_id"]) != set(sample["row_id"]) or len(test) != len(sample):
        raise SubmissionRuntimeError("test row IDs must exactly match sample submission")
    return test, sample


def _publish_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    if path.exists():
        raise SubmissionRuntimeError(f"output already exists: {path}")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["row_id", "control_success"])
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        # Exclusive final creation prevents replacement of an existing result.
        with path.open("xb") as output, temporary.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError as error:
        raise SubmissionRuntimeError(f"output already exists: {path}") from error
    except Exception:
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
        raise
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def run_evaluator(
    *,
    test_frame: "object",
    sample_submission: "object",
    adapter: object,
    output_path: str | Path,
    canary_count: int = 8,
    package_seed: str = "dacon-236743-row-canary-v1",
) -> dict[str, object]:
    """Predict all rows once and prove fixed singleton canaries agree."""

    if type(canary_count) is not int or canary_count < 1:
        raise SubmissionRuntimeError("canary_count must be positive")
    test, sample = _validate_frames(test_frame, sample_submission)
    before = _digest(adapter)
    predictions = _predict(adapter, test)
    after_full = _digest(adapter)
    if after_full != before:
        raise SubmissionRuntimeError("adapter state changed during inference")
    by_id = dict(zip(test["row_id"], predictions, strict=True))
    ordered_indices = sorted(
        range(len(test)),
        key=lambda index: sha256(
            (package_seed + "\0" + test.iloc[index]["row_id"]).encode("utf-8")
        ).hexdigest(),
    )[: min(canary_count, len(test))]
    for index in ordered_indices:
        singleton = test.iloc[[index]].reset_index(drop=True)
        candidate = _predict(adapter, singleton)[0]
        row_id = singleton.iloc[0]["row_id"]
        if candidate != by_id[row_id]:
            raise SubmissionRuntimeError("row-independence canary mismatch")
    if _digest(adapter) != before:
        raise SubmissionRuntimeError("adapter state changed during inference")
    rows = [(row_id, by_id[row_id]) for row_id in sample["row_id"]]
    output = Path(output_path)
    _publish_csv(output, rows)
    return {
        "status": "passed",
        "adapter_id": str(getattr(adapter, "adapter_id", "")),
        "row_count": len(test),
        "canary_count": len(ordered_indices),
        "state_sha256": before,
        "output_sha256": sha256(output.read_bytes()).hexdigest(),
    }


def render_script(
    *, adapter_id: str, artifact_metadata: Mapping[str, object]
) -> bytes:
    """Render the reviewed fixed template for a registered adapter."""

    from .adapters import resolve_adapter_factory
    from .tabm_candidate import CANDIDATE_ID, render_bound_script
    from .tree_expert_e2_candidate import (
        TREE_E2_ADAPTER_ID,
        render_bound_script as render_tree_e2_script,
    )

    resolve_adapter_factory(adapter_id)
    if not isinstance(artifact_metadata, Mapping):
        raise SubmissionRuntimeError("artifact metadata must be a mapping")
    if adapter_id == CANDIDATE_ID:
        try:
            return render_bound_script(artifact_metadata)
        except ValueError as error:
            raise SubmissionRuntimeError("TabM artifact metadata is invalid") from error
    if adapter_id == TREE_E2_ADAPTER_ID:
        try:
            return render_tree_e2_script(artifact_metadata)
        except ValueError as error:
            raise SubmissionRuntimeError("Tree E2 artifact metadata is invalid") from error
    raise SubmissionRuntimeError("registered adapter has no reviewed script template")
