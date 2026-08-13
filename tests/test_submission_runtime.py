from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from submission.adapters import AdapterRegistryError, resolve_adapter_factory
from submission.runtime import SubmissionRuntimeError, run_evaluator


class LinearFixtureAdapter:
    adapter_id = "fixture_linear_v1"

    def __init__(self) -> None:
        self.digest = "a" * 64

    def state_digest(self) -> str:
        return self.digest

    def predict_batch(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame["x"].fillna(0.0).to_numpy(dtype="float64")
        return 1.0 / (1.0 + np.exp(-x))


def _frames(n: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = [f"r-{index}" for index in range(n)]
    test = pd.DataFrame({"row_id": ids[::-1], "x": np.linspace(-1.0, 1.0, n)})
    sample = pd.DataFrame({"row_id": ids, "control_success": [0.0] * n})
    return test, sample


def test_runtime_supports_arbitrary_row_count_and_exact_output_order(tmp_path: Path) -> None:
    for count in (1, 3, 19):
        test, sample = _frames(count)
        output = tmp_path / str(count) / "output" / "submission.csv"
        report = run_evaluator(
            test_frame=test,
            sample_submission=sample,
            adapter=LinearFixtureAdapter(),
            output_path=output,
            canary_count=min(5, count),
        )
        assert report["status"] == "passed"
        result = pd.read_csv(output, dtype={"row_id": "string"})
        assert result.columns.tolist() == ["row_id", "control_success"]
        assert result["row_id"].tolist() == sample["row_id"].tolist()
        assert len(result) == count


def test_runtime_accepts_missing_values_and_is_deterministic(tmp_path: Path) -> None:
    test, sample = _frames(4)
    test.loc[1, "x"] = np.nan
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    run_evaluator(test_frame=test, sample_submission=sample, adapter=LinearFixtureAdapter(), output_path=first)
    run_evaluator(test_frame=test, sample_submission=sample, adapter=LinearFixtureAdapter(), output_path=second)
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize(
    "adapter, message",
    [
        (type("BadShape", (LinearFixtureAdapter,), {"predict_batch": lambda self, frame: [0.5]}), "row-aligned"),
        (type("NonFinite", (LinearFixtureAdapter,), {"predict_batch": lambda self, frame: np.full(len(frame), np.nan)}), "finite"),
        (type("OutOfRange", (LinearFixtureAdapter,), {"predict_batch": lambda self, frame: np.full(len(frame), 1.1)}), "inside"),
        (type("BatchDependent", (LinearFixtureAdapter,), {"predict_batch": lambda self, frame: np.full(len(frame), len(frame) / 10)}), "canary"),
    ],
)
def test_runtime_failures_leave_no_output(tmp_path: Path, adapter: type, message: str) -> None:
    test, sample = _frames(4)
    output = tmp_path / "output.csv"
    with pytest.raises(SubmissionRuntimeError, match=message):
        run_evaluator(test_frame=test, sample_submission=sample, adapter=adapter(), output_path=output)
    assert not output.exists()


def test_runtime_rejects_state_mutation_duplicate_ids_and_sample_mismatch(tmp_path: Path) -> None:
    class Mutating(LinearFixtureAdapter):
        def predict_batch(self, frame: pd.DataFrame) -> np.ndarray:
            self.digest = "b" * 64
            return super().predict_batch(frame)

    test, sample = _frames(3)
    with pytest.raises(SubmissionRuntimeError, match="state changed"):
        run_evaluator(test_frame=test, sample_submission=sample, adapter=Mutating(), output_path=tmp_path / "a.csv")

    duplicate = test.copy()
    duplicate.loc[1, "row_id"] = duplicate.loc[0, "row_id"]
    with pytest.raises(SubmissionRuntimeError, match="unique"):
        run_evaluator(test_frame=duplicate, sample_submission=sample, adapter=LinearFixtureAdapter(), output_path=tmp_path / "b.csv")

    wrong = sample.copy()
    wrong.loc[0, "row_id"] = "missing"
    with pytest.raises(SubmissionRuntimeError, match="exactly match"):
        run_evaluator(test_frame=test, sample_submission=wrong, adapter=LinearFixtureAdapter(), output_path=tmp_path / "c.csv")


def test_registry_is_closed_and_unknown_adapter_is_rejected() -> None:
    with pytest.raises(AdapterRegistryError, match="not registered"):
        resolve_adapter_factory("anything")
