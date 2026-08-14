from __future__ import annotations

import ast
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from competition_rules.code_gate import inspect_inference_source
from submission.adapters import AdapterRegistryError, resolve_adapter_factory
from submission.tabm_candidate import (
    CANDIDATE_ID,
    ImportedTabMCandidate,
    render_validation_script,
)
from submission import tabm_version_d_script as runtime


def _candidate(tmp_path: Path) -> ImportedTabMCandidate:
    model = tmp_path / "candidate/model"
    model.mkdir(parents=True)
    values = {
        "inference_manifest.json": b"{}\n",
        "numeric_embedding_0.json": b"{}\n",
        "preprocessing_state.json": b"{}\n",
        "tabm_member_0_seed_3407.pt": b"weights",
    }
    hashes = {}
    combined = sha256()
    for name, value in sorted(values.items()):
        (model / name).write_bytes(value)
        digest = sha256(value).hexdigest()
        hashes[name] = digest
        combined.update(name.encode("utf-8") + b"\0" + bytes.fromhex(digest))
    return ImportedTabMCandidate(
        candidate_id=CANDIDATE_ID,
        root=model.parent,
        model_dir=model,
        delivery_sha256="1" * 64,
        review_bundle_sha256="2" * 64,
        model_sha256=combined.hexdigest(),
        member_sha256=hashes,
    )


def test_rendered_script_is_deterministic_parseable_and_unregistered(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path)
    first = render_validation_script(candidate)

    assert first == render_validation_script(candidate)
    ast.parse(first.decode("utf-8"))
    assert candidate.model_sha256.encode() in first
    with pytest.raises(AdapterRegistryError, match="not registered"):
        resolve_adapter_factory(candidate.candidate_id)


def test_rendered_script_passes_inference_source_gate(tmp_path: Path) -> None:
    source = tmp_path / "script.py"
    source.write_bytes(render_validation_script(_candidate(tmp_path)))

    report = inspect_inference_source([source], project_root=tmp_path)

    assert report["status"] == "passed"
    assert report["file_count"] == 1


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    test = pd.DataFrame(
        {
            "row_id": ["r2", "r1", "r3"],
            "season": [2025, 2025, 2025],
            "pitcher_hand": ["1", "2", None],
            "batter_hand": ["2", "1", "1"],
        }
    )
    sample = pd.DataFrame(
        {"row_id": ["r1", "r2", "r3"], "control_success": [0.0, 0.0, 0.0]}
    )
    return test, sample


class _FixturePredictor:
    def state_digest(self) -> str:
        return "a" * 64

    def predict_batch(self, frame: pd.DataFrame, *, batch_size: int = 2048) -> np.ndarray:
        del batch_size
        return frame["row_id"].map({"r1": 0.1, "r2": 0.2, "r3": 0.3}).to_numpy()


def test_main_uses_exact_paths_and_sample_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    model = tmp_path / "model"
    data.mkdir()
    model.mkdir()
    test, sample = _frames()
    test.to_csv(data / "test.csv", index=False)
    sample.to_csv(data / "sample_submission.csv", index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime, "EMBEDDED_METADATA", {"candidate_id": CANDIDATE_ID})
    monkeypatch.setattr(runtime, "load_frozen_predictor", lambda path: _FixturePredictor())

    assert runtime.main() == 0

    result = pd.read_csv(tmp_path / "output/submission.csv", dtype={"row_id": "string"})
    assert result.columns.tolist() == ["row_id", "control_success"]
    assert result["row_id"].tolist() == sample["row_id"].tolist()
    assert result["control_success"].tolist() == [0.1, 0.2, 0.3]


def test_duplicate_or_mismatched_ids_are_rejected(tmp_path: Path) -> None:
    test, sample = _frames()
    duplicate = test.copy()
    duplicate.loc[1, "row_id"] = duplicate.loc[0, "row_id"]
    with pytest.raises(runtime.EvaluatorError, match="unique"):
        runtime.validate_inputs(duplicate, sample)

    wrong = sample.copy()
    wrong.loc[0, "row_id"] = "missing"
    with pytest.raises(runtime.EvaluatorError, match="exactly match"):
        runtime.validate_inputs(test, wrong)
    assert not (tmp_path / "output/submission.csv").exists()


def test_batch_dependent_predictions_fail_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BatchDependent(_FixturePredictor):
        def predict_batch(self, frame: pd.DataFrame, *, batch_size: int = 2048) -> np.ndarray:
            del batch_size
            return np.full(len(frame), len(frame) / 10.0)

    data = tmp_path / "data"
    (tmp_path / "model").mkdir()
    data.mkdir()
    test, sample = _frames()
    test.to_csv(data / "test.csv", index=False)
    sample.to_csv(data / "sample_submission.csv", index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime, "EMBEDDED_METADATA", {"candidate_id": CANDIDATE_ID})
    monkeypatch.setattr(runtime, "load_frozen_predictor", lambda path: BatchDependent())

    with pytest.raises(runtime.EvaluatorError, match="row-independence canary"):
        runtime.main()
    assert not (tmp_path / "output/submission.csv").exists()
