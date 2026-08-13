from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from competition_rules.evidence_gate import (
    AuditIdentity,
    RulesEvidenceError,
    run_full_independence_audit,
    validate_full_audit,
)


def _frame(n: int = 11) -> pd.DataFrame:
    return pd.DataFrame({"row_id": [f"r-{i}" for i in range(n)], "x": np.arange(n) / n})


def _identity() -> AuditIdentity:
    return AuditIdentity(
        policy_version="dacon-236743-2026-08-13", candidate_id="tiny",
        data_sha256="a" * 64, code_sha256="a" * 64,
        config_sha256="a" * 64, preprocessing_sha256="a" * 64,
        model_sha256="a" * 64, adapter_sha256="a" * 64,
        runtime_sha256="a" * 64,
    )


def _loader():
    return lambda frame: frame["x"].to_numpy(dtype="float64")


def test_interrupted_full_audit_resumes_verified_chunks(tmp_path: Path) -> None:
    first = run_full_independence_audit(
        frame=_frame(), output_dir=tmp_path, identity=_identity(),
        load_predictor=_loader, chunk_size=4, stop_after_chunks=2,
    )
    assert first.status == "incomplete"
    assert first.completed_chunks == 2

    second = run_full_independence_audit(
        frame=_frame(), output_dir=tmp_path, identity=_identity(),
        load_predictor=_loader, chunk_size=4,
    )
    assert second.status == "passed"
    assert second.row_count == 11
    assert second.reused_chunks == 2
    assert validate_full_audit(
        second.manifest_path, expected_identity=_identity()
    )["status"] == "passed"


def test_changed_chunk_or_identity_is_rejected(tmp_path: Path) -> None:
    result = run_full_independence_audit(
        frame=_frame(), output_dir=tmp_path, identity=_identity(),
        load_predictor=_loader, chunk_size=4,
    )
    chunk = sorted(tmp_path.glob("chunk-*.json"))[0]
    payload = json.loads(chunk.read_text())
    payload["prediction_sha256"] = "b" * 64
    chunk.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RulesEvidenceError):
        validate_full_audit(result.manifest_path, expected_identity=_identity())

    other = replace(_identity(), model_sha256="c" * 64)
    with pytest.raises(RulesEvidenceError):
        validate_full_audit(result.manifest_path, expected_identity=other)


def test_reordered_rows_missing_chunk_and_unexpected_chunk_are_rejected(tmp_path: Path) -> None:
    run_full_independence_audit(
        frame=_frame(), output_dir=tmp_path, identity=_identity(),
        load_predictor=_loader, chunk_size=4, stop_after_chunks=1,
    )
    reordered = _frame().iloc[::-1].reset_index(drop=True)
    with pytest.raises(RulesEvidenceError, match="identity differs"):
        run_full_independence_audit(
            frame=reordered, output_dir=tmp_path, identity=_identity(),
            load_predictor=_loader, chunk_size=4,
        )

    complete = tmp_path / "complete"
    result = run_full_independence_audit(
        frame=_frame(), output_dir=complete, identity=_identity(),
        load_predictor=_loader, chunk_size=4,
    )
    (complete / "chunk-000001.json").unlink()
    with pytest.raises(RulesEvidenceError):
        validate_full_audit(result.manifest_path, expected_identity=_identity())

    other = tmp_path / "unexpected"
    other.mkdir()
    (other / "chunk-999999.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RulesEvidenceError, match="unexpected audit chunks"):
        run_full_independence_audit(
            frame=_frame(), output_dir=other, identity=_identity(),
            load_predictor=_loader, chunk_size=4,
        )


def test_partial_temporary_file_is_ignored_and_existing_manifest_is_not_overwritten(
    tmp_path: Path,
) -> None:
    (tmp_path / "chunk-000000.json.partial").write_text("partial", encoding="utf-8")
    result = run_full_independence_audit(
        frame=_frame(3), output_dir=tmp_path, identity=_identity(),
        load_predictor=_loader, chunk_size=4,
    )
    original = result.manifest_path.read_bytes()
    result_again = run_full_independence_audit(
        frame=_frame(3), output_dir=tmp_path, identity=_identity(),
        load_predictor=_loader, chunk_size=4,
    )
    assert result_again.reused_chunks == 1
    assert result.manifest_path.read_bytes() == original


def test_duplicate_ids_and_batch_dependent_predictor_fail_without_manifest(tmp_path: Path) -> None:
    duplicate = _frame(3)
    duplicate.loc[2, "row_id"] = "r-0"
    with pytest.raises(RulesEvidenceError):
        run_full_independence_audit(
            frame=duplicate, output_dir=tmp_path, identity=_identity(),
            load_predictor=_loader, chunk_size=2,
        )
    assert not (tmp_path / "full_audit_manifest.json").exists()

    with pytest.raises(RulesEvidenceError, match="row independence"):
        run_full_independence_audit(
            frame=_frame(5), output_dir=tmp_path / "bad", identity=_identity(),
            load_predictor=lambda: lambda frame: np.full(len(frame), len(frame) / 10),
            chunk_size=3,
        )
