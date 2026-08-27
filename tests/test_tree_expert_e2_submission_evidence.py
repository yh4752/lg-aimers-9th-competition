from __future__ import annotations

from datetime import datetime
from pathlib import Path
import runpy
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest


class _Predictor:
    def state_digest(self) -> str:
        return "b" * 64

    def encode(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        numeric = np.asarray(
            [int(row_id.removeprefix("r")) for row_id in frame["row_id"]],
            dtype="float32",
        ).reshape(-1, 1)
        categorical = frame[["row_id"]].astype(str).to_numpy(dtype="<U32")
        return numeric, categorical

    def predict_batch(self, frame: pd.DataFrame, *, batch_size: int = 4096) -> np.ndarray:
        del batch_size
        return np.asarray(
            [0.4 + int(row_id.removeprefix("r")) * 0.01 for row_id in frame["row_id"]],
            dtype="float64",
        )


def _candidate(tmp_path: Path, monkeypatch):
    helper = runpy.run_path(
        str(Path(__file__).with_name("test_tree_expert_e2_submission_candidate.py"))
    )
    from submission import tree_expert_e2_candidate as module

    handoff = tmp_path / "handoff.zip"
    monkeypatch.setattr(module, "TREE_E2_HANDOFF_SHA256", helper["_handoff"](handoff))
    return module.import_tree_expert_e2_candidate(handoff, tmp_path / "candidate")


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    test = pd.DataFrame({"row_id": [f"r{i}" for i in range(5)], "value": range(5)})
    sample = pd.DataFrame(
        {"row_id": ["r2", "r0", "r4", "r1", "r3"], "control_success": 0.5}
    )
    return test, sample


def test_builds_standard_acceptance_from_exact_e2_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    from submission.tree_expert_e2_existing_evidence import build_tree_e2_acceptance

    candidate = _candidate(tmp_path, monkeypatch)
    test, sample = _frames()
    result = build_tree_e2_acceptance(
        project_root=Path.cwd(),
        candidate=candidate,
        test_frame=test,
        sample_frame=sample,
        runtime_bytes=b"def main():\n    return 0\n",
        output_dir=tmp_path / "evidence",
        load_predictor=_Predictor,
        python_probe={
            "status": "passed",
            "python": "3.11.15",
            "catboost": "1.2.10",
            "pandas": "2.0.3",
            "numpy": "1.26.4",
        },
        package_bytes=50_000_000,
        extracted_bytes=150_000_000,
        policy_path=Path("competition_rules/policy.json"),
        policy_review_path=Path("reports/rules/2026-08-27-final-policy-review.json"),
        package_time=datetime(2026, 8, 27, 15, 0, tzinfo=ZoneInfo("Asia/Seoul")),
    )

    assert result.acceptance["status"] == "passed"
    assert all(result.acceptance["gates"].values())
    assert result.benchmark["inference_seconds"] == pytest.approx(64.939)
    assert result.benchmark["peak_ram_bytes"] == 3_308_498_944
    assert result.audit_manifest.is_file()


def test_failed_full_scale_audit_blocks_acceptance_without_output(
    tmp_path: Path, monkeypatch
) -> None:
    from submission import tree_expert_e2_existing_evidence as module

    candidate = _candidate(tmp_path, monkeypatch)
    object.__setattr__(
        candidate,
        "inference_audit",
        {**candidate.inference_audit, "maximum_absolute_difference": 0.01},
    )
    test, sample = _frames()
    output = tmp_path / "evidence"

    with pytest.raises(module.TreeE2EvidenceError, match="inference audit"):
        module.build_tree_e2_acceptance(
            project_root=Path.cwd(),
            candidate=candidate,
            test_frame=test,
            sample_frame=sample,
            runtime_bytes=b"def main():\n    return 0\n",
            output_dir=output,
            load_predictor=_Predictor,
            python_probe={
                "status": "passed",
                "python": "3.11.15",
                "catboost": "1.2.10",
                "pandas": "2.0.3",
                "numpy": "1.26.4",
            },
            package_bytes=50_000_000,
            extracted_bytes=150_000_000,
            policy_path=Path("competition_rules/policy.json"),
            policy_review_path=Path("reports/rules/2026-08-27-final-policy-review.json"),
            package_time=datetime(2026, 8, 27, 15, 0, tzinfo=ZoneInfo("Asia/Seoul")),
        )
    assert not output.exists()
