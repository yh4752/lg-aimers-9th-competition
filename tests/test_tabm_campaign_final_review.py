from __future__ import annotations

import pandas as pd
from pathlib import Path
import sys
import json

import pytest

from experiments.tabm_campaign.dependency_probe import run_python_probe
from experiments.tabm_campaign.final_review import (
    FinalReviewError,
    final_epoch_count,
    run_final_review,
    synthetic_scale_frame,
    validate_frozen_manifest,
)


def test_final_epoch_count_is_sealed_at_three() -> None:
    assert final_epoch_count() == 3


def test_scale_frame_reuses_only_features_and_assigns_unique_ids() -> None:
    sample = pd.DataFrame({"row_id": ["a", "b"], "x": [1.0, 2.0]})
    scale = synthetic_scale_frame(sample, row_count=7)
    assert scale["row_id"].tolist() == [f"synthetic_{i:06d}" for i in range(7)]
    assert scale["row_id"].is_unique
    assert scale["x"].tolist() == [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0]


def test_clean_python_probe_records_output_hash(tmp_path: Path) -> None:
    result = run_python_probe(
        Path(sys.executable),
        "print('five-row-probe-ok')",
        timeout_seconds=10,
    )
    assert result["status"] == "passed"
    assert result["return_code"] == 0
    assert len(result["stdout_sha256"]) == 64


def test_review_rejects_wrong_fit_scope(tmp_path: Path) -> None:
    root = tmp_path / "frozen"
    root.mkdir()
    (root / "inference_manifest.json").write_text(
        json.dumps(
            {
                "fit_scope": "test_aware",
                "epochs": 3,
                "seeds": [3407],
                "scheduler": "constant",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FinalReviewError, match="fit scope differs"):
        validate_frozen_manifest(root)


def test_review_rejects_malformed_member_without_attribute_error(tmp_path: Path) -> None:
    root = tmp_path / "frozen"
    root.mkdir()
    (root / "inference_manifest.json").write_text(
        json.dumps(
            {
                "fit_scope": "official_train_2019_2024_only",
                "epochs": 3,
                "seeds": [3407],
                "scheduler": "constant",
                "members": [None],
                "files": {"weights.pt": "0" * 64},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FinalReviewError, match="member differs"):
        validate_frozen_manifest(root)


def test_legacy_runner_cannot_claim_version_d_review() -> None:
    with pytest.raises(FinalReviewError, match="sealed Colab Version D entrypoint"):
        run_final_review(None, {}, Path("."), Path("."), None, "0" * 64, 0.0)
