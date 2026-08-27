from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

import pandas as pd
import pytest

from competition_rules.evidence_gate import AuditIdentity
from submission.audit import _artifact_metadata
from submission.contract import PackageRequest
from submission.tree_expert_e2_candidate import TREE_E2_ADAPTER_ID


def test_audit_returns_exact_e2_candidate_metadata(tmp_path: Path) -> None:
    model_dir = tmp_path / "candidate/model"
    files = {
        "frozen_state/feature_state.json": b"state",
        "frozen_state/s1_batter.csv": b"batter",
        "frozen_state/s1_pitcher.csv": b"pitcher",
        "models/catboost_seed_42.cbm": b"42",
        "models/catboost_seed_2026.cbm": b"2026",
        "models/catboost_seed_3407.cbm": b"3407",
    }
    for name, value in files.items():
        path = model_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    member_sha = {name: sha256(value).hexdigest() for name, value in files.items()}
    combined = sha256()
    for name, value in sorted(files.items()):
        combined.update(name.encode() + b"\0" + sha256(value).digest())
    model_sha = combined.hexdigest()
    manifest = {
        "schema_version": 1,
        "artifact_kind": "tree_expert_e2_submission_candidate_v1",
        "candidate_id": "c1_anchor_residual",
        "adapter_id": TREE_E2_ADAPTER_ID,
        "handoff_sha256": "1" * 64,
        "delivery_sha256": "2" * 64,
        "delivery_manifest_sha256": "3" * 64,
        "model_sha256": model_sha,
        "members": member_sha,
        "seeds": [42, 2026, 3407],
        "iterations": {"42": 78, "2026": 86, "3407": 149},
    }
    model_dir.parent.joinpath("candidate_manifest.json").write_text(json.dumps(manifest))
    request = PackageRequest(
        project_root=tmp_path,
        policy_path=tmp_path / "policy",
        policy_review_path=tmp_path / "review",
        acceptance_path=tmp_path / "acceptance",
        full_audit_manifest_path=tmp_path / "audit",
        runtime_benchmark_path=tmp_path / "benchmark",
        model_dir=model_dir,
        requirements_path=tmp_path / "requirements",
        adapter_id=TREE_E2_ADAPTER_ID,
        archive_path=tmp_path / "submit.zip",
        receipt_path=tmp_path / "receipt.json",
        package_time=datetime.now().astimezone(),
    )
    identity = AuditIdentity(
        policy_version="p",
        candidate_id="c1_anchor_residual",
        data_sha256="4" * 64,
        code_sha256="5" * 64,
        config_sha256="6" * 64,
        preprocessing_sha256=member_sha["frozen_state/feature_state.json"],
        model_sha256=model_sha,
        adapter_sha256="7" * 64,
        runtime_sha256="8" * 64,
    )

    metadata = _artifact_metadata(
        request=request,
        root=tmp_path,
        model_dir=model_dir,
        model_files=tuple(sorted(files.items())),
        model_sha256=model_sha,
        identity=identity,
    )

    assert metadata == {
        key: manifest[key]
        for key in manifest
        if key not in {"schema_version", "artifact_kind"}
    }


def test_official_sample_loader_requires_exact_five_rows(tmp_path: Path) -> None:
    from tools.build_tree_expert_e2_submission import load_official_sample

    test = pd.DataFrame({"row_id": [f"r{i}" for i in range(5)], "value": range(5)})
    sample = pd.DataFrame(
        {"row_id": [f"r{i}" for i in range(5)], "control_success": 0.5}
    )
    test.to_csv(tmp_path / "test.csv", index=False)
    sample.to_csv(tmp_path / "sample_submission.csv", index=False)

    loaded_test, loaded_sample = load_official_sample(tmp_path)
    assert len(loaded_test) == len(loaded_sample) == 5

    test.iloc[:4].to_csv(tmp_path / "test.csv", index=False)
    with pytest.raises(ValueError, match="five rows"):
        load_official_sample(tmp_path)
