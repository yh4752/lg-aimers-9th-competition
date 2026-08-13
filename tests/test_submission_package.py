from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import zipfile
from zoneinfo import ZoneInfo

import pytest

from competition_rules.contract import load_policy, policy_digest
from competition_rules.evidence_gate import AuditIdentity
from submission.contract import PackageRequest
from submission.audit import SubmissionAuditError, audit_package_request
from submission.package import build_submission_package


ROOT = Path(__file__).resolve().parents[1]


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def _fixture_script() -> bytes:
    return b'''from pathlib import Path
import csv
data = Path("data")
out = Path("output")
out.mkdir(exist_ok=True)
with (data / "test.csv").open(newline="", encoding="utf-8") as handle:
    test = list(csv.DictReader(handle))
with (data / "sample_submission.csv").open(newline="", encoding="utf-8") as handle:
    sample = list(csv.DictReader(handle))
values = {row["row_id"]: f"{1 / (1 + 2.718281828 ** -float(row['x'])):.8f}" for row in test}
with (out / "submission.csv").open("x", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle, lineterminator="\\n")
    writer.writerow(["row_id", "control_success"])
    writer.writerows((row["row_id"], values[row["row_id"]]) for row in sample)
'''


def _valid_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PackageRequest:
    project = tmp_path / "project"
    (project / "competition_rules").mkdir(parents=True)
    policy_path = project / "competition_rules/policy.json"
    policy_path.write_bytes((ROOT / "competition_rules/policy.json").read_bytes())
    policy = load_policy(policy_path, project_root=project)
    review = {
        "schema_version": 1,
        "policy_version": policy["policy_version"],
        "policy_sha256": policy_digest(policy),
        "reviewed_at": "2026-08-13T18:00:00+09:00",
        "sources": policy["official_sources"],
        "verdict": "unchanged",
    }
    review_path = project / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    model_dir = project / "model"
    model_dir.mkdir()
    model = model_dir / "weights.bin"
    model.write_bytes(b"tiny-model")
    requirements = project / "requirements.txt"
    requirements.write_text("numpy==1.26.4\npandas==2.2.3\n", encoding="utf-8")

    identity = AuditIdentity(
        policy_version=policy["policy_version"], candidate_id="fixture",
        data_sha256="1" * 64, code_sha256="2" * 64, config_sha256="3" * 64,
        preprocessing_sha256="4" * 64, model_sha256=_sha(model.read_bytes()),
        adapter_sha256="6" * 64, runtime_sha256="7" * 64,
    )
    evidence = project / "evidence"
    evidence.mkdir()
    chunk = {
        "schema_version": 1, "chunk_index": 0, "identity": asdict(identity),
        "row_count": 2, "row_id_sha256": "8" * 64,
        "prediction_sha256": "9" * 64, "variant_count": 9, "status": "passed",
    }
    canonical = lambda value: (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    chunk["chunk_sha256"] = _sha(canonical(chunk))
    (evidence / "chunk-000000.json").write_bytes(canonical(chunk))
    manifest = {
        "schema_version": 1, "status": "passed", "identity": asdict(identity),
        "row_count": 2, "chunk_size": 2, "chunk_count": 1,
        "chunks": [{"path": "chunk-000000.json", "sha256": chunk["chunk_sha256"]}],
    }
    manifest_path = evidence / "full_audit_manifest.json"
    manifest_path.write_bytes(canonical(manifest))
    gates = {
        "temporal_validation": True, "performance": True, "provenance": True,
        "row_independence": True, "evaluator_runtime": True,
        "pretrained_license": True, "current_rules": True,
    }
    acceptance = {
        "schema_version": 1, "status": "passed", "candidate_id": "fixture",
        "policy_version": policy["policy_version"], "adapter_id": "fixture_linear_v1",
        "identity": asdict(identity), "gates": gates,
    }
    acceptance_path = project / "acceptance.json"
    acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")
    benchmark = {
        "schema_version": 1, "status": "passed", "candidate_id": "fixture",
        "adapter_id": "fixture_linear_v1", "identity": asdict(identity),
        "install_seconds": 1.0, "inference_seconds": 2.0,
        "peak_ram_bytes": 1000, "peak_vram_bytes": 1000,
        "extracted_bytes": 1000, "package_bytes": 1000,
    }
    benchmark_path = project / "benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")

    monkeypatch.setattr("submission.audit.resolve_adapter_factory", lambda adapter_id: object())
    monkeypatch.setattr("submission.package.render_script", lambda **kwargs: _fixture_script())
    return PackageRequest(
        project_root=project, policy_path=policy_path, policy_review_path=review_path,
        acceptance_path=acceptance_path, full_audit_manifest_path=manifest_path,
        runtime_benchmark_path=benchmark_path, model_dir=model_dir,
        requirements_path=requirements, adapter_id="fixture_linear_v1",
        archive_path=project / "out/submission.zip", receipt_path=project / "out/receipt.json",
        package_time=datetime(2026, 8, 13, 20, tzinfo=ZoneInfo("Asia/Seoul")),
    )


def test_valid_fixture_package_has_exact_top_level_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _valid_request(tmp_path, monkeypatch)
    result = build_submission_package(request)
    with zipfile.ZipFile(result.archive_path) as archive:
        names = archive.namelist()
    assert names == ["script.py", "requirements.txt", "model/weights.bin"]
    assert result.receipt_path.parent == result.archive_path.parent
    assert json.loads(result.receipt_path.read_text())["status"] == "packaged"


@pytest.mark.parametrize("row_count", [5, 7])
def test_rendered_fixture_obeys_official_input_and_output_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, row_count: int
) -> None:
    request = _valid_request(tmp_path, monkeypatch)
    result = build_submission_package(request)
    sandbox = tmp_path / f"sandbox-{row_count}"
    with zipfile.ZipFile(result.archive_path) as archive:
        archive.extractall(sandbox)
    data = sandbox / "data"
    data.mkdir()
    ids = [f"id-{index}" for index in range(row_count)]
    (data / "test.csv").write_text(
        "row_id,x\n" + "".join(f"{row_id},{index / 10}\n" for index, row_id in enumerate(ids)),
        encoding="utf-8",
    )
    ordered = ids[::-1]
    (data / "sample_submission.csv").write_text(
        "row_id,control_success\n" + "".join(f"{row_id},0\n" for row_id in ordered),
        encoding="utf-8",
    )
    subprocess.run([sys.executable, "script.py"], cwd=sandbox, check=True)
    output = (sandbox / "output/submission.csv").read_text(encoding="utf-8").splitlines()
    assert output[0] == "row_id,control_success"
    assert [line.split(",", 1)[0] for line in output[1:]] == ordered


def test_package_is_not_created_when_policy_review_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _valid_request(tmp_path, monkeypatch)
    review = json.loads(request.policy_review_path.read_text())
    review["reviewed_at"] = "2026-08-12T20:00:00+09:00"
    request.policy_review_path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(SubmissionAuditError, match="same KST date"):
        build_submission_package(request)
    assert not request.archive_path.exists()


@pytest.mark.parametrize(
    "target,key,value,message",
    [
        ("acceptance", "status", "rejected", "acceptance"),
        ("acceptance_gate", "performance", False, "gate"),
        ("benchmark", "inference_seconds", 481.0, "runtime safety"),
        ("benchmark", "install_seconds", 601.0, "install"),
        ("benchmark", "extracted_bytes", 32_000_000_001, "extracted"),
        ("benchmark", "package_bytes", 10_000_000_001, "package"),
    ],
)
def test_failed_gate_or_limit_never_creates_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    target: str, key: str, value: object, message: str,
) -> None:
    request = _valid_request(tmp_path, monkeypatch)
    path = request.acceptance_path if target.startswith("acceptance") else request.runtime_benchmark_path
    payload = json.loads(path.read_text())
    if target == "acceptance_gate":
        payload["gates"][key] = value
    else:
        payload[key] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SubmissionAuditError, match=message):
        build_submission_package(request)
    assert not request.archive_path.exists()


def test_live_hash_full_audit_unknown_adapter_and_existing_output_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _valid_request(tmp_path, monkeypatch)
    (request.model_dir / "weights.bin").write_bytes(b"changed")
    with pytest.raises(SubmissionAuditError, match="model SHA"):
        audit_package_request(request)

    request = _valid_request(tmp_path / "unknown", monkeypatch)
    monkeypatch.setattr(
        "submission.audit.resolve_adapter_factory",
        lambda adapter_id: (_ for _ in ()).throw(ValueError("unknown")),
    )
    with pytest.raises(SubmissionAuditError, match="adapter"):
        audit_package_request(request)

    request = _valid_request(tmp_path / "existing", monkeypatch)
    request.archive_path.parent.mkdir(parents=True)
    request.archive_path.write_bytes(b"existing")
    with pytest.raises(SubmissionAuditError, match="already exists"):
        build_submission_package(request)


def test_model_symlink_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _valid_request(tmp_path, monkeypatch)
    target = request.model_dir / "weights.bin"
    target.unlink()
    target.symlink_to(request.requirements_path)
    with pytest.raises(SubmissionAuditError, match="symlink"):
        build_submission_package(request)
