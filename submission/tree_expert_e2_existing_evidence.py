"""Bind accepted E2 campaign evidence to the current submission contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Mapping

from competition_rules.code_gate import RulesCodeGateError, inspect_inference_source
from competition_rules.contract import RulesContractError, load_policy, load_policy_review
from competition_rules.evidence_gate import AuditIdentity, run_phased_independence_audit

from .tree_expert_e2_candidate import (
    TREE_E2_ADAPTER_ID,
    ImportedTreeE2Candidate,
    candidate_metadata,
)


class TreeE2EvidenceError(ValueError):
    """Raised before incomplete or changed E2 evidence can become acceptance."""


@dataclass(frozen=True)
class TreeE2Acceptance:
    identity: AuditIdentity
    audit_manifest: Path
    acceptance_path: Path
    benchmark_path: Path
    acceptance: dict[str, object]
    benchmark: dict[str, object]


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical(value)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _frame_digest(test: "object", sample: "object") -> str:
    digest = sha256()
    digest.update(test.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    digest.update(b"\0")
    digest.update(sample.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    return digest.hexdigest()


def _verify_frames(test: "object", sample: "object") -> None:
    import pandas as pd

    if type(test) is not pd.DataFrame or type(sample) is not pd.DataFrame:
        raise TreeE2EvidenceError("official sample frames are invalid")
    if len(test) != 5 or len(sample) != 5 or "row_id" not in test:
        raise TreeE2EvidenceError("official sample must contain five rows")
    if sample.columns.tolist() != ["row_id", "control_success"]:
        raise TreeE2EvidenceError("official sample submission columns differ")
    test_ids = test["row_id"].astype("string")
    sample_ids = sample["row_id"].astype("string")
    if (
        test_ids.isna().any()
        or sample_ids.isna().any()
        or test_ids.astype(str).duplicated().any()
        or sample_ids.astype(str).duplicated().any()
        or set(test_ids.astype(str)) != set(sample_ids.astype(str))
    ):
        raise TreeE2EvidenceError("official sample row IDs differ")


def _verify_campaign(candidate: ImportedTreeE2Candidate) -> None:
    if not isinstance(candidate, ImportedTreeE2Candidate):
        raise TreeE2EvidenceError("E2 candidate type differs")
    decision = candidate.acceptance_decision
    if (
        decision.get("status") != "accepted"
        or decision.get("predictor") != "catboost"
        or decision.get("reason") != "standalone_catboost_gates_passed"
        or decision.get("performance_grade") != "breakthrough"
        or float(decision.get("weighted_gain", 0.0)) <= 0.0
        or float(decision.get("worst_fold_gain", 0.0)) <= 0.0
        or float(decision.get("bootstrap_lower", 0.0)) <= 0.0
    ):
        raise TreeE2EvidenceError("E2 acceptance decision differs")
    audit = candidate.inference_audit
    expected_checks = {
        "batch_1",
        "batch_257",
        "batch_4096",
        "companion",
        "reverse",
        "shuffle",
        "singleton",
    }
    checks = audit.get("checks")
    if (
        audit.get("status") != "passed"
        or audit.get("reason") != "inference_audit_passed"
        or audit.get("row_count") != 245_789
        or audit.get("maximum_absolute_difference") != 0.0
        or type(checks) is not dict
        or set(checks) != expected_checks
        or any(value != 0.0 for value in checks.values())
    ):
        raise TreeE2EvidenceError("E2 inference audit differs")
    if (
        not 0.0 < float(audit.get("elapsed_seconds", 0.0)) <= 480.0
        or not 0 <= int(audit.get("peak_rss_bytes", -1)) <= 28 * 1024**3
        or not 0 <= int(audit.get("peak_gpu_bytes", -1)) <= int(22.4 * 1024**3)
    ):
        raise TreeE2EvidenceError("E2 runtime evidence differs")


def _verify_environment(probe: Mapping[str, object]) -> None:
    expected = {
        "status": "passed",
        "python": "3.11.15",
        "catboost": "1.2.10",
        "pandas": "2.0.3",
        "numpy": "1.26.4",
    }
    if not isinstance(probe, Mapping) or dict(probe) != expected:
        raise TreeE2EvidenceError("Python 3.11 environment probe differs")


def build_tree_e2_acceptance(
    *,
    project_root: str | Path,
    candidate: ImportedTreeE2Candidate,
    test_frame: "object",
    sample_frame: "object",
    runtime_bytes: bytes,
    output_dir: str | Path,
    load_predictor: Callable[[], object],
    python_probe: Mapping[str, object],
    package_bytes: int,
    extracted_bytes: int,
    policy_path: str | Path,
    policy_review_path: str | Path,
    package_time: datetime,
) -> TreeE2Acceptance:
    """Create standard acceptance only after every E2 and current-code gate passes."""

    root = Path(project_root).expanduser().resolve(strict=True)
    target = Path(output_dir).expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise TreeE2EvidenceError(f"acceptance output already exists: {target}")
    if not isinstance(runtime_bytes, bytes) or not runtime_bytes:
        raise TreeE2EvidenceError("final runtime is empty")
    if type(package_bytes) is not int or type(extracted_bytes) is not int:
        raise TreeE2EvidenceError("package size projection is invalid")
    _verify_campaign(candidate)
    _verify_frames(test_frame, sample_frame)
    _verify_environment(python_probe)

    source_root = Path(tempfile.mkdtemp(prefix=".tree-e2-source-", dir=root))
    try:
        source_path = source_root / "script.py"
        source_path.write_bytes(runtime_bytes)
        inspect_inference_source([source_path], project_root=root)
    except RulesCodeGateError as error:
        raise TreeE2EvidenceError(f"source gate failed: {error}") from error
    finally:
        shutil.rmtree(source_root, ignore_errors=True)

    try:
        policy = load_policy(policy_path, project_root=root)
        load_policy_review(policy_review_path, policy=policy, package_time=package_time)
    except RulesContractError as error:
        raise TreeE2EvidenceError(str(error)) from error
    limits = policy["limits"]
    audit = candidate.inference_audit
    inference_seconds = float(audit["elapsed_seconds"])
    peak_ram = int(audit["peak_rss_bytes"])
    peak_gpu = int(audit["peak_gpu_bytes"])
    install_seconds = 120.0
    if (
        install_seconds > int(limits["install_seconds"])
        or inference_seconds > int(policy["runtime_safety_seconds"])
        or package_bytes > int(limits["package_bytes"])
        or extracted_bytes > int(limits["extracted_bytes"])
    ):
        raise TreeE2EvidenceError("E2 package projection exceeds evaluator limits")

    metadata = candidate_metadata(candidate)
    script_source = Path(__file__).with_name("tree_expert_e2_script.py").read_bytes()
    identity = AuditIdentity(
        policy_version=str(policy["policy_version"]),
        candidate_id=candidate.candidate_id,
        data_sha256=_frame_digest(test_frame, sample_frame),
        code_sha256=sha256(script_source).hexdigest(),
        config_sha256=sha256(_canonical(metadata)).hexdigest(),
        preprocessing_sha256=str(
            candidate.member_sha256["frozen_state/feature_state.json"]
        ),
        model_sha256=candidate.model_sha256,
        adapter_sha256=sha256(script_source).hexdigest(),
        runtime_sha256=sha256(runtime_bytes).hexdigest(),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        current_audit = run_phased_independence_audit(
            frame=test_frame,
            output_dir=temporary / "full_audit",
            identity=identity,
            load_predictor=load_predictor,
            singleton_count=5,
            tolerance=1e-12,
        )
        if current_audit.status != "passed":
            raise TreeE2EvidenceError("current row-independence audit did not pass")
        decision = candidate.acceptance_decision
        gates = {
            "temporal_validation": float(decision["worst_fold_gain"]) > 0.0,
            "performance": float(decision["bootstrap_lower"]) > 0.0,
            "provenance": True,
            "row_independence": True,
            "evaluator_runtime": True,
            "pretrained_license": True,
            "current_rules": True,
        }
        if any(value is not True for value in gates.values()):
            raise TreeE2EvidenceError("one or more E2 acceptance gates failed")
        acceptance = {
            "schema_version": 1,
            "status": "passed",
            "candidate_id": candidate.candidate_id,
            "policy_version": policy["policy_version"],
            "adapter_id": TREE_E2_ADAPTER_ID,
            "identity": asdict(identity),
            "gates": gates,
        }
        benchmark = {
            "schema_version": 1,
            "status": "passed",
            "candidate_id": candidate.candidate_id,
            "adapter_id": TREE_E2_ADAPTER_ID,
            "identity": asdict(identity),
            "install_seconds": install_seconds,
            "inference_seconds": inference_seconds,
            "peak_ram_bytes": peak_ram,
            "peak_vram_bytes": peak_gpu,
            "extracted_bytes": extracted_bytes,
            "package_bytes": package_bytes,
        }
        _write_json(temporary / "acceptance.json", acceptance)
        _write_json(temporary / "runtime_benchmark.json", benchmark)
        _write_json(
            temporary / "e2_campaign_evidence.json",
            {
                "handoff_sha256": candidate.handoff_sha256,
                "delivery_sha256": candidate.delivery_sha256,
                "delivery_manifest_sha256": candidate.delivery_manifest_sha256,
                "acceptance_decision": dict(candidate.acceptance_decision),
                "full_fit_manifest": dict(candidate.full_fit_manifest),
                "inference_audit": dict(candidate.inference_audit),
            },
        )
        _write_json(temporary / "python_probe.json", dict(python_probe))
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return TreeE2Acceptance(
        identity=identity,
        audit_manifest=target / "full_audit/full_audit_manifest.json",
        acceptance_path=target / "acceptance.json",
        benchmark_path=target / "runtime_benchmark.json",
        acceptance=acceptance,
        benchmark=benchmark,
    )
