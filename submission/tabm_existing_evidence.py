"""Verification and reuse of the completed Version D GPU evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import io
import json
from math import isfinite
from pathlib import Path, PurePosixPath
import os
import shutil
import stat
import tempfile
from types import MappingProxyType
from typing import Callable, Mapping
import zipfile

from competition_rules.contract import RulesContractError, load_policy, load_policy_review
from competition_rules.code_gate import RulesCodeGateError, inspect_inference_source
from competition_rules.evidence_gate import AuditIdentity, run_phased_independence_audit

from .tabm_candidate import CANDIDATE_ID, ImportedTabMCandidate


class ExistingEvidenceError(ValueError):
    """Raised when prior evidence cannot be bound to the current candidate."""


@dataclass(frozen=True)
class ExistingGpuEvidence:
    candidate_id: str
    stage_c_sha256: str
    stage_d_sha256: str
    model_sha256: str
    model_members: Mapping[str, str]
    gpu_name: str
    capacity_rows: int
    inference_seconds: float
    peak_ram_bytes: int
    peak_vram_bytes: int
    independence_delta: float
    install_seconds: float
    primary_brier: float
    older_brier: float
    gpu_sample_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class ReusedGpuAcceptance:
    identity: AuditIdentity
    audit_manifest: Path
    acceptance_path: Path
    benchmark_path: Path
    acceptance: dict[str, object]
    benchmark: dict[str, object]


_STAGE_C_NAMES = {
    "colab_stage_C.log",
    "delivery_manifest.json",
    "tabm_search_stage_C_resume_bundle.zip",
    "tabm_search_stage_C_review_bundle.zip",
}
_STAGE_D_NAMES = {
    "delivery_manifest.json",
    "tabm_hand_matchup_final_review_bundle.zip",
    "version_d.log",
}
_MODEL_NAMES = {
    "inference_manifest.json",
    "numeric_embedding_0.json",
    "preprocessing_state.json",
    "tabm_member_0_seed_3407.pt",
}
_STAGE_D_INNER_NAMES = {
    "final_review.json",
    "manifest.json",
    "policy/policy.json",
    "policy/policy_review.json",
    *(f"frozen_inference/{name}" for name in _MODEL_NAMES),
}
_MAX_EXPANDED_BYTES = 32_000_000_000


def _file_bytes(path: str | Path, label: str) -> bytes:
    candidate = Path(path).expanduser().resolve(strict=True)
    if candidate.is_symlink() or not candidate.is_file():
        raise ExistingEvidenceError(f"{label} is missing or unsafe")
    try:
        return candidate.read_bytes()
    except OSError as error:
        raise ExistingEvidenceError(f"cannot read {label}") from error


def _json(value: bytes, label: str) -> dict[str, object]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, item in pairs:
            if key in output:
                raise ExistingEvidenceError(f"duplicate JSON key in {label}")
            output[key] = item
        return output

    def finite(constant: str) -> object:
        raise ExistingEvidenceError(f"non-finite JSON in {label}: {constant}")

    try:
        payload = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=finite,
        )
    except ExistingEvidenceError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ExistingEvidenceError(f"invalid JSON: {label}") from error
    if not isinstance(payload, dict):
        raise ExistingEvidenceError(f"JSON must be an object: {label}")
    return payload


def _archive(value: bytes, label: str) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(value)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or len(names) > 128:
                raise ExistingEvidenceError(f"duplicate or excessive members in {label}")
            total = 0
            for info in infos:
                path = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if (
                    info.is_dir()
                    or path.is_absolute()
                    or ".." in path.parts
                    or not path.parts
                    or stat.S_ISLNK(mode)
                ):
                    raise ExistingEvidenceError(f"unsafe archive member in {label}")
                total += info.file_size
            if total > _MAX_EXPANDED_BYTES:
                raise ExistingEvidenceError(f"expanded archive exceeds limit in {label}")
            bad = archive.testzip()
            if bad is not None:
                raise ExistingEvidenceError(f"corrupt member in {label}: {bad}")
            return {info.filename: archive.read(info) for info in infos}
    except ExistingEvidenceError:
        raise
    except zipfile.BadZipFile as error:
        raise ExistingEvidenceError(f"invalid ZIP: {label}") from error


def _verify_size_hash_manifest(
    declared: object,
    members: Mapping[str, bytes],
    label: str,
) -> None:
    if not isinstance(declared, dict) or set(declared) != set(members):
        raise ExistingEvidenceError(f"{label} member set differs")
    for name, value in members.items():
        entry = declared[name]
        if not isinstance(entry, dict) or set(entry) != {"sha256", "size"}:
            raise ExistingEvidenceError(f"{label} member metadata differs")
        if entry["sha256"] != sha256(value).hexdigest() or entry["size"] != len(value):
            raise ExistingEvidenceError(f"{label} member SHA-256 differs: {name}")


def _verify_hash_manifest(
    declared: object,
    members: Mapping[str, bytes],
    label: str,
) -> None:
    if not isinstance(declared, dict) or set(declared) != set(members):
        raise ExistingEvidenceError(f"{label} member set differs")
    for name, value in members.items():
        if declared[name] != sha256(value).hexdigest():
            raise ExistingEvidenceError(f"{label} member SHA-256 differs: {name}")


def _stage_c_state(value: bytes) -> dict[str, object]:
    outer = _archive(value, "Stage C delivery")
    if set(outer) != _STAGE_C_NAMES:
        raise ExistingEvidenceError("Stage C delivery member set differs")
    manifest = _json(outer["delivery_manifest.json"], "Stage C delivery manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_kind") != "tabm_colab_stage_C_delivery"
        or manifest.get("final_stage_complete") is not True
    ):
        raise ExistingEvidenceError("Stage C delivery contract differs")
    declared_members = {name: outer[name] for name in _STAGE_C_NAMES - {"delivery_manifest.json"}}
    _verify_size_hash_manifest(manifest.get("members"), declared_members, "Stage C")

    resume = _archive(
        outer["tabm_search_stage_C_resume_bundle.zip"], "Stage C resume bundle"
    )
    if set(resume) != {"manifest.json", "stage_state.json"}:
        raise ExistingEvidenceError("Stage C resume member set differs")
    resume_manifest = _json(resume["manifest.json"], "Stage C resume manifest")
    if (
        resume_manifest.get("schema_version") != 1
        or resume_manifest.get("artifact_kind") != "resume"
        or resume_manifest.get("version") != "C"
        or resume_manifest.get("review_only") is not True
    ):
        raise ExistingEvidenceError("Stage C resume contract differs")
    _verify_hash_manifest(
        resume_manifest.get("members"),
        {"stage_state.json": resume["stage_state.json"]},
        "Stage C resume",
    )
    return _json(resume["stage_state.json"], "Stage C state")


def _stage_d_review(
    value: bytes,
) -> tuple[dict[str, object], dict[str, object], str]:
    outer = _archive(value, "Stage D delivery")
    if set(outer) != _STAGE_D_NAMES:
        raise ExistingEvidenceError("Stage D delivery member set differs")
    manifest = _json(outer["delivery_manifest.json"], "Stage D delivery manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_kind") != "tabm_version_D_review_delivery"
        or manifest.get("review_only") is not True
        or manifest.get("submission_package") is not False
    ):
        raise ExistingEvidenceError("Stage D delivery contract differs")
    declared = {
        "tabm_hand_matchup_final_review_bundle.zip": outer[
            "tabm_hand_matchup_final_review_bundle.zip"
        ],
        "version_d.log": outer["version_d.log"],
    }
    _verify_size_hash_manifest(manifest.get("members"), declared, "Stage D")

    inner = _archive(
        outer["tabm_hand_matchup_final_review_bundle.zip"], "Stage D review bundle"
    )
    if set(inner) != _STAGE_D_INNER_NAMES:
        raise ExistingEvidenceError("Stage D review member set differs")
    inner_manifest = _json(inner["manifest.json"], "Stage D review manifest")
    if (
        inner_manifest.get("schema_version") != 1
        or inner_manifest.get("artifact_kind") != "review"
        or inner_manifest.get("version") != "D"
        or inner_manifest.get("review_only") is not True
    ):
        raise ExistingEvidenceError("Stage D review contract differs")
    _verify_hash_manifest(
        inner_manifest.get("members"),
        {name: value for name, value in inner.items() if name != "manifest.json"},
        "Stage D review",
    )
    try:
        log = outer["version_d.log"].decode("utf-8")
    except UnicodeError as error:
        raise ExistingEvidenceError("Stage D log is not UTF-8") from error
    return manifest, _json(inner["final_review.json"], "Stage D final review"), log


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExistingEvidenceError(f"{label} is invalid")
    number = float(value)
    if not isfinite(number) or number < 0:
        raise ExistingEvidenceError(f"{label} is invalid")
    return number


def _current_model(candidate: ImportedTabMCandidate) -> tuple[dict[str, str], str]:
    if not isinstance(candidate, ImportedTabMCandidate) or candidate.candidate_id != CANDIDATE_ID:
        raise ExistingEvidenceError("candidate identity differs")
    paths = list(candidate.model_dir.iterdir()) if candidate.model_dir.is_dir() else []
    if {path.name for path in paths} != _MODEL_NAMES or any(
        path.is_symlink() or not path.is_file() for path in paths
    ):
        raise ExistingEvidenceError("candidate model member set differs")
    hashes: dict[str, str] = {}
    combined = sha256()
    for name in sorted(_MODEL_NAMES):
        value = (candidate.model_dir / name).read_bytes()
        digest = sha256(value).hexdigest()
        hashes[name] = digest
        combined.update(name.encode("utf-8") + b"\0" + bytes.fromhex(digest))
    if dict(candidate.member_sha256) != hashes or candidate.model_sha256 != combined.hexdigest():
        raise ExistingEvidenceError("candidate model SHA-256 differs")
    return hashes, combined.hexdigest()


def _accepted_predictor(state: dict[str, object]) -> tuple[float, float]:
    rows = state.get("predictor_evidence")
    if not isinstance(rows, list):
        raise ExistingEvidenceError("Stage C single_s3407 evidence is missing")
    selected = [row for row in rows if isinstance(row, dict) and row.get("predictor_id") == "single_s3407"]
    if len(selected) != 1 or selected[0].get("accepted") is not True:
        raise ExistingEvidenceError("Stage C single_s3407 is not accepted")
    members = selected[0].get("members")
    if members != [
        {
            "candidate_id": "a__p2__piecewise_linear__bce__plateau__s42",
            "seed": 3407,
        }
    ]:
        raise ExistingEvidenceError("Stage C single_s3407 member differs")
    final_members = state.get("final_members")
    if not isinstance(final_members, list) or not any(
        isinstance(item, dict)
        and item.get("seed") == 3407
        and item.get("status") == "completed"
        for item in final_members
    ):
        raise ExistingEvidenceError("Stage C seed 3407 final member is incomplete")
    return (
        _number(selected[0].get("primary_brier"), "primary Brier"),
        _number(selected[0].get("older_brier"), "older Brier"),
    )


def _sample_probabilities(review: dict[str, object]) -> tuple[float, ...]:
    probe = review.get("frozen_sample_probe")
    if not isinstance(probe, dict) or probe.get("status") != "passed":
        raise ExistingEvidenceError("Stage D sample probe did not pass")
    tail = probe.get("stdout_tail")
    if not isinstance(tail, str):
        raise ExistingEvidenceError("Stage D sample probe output is invalid")
    try:
        values = tuple(float(item) for item in tail.strip().split(","))
    except ValueError as error:
        raise ExistingEvidenceError("Stage D sample probe output is invalid") from error
    if len(values) != 5 or any(not isfinite(value) or value < 0 or value > 1 for value in values):
        raise ExistingEvidenceError("Stage D sample probe must contain five probabilities")
    return values


def verify_existing_gpu_evidence(
    *,
    stage_c_delivery: str | Path,
    stage_d_delivery: str | Path,
    candidate: ImportedTabMCandidate,
) -> ExistingGpuEvidence:
    """Verify both deliveries and return only hash-bound accepted fields."""

    stage_c_value = _file_bytes(stage_c_delivery, "Stage C delivery")
    stage_d_value = _file_bytes(stage_d_delivery, "Stage D delivery")
    state = _stage_c_state(stage_c_value)
    stage_d_manifest, review, log = _stage_d_review(stage_d_value)
    stage_c_digest = sha256(stage_c_value).hexdigest()
    if stage_d_manifest.get("stage_c_delivery_sha256") != stage_c_digest:
        raise ExistingEvidenceError("Stage C delivery SHA-256 differs from Stage D")

    model_members, model_digest = _current_model(candidate)
    if review.get("artifact_sha256") != model_members:
        raise ExistingEvidenceError("Stage D model SHA-256 differs from current candidate")
    fit = review.get("fit_report")
    if (
        review.get("acceptance_status") != "review_ready"
        or review.get("version") != "D"
        or review.get("review_only") is not True
        or review.get("fixed_epochs") != 3
        or not isinstance(fit, dict)
        or fit.get("status") != "completed"
        or fit.get("epochs") != 3
        or fit.get("rows") != 1_475_092
    ):
        raise ExistingEvidenceError("Stage D completed fit contract differs")

    gpu_name = "Tesla T4"
    if f"VERSION_D_GPU_READY device_count=1 name={gpu_name}" not in log:
        raise ExistingEvidenceError("Stage D Tesla T4 evidence is missing")
    if "VERSION_D_INDEPENDENCE_PASSED" not in log:
        raise ExistingEvidenceError("Stage D row independence marker is missing")
    if "VERSION_D_SCALE_GATE_PASSED rows=245789" not in log:
        raise ExistingEvidenceError("Stage D 245789-row scale marker is missing")

    scale = review.get("scale")
    independence = review.get("independence")
    dependency = review.get("dependency_probe")
    if not isinstance(scale, dict) or scale.get("rows") != 245_789:
        raise ExistingEvidenceError("Stage D capacity must contain 245789 rows")
    if not isinstance(independence, dict):
        raise ExistingEvidenceError("Stage D row independence evidence is missing")
    delta = _number(
        independence.get("max_abs_probability_delta"), "row independence delta"
    )
    if independence.get("row_count") != 5 or delta > 1e-6:
        raise ExistingEvidenceError("Stage D row independence limit failed")
    if not isinstance(dependency, dict) or dependency.get("status") != "passed":
        raise ExistingEvidenceError("Stage D dependency probe did not pass")

    primary, older = _accepted_predictor(state)
    return ExistingGpuEvidence(
        candidate_id=CANDIDATE_ID,
        stage_c_sha256=stage_c_digest,
        stage_d_sha256=sha256(stage_d_value).hexdigest(),
        model_sha256=model_digest,
        model_members=MappingProxyType(model_members),
        gpu_name=gpu_name,
        capacity_rows=245_789,
        inference_seconds=_number(scale.get("elapsed_seconds"), "inference seconds"),
        peak_ram_bytes=int(_number(scale.get("peak_rss_bytes"), "peak RAM")),
        peak_vram_bytes=int(
            _number(scale.get("peak_gpu_allocated_bytes"), "peak VRAM")
        ),
        independence_delta=delta,
        install_seconds=_number(dependency.get("elapsed_seconds"), "install seconds"),
        primary_brier=primary,
        older_brier=older,
        gpu_sample_probabilities=_sample_probabilities(review),
    )


def _canonical(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ExistingEvidenceError("evidence cannot be canonically encoded") from error


def _write_json(path: Path, value: object) -> None:
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


def _probe_probabilities(probe: Mapping[str, object]) -> tuple[float, ...]:
    if (
        probe.get("status") != "passed"
        or not str(probe.get("python", "")).startswith("3.11.")
    ):
        raise ExistingEvidenceError("Python 3.11 compatibility probe did not pass")
    if probe.get("tabm") != "0.0.3" or probe.get("rtdl_num_embeddings") != "0.0.12":
        raise ExistingEvidenceError("submitted dependency versions differ")
    values = probe.get("probabilities")
    if not isinstance(values, (list, tuple)) or len(values) != 5:
        raise ExistingEvidenceError("Python probe probabilities are invalid")
    probabilities = tuple(_number(value, "Python probe probability") for value in values)
    if any(value > 1 for value in probabilities):
        raise ExistingEvidenceError("Python probe probabilities are invalid")
    return probabilities


def _assert_probability_parity(
    left: tuple[float, ...], right: tuple[float, ...], *, tolerance: float = 1e-6
) -> None:
    if len(left) != len(right) or any(
        abs(first - second) > tolerance for first, second in zip(left, right, strict=True)
    ):
        raise ExistingEvidenceError("sample prediction parity differs")


def _gpu_evidence_payload(evidence: ExistingGpuEvidence) -> dict[str, object]:
    return {
        "candidate_id": evidence.candidate_id,
        "stage_c_sha256": evidence.stage_c_sha256,
        "stage_d_sha256": evidence.stage_d_sha256,
        "model_sha256": evidence.model_sha256,
        "model_members": dict(evidence.model_members),
        "gpu_name": evidence.gpu_name,
        "capacity_rows": evidence.capacity_rows,
        "inference_seconds": evidence.inference_seconds,
        "peak_ram_bytes": evidence.peak_ram_bytes,
        "peak_vram_bytes": evidence.peak_vram_bytes,
        "independence_delta": evidence.independence_delta,
        "install_seconds": evidence.install_seconds,
        "primary_brier": evidence.primary_brier,
        "older_brier": evidence.older_brier,
        "gpu_sample_probabilities": list(evidence.gpu_sample_probabilities),
    }


def build_reused_gpu_acceptance(
    *,
    project_root: str | Path,
    candidate: ImportedTabMCandidate,
    gpu_evidence: ExistingGpuEvidence,
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
) -> ReusedGpuAcceptance:
    """Publish acceptance only after sample parity and all existing gates pass."""

    import numpy as np
    import pandas as pd

    root = Path(project_root).expanduser().resolve(strict=True)
    target = Path(output_dir).expanduser().absolute()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ExistingEvidenceError("acceptance output is outside project root") from error
    if target.exists():
        raise ExistingEvidenceError(f"acceptance output already exists: {target}")
    if not isinstance(candidate, ImportedTabMCandidate):
        raise ExistingEvidenceError("candidate type differs")
    if (
        gpu_evidence.candidate_id != candidate.candidate_id
        or gpu_evidence.model_sha256 != candidate.model_sha256
        or dict(gpu_evidence.model_members) != dict(candidate.member_sha256)
    ):
        raise ExistingEvidenceError("GPU evidence model identity differs")
    if (
        gpu_evidence.gpu_name != "Tesla T4"
        or gpu_evidence.capacity_rows != 245_789
        or gpu_evidence.independence_delta > 1e-6
    ):
        raise ExistingEvidenceError("GPU capacity or row-independence evidence differs")
    if not isinstance(test_frame, pd.DataFrame) or not isinstance(sample_frame, pd.DataFrame):
        raise ExistingEvidenceError("official sample frames are invalid")
    if len(test_frame) != 5 or len(sample_frame) != 5 or "row_id" not in test_frame:
        raise ExistingEvidenceError("official sample must contain five rows")
    if sample_frame.columns.tolist() != ["row_id", "control_success"]:
        raise ExistingEvidenceError("official sample submission columns differ")
    test_ids = test_frame["row_id"].astype("string")
    sample_ids = sample_frame["row_id"].astype("string")
    if (
        test_ids.isna().any()
        or sample_ids.isna().any()
        or test_ids.astype(str).duplicated().any()
        or set(test_ids.astype(str)) != set(sample_ids.astype(str))
    ):
        raise ExistingEvidenceError("official sample row IDs differ")
    if not isinstance(runtime_bytes, bytes) or not runtime_bytes:
        raise ExistingEvidenceError("final runtime is empty")
    if type(package_bytes) is not int or type(extracted_bytes) is not int:
        raise ExistingEvidenceError("package size projection is invalid")

    target.parent.mkdir(parents=True, exist_ok=True)
    source_root = Path(
        tempfile.mkdtemp(prefix=".tabm-source-gate-", dir=target.parent)
    )
    try:
        source_path = source_root / "script.py"
        source_path.write_bytes(runtime_bytes)
        inspect_inference_source([source_path], project_root=root)
    except RulesCodeGateError as error:
        raise ExistingEvidenceError(f"source gate failed: {error}") from error
    finally:
        shutil.rmtree(source_root, ignore_errors=True)

    try:
        policy = load_policy(policy_path, project_root=root)
        load_policy_review(
            policy_review_path, policy=policy, package_time=package_time
        )
    except RulesContractError as error:
        raise ExistingEvidenceError(str(error)) from error

    probe_probabilities = _probe_probabilities(python_probe)
    _assert_probability_parity(
        probe_probabilities, gpu_evidence.gpu_sample_probabilities
    )
    predictor = load_predictor()
    current = np.asarray(
        predictor.predict_batch(test_frame.copy(deep=True), batch_size=2048),
        dtype="float64",
    )
    if current.shape != (5,) or not np.isfinite(current).all():
        raise ExistingEvidenceError("current sample predictions are invalid")
    current_probabilities = tuple(float(value) for value in current)
    _assert_probability_parity(
        current_probabilities, gpu_evidence.gpu_sample_probabilities
    )

    config_value = _canonical(
        {
            "candidate_id": candidate.candidate_id,
            "delivery_sha256": candidate.delivery_sha256,
            "review_bundle_sha256": candidate.review_bundle_sha256,
            "members": dict(candidate.member_sha256),
        }
    )
    adapter_source = Path(__file__).with_name("tabm_version_d_script.py").read_bytes()
    runtime_sha256 = sha256(runtime_bytes).hexdigest()
    identity = AuditIdentity(
        policy_version=str(policy["policy_version"]),
        candidate_id=candidate.candidate_id,
        data_sha256=_frame_digest(test_frame, sample_frame),
        code_sha256=sha256(adapter_source).hexdigest(),
        config_sha256=sha256(config_value).hexdigest(),
        preprocessing_sha256=str(
            candidate.member_sha256["preprocessing_state.json"]
        ),
        model_sha256=candidate.model_sha256,
        adapter_sha256=sha256(adapter_source).hexdigest(),
        runtime_sha256=runtime_sha256,
    )

    install = gpu_evidence.install_seconds
    inference = gpu_evidence.inference_seconds
    if install > int(policy["limits"]["install_seconds"]):
        raise ExistingEvidenceError("reused install evidence exceeds official limit")
    if inference > int(policy["runtime_safety_seconds"]):
        raise ExistingEvidenceError("reused inference evidence exceeds safety limit")
    if package_bytes > int(policy["limits"]["package_bytes"]):
        raise ExistingEvidenceError("package projection exceeds official limit")
    if extracted_bytes > int(policy["limits"]["extracted_bytes"]):
        raise ExistingEvidenceError("extracted projection exceeds official limit")
    if gpu_evidence.peak_ram_bytes > 28 * 1024**3:
        raise ExistingEvidenceError("reused RAM evidence exceeds official limit")
    if gpu_evidence.peak_vram_bytes > 22.4 * 1024**3:
        raise ExistingEvidenceError("reused VRAM evidence exceeds official limit")

    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        audit = run_phased_independence_audit(
            frame=test_frame,
            output_dir=temporary / "full_audit",
            identity=identity,
            load_predictor=load_predictor,
            singleton_count=5,
        )
        if audit.status != "passed":
            raise ExistingEvidenceError("current row-independence audit did not pass")
        gates = {
            "temporal_validation": gpu_evidence.primary_brier < 0.25,
            "performance": gpu_evidence.primary_brier < gpu_evidence.older_brier,
            "provenance": True,
            "row_independence": True,
            "evaluator_runtime": True,
            "pretrained_license": True,
            "current_rules": True,
        }
        if any(value is not True for value in gates.values()):
            raise ExistingEvidenceError("one or more acceptance gates failed")
        acceptance: dict[str, object] = {
            "schema_version": 1,
            "status": "passed",
            "candidate_id": candidate.candidate_id,
            "policy_version": policy["policy_version"],
            "adapter_id": CANDIDATE_ID,
            "identity": asdict(identity),
            "gates": gates,
        }
        benchmark: dict[str, object] = {
            "schema_version": 1,
            "status": "passed",
            "candidate_id": candidate.candidate_id,
            "adapter_id": CANDIDATE_ID,
            "identity": asdict(identity),
            "install_seconds": install,
            "inference_seconds": inference,
            "peak_ram_bytes": gpu_evidence.peak_ram_bytes,
            "peak_vram_bytes": gpu_evidence.peak_vram_bytes,
            "extracted_bytes": extracted_bytes,
            "package_bytes": package_bytes,
        }
        _write_json(temporary / "acceptance.json", acceptance)
        _write_json(temporary / "runtime_benchmark.json", benchmark)
        _write_json(temporary / "gpu_evidence.json", _gpu_evidence_payload(gpu_evidence))
        _write_json(temporary / "python_probe.json", dict(python_probe))
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return ReusedGpuAcceptance(
        identity=identity,
        audit_manifest=target / "full_audit/full_audit_manifest.json",
        acceptance_path=target / "acceptance.json",
        benchmark_path=target / "runtime_benchmark.json",
        acceptance=acceptance,
        benchmark=benchmark,
    )
