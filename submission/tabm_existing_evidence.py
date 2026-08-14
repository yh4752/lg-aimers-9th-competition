"""Verification and reuse of the completed Version D GPU evidence."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import json
from math import isfinite
from pathlib import Path, PurePosixPath
import stat
from types import MappingProxyType
from typing import Mapping
import zipfile

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
