"""Import the single accepted Tree Expert E2 delivery as a frozen candidate."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from types import MappingProxyType
from typing import Mapping
import zipfile


TREE_E2_CANDIDATE_ID = "c1_anchor_residual"
TREE_E2_ADAPTER_ID = "tree_expert_e2_c1_catboost_v1"
TREE_E2_HANDOFF_SHA256 = (
    "4dd0c9012b9a59e5235b76d6fe1f6ded4df4b404cab0082c0f3084b8c384050f"
)

_OUTER_NAMES = {
    "handoff_manifest.json",
    "tree_expert_e2.log",
    "tree_expert_e2_model_delivery.zip",
    "tree_expert_e2_resume.zip",
    "tree_expert_e2_review.zip",
}
_DELIVERY_PAYLOADS = {
    "evidence/acceptance_decision.json",
    "evidence/full_fit_manifest.json",
    "evidence/inference_audit.json",
    "frozen_state/feature_state.json",
    "frozen_state/s1_batter.csv",
    "frozen_state/s1_pitcher.csv",
    "models/catboost_seed_42.cbm",
    "models/catboost_seed_2026.cbm",
    "models/catboost_seed_3407.cbm",
}
_MODEL_PAYLOADS = {
    name for name in _DELIVERY_PAYLOADS if name.startswith(("frozen_state/", "models/"))
}
_MAX_TOTAL_BYTES = 8 * 1024**3


class TreeE2CandidateError(ValueError):
    """Raised before an unregistered or changed E2 artifact can be imported."""


@dataclass(frozen=True)
class ImportedTreeE2Candidate:
    candidate_id: str
    adapter_id: str
    root: Path
    model_dir: Path
    handoff_sha256: str
    delivery_sha256: str
    delivery_manifest_sha256: str
    model_sha256: str
    member_sha256: Mapping[str, str]
    seeds: tuple[int, ...]
    iterations: Mapping[int, int]
    acceptance_decision: Mapping[str, object]
    full_fit_manifest: Mapping[str, object]
    inference_audit: Mapping[str, object]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json(value: bytes, label: str) -> dict[str, object]:
    def unique(items: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, item in items:
            if key in output:
                raise TreeE2CandidateError(f"duplicate JSON key in {label}: {key}")
            output[key] = item
        return output

    def finite(constant: str) -> object:
        raise TreeE2CandidateError(f"non-finite JSON value in {label}: {constant}")

    try:
        payload = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=finite,
        )
    except TreeE2CandidateError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TreeE2CandidateError(f"invalid JSON in {label}") from error
    if type(payload) is not dict:
        raise TreeE2CandidateError(f"JSON object required in {label}")
    return payload


def _canonical(value: object) -> bytes:
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


def _safe_infos(archive: zipfile.ZipFile, label: str) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)) or len(names) > 32:
        raise TreeE2CandidateError(f"duplicate or excessive members in {label}")
    total = 0
    output: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if (
            info.is_dir()
            or path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or stat.S_ISLNK(mode)
            or info.file_size < 0
        ):
            raise TreeE2CandidateError(f"unsafe archive member in {label}: {info.filename}")
        total += info.file_size
        output[info.filename] = info
    if total > _MAX_TOTAL_BYTES:
        raise TreeE2CandidateError(f"expanded archive exceeds limit in {label}")
    return output


def _member_sha256(archive: zipfile.ZipFile, name: str) -> str:
    digest = sha256()
    with archive.open(name) as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _verify_declared(
    archive: zipfile.ZipFile,
    declared: object,
    names: set[str],
    label: str,
) -> None:
    if type(declared) is not dict or set(declared) != names:
        raise TreeE2CandidateError(f"{label} member set differs")
    for name in sorted(names):
        entry = declared[name]
        if type(entry) is not dict or set(entry) != {"sha256", "size"}:
            raise TreeE2CandidateError(f"{label} member metadata differs: {name}")
        info = archive.getinfo(name)
        if entry["size"] != info.file_size or entry["sha256"] != _member_sha256(archive, name):
            raise TreeE2CandidateError(f"{label} member SHA-256 differs: {name}")


def _model_digest(members: Mapping[str, bytes]) -> str:
    digest = sha256()
    for name, value in sorted(members.items()):
        digest.update(name.encode("utf-8") + b"\0" + sha256(value).digest())
    return digest.hexdigest()


def _verify_delivery(value: bytes) -> tuple[dict[str, bytes], dict[str, object]]:
    try:
        with zipfile.ZipFile(io.BytesIO(value)) as archive:
            infos = _safe_infos(archive, "E2 model delivery")
            if set(infos) != _DELIVERY_PAYLOADS | {"manifest.json"}:
                raise TreeE2CandidateError("E2 model delivery member set differs")
            manifest_bytes = archive.read("manifest.json")
            manifest = _json(manifest_bytes, "E2 delivery manifest")
            if (
                manifest.get("schema_version") != 1
                or manifest.get("artifact_kind") != "tree_expert_e2_model_delivery_v1"
                or manifest.get("campaign_id") != "tree_expert_e2_v1"
                or manifest.get("candidate_id") != TREE_E2_CANDIDATE_ID
                or manifest.get("predictor") != "catboost"
                or manifest.get("seeds") != [42, 2026, 3407]
                or manifest.get("iterations")
                != {"42": 78, "2026": 86, "3407": 149}
                or manifest.get("review_only") is not False
                or manifest.get("submission_package") is not False
            ):
                raise TreeE2CandidateError("E2 model delivery identity differs")
            _verify_declared(
                archive, manifest.get("members"), _DELIVERY_PAYLOADS, "E2 delivery"
            )
            payloads = {name: archive.read(name) for name in _DELIVERY_PAYLOADS}
    except TreeE2CandidateError:
        raise
    except zipfile.BadZipFile as error:
        raise TreeE2CandidateError("E2 model delivery is not a valid ZIP") from error

    decision = _json(payloads["evidence/acceptance_decision.json"], "acceptance decision")
    full_fit = _json(payloads["evidence/full_fit_manifest.json"], "full-fit manifest")
    audit = _json(payloads["evidence/inference_audit.json"], "inference audit")
    feature = _json(payloads["frozen_state/feature_state.json"], "feature state")
    if (
        manifest.get("decision_sha256")
        != sha256(payloads["evidence/acceptance_decision.json"]).hexdigest()
        or decision.get("status") != "accepted"
        or decision.get("reason") != "standalone_catboost_gates_passed"
        or decision.get("predictor") != "catboost"
    ):
        raise TreeE2CandidateError("E2 acceptance decision differs")
    expected_models = {
        str(seed): sha256(payloads[f"models/catboost_seed_{seed}.cbm"]).hexdigest()
        for seed in (42, 2026, 3407)
    }
    if (
        full_fit.get("candidate_id") != TREE_E2_CANDIDATE_ID
        or full_fit.get("predictor") != "catboost"
        or full_fit.get("model_sha256") != expected_models
    ):
        raise TreeE2CandidateError("E2 full-fit identity differs")
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
        raise TreeE2CandidateError("E2 inference audit differs")
    frozen_members = feature.get("members")
    if (
        feature.get("schema_version") != 1
        or feature.get("artifact_kind") != "tree_expert_e2_frozen_state_v1"
        or feature.get("candidate_id") != TREE_E2_CANDIDATE_ID
        or feature.get("valid_year") != 2025
        or feature.get("trackman") is not None
        or type(frozen_members) is not dict
    ):
        raise TreeE2CandidateError("E2 frozen-state identity differs")
    for name in ("s1_batter.csv", "s1_pitcher.csv"):
        entry = frozen_members.get(name)
        value = payloads[f"frozen_state/{name}"]
        if (
            type(entry) is not dict
            or entry.get("sha256") != sha256(value).hexdigest()
            or entry.get("size") != len(value)
        ):
            raise TreeE2CandidateError(f"E2 frozen-state member differs: {name}")
    return payloads, {
        "manifest": manifest,
        "manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "decision": decision,
        "full_fit": full_fit,
        "audit": audit,
    }


def import_tree_expert_e2_candidate(
    handoff_path: str | Path, destination: str | Path
) -> ImportedTreeE2Candidate:
    """Verify and exclusively publish the registered E2 inference files."""

    source = Path(handoff_path).expanduser().resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise TreeE2CandidateError("E2 handoff is missing or unsafe")
    handoff_sha = _file_sha256(source)
    if handoff_sha != TREE_E2_HANDOFF_SHA256:
        raise TreeE2CandidateError("E2 handoff SHA-256 differs from the registered artifact")
    try:
        with zipfile.ZipFile(source) as archive:
            infos = _safe_infos(archive, "E2 handoff")
            if set(infos) != _OUTER_NAMES:
                raise TreeE2CandidateError("E2 handoff member set differs")
            handoff = _json(archive.read("handoff_manifest.json"), "handoff manifest")
            if (
                handoff.get("schema_version") != 1
                or handoff.get("artifact_kind") != "tree_expert_e2_handoff_v1"
                or handoff.get("campaign_id") != "tree_expert_e2_v1"
                or handoff.get("status") != "accepted"
                or handoff.get("delivery") is not True
                or handoff.get("review_only") is not False
                or handoff.get("submission_package") is not False
            ):
                raise TreeE2CandidateError("E2 handoff is not accepted")
            outer_payloads = _OUTER_NAMES - {"handoff_manifest.json"}
            _verify_declared(
                archive, handoff.get("members"), outer_payloads, "E2 handoff"
            )
            delivery = archive.read("tree_expert_e2_model_delivery.zip")
    except TreeE2CandidateError:
        raise
    except zipfile.BadZipFile as error:
        raise TreeE2CandidateError("E2 handoff is not a valid ZIP") from error

    payloads, evidence = _verify_delivery(delivery)
    model_members = {name: payloads[name] for name in _MODEL_PAYLOADS}
    member_sha = {name: sha256(value).hexdigest() for name, value in model_members.items()}
    model_sha = _model_digest(model_members)
    output = Path(destination).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise TreeE2CandidateError(f"candidate output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        model_dir = temporary / "model"
        for name, value in sorted(model_members.items()):
            path = model_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        candidate_manifest = {
            "schema_version": 1,
            "artifact_kind": "tree_expert_e2_submission_candidate_v1",
            "candidate_id": TREE_E2_CANDIDATE_ID,
            "adapter_id": TREE_E2_ADAPTER_ID,
            "handoff_sha256": handoff_sha,
            "delivery_sha256": sha256(delivery).hexdigest(),
            "delivery_manifest_sha256": evidence["manifest_sha256"],
            "model_sha256": model_sha,
            "members": member_sha,
            "seeds": [42, 2026, 3407],
            "iterations": {"42": 78, "2026": 86, "3407": 149},
        }
        (temporary / "candidate_manifest.json").write_bytes(_canonical(candidate_manifest))
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return ImportedTreeE2Candidate(
        candidate_id=TREE_E2_CANDIDATE_ID,
        adapter_id=TREE_E2_ADAPTER_ID,
        root=output,
        model_dir=output / "model",
        handoff_sha256=handoff_sha,
        delivery_sha256=sha256(delivery).hexdigest(),
        delivery_manifest_sha256=str(evidence["manifest_sha256"]),
        model_sha256=model_sha,
        member_sha256=MappingProxyType(member_sha),
        seeds=(42, 2026, 3407),
        iterations=MappingProxyType({42: 78, 2026: 86, 3407: 149}),
        acceptance_decision=MappingProxyType(dict(evidence["decision"])),
        full_fit_manifest=MappingProxyType(dict(evidence["full_fit"])),
        inference_audit=MappingProxyType(dict(evidence["audit"])),
    )
