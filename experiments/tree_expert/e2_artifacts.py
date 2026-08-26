from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from types import MappingProxyType
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from .e2_full_fit import AcceptedForFullFit


class E2ArtifactError(ValueError):
    """Raised when E2 evidence is incomplete, unsafe, or changes identity."""


_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)
_BINDING_KEYS = {
    "contract_sha256",
    "code_sha256",
    "input_sha256",
    "train_sha256",
    "history_sha256",
}
_MAX_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
_MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
_MAX_RATIO = 250.0


@dataclass(frozen=True)
class E2DeliverySources:
    token: AcceptedForFullFit
    models: Mapping[int, Path]
    frozen_state: Path
    full_fit_manifest: Path
    acceptance_decision: Path
    inference_audit: Path
    tabm_root: Path | None


@dataclass(frozen=True)
class E2BundlePaths:
    review: Path
    resume: Path
    handoff: Path
    delivery: Path | None
    review_sha256: str
    resume_sha256: str
    handoff_sha256: str
    delivery_sha256: str | None


@dataclass(frozen=True)
class VerifiedE2Resume:
    path: Path
    manifest_sha256: str
    bindings: Mapping[str, str]
    status: str


@dataclass(frozen=True)
class VerifiedE2Handoff:
    path: Path
    manifest_sha256: str
    status: str
    delivery: bool


def file_sha256(path: Path) -> str:
    source = _source(path)
    digest = sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source(path: Path) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise E2ArtifactError(f"artifact source is not a regular file: {source}")
    return source


def _valid_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _bindings(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _BINDING_KEYS:
        raise E2ArtifactError("artifact binding keys differ")
    output = dict(value)
    if any(not _valid_sha(item) for item in output.values()):
        raise E2ArtifactError("artifact binding SHA-256 differs")
    return output


def _json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise E2ArtifactError(f"artifact JSON differs: {error}") from error


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(_source(path).read_text(encoding="utf-8"))
    except Exception as error:
        raise E2ArtifactError(f"cannot read {label}: {error}") from error
    if type(value) is not dict:
        raise E2ArtifactError(f"{label} must be an object")
    return value


def _evidence(value: bytes | Path) -> dict[str, object]:
    if isinstance(value, bytes):
        return {"size": len(value), "sha256": sha256(value).hexdigest()}
    source = _source(value)
    return {"size": source.stat().st_size, "sha256": file_sha256(source)}


def _info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _write_member(archive: ZipFile, name: str, value: bytes | Path) -> None:
    if isinstance(value, bytes):
        archive.writestr(_info(name), value)
        return
    with _source(value).open("rb") as source, archive.open(_info(name), "w") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def _write_zip(path: Path, members: Mapping[str, bytes | Path]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for name, value in sorted(members.items()):
                _write_member(archive, name, value)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _safe_names(archive: ZipFile) -> list[str]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise E2ArtifactError("artifact has duplicate members")
    total = 0
    for info in infos:
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        ratio = info.file_size / max(1, info.compress_size)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or info.is_dir()
            or stat.S_ISLNK(mode)
            or info.file_size > _MAX_MEMBER_BYTES
            or (info.file_size >= 64 * 1024 and ratio > _MAX_RATIO)
        ):
            raise E2ArtifactError(f"unsafe artifact member: {info.filename}")
        total += info.file_size
    if total > _MAX_TOTAL_BYTES:
        raise E2ArtifactError("artifact uncompressed size is too large")
    return names


def _verify_manifest_members(
    archive: ZipFile,
    manifest: Mapping[str, object],
    names: list[str],
    manifest_name: str,
) -> None:
    members = manifest.get("members")
    if type(members) is not dict or set(members) != set(names) - {manifest_name}:
        raise E2ArtifactError("artifact member manifest differs")
    for name, evidence in members.items():
        if type(evidence) is not dict or set(evidence) != {"size", "sha256"}:
            raise E2ArtifactError(f"artifact member evidence differs: {name}")
        digest = sha256()
        with archive.open(name) as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        if (
            evidence["size"] != archive.getinfo(name).file_size
            or evidence["sha256"] != digest.hexdigest()
        ):
            raise E2ArtifactError(f"artifact member differs: {name}")


def _state(path: Path, bindings: Mapping[str, str]) -> dict[str, object]:
    value = _read_json(path, "E2 state")
    expected = {
        "schema_version",
        "campaign_id",
        "status",
        "phase",
        "bindings",
        "completed",
        "skipped",
        "failed",
        "active",
        "decisions",
        "artifact_paths",
    }
    if (
        set(value) != expected
        or value["schema_version"] != 1
        or value["campaign_id"] != "tree_expert_e2_v1"
        or value["bindings"] != _bindings(bindings)
        or type(value["status"]) is not str
        or type(value["phase"]) is not str
        or any(type(value[name]) is not list for name in ("completed", "skipped", "failed"))
        or any(
            type(value[name]) is not dict
            for name in ("active", "decisions", "artifact_paths")
        )
    ):
        raise E2ArtifactError("E2 state identity or binding differs")
    return value


def _bundle_manifest(
    *,
    kind: str,
    state: Mapping[str, object],
    bindings: Mapping[str, str],
    members: Mapping[str, bytes | Path],
) -> bytes:
    return _json_bytes(
        {
            "schema_version": 1,
            "artifact_kind": kind,
            "campaign_id": "tree_expert_e2_v1",
            "review_only": True,
            "submission_package": False,
            "status": state["status"],
            "phase": state["phase"],
            "bindings": dict(bindings),
            "members": {name: _evidence(value) for name, value in sorted(members.items())},
        }
    )


def _write_evidence_bundle(
    path: Path,
    *,
    kind: str,
    state: Mapping[str, object],
    bindings: Mapping[str, str],
    members: Mapping[str, bytes | Path],
) -> Path:
    payloads = dict(members)
    payloads["manifest.json"] = _bundle_manifest(
        kind=kind,
        state=state,
        bindings=bindings,
        members=members,
    )
    output = _write_zip(path, payloads)
    _verify_evidence_bundle(output, kind, bindings)
    return output


def _verify_evidence_bundle(
    path: Path,
    kind: str,
    expected_bindings: Mapping[str, str],
) -> dict[str, object]:
    try:
        with ZipFile(_source(path)) as archive:
            names = _safe_names(archive)
            if "manifest.json" not in names:
                raise E2ArtifactError("artifact manifest is absent")
            manifest_bytes = archive.read("manifest.json")
            manifest = json.loads(manifest_bytes)
            expected = {
                "schema_version",
                "artifact_kind",
                "campaign_id",
                "review_only",
                "submission_package",
                "status",
                "phase",
                "bindings",
                "members",
            }
            if (
                type(manifest) is not dict
                or set(manifest) != expected
                or manifest["schema_version"] != 1
                or manifest["artifact_kind"] != kind
                or manifest["campaign_id"] != "tree_expert_e2_v1"
                or manifest["review_only"] is not True
                or manifest["submission_package"] is not False
                or manifest["bindings"] != _bindings(expected_bindings)
            ):
                raise E2ArtifactError("artifact identity or binding differs")
            _verify_manifest_members(archive, manifest, names, "manifest.json")
            manifest["__manifest_sha256__"] = sha256(manifest_bytes).hexdigest()
            return manifest
    except E2ArtifactError:
        raise
    except Exception as error:
        raise E2ArtifactError(f"cannot verify E2 artifact: {error}") from error


def verify_e2_resume(path: Path, expected_bindings: Mapping[str, str]) -> VerifiedE2Resume:
    manifest = _verify_evidence_bundle(path, "tree_expert_e2_resume_v1", expected_bindings)
    return VerifiedE2Resume(
        path=Path(path),
        manifest_sha256=str(manifest["__manifest_sha256__"]),
        bindings=MappingProxyType(dict(manifest["bindings"])),
        status=str(manifest["status"]),
    )


def _directory_members(root: Path, prefix: str) -> dict[str, Path]:
    source = Path(root)
    if source.is_symlink() or not source.is_dir():
        raise E2ArtifactError(f"delivery directory differs: {source}")
    output: dict[str, Path] = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise E2ArtifactError(f"delivery symlink is forbidden: {path}")
        if path.is_file():
            relative = path.relative_to(source).as_posix()
            output[f"{prefix}/{relative}"] = path
    if not output:
        raise E2ArtifactError(f"delivery directory is empty: {source}")
    return output


def _validated_delivery_members(
    token: AcceptedForFullFit,
    sources: E2DeliverySources,
) -> dict[str, Path]:
    if type(sources) is not E2DeliverySources or sources.token != token:
        raise E2ArtifactError("delivery token and source identity differ")
    if set(sources.models) != set(token.seeds):
        raise E2ArtifactError("delivery model seed identities differ")
    acceptance_sha = file_sha256(sources.acceptance_decision)
    if acceptance_sha != token.decision_sha256:
        raise E2ArtifactError("acceptance decision SHA-256 differs")
    audit = _read_json(sources.inference_audit, "inference audit")
    if audit.get("status") != "passed":
        raise E2ArtifactError("inference audit has not passed")
    full_fit = _read_json(sources.full_fit_manifest, "full-fit manifest")
    model_hashes = full_fit.get("model_sha256")
    expected_hashes = {
        str(seed): file_sha256(sources.models[seed]) for seed in token.seeds
    }
    if (
        full_fit.get("candidate_id") != token.candidate_id
        or full_fit.get("predictor") != token.predictor
        or model_hashes != expected_hashes
    ):
        raise E2ArtifactError("full-fit manifest differs")
    members = {
        f"models/catboost_seed_{seed}.cbm": sources.models[seed]
        for seed in token.seeds
    }
    members.update(_directory_members(sources.frozen_state, "frozen_state"))
    members.update(
        {
            "evidence/full_fit_manifest.json": sources.full_fit_manifest,
            "evidence/acceptance_decision.json": sources.acceptance_decision,
            "evidence/inference_audit.json": sources.inference_audit,
        }
    )
    if token.predictor == "blend":
        if sources.tabm_root is None:
            raise E2ArtifactError("accepted blend is missing frozen TabM")
        members.update(_directory_members(sources.tabm_root, "tabm"))
    elif sources.tabm_root is not None:
        raise E2ArtifactError("CatBoost-only delivery contains TabM")
    return members


def write_e2_delivery(
    token: AcceptedForFullFit,
    sources: E2DeliverySources,
    output: Path,
) -> Path:
    if type(token) is not AcceptedForFullFit:
        raise TypeError("token must be AcceptedForFullFit")
    members = _validated_delivery_members(token, sources)
    manifest = _json_bytes(
        {
            "schema_version": 1,
            "artifact_kind": "tree_expert_e2_model_delivery_v1",
            "campaign_id": "tree_expert_e2_v1",
            "review_only": False,
            "submission_package": False,
            "candidate_id": token.candidate_id,
            "predictor": token.predictor,
            "seeds": list(token.seeds),
            "iterations": {str(key): value for key, value in token.iterations.items()},
            "decision_sha256": token.decision_sha256,
            "members": {name: _evidence(value) for name, value in sorted(members.items())},
        }
    )
    output_path = _write_zip(output, {**members, "manifest.json": manifest})
    with ZipFile(output_path) as archive:
        names = _safe_names(archive)
        loaded = json.loads(archive.read("manifest.json"))
        _verify_manifest_members(archive, loaded, names, "manifest.json")
        if "script.py" in names or "submission.csv" in names:
            raise E2ArtifactError("delivery contains a submission entry point")
    return output_path


def _write_handoff(
    path: Path,
    *,
    review: Path,
    resume: Path,
    log: Path,
    status: str,
    delivery: Path | None,
) -> Path:
    members: dict[str, Path] = {
        "tree_expert_e2_review.zip": review,
        "tree_expert_e2_resume.zip": resume,
        "tree_expert_e2.log": log,
    }
    if delivery is not None:
        members["tree_expert_e2_model_delivery.zip"] = delivery
    manifest = _json_bytes(
        {
            "schema_version": 1,
            "artifact_kind": "tree_expert_e2_handoff_v1",
            "campaign_id": "tree_expert_e2_v1",
            "review_only": delivery is None,
            "submission_package": False,
            "status": status,
            "delivery": delivery is not None,
            "members": {name: _evidence(value) for name, value in sorted(members.items())},
        }
    )
    output = _write_zip(path, {**members, "handoff_manifest.json": manifest})
    verify_e2_handoff(output)
    return output


def verify_e2_handoff(path: Path) -> VerifiedE2Handoff:
    try:
        with ZipFile(_source(path)) as archive:
            names = _safe_names(archive)
            if "handoff_manifest.json" not in names:
                raise E2ArtifactError("handoff manifest is absent")
            manifest_bytes = archive.read("handoff_manifest.json")
            manifest = json.loads(manifest_bytes)
            expected = {
                "schema_version",
                "artifact_kind",
                "campaign_id",
                "review_only",
                "submission_package",
                "status",
                "delivery",
                "members",
            }
            if (
                type(manifest) is not dict
                or set(manifest) != expected
                or manifest["schema_version"] != 1
                or manifest["artifact_kind"] != "tree_expert_e2_handoff_v1"
                or manifest["campaign_id"] != "tree_expert_e2_v1"
                or manifest["submission_package"] is not False
                or type(manifest["delivery"]) is not bool
                or manifest["review_only"] is not (not manifest["delivery"])
            ):
                raise E2ArtifactError("handoff identity differs")
            expected_names = {
                "handoff_manifest.json",
                "tree_expert_e2_review.zip",
                "tree_expert_e2_resume.zip",
                "tree_expert_e2.log",
            }
            if manifest["delivery"]:
                expected_names.add("tree_expert_e2_model_delivery.zip")
            if set(names) != expected_names:
                raise E2ArtifactError("handoff members differ")
            _verify_manifest_members(archive, manifest, names, "handoff_manifest.json")
            return VerifiedE2Handoff(
                path=Path(path),
                manifest_sha256=sha256(manifest_bytes).hexdigest(),
                status=str(manifest["status"]),
                delivery=bool(manifest["delivery"]),
            )
    except E2ArtifactError:
        raise
    except Exception as error:
        raise E2ArtifactError(f"cannot verify E2 handoff: {error}") from error


def write_e2_bundles(
    *,
    output_dir: Path,
    bindings: Mapping[str, str],
    state_path: Path,
    log_path: Path,
    review_sources: Mapping[str, Path],
    resume_sources: Mapping[str, Path],
    delivery_sources: E2DeliverySources | None = None,
) -> E2BundlePaths:
    exact_bindings = _bindings(bindings)
    state = _state(state_path, exact_bindings)
    output = Path(output_dir)
    review_members = {"state/stage_state.json": _source(state_path)}
    review_members.update({name: _source(path) for name, path in review_sources.items()})
    resume_members = dict(review_members)
    resume_members.update({name: _source(path) for name, path in resume_sources.items()})
    review = _write_evidence_bundle(
        output / "tree_expert_e2_review.zip",
        kind="tree_expert_e2_review_v1",
        state=state,
        bindings=exact_bindings,
        members=review_members,
    )
    resume = _write_evidence_bundle(
        output / "tree_expert_e2_resume.zip",
        kind="tree_expert_e2_resume_v1",
        state=state,
        bindings=exact_bindings,
        members=resume_members,
    )
    delivery: Path | None = None
    if state["status"] == "accepted":
        if delivery_sources is None:
            raise E2ArtifactError("accepted state is missing delivery sources")
        delivery = write_e2_delivery(
            delivery_sources.token,
            delivery_sources,
            output / "tree_expert_e2_model_delivery.zip",
        )
    elif delivery_sources is not None:
        raise E2ArtifactError("non-accepted state cannot create delivery")
    handoff = _write_handoff(
        output / "tree_expert_e2_handoff.zip",
        review=review,
        resume=resume,
        log=_source(log_path),
        status=str(state["status"]),
        delivery=delivery,
    )
    return E2BundlePaths(
        review=review,
        resume=resume,
        handoff=handoff,
        delivery=delivery,
        review_sha256=file_sha256(review),
        resume_sha256=file_sha256(resume),
        handoff_sha256=file_sha256(handoff),
        delivery_sha256=file_sha256(delivery) if delivery is not None else None,
    )
