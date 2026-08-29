from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

from .s4_state import load_s4_state


class S4ArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class S4Bindings:
    contract_sha256: str
    code_sha256: str
    input_manifest_sha256: str
    train_sha256: str
    history_sha256: str
    e2_handoff_sha256: str


_KINDS = {
    "resume": "tree_s4_resume_v1",
    "review": "tree_s4_review_v1",
    "delivery": "tree_s4_model_delivery_v1",
    "handoff": "tree_s4_handoff_v1",
}
_MAX_FILES = 4096
_MAX_BYTES = 12 * 1024 * 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _validate_bindings(bindings: S4Bindings) -> None:
    if type(bindings) is not S4Bindings or any(
        type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in asdict(bindings).values()
    ):
        raise S4ArtifactError("artifact bindings differ")


def _write_payloads(
    payloads: Mapping[str, bytes], destination: Path, kind: str, bindings: S4Bindings,
    *, extra: Mapping[str, object] | None = None,
) -> Path:
    _validate_bindings(bindings)
    if kind not in _KINDS or not payloads or "manifest.json" in payloads:
        raise S4ArtifactError("artifact payload differs")
    names = sorted(payloads)
    if len(names) > _MAX_FILES or sum(len(payloads[name]) for name in names) > _MAX_BYTES:
        raise S4ArtifactError("artifact expansion exceeds limit")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or not name:
            raise S4ArtifactError("unsafe artifact member")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": _KINDS[kind],
        "bindings": asdict(bindings),
        "members": {
            name: {"sha256": sha256(payloads[name]).hexdigest(), "size": len(payloads[name])}
            for name in names
        },
    }
    if extra:
        for key, value in extra.items():
            if key in manifest:
                raise S4ArtifactError("artifact metadata collides")
            manifest[key] = value
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with ZipFile(temporary, "w") as archive:
            for name in names:
                archive.writestr(_zip_info(name), payloads[name])
            archive.writestr(_zip_info("manifest.json"), _canonical(manifest))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _root_payloads(root: Path, include) -> dict[str, bytes]:
    source = Path(root)
    if not source.is_dir():
        raise S4ArtifactError("campaign root differs")
    output = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise S4ArtifactError("campaign source contains a symlink")
        if not path.is_file():
            continue
        name = path.relative_to(source).as_posix()
        if include(name):
            output[name] = path.read_bytes()
    if not output:
        raise S4ArtifactError("campaign source is empty")
    return output


def create_s4_resume(root: Path, destination: Path, bindings: S4Bindings) -> Path:
    state = load_s4_state(Path(root) / "state/state.json")
    completed = set(state.completed_jobs)

    def include(name: str) -> bool:
        parts = PurePosixPath(name).parts
        if name.startswith("bundles/") or name.endswith(".tmp"):
            return False
        if name == "verified_e2_input.zip" or name.startswith("verified_e2/"):
            return False
        if parts and parts[0] == "jobs":
            return (
                len(parts) >= 3
                and parts[1] in completed
                and parts[-1] == "result.json"
            )
        if parts and parts[0] == "anchor_basis" and state.phase != "anchors":
            return False
        return True

    return _write_payloads(_root_payloads(root, include), destination, "resume", bindings)


def create_s4_review(root: Path, destination: Path, bindings: S4Bindings) -> Path:
    def include(name: str) -> bool:
        parts = PurePosixPath(name).parts
        if not parts or name.endswith((".zip", ".tmp", ".cbm", ".pt", ".bin", ".pkl", ".joblib")):
            return False
        if name in {"s4_campaign.log"}:
            return True
        if parts[0] in {"state", "diagnostics", "decisions", "confirmation"}:
            return True
        if parts[0] == "full_chains":
            return parts[-1] == "config.json"
        if parts[0] == "jobs":
            return parts[-1] == "result.json"
        return False

    return _write_payloads(
        _root_payloads(root, include),
        destination,
        "review",
        bindings,
    )


def _accepted_token(root: Path) -> bytes:
    path = Path(root) / "accepted/token.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise S4ArtifactError("accepted token is required") from error
    if type(payload) is not dict or payload.get("status") != "accepted" or type(payload.get("candidate_id")) is not str:
        raise S4ArtifactError("accepted token is required")
    return _canonical(payload)


def create_s4_model_delivery(root: Path, destination: Path, bindings: S4Bindings) -> Path:
    token = _accepted_token(root)
    models = _root_payloads(Path(root) / "models", lambda _name: True)
    payloads = {f"models/{name}": value for name, value in models.items()}
    payloads["accepted/token.json"] = token
    return _write_payloads(payloads, destination, "delivery", bindings, extra={"delivery": True})


def _safe_infos(archive: ZipFile) -> dict[str, ZipInfo]:
    infos: dict[str, ZipInfo] = {}
    total = 0
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if info.filename in infos or info.is_dir() or info.flag_bits & 1 or path.is_absolute() or ".." in path.parts or "\\" in info.filename:
            raise S4ArtifactError("unsafe or duplicate archive member")
        total += info.file_size
        if len(infos) >= _MAX_FILES or total > _MAX_BYTES:
            raise S4ArtifactError("artifact expansion exceeds limit")
        infos[info.filename] = info
    return infos


def _verify(path: Path, kind: str, bindings: S4Bindings) -> dict[str, object]:
    _validate_bindings(bindings)
    try:
        with ZipFile(path) as archive:
            infos = _safe_infos(archive)
            if "manifest.json" not in infos:
                raise S4ArtifactError("artifact manifest is missing")
            manifest = json.loads(archive.read(infos["manifest.json"]))
            if type(manifest) is not dict or manifest.get("artifact_kind") != _KINDS[kind]:
                raise S4ArtifactError("artifact identity differs")
            if manifest.get("bindings") != asdict(bindings):
                raise S4ArtifactError("artifact bindings differ")
            members = manifest.get("members")
            if type(members) is not dict or set(members) != set(infos) - {"manifest.json"}:
                raise S4ArtifactError("artifact member set differs")
            for name, metadata in members.items():
                payload = archive.read(infos[name])
                if metadata != {"sha256": sha256(payload).hexdigest(), "size": len(payload)}:
                    raise S4ArtifactError(f"member SHA-256 differs: {name}")
            return manifest
    except (OSError, BadZipFile, json.JSONDecodeError) as error:
        if isinstance(error, S4ArtifactError):
            raise
        raise S4ArtifactError("artifact is not a valid ZIP") from error


def verify_s4_resume(path: Path, bindings: S4Bindings) -> dict[str, object]:
    return _verify(path, "resume", bindings)


def verify_s4_review(path: Path, bindings: S4Bindings) -> dict[str, object]:
    return _verify(path, "review", bindings)


def verify_s4_handoff(path: Path, bindings: S4Bindings) -> dict[str, object]:
    return _verify(path, "handoff", bindings)


def restore_s4_resume(path: Path, destination: Path, bindings: S4Bindings) -> Path:
    verify_s4_resume(path, bindings)
    output = Path(destination)
    if output.exists() or output.is_symlink():
        raise S4ArtifactError("resume destination already exists")
    output.mkdir(parents=True)
    with ZipFile(path) as archive:
        for name, info in _safe_infos(archive).items():
            if name == "manifest.json":
                continue
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
    return output


def create_s4_handoff(
    root: Path,
    destination: Path,
    bindings: S4Bindings,
    *,
    include_delivery: bool = False,
    log_path: Path | None = None,
) -> Path:
    source = Path(root)
    bundles = source / "bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    review = create_s4_review(source, bundles / "review.zip", bindings)
    resume = create_s4_resume(source, bundles / "resume.zip", bindings)
    payloads = {"review.zip": review.read_bytes(), "resume.zip": resume.read_bytes()}
    status = "review_ready"
    if include_delivery:
        delivery = create_s4_model_delivery(source, bundles / "model_delivery.zip", bindings)
        payloads["model_delivery.zip"] = delivery.read_bytes()
        status = "delivery_ready"
    else:
        state = load_s4_state(source / "state/state.json")
        if any(value == "accepted" for value in state.decisions.values()):
            status = "accepted_review_ready"
        elif state.phase == "completed":
            status = "completed_no_candidate"
    if log_path is not None:
        payloads["campaign.log"] = Path(log_path).read_bytes()
    return _write_payloads(
        payloads,
        destination,
        "handoff",
        bindings,
        extra={"status": status, "delivery": include_delivery, "submission_package": False},
    )
