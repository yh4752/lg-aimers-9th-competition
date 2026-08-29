from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Callable
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo


class HeteroArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class HeteroBindings:
    contract_sha256: str
    code_sha256: str
    input_manifest_sha256: str
    train_sha256: str
    history_sha256: str
    e2_handoff_sha256: str


_KINDS = {
    "resume": "tree_hetero_residual_resume_v1",
    "review": "tree_hetero_residual_review_v1",
    "handoff": "tree_hetero_residual_handoff_v1",
}
_MAX_FILES = 2048
_MAX_EXPANDED = 8 * 1024 * 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _info(name: str) -> ZipInfo:
    value = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
    value.compress_type = ZIP_DEFLATED
    value.external_attr = 0o100644 << 16
    return value


def _safe_infos(archive: ZipFile) -> dict[str, ZipInfo]:
    output: dict[str, ZipInfo] = {}
    total = 0
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if (
            info.filename in output or info.is_dir() or info.flag_bits & 1
            or path.is_absolute() or ".." in path.parts or "\\" in info.filename
        ):
            raise HeteroArtifactError("unsafe or duplicate archive member")
        total += info.file_size
        if len(output) >= _MAX_FILES or total > _MAX_EXPANDED:
            raise HeteroArtifactError("artifact expansion exceeds limit")
        output[info.filename] = info
    return output


def _write(
    source: Path,
    destination: Path,
    kind: str,
    bindings: HeteroBindings,
    include: Callable[[str], bool],
    extra: dict[str, object] | None = None,
) -> Path:
    if kind not in _KINDS or not Path(source).is_dir():
        raise HeteroArtifactError("artifact source differs")
    files = {
        path.relative_to(source).as_posix(): path
        for path in sorted(Path(source).rglob("*"))
        if path.is_file() and not path.is_symlink()
        and include(path.relative_to(source).as_posix())
    }
    if not files:
        raise HeteroArtifactError("artifact source is empty")
    manifest: dict[str, object] = {
        "schema_version": 1, "artifact_kind": _KINDS[kind],
        "bindings": asdict(bindings),
        "members": {
            name: {"sha256": file_sha256(path), "size": path.stat().st_size}
            for name, path in files.items()
        },
    }
    if extra:
        manifest.update(extra)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with ZipFile(temporary, "w") as archive:
            for name, path in files.items():
                archive.writestr(_info(name), path.read_bytes())
            archive.writestr(_info("manifest.json"), _canonical(manifest))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _state_completed(root: Path) -> frozenset[str]:
    try:
        state = json.loads((Path(root) / "state/state.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise HeteroArtifactError("campaign state cannot be loaded") from error
    if type(state) is not dict or type(state.get("completed_jobs")) is not list:
        raise HeteroArtifactError("campaign state schema differs")
    return frozenset(state["completed_jobs"])


def create_resume_bundle(root: Path, destination: Path, bindings: HeteroBindings) -> Path:
    completed = _state_completed(root)
    def include(name: str) -> bool:
        parts = PurePosixPath(name).parts
        if name.endswith((".zip", ".tmp")) or name.startswith("bundles/"):
            return False
        return not parts or parts[0] != "jobs" or (len(parts) >= 3 and parts[1] in completed)
    return _write(root, destination, "resume", bindings, include)


def create_review_bundle(root: Path, destination: Path, bindings: HeteroBindings) -> Path:
    def include(name: str) -> bool:
        path = PurePosixPath(name)
        return (
            not name.startswith("bundles/") and not name.endswith((".zip", ".tmp", ".cbm", ".pt"))
            and path.name not in {"model.json", "model.txt"}
        )
    return _write(root, destination, "review", bindings, include)


def _verify(
    path: Path,
    kind: str,
    bindings: HeteroBindings,
    *,
    compatible_code_sha256s: frozenset[str] = frozenset(),
) -> dict[str, object]:
    try:
        with ZipFile(path) as archive:
            infos = _safe_infos(archive)
            if "manifest.json" not in infos:
                raise HeteroArtifactError("artifact manifest is missing")
            manifest = json.loads(archive.read(infos["manifest.json"]))
            if type(manifest) is not dict or manifest.get("artifact_kind") != _KINDS[kind]:
                raise HeteroArtifactError("artifact identity differs")
            actual_bindings = manifest.get("bindings")
            expected_bindings = asdict(bindings)
            if actual_bindings != expected_bindings:
                compatible = False
                if kind == "resume" and type(actual_bindings) is dict:
                    legacy_code = actual_bindings.get("code_sha256")
                    rebound = dict(actual_bindings)
                    rebound["code_sha256"] = bindings.code_sha256
                    compatible = (
                        legacy_code in compatible_code_sha256s
                        and rebound == expected_bindings
                    )
                if not compatible:
                    raise HeteroArtifactError("artifact bindings differ")
            members = manifest.get("members")
            if type(members) is not dict or set(members) != set(infos) - {"manifest.json"}:
                raise HeteroArtifactError("artifact member set differs")
            for name, metadata in members.items():
                payload = archive.read(infos[name])
                if metadata != {"sha256": sha256(payload).hexdigest(), "size": len(payload)}:
                    raise HeteroArtifactError(f"member SHA-256 differs: {name}")
            return manifest
    except (OSError, BadZipFile, json.JSONDecodeError) as error:
        if isinstance(error, HeteroArtifactError):
            raise
        raise HeteroArtifactError("artifact is not a valid ZIP") from error


def verify_resume_bundle(
    path: Path,
    bindings: HeteroBindings,
    *,
    compatible_code_sha256s: frozenset[str] = frozenset(),
) -> dict[str, object]:
    return _verify(
        path,
        "resume",
        bindings,
        compatible_code_sha256s=compatible_code_sha256s,
    )


def verify_review_bundle(path: Path, bindings: HeteroBindings) -> dict[str, object]:
    return _verify(path, "review", bindings)


def verify_handoff_bundle(path: Path, bindings: HeteroBindings) -> dict[str, object]:
    return _verify(path, "handoff", bindings)


def restore_resume_bundle(
    path: Path,
    destination: Path,
    bindings: HeteroBindings,
    *,
    compatible_code_sha256s: frozenset[str] = frozenset(),
) -> Path:
    verify_resume_bundle(
        path,
        bindings,
        compatible_code_sha256s=compatible_code_sha256s,
    )
    output = Path(destination)
    if output.exists() or output.is_symlink():
        raise HeteroArtifactError("resume destination already exists")
    output.mkdir(parents=True)
    with ZipFile(path) as archive:
        for name, info in _safe_infos(archive).items():
            if name == "manifest.json":
                continue
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
    return output


def create_handoff_bundle(
    review: Path,
    resume: Path,
    destination: Path,
    bindings: HeteroBindings,
    log: Path | None = None,
) -> Path:
    staging = Path(destination).with_name(f".{Path(destination).stem}_staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        shutil.copy2(review, staging / "tree_hetero_review.zip")
        shutil.copy2(resume, staging / "tree_hetero_resume.zip")
        if log is not None:
            shutil.copy2(log, staging / "tree_hetero.log")
        return _write(
            staging, destination, "handoff", bindings, lambda _name: True,
            {"status": "review_ready", "delivery": False, "review_only": True, "submission_package": False},
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
