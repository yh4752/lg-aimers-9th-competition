from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Callable
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo


class T3ArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class T3Bindings:
    contract_sha256: str
    code_sha256: str
    input_manifest_sha256: str
    train_sha256: str
    history_sha256: str
    e2_handoff_sha256: str


_KINDS = {
    "resume": "tree_expert_t3_resume_v1",
    "review": "tree_expert_t3_review_v1",
    "model_delivery": "tree_expert_t3_model_delivery_v1",
    "handoff": "tree_expert_t3_handoff_v1",
}
_MAX_FILES = 2048
_MAX_EXPANDED = 4 * 1024 * 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _safe_infos(archive: ZipFile) -> dict[str, ZipInfo]:
    infos: dict[str, ZipInfo] = {}
    total = 0
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if (
            info.filename in infos or info.flag_bits & 0x1 or info.is_dir()
            or path.is_absolute() or ".." in path.parts or "\\" in info.filename
        ):
            raise T3ArtifactError("unsafe or duplicate archive member")
        total += info.file_size
        if len(infos) >= _MAX_FILES or total > _MAX_EXPANDED:
            raise T3ArtifactError("artifact expansion exceeds limit")
        infos[info.filename] = info
    return infos


def _source_files(root: Path, include: Callable[[str], bool]) -> dict[str, Path]:
    source = Path(root)
    if not source.is_dir():
        raise T3ArtifactError("artifact source directory is missing")
    files: dict[str, Path] = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        name = path.relative_to(source).as_posix()
        if include(name):
            files[name] = path
    if not files:
        raise T3ArtifactError("artifact source is empty")
    return files


def _write_bundle(
    source: Path,
    destination: Path,
    *,
    kind: str,
    bindings: T3Bindings,
    include: Callable[[str], bool],
    extra: dict[str, object] | None = None,
) -> Path:
    if kind not in _KINDS:
        raise T3ArtifactError("artifact kind differs")
    files = _source_files(source, include)
    members = {
        name: {"sha256": file_sha256(path), "size": path.stat().st_size}
        for name, path in files.items()
    }
    manifest = {
        "schema_version": 1, "artifact_kind": _KINDS[kind],
        "bindings": asdict(bindings), "members": members,
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
                archive.writestr(_zip_info(name), path.read_bytes())
            archive.writestr(_zip_info("manifest.json"), _canonical(manifest))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def create_resume_bundle(campaign_root: Path, destination: Path, bindings: T3Bindings) -> Path:
    return _write_bundle(
        campaign_root, destination, kind="resume", bindings=bindings,
        include=lambda name: not name.endswith(".zip") and not name.startswith("bundles/"),
    )


def create_review_bundle(campaign_root: Path, destination: Path, bindings: T3Bindings) -> Path:
    return _write_bundle(
        campaign_root, destination, kind="review", bindings=bindings,
        include=lambda name: (
            not name.endswith((".cbm", ".pt", ".zip"))
            and "/catboost_info/" not in f"/{name}/"
            and not name.startswith("bundles/")
        ),
    )


def create_model_delivery(full_fit_root: Path, destination: Path, bindings: T3Bindings) -> Path:
    return _write_bundle(
        full_fit_root, destination, kind="model_delivery", bindings=bindings,
        include=lambda name: not name.endswith(".zip"),
        extra={"candidate_id": "t3_temporal_dual", "status": "accepted"},
    )


def _verify(path: Path, kind: str, expected: T3Bindings) -> dict[str, object]:
    try:
        with ZipFile(path) as archive:
            infos = _safe_infos(archive)
            if "manifest.json" not in infos:
                raise T3ArtifactError("artifact manifest is missing")
            try:
                manifest = json.loads(archive.read(infos["manifest.json"]))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise T3ArtifactError("artifact manifest is invalid") from error
            if type(manifest) is not dict or manifest.get("artifact_kind") != _KINDS[kind]:
                raise T3ArtifactError("artifact identity differs")
            if manifest.get("bindings") != asdict(expected):
                raise T3ArtifactError("artifact bindings differ")
            members = manifest.get("members")
            if type(members) is not dict or set(members) != set(infos) - {"manifest.json"}:
                raise T3ArtifactError("artifact member set differs")
            for name, metadata in members.items():
                payload = archive.read(infos[name])
                if (
                    type(metadata) is not dict or metadata.get("size") != len(payload)
                    or metadata.get("sha256") != sha256(payload).hexdigest()
                ):
                    raise T3ArtifactError(f"member SHA-256 differs: {name}")
            return manifest
    except (OSError, BadZipFile) as error:
        raise T3ArtifactError("artifact is not a valid ZIP") from error


def verify_resume_bundle(path: Path, expected: T3Bindings) -> dict[str, object]:
    return _verify(Path(path), "resume", expected)


def verify_review_bundle(path: Path, expected: T3Bindings) -> dict[str, object]:
    return _verify(Path(path), "review", expected)


def verify_model_delivery(path: Path, expected: T3Bindings) -> dict[str, object]:
    return _verify(Path(path), "model_delivery", expected)


def restore_resume_bundle(path: Path, destination: Path, expected: T3Bindings) -> Path:
    verify_resume_bundle(path, expected)
    output = Path(destination)
    if output.exists() or output.is_symlink():
        raise T3ArtifactError("resume destination already exists")
    output.mkdir(parents=True)
    with ZipFile(path) as archive:
        infos = _safe_infos(archive)
        for name, info in infos.items():
            if name == "manifest.json":
                continue
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
    return output


def publish_stable_resume(campaign_root: Path, snapshot_root: Path, bindings: T3Bindings) -> Path:
    root = Path(snapshot_root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / ".next_resume.zip"
    create_resume_bundle(campaign_root, temporary, bindings)
    verify_resume_bundle(temporary, bindings)
    final = root / "tree_expert_t3_resume.zip"
    os.replace(temporary, final)
    for stale in root.glob("*.zip"):
        if stale != final:
            stale.unlink()
    return final


def create_handoff_bundle(
    *,
    review: Path,
    resume: Path,
    destination: Path,
    bindings: T3Bindings,
    model_delivery: Path | None = None,
    log: Path | None = None,
) -> Path:
    staging = Path(destination).with_name(f".{Path(destination).stem}_staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copy2(review, staging / "tree_expert_t3_review.zip")
    shutil.copy2(resume, staging / "tree_expert_t3_resume.zip")
    if model_delivery is not None:
        shutil.copy2(model_delivery, staging / "tree_expert_t3_model_delivery.zip")
    if log is not None:
        shutil.copy2(log, staging / "tree_expert_t3.log")
    try:
        return _write_bundle(
            staging, destination, kind="handoff", bindings=bindings,
            include=lambda _name: True,
            extra={"status": "completed", "delivery": model_delivery is not None},
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
