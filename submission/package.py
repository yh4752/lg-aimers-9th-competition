"""The sole writer for the official script.py/requirements.txt/model archive."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import zipfile

from .audit import AuditSnapshot, SubmissionAuditError, audit_package_request
from .contract import PackageRequest, PackageResult
from .runtime import render_script


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as output, source.open("rb") as incoming:
            for block in iter(lambda: incoming.read(1024 * 1024), b""):
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError as error:
        raise SubmissionAuditError(f"output already exists: {destination}") from error


def _verify_layout(path: Path, expected: list[str]) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if names != expected or len(set(names)) != len(names):
                raise SubmissionAuditError("archive layout differs from audited inputs")
            if any(
                name.startswith("/") or ".." in Path(name).parts
                for name in names
            ):
                raise SubmissionAuditError("archive contains an unsafe path")
            bad = archive.testzip()
            if bad is not None:
                raise SubmissionAuditError(f"archive member is corrupt: {bad}")
    except zipfile.BadZipFile as error:
        raise SubmissionAuditError("created archive is invalid") from error


def build_submission_package(request: PackageRequest) -> PackageResult:
    """Audit first, then deterministically create one local archive and receipt."""

    snapshot: AuditSnapshot = audit_package_request(request)
    script = render_script(
        adapter_id=request.adapter_id,
        artifact_metadata={
            "candidate_id": snapshot.identity.candidate_id,
            "model_sha256": snapshot.model_sha256,
            "adapter_sha256": snapshot.identity.adapter_sha256,
            "runtime_sha256": snapshot.identity.runtime_sha256,
        },
    )
    if not isinstance(script, bytes) or not script:
        raise SubmissionAuditError("rendered script is empty")
    temporary = request.archive_path.with_name(
        request.archive_path.name + f".{os.getpid()}.partial"
    )
    expected = [
        "script.py",
        "requirements.txt",
        *(f"model/{name}" for name, _ in snapshot.model_files),
    ]
    archive_published = False
    archive_digest = ""
    try:
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(temporary, "x") as archive:
            archive.writestr(_zip_info("script.py"), script)
            archive.writestr(_zip_info("requirements.txt"), snapshot.requirements_bytes)
            for name, data in snapshot.model_files:
                archive.writestr(_zip_info(f"model/{name}"), data)
        _verify_layout(temporary, expected)
        archive_digest = sha256(temporary.read_bytes()).hexdigest()
        if temporary.stat().st_size > snapshot.policy["limits"]["package_bytes"]:
            raise SubmissionAuditError("created package exceeds size limit")
        _copy_exclusive(temporary, request.archive_path)
        archive_published = True
        if sha256(request.archive_path.read_bytes()).hexdigest() != archive_digest:
            raise SubmissionAuditError("published archive hash differs")
        receipt = {
            "schema_version": 1,
            "status": "packaged",
            "candidate_id": snapshot.identity.candidate_id,
            "policy_version": snapshot.identity.policy_version,
            "adapter_id": request.adapter_id,
            "archive_sha256": archive_digest,
            "archive_bytes": request.archive_path.stat().st_size,
            "identity": snapshot.acceptance["identity"],
        }
        receipt_temp = temporary.with_name(temporary.name + ".receipt")
        receipt_temp.write_text(
            json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _copy_exclusive(receipt_temp, request.receipt_path)
        receipt_temp.unlink()
    except Exception:
        if archive_published and request.archive_path.exists():
            try:
                if sha256(request.archive_path.read_bytes()).hexdigest() == archive_digest:
                    request.archive_path.unlink()
            except OSError:
                pass
        raise
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return PackageResult(
        archive_path=request.archive_path,
        receipt_path=request.receipt_path,
        archive_sha256=archive_digest,
        archive_bytes=request.archive_path.stat().st_size,
    )
