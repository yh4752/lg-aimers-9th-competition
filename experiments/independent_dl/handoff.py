"""Create a small, hash-bound candidate result bundle for human review."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Mapping
import zipfile


class HandoffError(ValueError):
    """Raised before an untrusted or incomplete handoff can be published."""


@dataclass(frozen=True)
class HandoffResult:
    path: Path
    size_bytes: int
    sha256: str


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HandoffError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise HandoffError(f"non-finite JSON number: {value}")


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HandoffError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise HandoffError(f"{label} must be a JSON object")
    return value


def _digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _digest_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise HandoffError(f"{label} is not a readable regular file: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise HandoffError(f"{label} must be a regular file: {path}")
    return path.resolve()


def _campaign_artifact(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise HandoffError(f"{label} path is missing")
    raw = root / value
    if raw.is_symlink():
        raise HandoffError(f"{label} must be a regular file: {raw}")
    path = _regular_file(raw, label)
    if path == root or root not in path.parents:
        raise HandoffError(f"{label} escapes campaign root")
    return path


def _member(value: bytes) -> dict[str, object]:
    return {"sha256": _digest_bytes(value), "size_bytes": len(value)}


def write_candidate_handoff(
    campaign_root: str | Path,
    candidate_id: str,
    result_path: str | Path,
    runtime_sha256: str,
    requirements_path: str | Path,
    environment_path: str | Path,
) -> HandoffResult:
    root = Path(campaign_root).resolve()
    result = Path(result_path).resolve()
    if result.exists():
        raise HandoffError(f"result already exists: {result}")
    if (
        len(runtime_sha256) != 64
        or runtime_sha256.lower() != runtime_sha256
        or any(character not in "0123456789abcdef" for character in runtime_sha256)
    ):
        raise HandoffError("runtime_sha256 must be 64 lowercase hexadecimal characters")

    manifest_path = _regular_file(root / "campaign_manifest.json", "campaign manifest")
    manifest = _read_json(manifest_path, "campaign manifest")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, dict) or candidate_id not in candidates:
        raise HandoffError(f"candidate_id is not registered: {candidate_id}")
    entry = candidates[candidate_id]
    if not isinstance(entry, dict) or entry.get("state") != "completed":
        raise HandoffError(f"candidate must be completed: {candidate_id}")

    metrics_path = _campaign_artifact(root, entry.get("metrics_path"), "metrics")
    predictions_path = _campaign_artifact(
        root, entry.get("predictions_path"), "predictions"
    )
    requirements = _regular_file(Path(requirements_path), "requirements")
    environment = _regular_file(Path(environment_path), "environment")
    environment_payload = _read_json(environment, "environment")

    metrics_bytes = metrics_path.read_bytes()
    predictions_bytes = predictions_path.read_bytes()
    requirements_bytes = requirements.read_bytes()
    environment_bytes = environment.read_bytes()
    expected_metrics = entry.get("metrics_sha256")
    expected_predictions = entry.get("predictions_sha256")
    if expected_metrics != _digest_bytes(metrics_bytes):
        raise HandoffError("metrics SHA-256 does not match campaign manifest")
    if expected_predictions != _digest_bytes(predictions_bytes):
        raise HandoffError("predictions SHA-256 does not match campaign manifest")

    entry_bytes = (
        json.dumps(entry, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    members = {
        "campaign_entry.json": entry_bytes,
        "metrics.json": metrics_bytes,
        "predictions.csv": predictions_bytes,
        "requirements-colab.txt": requirements_bytes,
        "environment.json": environment_bytes,
    }
    metrics_payload = _read_json(metrics_path, "metrics")
    handoff_manifest = {
        "schema_version": 1,
        "status": "handoff_ready",
        "candidate_id": candidate_id,
        "campaign_id": manifest.get("campaign_id"),
        "protocol": manifest.get("protocol"),
        "runtime_sha256": runtime_sha256,
        "requirements_sha256": _digest_bytes(requirements_bytes),
        "environment_sha256": _digest_bytes(environment_bytes),
        "environment": environment_payload,
        "hardware": metrics_payload.get("hardware"),
        "members": {name: _member(value) for name, value in members.items()},
    }
    handoff_bytes = (
        json.dumps(handoff_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")

    result.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(
            result, mode="x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as bundle:
            for name, value in members.items():
                bundle.writestr(name, value)
            bundle.writestr("handoff_manifest.json", handoff_bytes)
    except Exception:
        if result.exists():
            result.unlink()
        raise
    return HandoffResult(result, result.stat().st_size, _digest_file(result))
