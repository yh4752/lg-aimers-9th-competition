"""Restartable, hash-bound full row-independence evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Callable

from .code_gate import RulesCodeGateError, assert_row_independent


class RulesEvidenceError(ValueError):
    """Raised when full audit evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class AuditIdentity:
    policy_version: str
    candidate_id: str
    data_sha256: str
    code_sha256: str
    config_sha256: str
    preprocessing_sha256: str
    model_sha256: str
    adapter_sha256: str
    runtime_sha256: str

    def __post_init__(self) -> None:
        if not self.policy_version or not self.candidate_id:
            raise RulesEvidenceError("audit identity names are required")
        for name, value in asdict(self).items():
            if name.endswith("sha256") and (
                len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            ):
                raise RulesEvidenceError(f"{name} is not a lowercase SHA-256")


@dataclass(frozen=True)
class FullAuditResult:
    status: str
    row_count: int
    completed_chunks: int
    reused_chunks: int
    manifest_path: Path


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_new(path: Path, payload: object) -> None:
    """Publish one small evidence file without ever replacing a destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical(payload)
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise RulesEvidenceError(f"audit evidence already exists: {path}") from error


def _load(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise RulesEvidenceError(f"invalid audit evidence: {path}") from error
    if not isinstance(payload, dict):
        raise RulesEvidenceError("audit evidence must be an object")
    return payload


def _chunk_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("chunk_sha256", None)
    return sha256(_canonical(body)).hexdigest()


def run_full_independence_audit(
    *, frame: "object", output_dir: str | Path, identity: AuditIdentity,
    load_predictor: Callable[[], object], chunk_size: int = 4096,
    stop_after_chunks: int | None = None,
) -> FullAuditResult:
    import pandas as pd

    if not isinstance(frame, pd.DataFrame) or frame.empty or "row_id" not in frame:
        raise RulesEvidenceError("audit frame must be non-empty with row_id")
    ids = frame["row_id"].astype("string")
    if ids.isna().any() or ids.astype(str).duplicated().any():
        raise RulesEvidenceError("audit row_id must be non-null and unique")
    if type(chunk_size) is not int or chunk_size < 1:
        raise RulesEvidenceError("chunk_size must be positive")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    expected = (len(frame) + chunk_size - 1) // chunk_size
    expected_names = {f"chunk-{index:06d}.json" for index in range(expected)}
    found_names = {path.name for path in root.glob("chunk-*.json")}
    unexpected = found_names - expected_names
    if unexpected:
        raise RulesEvidenceError(
            "unexpected audit chunks: " + ", ".join(sorted(unexpected))
        )
    reused = 0
    completed = 0
    chunk_entries = []
    for index, start in enumerate(range(0, len(frame), chunk_size)):
        part = frame.iloc[start : start + chunk_size].reset_index(drop=True)
        path = root / f"chunk-{index:06d}.json"
        id_digest = sha256("\n".join(part["row_id"].astype(str)).encode()).hexdigest()
        if path.exists():
            payload = _load(path)
            if payload.get("identity") != asdict(identity) or payload.get("row_id_sha256") != id_digest:
                raise RulesEvidenceError("existing audit chunk identity differs")
            if payload.get("chunk_sha256") != _chunk_hash(payload):
                raise RulesEvidenceError("existing audit chunk hash differs")
            reused += 1
        else:
            try:
                report = assert_row_independent(part, load_predictor=load_predictor)
            except RulesCodeGateError as error:
                raise RulesEvidenceError(str(error)) from error
            payload = {
                "schema_version": 1, "chunk_index": index,
                "identity": asdict(identity), "row_count": len(part),
                "row_id_sha256": id_digest,
                "prediction_sha256": report["prediction_sha256"],
                "variant_count": report["variant_count"],
                "status": "passed",
            }
            payload["chunk_sha256"] = _chunk_hash(payload)
            _write_new(path, payload)
        completed += 1
        chunk_entries.append({"path": path.name, "sha256": payload["chunk_sha256"]})
        if stop_after_chunks is not None and completed >= stop_after_chunks:
            return FullAuditResult("incomplete", len(frame), completed, reused, root / "full_audit_manifest.json")
    manifest = {
        "schema_version": 1, "status": "passed", "identity": asdict(identity),
        "row_count": len(frame), "chunk_size": chunk_size,
        "chunk_count": expected, "chunks": chunk_entries,
    }
    manifest_path = root / "full_audit_manifest.json"
    if manifest_path.exists():
        if _load(manifest_path) != manifest:
            raise RulesEvidenceError("existing full audit manifest differs")
    else:
        _write_new(manifest_path, manifest)
    validate_full_audit(manifest_path, expected_identity=identity)
    return FullAuditResult("passed", len(frame), completed, reused, manifest_path)


def validate_full_audit(
    manifest_path: str | Path, *, expected_identity: AuditIdentity
) -> dict[str, object]:
    path = Path(manifest_path)
    manifest = _load(path)
    if manifest.get("status") != "passed" or manifest.get("identity") != asdict(expected_identity):
        raise RulesEvidenceError("full audit identity or status differs")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != manifest.get("chunk_count"):
        raise RulesEvidenceError("full audit chunks are incomplete")
    names = []
    rows = 0
    for index, entry in enumerate(chunks):
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise RulesEvidenceError("full audit chunk entry is invalid")
        chunk_path = path.parent / str(entry["path"])
        if chunk_path.name != f"chunk-{index:06d}.json":
            raise RulesEvidenceError("audit chunk order is invalid")
        payload = _load(chunk_path)
        if payload.get("identity") != asdict(expected_identity):
            raise RulesEvidenceError("audit chunk identity differs")
        if payload.get("chunk_sha256") != entry["sha256"] or _chunk_hash(payload) != entry["sha256"]:
            raise RulesEvidenceError("audit chunk hash differs")
        if payload.get("status") != "passed" or payload.get("chunk_index") != index:
            raise RulesEvidenceError("audit chunk status or index differs")
        if not isinstance(payload.get("variant_count"), int) or payload["variant_count"] < 2:
            raise RulesEvidenceError("audit chunk variants are incomplete")
        names.append(entry["path"])
        rows += int(payload.get("row_count", -1))
    if len(set(names)) != len(names) or rows != manifest.get("row_count"):
        raise RulesEvidenceError("audit chunks duplicate or omit rows")
    return manifest
