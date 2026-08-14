"""Restartable, hash-bound full row-independence evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from time import monotonic
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


@dataclass(frozen=True)
class PhasedAuditResult:
    status: str
    row_count: int
    completed_phases: int
    reused_phases: int
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
    if manifest.get("schema_version") == 2:
        return _validate_phased_audit(path, manifest, expected_identity)
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


_PHASES = (
    "baseline",
    "reverse",
    "shuffle",
    "batch_257",
    "batch_2048",
    "singleton_canaries",
)


def _phase_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("phase_sha256", None)
    return sha256(_canonical(body)).hexdigest()


def _row_hashes(
    frame: "object", x_num: object, x_cat: object
) -> dict[str, str]:
    import numpy as np

    numeric = np.asarray(x_num)
    categorical = np.asarray(x_cat)
    if (
        numeric.ndim != 2
        or categorical.ndim != 2
        or len(numeric) != len(frame)
        or len(categorical) != len(frame)
    ):
        raise RulesEvidenceError("encoded features must be aligned matrices")
    output: dict[str, str] = {}
    for index, row_id in enumerate(frame["row_id"].astype(str)):
        digest = sha256()
        digest.update(numeric[index].tobytes())
        digest.update(categorical[index].tobytes())
        output[row_id] = digest.hexdigest()
    return output


def _phase_payload(
    *,
    phase: str,
    frame: "object",
    identity: AuditIdentity,
    load_predictor: Callable[[], object],
) -> dict[str, object]:
    import numpy as np

    predictor = load_predictor()
    digest = getattr(predictor, "state_digest", None)
    encode = getattr(predictor, "encode", None)
    predict = getattr(predictor, "predict_batch", None)
    if not callable(digest) or not callable(encode) or not callable(predict):
        raise RulesEvidenceError("phased audit predictor contract differs")
    before = str(digest())
    started = monotonic()
    if phase == "baseline":
        selected = frame.reset_index(drop=True)
        values = np.asarray(predict(selected, batch_size=2048), dtype="float64")
    elif phase == "reverse":
        selected = frame.iloc[::-1].reset_index(drop=True)
        values = np.asarray(predict(selected, batch_size=2048), dtype="float64")
    elif phase == "shuffle":
        order = sorted(
            range(len(frame)),
            key=lambda index: sha256(
                (identity.runtime_sha256 + "\0" + str(frame.iloc[index]["row_id"])).encode()
            ).hexdigest(),
        )
        selected = frame.iloc[order].reset_index(drop=True)
        values = np.asarray(predict(selected, batch_size=2048), dtype="float64")
    elif phase in {"batch_257", "batch_2048"}:
        selected = frame.reset_index(drop=True)
        size = int(phase.split("_", 1)[1])
        parts = [
            np.asarray(
                predict(selected.iloc[start : start + size].reset_index(drop=True), batch_size=size),
                dtype="float64",
            )
            for start in range(0, len(selected), size)
        ]
        values = np.concatenate(parts)
    else:
        selected = frame.reset_index(drop=True)
        parts = [
            np.asarray(predict(selected.iloc[[index]].reset_index(drop=True), batch_size=1), dtype="float64")
            for index in range(len(selected))
        ]
        values = np.concatenate(parts)
    if values.shape != (len(selected),) or not np.isfinite(values).all():
        raise RulesEvidenceError("phased audit predictions are not finite and aligned")
    if ((values < 0) | (values > 1)).any():
        raise RulesEvidenceError("phased audit predictions are outside [0, 1]")
    x_num, x_cat = encode(selected.copy(deep=True))
    after = str(digest())
    if before != after:
        raise RulesEvidenceError("predictor state changed during phased audit")
    ids = selected["row_id"].astype(str).tolist()
    payload: dict[str, object] = {
        "schema_version": 2,
        "status": "passed",
        "identity": asdict(identity),
        "phase": phase,
        "row_count": len(selected),
        "row_id_sha256": sha256("\n".join(ids).encode()).hexdigest(),
        "predictions": dict(zip(ids, (float(item) for item in values), strict=True)),
        "feature_sha256": _row_hashes(selected, x_num, x_cat),
        "state_sha256_before": before,
        "state_sha256_after": after,
        "elapsed_seconds": monotonic() - started,
    }
    payload["phase_sha256"] = _phase_hash(payload)
    return payload


def _load_phase(
    path: Path,
    *,
    phase: str,
    identity: AuditIdentity,
) -> dict[str, object]:
    payload = _load(path)
    if (
        payload.get("schema_version") != 2
        or payload.get("status") != "passed"
        or payload.get("identity") != asdict(identity)
        or payload.get("phase") != phase
        or payload.get("phase_sha256") != _phase_hash(payload)
    ):
        raise RulesEvidenceError(f"existing audit phase differs: {phase}")
    if payload.get("state_sha256_before") != payload.get("state_sha256_after"):
        raise RulesEvidenceError(f"audit phase state changed: {phase}")
    predictions = payload.get("predictions")
    features = payload.get("feature_sha256")
    if not isinstance(predictions, dict) or not isinstance(features, dict):
        raise RulesEvidenceError(f"audit phase evidence is incomplete: {phase}")
    if set(predictions) != set(features) or len(predictions) != payload.get("row_count"):
        raise RulesEvidenceError(f"audit phase rows differ: {phase}")
    return payload


def run_phased_independence_audit(
    *,
    frame: "object",
    output_dir: str | Path,
    identity: AuditIdentity,
    load_predictor: Callable[[], object],
    singleton_count: int,
    stop_after_phases: int | None = None,
    tolerance: float = 1e-6,
) -> PhasedAuditResult:
    """Audit every sample row under reorder, rebatch, and singleton variants."""

    import pandas as pd

    if not isinstance(frame, pd.DataFrame) or frame.empty or "row_id" not in frame:
        raise RulesEvidenceError("phased audit frame must be non-empty with row_id")
    ids = frame["row_id"].astype("string")
    if ids.isna().any() or ids.astype(str).duplicated().any():
        raise RulesEvidenceError("phased audit row_id must be non-null and unique")
    if type(singleton_count) is not int or singleton_count < 1:
        raise RulesEvidenceError("singleton_count must be positive")
    if singleton_count != len(frame):
        raise RulesEvidenceError("all official sample rows must be singleton-audited")
    if not isinstance(tolerance, float) or tolerance <= 0:
        raise RulesEvidenceError("audit tolerance must be positive")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    source = frame.copy(deep=True).reset_index(drop=True)
    source["row_id"] = ids.astype(str).to_numpy()
    completed = 0
    reused = 0
    payloads: list[dict[str, object]] = []
    entries: list[dict[str, str]] = []
    for phase in _PHASES:
        path = root / f"phase-{phase}.json"
        if path.exists():
            payload = _load_phase(path, phase=phase, identity=identity)
            reused += 1
        else:
            payload = _phase_payload(
                phase=phase,
                frame=source,
                identity=identity,
                load_predictor=load_predictor,
            )
            _write_new(path, payload)
        payloads.append(payload)
        entries.append({"path": path.name, "sha256": str(payload["phase_sha256"])})
        completed += 1
        if stop_after_phases is not None and completed >= stop_after_phases:
            return PhasedAuditResult(
                "incomplete", len(source), completed, reused, root / "full_audit_manifest.json"
            )
    baseline_predictions = payloads[0]["predictions"]
    baseline_features = payloads[0]["feature_sha256"]
    if not isinstance(baseline_predictions, dict) or not isinstance(baseline_features, dict):
        raise RulesEvidenceError("baseline audit evidence is incomplete")
    for payload in payloads[1:]:
        predictions = payload["predictions"]
        features = payload["feature_sha256"]
        if set(predictions) != set(baseline_predictions) or features != baseline_features:
            raise RulesEvidenceError(f"row independence mismatch: {payload['phase']}")
        maximum = max(
            abs(float(predictions[row_id]) - float(baseline_predictions[row_id]))
            for row_id in baseline_predictions
        )
        if maximum > tolerance:
            raise RulesEvidenceError(f"row independence mismatch: {payload['phase']}")
    manifest = {
        "schema_version": 2,
        "status": "passed",
        "audit_scope": "official_sample_plus_synthetic_scale",
        "identity": asdict(identity),
        "row_count": len(source),
        "singleton_count": singleton_count,
        "tolerance": tolerance,
        "phases": entries,
    }
    manifest_path = root / "full_audit_manifest.json"
    if manifest_path.exists():
        if _load(manifest_path) != manifest:
            raise RulesEvidenceError("existing phased audit manifest differs")
    else:
        _write_new(manifest_path, manifest)
    validate_full_audit(manifest_path, expected_identity=identity)
    return PhasedAuditResult("passed", len(source), completed, reused, manifest_path)


def _validate_phased_audit(
    path: Path,
    manifest: dict[str, object],
    identity: AuditIdentity,
) -> dict[str, object]:
    expected_keys = {
        "schema_version",
        "status",
        "audit_scope",
        "identity",
        "row_count",
        "singleton_count",
        "tolerance",
        "phases",
    }
    if set(manifest) != expected_keys:
        raise RulesEvidenceError("phased audit manifest fields differ")
    if (
        manifest["status"] != "passed"
        or manifest["audit_scope"] != "official_sample_plus_synthetic_scale"
        or manifest["identity"] != asdict(identity)
        or manifest["singleton_count"] != manifest["row_count"]
    ):
        raise RulesEvidenceError("phased audit manifest identity or scope differs")
    phases = manifest["phases"]
    if not isinstance(phases, list) or len(phases) != len(_PHASES):
        raise RulesEvidenceError("phased audit phase list differs")
    for expected_phase, entry in zip(_PHASES, phases, strict=True):
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise RulesEvidenceError("phased audit phase entry differs")
        phase_path = path.parent / str(entry["path"])
        if phase_path.name != f"phase-{expected_phase}.json":
            raise RulesEvidenceError("phased audit phase order differs")
        payload = _load_phase(phase_path, phase=expected_phase, identity=identity)
        if payload["phase_sha256"] != entry["sha256"]:
            raise RulesEvidenceError(f"phased audit phase hash differs: {expected_phase}")
        if payload["row_count"] != manifest["row_count"]:
            raise RulesEvidenceError(f"phased audit phase row count differs: {expected_phase}")
    return manifest
