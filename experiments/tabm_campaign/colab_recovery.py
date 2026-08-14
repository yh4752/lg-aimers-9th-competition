"""Trusted direct-upload and restart primitives for the Colab Stage C handoff."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from typing import Mapping
import uuid
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from .artifacts import (
    StageEvidence,
    verify_resume_bundle,
    verify_review_bundle,
    write_stage_bundles,
)


class ColabRecoveryError(RuntimeError):
    """Raised before untrusted or incompatible Colab state can be consumed."""


@dataclass(frozen=True)
class SanitizedResume:
    path: Path
    source_sha256: str
    source_manifest_sha256: str
    sanitized_sha256: str
    sanitized_manifest_sha256: str


@dataclass(frozen=True)
class RuntimeIdentity:
    base_resume_sha256: str
    base_manifest_sha256: str
    sanitized_resume_sha256: str
    campaign_config_sha256: str
    runtime_sha256: str
    training_source_sha256: str
    cache_sha256: str | None
    python: str
    torch: str
    cuda_runtime: str
    numpy: str
    pandas: str
    tabm: str
    rtdl_num_embeddings: str
    gpu_name: str
    target_candidate_id: str


@dataclass(frozen=True)
class EmergencySnapshot:
    path: Path
    epoch: int
    sha256: str
    manifest: Mapping[str, object]


@dataclass(frozen=True)
class ProcessReceipt:
    pid: int
    process_start_identity: str
    command_sha256: str
    run_uuid: str


@dataclass(frozen=True)
class VerifiedDelivery:
    path: Path
    sha256: str
    member_sha256: Mapping[str, str]
    final_stage_complete: bool


@dataclass(frozen=True)
class SupervisorResult:
    interrupted: bool
    returncode: int
    latest_snapshot: EmergencySnapshot | None
    receipt: ProcessReceipt
    log_path: Path


_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_colab_contract(
    path: Path = Path(__file__).with_name("colab_stage_c_contract.json"),
) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ColabRecoveryError(f"cannot load Colab recovery contract: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ColabRecoveryError("unsupported Colab recovery contract")
    return value


def _declared_data_members(
    contract: Mapping[str, object],
) -> tuple[Mapping[str, object], int, int]:
    archive = contract.get("data_archive")
    if not isinstance(archive, Mapping):
        raise ColabRecoveryError("data archive contract is missing")
    members = archive.get("members")
    if not isinstance(members, Mapping) or not members:
        raise ColabRecoveryError("data archive member contract is invalid")
    try:
        max_count = int(archive["max_member_count"])
        max_bytes = int(archive["max_uncompressed_bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise ColabRecoveryError("data archive limits are invalid") from error
    if max_count != len(members) or max_bytes != sum(
        int(evidence["size"])
        for evidence in members.values()
        if isinstance(evidence, Mapping)
    ):
        raise ColabRecoveryError("data archive limits differ from declared members")
    return members, max_count, max_bytes


def _safe_flat_member(info: ZipInfo) -> bool:
    path = PurePosixPath(info.filename)
    unix_mode = info.external_attr >> 16
    is_link = stat.S_ISLNK(unix_mode)
    return bool(
        info.filename
        and "\\" not in info.filename
        and not path.is_absolute()
        and ".." not in path.parts
        and len(path.parts) == 1
        and not info.is_dir()
        and not is_link
    )


def _verify_live_data_directory(
    destination: Path, members: Mapping[str, object]
) -> None:
    actual: set[str] = set()
    for path in destination.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ColabRecoveryError("published data directory has unsafe members")
        actual.add(path.name)
    if actual != set(members):
        raise ColabRecoveryError("published data member names differ")
    for name, raw_evidence in members.items():
        if not isinstance(raw_evidence, Mapping):
            raise ColabRecoveryError(f"data member contract is invalid: {name}")
        path = destination / str(name)
        if path.stat().st_size != int(raw_evidence["size"]):
            raise ColabRecoveryError(f"published data member size differs: {name}")
        if file_sha256(path) != str(raw_evidence["sha256"]):
            raise ColabRecoveryError(f"published data member SHA-256 differs: {name}")


def verify_and_extract_data_archive(
    archive: Path,
    destination: Path,
    contract: Mapping[str, object],
) -> Path:
    """Verify the exact four-file upload and publish an atomic data directory."""

    members, max_count, max_bytes = _declared_data_members(contract)
    destination = destination.resolve()
    if destination.exists():
        if not destination.is_dir() or destination.is_symlink():
            raise ColabRecoveryError("published data path is not a safe directory")
        _verify_live_data_directory(destination, members)
        return destination

    temporary = destination.parent / f".{destination.name}.extracting"
    if temporary.exists():
        if temporary.is_symlink() or not temporary.is_dir():
            raise ColabRecoveryError("stale extraction path is unsafe")
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        with ZipFile(archive, "r") as source:
            infos = source.infolist()
            names = [info.filename for info in infos]
            if len(infos) > max_count or len(names) != len(set(names)):
                raise ColabRecoveryError("data archive has duplicate or excessive members")
            if any(not _safe_flat_member(info) for info in infos):
                raise ColabRecoveryError("data archive has an unsafe member")
            if set(names) != set(members):
                raise ColabRecoveryError("data archive member names differ")
            if sum(info.file_size for info in infos) != max_bytes:
                raise ColabRecoveryError("data archive uncompressed size differs")

            for info in infos:
                evidence = members[info.filename]
                if not isinstance(evidence, Mapping):
                    raise ColabRecoveryError(
                        f"data member contract is invalid: {info.filename}"
                    )
                expected_size = int(evidence["size"])
                if info.file_size != expected_size:
                    raise ColabRecoveryError(
                        f"data archive member size differs: {info.filename}"
                    )
                target = temporary / info.filename
                digest = sha256()
                written = 0
                with source.open(info, "r") as input_stream, target.open("xb") as output:
                    while chunk := input_stream.read(1024 * 1024):
                        written += len(chunk)
                        if written > expected_size:
                            raise ColabRecoveryError(
                                f"data archive member expanded beyond contract: {info.filename}"
                            )
                        output.write(chunk)
                        digest.update(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if written != expected_size or digest.hexdigest() != str(
                    evidence["sha256"]
                ):
                    raise ColabRecoveryError(
                        f"data archive member SHA-256 differs: {info.filename}"
                    )
        _verify_live_data_directory(temporary, members)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
        return destination
    except Exception:
        if temporary.is_dir() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise


def _write_verified_resume_members(
    destination: Path,
    *,
    version: str,
    campaign_config_sha256: str,
    prior_manifest_sha256: str | None,
    members: Mapping[str, bytes],
) -> tuple[Path, str]:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        bundles = write_stage_bundles(
            temporary,
            StageEvidence(
                version,
                campaign_config_sha256,
                prior_manifest_sha256,
                {"audit/resume_rebuild.json": canonical_json({"version": version})},
                members,
            ),
        )
        if bundles.resume is None:
            raise ColabRecoveryError("rebuilt Stage C resume was not created")
        os.replace(bundles.resume, destination)
    verified = verify_resume_bundle(destination)
    return destination, verified.manifest_sha256


def sanitize_stage_c_resume(
    source: Path,
    destination: Path,
    contract: Mapping[str, object],
) -> SanitizedResume:
    """Discard only the unbound Kaggle checkpoint from the fixed Stage C resume."""

    base_contract = contract.get("base_resume")
    if not isinstance(base_contract, Mapping):
        raise ColabRecoveryError("base resume contract is missing")
    source_sha = file_sha256(source)
    if source_sha != str(base_contract.get("sha256")):
        raise ColabRecoveryError("base Stage C resume SHA-256 differs")
    verified = verify_resume_bundle(source)
    if verified.version != "C" or base_contract.get("version") != "C":
        raise ColabRecoveryError("base resume is not Stage C")
    with ZipFile(source, "r") as archive:
        members = {
            name: archive.read(name)
            for name in archive.namelist()
            if name != "manifest.json"
        }
    try:
        state = json.loads(members["stage_state.json"])
    except (KeyError, json.JSONDecodeError) as error:
        raise ColabRecoveryError("base Stage C state is unreadable") from error
    rows = state.get("results")
    if (
        state.get("version") != "C"
        or state.get("stage_complete") is not False
        or state.get("reason") != "older_fold_seed_confirmation_incomplete"
        or not isinstance(rows, list)
    ):
        raise ColabRecoveryError(
            "base Stage C state is not the expected incomplete state"
        )
    target_id = str(contract.get("target_candidate_id"))
    completed = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("status") == "completed"
    ]
    incomplete = [
        row
        for row in rows
        if not isinstance(row, dict) or row.get("status") != "completed"
    ]
    if (
        len(completed) != int(contract.get("expected_completed_jobs", -1))
        or len(incomplete) != 1
        or incomplete[0].get("candidate_id") != target_id
    ):
        raise ColabRecoveryError("Stage C completed/pending job identity differs")
    artifacts = state.get("resume_artifacts")
    if not isinstance(artifacts, dict):
        raise ColabRecoveryError("Stage C resume artifact map is invalid")
    target = incomplete[0]
    target.update(
        {
            "status": "inconclusive",
            "brier": None,
            "best_epoch": None,
            "completed_epochs": 0,
            "checkpoint": None,
            "predictions": None,
            "resource_evidence": {},
            "failure": "cross_runtime_restart_required",
        }
    )
    artifacts.pop(target_id, None)
    target_prefix = f"training/{target_id}/"
    members = {
        name: value
        for name, value in members.items()
        if not name.startswith(target_prefix)
    }
    members["stage_state.json"] = canonical_json(state)
    path, manifest_sha = _write_verified_resume_members(
        destination,
        version="C",
        campaign_config_sha256=verified.campaign_config_sha256,
        prior_manifest_sha256=verified.prior_manifest_sha256,
        members=members,
    )
    return SanitizedResume(
        path=path,
        source_sha256=source_sha,
        source_manifest_sha256=verified.manifest_sha256,
        sanitized_sha256=file_sha256(path),
        sanitized_manifest_sha256=manifest_sha,
    )


def _deterministic_zip_bytes(members: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(members):
            info = ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, members[name])
    return buffer.getvalue()


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _checkpoint_binding(
    meta: Mapping[str, object], identity: RuntimeIdentity
) -> Mapping[str, object]:
    if meta.get("candidate_id") != identity.target_candidate_id:
        raise ColabRecoveryError("checkpoint candidate differs from recovery target")
    epoch = meta.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ColabRecoveryError("checkpoint epoch is invalid")
    if meta.get("checkpoint") != "checkpoint.pt":
        raise ColabRecoveryError("checkpoint metadata points to an unexpected file")
    binding = meta.get("checkpoint_binding")
    if not isinstance(binding, Mapping):
        raise ColabRecoveryError("checkpoint binding is absent")
    required = {
        "config_sha256": identity.campaign_config_sha256,
        "training_source_sha256": identity.training_source_sha256,
    }
    for key, expected in required.items():
        if binding.get(key) != expected:
            raise ColabRecoveryError(f"checkpoint {key} binding differs")
    cache_sha = binding.get("cache_sha256")
    if not isinstance(cache_sha, str) or len(cache_sha) != 64:
        raise ColabRecoveryError("checkpoint cache binding is invalid")
    if identity.cache_sha256 is not None and cache_sha != identity.cache_sha256:
        raise ColabRecoveryError("checkpoint cache binding differs")
    return binding


def create_emergency_snapshot(
    *,
    training_dir: Path,
    log_path: Path,
    output_dir: Path,
    identity: RuntimeIdentity,
) -> EmergencySnapshot:
    """Publish one hash-bound snapshot from a stable complete epoch."""

    meta_path = training_dir / "checkpoint_meta.json"
    checkpoint_path = training_dir / "checkpoint.pt"
    best_path = training_dir / "best_checkpoint.pt"
    first_meta = meta_path.read_bytes()
    try:
        meta = json.loads(first_meta)
    except json.JSONDecodeError as error:
        raise ColabRecoveryError("checkpoint metadata is unreadable") from error
    if not isinstance(meta, Mapping):
        raise ColabRecoveryError("checkpoint metadata is invalid")
    binding = _checkpoint_binding(meta, identity)
    payload_members = {
        f"training/{identity.target_candidate_id}/checkpoint.pt": checkpoint_path.read_bytes(),
        f"training/{identity.target_candidate_id}/checkpoint_meta.json": first_meta,
        f"training/{identity.target_candidate_id}/best_checkpoint.pt": best_path.read_bytes(),
        "logs/colab_stage_C.log": log_path.read_bytes(),
    }
    if meta_path.read_bytes() != first_meta:
        raise ColabRecoveryError("checkpoint changed while snapshotting")
    identity_payload = asdict(identity)
    identity_payload["cache_sha256"] = str(binding["cache_sha256"])
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "tabm_colab_emergency",
        "epoch": int(meta["epoch"]),
        "candidate_id": identity.target_candidate_id,
        "runtime_identity": identity_payload,
        "members": {
            name: sha256(value).hexdigest()
            for name, value in sorted(payload_members.items())
        },
    }
    archive_bytes = _deterministic_zip_bytes(
        {**payload_members, "emergency_manifest.json": canonical_json(manifest)}
    )
    digest = sha256(archive_bytes).hexdigest()
    output = output_dir / (
        f"tabm_colab_emergency_epoch_{int(meta['epoch']):03d}_{digest[:12]}.zip"
    )
    if output.exists():
        if file_sha256(output) != digest:
            raise ColabRecoveryError("existing emergency snapshot hash differs")
    else:
        _atomic_write_bytes(output, archive_bytes)
    return EmergencySnapshot(output, int(meta["epoch"]), digest, manifest)


def _verify_emergency_snapshot(
    path: Path,
    sanitized: SanitizedResume,
    expected_identity: RuntimeIdentity,
) -> EmergencySnapshot:
    digest = file_sha256(path)
    try:
        with ZipFile(path, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or "emergency_manifest.json" not in names:
                raise ColabRecoveryError("emergency snapshot member list is invalid")
            manifest = json.loads(archive.read("emergency_manifest.json"))
            if (
                not isinstance(manifest, dict)
                or manifest.get("schema_version") != 1
                or manifest.get("artifact_kind") != "tabm_colab_emergency"
            ):
                raise ColabRecoveryError("emergency snapshot manifest is invalid")
            member_hashes = manifest.get("members")
            if not isinstance(member_hashes, dict) or set(member_hashes) != set(names) - {
                "emergency_manifest.json"
            }:
                raise ColabRecoveryError("emergency snapshot member manifest differs")
            for name, expected_hash in member_hashes.items():
                if sha256(archive.read(name)).hexdigest() != expected_hash:
                    raise ColabRecoveryError(
                        f"emergency snapshot member SHA-256 differs: {name}"
                    )
            identity = manifest.get("runtime_identity")
            if not isinstance(identity, dict):
                raise ColabRecoveryError("emergency snapshot environment is absent")
            expected = asdict(expected_identity)
            for key, expected_value in expected.items():
                if key == "cache_sha256" and expected_value is None:
                    continue
                if identity.get(key) != expected_value:
                    raise ColabRecoveryError(
                        f"emergency snapshot environment differs: {key}"
                    )
            if (
                identity.get("base_resume_sha256") != sanitized.source_sha256
                or identity.get("base_manifest_sha256")
                != sanitized.source_manifest_sha256
                or identity.get("sanitized_resume_sha256")
                != sanitized.sanitized_sha256
            ):
                raise ColabRecoveryError("emergency snapshot resume binding differs")
            candidate = str(manifest.get("candidate_id"))
            if candidate != expected_identity.target_candidate_id:
                raise ColabRecoveryError("emergency snapshot candidate differs")
            meta_name = f"training/{candidate}/checkpoint_meta.json"
            meta = json.loads(archive.read(meta_name))
            if not isinstance(meta, Mapping):
                raise ColabRecoveryError("emergency checkpoint metadata is invalid")
            snapshot_identity = RuntimeIdentity(**identity)
            _checkpoint_binding(meta, snapshot_identity)
            epoch = int(manifest.get("epoch", -1))
            if epoch < 0 or meta.get("epoch") != epoch:
                raise ColabRecoveryError("emergency snapshot epoch differs")
    except ColabRecoveryError:
        raise
    except Exception as error:
        raise ColabRecoveryError(f"cannot verify emergency snapshot: {error}") from error
    return EmergencySnapshot(path, epoch, digest, manifest)


def verify_emergency_snapshots(
    paths: list[Path] | tuple[Path, ...],
    sanitized: SanitizedResume,
    expected_identity: RuntimeIdentity,
) -> EmergencySnapshot:
    if not paths:
        raise ColabRecoveryError("no emergency snapshots were supplied")
    verified = [
        _verify_emergency_snapshot(path, sanitized, expected_identity)
        for path in paths
    ]
    by_epoch: dict[int, set[str]] = {}
    for item in verified:
        by_epoch.setdefault(item.epoch, set()).add(item.sha256)
    if any(len(hashes) > 1 for hashes in by_epoch.values()):
        raise ColabRecoveryError("different emergency snapshots claim the same epoch")
    return max(verified, key=lambda item: (item.epoch, item.sha256))


def merge_emergency_snapshot(
    sanitized: SanitizedResume,
    snapshot: EmergencySnapshot,
    destination: Path,
) -> Path:
    """Bind verified Colab training files into the sanitized Stage C resume."""

    if file_sha256(sanitized.path) != sanitized.sanitized_sha256:
        raise ColabRecoveryError("sanitized resume changed before snapshot merge")
    sanitized_verified = verify_resume_bundle(sanitized.path)
    candidate = str(snapshot.manifest["candidate_id"])
    with ZipFile(sanitized.path, "r") as source:
        members = {
            name: source.read(name)
            for name in source.namelist()
            if name != "manifest.json"
        }
    with ZipFile(snapshot.path, "r") as emergency:
        training_names = [
            f"training/{candidate}/checkpoint.pt",
            f"training/{candidate}/checkpoint_meta.json",
            f"training/{candidate}/best_checkpoint.pt",
        ]
        for name in training_names:
            members[name] = emergency.read(name)
    state = json.loads(members["stage_state.json"])
    rows = state.get("results")
    if not isinstance(rows, list):
        raise ColabRecoveryError("sanitized Stage C result rows are invalid")
    matching = [
        row for row in rows if isinstance(row, dict) and row.get("candidate_id") == candidate
    ]
    if len(matching) != 1:
        raise ColabRecoveryError("snapshot target row is not unique")
    matching[0].update(
        {
            "status": "inconclusive",
            "brier": None,
            "best_epoch": None,
            "completed_epochs": snapshot.epoch + 1,
            "checkpoint": "best_checkpoint.pt",
            "predictions": None,
            "resource_evidence": {"recovered_from_emergency": snapshot.sha256},
            "failure": "colab_emergency_resume",
        }
    )
    artifacts = state.get("resume_artifacts")
    if not isinstance(artifacts, dict):
        raise ColabRecoveryError("sanitized Stage C artifact map is invalid")
    artifacts[candidate] = {"training_files": training_names}
    members["stage_state.json"] = canonical_json(state)
    path, _ = _write_verified_resume_members(
        destination,
        version="C",
        campaign_config_sha256=sanitized_verified.campaign_config_sha256,
        prior_manifest_sha256=sanitized_verified.prior_manifest_sha256,
        members=members,
    )
    return path


def process_identity(pid: int) -> tuple[str, str] | None:
    """Return process-start and command identities without trusting PID alone."""

    proc_root = Path("/proc") / str(pid)
    try:
        stat_text = (proc_root / "stat").read_text(encoding="utf-8")
        command = (proc_root / "cmdline").read_bytes()
        close = stat_text.rfind(")")
        if close < 0:
            return None
        fields_after_command = stat_text[close + 2 :].split()
        start_ticks = fields_after_command[19]
        return f"proc-start-ticks:{start_ticks}", sha256(command).hexdigest()
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
        pass

    try:
        started = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            capture_output=True,
            text=True,
            check=False,
        )
        command_result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    start_value = started.stdout.strip()
    command_value = command_result.stdout.strip()
    if started.returncode or command_result.returncode or not start_value or not command_value:
        return None
    return f"ps-start:{start_value}", sha256(command_value).hexdigest()


def write_process_receipt(
    path: Path, pid: int, run_uuid: str
) -> ProcessReceipt:
    identity = process_identity(pid)
    if identity is None:
        raise ColabRecoveryError(f"cannot identify process pid={pid}")
    receipt = ProcessReceipt(pid, identity[0], identity[1], run_uuid)
    _atomic_write_bytes(path, canonical_json(asdict(receipt)))
    return receipt


def _load_process_receipt(path: Path) -> ProcessReceipt:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return ProcessReceipt(
            pid=int(value["pid"]),
            process_start_identity=str(value["process_start_identity"]),
            command_sha256=str(value["command_sha256"]),
            run_uuid=str(value["run_uuid"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ColabRecoveryError(f"process lock receipt is invalid: {error}") from error


def acquire_process_lock(path: Path, run_uuid: str) -> ProcessReceipt:
    """Refuse a live matching owner and archive only demonstrably stale locks."""

    if path.exists():
        existing = _load_process_receipt(path)
        live = process_identity(existing.pid)
        if live == (
            existing.process_start_identity,
            existing.command_sha256,
        ):
            raise ColabRecoveryError(
                f"Colab Stage C process is already running pid={existing.pid}"
            )
        stale = path.with_name(f"{path.name}.stale-{uuid.uuid4().hex}")
        os.replace(path, stale)
    return write_process_receipt(path, os.getpid(), run_uuid)


def terminate_matching_process_group(
    receipt: ProcessReceipt, *, grace_seconds: float = 10.0
) -> None:
    live = process_identity(receipt.pid)
    expected = (receipt.process_start_identity, receipt.command_sha256)
    if live is None:
        return
    if live != expected:
        raise ColabRecoveryError("refusing to terminate a reused process ID")
    try:
        os.killpg(receipt.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if process_identity(receipt.pid) is None:
            return
        time.sleep(0.05)
    if process_identity(receipt.pid) == expected:
        try:
            os.killpg(receipt.pid, signal.SIGKILL)
        except ProcessLookupError:
            return


def _checkpoint_epoch(training_dir: Path) -> int | None:
    path = training_dir / "checkpoint_meta.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        epoch = value.get("epoch")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        return None
    if not (training_dir / "checkpoint.pt").is_file() or not (
        training_dir / "best_checkpoint.pt"
    ).is_file():
        return None
    return epoch


def supervise_campaign(
    *,
    command: list[str],
    work_root: Path,
    identity: RuntimeIdentity,
    snapshot_interval_seconds: float = 1200.0,
    interrupt_after_seconds: float | None = None,
    interrupt_grace_seconds: float = 180.0,
    poll_seconds: float = 2.0,
) -> SupervisorResult:
    """Run one campaign process with flushed logs and bounded safe interruption."""

    work_root.mkdir(parents=True, exist_ok=True)
    run_uuid = uuid.uuid4().hex
    lock_path = work_root / "campaign.lock"
    owner = acquire_process_lock(lock_path, run_uuid)
    log_path = work_root / "logs" / "colab_stage_C.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stop_marker = work_root / "stop-after-epoch.request"
    stop_marker.unlink(missing_ok=True)
    training_dir = (
        work_root
        / "outputs"
        / "stage_C"
        / "jobs"
        / identity.target_candidate_id
    )
    snapshot_root = work_root / "snapshots"
    environment = os.environ.copy()
    environment["TABM_STOP_AFTER_EPOCH_FILE"] = str(stop_marker.resolve())
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
        start_new_session=True,
    )
    receipt = write_process_receipt(
        work_root / "campaign_process.json", process.pid, run_uuid
    )
    reader_errors: list[BaseException] = []

    def stream_output() -> None:
        try:
            assert process.stdout is not None
            with log_path.open("a", encoding="utf-8") as log:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line.rstrip(), flush=True)
        except BaseException as error:
            reader_errors.append(error)

    reader = threading.Thread(target=stream_output, daemon=True)
    reader.start()
    started = time.monotonic()
    last_export_time = started
    last_export_epoch = -1
    latest: EmergencySnapshot | None = None
    interrupted = False
    interrupt_deadline: float | None = None

    try:
        while True:
            now = time.monotonic()
            if (
                not interrupted
                and interrupt_after_seconds is not None
                and now - started >= interrupt_after_seconds
            ):
                interrupted = True
                _atomic_write_bytes(stop_marker, b"stop-after-complete-epoch\n")
                interrupt_deadline = now + interrupt_grace_seconds
            epoch = _checkpoint_epoch(training_dir)
            if (
                epoch is not None
                and epoch > last_export_epoch
                and now - last_export_time >= snapshot_interval_seconds
            ):
                try:
                    latest = create_emergency_snapshot(
                        training_dir=training_dir,
                        log_path=log_path,
                        output_dir=snapshot_root,
                        identity=identity,
                    )
                    last_export_epoch = latest.epoch
                    last_export_time = now
                    print(
                        "EMERGENCY_SNAPSHOT_READY "
                        f"epoch={latest.epoch} path={latest.path} "
                        f"sha256={latest.sha256}",
                        flush=True,
                    )
                except (OSError, ColabRecoveryError) as error:
                    print(
                        f"EMERGENCY_SNAPSHOT_DEFERRED type={type(error).__name__} "
                        f"message={str(error).replace(' ', '_')}",
                        flush=True,
                    )
            returncode = process.poll()
            if returncode is not None:
                break
            if interrupted and interrupt_deadline is not None and now >= interrupt_deadline:
                terminate_matching_process_group(receipt)
                returncode = process.wait(timeout=15)
                break
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        interrupted = True
        _atomic_write_bytes(stop_marker, b"stop-after-complete-epoch\n")
        deadline = time.monotonic() + interrupt_grace_seconds
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(min(poll_seconds, 0.25))
        if process.poll() is None:
            terminate_matching_process_group(receipt)
        returncode = process.wait(timeout=15)
    finally:
        reader.join(timeout=5)

    epoch = _checkpoint_epoch(training_dir)
    if epoch is not None and epoch > last_export_epoch:
        latest = create_emergency_snapshot(
            training_dir=training_dir,
            log_path=log_path,
            output_dir=snapshot_root,
            identity=identity,
        )
        print(
            "EMERGENCY_SNAPSHOT_READY "
            f"epoch={latest.epoch} path={latest.path} sha256={latest.sha256}",
            flush=True,
        )
    if reader_errors:
        raise ColabRecoveryError(f"campaign log reader failed: {reader_errors[0]}")
    try:
        current_owner = _load_process_receipt(lock_path)
        if current_owner == owner:
            lock_path.unlink()
    except FileNotFoundError:
        pass
    if returncode != 0 and not interrupted:
        raise ColabRecoveryError(f"campaign process failed returncode={returncode}")
    return SupervisorResult(interrupted, returncode, latest, receipt, log_path)


def _stage_state_bytes(path: Path) -> bytes:
    try:
        with ZipFile(path, "r") as archive:
            return archive.read("stage_state.json")
    except Exception as error:
        raise ColabRecoveryError(f"cannot read Stage C state: {error}") from error


def write_delivery_bundle(
    *,
    review: Path,
    resume: Path,
    log: Path,
    identity: RuntimeIdentity,
    run_uuid: str,
    destination: Path,
) -> Path:
    review_verified = verify_review_bundle(review)
    resume_verified = verify_resume_bundle(resume)
    review_state = _stage_state_bytes(review)
    resume_state = _stage_state_bytes(resume)
    try:
        state = json.loads(review_state)
    except json.JSONDecodeError as error:
        raise ColabRecoveryError("final Stage C state is unreadable") from error
    if (
        review_verified.version != "C"
        or resume_verified.version != "C"
        or review_state != resume_state
        or state.get("stage_complete") is not True
    ):
        raise ColabRecoveryError(
            "delivery bundles do not share one complete Stage C state"
        )
    if (
        review_verified.campaign_config_sha256
        != resume_verified.campaign_config_sha256
        or review_verified.prior_manifest_sha256
        != resume_verified.prior_manifest_sha256
    ):
        raise ColabRecoveryError("delivery bundle lineage differs")
    payloads = {
        "tabm_search_stage_C_review_bundle.zip": review.read_bytes(),
        "tabm_search_stage_C_resume_bundle.zip": resume.read_bytes(),
        "colab_stage_C.log": log.read_bytes(),
    }
    manifest = {
        "schema_version": 1,
        "artifact_kind": "tabm_colab_stage_C_delivery",
        "run_uuid": run_uuid,
        "runtime_identity": asdict(identity),
        "final_stage_complete": True,
        "members": {
            name: {"size": len(value), "sha256": sha256(value).hexdigest()}
            for name, value in sorted(payloads.items())
        },
    }
    archive_bytes = _deterministic_zip_bytes(
        {**payloads, "delivery_manifest.json": canonical_json(manifest)}
    )
    _atomic_write_bytes(destination, archive_bytes)
    verify_delivery_bundle(destination)
    return destination


def verify_delivery_bundle(path: Path) -> VerifiedDelivery:
    try:
        with ZipFile(path, "r") as archive:
            names = archive.namelist()
            payload_names = {
                "tabm_search_stage_C_review_bundle.zip",
                "tabm_search_stage_C_resume_bundle.zip",
                "colab_stage_C.log",
            }
            if len(names) != len(set(names)) or set(names) != payload_names | {
                "delivery_manifest.json"
            }:
                raise ColabRecoveryError("delivery member names differ")
            manifest = json.loads(archive.read("delivery_manifest.json"))
            if (
                manifest.get("schema_version") != 1
                or manifest.get("artifact_kind")
                != "tabm_colab_stage_C_delivery"
                or manifest.get("final_stage_complete") is not True
            ):
                raise ColabRecoveryError("delivery manifest is invalid")
            expected = manifest.get("members")
            if not isinstance(expected, dict) or set(expected) != payload_names:
                raise ColabRecoveryError("delivery member manifest differs")
            hashes: dict[str, str] = {}
            for name in sorted(payload_names):
                value = archive.read(name)
                evidence = expected[name]
                if not isinstance(evidence, Mapping):
                    raise ColabRecoveryError(f"delivery evidence is invalid: {name}")
                digest = sha256(value).hexdigest()
                if len(value) != int(evidence.get("size", -1)) or digest != evidence.get(
                    "sha256"
                ):
                    raise ColabRecoveryError(f"delivery member hash differs: {name}")
                hashes[name] = digest
    except ColabRecoveryError:
        raise
    except Exception as error:
        raise ColabRecoveryError(f"cannot verify delivery bundle: {error}") from error
    return VerifiedDelivery(path, file_sha256(path), hashes, True)
