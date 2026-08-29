from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import stat
import tempfile
from types import MappingProxyType
from typing import Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

from .s4_artifacts import S4Bindings


class S4RecoveryError(ValueError):
    pass


_CONTRACT_KEYS = {
    "schema_version",
    "artifact_kind",
    "source_handoff_sha256",
    "predecessor_code_sha256",
    "s4_contract_sha256",
    "source_bindings",
}
_BINDING_KEYS = {
    "contract_sha256",
    "code_sha256",
    "input_manifest_sha256",
    "train_sha256",
    "history_sha256",
    "e2_handoff_sha256",
}
RECOVERY_INPUT_KIND = "tree_s4_recovery_input_v1"
_MAX_FILES = 4096
_MAX_BYTES = 12 * 1024 * 1024 * 1024
_MAX_MEMBER_BYTES = 8 * 1024 * 1024 * 1024
_ZIP_TIME = (2026, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class S4RecoveryContract:
    schema_version: int
    artifact_kind: str
    source_handoff_sha256: str
    predecessor_code_sha256: str
    s4_contract_sha256: str
    source_bindings: Mapping[str, str]


@dataclass(frozen=True)
class RecoveryResult:
    path: Path
    sha256: str
    source_sha256: str
    retained_bytes: int
    dropped_bytes: int


@dataclass(frozen=True)
class VerifiedRecoveryInput:
    source: Path
    manifest_sha256: str
    source_handoff_sha256: str
    destination_bindings: Mapping[str, str]
    state_phase: str
    completed_full_chains: tuple[int, ...]
    resume_members: tuple[str, ...]


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def parse_recovery_contract(payload: object) -> S4RecoveryContract:
    if type(payload) is not dict or set(payload) != _CONTRACT_KEYS:
        raise S4RecoveryError("recovery contract keys differ")
    bindings = payload["source_bindings"]
    if type(bindings) is not dict or set(bindings) != _BINDING_KEYS:
        raise S4RecoveryError("recovery contract binding keys differ")
    hashes = (
        payload["source_handoff_sha256"],
        payload["predecessor_code_sha256"],
        payload["s4_contract_sha256"],
        *bindings.values(),
    )
    if any(not _is_sha256(value) for value in hashes):
        raise S4RecoveryError("recovery contract SHA-256 differs")
    if (
        payload["schema_version"] != 1
        or payload["artifact_kind"] != "tree_s4_recovery_contract_v1"
        or payload["predecessor_code_sha256"] != bindings["code_sha256"]
        or payload["s4_contract_sha256"] != bindings["contract_sha256"]
    ):
        raise S4RecoveryError("recovery contract identity differs")
    return S4RecoveryContract(
        schema_version=1,
        artifact_kind="tree_s4_recovery_contract_v1",
        source_handoff_sha256=str(payload["source_handoff_sha256"]),
        predecessor_code_sha256=str(payload["predecessor_code_sha256"]),
        s4_contract_sha256=str(payload["s4_contract_sha256"]),
        source_bindings=MappingProxyType(dict(bindings)),
    )


def load_recovery_contract(path: Path | None = None) -> S4RecoveryContract:
    source = Path(__file__).with_name("s4_recovery_contract.json") if path is None else Path(path)
    if source.is_symlink() or not source.is_file():
        raise S4RecoveryError("recovery contract source differs")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S4RecoveryError("recovery contract is invalid") from error
    return parse_recovery_contract(payload)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, _ZIP_TIME)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _file_sha256(path: Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise S4RecoveryError("artifact source is not a regular file")
    digest = sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_infos(archive: ZipFile, label: str) -> dict[str, ZipInfo]:
    infos: dict[str, ZipInfo] = {}
    total = 0
    for info in archive.infolist():
        pure = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if (
            info.filename in infos
            or info.flag_bits & 1
            or info.is_dir()
            or pure.is_absolute()
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
            or "\\" in info.filename
            or stat.S_ISLNK(mode)
            or info.file_size > _MAX_MEMBER_BYTES
        ):
            raise S4RecoveryError(f"unsafe or duplicate member in {label}")
        total += info.file_size
        if len(infos) >= _MAX_FILES or total > _MAX_BYTES:
            raise S4RecoveryError(f"{label} expansion exceeds limit")
        infos[info.filename] = info
    return infos


def _json(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S4RecoveryError(f"{label} is invalid") from error
    if type(value) is not dict:
        raise S4RecoveryError(f"{label} differs")
    return value


def _member_evidence(value: object, label: str) -> tuple[int, str]:
    if (
        type(value) is not dict
        or set(value) != {"size", "sha256"}
        or type(value["size"]) is not int
        or value["size"] < 0
        or not _is_sha256(value["sha256"])
    ):
        raise S4RecoveryError(f"{label} evidence differs")
    return int(value["size"]), str(value["sha256"])


def _copy_and_digest(source, target=None) -> tuple[int, str]:
    size = 0
    digest = sha256()
    for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
        size += len(block)
        digest.update(block)
        if target is not None:
            target.write(block)
    return size, digest.hexdigest()


def _destination_bindings(
    contract: S4RecoveryContract, code_sha256: str
) -> S4Bindings:
    if not _is_sha256(code_sha256):
        raise S4RecoveryError("destination code SHA-256 differs")
    values = dict(contract.source_bindings)
    values["code_sha256"] = code_sha256
    return S4Bindings(**values)


def _extract_verified_source_resume(
    source: Path,
    destination: Path,
    contract: S4RecoveryContract,
) -> bytes:
    if _file_sha256(source) != contract.source_handoff_sha256:
        raise S4RecoveryError("source handoff SHA-256 differs")
    try:
        with ZipFile(source) as archive:
            infos = _safe_infos(archive, "source handoff")
            if "manifest.json" not in infos or "resume.zip" not in infos:
                raise S4RecoveryError("source handoff members differ")
            manifest_bytes = archive.read(infos["manifest.json"])
            manifest = _json(manifest_bytes, "source handoff manifest")
            if (
                manifest.get("artifact_kind") != "tree_s4_handoff_v1"
                or manifest.get("bindings") != dict(contract.source_bindings)
                or manifest.get("submission_package") is not False
                or manifest.get("delivery") is not False
            ):
                raise S4RecoveryError("source handoff identity differs")
            declared = manifest.get("members")
            if type(declared) is not dict or set(declared) != set(infos) - {"manifest.json"}:
                raise S4RecoveryError("source handoff member manifest differs")
            with Path(destination).open("wb") as resume_target:
                for name, info in sorted(infos.items()):
                    if name == "manifest.json":
                        continue
                    expected_size, expected_sha = _member_evidence(
                        declared[name], f"source handoff member {name}"
                    )
                    with archive.open(info) as member:
                        size, digest = _copy_and_digest(
                            member, resume_target if name == "resume.zip" else None
                        )
                    if size != expected_size or digest != expected_sha:
                        raise S4RecoveryError(f"source handoff member differs: {name}")
            return manifest_bytes
    except (OSError, BadZipFile) as error:
        if isinstance(error, S4RecoveryError):
            raise
        raise S4RecoveryError("source handoff is not a valid ZIP") from error


def _state_identity(payload: bytes) -> tuple[str, tuple[int, ...]]:
    state = _json(payload, "S4 recovery state")
    if set(state) != {
        "schema_version", "phase", "completed_jobs", "failed_jobs", "decisions"
    } or state.get("schema_version") != 1:
        raise S4RecoveryError("S4 recovery state schema differs")
    phase = state.get("phase")
    completed = state.get("completed_jobs")
    failed = state.get("failed_jobs")
    decisions = state.get("decisions")
    if (
        phase not in {"anchors", "residuals", "full_chains", "confirmation", "full_fit", "completed"}
        or type(completed) is not list
        or type(failed) is not list
        or type(decisions) is not dict
    ):
        raise S4RecoveryError("S4 recovery state values differ")
    jobs = tuple(str(value) for value in completed)
    indices = []
    for name in jobs:
        if name.startswith("full_chains__"):
            token = name.removeprefix("full_chains__")
            if len(token) != 2 or not token.isdigit():
                raise S4RecoveryError("completed full-chain identity differs")
            indices.append(int(token))
    return str(phase), tuple(sorted(indices))


def _retain_resume_member(name: str, phase: str) -> bool:
    if name == "s4_campaign.log":
        return True
    top = PurePosixPath(name).parts[0]
    common = {"state", "diagnostics", "decisions"}
    if phase == "anchors":
        return top in common | {"anchor_basis"}
    if phase == "residuals":
        return top in common | {"anchors", "residual_predictions"}
    if phase in {"full_chains", "confirmation"}:
        return top in common | {"anchors", "residual_predictions", "full_chains", "confirmation"}
    return top in common | {"confirmation", "accepted", "models"}


def _required_full_chain_members(indices: tuple[int, ...]) -> set[str]:
    required = set()
    for index in indices:
        prefix = f"full_chains/c{index:02d}"
        required.add(f"{prefix}/config.json")
        required.update(f"{prefix}/{year}.csv" for year in (2022, 2023, 2024))
    return required


def _compact_resume(
    source_resume: Path,
    destination: Path,
    contract: S4RecoveryContract,
    destination_bindings: S4Bindings,
) -> tuple[int, int, str, tuple[int, ...], tuple[str, ...]]:
    try:
        with ZipFile(source_resume) as source:
            infos = _safe_infos(source, "source resume")
            if "manifest.json" not in infos or "state/state.json" not in infos:
                raise S4RecoveryError("source resume members differ")
            manifest = _json(source.read(infos["manifest.json"]), "source resume manifest")
            if (
                manifest.get("artifact_kind") != "tree_s4_resume_v1"
                or manifest.get("bindings") != dict(contract.source_bindings)
            ):
                raise S4RecoveryError("source resume identity differs")
            declared = manifest.get("members")
            if type(declared) is not dict or set(declared) != set(infos) - {"manifest.json"}:
                raise S4RecoveryError("source resume member manifest differs")
            state_payload = source.read(infos["state/state.json"])
            phase, completed_full_chains = _state_identity(state_payload)
            retained_names = tuple(
                name for name in sorted(declared) if _retain_resume_member(name, phase)
            )
            missing = _required_full_chain_members(completed_full_chains) - set(retained_names)
            if missing:
                raise S4RecoveryError(
                    f"completed full-chain output is absent: {sorted(missing)[0]}"
                )
            retained = 0
            dropped = 0
            evidence: dict[str, dict[str, object]] = {}
            with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as target:
                for name, info in sorted(infos.items()):
                    if name == "manifest.json":
                        continue
                    expected_size, expected_sha = _member_evidence(
                        declared[name], f"source resume member {name}"
                    )
                    keep = name in retained_names
                    with source.open(info) as member:
                        if keep:
                            with target.open(_zip_info(name), "w") as output:
                                size, digest = _copy_and_digest(member, output)
                        else:
                            size, digest = _copy_and_digest(member)
                    if size != expected_size or digest != expected_sha:
                        raise S4RecoveryError(f"source resume member differs: {name}")
                    if keep:
                        retained += size
                        evidence[name] = {"size": size, "sha256": digest}
                    else:
                        dropped += size
                compact_manifest = {
                    "schema_version": 1,
                    "artifact_kind": "tree_s4_resume_v1",
                    "bindings": asdict(destination_bindings),
                    "members": evidence,
                    "recovery_source_handoff_sha256": contract.source_handoff_sha256,
                }
                target.writestr(_zip_info("manifest.json"), _canonical(compact_manifest))
            return retained, dropped, phase, completed_full_chains, retained_names
    except (OSError, BadZipFile) as error:
        if isinstance(error, S4RecoveryError):
            raise
        raise S4RecoveryError("source resume is not a valid ZIP") from error


def _write_recovery_input(
    destination: Path,
    compact_resume: Path,
    source_manifest: bytes,
    contract: S4RecoveryContract,
    bindings: S4Bindings,
    phase: str,
    completed_full_chains: tuple[int, ...],
) -> None:
    resume_size = compact_resume.stat().st_size
    resume_sha = _file_sha256(compact_resume)
    payload_evidence = {
        "resume.zip": {"size": resume_size, "sha256": resume_sha},
        "source_handoff_manifest.json": {
            "size": len(source_manifest),
            "sha256": sha256(source_manifest).hexdigest(),
        },
    }
    manifest = {
        "schema_version": 1,
        "artifact_kind": RECOVERY_INPUT_KIND,
        "source_handoff_sha256": contract.source_handoff_sha256,
        "predecessor_code_sha256": contract.predecessor_code_sha256,
        "destination_bindings": asdict(bindings),
        "state_phase": phase,
        "completed_full_chains": list(completed_full_chains),
        "members": payload_evidence,
    }
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        with compact_resume.open("rb") as source, archive.open(_zip_info("resume.zip"), "w") as target:
            shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
        archive.writestr(_zip_info("source_handoff_manifest.json"), source_manifest)
        archive.writestr(_zip_info("manifest.json"), _canonical(manifest))


def compact_recovery_handoff(
    source_handoff: Path,
    output: Path,
    *,
    destination_code_sha256: str,
    contract: S4RecoveryContract | None = None,
) -> RecoveryResult:
    recovery = load_recovery_contract() if contract is None else contract
    source = Path(source_handoff)
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bindings = _destination_bindings(recovery, destination_code_sha256)
    with tempfile.TemporaryDirectory(prefix=".s4_recovery_", dir=destination.parent) as temporary_name:
        temporary = Path(temporary_name)
        source_resume = temporary / "source_resume.zip"
        compact_resume = temporary / "compact_resume.zip"
        source_manifest = _extract_verified_source_resume(source, source_resume, recovery)
        retained, dropped, phase, completed, _ = _compact_resume(
            source_resume, compact_resume, recovery, bindings
        )
        target = temporary / "recovery_input.zip"
        _write_recovery_input(
            target, compact_resume, source_manifest, recovery, bindings, phase, completed
        )
        os.link(target, destination)
    return RecoveryResult(
        path=destination,
        sha256=_file_sha256(destination),
        source_sha256=recovery.source_handoff_sha256,
        retained_bytes=retained,
        dropped_bytes=dropped,
    )


def verify_recovery_input(
    source: Path,
    *,
    expected_code_sha256: str,
    contract: S4RecoveryContract | None = None,
) -> VerifiedRecoveryInput:
    recovery = load_recovery_contract() if contract is None else contract
    expected_bindings = asdict(_destination_bindings(recovery, expected_code_sha256))
    path = Path(source)
    try:
        with ZipFile(path) as archive:
            infos = _safe_infos(archive, "S4 recovery input")
            required = {"resume.zip", "source_handoff_manifest.json", "manifest.json"}
            if set(infos) != required:
                raise S4RecoveryError("S4 recovery input members differ")
            manifest_bytes = archive.read(infos["manifest.json"])
            manifest = _json(manifest_bytes, "S4 recovery input manifest")
            if (
                manifest.get("schema_version") != 1
                or manifest.get("artifact_kind") != RECOVERY_INPUT_KIND
                or manifest.get("source_handoff_sha256") != recovery.source_handoff_sha256
                or manifest.get("predecessor_code_sha256") != recovery.predecessor_code_sha256
                or manifest.get("destination_bindings") != expected_bindings
            ):
                raise S4RecoveryError("S4 recovery input identity differs")
            declared = manifest.get("members")
            if type(declared) is not dict or set(declared) != required - {"manifest.json"}:
                raise S4RecoveryError("S4 recovery input member manifest differs")
            with tempfile.TemporaryDirectory(prefix=".s4_verify_", dir=path.parent) as temp_name:
                resume_path = Path(temp_name) / "resume.zip"
                for name in ("resume.zip", "source_handoff_manifest.json"):
                    expected_size, expected_sha = _member_evidence(
                        declared[name], f"S4 recovery input member {name}"
                    )
                    with archive.open(infos[name]) as member:
                        if name == "resume.zip":
                            with resume_path.open("wb") as target:
                                size, digest = _copy_and_digest(member, target)
                        else:
                            size, digest = _copy_and_digest(member)
                    if size != expected_size or digest != expected_sha:
                        raise S4RecoveryError(f"S4 recovery input member differs: {name}")
                with ZipFile(resume_path) as resume:
                    resume_infos = _safe_infos(resume, "compact S4 resume")
                    resume_manifest = _json(
                        resume.read(resume_infos["manifest.json"]), "compact S4 resume manifest"
                    )
                    if (
                        resume_manifest.get("artifact_kind") != "tree_s4_resume_v1"
                        or resume_manifest.get("bindings") != expected_bindings
                        or resume_manifest.get("recovery_source_handoff_sha256")
                        != recovery.source_handoff_sha256
                    ):
                        raise S4RecoveryError("compact S4 resume identity differs")
                    resume_members = resume_manifest.get("members")
                    if type(resume_members) is not dict or set(resume_members) != set(resume_infos) - {"manifest.json"}:
                        raise S4RecoveryError("compact S4 resume member manifest differs")
                    phase, completed = _state_identity(resume.read("state/state.json"))
                    if phase != manifest.get("state_phase") or list(completed) != manifest.get("completed_full_chains"):
                        raise S4RecoveryError("S4 recovery state binding differs")
            return VerifiedRecoveryInput(
                source=path,
                manifest_sha256=sha256(manifest_bytes).hexdigest(),
                source_handoff_sha256=recovery.source_handoff_sha256,
                destination_bindings=MappingProxyType(expected_bindings),
                state_phase=phase,
                completed_full_chains=completed,
                resume_members=tuple(sorted(resume_members)),
            )
    except (OSError, BadZipFile, KeyError) as error:
        if isinstance(error, S4RecoveryError):
            raise
        raise S4RecoveryError("S4 recovery input is not a valid ZIP") from error
