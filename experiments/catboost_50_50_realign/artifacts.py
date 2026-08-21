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
from typing import Callable, Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

from .state import Bindings, CampaignState, deserialize_state, bindings_payload


class RealignArtifactError(ValueError):
    """Raised when a realignment artifact is incomplete or untrusted."""


@dataclass(frozen=True)
class TrustedFile:
    path: Path
    sha256: str


@dataclass(frozen=True)
class CampaignFiles:
    status: str
    resume: Mapping[str, TrustedFile]
    review: Mapping[str, TrustedFile]
    delivery: Mapping[str, TrustedFile]

    def resume_members(self) -> Mapping[str, TrustedFile]:
        return self.resume

    def review_members(self) -> Mapping[str, TrustedFile]:
        return self.review

    def delivery_members(self) -> Mapping[str, TrustedFile]:
        return self.delivery

    def require_completed_delivery(self) -> None:
        if self.status != "completed" or not self.delivery:
            raise RealignArtifactError("completed delivery evidence is required")


@dataclass(frozen=True)
class VerifiedBundle:
    path: Path
    artifact_kind: str
    bindings: Bindings
    status: str
    selected_tree_count: int | None
    submission_package: bool
    members: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class RestoredRun:
    root: Path
    status: str
    state_path: Path


_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
_SHA_CHARS = set("0123456789abcdef")
_MAX_MEMBER = 8 * 1024**3
_MAX_TOTAL = 12 * 1024**3
_MAX_RATIO = 200.0
_MAX_COUNT = 256
_DELIVERY_NAMES = {
    "decision/alignment_decision.json",
    "decision/fold_metrics.json",
    "decision/bootstrap_metrics.json",
    "decision/segment_metrics.json",
    "frozen_catboost/model.cbm",
    "frozen_catboost/preprocessing_state.json",
    "frozen_catboost/inference_manifest.json",
    "logs/campaign.log",
    "policy/policy.json",
}
_DECISION_NAMES = {
    "decision/alignment_decision.json",
    "decision/fold_metrics.json",
    "decision/bootstrap_metrics.json",
    "decision/segment_metrics.json",
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, _ZIP_TIME)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100600 << 16
    return info


def _valid_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= _SHA_CHARS


def _safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        type(name) is str
        and bool(name)
        and "\\" not in name
        and not path.is_absolute()
        and ".." not in path.parts
        and "." not in path.parts
        and not name.endswith("/")
    )


def _reject_symlink_ancestors(path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise RealignArtifactError(f"trusted source has a symlink ancestor: {path}")


def _read_decision(path: Path) -> tuple[str, int | None]:
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RealignArtifactError("alignment decision is unreadable") from error
    if type(root) is not dict:
        raise RealignArtifactError("alignment decision differs")
    status = root.get("status")
    selected = root.get("selected_tree_count")
    if status == "promoted" and selected in (4, 8, 12, 16, 20, 24, 28, 32):
        return status, selected
    if status == "rejected" and selected is None:
        return status, None
    raise RealignArtifactError("alignment decision differs")


def _state_from_member(members: Mapping[str, TrustedFile]) -> CampaignState:
    item = members.get("state/stage_state.json")
    if item is None:
        raise RealignArtifactError("stage state is missing")
    try:
        return deserialize_state(item.path.read_bytes())
    except Exception as error:
        raise RealignArtifactError(f"stage state differs: {error}") from error


def _validate_delivery_semantics(
    members: Mapping[str, TrustedFile], selected_tree_count: int
) -> None:
    try:
        inference = json.loads(
            members["frozen_catboost/inference_manifest.json"].path.read_text(
                encoding="utf-8"
            )
        )
        policy = json.loads(
            members["policy/policy.json"].path.read_text(encoding="utf-8")
        )
    except Exception as error:
        raise RealignArtifactError("delivery policy or inference manifest is unreadable") from error
    if inference != {
        "schema_version": 1,
        "selected_tree_count": selected_tree_count,
        "model_sha256": members["frozen_catboost/model.cbm"].sha256,
        "preprocessing_sha256": members[
            "frozen_catboost/preprocessing_state.json"
        ].sha256,
    }:
        raise RealignArtifactError("inference manifest differs")
    if policy != {
        "campaign_id": "catboost_50_50_realign_v2",
        "rule_safe": True,
        "submission_package": False,
        "tabm_weight": "0.50",
        "test_independent": True,
    }:
        raise RealignArtifactError("delivery policy differs")


def _validate_member_allowlist(kind: str, members: Mapping[str, TrustedFile]) -> None:
    names = set(members)
    if not names or len(names) > _MAX_COUNT or any(not _safe_name(name) for name in names):
        raise RealignArtifactError("artifact member set differs")
    forbidden = ("test.csv", "submission.csv", "sample_submission.csv")
    if any(any(part.lower() == item for item in forbidden) for name in names for part in PurePosixPath(name).parts):
        raise RealignArtifactError("artifact contains forbidden inference material")
    if kind == "realign_delivery_v1" and names != _DELIVERY_NAMES:
        raise RealignArtifactError("delivery member set differs")
    if kind in {"realign_resume_v1", "realign_review_v1"}:
        if "state/stage_state.json" not in names:
            raise RealignArtifactError("stage state is missing")
        allowed_roots = {"state", "jobs", "decision", "logs"}
        if any(PurePosixPath(name).parts[0] not in allowed_roots for name in names):
            raise RealignArtifactError("artifact member set differs")
        if kind == "realign_review_v1" and any(
            name.endswith((".pt", ".cbsnapshot", ".cbm")) for name in names
        ):
            raise RealignArtifactError("review contains restart or model bytes")


_FULL_MODEL_MEMBERS = {
    "jobs/catboost_full_2024/job.json",
    "jobs/catboost_full_2024/metrics.json",
    "jobs/catboost_full_2024/model.cbm",
    "jobs/catboost_full_2024/preprocessing_state.json",
}


def _validate_completed_resume_semantics(
    state: CampaignState,
    member_names: set[str],
    member_sha256: Callable[[str], str],
    read_json: Callable[[str], object],
) -> None:
    if not _FULL_MODEL_MEMBERS.issubset(member_names):
        raise RealignArtifactError("completed resume requires full model evidence")
    try:
        job = read_json("jobs/catboost_full_2024/job.json")
        metrics = read_json("jobs/catboost_full_2024/metrics.json")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RealignArtifactError("full model evidence is unreadable") from error
    expected_job = {
        "schema_version": 1,
        "campaign_id": "catboost_50_50_realign_v2",
        "job_id": "catboost_full_2024",
        "kind": "full_fit",
        "contract_sha256": state.bindings.contract_sha256,
        "input_manifest_sha256": state.bindings.input_manifest_sha256,
        "code_sha256": state.bindings.code_sha256,
        "decision_sha256": state.decision_sha256,
        "selected_tree_count": state.selected_tree_count,
    }
    metric_keys = {
        "schema_version",
        "job_id",
        "kind",
        "train_rows",
        "valid_rows",
        "selected_tree_count",
        "model_sha256",
        "preprocessing_sha256",
        "snapshot_sha256",
        "decision_sha256",
    }
    valid_metrics = (
        type(metrics) is dict
        and set(metrics) == metric_keys
        and metrics["schema_version"] == 1
        and metrics["job_id"] == "catboost_full_2024"
        and metrics["kind"] == "full_fit"
        and type(metrics["train_rows"]) is int
        and metrics["train_rows"] > 0
        and metrics["valid_rows"] is None
        and metrics["selected_tree_count"] == state.selected_tree_count
        and metrics["decision_sha256"] == state.decision_sha256
        and metrics["model_sha256"]
        == member_sha256("jobs/catboost_full_2024/model.cbm")
        and metrics["preprocessing_sha256"]
        == member_sha256("jobs/catboost_full_2024/preprocessing_state.json")
        and (
            metrics["snapshot_sha256"] is None
            or _valid_sha(metrics["snapshot_sha256"])
        )
    )
    if job != expected_job or not valid_metrics:
        raise RealignArtifactError("full model evidence differs")


def _bundle_identity(
    kind: str, source: CampaignFiles, members: Mapping[str, TrustedFile], bindings: Bindings
) -> tuple[str, int | None]:
    if kind == "realign_delivery_v1":
        source.require_completed_delivery()
        decision_status, selected = _read_decision(
            members["decision/alignment_decision.json"].path
        )
        if decision_status != "promoted" or selected is None:
            raise RealignArtifactError("completed delivery requires a promoted decision")
        _validate_delivery_semantics(members, selected)
        return "completed", selected
    state = _state_from_member(members)
    if state.bindings != bindings or state.status != source.status:
        raise RealignArtifactError("stage state bindings or status differ")
    decision_member = members.get("decision/alignment_decision.json")
    if state.decision_sha256 is not None and (
        decision_member is None or decision_member.sha256 != state.decision_sha256
    ):
        raise RealignArtifactError("stage decision SHA-256 differs")
    if (
        kind == "realign_resume_v1"
        and state.status == "completed"
    ):
        _validate_completed_resume_semantics(
            state,
            set(members),
            lambda name: members[name].sha256,
            lambda name: json.loads(members[name].path.read_text(encoding="utf-8")),
        )
    return state.status, state.selected_tree_count


def _write_verified_bundle(
    kind: str,
    source: CampaignFiles,
    members: Mapping[str, TrustedFile],
    output: Path,
    bindings: Bindings,
    check_deadline: Callable[[], None] | None = None,
) -> Path:
    _validate_member_allowlist(kind, members)
    binding_values = bindings_payload(bindings)
    status, selected = _bundle_identity(kind, source, members, bindings)
    evidence = {
        name: {"size": item.path.stat().st_size, "sha256": item.sha256}
        for name, item in sorted(members.items())
    }
    manifest = {
        "schema_version": 1,
        "artifact_kind": kind,
        "submission_package": False,
        "bindings": binding_values,
        "status": status,
        "selected_tree_count": selected,
        "members": evidence,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise RealignArtifactError("artifact output already exists")
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RealignArtifactError("artifact temporary output already exists")
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr(_zip_info("manifest.json"), _canonical_json(manifest))
            for name, item in sorted(members.items()):
                if check_deadline is not None:
                    check_deadline()
                path = Path(item.path)
                _reject_symlink_ancestors(path)
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
                try:
                    descriptor = os.open(path, flags)
                except OSError as error:
                    raise RealignArtifactError(f"trusted member cannot be opened: {name}") from error
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    os.close(descriptor)
                    raise RealignArtifactError(f"trusted member is not regular: {name}")
                digest = sha256()
                size = 0
                with os.fdopen(descriptor, "rb") as stream, archive.open(_zip_info(name), "w") as sink:
                    while chunk := stream.read(1024 * 1024):
                        if check_deadline is not None:
                            check_deadline()
                        digest.update(chunk)
                        size += len(chunk)
                        sink.write(chunk)
                if evidence[name] != {"size": size, "sha256": digest.hexdigest()}:
                    raise RealignArtifactError(f"trusted member changed while reading: {name}")
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def write_resume_bundle(source: CampaignFiles, output: Path, bindings: Bindings) -> Path:
    return _write_verified_bundle(
        "realign_resume_v1", source, source.resume_members(), output, bindings
    )


def write_review_bundle(source: CampaignFiles, output: Path, bindings: Bindings) -> Path:
    return _write_verified_bundle(
        "realign_review_v1", source, source.review_members(), output, bindings
    )


def write_delivery_bundle(source: CampaignFiles, output: Path, bindings: Bindings) -> Path:
    source.require_completed_delivery()
    return _write_verified_bundle(
        "realign_delivery_v1", source, source.delivery_members(), output, bindings
    )


def _checked_infos(archive: ZipFile) -> dict[str, ZipInfo]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise RealignArtifactError("duplicate ZIP member")
    if len(infos) > _MAX_COUNT + 1:
        raise RealignArtifactError("ZIP member count exceeds limit")
    total = 0
    output: dict[str, ZipInfo] = {}
    for info in infos:
        if not _safe_name(info.filename):
            raise RealignArtifactError("unsafe ZIP member")
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise RealignArtifactError("unsafe ZIP member")
        if info.file_size < 0 or info.file_size > _MAX_MEMBER:
            raise RealignArtifactError("ZIP member size exceeds limit")
        ratio = info.file_size / max(info.compress_size, 1)
        if ratio > _MAX_RATIO:
            raise RealignArtifactError("ZIP compression ratio exceeds limit")
        total += info.file_size
        output[info.filename] = info
    if total > _MAX_TOTAL:
        raise RealignArtifactError("ZIP total size exceeds limit")
    return output


def _verify_bundle(
    path: Path, kind: str, expected_bindings: Bindings
) -> VerifiedBundle:
    try:
        with ZipFile(path) as archive:
            infos = _checked_infos(archive)
            if "manifest.json" not in infos:
                raise RealignArtifactError("artifact manifest is missing")
            manifest = json.loads(archive.read("manifest.json"))
            expected_keys = {
                "schema_version",
                "artifact_kind",
                "submission_package",
                "bindings",
                "status",
                "selected_tree_count",
                "members",
            }
            if (
                type(manifest) is not dict
                or set(manifest) != expected_keys
                or manifest["schema_version"] != 1
                or manifest["artifact_kind"] != kind
                or manifest["submission_package"] is not False
                or manifest["bindings"] != bindings_payload(expected_bindings)
                or type(manifest["members"]) is not dict
                or set(manifest["members"]) != set(infos) - {"manifest.json"}
            ):
                if type(manifest) is dict and manifest.get("bindings") != bindings_payload(expected_bindings):
                    raise RealignArtifactError("artifact bindings differ")
                raise RealignArtifactError("artifact manifest differs")
            dummy = {name: TrustedFile(Path(name), "0" * 64) for name in manifest["members"]}
            _validate_member_allowlist(kind, dummy)
            for name, evidence in manifest["members"].items():
                if (
                    type(evidence) is not dict
                    or set(evidence) != {"size", "sha256"}
                    or type(evidence["size"]) is not int
                    or evidence["size"] < 0
                    or not _valid_sha(evidence["sha256"])
                ):
                    raise RealignArtifactError(f"member evidence differs: {name}")
                digest = sha256()
                size = 0
                with archive.open(name) as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                        size += len(chunk)
                if evidence != {"size": size, "sha256": digest.hexdigest()}:
                    raise RealignArtifactError(f"member content differs: {name}")
            if kind == "realign_delivery_v1":
                decision = json.loads(archive.read("decision/alignment_decision.json"))
                if (
                    type(decision) is not dict
                    or decision.get("status") != "promoted"
                    or decision.get("selected_tree_count") != manifest["selected_tree_count"]
                    or manifest["status"] != "completed"
                ):
                    raise RealignArtifactError("delivery decision differs")
                inference = json.loads(
                    archive.read("frozen_catboost/inference_manifest.json")
                )
                policy = json.loads(archive.read("policy/policy.json"))
                if inference != {
                    "schema_version": 1,
                    "selected_tree_count": manifest["selected_tree_count"],
                    "model_sha256": manifest["members"][
                        "frozen_catboost/model.cbm"
                    ]["sha256"],
                    "preprocessing_sha256": manifest["members"][
                        "frozen_catboost/preprocessing_state.json"
                    ]["sha256"],
                }:
                    raise RealignArtifactError("inference manifest differs")
                if policy != {
                    "campaign_id": "catboost_50_50_realign_v2",
                    "rule_safe": True,
                    "submission_package": False,
                    "tabm_weight": "0.50",
                    "test_independent": True,
                }:
                    raise RealignArtifactError("delivery policy differs")
            else:
                state = deserialize_state(archive.read("state/stage_state.json"))
                if (
                    state.bindings != expected_bindings
                    or state.status != manifest["status"]
                    or state.selected_tree_count != manifest["selected_tree_count"]
                    or (
                        state.decision_sha256 is not None
                        and (
                            "decision/alignment_decision.json"
                            not in manifest["members"]
                            or manifest["members"][
                                "decision/alignment_decision.json"
                            ]["sha256"]
                            != state.decision_sha256
                        )
                    )
                ):
                    raise RealignArtifactError("artifact state differs")
                if kind == "realign_resume_v1" and state.status == "completed":
                    _validate_completed_resume_semantics(
                        state,
                        set(manifest["members"]),
                        lambda name: manifest["members"][name]["sha256"],
                        lambda name: json.loads(archive.read(name)),
                    )
    except RealignArtifactError:
        raise
    except (BadZipFile, OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RealignArtifactError(f"artifact ZIP is invalid: {error}") from error
    return VerifiedBundle(
        path=Path(path),
        artifact_kind=kind,
        bindings=expected_bindings,
        status=manifest["status"],
        selected_tree_count=manifest["selected_tree_count"],
        submission_package=False,
        members=MappingProxyType(dict(manifest["members"])),
    )


def verify_resume_bundle(path: Path, expected_bindings: Bindings) -> VerifiedBundle:
    return _verify_bundle(path, "realign_resume_v1", expected_bindings)


def verify_review_bundle(path: Path, expected_bindings: Bindings) -> VerifiedBundle:
    return _verify_bundle(path, "realign_review_v1", expected_bindings)


def verify_delivery_bundle(path: Path, expected_bindings: Bindings) -> VerifiedBundle:
    return _verify_bundle(path, "realign_delivery_v1", expected_bindings)


def restore_resume(
    path: Path, destination: Path, expected_bindings: Bindings
) -> RestoredRun:
    verified = verify_resume_bundle(path, expected_bindings)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise RealignArtifactError("resume destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        with ZipFile(path) as archive:
            infos = _checked_infos(archive)
            for name in sorted(verified.members):
                target = temporary.joinpath(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix(target.suffix + ".tmp")
                digest = sha256()
                size = 0
                with archive.open(infos[name]) as source, partial.open("xb") as sink:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                        size += len(chunk)
                        sink.write(chunk)
                expected = verified.members[name]
                if expected != {"size": size, "sha256": digest.hexdigest()}:
                    raise RealignArtifactError(f"member content differs: {name}")
                os.replace(partial, target)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    state_path = destination / "state/stage_state.json"
    state = deserialize_state(state_path.read_bytes())
    return RestoredRun(destination, state.status, state_path)


__all__ = [
    "Bindings",
    "CampaignFiles",
    "RealignArtifactError",
    "RestoredRun",
    "TrustedFile",
    "VerifiedBundle",
    "restore_resume",
    "verify_delivery_bundle",
    "verify_resume_bundle",
    "verify_review_bundle",
    "write_delivery_bundle",
    "write_resume_bundle",
    "write_review_bundle",
]
