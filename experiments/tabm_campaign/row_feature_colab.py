"""Direct-upload and recovery primitives for the Stage P Colab handoff."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tempfile
import threading
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

from .artifacts import (
    ArtifactError,
    StageEvidence,
    VerifiedResume,
    verify_resume_bundle,
    verify_review_bundle,
    write_stage_bundles,
)


class RowFeatureColabError(RuntimeError):
    """Raised before an untrusted Colab input can reach training."""


@dataclass(frozen=True)
class VerifiedInput:
    data_dir: Path
    input_manifest_sha256: str
    train_sha256: str
    history_sha256: str


@dataclass(frozen=True)
class VerifiedDelivery:
    path: Path
    sha256: str
    member_sha256: Mapping[str, str]


@dataclass(frozen=True)
class EmergencySnapshot:
    path: Path
    sha256: str
    manifest_sha256: str
    candidate_id: str
    epoch: int


_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)
_DATA_MEMBERS = {"input_manifest.json", "train.csv", "trackman_history.csv"}
_CSV_MEMBERS = {"train.csv", "trackman_history.csv"}
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_DATA_MEMBER_BYTES = 8 * 1024 * 1024 * 1024
_MAX_DATA_TOTAL_BYTES = 12 * 1024 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 200.0
_MIN_RATIO_BYTES = 64 * 1024
_DELIVERY_MEMBERS = {
    "tabm_row_feature_stage_P_review_bundle.zip",
    "tabm_row_feature_stage_P_resume_bundle.zip",
    "row_feature_proxy.log",
    "delivery_manifest.json",
}
_DELIVERY_PAYLOAD_MEMBERS = _DELIVERY_MEMBERS - {"delivery_manifest.json"}
_ACTIVE_FILES = {
    "job.json",
    "checkpoint.pt",
    "checkpoint_meta.json",
    "best_checkpoint.pt",
    "worker.log",
    "progress.jsonl",
}


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise RowFeatureColabError(f"{label} must be a lowercase SHA-256")
    return value


def _safe_source_files(data_dir: Path) -> dict[str, Path]:
    try:
        root = Path(os.path.abspath(os.fspath(data_dir)))
    except (OSError, TypeError, ValueError) as error:
        raise RowFeatureColabError("data directory path is invalid") from error
    for component in (root, *root.parents):
        if component.is_symlink():
            raise RowFeatureColabError("data directory must not contain a symlink ancestor")
    if not root.is_dir():
        raise RowFeatureColabError("data directory must be a regular directory")

    candidates: dict[str, list[Path]] = {name: [] for name in _CSV_MEMBERS}
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        for name in [*names, *files]:
            if (directory_path / name).is_symlink():
                raise RowFeatureColabError("data directory must not contain symlinks")
        names[:] = sorted(names)
        for name in _CSV_MEMBERS.intersection(files):
            candidates[name].append(directory_path / name)

    result: dict[str, Path] = {}
    for name, matches in candidates.items():
        if len(matches) != 1:
            raise RowFeatureColabError(
                f"data directory must contain exactly one {name}; found={len(matches)}"
            )
        path = matches[0]
        if path.parent != root:
            raise RowFeatureColabError(f"{name} must be at the data directory top level")
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise RowFeatureColabError(f"{name} must be a regular file")
        result[name] = path
    return result


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _publish_prepared_archive(
    temporary: str,
    destination: Path,
    *,
    replace: bool,
) -> None:
    if replace:
        if destination.exists() and not destination.is_symlink():
            try:
                mode = destination.lstat().st_mode
            except OSError as error:
                raise RowFeatureColabError("cannot inspect existing output") from error
            if not stat.S_ISREG(mode):
                raise RowFeatureColabError(
                    "replace output must be a regular file or symlink entry"
                )
        os.replace(temporary, destination)
        return
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise RowFeatureColabError(f"output already exists: {destination}") from error
    except OSError as error:
        raise RowFeatureColabError(f"cannot publish output exclusively: {error}") from error
    os.unlink(temporary)


def prepare_input_archive(
    data_dir: str | Path,
    output: str | Path,
    *,
    expected_train_sha256: str,
    campaign_config_sha256: str,
    replace: bool = False,
) -> Path:
    """Write the exact deterministic three-member Stage P input archive."""

    expected_train_sha256 = _require_sha(
        expected_train_sha256, "expected train SHA-256"
    )
    campaign_config_sha256 = _require_sha(
        campaign_config_sha256, "campaign config SHA-256"
    )
    sources = _safe_source_files(Path(data_dir))
    evidence = {
        name: {"size": path.stat().st_size, "sha256": file_sha256(path)}
        for name, path in sorted(sources.items())
    }
    if evidence["train.csv"]["sha256"] != expected_train_sha256:
        raise RowFeatureColabError("train.csv SHA-256 differs from the frozen contract")
    manifest = canonical_json(
        {
            "schema_version": 1,
            "artifact_kind": "tabm_row_feature_input",
            "stage": "P",
            "campaign_config_sha256": campaign_config_sha256,
            "members": evidence,
        }
    )

    destination = Path(output)
    if (destination.exists() or destination.is_symlink()) and not replace:
        raise RowFeatureColabError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}-", dir=destination.parent
    )
    os.close(descriptor)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            archive.writestr(_zip_info("input_manifest.json"), manifest)
            for name in sorted(sources):
                _stream_zip_member(archive, name, sources[name])
        for name, path in sources.items():
            if file_sha256(path) != evidence[name]["sha256"]:
                raise RowFeatureColabError(f"source changed during publication: {name}")
        _publish_prepared_archive(temporary, destination, replace=replace)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def _safe_zip_info(info: ZipInfo) -> bool:
    name = info.filename
    path = PurePosixPath(name)
    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    return bool(
        name
        and "\\" not in name
        and not path.is_absolute()
        and ".." not in path.parts
        and len(path.parts) == 1
        and not info.is_dir()
        and not stat.S_ISLNK(mode)
        and file_type in {0, stat.S_IFREG}
    )


def _validated_data_archive(
    archive: ZipFile,
    *,
    expected_train_sha256: str,
    campaign_config_sha256: str,
) -> tuple[dict[str, Mapping[str, object]], bytes]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise RowFeatureColabError("data archive has duplicate members")
    if set(names) != _DATA_MEMBERS:
        raise RowFeatureColabError("data archive must contain exactly the three declared members")
    if any(not _safe_zip_info(info) for info in infos):
        raise RowFeatureColabError("data archive contains an unsafe member")

    total = 0
    for info in infos:
        limit = _MAX_MANIFEST_BYTES if info.filename == "input_manifest.json" else _MAX_DATA_MEMBER_BYTES
        if info.file_size > limit:
            raise RowFeatureColabError(f"data archive member exceeds its size limit: {info.filename}")
        total += info.file_size
        if total > _MAX_DATA_TOTAL_BYTES:
            raise RowFeatureColabError("data archive exceeds its total size limit")
        if info.file_size >= _MIN_RATIO_BYTES:
            if info.compress_size <= 0 or info.file_size / info.compress_size > _MAX_COMPRESSION_RATIO:
                raise RowFeatureColabError(
                    f"data archive member exceeds the compression ratio limit: {info.filename}"
                )

    try:
        manifest_bytes = archive.read("input_manifest.json")
        manifest = json.loads(manifest_bytes)
    except (KeyError, UnicodeError, json.JSONDecodeError) as error:
        raise RowFeatureColabError("input manifest is unreadable") from error
    expected_keys = {
        "schema_version",
        "artifact_kind",
        "stage",
        "campaign_config_sha256",
        "members",
    }
    if (
        type(manifest) is not dict
        or set(manifest) != expected_keys
        or manifest["schema_version"] != 1
        or manifest["artifact_kind"] != "tabm_row_feature_input"
        or manifest["stage"] != "P"
        or manifest["campaign_config_sha256"] != campaign_config_sha256
        or type(manifest["members"]) is not dict
        or set(manifest["members"]) != _CSV_MEMBERS
    ):
        raise RowFeatureColabError("input manifest contract or member schema differs")
    members: dict[str, Mapping[str, object]] = {}
    for name in sorted(_CSV_MEMBERS):
        raw = manifest["members"][name]
        info = archive.getinfo(name)
        if (
            type(raw) is not dict
            or set(raw) != {"size", "sha256"}
            or type(raw["size"]) is not int
            or raw["size"] < 0
            or raw["size"] != info.file_size
            or type(raw["sha256"]) is not str
            or _SHA_RE.fullmatch(raw["sha256"]) is None
        ):
            raise RowFeatureColabError(f"input manifest evidence is invalid: {name}")
        members[name] = raw
    if members["train.csv"]["sha256"] != expected_train_sha256:
        raise RowFeatureColabError("train.csv SHA-256 differs from the frozen contract")
    return members, manifest_bytes


def verify_and_extract_input_archive(
    source: str | Path,
    destination: str | Path,
    *,
    expected_train_sha256: str,
    campaign_config_sha256: str,
) -> VerifiedInput:
    """Recursively verify a flat input ZIP and atomically publish its files."""

    expected_train_sha256 = _require_sha(
        expected_train_sha256, "expected train SHA-256"
    )
    campaign_config_sha256 = _require_sha(
        campaign_config_sha256, "campaign config SHA-256"
    )
    source = Path(source)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise RowFeatureColabError("input extraction destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        with ZipFile(source, "r") as archive:
            members, manifest_bytes = _validated_data_archive(
                archive,
                expected_train_sha256=expected_train_sha256,
                campaign_config_sha256=campaign_config_sha256,
            )
            for name in sorted(_CSV_MEMBERS):
                target = temporary / name
                expected = members[name]
                digest = sha256()
                written = 0
                with archive.open(name, "r") as input_stream, target.open("xb") as output:
                    while chunk := input_stream.read(1024 * 1024):
                        written += len(chunk)
                        if written > expected["size"]:
                            raise RowFeatureColabError(f"data member expanded beyond its manifest: {name}")
                        output.write(chunk)
                        digest.update(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if written != expected["size"] or digest.hexdigest() != expected["sha256"]:
                    raise RowFeatureColabError(f"data member SHA-256 differs: {name}")
            (temporary / "input_manifest.json").write_bytes(manifest_bytes)
        os.replace(temporary, destination)
    except RowFeatureColabError:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    except (BadZipFile, EOFError, OSError, RuntimeError, ValueError) as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise RowFeatureColabError(f"cannot verify data archive: {error}") from error
    return VerifiedInput(
        data_dir=destination,
        input_manifest_sha256=sha256(manifest_bytes).hexdigest(),
        train_sha256=str(members["train.csv"]["sha256"]),
        history_sha256=str(members["trackman_history.csv"]["sha256"]),
    )


def _read_resume_state(path: Path, expected_sha256: str) -> dict[str, object]:
    try:
        with ZipFile(path, "r") as archive:
            info = archive.getinfo("state/stage_state.json")
            if info.file_size > _MAX_MANIFEST_BYTES:
                raise RowFeatureColabError("resume stage state exceeds its size limit")
            value = archive.read(info)
    except RowFeatureColabError:
        raise
    except (BadZipFile, KeyError, OSError) as error:
        raise RowFeatureColabError("resume has no readable Stage P state") from error
    if sha256(value).hexdigest() != expected_sha256:
        raise RowFeatureColabError("resume stage state changed after verification")
    try:
        state = json.loads(value)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RowFeatureColabError("resume Stage P state is not valid JSON") from error
    if type(state) is not dict:
        raise RowFeatureColabError("resume Stage P state must be an object")
    return state


def verify_stage_p_resume(
    source: str | Path,
    *,
    expected_contract_sha256: str,
    expected_code_sha256: str,
    verified_input: VerifiedInput,
) -> VerifiedResume:
    """Reject a resume unless its Stage P code, contract, and train input match."""

    expected_contract_sha256 = _require_sha(
        expected_contract_sha256, "expected contract SHA-256"
    )
    expected_code_sha256 = _require_sha(expected_code_sha256, "expected code SHA-256")
    if not isinstance(verified_input, VerifiedInput):
        raise RowFeatureColabError("verified input receipt is required")
    try:
        verified = verify_resume_bundle(source)
    except (ArtifactError, OSError) as error:
        raise RowFeatureColabError(f"resume bundle is untrusted: {error}") from error
    if verified.version != "P":
        raise RowFeatureColabError("resume bundle is not Stage P")
    if verified.campaign_config_sha256 != expected_contract_sha256:
        raise RowFeatureColabError("resume contract SHA-256 differs")
    if (
        verified.member_sha256.get("config/row_feature_proxy_v1.json")
        != expected_contract_sha256
    ):
        raise RowFeatureColabError("resume does not contain the exact contract bytes")
    state_sha = verified.member_sha256.get("state/stage_state.json")
    if state_sha is None:
        raise RowFeatureColabError("resume has no Stage P state member")
    state = _read_resume_state(Path(source), state_sha)
    expected = {
        "version": "P",
        "campaign_config_sha256": expected_contract_sha256,
        "official_train_sha256": verified_input.train_sha256,
        "trackman_history_sha256": verified_input.history_sha256,
        "input_manifest_sha256": verified_input.input_manifest_sha256,
        "code_sha256": expected_code_sha256,
    }
    for name, value in expected.items():
        if state.get(name) != value:
            raise RowFeatureColabError(f"resume {name} binding differs")
    try:
        from . import row_feature_proxy as proxy
        from .row_feature_contracts import load_row_feature_proxy_contract

        contract = load_row_feature_proxy_contract()
        jobs = proxy.build_proxy_jobs(contract)
        state = proxy._validate_state_header(
            state,
            contract_sha=expected_contract_sha256,
            train_sha=verified_input.train_sha256,
            history_sha=verified_input.history_sha256,
            input_manifest_sha=verified_input.input_manifest_sha256,
            code_sha=expected_code_sha256,
            jobs=jobs,
        )
        rows = proxy._validated_rows(state, jobs)
        decision = proxy._decision(jobs, rows, contract)
        if state["stage_complete"] != (decision.status in {"complete", "blocked"}):
            raise RowFeatureColabError(
                "resume stage_complete contradicts its results"
            )
        if state["prior_manifest_sha256"] != verified.prior_manifest_sha256:
            raise RowFeatureColabError("resume prior manifest binding differs")
        if set(verified.member_sha256) != proxy._resume_expected_members(rows):
            raise RowFeatureColabError(
                "resume member set differs from its Stage P state"
            )
        for row in rows.values():
            for binding in row["artifacts"].values():  # type: ignore[union-attr]
                if verified.member_sha256.get(binding["path"]) != binding["sha256"]:
                    raise RowFeatureColabError("resume artifact binding differs")
        with ZipFile(source, "r") as archive:
            if archive.read("metrics/job_results.json") != proxy._canonical_json(
                state["results"]
            ):
                raise RowFeatureColabError("resume metrics differ from Stage P state")
            if archive.read("logs/stage.log") != proxy._stage_log(state, decision):
                raise RowFeatureColabError("resume log differs from Stage P state")
            for row in rows.values():
                if row["status"] != "completed":
                    continue
                prediction = f"predictions/{row['candidate_id']}.csv"
                artifact = row["artifacts"]["predictions.csv"]  # type: ignore[index]
                if (
                    verified.member_sha256.get(prediction) != artifact["sha256"]
                    or proxy._zip_member_sha256(archive, prediction)
                    != artifact["sha256"]
                ):
                    raise RowFeatureColabError(
                        "resume completed prediction copies differ"
                    )
    except RowFeatureColabError:
        raise
    except Exception as error:
        raise RowFeatureColabError(f"resume Stage P state is invalid: {error}") from error
    return verified


def register_verified_uploaded_resume(
    source: str | Path,
    *,
    expected_contract_sha256: str,
    expected_code_sha256: str,
    verified_input: VerifiedInput,
    on_verified_resume: Callable[[Path], None],
) -> Path:
    """Register an uploaded resume only after all Stage P bindings verify."""

    if not callable(on_verified_resume):
        raise RowFeatureColabError("verified resume callback must be callable")
    path = Path(source)
    verify_stage_p_resume(
        path,
        expected_contract_sha256=expected_contract_sha256,
        expected_code_sha256=expected_code_sha256,
        verified_input=verified_input,
    )
    on_verified_resume(path)
    return path


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RowFeatureColabError(f"{label} must be a regular non-symlink file")
    return path


def _stream_zip_member(archive: ZipFile, name: str, path: Path) -> None:
    info = _zip_info(name)
    info.file_size = path.stat().st_size
    with path.open("rb") as source, archive.open(info, "w") as destination:
        while chunk := source.read(1024 * 1024):
            destination.write(chunk)


def build_delivery(
    *,
    review_bundle: str | Path,
    resume_bundle: str | Path,
    log_path: str | Path,
    output_path: str | Path,
    input_manifest_sha256: str,
    embedded_runtime_sha256: str,
    campaign_config_sha256: str,
    code_sha256: str,
) -> Path:
    """Create the exact four-member delivery only after nested verification."""

    bindings = {
        "input_manifest_sha256": _require_sha(
            input_manifest_sha256, "input manifest SHA-256"
        ),
        "embedded_runtime_sha256": _require_sha(
            embedded_runtime_sha256, "embedded runtime SHA-256"
        ),
        "campaign_config_sha256": _require_sha(
            campaign_config_sha256, "campaign config SHA-256"
        ),
        "code_sha256": _require_sha(code_sha256, "code SHA-256"),
    }
    review_bundle = _regular_file(Path(review_bundle), "review bundle")
    resume_bundle = _regular_file(Path(resume_bundle), "resume bundle")
    log_path = _regular_file(Path(log_path), "proxy log")
    try:
        review = verify_review_bundle(review_bundle)
        resume = verify_resume_bundle(resume_bundle)
    except ArtifactError as error:
        raise RowFeatureColabError(f"nested bundle is untrusted: {error}") from error
    if (
        review.version != "P"
        or resume.version != "P"
        or review.campaign_config_sha256 != bindings["campaign_config_sha256"]
        or resume.campaign_config_sha256 != bindings["campaign_config_sha256"]
        or review.member_sha256.get("config/row_feature_proxy_v1.json")
        != bindings["campaign_config_sha256"]
        or resume.member_sha256.get("config/row_feature_proxy_v1.json")
        != bindings["campaign_config_sha256"]
    ):
        raise RowFeatureColabError("nested Stage P version or contract binding differs")

    sources = {
        "tabm_row_feature_stage_P_review_bundle.zip": review_bundle,
        "tabm_row_feature_stage_P_resume_bundle.zip": resume_bundle,
        "row_feature_proxy.log": log_path,
    }
    evidence = {
        name: {"size": path.stat().st_size, "sha256": file_sha256(path)}
        for name, path in sources.items()
    }
    manifest = canonical_json(
        {
            "schema_version": 1,
            "artifact_kind": "tabm_row_feature_stage_P_delivery",
            "review_only": True,
            "stage": "P",
            "bindings": bindings,
            "members": evidence,
        }
    )
    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise RowFeatureColabError("delivery output already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}-", dir=output_path.parent
    )
    os.close(descriptor)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for name in sorted(sources):
                _stream_zip_member(archive, name, sources[name])
            archive.writestr(_zip_info("delivery_manifest.json"), manifest)
        os.replace(temporary, output_path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    verify_delivery(output_path, **bindings)
    return output_path


def verify_delivery(
    source: str | Path,
    *,
    input_manifest_sha256: str,
    embedded_runtime_sha256: str,
    campaign_config_sha256: str,
    code_sha256: str,
) -> VerifiedDelivery:
    """Verify the outer manifest and recursively verify both nested bundles."""

    expected_bindings = {
        "input_manifest_sha256": _require_sha(
            input_manifest_sha256, "input manifest SHA-256"
        ),
        "embedded_runtime_sha256": _require_sha(
            embedded_runtime_sha256, "embedded runtime SHA-256"
        ),
        "campaign_config_sha256": _require_sha(
            campaign_config_sha256, "campaign config SHA-256"
        ),
        "code_sha256": _require_sha(code_sha256, "code SHA-256"),
    }
    source = _regular_file(Path(source), "delivery")
    temporary_root: Path | None = None
    try:
        with ZipFile(source, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or set(names) != _DELIVERY_MEMBERS:
                raise RowFeatureColabError("delivery member set differs")
            if any(not _safe_zip_info(info) for info in infos):
                raise RowFeatureColabError("delivery contains an unsafe member")
            manifest_info = archive.getinfo("delivery_manifest.json")
            if manifest_info.file_size > _MAX_MANIFEST_BYTES:
                raise RowFeatureColabError("delivery manifest exceeds its size limit")
            manifest_bytes = archive.read(manifest_info)
            manifest = json.loads(manifest_bytes)
            expected_keys = {
                "schema_version",
                "artifact_kind",
                "review_only",
                "stage",
                "bindings",
                "members",
            }
            if (
                type(manifest) is not dict
                or set(manifest) != expected_keys
                or manifest["schema_version"] != 1
                or manifest["artifact_kind"] != "tabm_row_feature_stage_P_delivery"
                or manifest["review_only"] is not True
                or manifest["stage"] != "P"
                or manifest["bindings"] != expected_bindings
                or type(manifest["members"]) is not dict
                or set(manifest["members"]) != _DELIVERY_PAYLOAD_MEMBERS
            ):
                raise RowFeatureColabError("delivery manifest schema or bindings differ")

            temporary_root = Path(tempfile.mkdtemp(prefix="row-feature-delivery-"))
            observed: dict[str, str] = {}
            for name in sorted(_DELIVERY_PAYLOAD_MEMBERS):
                raw = manifest["members"][name]
                info = archive.getinfo(name)
                if (
                    type(raw) is not dict
                    or set(raw) != {"size", "sha256"}
                    or type(raw["size"]) is not int
                    or raw["size"] != info.file_size
                    or type(raw["sha256"]) is not str
                    or _SHA_RE.fullmatch(raw["sha256"]) is None
                ):
                    raise RowFeatureColabError(f"delivery member evidence is invalid: {name}")
                target = temporary_root / name
                digest = sha256()
                written = 0
                with archive.open(info, "r") as input_stream, target.open("xb") as output:
                    while chunk := input_stream.read(1024 * 1024):
                        written += len(chunk)
                        if written > raw["size"]:
                            raise RowFeatureColabError(f"delivery member expanded beyond its manifest: {name}")
                        output.write(chunk)
                        digest.update(chunk)
                if written != raw["size"] or digest.hexdigest() != raw["sha256"]:
                    raise RowFeatureColabError(f"delivery member SHA-256 differs: {name}")
                observed[name] = digest.hexdigest()
            review = verify_review_bundle(
                temporary_root / "tabm_row_feature_stage_P_review_bundle.zip"
            )
            resume = verify_resume_bundle(
                temporary_root / "tabm_row_feature_stage_P_resume_bundle.zip"
            )
            if (
                review.version != "P"
                or resume.version != "P"
                or review.campaign_config_sha256 != expected_bindings["campaign_config_sha256"]
                or resume.campaign_config_sha256 != expected_bindings["campaign_config_sha256"]
                or review.member_sha256.get("config/row_feature_proxy_v1.json")
                != expected_bindings["campaign_config_sha256"]
                or resume.member_sha256.get("config/row_feature_proxy_v1.json")
                != expected_bindings["campaign_config_sha256"]
            ):
                raise RowFeatureColabError("recursively verified Stage P binding differs")
    except RowFeatureColabError:
        raise
    except (ArtifactError, BadZipFile, KeyError, OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RowFeatureColabError(f"cannot verify delivery: {error}") from error
    finally:
        if temporary_root is not None:
            shutil.rmtree(temporary_root, ignore_errors=True)
    return VerifiedDelivery(source, file_sha256(source), observed)


def _read_json_object(value: bytes, label: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RowFeatureColabError(f"{label} is not valid JSON") from error
    if type(parsed) is not dict:
        raise RowFeatureColabError(f"{label} must be a JSON object")
    return parsed


def _copy_active_candidate(source: Path, destination: Path, job) -> dict[str, object]:
    """Copy one atomic epoch publication without racing the next epoch."""

    from .worker import _job_payload, _job_sha, _training_source_sha256

    if source.is_symlink() or not source.is_dir():
        raise RowFeatureColabError("active candidate directory is unsafe")
    paths = sorted(source.iterdir(), key=lambda path: path.name)
    for path in paths:
        if path.is_symlink() or not path.is_file() or path.name not in _ACTIVE_FILES:
            raise RowFeatureColabError(f"active candidate has an unsafe artifact: {path.name}")
    required = {"job.json", "checkpoint.pt", "checkpoint_meta.json", "best_checkpoint.pt"}
    if not required.issubset(path.name for path in paths):
        raise RowFeatureColabError("active candidate has no complete epoch checkpoint")

    metadata_before = (source / "checkpoint_meta.json").read_bytes()
    metadata = _read_json_object(metadata_before, "active checkpoint metadata")
    expected_meta_keys = {
        "candidate_id",
        "epoch",
        "checkpoint",
        "adapter_state",
        "checkpoint_binding",
    }
    epoch = metadata.get("epoch")
    binding = metadata.get("checkpoint_binding")
    if (
        set(metadata) != expected_meta_keys
        or metadata.get("candidate_id") != job.candidate_id
        or type(epoch) is not int
        or epoch < 0
        or metadata.get("checkpoint") != "checkpoint.pt"
        or type(binding) is not dict
        or set(binding) != {
            "config_sha256",
            "cache_sha256",
            "training_source_sha256",
        }
        or binding.get("config_sha256") != _job_sha(job)
        or type(binding.get("cache_sha256")) is not str
        or _SHA_RE.fullmatch(binding["cache_sha256"]) is None
        or binding.get("training_source_sha256") != _training_source_sha256()
    ):
        raise RowFeatureColabError("active checkpoint binding is invalid")
    if (source / "job.json").read_bytes() != canonical_json(_job_payload(job)):
        raise RowFeatureColabError("active checkpoint job binding differs")

    destination.mkdir(parents=True, exist_ok=False)
    for path in paths:
        shutil.copyfile(path, destination / path.name)
    metadata_after = (source / "checkpoint_meta.json").read_bytes()
    if metadata_after != metadata_before:
        raise RowFeatureColabError("active checkpoint metadata changed during snapshot")
    if (destination / "checkpoint_meta.json").read_bytes() != metadata_before:
        raise RowFeatureColabError("copied active checkpoint metadata differs")
    return metadata


def _checkpoint_progress(path: Path, *, candidate_id: str, epoch: int) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise RowFeatureColabError("active progress evidence is unsafe")
    latest: dict[str, object] | None = None
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                if (
                    type(event) is dict
                    and event.get("event") == "EPOCH_CHECKPOINTED"
                    and event.get("candidate_id") == candidate_id
                    and event.get("epoch") == epoch
                ):
                    latest = event
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RowFeatureColabError("active checkpoint progress is unreadable") from error
    if latest is None:
        raise RowFeatureColabError("active checkpoint progress has not published this epoch")
    return latest


def _validated_active_row(snapshot_job_dir: Path, job) -> dict[str, object]:
    """Convert a copied epoch checkpoint into a resume row via Stage P validators."""

    from .row_feature_runtime import CampaignJobResult
    from . import row_feature_proxy as proxy
    from .worker import (
        _atomic_json,
        _current_code_provenance,
        _job_sha,
        _serialize_result,
        _training_source_sha256,
    )

    metadata = _read_json_object(
        (snapshot_job_dir / "checkpoint_meta.json").read_bytes(),
        "copied checkpoint metadata",
    )
    epoch = metadata["epoch"]
    progress = _checkpoint_progress(
        snapshot_job_dir / "progress.jsonl",
        candidate_id=job.candidate_id,
        epoch=epoch,
    )
    best_epoch = progress.get("best_epoch")
    best_brier = progress.get("best_brier")
    if (
        type(best_epoch) is not int
        or not 0 <= best_epoch <= epoch
        or type(best_brier) is not float
        or not 0.0 <= best_brier <= 1.0
    ):
        raise RowFeatureColabError("active checkpoint progress metric is invalid")
    try:
        import torch

        best_payload = torch.load(
            snapshot_job_dir / "best_checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise RowFeatureColabError("active best checkpoint cannot be loaded safely") from error
    if (
        type(best_payload) is not dict
        or set(best_payload) != {"model", "epoch"}
        or best_payload["epoch"] != best_epoch
        or not isinstance(best_payload["model"], Mapping)
    ):
        raise RowFeatureColabError("active best checkpoint epoch binding differs")
    binding = metadata["checkpoint_binding"]
    training_sha = _training_source_sha256()
    evidence = {
        "cache_digest": binding["cache_sha256"],
        **_current_code_provenance(training_sha),
    }
    result = CampaignJobResult(
        candidate_id=job.candidate_id,
        status="inconclusive",
        brier=best_brier,
        best_epoch=best_epoch,
        completed_epochs=epoch + 1,
        checkpoint=snapshot_job_dir / "checkpoint.pt",
        predictions_path=None,
        resource_evidence=evidence,
        failure="active_checkpoint_snapshot",
    )
    _atomic_json(
        snapshot_job_dir / "worker_result.json",
        _serialize_result(result, _job_sha(job)),
    )
    return proxy._validate_runtime_result(job, result, snapshot_job_dir)


def _copy_bound_artifacts(
    live_root: Path,
    snapshot_root: Path,
    rows: Mapping[str, dict[str, object]],
    *,
    excluded_candidate_id: str,
) -> None:
    for candidate_id, row in rows.items():
        if candidate_id == excluded_candidate_id:
            continue
        artifacts = row["artifacts"]
        for binding in artifacts.values():  # type: ignore[union-attr]
            source = live_root / str(binding["path"])
            if source.is_symlink() or not source.is_file():
                raise RowFeatureColabError(f"prior snapshot artifact is unsafe: {binding['path']}")
            target = snapshot_root / str(binding["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if file_sha256(target) != binding["sha256"]:
                raise RowFeatureColabError(f"prior snapshot artifact hash differs: {binding['path']}")


def publish_active_checkpoint_snapshot(
    *,
    live_output_dir: str | Path,
    snapshot_dir: str | Path,
    active_job,
    contract,
    contract_path: str | Path,
    contract_sha256: str,
    train_sha256: str,
    history_sha256: str,
    input_manifest_sha256: str,
    code_sha256: str,
    sequence: int,
) -> EmergencySnapshot:
    """Promote one complete active epoch, replacing no prior verified snapshot."""

    from . import row_feature_proxy as proxy

    contract_sha256 = _require_sha(contract_sha256, "contract SHA-256")
    train_sha256 = _require_sha(train_sha256, "train SHA-256")
    history_sha256 = _require_sha(history_sha256, "history SHA-256")
    input_manifest_sha256 = _require_sha(
        input_manifest_sha256, "input manifest SHA-256"
    )
    code_sha256 = _require_sha(code_sha256, "code SHA-256")
    if type(sequence) is not int or sequence < 0:
        raise RowFeatureColabError("snapshot sequence must be nonnegative")
    live_root = Path(live_output_dir)
    state_path = live_root / "stage_state.json"
    try:
        state = proxy._validate_state_header(
            proxy._read_json_bytes(state_path.read_bytes(), "local stage state"),
            contract_sha=contract_sha256,
            train_sha=train_sha256,
            history_sha=history_sha256,
            input_manifest_sha=input_manifest_sha256,
            code_sha=code_sha256,
            jobs=proxy.build_proxy_jobs(contract),
        )
        jobs = proxy.build_proxy_jobs(contract)
        rows = proxy._validated_rows(state, jobs)
        proxy._verify_local_artifacts(
            live_root,
            {
                key: value
                for key, value in rows.items()
                if key != active_job.candidate_id
            },
        )
    except Exception as error:
        raise RowFeatureColabError(f"cannot trust live Stage P state: {error}") from error

    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".active-", dir=snapshot_dir) as temporary:
        snapshot_root = Path(temporary) / "stage"
        snapshot_root.mkdir()
        _copy_bound_artifacts(
            live_root,
            snapshot_root,
            rows,
            excluded_candidate_id=active_job.candidate_id,
        )
        source_job_dir = live_root / "jobs" / active_job.candidate_id
        target_job_dir = snapshot_root / "jobs" / active_job.candidate_id
        metadata = _copy_active_candidate(source_job_dir, target_job_dir, active_job)
        active_row = _validated_active_row(target_job_dir, active_job)
        rows[active_job.candidate_id] = active_row
        decision = proxy._decision(jobs, rows, contract)
        snapshot_state = proxy._state_payload(
            contract_sha=contract_sha256,
            train_sha=train_sha256,
            history_sha=history_sha256,
            input_manifest_sha=input_manifest_sha256,
            code_sha=code_sha256,
            jobs=jobs,
            rows=rows,
            prior_manifest_sha=state["prior_manifest_sha256"],
            decision=decision,
        )
        config_bytes = Path(contract_path).read_bytes()
        if sha256(config_bytes).hexdigest() != contract_sha256:
            raise RowFeatureColabError("exact contract bytes changed during snapshot")
        review, resume = proxy._bundle_members(
            output_dir=snapshot_root,
            config_bytes=config_bytes,
            state=snapshot_state,
            rows=rows,
            decision=decision,
        )
        bundles = write_stage_bundles(
            Path(temporary) / "bundles",
            StageEvidence(
                "P",
                contract_sha256,
                state["prior_manifest_sha256"],
                review,
                resume,
            ),
            bundle_prefix="tabm_row_feature_stage",
        )
        if bundles.resume is None:
            raise RowFeatureColabError("active snapshot did not produce resume evidence")
        verified = verify_resume_bundle(bundles.resume)
        digest = file_sha256(bundles.resume)
        destination = snapshot_dir / (
            f"tabm_row_feature_stage_P_emergency_resume_{sequence:04d}_{digest[:12]}.zip"
        )
        if destination.exists() or destination.is_symlink():
            raise RowFeatureColabError("emergency snapshot destination already exists")
        os.replace(bundles.resume, destination)
        final = verify_resume_bundle(destination)
        if final.manifest_sha256 != verified.manifest_sha256:
            raise RowFeatureColabError("emergency snapshot changed during publication")
    return EmergencySnapshot(
        destination,
        digest,
        verified.manifest_sha256,
        active_job.candidate_id,
        int(metadata["epoch"]),
    )


def classify_and_verify_uploads(
    paths: list[Path],
    *,
    data_destination: Path,
    expected_train_sha256: str,
    campaign_config_sha256: str,
    expected_code_sha256: str,
) -> tuple[VerifiedInput, Path | None]:
    """Require exactly one data ZIP and at most one compatible Stage P resume."""

    if type(paths) is not list or len(paths) not in {1, 2}:
        raise RowFeatureColabError("upload must contain one data ZIP and optional resume ZIP")
    data_candidates: list[Path] = []
    other: list[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise RowFeatureColabError("uploaded archive must be a regular file")
        try:
            with ZipFile(path, "r") as archive:
                names = archive.namelist()
        except (BadZipFile, OSError) as error:
            raise RowFeatureColabError(f"uploaded ZIP is unreadable: {path.name}") from error
        if len(names) == len(set(names)) and set(names) == _DATA_MEMBERS:
            data_candidates.append(path)
        else:
            other.append(path)
    if len(data_candidates) != 1 or len(other) > 1:
        raise RowFeatureColabError(
            "upload must contain exactly one data ZIP and zero or one Stage P resume ZIP"
        )
    verified_input = verify_and_extract_input_archive(
        data_candidates[0],
        data_destination,
        expected_train_sha256=expected_train_sha256,
        campaign_config_sha256=campaign_config_sha256,
    )
    resume = other[0] if other else None
    if resume is not None:
        verify_stage_p_resume(
            resume,
            expected_contract_sha256=campaign_config_sha256,
            expected_code_sha256=expected_code_sha256,
            verified_input=verified_input,
        )
    return verified_input, resume


def publish_stable_state_snapshot(
    *,
    live_output_dir: str | Path,
    snapshot_dir: str | Path,
    contract,
    contract_path: str | Path,
    contract_sha256: str,
    train_sha256: str,
    history_sha256: str,
    input_manifest_sha256: str,
    code_sha256: str,
    sequence: int,
    candidate_id: str,
    allow_not_started: bool = False,
) -> EmergencySnapshot:
    """Publish the stable state written immediately after a candidate returns."""

    from . import row_feature_proxy as proxy

    live_root = Path(live_output_dir)
    jobs = proxy.build_proxy_jobs(contract)
    try:
        state, rows = proxy._load_local_state(
            live_root / "stage_state.json",
            contract_sha=contract_sha256,
            train_sha=train_sha256,
            history_sha=history_sha256,
            input_manifest_sha=input_manifest_sha256,
            code_sha=code_sha256,
            jobs=jobs,
            output_dir=live_root,
        )
    except Exception as error:
        raise RowFeatureColabError(f"cannot trust stable Stage P state: {error}") from error
    row = rows.get(candidate_id)
    if row is None or (
        row["disposition"] == "not_started" and not allow_not_started
    ):
        raise RowFeatureColabError("completed-candidate snapshot state was not advanced")
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stable-", dir=snapshot_dir) as temporary:
        snapshot_root = Path(temporary) / "stage"
        snapshot_root.mkdir()
        _copy_bound_artifacts(
            live_root,
            snapshot_root,
            rows,
            excluded_candidate_id="",
        )
        decision = proxy._decision(jobs, rows, contract)
        config_bytes = Path(contract_path).read_bytes()
        if sha256(config_bytes).hexdigest() != contract_sha256:
            raise RowFeatureColabError("exact contract bytes changed during snapshot")
        review, resume = proxy._bundle_members(
            output_dir=snapshot_root,
            config_bytes=config_bytes,
            state=state,
            rows=rows,
            decision=decision,
        )
        bundles = write_stage_bundles(
            Path(temporary) / "bundles",
            StageEvidence(
                "P",
                contract_sha256,
                state["prior_manifest_sha256"],
                review,
                resume,
            ),
            bundle_prefix="tabm_row_feature_stage",
        )
        if bundles.resume is None:
            raise RowFeatureColabError("stable snapshot did not produce resume evidence")
        verified = verify_resume_bundle(bundles.resume)
        digest = file_sha256(bundles.resume)
        destination = snapshot_dir / (
            f"tabm_row_feature_stage_P_emergency_resume_{sequence:04d}_{digest[:12]}.zip"
        )
        if destination.exists() or destination.is_symlink():
            raise RowFeatureColabError("emergency snapshot destination already exists")
        os.replace(bundles.resume, destination)
        if verify_resume_bundle(destination).manifest_sha256 != verified.manifest_sha256:
            raise RowFeatureColabError("stable snapshot changed during publication")
    return EmergencySnapshot(
        destination,
        digest,
        verified.manifest_sha256,
        candidate_id,
        max(-1, int(row["completed_epochs"]) - 1),
    )


class _VerifiedSnapshotStore:
    def __init__(
        self,
        *,
        snapshot_dir: Path,
        on_verified_snapshot: Callable[[EmergencySnapshot], None] | None,
    ) -> None:
        self.snapshot_dir = snapshot_dir
        self.on_verified_snapshot = on_verified_snapshot
        self.latest: EmergencySnapshot | None = None
        self._sequence = 0
        self._lock = threading.Lock()

    def next_sequence(self) -> int:
        with self._lock:
            value = self._sequence
            self._sequence += 1
            return value

    def accept(self, snapshot: EmergencySnapshot) -> EmergencySnapshot:
        verify_resume_bundle(snapshot.path)
        with self._lock:
            self.latest = snapshot
        if self.on_verified_snapshot is not None:
            self.on_verified_snapshot(snapshot)
        return snapshot

    def republish_latest(self) -> EmergencySnapshot | None:
        with self._lock:
            current = self.latest
        if current is None:
            return None
        verify_resume_bundle(current.path)
        sequence = self.next_sequence()
        destination = self.snapshot_dir / (
            f"tabm_row_feature_stage_P_emergency_resume_{sequence:04d}_{current.sha256[:12]}.zip"
        )
        if destination.exists() or destination.is_symlink():
            raise RowFeatureColabError("republished snapshot destination already exists")
        shutil.copyfile(current.path, destination)
        verified = verify_resume_bundle(destination)
        if file_sha256(destination) != current.sha256:
            destination.unlink()
            raise RowFeatureColabError("republished snapshot SHA-256 differs")
        return self.accept(
            EmergencySnapshot(
                destination,
                current.sha256,
                verified.manifest_sha256,
                current.candidate_id,
                current.epoch,
            )
        )


class SnapshottingCampaignRuntime:
    """Delegate GPU work while promoting each atomic epoch at a bounded cadence."""

    def __init__(
        self,
        data_dir: Path,
        *,
        live_output_dir: Path,
        store: _VerifiedSnapshotStore,
        contract,
        contract_path: Path,
        contract_sha256: str,
        train_sha256: str,
        history_sha256: str,
        input_manifest_sha256: str,
        code_sha256: str,
        snapshot_interval_seconds: float,
        poll_seconds: float = 1.0,
        delegate=None,
    ) -> None:
        from .worker import SubprocessCampaignRuntime

        if snapshot_interval_seconds <= 0 or poll_seconds <= 0:
            raise RowFeatureColabError("snapshot cadence must be positive")
        self.delegate = delegate or SubprocessCampaignRuntime(data_dir)
        self.live_output_dir = live_output_dir
        self.store = store
        self.contract = contract
        self.contract_path = contract_path
        self.contract_sha256 = contract_sha256
        self.train_sha256 = train_sha256
        self.history_sha256 = history_sha256
        self.input_manifest_sha256 = input_manifest_sha256
        self.code_sha256 = code_sha256
        self.snapshot_interval_seconds = float(snapshot_interval_seconds)
        self.poll_seconds = float(poll_seconds)

    def _active_snapshot(self, job) -> EmergencySnapshot:
        return publish_active_checkpoint_snapshot(
            live_output_dir=self.live_output_dir,
            snapshot_dir=self.store.snapshot_dir,
            active_job=job,
            contract=self.contract,
            contract_path=self.contract_path,
            contract_sha256=self.contract_sha256,
            train_sha256=self.train_sha256,
            history_sha256=self.history_sha256,
            input_manifest_sha256=self.input_manifest_sha256,
            code_sha256=self.code_sha256,
            sequence=self.store.next_sequence(),
        )

    def run_jobs(self, version, jobs, output_dir, *, gpu_count, job_deadline):
        if len(jobs) != 1:
            raise RowFeatureColabError("snapshot runtime requires one sequential Stage P job")
        job = jobs[0]
        stopped = threading.Event()
        if self.store.latest is None:
            initial = publish_stable_state_snapshot(
                live_output_dir=self.live_output_dir,
                snapshot_dir=self.store.snapshot_dir,
                contract=self.contract,
                contract_path=self.contract_path,
                contract_sha256=self.contract_sha256,
                train_sha256=self.train_sha256,
                history_sha256=self.history_sha256,
                input_manifest_sha256=self.input_manifest_sha256,
                code_sha256=self.code_sha256,
                sequence=self.store.next_sequence(),
                candidate_id=job.candidate_id,
                allow_not_started=True,
            )
            self.store.accept(initial)

        def monitor() -> None:
            last_epoch = -1
            next_periodic = time.monotonic() + self.snapshot_interval_seconds
            while True:
                wait_seconds = min(
                    self.poll_seconds,
                    max(0.0, next_periodic - time.monotonic()),
                )
                if stopped.wait(wait_seconds):
                    break
                meta_path = output_dir / job.candidate_id / "checkpoint_meta.json"
                observed_epoch = -1
                try:
                    raw = _read_json_object(meta_path.read_bytes(), "active checkpoint metadata")
                    if type(raw.get("epoch")) is int:
                        observed_epoch = int(raw["epoch"])
                except (OSError, RowFeatureColabError):
                    pass
                now = time.monotonic()
                if observed_epoch > last_epoch:
                    try:
                        snapshot = self._active_snapshot(job)
                    except Exception as error:
                        print(
                            f"ROW_FEATURE_SNAPSHOT_DEFERRED type={type(error).__name__} message={str(error).replace(' ', '_')}",
                            flush=True,
                        )
                    else:
                        self.store.accept(snapshot)
                        last_epoch = snapshot.epoch
                        next_periodic = now + self.snapshot_interval_seconds
                elif now >= next_periodic:
                    try:
                        snapshot = self._active_snapshot(job)
                    except Exception as active_error:
                        try:
                            snapshot = self.store.republish_latest()
                        except Exception as fallback_error:
                            print(
                                "ROW_FEATURE_SNAPSHOT_DEFERRED "
                                f"type={type(fallback_error).__name__} "
                                f"message={str(fallback_error).replace(' ', '_')} "
                                f"active_type={type(active_error).__name__}",
                                flush=True,
                            )
                            snapshot = None
                    else:
                        self.store.accept(snapshot)
                    next_periodic = now + self.snapshot_interval_seconds

        thread = threading.Thread(target=monitor, name="row-feature-snapshot", daemon=True)
        thread.start()
        try:
            return self.delegate.run_jobs(
                version,
                jobs,
                output_dir,
                gpu_count=gpu_count,
                job_deadline=job_deadline,
            )
        finally:
            stopped.set()
            thread.join(timeout=max(5.0, self.poll_seconds * 2.0))
            if thread.is_alive():
                raise RowFeatureColabError("snapshot monitor did not stop")


class _Tee:
    def __init__(self, console, log) -> None:
        self.console = console
        self.log = log
        self.lock = threading.Lock()

    def write(self, value: str) -> int:
        with self.lock:
            self.console.write(value)
            self.log.write(value)
        return len(value)

    def flush(self) -> None:
        with self.lock:
            self.console.flush()
            self.log.flush()


def run_supervised_stage(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    resume_bundle: str | Path | None,
    snapshot_dir: str | Path,
    wall_deadline: float,
    snapshot_interval_seconds: float,
    on_verified_snapshot: Callable[[EmergencySnapshot], None] | None,
    log_path: str | Path,
):
    """Run Stage P with streamed stdout and verified epoch/candidate recovery."""

    import sys

    from . import row_feature_proxy as proxy
    from .row_feature_contracts import (
        DEFAULT_ROW_FEATURE_PROXY_CONTRACT,
        load_row_feature_proxy_contract,
        row_feature_contract_sha256,
    )

    contract = load_row_feature_proxy_contract()
    contract_sha = row_feature_contract_sha256()
    code_sha = proxy._code_sha256()
    data_dir = Path(data_dir)
    history_sha = file_sha256(_regular_file(data_dir / "trackman_history.csv", "history"))
    input_manifest_sha = file_sha256(
        _regular_file(data_dir / "input_manifest.json", "input manifest")
    )
    output_dir = Path(output_dir)
    store = _VerifiedSnapshotStore(
        snapshot_dir=Path(snapshot_dir),
        on_verified_snapshot=on_verified_snapshot,
    )
    runtime = SnapshottingCampaignRuntime(
        Path(data_dir),
        live_output_dir=output_dir,
        store=store,
        contract=contract,
        contract_path=DEFAULT_ROW_FEATURE_PROXY_CONTRACT,
        contract_sha256=contract_sha,
        train_sha256=contract.official_train_sha256,
        history_sha256=history_sha,
        input_manifest_sha256=input_manifest_sha,
        code_sha256=code_sha,
        snapshot_interval_seconds=snapshot_interval_seconds,
    )

    def completed(job, _state_path: Path) -> None:
        snapshot = publish_stable_state_snapshot(
            live_output_dir=output_dir,
            snapshot_dir=store.snapshot_dir,
            contract=contract,
            contract_path=DEFAULT_ROW_FEATURE_PROXY_CONTRACT,
            contract_sha256=contract_sha,
            train_sha256=contract.official_train_sha256,
            history_sha256=history_sha,
            input_manifest_sha256=input_manifest_sha,
            code_sha256=code_sha,
            sequence=store.next_sequence(),
            candidate_id=job.candidate_id,
        )
        store.accept(snapshot)

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        with redirect_stdout(_Tee(sys.stdout, log)):
            result = proxy.run_row_feature_proxy(
                data_dir=data_dir,
                output_dir=output_dir,
                resume_bundle=resume_bundle,
                runtime=runtime,
                gpu_count=1,
                wall_deadline=wall_deadline,
                on_candidate_complete=completed,
            )
    return result, store.latest
