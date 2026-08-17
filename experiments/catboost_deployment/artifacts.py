from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Callable, Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd

from .metrics import CATBOOST_PREDICTION_COLUMNS
from .state import DeploymentStateError, deserialize_feature_state


class DeploymentArtifactError(ValueError):
    """Raised when deployment evidence is incomplete or corrupted."""


@dataclass(frozen=True)
class DeploymentBundlePaths:
    review: Path | None
    resume: Path
    delivery: Path | None
    review_sha256: str | None
    resume_sha256: str
    delivery_sha256: str | None


@dataclass(frozen=True)
class VerifiedDeploymentResume:
    path: Path
    bindings: Mapping[str, str]
    status: str
    completed_job_ids: tuple[str, ...]
    active_job_id: str | None
    selected_tree_count: int | None
    decision_sha256: str | None


_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
_SHA_CHARS = set("0123456789abcdef")
_BINDING_KEYS = {
    "contract_sha256",
    "code_sha256",
    "input_manifest_sha256",
    "train_sha256",
    "history_sha256",
    "stage_c_delivery_sha256",
    "stage_c_review_sha256",
    "stage_c_state_sha256",
    "source_blend_delivery_sha256",
    "source_blend_manifest_sha256",
    "source_decision_sha256",
}
_STATE_KEYS = {
    "schema_version",
    "campaign_id",
    "status",
    "completed_job_ids",
    "active_job_id",
    "decision_sha256",
    "selected_tree_count",
    "bindings",
}
_ALIGNMENT_REQUIRED = {
    "job.json",
    "metrics.json",
    "model.cbm",
    "predictions.csv",
    "preprocessing_state.json",
    "worker.log",
    "worker_result.json",
}
_FULL_REQUIRED = _ALIGNMENT_REQUIRED - {"predictions.csv"}
_ACTIVE_REQUIRED = {"job.json", "worker.log", "experiment.cbsnapshot"}
_MAX_MANIFEST = 1024 * 1024
_MAX_MEMBER = 8 * 1024**3
_MAX_TOTAL = 12 * 1024**3
_MAX_RATIO = 200.0


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _valid_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= _SHA_CHARS


def _file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except Exception as error:
        raise DeploymentArtifactError(f"{label} is unreadable") from error
    if type(value) is not dict:
        raise DeploymentArtifactError(f"{label} must be an object")
    return value


def _json_file(path: Path, label: str) -> dict[str, object]:
    try:
        return _json_bytes(path.read_bytes(), label)
    except OSError as error:
        raise DeploymentArtifactError(f"{label} is missing") from error


def _validate_bindings(value: Mapping[str, str]) -> dict[str, str]:
    bindings = dict(value)
    if set(bindings) != _BINDING_KEYS or any(
        not _valid_sha(item) for item in bindings.values()
    ):
        raise DeploymentArtifactError("artifact bindings differ")
    return bindings


def _validate_state(path: Path, bindings: Mapping[str, str]) -> dict[str, object]:
    state = _json_file(path, "stage state")
    if (
        set(state) != _STATE_KEYS
        or state["schema_version"] != 1
        or state["campaign_id"] != "catboost_deployment_v1"
        or state["bindings"] != dict(bindings)
        or type(state["completed_job_ids"]) is not list
        or any(type(item) is not str for item in state["completed_job_ids"])
        or len(state["completed_job_ids"]) != len(set(state["completed_job_ids"]))
        or (state["active_job_id"] is not None and type(state["active_job_id"]) is not str)
        or (state["decision_sha256"] is not None and not _valid_sha(state["decision_sha256"]))
        or (
            state["selected_tree_count"] is not None
            and type(state["selected_tree_count"]) is not int
        )
    ):
        raise DeploymentArtifactError("stage state differs")
    completed = state["completed_job_ids"]
    status = state["status"]
    active = state["active_job_id"]
    expected_alignment = ["align_2022_2023", "align_2023_2024"]
    valid = (
        (status == "fresh" and completed == [] and active is None and state["decision_sha256"] is None)
        or (
            status == "alignment_active"
            and completed in ([], expected_alignment[:1])
            and active in set(expected_alignment) - set(completed)
            and state["decision_sha256"] is None
        )
        or (
            status == "alignment_incomplete"
            and completed in ([], expected_alignment[:1])
            and active is None
            and state["decision_sha256"] is None
        )
        or (
            status in {"deployment_aligned", "deployment_blocked"}
            and completed == expected_alignment
            and active is None
            and state["decision_sha256"] is not None
        )
        or (
            status == "full_fit_active"
            and completed == expected_alignment
            and active == "full_2024"
            and state["decision_sha256"] is not None
        )
        or (
            status == "full_training_complete"
            and completed == [*expected_alignment, "full_2024"]
            and active is None
            and state["decision_sha256"] is not None
        )
    )
    if not valid:
        raise DeploymentArtifactError("stage transition differs")
    if status == "deployment_blocked" and state["selected_tree_count"] is not None:
        raise DeploymentArtifactError("blocked stage selected a tree count")
    if status in {"deployment_aligned", "full_fit_active", "full_training_complete"} and type(
        state["selected_tree_count"]
    ) is not int:
        raise DeploymentArtifactError("aligned stage is missing tree count")
    return state


def _validate_preprocessing(path: Path) -> str:
    try:
        deserialize_feature_state(path.read_bytes())
    except (OSError, DeploymentStateError) as error:
        raise DeploymentArtifactError("preprocessing state is invalid") from error
    return _file_sha(path)


def _validate_completed_job(directory: Path, job_id: str) -> dict[str, object]:
    if directory.is_symlink() or not directory.is_dir():
        raise DeploymentArtifactError(f"job directory is unsafe: {job_id}")
    job = _json_file(directory / "job.json", "job identity")
    result = _json_file(directory / "worker_result.json", "worker result")
    metrics = _json_file(directory / "metrics.json", "job metrics")
    kind = "full_fit" if job_id == "full_2024" else "alignment"
    required = _FULL_REQUIRED if kind == "full_fit" else _ALIGNMENT_REQUIRED
    if not all((directory / name).is_file() for name in required):
        raise DeploymentArtifactError(f"completed job artifacts are missing: {job_id}")
    if (
        job.get("job_id") != job_id
        or job.get("kind") != kind
        or result.get("job_id") != job_id
        or result.get("kind") != kind
        or result.get("status") != "completed"
        or metrics.get("job_id") != job_id
        or metrics.get("kind") != kind
    ):
        raise DeploymentArtifactError(f"completed job identity differs: {job_id}")
    model_sha = _file_sha(directory / "model.cbm")
    if metrics.get("model_sha256") != model_sha:
        raise DeploymentArtifactError(f"model SHA differs: {job_id}")
    preprocessing_sha = _validate_preprocessing(directory / "preprocessing_state.json")
    if metrics.get("preprocessing_sha256") != preprocessing_sha:
        raise DeploymentArtifactError(f"preprocessing SHA differs: {job_id}")
    if kind == "alignment":
        frame = pd.read_csv(directory / "predictions.csv")
        if tuple(frame.columns) != CATBOOST_PREDICTION_COLUMNS or frame.empty:
            raise DeploymentArtifactError(f"prediction schema differs: {job_id}")
        target = pd.to_numeric(frame["target"], errors="coerce").to_numpy(dtype="float64")
        declared = metrics.get("prefix_brier")
        if type(declared) is not dict or set(declared) != {
            "4", "32", "64", "128", "192", "296", "400"
        }:
            raise DeploymentArtifactError(f"prefix metric set differs: {job_id}")
        for prefix in (4, 32, 64, 128, 192, 296, 400):
            probability = pd.to_numeric(
                frame[f"p_{prefix}"], errors="coerce"
            ).to_numpy(dtype="float64")
            brier = float(np.mean(np.square(probability - target), dtype=np.float64))
            if (
                not math.isfinite(brier)
                or not math.isclose(float(declared[str(prefix)]), brier, abs_tol=1e-12)
            ):
                raise DeploymentArtifactError(f"prefix Brier differs: {job_id}")
        if metrics.get("predictions_sha256") != _file_sha(directory / "predictions.csv"):
            raise DeploymentArtifactError(f"prediction SHA differs: {job_id}")
    else:
        if result.get("predictions") is not None or metrics.get("valid_rows") is not None:
            raise DeploymentArtifactError("full fit unexpectedly has validation evidence")
        if (
            type(metrics.get("train_rows")) is not int
            or metrics["train_rows"] <= 0
            or result.get("train_rows") != metrics["train_rows"]
        ):
            raise DeploymentArtifactError("full fit row count differs")
    return result


def _validate_decision(path: Path, state: Mapping[str, object]) -> dict[str, object]:
    decision = _json_file(path, "alignment decision")
    if _file_sha(path) != state["decision_sha256"]:
        raise DeploymentArtifactError("alignment decision SHA differs")
    if decision.get("status") != state["status"] and not (
        state["status"] in {"full_fit_active", "full_training_complete"}
        and decision.get("status") == "deployment_aligned"
    ):
        raise DeploymentArtifactError("alignment decision status differs")
    if decision.get("selected_tree_count") != state["selected_tree_count"]:
        raise DeploymentArtifactError("selected tree count differs")
    return decision


def _zip_info(name: str, size: int) -> ZipInfo:
    info = ZipInfo(name, date_time=_ZIP_TIME)
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.file_size = size
    return info


def _write_archive(
    destination: Path,
    *,
    kind: str,
    bindings: Mapping[str, str],
    sources: Mapping[str, bytes | Path],
    check_deadline: Callable[[], None] | None,
    verifier: Callable[[Path], None],
) -> None:
    evidence: dict[str, dict[str, object]] = {}
    for name, source in sorted(sources.items()):
        if type(source) is bytes:
            size, digest = len(source), sha256(source).hexdigest()
        else:
            if source.is_symlink() or not source.is_file():
                raise DeploymentArtifactError(f"bundle source is unsafe: {name}")
            size, digest = source.stat().st_size, _file_sha(source)
        evidence[name] = {"size": size, "sha256": digest}
    manifest = _canonical_json(
        {
            "schema_version": 1,
            "artifact_kind": kind,
            "campaign_id": "catboost_deployment_v1",
            "review_only": True,
            "submission_package": False,
            "bindings": dict(bindings),
            "members": evidence,
        }
    )
    all_sources = {**sources, "manifest.json": manifest}
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}-", dir=destination.parent)
    os.close(descriptor)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for name, source in sorted(all_sources.items()):
                if check_deadline is not None:
                    check_deadline()
                size = len(source) if type(source) is bytes else source.stat().st_size
                with archive.open(_zip_info(name, size), "w") as output:
                    if type(source) is bytes:
                        output.write(source)
                    else:
                        with source.open("rb") as input_stream:
                            while chunk := input_stream.read(1024 * 1024):
                                if check_deadline is not None:
                                    check_deadline()
                                output.write(chunk)
        verifier(Path(temporary))
        if check_deadline is not None:
            check_deadline()
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _safe_archive(path: Path) -> tuple[dict[str, ZipInfo], dict[str, object], bytes]:
    try:
        with ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or "manifest.json" not in names:
                raise DeploymentArtifactError("bundle member set is invalid")
            if sum(info.file_size for info in infos) > _MAX_TOTAL:
                raise DeploymentArtifactError("bundle total size exceeds limit")
            for info in infos:
                pure = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if (
                    not info.filename
                    or "\\" in info.filename
                    or pure.is_absolute()
                    or ".." in pure.parts
                    or info.is_dir()
                    or stat.S_IFMT(mode) == stat.S_IFLNK
                    or info.file_size > _MAX_MEMBER
                    or (
                        info.file_size >= 64 * 1024
                        and info.file_size / max(1, info.compress_size) > _MAX_RATIO
                    )
                ):
                    raise DeploymentArtifactError(f"unsafe bundle member: {info.filename}")
            manifest_info = archive.getinfo("manifest.json")
            if manifest_info.file_size > _MAX_MANIFEST:
                raise DeploymentArtifactError("bundle manifest exceeds limit")
            manifest_bytes = archive.read(manifest_info)
            manifest = _json_bytes(manifest_bytes, "bundle manifest")
            return {info.filename: info for info in infos}, manifest, manifest_bytes
    except DeploymentArtifactError:
        raise
    except (BadZipFile, OSError, KeyError) as error:
        raise DeploymentArtifactError("bundle is unreadable") from error


def _verify_inner(
    path: Path, *, kind: str, expected_bindings: Mapping[str, str]
) -> tuple[dict[str, ZipInfo], dict[str, object]]:
    bindings = _validate_bindings(expected_bindings)
    infos, manifest, _ = _safe_archive(path)
    if (
        set(manifest)
        != {
            "schema_version", "artifact_kind", "campaign_id", "review_only",
            "submission_package", "bindings", "members",
        }
        or manifest["schema_version"] != 1
        or manifest["artifact_kind"] != kind
        or manifest["campaign_id"] != "catboost_deployment_v1"
        or manifest["review_only"] is not True
        or manifest["submission_package"] is not False
        or manifest["bindings"] != bindings
        or type(manifest["members"]) is not dict
        or set(manifest["members"]) != set(infos) - {"manifest.json"}
    ):
        raise DeploymentArtifactError("bundle manifest differs")
    with ZipFile(path) as archive:
        for name, evidence in manifest["members"].items():
            if type(evidence) is not dict or set(evidence) != {"size", "sha256"}:
                raise DeploymentArtifactError(f"bundle evidence differs: {name}")
            digest = sha256()
            size = 0
            with archive.open(name) as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
            if evidence != {"size": size, "sha256": digest.hexdigest()}:
                raise DeploymentArtifactError(f"bundle member differs: {name}")
        state = _json_bytes(archive.read("state/stage_state.json"), "bundled state")
    if state.get("bindings") != bindings:
        raise DeploymentArtifactError("bundled state binding differs")
    return infos, state


def verify_deployment_resume(
    path: Path, *, expected_bindings: Mapping[str, str]
) -> VerifiedDeploymentResume:
    infos, state = _verify_inner(
        Path(path), kind="catboost_deployment_resume", expected_bindings=expected_bindings
    )
    completed = tuple(state.get("completed_job_ids", ()))
    active = state.get("active_job_id")
    expected = {"manifest.json", "contract/contract.json", "state/stage_state.json"}
    if state.get("decision_sha256") is not None:
        expected.add("decision/alignment_decision.json")
    for job_id in completed:
        required = _FULL_REQUIRED if job_id == "full_2024" else _ALIGNMENT_REQUIRED
        expected.update(f"jobs/{job_id}/{name}" for name in required)
        snapshot = f"jobs/{job_id}/experiment.cbsnapshot"
        if snapshot in infos:
            expected.add(snapshot)
    if active is not None:
        expected.update(f"jobs/{active}/{name}" for name in _ACTIVE_REQUIRED)
    if set(infos) != expected:
        raise DeploymentArtifactError("resume member set differs from state")
    temporary = Path(tempfile.mkdtemp(prefix="catboost-deployment-resume-"))
    try:
        with ZipFile(path) as archive:
            for name in sorted(set(infos) - {"manifest.json"}):
                target = temporary.joinpath(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        validated_state = _validate_state(
            temporary / "state/stage_state.json", expected_bindings
        )
        if _file_sha(temporary / "contract/contract.json") != expected_bindings["contract_sha256"]:
            raise DeploymentArtifactError("bundled contract SHA differs")
        if validated_state["decision_sha256"] is not None:
            _validate_decision(
                temporary / "decision/alignment_decision.json", validated_state
            )
        for job_id in completed:
            _validate_completed_job(temporary / "jobs" / job_id, job_id)
        if active is not None:
            active_dir = temporary / "jobs" / str(active)
            job = _json_file(active_dir / "job.json", "active job identity")
            if job.get("job_id") != active or not all(
                (active_dir / name).is_file() for name in _ACTIVE_REQUIRED
            ):
                raise DeploymentArtifactError("active job snapshot differs")
        return VerifiedDeploymentResume(
            path=Path(path),
            bindings=dict(expected_bindings),
            status=str(validated_state["status"]),
            completed_job_ids=completed,
            active_job_id=None if active is None else str(active),
            selected_tree_count=validated_state["selected_tree_count"],
            decision_sha256=validated_state["decision_sha256"],
        )
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def verify_deployment_review(
    path: Path, *, expected_bindings: Mapping[str, str]
) -> None:
    infos, state = _verify_inner(
        Path(path), kind="catboost_deployment_review", expected_bindings=expected_bindings
    )
    completed = tuple(state.get("completed_job_ids", ()))
    expected = {
        "manifest.json",
        "contract/contract.json",
        "decision/alignment_decision.json",
        "logs/campaign.log",
        "metrics/job_results.json",
        "state/stage_state.json",
        "predictions/align_2022_2023.csv",
        "predictions/align_2023_2024.csv",
    }
    if state.get("status") == "full_training_complete":
        expected.update(
            {
                "model/model.cbm",
                "model/preprocessing_state.json",
                "model/inference_manifest.json",
            }
        )
    if set(completed[:2]) != {"align_2022_2023", "align_2023_2024"} or set(infos) != expected:
        raise DeploymentArtifactError("review member set differs")
    temporary = Path(tempfile.mkdtemp(prefix="catboost-deployment-review-"))
    try:
        with ZipFile(path) as archive:
            for name in sorted(set(infos) - {"manifest.json"}):
                target = temporary.joinpath(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        validated_state = _validate_state(
            temporary / "state/stage_state.json", expected_bindings
        )
        if _file_sha(temporary / "contract/contract.json") != expected_bindings["contract_sha256"]:
            raise DeploymentArtifactError("bundled contract SHA differs")
        _validate_decision(
            temporary / "decision/alignment_decision.json", validated_state
        )
        results = json.loads(
            (temporary / "metrics/job_results.json").read_text(encoding="utf-8")
        )
        if type(results) is not list or len(results) != len(completed):
            raise DeploymentArtifactError("review job result set differs")
        for job_id in ("align_2022_2023", "align_2023_2024"):
            frame = pd.read_csv(temporary / "predictions" / f"{job_id}.csv")
            if tuple(frame.columns) != CATBOOST_PREDICTION_COLUMNS or frame.empty:
                raise DeploymentArtifactError(f"review prediction schema differs: {job_id}")
        if validated_state["status"] == "full_training_complete":
            inference = _json_file(
                temporary / "model/inference_manifest.json", "inference manifest"
            )
            if (
                set(inference)
                != {
                    "schema_version", "artifact_kind", "tree_count",
                    "alignment_decision_sha256", "model_sha256",
                    "preprocessing_sha256", "row_independent",
                }
                or inference["schema_version"] != 1
                or inference["artifact_kind"] != "frozen_catboost_model"
                or inference["tree_count"] != validated_state["selected_tree_count"]
                or inference["alignment_decision_sha256"] != validated_state["decision_sha256"]
                or inference["model_sha256"] != _file_sha(temporary / "model/model.cbm")
            ):
                raise DeploymentArtifactError("frozen model SHA differs")
            preprocessing_sha = _validate_preprocessing(
                temporary / "model/preprocessing_state.json"
            )
            if inference["preprocessing_sha256"] != preprocessing_sha:
                raise DeploymentArtifactError("frozen preprocessing SHA differs")
            if inference["row_independent"] is not True:
                raise DeploymentArtifactError("frozen inference policy differs")
    except DeploymentArtifactError:
        raise
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise DeploymentArtifactError("review evidence is unreadable") from error
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _write_delivery(
    destination: Path,
    *,
    log: Path,
    review: Path,
    resume: Path,
    bindings: Mapping[str, str],
    check_deadline: Callable[[], None] | None,
) -> None:
    sources = {
        "catboost_deployment.log": log,
        "catboost_deployment_review.zip": review,
        "catboost_deployment_resume.zip": resume,
    }
    evidence = {
        name: {"size": path.stat().st_size, "sha256": _file_sha(path)}
        for name, path in sources.items()
    }
    manifest = _canonical_json(
        {
            "schema_version": 1,
            "artifact_kind": "catboost_full_training_delivery",
            "review_only": True,
            "submission_package": False,
            "bindings": dict(bindings),
            "members": evidence,
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}-", dir=destination.parent)
    os.close(descriptor)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for name, path in sorted(sources.items()):
                with path.open("rb") as source, archive.open(_zip_info(name, path.stat().st_size), "w") as output:
                    while chunk := source.read(1024 * 1024):
                        if check_deadline is not None:
                            check_deadline()
                        output.write(chunk)
            archive.writestr(_zip_info("delivery_manifest.json", len(manifest)), manifest)
        verify_full_training_delivery(Path(temporary), expected_bindings=bindings)
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def verify_full_training_delivery(
    path: Path, *, expected_bindings: Mapping[str, str]
) -> None:
    expected_names = {
        "catboost_deployment.log",
        "catboost_deployment_review.zip",
        "catboost_deployment_resume.zip",
        "delivery_manifest.json",
    }
    temporary = Path(tempfile.mkdtemp(prefix="catboost-deployment-delivery-"))
    try:
        with ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or set(names) != expected_names:
                raise DeploymentArtifactError("delivery member set differs")
            manifest = _json_bytes(archive.read("delivery_manifest.json"), "delivery manifest")
            if (
                manifest.get("artifact_kind") != "catboost_full_training_delivery"
                or manifest.get("review_only") is not True
                or manifest.get("submission_package") is not False
                or manifest.get("bindings") != dict(expected_bindings)
                or set(manifest.get("members", {})) != expected_names - {"delivery_manifest.json"}
            ):
                raise DeploymentArtifactError("delivery manifest differs")
            extracted: dict[str, Path] = {}
            for name, evidence in manifest["members"].items():
                target = temporary / name
                digest = sha256()
                size = 0
                with archive.open(name) as source, target.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                if evidence != {"size": size, "sha256": digest.hexdigest()}:
                    raise DeploymentArtifactError(f"delivery member differs: {name}")
                extracted[name] = target
        verify_deployment_review(
            extracted["catboost_deployment_review.zip"], expected_bindings=expected_bindings
        )
        verified = verify_deployment_resume(
            extracted["catboost_deployment_resume.zip"], expected_bindings=expected_bindings
        )
        if verified.status != "full_training_complete":
            raise DeploymentArtifactError("delivery resume is not complete")
    except DeploymentArtifactError:
        raise
    except (BadZipFile, OSError, KeyError) as error:
        raise DeploymentArtifactError("delivery is unreadable") from error
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def write_deployment_bundles(
    *,
    output_dir: Path,
    bindings: Mapping[str, str],
    contract_path: Path,
    stage_state_path: Path,
    campaign_log_path: Path,
    job_directories: Mapping[str, Path],
    decision_path: Path | None,
    check_deadline: Callable[[], None] | None = None,
) -> DeploymentBundlePaths:
    bindings = _validate_bindings(bindings)
    state = _validate_state(Path(stage_state_path), bindings)
    completed = list(state["completed_job_ids"])
    active = state["active_job_id"]
    expected_directories = set(completed) | ({active} if active is not None else set())
    if set(job_directories) != expected_directories:
        raise DeploymentArtifactError("job directory set differs")
    results = [
        _validate_completed_job(Path(job_directories[job_id]), job_id)
        for job_id in completed
    ]
    if active is not None:
        directory = Path(job_directories[active])
        if not all((directory / name).is_file() for name in _ACTIVE_REQUIRED):
            raise DeploymentArtifactError("active job snapshot is incomplete")
    decision: dict[str, object] | None = None
    if state["decision_sha256"] is not None:
        if decision_path is None:
            raise DeploymentArtifactError("alignment decision is missing")
        decision = _validate_decision(Path(decision_path), state)
    elif decision_path is not None:
        raise DeploymentArtifactError("unexpected alignment decision")

    resume_sources: dict[str, bytes | Path] = {
        "contract/contract.json": Path(contract_path),
        "state/stage_state.json": Path(stage_state_path),
    }
    if decision_path is not None:
        resume_sources["decision/alignment_decision.json"] = Path(decision_path)
    for job_id in completed:
        required = _FULL_REQUIRED if job_id == "full_2024" else _ALIGNMENT_REQUIRED
        for name in sorted(required):
            resume_sources[f"jobs/{job_id}/{name}"] = Path(job_directories[job_id]) / name
        snapshot = Path(job_directories[job_id]) / "experiment.cbsnapshot"
        if snapshot.is_file():
            resume_sources[f"jobs/{job_id}/experiment.cbsnapshot"] = snapshot
    if active is not None:
        for name in sorted(_ACTIVE_REQUIRED):
            resume_sources[f"jobs/{active}/{name}"] = Path(job_directories[active]) / name
    output_dir = Path(output_dir)
    resume = output_dir / "catboost_deployment_resume.zip"
    _write_archive(
        resume,
        kind="catboost_deployment_resume",
        bindings=bindings,
        sources=resume_sources,
        check_deadline=check_deadline,
        verifier=lambda path: verify_deployment_resume(path, expected_bindings=bindings),
    )

    review: Path | None = None
    if decision is not None:
        review_sources: dict[str, bytes | Path] = {
            "contract/contract.json": Path(contract_path),
            "decision/alignment_decision.json": Path(decision_path),
            "logs/campaign.log": Path(campaign_log_path),
            "metrics/job_results.json": _canonical_json(results),
            "state/stage_state.json": Path(stage_state_path),
            "predictions/align_2022_2023.csv": Path(job_directories["align_2022_2023"]) / "predictions.csv",
            "predictions/align_2023_2024.csv": Path(job_directories["align_2023_2024"]) / "predictions.csv",
        }
        if state["status"] == "full_training_complete":
            full = Path(job_directories["full_2024"])
            inference_manifest = _canonical_json(
                {
                    "schema_version": 1,
                    "artifact_kind": "frozen_catboost_model",
                    "tree_count": state["selected_tree_count"],
                    "alignment_decision_sha256": state["decision_sha256"],
                    "model_sha256": _file_sha(full / "model.cbm"),
                    "preprocessing_sha256": _file_sha(full / "preprocessing_state.json"),
                    "row_independent": True,
                }
            )
            review_sources.update(
                {
                    "model/model.cbm": full / "model.cbm",
                    "model/preprocessing_state.json": full / "preprocessing_state.json",
                    "model/inference_manifest.json": inference_manifest,
                }
            )
        review = output_dir / "catboost_deployment_review.zip"
        _write_archive(
            review,
            kind="catboost_deployment_review",
            bindings=bindings,
            sources=review_sources,
            check_deadline=check_deadline,
            verifier=lambda path: verify_deployment_review(path, expected_bindings=bindings),
        )

    delivery: Path | None = None
    if state["status"] == "full_training_complete":
        if review is None:
            raise DeploymentArtifactError("complete training review is missing")
        delivery = output_dir / "catboost_full_training_delivery.zip"
        _write_delivery(
            delivery,
            log=Path(campaign_log_path),
            review=review,
            resume=resume,
            bindings=bindings,
            check_deadline=check_deadline,
        )
    return DeploymentBundlePaths(
        review=review,
        resume=resume,
        delivery=delivery,
        review_sha256=None if review is None else _file_sha(review),
        resume_sha256=_file_sha(resume),
        delivery_sha256=None if delivery is None else _file_sha(delivery),
    )
