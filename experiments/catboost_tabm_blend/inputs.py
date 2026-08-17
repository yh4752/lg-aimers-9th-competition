from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Mapping
from types import MappingProxyType
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd

from experiments.tabm_campaign.artifacts import ArtifactError, verify_review_bundle

from .contracts import BlendContract, contract_sha256
from .metrics import PREDICTION_COLUMNS


class BlendInputError(ValueError):
    """Raised when an uploaded blend input cannot be trusted."""


@dataclass(frozen=True)
class VerifiedTrainingInput:
    data_dir: Path
    manifest_sha256: str
    train_sha256: str
    history_sha256: str


@dataclass(frozen=True)
class VerifiedStageC:
    delivery_sha256: str
    review_sha256: str
    resume_sha256: str
    stage_state_sha256: str
    prediction_paths: Mapping[str, Path]


_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
_DATA_MEMBERS = {"input_manifest.json", "train.csv", "trackman_history.csv"}
_DELIVERY_MEMBERS = {
    "delivery_manifest.json",
    "colab_stage_C.log",
    "tabm_search_stage_C_review_bundle.zip",
    "tabm_search_stage_C_resume_bundle.zip",
}
_MAX_MANIFEST = 1024 * 1024
_MAX_MEMBER = 8 * 1024 * 1024 * 1024
_MAX_TOTAL = 12 * 1024 * 1024 * 1024
_MAX_RATIO = 200.0


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(info: ZipInfo) -> None:
    path = PurePosixPath(info.filename)
    mode = info.external_attr >> 16
    if (
        not info.filename
        or "\\" in info.filename
        or path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 1
        or info.is_dir()
        or stat.S_IFMT(mode) == stat.S_IFLNK
        or info.file_size > _MAX_MEMBER
        or (
            info.file_size >= 64 * 1024
            and info.file_size / max(1, info.compress_size) > _MAX_RATIO
        )
    ):
        raise BlendInputError(f"unsafe ZIP member: {info.filename}")


def _checked_infos(archive: ZipFile, expected: set[str]) -> dict[str, ZipInfo]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)) or set(names) != expected:
        raise BlendInputError("ZIP member set differs")
    if sum(info.file_size for info in infos) > _MAX_TOTAL:
        raise BlendInputError("ZIP total size exceeds limit")
    for info in infos:
        _safe_member(info)
    return {info.filename: info for info in infos}


def _safe_sources(data_dir: Path) -> dict[str, Path]:
    root = Path(os.path.abspath(os.fspath(data_dir)))
    for component in (root, *root.parents):
        if component.is_symlink():
            raise BlendInputError("data path has a symlink ancestor")
    if not root.is_dir():
        raise BlendInputError("data directory is missing")
    matches: dict[str, list[Path]] = {"train.csv": [], "trackman_history.csv": []}
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        for name in [*names, *files]:
            if (base / name).is_symlink():
                raise BlendInputError("data directory contains a symlink")
        for name in matches:
            if name in files:
                matches[name].append(base / name)
    output: dict[str, Path] = {}
    for name, paths in matches.items():
        if len(paths) != 1 or paths[0].parent != root:
            raise BlendInputError(f"data directory must contain one top-level {name}")
        metadata = paths[0].stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise BlendInputError(f"{name} is not a regular file")
        output[name] = paths[0]
    return output


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=_ZIP_TIME)
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _stream_file(archive: ZipFile, name: str, path: Path) -> None:
    info = _zip_info(name)
    info.file_size = path.stat().st_size
    with path.open("rb") as source, archive.open(info, "w") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)


def prepare_input_archive(
    data_dir: Path,
    output: Path,
    contract: BlendContract,
    *,
    replace: bool = False,
) -> Path:
    sources = _safe_sources(Path(data_dir))
    evidence = {
        name: {"size": path.stat().st_size, "sha256": file_sha256(path)}
        for name, path in sorted(sources.items())
    }
    if evidence["train.csv"]["sha256"] != contract.official_train_sha256:
        raise BlendInputError("train.csv SHA-256 differs")
    if evidence["trackman_history.csv"]["sha256"] != contract.official_history_sha256:
        raise BlendInputError("trackman_history.csv SHA-256 differs")
    manifest = canonical_json(
        {
            "schema_version": 1,
            "artifact_kind": "catboost_tabm_blend_input",
            "campaign_config_sha256": contract_sha256(),
            "members": evidence,
        }
    )
    output = Path(output)
    if (output.exists() or output.is_symlink()) and not replace:
        raise BlendInputError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}-", dir=output.parent)
    os.close(descriptor)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            archive.writestr(_zip_info("input_manifest.json"), manifest)
            for name, path in sorted(sources.items()):
                _stream_file(archive, name, path)
        if any(file_sha256(path) != evidence[name]["sha256"] for name, path in sources.items()):
            raise BlendInputError("source changed during archive publication")
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return output


def _manifest_object(value: bytes, label: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except Exception as error:
        raise BlendInputError(f"{label} is unreadable") from error
    if type(parsed) is not dict:
        raise BlendInputError(f"{label} must be an object")
    return parsed


def verify_and_extract_training_input(
    source: Path, destination: Path, contract: BlendContract
) -> VerifiedTrainingInput:
    source = Path(source)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise BlendInputError("training destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        with ZipFile(source) as archive:
            infos = _checked_infos(archive, _DATA_MEMBERS)
            if infos["input_manifest.json"].file_size > _MAX_MANIFEST:
                raise BlendInputError("input manifest exceeds limit")
            manifest_bytes = archive.read("input_manifest.json")
            manifest = _manifest_object(manifest_bytes, "input manifest")
            if set(manifest) != {"schema_version", "artifact_kind", "campaign_config_sha256", "members"}:
                raise BlendInputError("input manifest keys differ")
            if (
                manifest["schema_version"] != 1
                or manifest["artifact_kind"] != "catboost_tabm_blend_input"
                or manifest["campaign_config_sha256"] != contract_sha256()
                or type(manifest["members"]) is not dict
                or set(manifest["members"]) != {"train.csv", "trackman_history.csv"}
            ):
                raise BlendInputError("input manifest identity differs")
            for name in ("train.csv", "trackman_history.csv"):
                declared = manifest["members"][name]
                if type(declared) is not dict or set(declared) != {"size", "sha256"}:
                    raise BlendInputError(f"input member evidence differs: {name}")
                target = temporary / name
                digest = sha256()
                size = 0
                with archive.open(name) as input_stream, target.open("xb") as output_stream:
                    while chunk := input_stream.read(1024 * 1024):
                        output_stream.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                expected_sha = (
                    contract.official_train_sha256
                    if name == "train.csv"
                    else contract.official_history_sha256
                )
                if (
                    type(declared["size"]) is not int
                    or size != declared["size"]
                    or digest.hexdigest() != declared["sha256"]
                    or digest.hexdigest() != expected_sha
                ):
                    raise BlendInputError(f"input member differs: {name}")
            (temporary / "input_manifest.json").write_bytes(manifest_bytes)
        os.replace(temporary, destination)
    except (BlendInputError, BadZipFile):
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    except Exception as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise BlendInputError(f"cannot verify training input: {error}") from error
    return VerifiedTrainingInput(
        data_dir=destination,
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
        train_sha256=contract.official_train_sha256,
        history_sha256=contract.official_history_sha256,
    )


def _validate_prediction(path: Path) -> None:
    frame = pd.read_csv(path)
    if tuple(frame.columns) != PREDICTION_COLUMNS or frame.empty:
        raise BlendInputError("Stage C prediction schema differs")
    if frame["row_id"].isna().any() or frame["row_id"].astype(str).duplicated().any():
        raise BlendInputError("Stage C row_id is invalid")
    target = pd.to_numeric(frame["target"], errors="coerce").to_numpy(dtype="float64")
    probability = pd.to_numeric(frame["probability"], errors="coerce").to_numpy(dtype="float64")
    if (
        not np.isfinite(target).all()
        or not np.isin(target, (0.0, 1.0)).all()
        or not np.isfinite(probability).all()
        or np.any(probability < 0.0)
        or np.any(probability > 1.0)
    ):
        raise BlendInputError("Stage C prediction values are invalid")


def verify_and_extract_stage_c(
    source: Path, destination: Path, contract: BlendContract
) -> VerifiedStageC:
    source = Path(source)
    destination = Path(destination)
    observed_delivery_sha = file_sha256(source)
    if observed_delivery_sha != contract.stage_c["delivery_sha256"]:
        raise BlendInputError("Stage C delivery SHA-256 differs")
    if destination.exists() or destination.is_symlink():
        raise BlendInputError("Stage C destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    nested = temporary / "nested"
    nested.mkdir()
    try:
        with ZipFile(source) as archive:
            infos = _checked_infos(archive, _DELIVERY_MEMBERS)
            if infos["delivery_manifest.json"].file_size > _MAX_MANIFEST:
                raise BlendInputError("Stage C delivery manifest exceeds limit")
            manifest = _manifest_object(
                archive.read("delivery_manifest.json"), "Stage C delivery manifest"
            )
            if set(manifest) != {
                "schema_version",
                "artifact_kind",
                "run_uuid",
                "runtime_identity",
                "final_stage_complete",
                "members",
            }:
                raise BlendInputError("Stage C delivery manifest keys differ")
            payload_names = _DELIVERY_MEMBERS - {"delivery_manifest.json"}
            if (
                manifest["schema_version"] != 1
                or manifest["artifact_kind"] != "tabm_colab_stage_C_delivery"
                or manifest["final_stage_complete"] is not True
                or type(manifest["members"]) is not dict
                or set(manifest["members"]) != payload_names
            ):
                raise BlendInputError("Stage C delivery identity differs")
            materialized: dict[str, Path] = {}
            for name in sorted(payload_names):
                declared = manifest["members"][name]
                if type(declared) is not dict or set(declared) != {"size", "sha256"}:
                    raise BlendInputError(f"Stage C member evidence differs: {name}")
                target = nested / name
                digest = sha256()
                size = 0
                with archive.open(name) as input_stream, target.open("xb") as output_stream:
                    while chunk := input_stream.read(1024 * 1024):
                        output_stream.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                if size != declared["size"] or digest.hexdigest() != declared["sha256"]:
                    raise BlendInputError(f"Stage C member differs: {name}")
                materialized[name] = target
        review_path = materialized["tabm_search_stage_C_review_bundle.zip"]
        resume_path = materialized["tabm_search_stage_C_resume_bundle.zip"]
        review_sha = file_sha256(review_path)
        resume_sha = file_sha256(resume_path)
        if review_sha != contract.stage_c["review_sha256"]:
            raise BlendInputError("Stage C review SHA-256 differs")
        if resume_sha != contract.stage_c["resume_sha256"]:
            raise BlendInputError("Stage C resume SHA-256 differs")
        try:
            verified_review = verify_review_bundle(review_path)
        except ArtifactError as error:
            raise BlendInputError(f"Stage C review is invalid: {error}") from error
        if verified_review.version != "C":
            raise BlendInputError("Stage C review version differs")

        prediction_members = dict(contract.stage_c["prediction_members"])
        output_root = temporary / "selected"
        output_root.mkdir()
        with ZipFile(review_path) as review:
            state_bytes = review.read("stage_state.json")
            if sha256(state_bytes).hexdigest() != contract.stage_c["stage_state_sha256"]:
                raise BlendInputError("Stage C state SHA-256 differs")
            state = _manifest_object(state_bytes, "Stage C state")
            final = state.get("final_members")
            if (
                state.get("version") != "C"
                or state.get("stage_complete") is not True
                or state.get("selected_predictor") != contract.stage_c["selected_predictor"]
                or type(final) is not list
                or len(final) != 1
                or type(final[0]) is not dict
                or final[0].get("status") != "completed"
                or final[0].get("seed") != 3407
                or final[0].get("temporal_best_epochs") != [3, 0]
            ):
                raise BlendInputError("Stage C selected predictor differs")
            prediction_paths: dict[str, Path] = {}
            for fold, member in sorted(prediction_members.items()):
                if member not in review.namelist():
                    raise BlendInputError(f"Stage C prediction is missing: {fold}")
                target = output_root / f"tabm_{fold.replace('->', '_to_')}.csv"
                with review.open(member) as source_stream, target.open("xb") as output_stream:
                    shutil.copyfileobj(source_stream, output_stream, length=1024 * 1024)
                _validate_prediction(target)
                prediction_paths[fold] = target
        shutil.rmtree(nested)
        os.replace(output_root, destination)
        shutil.rmtree(temporary, ignore_errors=True)
    except (BlendInputError, BadZipFile):
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    except Exception as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise BlendInputError(f"cannot verify Stage C delivery: {error}") from error
    return VerifiedStageC(
        delivery_sha256=observed_delivery_sha,
        review_sha256=review_sha,
        resume_sha256=resume_sha,
        stage_state_sha256=str(contract.stage_c["stage_state_sha256"]),
        prediction_paths=MappingProxyType(
            {fold: destination / path.name for fold, path in prediction_paths.items()}
        ),
    )
