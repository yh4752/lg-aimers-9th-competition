from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from types import MappingProxyType
from typing import Mapping, Sequence
from zipfile import BadZipFile, ZipFile, ZipInfo

import numpy as np
import pandas as pd

from experiments.catboost_tabm_blend.contracts import load_contract as load_blend_contract
from experiments.catboost_tabm_blend.inputs import (
    BlendInputError,
    VerifiedStageC,
    VerifiedTrainingInput,
    file_sha256,
    verify_and_extract_stage_c,
    verify_and_extract_training_input,
)

from .contracts import HierarchicalContract, contract_sha256


class HierarchicalInputError(ValueError):
    """Raised when uploaded campaign evidence cannot be trusted."""


@dataclass(frozen=True)
class VerifiedHierarchicalInputs:
    training: VerifiedTrainingInput
    stage_c: VerifiedStageC
    source_stage_c_path: Path
    source_stage_c_sha256: str
    anchor_predictions: Mapping[str, Path]


EXPECTED_BINDING_KEYS = {
    "contract_sha256",
    "code_sha256",
    "environment_sha256",
    "input_manifest_sha256",
    "train_sha256",
    "history_sha256",
    "stage_c_delivery_sha256",
    "stage_c_review_sha256",
    "stage_c_state_sha256",
    "anchor_2022_2023_sha256",
    "anchor_2023_2024_sha256",
}
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_TRAINING_MEMBERS = {"input_manifest.json", "train.csv", "trackman_history.csv"}
_STAGE_C_MEMBERS = {
    "delivery_manifest.json",
    "colab_stage_C.log",
    "tabm_search_stage_C_review_bundle.zip",
    "tabm_search_stage_C_resume_bundle.zip",
}
_ANCHOR_COLUMNS = (
    "row_id",
    "target",
    "probability",
    "game_type",
    "game_month",
    "pitcher_id_known",
    "batter_id_known",
)
_MAX_MEMBER = 8 * 1024 * 1024 * 1024
_MAX_TOTAL = 12 * 1024 * 1024 * 1024
_MAX_RATIO = 200.0
_MAX_MANIFEST = 1024 * 1024


def _safe_info(info: ZipInfo) -> None:
    path = PurePosixPath(info.filename)
    mode = info.external_attr >> 16
    lowered = info.filename.lower()
    if (
        not info.filename
        or "\\" in info.filename
        or path.is_absolute()
        or ".." in path.parts
        or info.is_dir()
        or stat.S_IFMT(mode) == stat.S_IFLNK
        or any("test" in part.lower() or "submit" in part.lower() for part in path.parts)
        or info.file_size > _MAX_MEMBER
        or (
            info.file_size >= 64 * 1024
            and info.file_size / max(1, info.compress_size) > _MAX_RATIO
        )
    ):
        raise HierarchicalInputError(f"unsafe ZIP member: {info.filename}")
    del lowered


def _infos(path: Path) -> tuple[ZipFile, dict[str, ZipInfo]]:
    try:
        archive = ZipFile(path)
        infos = archive.infolist()
    except (OSError, BadZipFile) as error:
        raise HierarchicalInputError(f"unknown upload: {error}") from error
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        archive.close()
        raise HierarchicalInputError("duplicate ZIP members")
    if not names or sum(info.file_size for info in infos) > _MAX_TOTAL:
        archive.close()
        raise HierarchicalInputError("unsafe ZIP total size")
    try:
        for info in infos:
            _safe_info(info)
    except Exception:
        archive.close()
        raise
    return archive, {info.filename: info for info in infos}


def _regular_source(path: Path) -> Path:
    source = Path(os.path.abspath(os.fspath(path)))
    if source.is_symlink() or not source.is_file():
        raise HierarchicalInputError("upload must be a regular file")
    metadata = source.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise HierarchicalInputError("upload must be a regular file")
    return source


def classify_upload(path: Path) -> str:
    source = _regular_source(path)
    archive, infos = _infos(source)
    try:
        names = set(infos)
        if names == _TRAINING_MEMBERS:
            return "training_input"
        if names == _STAGE_C_MEMBERS:
            return "stage_c_delivery"
        manifest_info = infos.get("manifest.json")
        if manifest_info is not None and manifest_info.file_size <= _MAX_MANIFEST:
            try:
                manifest = json.loads(archive.read(manifest_info))
            except Exception as error:
                raise HierarchicalInputError("resume manifest is unreadable") from error
            if (
                type(manifest) is dict
                and manifest.get("schema_version") == 1
                and manifest.get("artifact_kind") == "hierarchical_tabm_resume_v1"
                and "stage_state.json" in infos
            ):
                return "campaign_resume"
    finally:
        archive.close()
    raise HierarchicalInputError("unknown upload member set")


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise HierarchicalInputError(f"{label} must be a lowercase SHA-256")
    return value


def _exclusive_root(path: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(path)))
    if root.exists() or root.is_symlink():
        raise HierarchicalInputError("run root must not already exist")
    for parent in root.parents:
        if parent.is_symlink():
            raise HierarchicalInputError("run root has a symlink ancestor")
    root.mkdir(parents=True)
    return root


def _validate_anchor(path: Path) -> None:
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        raise HierarchicalInputError(f"anchor prediction is unreadable: {error}") from error
    if tuple(frame.columns) != _ANCHOR_COLUMNS or frame.empty:
        raise HierarchicalInputError("anchor prediction schema differs")
    if frame["row_id"].isna().any() or frame["row_id"].astype(str).duplicated().any():
        raise HierarchicalInputError("anchor row_id is invalid")
    target = pd.to_numeric(frame["target"], errors="coerce").to_numpy(dtype="float64")
    probability = pd.to_numeric(frame["probability"], errors="coerce").to_numpy(dtype="float64")
    if (
        not np.isfinite(target).all()
        or not np.isin(target, (0.0, 1.0)).all()
        or not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise HierarchicalInputError("anchor prediction values are invalid")


def classify_and_verify_uploads(
    paths: Sequence[Path],
    *,
    run_root: Path,
    contract: HierarchicalContract,
    expected_contract_sha256: str,
    expected_code_sha256: str,
) -> tuple[VerifiedHierarchicalInputs, Path | None]:
    if len(paths) not in (2, 3):
        raise HierarchicalInputError("upload count must be two or three")
    if expected_contract_sha256 != contract_sha256():
        raise HierarchicalInputError("contract identity differs")
    _sha(expected_code_sha256, "code identity")
    classified: dict[str, Path] = {}
    for raw_path in paths:
        source = _regular_source(Path(raw_path))
        kind = classify_upload(source)
        if kind in classified:
            raise HierarchicalInputError(f"duplicate upload kind: {kind}")
        classified[kind] = source
    required = {"training_input", "stage_c_delivery"}
    if not required.issubset(classified) or set(classified).difference(
        {*required, "campaign_resume"}
    ):
        raise HierarchicalInputError("required upload kinds differ")

    blend = load_blend_contract()
    if (
        blend.official_train_sha256 != contract.official_train_sha256
        or blend.official_history_sha256 != contract.official_history_sha256
        or blend.stage_c["delivery_sha256"] != contract.source_stage_c_delivery_sha256
    ):
        raise HierarchicalInputError("source verifier identity differs")
    stage_source = classified["stage_c_delivery"]
    stage_before = file_sha256(stage_source)
    if stage_before != contract.source_stage_c_delivery_sha256:
        raise HierarchicalInputError("Stage C delivery SHA-256 differs")
    training_source = classified["training_input"]
    training_before = file_sha256(training_source)
    root = _exclusive_root(run_root)
    try:
        training = verify_and_extract_training_input(
            training_source, root / "official_data", blend
        )
        stage_c = verify_and_extract_stage_c(
            stage_source, root / "stage_c_anchor", blend
        )
        if (
            file_sha256(training_source) != training_before
            or file_sha256(stage_source) != stage_before
        ):
            raise HierarchicalInputError("upload changed during verification")
        if (
            training.train_sha256 != contract.official_train_sha256
            or training.history_sha256 != contract.official_history_sha256
            or stage_c.delivery_sha256 != contract.source_stage_c_delivery_sha256
        ):
            raise HierarchicalInputError("verified source identity differs")
        if tuple(stage_c.prediction_paths) != ("2022->2023", "2023->2024"):
            raise HierarchicalInputError("anchor fold set differs")
        anchors: dict[str, Path] = {}
        for fold, anchor_path in stage_c.prediction_paths.items():
            _validate_anchor(anchor_path)
            anchors[fold] = Path(anchor_path)
        verified = VerifiedHierarchicalInputs(
            training,
            stage_c,
            stage_source.resolve(),
            stage_before,
            MappingProxyType(anchors),
        )
        return verified, classified.get("campaign_resume")
    except (BlendInputError, HierarchicalInputError):
        shutil.rmtree(root, ignore_errors=True)
        raise
    except Exception as error:
        shutil.rmtree(root, ignore_errors=True)
        raise HierarchicalInputError(f"cannot verify uploads: {error}") from error
