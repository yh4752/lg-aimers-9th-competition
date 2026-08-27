from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from types import MappingProxyType
from typing import Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd

from .rf_contracts import contract_sha256, load_rf_contract


class RFInputError(ValueError):
    pass


FOLDS = ((2021, 2022), (2022, 2023), (2023, 2024))
INPUT_KIND = "tree_expert_rf_input_v1"
_FOLD_MEMBERS = {fold: f"e2/fold_{fold[0]}_{fold[1]}.csv" for fold in FOLDS}
_MODEL_DELIVERY = "e2/model_delivery.zip"
_PAYLOAD_NAMES = {
    "e2/acceptance.json",
    "e2/handoff_manifest.json",
    _MODEL_DELIVERY,
    *_FOLD_MEMBERS.values(),
}
_MAX_INPUT_EXPANDED = 512 * 1024 * 1024
_MAX_HANDOFF_EXPANDED = 1024 * 1024 * 1024
_MAX_NESTED_EXPANDED = 512 * 1024 * 1024


@dataclass(frozen=True)
class VerifiedRFInput:
    root: Path
    manifest_sha256: str
    e2_handoff_sha256: str
    e2_candidate_id: str
    fold_predictions: Mapping[tuple[int, int], Path]
    acceptance: Path
    full_fit_root: Path


@dataclass(frozen=True)
class VerifiedOfficialData:
    root: Path
    train: Path
    history: Path
    train_sha256: str
    history_sha256: str


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RFInputError(f"{label} is invalid JSON") from error
    if type(value) is not dict:
        raise RFInputError(f"{label} must be an object")
    return value


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _safe_infos(archive: ZipFile, label: str, limit: int) -> dict[str, ZipInfo]:
    result: dict[str, ZipInfo] = {}
    total = 0
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        is_symlink = mode & 0o170000 == 0o120000
        if (
            info.filename in result
            or info.flag_bits & 0x1
            or info.is_dir()
            or is_symlink
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in info.filename
        ):
            raise RFInputError(f"unsafe or duplicate member in {label}")
        total += info.file_size
        if total > limit:
            raise RFInputError(f"{label} expanded size exceeds limit")
        result[info.filename] = info
    return result


def _verify_declared_members(
    archive: ZipFile,
    manifest: dict[str, object],
    infos: Mapping[str, ZipInfo],
    *,
    manifest_name: str = "manifest.json",
) -> None:
    names = set(infos) - {manifest_name}
    declared = manifest.get("members")
    if type(declared) is not dict or set(declared) != names:
        raise RFInputError("nested manifest members differ")
    for name in sorted(names):
        metadata = declared[name]
        payload = archive.read(infos[name])
        if (
            type(metadata) is not dict
            or metadata.get("size") != len(payload)
            or metadata.get("sha256") != sha256(payload).hexdigest()
        ):
            raise RFInputError(f"nested member SHA-256 differs: {name}")


def _prediction(payload: bytes, label: str) -> tuple[pd.DataFrame, str]:
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except Exception as error:
        raise RFInputError(f"{label} prediction CSV is invalid") from error
    required = {"row_id", "target", "probability", "game_type"}
    if not required.issubset(frame.columns) or frame.empty:
        raise RFInputError(f"{label} prediction schema differs")
    if frame["row_id"].isna().any() or not frame["row_id"].is_unique:
        raise RFInputError(f"{label} row identity differs")
    target = pd.to_numeric(frame["target"], errors="coerce")
    probability = pd.to_numeric(frame["probability"], errors="coerce")
    if (
        not target.isin([0, 1]).all()
        or not probability.between(0, 1).all()
        or not frame["game_type"].astype(str).isin(["R", "F"]).all()
    ):
        raise RFInputError(f"{label} prediction values differ")
    row_hash = sha256("\n".join(frame["row_id"].astype(str)).encode()).hexdigest()
    return frame, row_hash


def _validate_delivery(payload: bytes) -> None:
    try:
        with ZipFile(io.BytesIO(payload)) as archive:
            infos = _safe_infos(archive, "E2 delivery", _MAX_NESTED_EXPANDED)
            if "manifest.json" not in infos:
                raise RFInputError("E2 delivery manifest is absent")
            manifest = _json(archive.read(infos["manifest.json"]), "E2 delivery manifest")
            required = {
                "evidence/acceptance_decision.json",
                "evidence/full_fit_manifest.json",
                "evidence/inference_audit.json",
                "frozen_state/feature_state.json",
                *(f"models/catboost_seed_{seed}.cbm" for seed in (42, 2026, 3407)),
            }
            if (
                manifest.get("artifact_kind") != "tree_expert_e2_model_delivery_v1"
                or manifest.get("candidate_id") != "c1_anchor_residual"
                or manifest.get("predictor") != "catboost"
                or manifest.get("seeds") != [42, 2026, 3407]
                or manifest.get("submission_package") is not False
                or not required.issubset(infos)
            ):
                raise RFInputError("E2 delivery identity differs")
            _verify_declared_members(archive, manifest, infos)
    except BadZipFile as error:
        raise RFInputError("E2 delivery is not a valid ZIP") from error


def _read_e2_handoff(path: Path, expected_sha: str) -> dict[str, bytes]:
    source = Path(path)
    if not source.is_file() or file_sha256(source) != expected_sha:
        raise RFInputError("E2 handoff SHA-256 differs")
    try:
        with ZipFile(source) as outer:
            infos = _safe_infos(outer, "E2 handoff", _MAX_HANDOFF_EXPANDED)
            handoff_manifest = outer.read(infos["handoff_manifest.json"])
            handoff = _json(handoff_manifest, "E2 handoff manifest")
            expected_names = {
                "handoff_manifest.json", "tree_expert_e2_model_delivery.zip",
                "tree_expert_e2_resume.zip", "tree_expert_e2_review.zip", "tree_expert_e2.log",
            }
            if (
                set(infos) != expected_names
                or handoff.get("artifact_kind") != "tree_expert_e2_handoff_v1"
                or handoff.get("campaign_id") != "tree_expert_e2_v1"
                or handoff.get("status") != "accepted"
                or handoff.get("delivery") is not True
                or handoff.get("submission_package") is not False
            ):
                raise RFInputError("E2 handoff is not accepted")
            declared = handoff.get("members")
            if type(declared) is not dict or set(declared) != expected_names - {"handoff_manifest.json"}:
                raise RFInputError("E2 handoff members differ")
            for name in sorted(declared):
                payload = outer.read(infos[name])
                metadata = declared[name]
                if (
                    type(metadata) is not dict
                    or metadata.get("size") != len(payload)
                    or metadata.get("sha256") != sha256(payload).hexdigest()
                ):
                    raise RFInputError(f"E2 handoff member differs: {name}")
            review = outer.read(infos["tree_expert_e2_review.zip"])
            delivery = outer.read(infos["tree_expert_e2_model_delivery.zip"])
        _validate_delivery(delivery)
        with ZipFile(io.BytesIO(review)) as nested:
            nested_infos = _safe_infos(nested, "E2 review", _MAX_NESTED_EXPANDED)
            acceptance = nested.read(nested_infos["decisions/acceptance.json"])
            decision = _json(acceptance, "E2 acceptance")
            if decision.get("status") != "accepted" or decision.get("predictor") != "catboost":
                raise RFInputError("E2 acceptance differs")
            payloads = {
                "e2/acceptance.json": acceptance,
                "e2/handoff_manifest.json": handoff_manifest,
                _MODEL_DELIVERY: delivery,
            }
            for fold, output_name in _FOLD_MEMBERS.items():
                source_name = f"ensembles/{fold[0]}_{fold[1]}.csv"
                payload = nested.read(nested_infos[source_name])
                _prediction(payload, source_name)
                payloads[output_name] = payload
            return payloads
    except (BadZipFile, KeyError) as error:
        raise RFInputError("E2 handoff nested members differ") from error


def prepare_rf_input(
    *,
    e2_handoff: Path,
    output: Path,
    expected_e2_sha256: str | None = None,
) -> Path:
    contract = load_rf_contract()
    expected = expected_e2_sha256 or contract.inputs["e2_handoff_sha256"]
    payloads = _read_e2_handoff(Path(e2_handoff), expected)
    members = {
        name: {"sha256": sha256(payload).hexdigest(), "size": len(payload)}
        for name, payload in sorted(payloads.items())
    }
    row_hashes = {
        f"{fold[0]}->{fold[1]}": _prediction(payloads[name], name)[1]
        for fold, name in _FOLD_MEMBERS.items()
    }
    manifest = {
        "schema_version": 1,
        "artifact_kind": INPUT_KIND,
        "campaign_id": contract.campaign_id,
        "contract_sha256": contract_sha256(),
        "e2_handoff_sha256": expected,
        "e2_candidate_id": "c1_anchor_residual",
        "official_train_sha256": contract.inputs["official_train_sha256"],
        "official_history_sha256": contract.inputs["official_history_sha256"],
        "row_hashes": row_hashes,
        "members": members,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with ZipFile(temporary, "w") as archive:
            for name, payload in sorted(payloads.items()):
                archive.writestr(_zip_info(name), payload)
            archive.writestr(_zip_info("manifest.json"), _canonical(manifest))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _source_payloads(path: Path) -> dict[str, bytes]:
    source = Path(path)
    if source.is_dir():
        manifest_path = source / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise RFInputError("RF input directory manifest is absent")
        manifest_payload = manifest_path.read_bytes()
        manifest = _json(manifest_payload, "RF input manifest")
        declared = manifest.get("members")
        if type(declared) is not dict:
            raise RFInputError("RF input directory manifest members differ")
        payloads = {"manifest.json": manifest_payload}
        total = len(manifest_payload)
        root = source.resolve()
        for name in sorted(declared):
            member = PurePosixPath(name)
            if (
                type(name) is not str
                or member.is_absolute()
                or ".." in member.parts
                or "\\" in name
                or name == "manifest.json"
            ):
                raise RFInputError("unsafe RF input directory member")
            target = source / name
            resolved = target.resolve()
            if (
                target.is_symlink()
                or not target.is_file()
                or not resolved.is_relative_to(root)
            ):
                raise RFInputError(f"RF input directory member is absent: {name}")
            payload = target.read_bytes()
            total += len(payload)
            if total > _MAX_INPUT_EXPANDED:
                raise RFInputError("RF input expanded size exceeds limit")
            payloads[name] = payload
        return payloads
    try:
        with ZipFile(source) as archive:
            infos = _safe_infos(archive, "RF input", _MAX_INPUT_EXPANDED)
            return {name: archive.read(info) for name, info in infos.items()}
    except (OSError, BadZipFile) as error:
        raise RFInputError("RF input is not a valid archive or directory") from error


def _extract_delivery(payload: bytes, destination: Path) -> Path:
    _validate_delivery(payload)
    root = destination / "e2/model_delivery"
    root.mkdir(parents=True)
    with ZipFile(io.BytesIO(payload)) as archive:
        infos = _safe_infos(archive, "E2 delivery", _MAX_NESTED_EXPANDED)
        for name, info in infos.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
    return root


def verify_and_extract_rf_input(
    path: Path,
    destination: Path,
    *,
    expected_e2_sha256: str | None = None,
) -> VerifiedRFInput:
    payloads = _source_payloads(Path(path))
    if set(payloads) != {*_PAYLOAD_NAMES, "manifest.json"}:
        raise RFInputError("RF input member set differs")
    manifest_bytes = payloads["manifest.json"]
    manifest = _json(manifest_bytes, "RF input manifest")
    contract = load_rf_contract()
    expected = expected_e2_sha256 or contract.inputs["e2_handoff_sha256"]
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_kind") != INPUT_KIND
        or manifest.get("campaign_id") != contract.campaign_id
        or manifest.get("contract_sha256") != contract_sha256()
        or manifest.get("e2_handoff_sha256") != expected
        or manifest.get("e2_candidate_id") != "c1_anchor_residual"
        or manifest.get("official_train_sha256") != contract.inputs["official_train_sha256"]
        or manifest.get("official_history_sha256") != contract.inputs["official_history_sha256"]
    ):
        raise RFInputError("RF input identity differs")
    declared = manifest.get("members")
    if type(declared) is not dict or set(declared) != _PAYLOAD_NAMES:
        raise RFInputError("RF input manifest members differ")
    for name in sorted(_PAYLOAD_NAMES):
        metadata = declared[name]
        payload = payloads[name]
        if (
            type(metadata) is not dict
            or metadata.get("size") != len(payload)
            or metadata.get("sha256") != sha256(payload).hexdigest()
        ):
            raise RFInputError(f"member SHA-256 differs: {name}")
    row_hashes = manifest.get("row_hashes")
    if type(row_hashes) is not dict or set(row_hashes) != {f"{a}->{b}" for a, b in FOLDS}:
        raise RFInputError("RF row hashes differ")
    for fold, name in _FOLD_MEMBERS.items():
        _, digest = _prediction(payloads[name], name)
        if row_hashes[f"{fold[0]}->{fold[1]}"] != digest:
            raise RFInputError("RF row identity hash differs")

    root = Path(destination)
    if root.exists():
        raise RFInputError("RF input destination already exists")
    root.mkdir(parents=True)
    for name, payload in payloads.items():
        if name == _MODEL_DELIVERY:
            continue
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    full_fit_root = _extract_delivery(payloads[_MODEL_DELIVERY], root)
    return VerifiedRFInput(
        root=root,
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
        e2_handoff_sha256=expected,
        e2_candidate_id="c1_anchor_residual",
        fold_predictions=MappingProxyType({fold: root / name for fold, name in _FOLD_MEMBERS.items()}),
        acceptance=root / "e2/acceptance.json",
        full_fit_root=full_fit_root,
    )


def verify_official_data(root: Path, *, testing: bool = False) -> VerifiedOfficialData:
    source = Path(root)
    if source.is_symlink() or not source.is_dir():
        raise RFInputError("official data directory is missing")
    resolved = source.resolve()
    if testing and not resolved.is_relative_to(Path(tempfile.gettempdir()).resolve()):
        raise RFInputError("testing hash bypass requires temporary data")
    found: dict[str, list[Path]] = {"train.csv": [], "trackman_history.csv": []}
    for directory, directory_names, file_names in os.walk(resolved, followlinks=False):
        current = Path(directory)
        if any((current / name).is_symlink() for name in directory_names):
            raise RFInputError("official data contains a symlink directory")
        for name in found:
            if name in file_names:
                candidate = current / name
                if candidate.is_symlink() or not candidate.is_file():
                    raise RFInputError(f"official {name} is not a regular file")
                found[name].append(candidate)
    for name, candidates in found.items():
        if len(candidates) != 1 or candidates[0].parent != resolved:
            raise RFInputError(f"official {name} must occur exactly once at the data root")
    train = found["train.csv"][0]
    history = found["trackman_history.csv"][0]
    train_hash = file_sha256(train)
    history_hash = file_sha256(history)
    contract = load_rf_contract()
    if not testing and (
        train_hash != contract.inputs["official_train_sha256"]
        or history_hash != contract.inputs["official_history_sha256"]
    ):
        raise RFInputError("official data SHA-256 differs")
    return VerifiedOfficialData(resolved, train, history, train_hash, history_hash)
