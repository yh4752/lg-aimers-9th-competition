from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from types import MappingProxyType
from typing import Iterable, Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd

from .e2_artifacts import E2ArtifactError, verify_e2_handoff
from .hc_contracts import contract_sha256, file_sha256, load_hc_contract
from .inputs import PREDICTION_COLUMNS


class HCInputError(ValueError):
    """Raised when hierarchical campaign inputs are unsafe or change identity."""


INPUT_KIND = "tree_hierarchical_input_v1"
_FOLDS = ((2021, 2022), (2022, 2023), (2023, 2024))
_FOLD_MEMBERS = {fold: f"e2/fold_{fold[0]}_{fold[1]}.csv" for fold in _FOLDS}
_PAYLOAD_NAMES = {
    "e2/acceptance.json",
    "e2/handoff_manifest.json",
    "e2/model_delivery.zip",
    *_FOLD_MEMBERS.values(),
}
_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)
_MAX_MEMBER = 2 * 1024 * 1024 * 1024
_MAX_TOTAL = 8 * 1024 * 1024 * 1024
_MAX_RATIO = 250.0


@dataclass(frozen=True)
class VerifiedHCEvidence:
    root: Path
    manifest_sha256: str
    e2_handoff_sha256: str
    candidate_id: str
    acceptance: Path
    e2_delivery: Path
    fold_predictions: Mapping[tuple[int, int], Path]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HCInputError(f"{label} is invalid JSON") from error
    if type(value) is not dict:
        raise HCInputError(f"{label} must be an object")
    return value


def _info(name: str) -> ZipInfo:
    info = ZipInfo(name, _ZIP_TIMESTAMP)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _safe_infos(archive: ZipFile, label: str) -> dict[str, ZipInfo]:
    result: dict[str, ZipInfo] = {}
    total = 0
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        ratio = info.file_size / max(1, info.compress_size)
        if (
            info.filename in result
            or info.flag_bits & 0x1
            or info.is_dir()
            or path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or "\\" in info.filename
            or stat.S_ISLNK(mode)
            or info.file_size > _MAX_MEMBER
            or (info.file_size >= 64 * 1024 and ratio > _MAX_RATIO)
        ):
            raise HCInputError(f"unsafe or duplicate member in {label}")
        total += info.file_size
        if total > _MAX_TOTAL:
            raise HCInputError(f"{label} expanded size exceeds limit")
        result[info.filename] = info
    return result


def _source_payloads(path: Path, label: str) -> dict[str, bytes]:
    source = Path(path)
    if source.is_symlink():
        raise HCInputError(f"unsafe {label} source")
    if source.is_dir():
        payloads: dict[str, bytes] = {}
        total = 0
        for item in sorted(source.rglob("*")):
            if item.is_symlink():
                raise HCInputError(f"unsafe {label} member")
            if item.is_file():
                name = item.relative_to(source).as_posix()
                payload = item.read_bytes()
                total += len(payload)
                if len(payload) > _MAX_MEMBER or total > _MAX_TOTAL:
                    raise HCInputError(f"{label} expanded size exceeds limit")
                payloads[name] = payload
        return payloads
    if not source.is_file():
        raise HCInputError(f"{label} source is absent")
    try:
        with ZipFile(source) as archive:
            infos = _safe_infos(archive, label)
            return {name: archive.read(info) for name, info in infos.items()}
    except BadZipFile as error:
        raise HCInputError(f"{label} is not a valid ZIP") from error


def _verify_declared_members(
    payloads: Mapping[str, bytes], manifest: Mapping[str, object], manifest_name: str
) -> None:
    declared = manifest.get("members")
    if type(declared) is not dict or set(declared) != set(payloads) - {manifest_name}:
        raise HCInputError("member set differs")
    for name, evidence in declared.items():
        payload = payloads[name]
        if (
            type(evidence) is not dict
            or set(evidence) != {"size", "sha256"}
            or evidence["size"] != len(payload)
            or evidence["sha256"] != sha256(payload).hexdigest()
        ):
            raise HCInputError(f"member SHA-256 differs: {name}")


def _prediction(payload: bytes, label: str) -> tuple[pd.DataFrame, str]:
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except Exception as error:
        raise HCInputError(f"{label} prediction CSV is invalid") from error
    if tuple(frame.columns) != PREDICTION_COLUMNS or frame.empty:
        raise HCInputError(f"{label} prediction schema differs")
    if frame["row_id"].isna().any() or not frame["row_id"].is_unique:
        raise HCInputError(f"{label} row identity differs")
    target = pd.to_numeric(frame["target"], errors="coerce")
    probability = pd.to_numeric(frame["probability"], errors="coerce")
    if not target.isin([0, 1]).all() or not probability.between(0, 1).all():
        raise HCInputError(f"{label} prediction values differ")
    row_hash = sha256("\n".join(frame["row_id"].astype(str)).encode()).hexdigest()
    return frame, row_hash


def _validate_delivery(payload: bytes) -> None:
    try:
        with ZipFile(io.BytesIO(payload)) as archive:
            infos = _safe_infos(archive, "E2 delivery")
            if "manifest.json" not in infos:
                raise HCInputError("E2 delivery manifest is absent")
            payloads = {name: archive.read(info) for name, info in infos.items()}
    except BadZipFile as error:
        raise HCInputError("E2 delivery is not a valid ZIP") from error
    manifest = _json(payloads["manifest.json"], "E2 delivery manifest")
    required = {
        "evidence/acceptance_decision.json",
        "evidence/full_fit_manifest.json",
        "evidence/inference_audit.json",
        "frozen_state/feature_state.json",
        *(f"models/catboost_seed_{seed}.cbm" for seed in (42, 2026, 3407)),
    }
    if (
        manifest.get("artifact_kind") != "tree_expert_e2_model_delivery_v1"
        or manifest.get("campaign_id") != "tree_expert_e2_v1"
        or manifest.get("candidate_id") != "c1_anchor_residual"
        or manifest.get("predictor") != "catboost"
        or manifest.get("seeds") != [42, 2026, 3407]
        or manifest.get("submission_package") is not False
        or not required.issubset(payloads)
    ):
        raise HCInputError("E2 delivery identity differs")
    _verify_declared_members(payloads, manifest, "manifest.json")


def _read_accepted_e2_handoff(path: Path, expected_sha256: str) -> dict[str, bytes]:
    source = Path(path)
    if (
        source.is_symlink()
        or not source.is_file()
        or file_sha256(source) != expected_sha256
    ):
        raise HCInputError("E2 handoff SHA-256 differs")
    try:
        verified = verify_e2_handoff(source)
    except E2ArtifactError as error:
        raise HCInputError(f"E2 handoff verification failed: {error}") from error
    if verified.status != "accepted" or verified.delivery is not True:
        raise HCInputError("E2 handoff is not accepted")
    try:
        with ZipFile(source) as outer:
            infos = _safe_infos(outer, "E2 handoff")
            review = outer.read(infos["tree_expert_e2_review.zip"])
            delivery = outer.read(infos["tree_expert_e2_model_delivery.zip"])
            handoff_manifest = outer.read(infos["handoff_manifest.json"])
        _validate_delivery(delivery)
        with ZipFile(io.BytesIO(review)) as nested:
            nested_infos = _safe_infos(nested, "E2 review")
            acceptance = nested.read(nested_infos["decisions/acceptance.json"])
            decision = _json(acceptance, "E2 acceptance")
            if decision.get("status") != "accepted" or decision.get("predictor") != "catboost":
                raise HCInputError("E2 acceptance differs")
            payloads = {
                "e2/acceptance.json": acceptance,
                "e2/handoff_manifest.json": handoff_manifest,
                "e2/model_delivery.zip": delivery,
            }
            for fold, output_name in _FOLD_MEMBERS.items():
                source_name = f"ensembles/{fold[0]}_{fold[1]}.csv"
                payload = nested.read(nested_infos[source_name])
                _prediction(payload, source_name)
                payloads[output_name] = payload
            return payloads
    except (BadZipFile, KeyError) as error:
        raise HCInputError("E2 handoff nested members differ") from error


def _write_zip(path: Path, members: Mapping[str, bytes]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for name, payload in sorted(members.items()):
                archive.writestr(_info(name), payload)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def prepare_hc_input(
    *, e2_handoff: Path, output: Path, expected_e2_sha256: str | None = None
) -> Path:
    contract = load_hc_contract()
    expected = expected_e2_sha256 or contract.inputs["e2_handoff_sha256"]
    payloads = _read_accepted_e2_handoff(Path(e2_handoff), expected)
    members = {
        name: {"size": len(payload), "sha256": sha256(payload).hexdigest()}
        for name, payload in sorted(payloads.items())
    }
    row_hashes = {
        f"{fold[0]}->{fold[1]}": _prediction(payloads[name], name)[1]
        for fold, name in _FOLD_MEMBERS.items()
    }
    manifest = _canonical(
        {
            "schema_version": 1,
            "artifact_kind": INPUT_KIND,
            "campaign_id": contract.campaign_id,
            "contract_sha256": contract_sha256(),
            "e2_handoff_sha256": expected,
            "candidate_id": "c1_anchor_residual",
            "official_train_sha256": contract.inputs["official_train_sha256"],
            "official_history_sha256": contract.inputs["official_history_sha256"],
            "row_hashes": row_hashes,
            "members": members,
        }
    )
    return _write_zip(Path(output), {**payloads, "manifest.json": manifest})


def verify_and_extract_hc_input(
    path: Path,
    destination: Path,
    *,
    expected_e2_sha256: str | None = None,
) -> VerifiedHCEvidence:
    contract = load_hc_contract()
    expected = expected_e2_sha256 or contract.inputs["e2_handoff_sha256"]
    payloads = _source_payloads(Path(path), "HC input")
    if set(payloads) != {*_PAYLOAD_NAMES, "manifest.json"}:
        raise HCInputError("HC input member set differs")
    manifest_bytes = payloads["manifest.json"]
    manifest = _json(manifest_bytes, "HC input manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_kind") != INPUT_KIND
        or manifest.get("campaign_id") != contract.campaign_id
        or manifest.get("contract_sha256") != contract_sha256()
        or manifest.get("e2_handoff_sha256") != expected
        or manifest.get("candidate_id") != "c1_anchor_residual"
        or manifest.get("official_train_sha256") != contract.inputs["official_train_sha256"]
        or manifest.get("official_history_sha256") != contract.inputs["official_history_sha256"]
    ):
        raise HCInputError("HC input identity differs")
    _verify_declared_members(payloads, manifest, "manifest.json")
    _validate_delivery(payloads["e2/model_delivery.zip"])
    row_hashes = manifest.get("row_hashes")
    if type(row_hashes) is not dict or set(row_hashes) != {
        f"{fold[0]}->{fold[1]}" for fold in _FOLDS
    }:
        raise HCInputError("HC input row hashes differ")
    for fold, name in _FOLD_MEMBERS.items():
        _, digest = _prediction(payloads[name], name)
        if row_hashes[f"{fold[0]}->{fold[1]}"] != digest:
            raise HCInputError(f"HC input row hash differs: {name}")

    output = Path(destination)
    if output.exists() and (output.is_symlink() or any(output.iterdir())):
        raise HCInputError("HC input destination is not empty")
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        for name, payload in sorted(payloads.items()):
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return VerifiedHCEvidence(
        root=output,
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
        e2_handoff_sha256=expected,
        candidate_id="c1_anchor_residual",
        acceptance=output / "e2/acceptance.json",
        e2_delivery=output / "e2/model_delivery.zip",
        fold_predictions=MappingProxyType(
            {fold: output / name for fold, name in _FOLD_MEMBERS.items()}
        ),
    )


def _input_identity(path: Path) -> tuple[str, str] | None:
    try:
        payloads = _source_payloads(Path(path), "HC input candidate")
    except HCInputError:
        return None
    manifest_bytes = payloads.get("manifest.json")
    if manifest_bytes is None:
        return None
    manifest = _json(manifest_bytes, "HC input manifest")
    if manifest.get("artifact_kind") != INPUT_KIND:
        return None
    return str(manifest.get("artifact_kind")), sha256(manifest_bytes).hexdigest()


def choose_unique_hc_input(paths: Iterable[Path]) -> Path:
    found: list[tuple[Path, tuple[str, str]]] = []
    for path in paths:
        identity = _input_identity(Path(path))
        if identity is not None:
            found.append((Path(path), identity))
    identities = {identity for _, identity in found}
    if not found:
        raise HCInputError("HC input is absent")
    if len(identities) != 1:
        raise HCInputError("distinct HC input identities")
    return found[0][0]
