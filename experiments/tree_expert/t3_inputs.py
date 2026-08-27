from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd

from .t3_contracts import contract_sha256, load_t3_contract


class T3InputError(ValueError):
    pass


FOLDS = ((2021, 2022), (2022, 2023), (2023, 2024))
INPUT_KIND = "tree_expert_t3_input_v1"
_FOLD_MEMBERS = {fold: f"e2/fold_{fold[0]}_{fold[1]}.csv" for fold in FOLDS}
_PAYLOAD_NAMES = {
    "e2/acceptance.json", "e2/handoff_manifest.json", *_FOLD_MEMBERS.values(),
}
_MAX_EXPANDED = 256 * 1024 * 1024
_MAX_E2_HANDOFF_EXPANDED = 1024 * 1024 * 1024
_MAX_E2_REVIEW_EXPANDED = 512 * 1024 * 1024


@dataclass(frozen=True)
class VerifiedT3Input:
    root: Path
    manifest_sha256: str
    e2_handoff_sha256: str
    e2_candidate_id: str
    fold_predictions: Mapping[tuple[int, int], Path]
    acceptance: Path
    handoff_manifest: Path


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
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _json(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise T3InputError(f"{label} is invalid JSON") from error
    if type(value) is not dict:
        raise T3InputError(f"{label} must be an object")
    return value


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _safe_infos(
    archive: ZipFile,
    label: str,
    *,
    max_expanded: int | None = None,
) -> dict[str, ZipInfo]:
    result: dict[str, ZipInfo] = {}
    total = 0
    limit = _MAX_EXPANDED if max_expanded is None else max_expanded
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if (
            info.filename in result or info.flag_bits & 0x1 or info.is_dir()
            or path.is_absolute() or ".." in path.parts or "\\" in info.filename
        ):
            raise T3InputError(f"unsafe or duplicate member in {label}")
        total += info.file_size
        if total > limit:
            raise T3InputError(f"{label} expanded size exceeds limit")
        result[info.filename] = info
    return result


def _prediction(payload: bytes, label: str) -> tuple[pd.DataFrame, str]:
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except Exception as error:
        raise T3InputError(f"{label} prediction CSV is invalid") from error
    required = {"row_id", "target", "probability", "game_type"}
    if not required.issubset(frame.columns) or frame.empty:
        raise T3InputError(f"{label} prediction schema differs")
    if frame["row_id"].isna().any() or not frame["row_id"].is_unique:
        raise T3InputError(f"{label} row identity differs")
    target = pd.to_numeric(frame["target"], errors="coerce")
    probability = pd.to_numeric(frame["probability"], errors="coerce")
    if not target.isin([0, 1]).all() or not probability.between(0, 1).all():
        raise T3InputError(f"{label} prediction values differ")
    row_hash = sha256("\n".join(frame["row_id"].astype(str)).encode("utf-8")).hexdigest()
    return frame, row_hash


def _read_e2_handoff(path: Path, expected_sha: str) -> dict[str, bytes]:
    if not Path(path).is_file() or file_sha256(path) != expected_sha:
        raise T3InputError("E2 handoff SHA-256 differs")
    try:
        with ZipFile(path) as outer:
            infos = _safe_infos(
                outer,
                "E2 handoff",
                max_expanded=_MAX_E2_HANDOFF_EXPANDED,
            )
            if "handoff_manifest.json" not in infos or "tree_expert_e2_review.zip" not in infos:
                raise T3InputError("E2 handoff members differ")
            handoff_bytes = outer.read(infos["handoff_manifest.json"])
            handoff = _json(handoff_bytes, "E2 handoff manifest")
            if handoff.get("status") != "accepted" or handoff.get("artifact_kind") != "tree_expert_e2_handoff_v1":
                raise T3InputError("E2 handoff is not accepted")
            review = outer.read(infos["tree_expert_e2_review.zip"])
            declared = handoff.get("members", {}).get("tree_expert_e2_review.zip", {})
            if declared.get("sha256") != sha256(review).hexdigest() or declared.get("size") != len(review):
                raise T3InputError("E2 review binding differs")
        with ZipFile(io.BytesIO(review)) as nested:
            nested_infos = _safe_infos(
                nested,
                "E2 review",
                max_expanded=_MAX_E2_REVIEW_EXPANDED,
            )
            names = {
                "decisions/acceptance.json",
                "ensembles/2021_2022.csv", "ensembles/2022_2023.csv", "ensembles/2023_2024.csv",
            }
            if not names.issubset(nested_infos):
                raise T3InputError("E2 review members differ")
            acceptance = nested.read(nested_infos["decisions/acceptance.json"])
            decision = _json(acceptance, "E2 acceptance")
            if decision.get("status") != "accepted" or decision.get("predictor") != "catboost":
                raise T3InputError("E2 acceptance differs")
            payloads = {
                "e2/acceptance.json": acceptance,
                "e2/handoff_manifest.json": handoff_bytes,
            }
            for fold, output_name in _FOLD_MEMBERS.items():
                source_name = f"ensembles/{fold[0]}_{fold[1]}.csv"
                payload = nested.read(nested_infos[source_name])
                _prediction(payload, source_name)
                payloads[output_name] = payload
            return payloads
    except BadZipFile as error:
        raise T3InputError("E2 handoff is not a valid ZIP") from error


def prepare_t3_input(
    *,
    e2_handoff: Path,
    output: Path,
    expected_e2_sha256: str | None = None,
) -> Path:
    contract = load_t3_contract()
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
        "schema_version": 1, "artifact_kind": INPUT_KIND,
        "campaign_id": contract.campaign_id, "contract_sha256": contract_sha256(),
        "e2_handoff_sha256": expected, "e2_candidate_id": "c1_anchor_residual",
        "official_train_sha256": contract.inputs["official_train_sha256"],
        "official_history_sha256": contract.inputs["official_history_sha256"],
        "row_hashes": row_hashes, "members": members,
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
        names = sorted(item.relative_to(source).as_posix() for item in source.rglob("*") if item.is_file())
        return {name: (source / name).read_bytes() for name in names}
    try:
        with ZipFile(source) as archive:
            infos = _safe_infos(archive, "T3 input")
            return {name: archive.read(info) for name, info in infos.items()}
    except (OSError, BadZipFile) as error:
        raise T3InputError("T3 input is not a valid archive or directory") from error


def verify_and_extract_t3_input(
    path: Path,
    destination: Path,
    *,
    expected_e2_sha256: str | None = None,
) -> VerifiedT3Input:
    payloads = _source_payloads(Path(path))
    if set(payloads) != {*_PAYLOAD_NAMES, "manifest.json"}:
        raise T3InputError("T3 input member set differs")
    manifest_bytes = payloads["manifest.json"]
    manifest = _json(manifest_bytes, "T3 input manifest")
    contract = load_t3_contract()
    expected = expected_e2_sha256 or contract.inputs["e2_handoff_sha256"]
    if (
        manifest.get("schema_version") != 1 or manifest.get("artifact_kind") != INPUT_KIND
        or manifest.get("campaign_id") != contract.campaign_id
        or manifest.get("contract_sha256") != contract_sha256()
        or manifest.get("e2_handoff_sha256") != expected
        or manifest.get("e2_candidate_id") != "c1_anchor_residual"
        or manifest.get("official_train_sha256") != contract.inputs["official_train_sha256"]
        or manifest.get("official_history_sha256") != contract.inputs["official_history_sha256"]
    ):
        raise T3InputError("T3 input identity differs")
    declared = manifest.get("members")
    if type(declared) is not dict or set(declared) != _PAYLOAD_NAMES:
        raise T3InputError("T3 input manifest members differ")
    for name in sorted(_PAYLOAD_NAMES):
        metadata = declared[name]
        payload = payloads[name]
        if type(metadata) is not dict or metadata.get("size") != len(payload) or metadata.get("sha256") != sha256(payload).hexdigest():
            raise T3InputError(f"member SHA-256 differs: {name}")
    row_hashes = manifest.get("row_hashes")
    if type(row_hashes) is not dict or set(row_hashes) != {f"{a}->{b}" for a, b in FOLDS}:
        raise T3InputError("T3 row hashes differ")
    for fold, name in _FOLD_MEMBERS.items():
        _, digest = _prediction(payloads[name], name)
        if row_hashes[f"{fold[0]}->{fold[1]}"] != digest:
            raise T3InputError("T3 row identity hash differs")
    root = Path(destination)
    if root.exists():
        raise T3InputError("T3 input destination already exists")
    root.mkdir(parents=True)
    for name, payload in payloads.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return VerifiedT3Input(
        root=root,
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
        e2_handoff_sha256=expected,
        e2_candidate_id="c1_anchor_residual",
        fold_predictions=MappingProxyType({fold: root / name for fold, name in _FOLD_MEMBERS.items()}),
        acceptance=root / "e2/acceptance.json",
        handoff_manifest=root / "e2/handoff_manifest.json",
    )


def verify_official_data(root: Path) -> VerifiedOfficialData:
    source = Path(root)
    if not source.is_dir():
        raise T3InputError("official data directory is missing")
    train_candidates = [path for path in source.rglob("train.csv") if path.is_file()]
    history_candidates = [path for path in source.rglob("trackman_history.csv") if path.is_file()]
    if len(train_candidates) != 1 or len(history_candidates) != 1:
        raise T3InputError("official data file count differs")
    train = train_candidates[0]
    history = history_candidates[0]
    if train.parent != history.parent:
        raise T3InputError("official data roots differ")
    contract = load_t3_contract()
    train_hash = file_sha256(train)
    history_hash = file_sha256(history)
    if train_hash != contract.inputs["official_train_sha256"] or history_hash != contract.inputs["official_history_sha256"]:
        raise T3InputError("official data SHA-256 differs")
    return VerifiedOfficialData(train.parent, train, history, train_hash, history_hash)
