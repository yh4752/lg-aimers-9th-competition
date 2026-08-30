from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from types import MappingProxyType
from typing import BinaryIO, Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd


class DirectExpertInputError(ValueError):
    pass


EXPECTED_S4_HANDOFF_SHA256 = "5a410548de99d5d9c56f9b0d1940d5080e9167eafd97a2c94c45e31deacb4911"
EXPECTED_E2_SUBMISSION_SHA256 = "8bc33042b9258049f905df0c733b1153d1cedfac39cb0732b2a7e154629d779a"
EXPECTED_LOGICAL_E2_HANDOFF_SHA256 = "4dd0c9012b9a59e5235b76d6fe1f6ded4df4b404cab0082c0f3084b8c384050f"
INPUT_MEMBERS = frozenset(
    {
        "e2_oof/2021.csv",
        "e2_oof/2022.csv",
        "e2_oof/2023.csv",
        "e2_oof/2024.csv",
        "e2_submission/catboost_3seed_v1.zip",
        "evidence/s4_manifest.json",
        "evidence/e2_submission_receipt.json",
        "manifest.json",
    }
)
_PAYLOAD_MEMBERS = INPUT_MEMBERS - {"manifest.json"}
_S4_MEMBERS = {"campaign.log", "resume.zip", "review.zip", "manifest.json"}
_E2_MEMBERS = {
    "script.py",
    "requirements.txt",
    "model/frozen_state/feature_state.json",
    "model/frozen_state/s1_batter.csv",
    "model/frozen_state/s1_pitcher.csv",
    "model/models/catboost_seed_42.cbm",
    "model/models/catboost_seed_2026.cbm",
    "model/models/catboost_seed_3407.cbm",
}
_E2_SUBMISSION_MEMBER = "e2_submission/catboost_3seed_v1.zip"
_OOF_COLUMNS = (
    "row_id",
    "game_type",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "balls_before",
    "strikes_before",
    "outs_before",
    "base_state",
    "target",
    "p_anchor",
    "oof_year",
)
_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
_BLOCK = 1024 * 1024
_MAX_FILES = 4096
_MAX_MEMBER_BYTES = 4 * 1024 * 1024 * 1024
_MAX_TOTAL_BYTES = 12 * 1024 * 1024 * 1024
_MAX_RATIO = 300.0


@dataclass(frozen=True)
class VerifiedDirectExpertInput:
    root: Path
    manifest_sha256: str
    source_s4_sha256: str
    e2_submission_sha256: str
    logical_e2_handoff_sha256: str
    e2_oof_years: tuple[int, ...]
    e2_oof_paths: Mapping[int, Path]
    e2_submission_path: Path
    train_sha256: str
    history_sha256: str


def canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DirectExpertInputError("JSON value is not canonicalizable") from error


def file_sha256(path: Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise DirectExpertInputError(f"source is not a regular file: {source}")
    digest = sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _hex(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DirectExpertInputError(f"{label} differs")
    return value


def _json(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DirectExpertInputError(f"{label} is unreadable") from error
    if type(value) is not dict:
        raise DirectExpertInputError(f"{label} differs")
    return value


def _safe_infos(archive: ZipFile, label: str) -> dict[str, ZipInfo]:
    output: dict[str, ZipInfo] = {}
    total = 0
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        ratio = info.file_size / max(1, info.compress_size)
        if (
            info.filename in output
            or info.is_dir()
            or info.flag_bits & 1
            or path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or "\\" in info.filename
            or stat.S_ISLNK(mode)
            or info.file_size > _MAX_MEMBER_BYTES
            or (info.file_size >= 64 * 1024 and ratio > _MAX_RATIO)
        ):
            raise DirectExpertInputError(f"unsafe {label} member: {info.filename}")
        total += info.file_size
        if len(output) >= _MAX_FILES or total > _MAX_TOTAL_BYTES:
            raise DirectExpertInputError(f"{label} expansion exceeds limit")
        output[info.filename] = info
    return output


def _digest_stream(source: BinaryIO, target: BinaryIO | None = None) -> tuple[int, str]:
    digest = sha256()
    size = 0
    for block in iter(lambda: source.read(_BLOCK), b""):
        size += len(block)
        digest.update(block)
        if target is not None:
            target.write(block)
    return size, digest.hexdigest()


def _metadata(manifest: Mapping[str, object], names: set[str], label: str) -> Mapping[str, object]:
    members = manifest.get("members")
    if type(members) is not dict or set(members) != names:
        raise DirectExpertInputError(f"{label} member set differs")
    return members


def _verify_archive_members(
    archive: ZipFile,
    infos: Mapping[str, ZipInfo],
    members: Mapping[str, object],
    *,
    capture: Mapping[str, Path] | None = None,
) -> None:
    capture = capture or {}
    for name in sorted(members):
        declared = members[name]
        if type(declared) is not dict:
            raise DirectExpertInputError(f"member metadata differs: {name}")
        target_path = capture.get(name)
        if target_path is not None:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(infos[name]) as source, target_path.open("wb") as target:
                size, digest = _digest_stream(source, target)
        else:
            with archive.open(infos[name]) as source:
                size, digest = _digest_stream(source)
        if declared != {"sha256": digest, "size": size}:
            raise DirectExpertInputError(f"member digest differs: {name}")


def _verify_e2_submission(path: Path, expected_sha256: str) -> None:
    if file_sha256(path) != expected_sha256:
        raise DirectExpertInputError("E2 submission SHA-256 differs")
    try:
        with ZipFile(path) as archive:
            infos = _safe_infos(archive, "E2 submission")
            if set(infos) != _E2_MEMBERS:
                raise DirectExpertInputError("E2 submission member set differs")
    except (OSError, BadZipFile) as error:
        if isinstance(error, DirectExpertInputError):
            raise
        raise DirectExpertInputError("E2 submission is not a valid ZIP") from error


def _validate_oof(path: Path, year: int) -> None:
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        raise DirectExpertInputError(f"E2 OOF {year} is unreadable") from error
    if tuple(frame.columns) != _OOF_COLUMNS or frame.empty:
        raise DirectExpertInputError(f"E2 OOF {year} columns differ")
    if frame["row_id"].isna().any() or not frame["row_id"].is_unique:
        raise DirectExpertInputError(f"E2 OOF {year} row identity differs")
    target = pd.to_numeric(frame["target"], errors="coerce")
    probability = pd.to_numeric(frame["p_anchor"], errors="coerce")
    oof_year = pd.to_numeric(frame["oof_year"], errors="coerce")
    if (
        not target.isin((0, 1)).all()
        or probability.isna().any()
        or not probability.between(0, 1).all()
        or not oof_year.eq(year).all()
    ):
        raise DirectExpertInputError(f"E2 OOF {year} values differ")


def _read_s4_oof(
    source: Path,
    temporary_root: Path,
    *,
    expected_s4_sha256: str,
) -> tuple[dict[int, Path], bytes, dict[str, str]]:
    if file_sha256(source) != expected_s4_sha256:
        raise DirectExpertInputError("S4 handoff SHA-256 differs")
    resume_path = temporary_root / "s4_resume.zip"
    try:
        with ZipFile(source) as outer:
            infos = _safe_infos(outer, "S4 handoff")
            if set(infos) != _S4_MEMBERS:
                raise DirectExpertInputError("S4 handoff member set differs")
            manifest_bytes = outer.read(infos["manifest.json"])
            manifest = _json(manifest_bytes, "S4 handoff manifest")
            if manifest.get("artifact_kind") != "tree_s4_handoff_v1":
                raise DirectExpertInputError("S4 handoff identity differs")
            bindings = manifest.get("bindings")
            if type(bindings) is not dict:
                raise DirectExpertInputError("S4 handoff bindings differ")
            for key in (
                "code_sha256",
                "contract_sha256",
                "e2_handoff_sha256",
                "history_sha256",
                "input_manifest_sha256",
                "train_sha256",
            ):
                _hex(bindings.get(key), f"S4 binding {key}")
            members = _metadata(manifest, _S4_MEMBERS - {"manifest.json"}, "S4 handoff")
            _verify_archive_members(
                outer,
                infos,
                members,
                capture={"resume.zip": resume_path},
            )
        with ZipFile(resume_path) as resume:
            infos = _safe_infos(resume, "S4 resume")
            if "manifest.json" not in infos:
                raise DirectExpertInputError("S4 resume manifest is absent")
            nested = _json(resume.read(infos["manifest.json"]), "S4 resume manifest")
            if nested.get("artifact_kind") != "tree_s4_resume_v1" or nested.get("bindings") != bindings:
                raise DirectExpertInputError("S4 resume identity differs")
            members = _metadata(nested, set(infos) - {"manifest.json"}, "S4 resume")
            captures = {
                f"anchors/e2/{year}.csv": temporary_root / f"e2_oof_{year}.csv"
                for year in (2021, 2022, 2023, 2024)
            }
            if not set(captures).issubset(infos):
                raise DirectExpertInputError("S4 E2 OOF members are absent")
            _verify_archive_members(resume, infos, members, capture=captures)
    except (OSError, BadZipFile) as error:
        if isinstance(error, DirectExpertInputError):
            raise
        raise DirectExpertInputError("S4 handoff is not a valid ZIP") from error
    output = {year: captures[f"anchors/e2/{year}.csv"] for year in (2021, 2022, 2023, 2024)}
    for year, path in output.items():
        _validate_oof(path, year)
    lineage = {
        "logical_e2_handoff_sha256": _hex(bindings["e2_handoff_sha256"], "logical E2 handoff"),
        "train_sha256": _hex(bindings["train_sha256"], "train binding"),
        "history_sha256": _hex(bindings["history_sha256"], "history binding"),
    }
    return output, manifest_bytes, lineage


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, _ZIP_TIME)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _payload_evidence(path: Path) -> dict[str, object]:
    return {"sha256": file_sha256(path), "size": path.stat().st_size}


def _write_archive(destination: Path, payloads: Mapping[str, Path], manifest: bytes) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for name, path in sorted(payloads.items()):
                with path.open("rb") as source, archive.open(_zip_info(name), "w") as target:
                    shutil.copyfileobj(source, target, _BLOCK)
            archive.writestr(_zip_info("manifest.json"), manifest)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def prepare_direct_expert_input(
    s4_handoff: Path,
    e2_submission: Path,
    e2_receipt: Path,
    output: Path,
    *,
    expected_s4_sha256: str = EXPECTED_S4_HANDOFF_SHA256,
    expected_e2_sha256: str = EXPECTED_E2_SUBMISSION_SHA256,
) -> Path:
    s4 = Path(s4_handoff)
    e2 = Path(e2_submission)
    receipt_path = Path(e2_receipt)
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise DirectExpertInputError("output already exists")
    _hex(expected_s4_sha256, "expected S4 SHA-256")
    _hex(expected_e2_sha256, "expected E2 SHA-256")
    _verify_e2_submission(e2, expected_e2_sha256)
    try:
        receipt_bytes = receipt_path.read_bytes()
    except OSError as error:
        raise DirectExpertInputError("E2 receipt is unreadable") from error
    receipt = _json(receipt_bytes, "E2 receipt")
    if (
        receipt.get("status") != "packaged"
        or receipt.get("candidate_id") != "c1_anchor_residual"
        or receipt.get("archive_sha256") != expected_e2_sha256
        or receipt.get("archive_bytes") != e2.stat().st_size
    ):
        raise DirectExpertInputError("E2 receipt differs")
    with tempfile.TemporaryDirectory(prefix="direct-expert-input-") as temporary_name:
        temporary = Path(temporary_name)
        oof, s4_manifest, lineage = _read_s4_oof(
            s4,
            temporary,
            expected_s4_sha256=expected_s4_sha256,
        )
        if (
            expected_s4_sha256 == EXPECTED_S4_HANDOFF_SHA256
            and lineage["logical_e2_handoff_sha256"] != EXPECTED_LOGICAL_E2_HANDOFF_SHA256
        ):
            raise DirectExpertInputError("logical E2 handoff SHA-256 differs")
        s4_manifest_path = temporary / "s4_manifest.json"
        receipt_copy = temporary / "e2_submission_receipt.json"
        s4_manifest_path.write_bytes(s4_manifest)
        receipt_copy.write_bytes(receipt_bytes)
        payloads = {
            **{f"e2_oof/{year}.csv": path for year, path in oof.items()},
            "e2_submission/catboost_3seed_v1.zip": e2,
            "evidence/s4_manifest.json": s4_manifest_path,
            "evidence/e2_submission_receipt.json": receipt_copy,
        }
        manifest = canonical_json(
            {
                "schema_version": 1,
                "artifact_kind": "direct_expert_input_v1",
                "bindings": {
                    "source_s4_sha256": expected_s4_sha256,
                    "logical_e2_handoff_sha256": lineage["logical_e2_handoff_sha256"],
                    "e2_submission_sha256": expected_e2_sha256,
                    "train_sha256": lineage["train_sha256"],
                    "history_sha256": lineage["history_sha256"],
                },
                "members": {name: _payload_evidence(path) for name, path in sorted(payloads.items())},
            }
        )
        _write_archive(destination, payloads, manifest)
    verify_and_extract_input(destination, destination.parent / f".{destination.name}.verification")
    shutil.rmtree(destination.parent / f".{destination.name}.verification")
    return destination


def _submission_zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _rebuild_expanded_e2_submission(source: Path, destination: Path) -> Path:
    candidates = (source.with_suffix(""), source.parent)
    roots = []
    for candidate in candidates:
        if candidate.is_symlink() or not candidate.is_dir():
            continue
        root = candidate.resolve()
        if all(
            not (candidate / name).is_symlink()
            and (candidate / name).is_file()
            and (candidate / name).resolve().is_relative_to(root)
            for name in _E2_MEMBERS
        ):
            roots.append(candidate)
    if len(roots) != 1:
        raise DirectExpertInputError("expanded E2 submission directory differs")
    root = roots[0]
    ordered = (
        "script.py",
        "requirements.txt",
        *sorted(_E2_MEMBERS - {"script.py", "requirements.txt"}),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "x") as archive:
        for name in ordered:
            archive.writestr(_submission_zip_info(name), (root / name).read_bytes())
    return destination


def _directory_payloads(source: Path, materialized_root: Path) -> dict[str, Path]:
    if source.is_symlink() or not source.is_dir():
        raise DirectExpertInputError("input directory differs")
    root = source.resolve()
    output: dict[str, Path] = {}
    for name in sorted(INPUT_MEMBERS):
        path = source / name
        if name == _E2_SUBMISSION_MEMBER and not path.is_file():
            path = _rebuild_expanded_e2_submission(path, materialized_root / name)
            output[name] = path
            continue
        resolved = path.resolve()
        if path.is_symlink() or not path.is_file() or not resolved.is_relative_to(root):
            raise DirectExpertInputError(f"input member is absent: {name}")
        output[name] = path
    return output


def _extract_archive(source: Path, destination: Path) -> dict[str, Path]:
    try:
        with ZipFile(source) as archive:
            infos = _safe_infos(archive, "direct expert input")
            if set(infos) != INPUT_MEMBERS:
                raise DirectExpertInputError("input member set differs")
            for name in sorted(infos):
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(infos[name]) as incoming, target.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, _BLOCK)
    except (OSError, BadZipFile) as error:
        if isinstance(error, DirectExpertInputError):
            raise
        raise DirectExpertInputError("direct expert input is not a valid ZIP") from error
    return {name: destination / name for name in INPUT_MEMBERS}


def verify_and_extract_input(source: Path, destination: Path) -> VerifiedDirectExpertInput:
    source = Path(source)
    output = Path(destination)
    if output.exists() or output.is_symlink():
        raise DirectExpertInputError("destination already exists")
    output.mkdir(parents=True)
    try:
        if source.is_dir():
            originals = _directory_payloads(source, output)
            for name, path in originals.items():
                target = output / name
                if path == target:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            payloads = {name: output / name for name in INPUT_MEMBERS}
        else:
            payloads = _extract_archive(source, output)
        manifest_bytes = payloads["manifest.json"].read_bytes()
        manifest = _json(manifest_bytes, "direct expert input manifest")
        if manifest.get("schema_version") != 1 or manifest.get("artifact_kind") != "direct_expert_input_v1":
            raise DirectExpertInputError("input identity differs")
        members = _metadata(manifest, set(_PAYLOAD_MEMBERS), "input")
        for name in sorted(_PAYLOAD_MEMBERS):
            declared = members[name]
            actual = _payload_evidence(payloads[name])
            if declared != actual:
                raise DirectExpertInputError(f"member digest differs: {name}")
        bindings = manifest.get("bindings")
        if type(bindings) is not dict or set(bindings) != {
            "source_s4_sha256",
            "logical_e2_handoff_sha256",
            "e2_submission_sha256",
            "train_sha256",
            "history_sha256",
        }:
            raise DirectExpertInputError("input bindings differ")
        for key in bindings:
            _hex(bindings[key], f"input binding {key}")
        _verify_e2_submission(
            payloads["e2_submission/catboost_3seed_v1.zip"],
            bindings["e2_submission_sha256"],
        )
        receipt = _json(payloads["evidence/e2_submission_receipt.json"].read_bytes(), "E2 receipt")
        if receipt.get("archive_sha256") != bindings["e2_submission_sha256"]:
            raise DirectExpertInputError("E2 receipt binding differs")
        years = (2021, 2022, 2023, 2024)
        oof_paths = {year: payloads[f"e2_oof/{year}.csv"] for year in years}
        for year, path in oof_paths.items():
            _validate_oof(path, year)
        return VerifiedDirectExpertInput(
            root=output,
            manifest_sha256=sha256(manifest_bytes).hexdigest(),
            source_s4_sha256=bindings["source_s4_sha256"],
            e2_submission_sha256=bindings["e2_submission_sha256"],
            logical_e2_handoff_sha256=bindings["logical_e2_handoff_sha256"],
            e2_oof_years=years,
            e2_oof_paths=MappingProxyType(oof_paths),
            e2_submission_path=payloads["e2_submission/catboost_3seed_v1.zip"],
            train_sha256=bindings["train_sha256"],
            history_sha256=bindings["history_sha256"],
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
