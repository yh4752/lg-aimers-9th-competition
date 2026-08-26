from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from types import MappingProxyType
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd

from experiments.tabm_campaign.artifacts import verify_review_bundle
from experiments.tabm_campaign.colab_recovery import verify_delivery_bundle

from .artifacts import verify_e1_handoff
from .e2_contracts import E2Contract, load_e2_contract
from .inputs import PREDICTION_COLUMNS, VerifiedOfficialData


class E2InputError(ValueError):
    """Raised when E2 source or compact input evidence is not trustworthy."""


_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)
_MAX_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
_MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
_MAX_RATIO = 250.0
_E1_REVIEW = "tree_expert_e1_review.zip"
_E1_RESUME = "tree_expert_e1_resume.zip"
_STAGE_C_REVIEW = "tabm_search_stage_C_review_bundle.zip"
_E1_DECISION_MEMBER = "campaign/decision.json"
_E1_PREDICTION_MEMBERS = {
    "c1_anchor_residual": (
        "jobs/e1__c1_anchor_residual__tr2023__va2024__s3407/predictions.csv"
    ),
    "c2_trackman_residual": (
        "jobs/e1__c2_trackman_residual__tr2023__va2024__s3407/predictions.csv"
    ),
}
_STAGE_C_PREDICTION_MEMBERS = {
    "2022->2023": (
        "predictions/c_final__a__p2__piecewise_linear__bce__plateau__s42"
        "__s3407__tr2022__va2023.csv"
    ),
    "2023->2024": (
        "predictions/c_final__a__p2__piecewise_linear__bce__plateau__s42"
        "__s3407__tr2023__va2024.csv"
    ),
}
_TABM_MEMBERS = (
    "script.py",
    "requirements.txt",
    "model/inference_manifest.json",
    "model/numeric_embedding_0.json",
    "model/preprocessing_state.json",
    "model/tabm_member_0_seed_3407.pt",
)
E2_INPUT_MEMBERS = (
    "manifest.json",
    "e1/decision.json",
    "e1/c1_f3_predictions.csv",
    "e1/c2_f3_predictions.csv",
    "stage_c/tabm_f2_predictions.csv",
    "stage_c/tabm_f3_predictions.csv",
    *(f"tabm/{name}" for name in _TABM_MEMBERS),
)
_E1_COMPACT_PREDICTIONS = {
    "c1_anchor_residual": "e1/c1_f3_predictions.csv",
    "c2_trackman_residual": "e1/c2_f3_predictions.csv",
}
_TABM_COMPACT_PREDICTIONS = {
    "2022->2023": "stage_c/tabm_f2_predictions.csv",
    "2023->2024": "stage_c/tabm_f3_predictions.csv",
}


@dataclass(frozen=True)
class VerifiedE2Input:
    root: Path
    archive_sha256: str
    manifest_sha256: str
    e1_predictions: Mapping[str, Path]
    tabm_predictions: Mapping[str, Path]
    tabm_runtime_root: Path
    lineage: Mapping[str, str]


def file_sha256(path: Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise E2InputError(f"source is not a regular file: {source}")
    digest = sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise E2InputError(f"cannot encode E2 input JSON: {error}") from error


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _write_archive(path: Path, members: Mapping[str, bytes]) -> Path:
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
                archive.writestr(_zip_info(name), payload)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _safe_names(archive: ZipFile, expected: set[str] | None = None) -> list[str]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise E2InputError("archive has duplicate members")
    if expected is not None and set(names) != expected:
        raise E2InputError("archive member names differ")
    total = 0
    for info in infos:
        pure = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        ratio = info.file_size / max(1, info.compress_size)
        if (
            pure.is_absolute()
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
            or info.is_dir()
            or stat.S_ISLNK(mode)
            or info.file_size > _MAX_MEMBER_BYTES
            or (info.file_size >= 64 * 1024 and ratio > _MAX_RATIO)
        ):
            raise E2InputError(f"unsafe archive member: {info.filename}")
        total += info.file_size
    if total > _MAX_TOTAL_BYTES:
        raise E2InputError("archive uncompressed size is too large")
    return names


def _prediction(payload: bytes, label: str) -> tuple[pd.DataFrame, str]:
    try:
        frame = pd.read_csv(BytesIO(payload))
    except Exception as error:
        raise E2InputError(f"{label} prediction is unreadable: {error}") from error
    if tuple(frame.columns) != PREDICTION_COLUMNS:
        raise E2InputError(f"{label} prediction columns differ")
    if frame.empty or frame["row_id"].isna().any() or not frame["row_id"].is_unique:
        raise E2InputError(f"{label} prediction row_id values differ")
    target = pd.to_numeric(frame["target"], errors="coerce")
    probability = pd.to_numeric(frame["probability"], errors="coerce")
    if (
        not target.isin([0, 1]).all()
        or probability.isna().any()
        or not probability.between(0, 1).all()
        or frame.loc[:, PREDICTION_COLUMNS[3:]].isna().any().any()
    ):
        raise E2InputError(f"{label} prediction values differ")
    row_target = frame.loc[:, ["row_id", "target"]].to_csv(index=False).encode("utf-8")
    return frame, sha256(row_target).hexdigest()


def _decision(payload: bytes) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except Exception as error:
        raise E2InputError(f"E1 decision is unreadable: {error}") from error
    if (
        type(value) is not dict
        or value.get("status") != "completed"
        or value.get("promoted")
        != ["c1_anchor_residual", "c2_trackman_residual"]
    ):
        raise E2InputError("E1 decision differs")
    return value


def _verified_e1_archive(payload: bytes, *, kind: str) -> ZipFile:
    try:
        archive = ZipFile(BytesIO(payload), "r")
        names = _safe_names(archive)
        if "manifest.json" not in names:
            raise E2InputError("E1 nested archive has no manifest")
        manifest = json.loads(archive.read("manifest.json"))
        if (
            type(manifest) is not dict
            or manifest.get("artifact_kind") != kind
            or manifest.get("submission_package") is not False
        ):
            raise E2InputError("E1 nested archive identity differs")
        members = manifest.get("members")
        if type(members) is not dict or set(members) != set(names) - {"manifest.json"}:
            raise E2InputError("E1 nested member manifest differs")
        for name, evidence in members.items():
            member = archive.read(name)
            if (
                type(evidence) is not dict
                or evidence.get("size") != len(member)
                or evidence.get("sha256") != sha256(member).hexdigest()
            ):
                raise E2InputError(f"E1 nested member differs: {name}")
        return archive
    except E2InputError:
        raise
    except Exception as error:
        raise E2InputError(f"cannot verify E1 nested archive: {error}") from error


def _stage_c_review(payload: bytes) -> ZipFile:
    with tempfile.TemporaryDirectory(prefix="tree-e2-stage-c-") as temporary:
        review_path = Path(temporary) / "review.zip"
        review_path.write_bytes(payload)
        try:
            verified = verify_review_bundle(review_path)
        except Exception as error:
            raise E2InputError(f"cannot verify Stage C review: {error}") from error
        if verified.version != "C":
            raise E2InputError("Stage C review version differs")
    return ZipFile(BytesIO(payload), "r")


def _tabm_payloads(path: Path, contract: E2Contract) -> dict[str, bytes]:
    if file_sha256(path) != contract.input_hashes["tabm_submission_sha256"]:
        raise E2InputError("TabM submission SHA-256 differs")
    try:
        with ZipFile(path, "r") as archive:
            _safe_names(archive, set(_TABM_MEMBERS))
            payloads = {name: archive.read(name) for name in _TABM_MEMBERS}
    except E2InputError:
        raise
    except Exception as error:
        raise E2InputError(f"cannot verify TabM submission: {error}") from error
    if (
        sha256(payloads["model/tabm_member_0_seed_3407.pt"]).hexdigest()
        != contract.input_hashes["tabm_weight_sha256"]
    ):
        raise E2InputError("TabM weight SHA-256 differs")
    try:
        manifest = json.loads(payloads["model/inference_manifest.json"])
    except Exception as error:
        raise E2InputError(f"TabM inference manifest is unreadable: {error}") from error
    if (
        manifest.get("fit_scope") != "official_train_2019_2024_only"
        or manifest.get("identity", {}).get("train_sha256")
        != contract.input_hashes["official_train_sha256"]
        or manifest.get("files", {}).get("tabm_member_0_seed_3407.pt")
        != contract.input_hashes["tabm_weight_sha256"]
    ):
        raise E2InputError("TabM inference identity differs")
    for name in (
        "numeric_embedding_0.json",
        "preprocessing_state.json",
        "tabm_member_0_seed_3407.pt",
    ):
        if manifest["files"].get(name) != sha256(payloads[f"model/{name}"]).hexdigest():
            raise E2InputError(f"TabM inference member differs: {name}")
    return {f"tabm/{name}": payload for name, payload in payloads.items()}


def prepare_e2_input(
    e1_handoff: Path,
    stage_c_delivery: Path,
    tabm_submission: Path,
    output: Path,
    *,
    contract: E2Contract | None = None,
) -> Path:
    active = load_e2_contract() if contract is None else contract
    if file_sha256(e1_handoff) != active.input_hashes["e1_handoff_sha256"]:
        raise E2InputError("E1 handoff SHA-256 differs")
    try:
        verify_e1_handoff(e1_handoff)
        with ZipFile(e1_handoff, "r") as outer:
            review_bytes = outer.read(_E1_REVIEW)
            resume_bytes = outer.read(_E1_RESUME)
    except E2InputError:
        raise
    except Exception as error:
        raise E2InputError(f"cannot verify E1 handoff: {error}") from error
    if sha256(review_bytes).hexdigest() != active.input_hashes["e1_review_sha256"]:
        raise E2InputError("E1 review SHA-256 differs")
    if sha256(resume_bytes).hexdigest() != active.input_hashes["e1_resume_sha256"]:
        raise E2InputError("E1 resume SHA-256 differs")
    review = _verified_e1_archive(review_bytes, kind="tree_expert_e1_review_v1")
    resume = _verified_e1_archive(resume_bytes, kind="tree_expert_e1_resume_v1")
    resume.close()
    decision = review.read(_E1_DECISION_MEMBER)
    _decision(decision)
    payloads: dict[str, bytes] = {"e1/decision.json": decision}
    for candidate_id, source_name in _E1_PREDICTION_MEMBERS.items():
        payloads[_E1_COMPACT_PREDICTIONS[candidate_id]] = review.read(source_name)
    review.close()

    if file_sha256(stage_c_delivery) != active.input_hashes["stage_c_delivery_sha256"]:
        raise E2InputError("Stage C delivery SHA-256 differs")
    try:
        verified_delivery = verify_delivery_bundle(stage_c_delivery)
        if (
            verified_delivery.member_sha256.get(_STAGE_C_REVIEW)
            != active.input_hashes["stage_c_review_sha256"]
        ):
            raise E2InputError("Stage C review SHA-256 differs")
        with ZipFile(stage_c_delivery, "r") as outer:
            stage_c_bytes = outer.read(_STAGE_C_REVIEW)
    except E2InputError:
        raise
    except Exception as error:
        raise E2InputError(f"cannot verify Stage C delivery: {error}") from error
    stage_c = _stage_c_review(stage_c_bytes)
    state = json.loads(stage_c.read("stage_state.json"))
    if state.get("stage_complete") is not True:
        raise E2InputError("Stage C state is incomplete")
    for fold, source_name in _STAGE_C_PREDICTION_MEMBERS.items():
        payloads[_TABM_COMPACT_PREDICTIONS[fold]] = stage_c.read(source_name)
    stage_c.close()
    payloads.update(_tabm_payloads(Path(tabm_submission), active))

    prediction_bindings: dict[str, dict[str, str]] = {}
    for name in (*_E1_COMPACT_PREDICTIONS.values(), *_TABM_COMPACT_PREDICTIONS.values()):
        _, row_target_sha = _prediction(payloads[name], name)
        prediction_bindings[name] = {"row_target_sha256": row_target_sha}
    f3_hashes = {
        prediction_bindings[name]["row_target_sha256"]
        for name in (
            _E1_COMPACT_PREDICTIONS["c1_anchor_residual"],
            _E1_COMPACT_PREDICTIONS["c2_trackman_residual"],
            _TABM_COMPACT_PREDICTIONS["2023->2024"],
        )
    }
    if len(f3_hashes) != 1:
        raise E2InputError("F3 prediction alignment differs")
    manifest = {
        "schema_version": 1,
        "artifact_kind": "tree_expert_e2_input_v1",
        "review_only": False,
        "submission_package": False,
        "campaign_id": active.campaign_id,
        "lineage": dict(active.input_hashes),
        "prediction_bindings": prediction_bindings,
        "members": {
            name: {"size": len(value), "sha256": sha256(value).hexdigest()}
            for name, value in sorted(payloads.items())
        },
    }
    destination = _write_archive(
        Path(output),
        {"manifest.json": _canonical_json(manifest), **payloads},
    )
    with tempfile.TemporaryDirectory(prefix="tree-e2-input-check-") as temporary:
        verify_and_extract_e2_input(destination, Path(temporary), contract=active)
    return destination


def verify_and_extract_e2_input(
    archive_path: Path,
    destination: Path,
    *,
    contract: E2Contract | None = None,
) -> VerifiedE2Input:
    active = load_e2_contract() if contract is None else contract
    source = Path(archive_path)
    archive_sha = file_sha256(source)
    try:
        with ZipFile(source, "r") as archive:
            _safe_names(archive, set(E2_INPUT_MEMBERS))
            manifest_bytes = archive.read("manifest.json")
            manifest = json.loads(manifest_bytes)
            payloads = {
                name: archive.read(name)
                for name in E2_INPUT_MEMBERS
                if name != "manifest.json"
            }
    except E2InputError:
        raise
    except Exception as error:
        raise E2InputError(f"cannot read compact E2 input: {error}") from error
    expected_manifest_keys = {
        "schema_version",
        "artifact_kind",
        "review_only",
        "submission_package",
        "campaign_id",
        "lineage",
        "prediction_bindings",
        "members",
    }
    if type(manifest) is not dict or set(manifest) != expected_manifest_keys:
        raise E2InputError("E2 input manifest keys differ")
    if (
        manifest["schema_version"] != 1
        or manifest["artifact_kind"] != "tree_expert_e2_input_v1"
        or manifest["review_only"] is not False
        or manifest["submission_package"] is not False
        or manifest["campaign_id"] != active.campaign_id
        or manifest["lineage"] != dict(active.input_hashes)
    ):
        raise E2InputError("E2 input identity or lineage differs")
    members = manifest["members"]
    if type(members) is not dict or set(members) != set(payloads):
        raise E2InputError("E2 input member manifest differs")
    for name, payload in payloads.items():
        evidence = members[name]
        if (
            type(evidence) is not dict
            or set(evidence) != {"size", "sha256"}
            or evidence["size"] != len(payload)
            or evidence["sha256"] != sha256(payload).hexdigest()
        ):
            raise E2InputError(f"E2 input member differs: {name}")
    _decision(payloads["e1/decision.json"])
    bindings = manifest["prediction_bindings"]
    prediction_names = {
        *_E1_COMPACT_PREDICTIONS.values(),
        *_TABM_COMPACT_PREDICTIONS.values(),
    }
    if type(bindings) is not dict or set(bindings) != prediction_names:
        raise E2InputError("prediction bindings differ")
    row_hashes: dict[str, str] = {}
    for name in prediction_names:
        _, row_hash = _prediction(payloads[name], name)
        binding = bindings[name]
        if (
            type(binding) is not dict
            or set(binding) != {"row_target_sha256"}
            or binding["row_target_sha256"] != row_hash
        ):
            raise E2InputError("fold prediction alignment differs")
        row_hashes[name] = row_hash
    if len(
        {
            row_hashes[_E1_COMPACT_PREDICTIONS["c1_anchor_residual"]],
            row_hashes[_E1_COMPACT_PREDICTIONS["c2_trackman_residual"]],
            row_hashes[_TABM_COMPACT_PREDICTIONS["2023->2024"]],
        }
    ) != 1:
        raise E2InputError("F3 prediction alignment differs")
    if (
        sha256(payloads["tabm/model/tabm_member_0_seed_3407.pt"]).hexdigest()
        != active.input_hashes["tabm_weight_sha256"]
    ):
        raise E2InputError("TabM weight SHA-256 differs")

    root = Path(destination)
    if root.exists() and (root.is_symlink() or any(root.iterdir())):
        raise E2InputError("E2 input destination is not empty")
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        output = root.joinpath(*PurePosixPath(name).parts)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
    return VerifiedE2Input(
        root=root,
        archive_sha256=archive_sha,
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
        e1_predictions=MappingProxyType(
            {
                key: root / name
                for key, name in _E1_COMPACT_PREDICTIONS.items()
            }
        ),
        tabm_predictions=MappingProxyType(
            {key: root / name for key, name in _TABM_COMPACT_PREDICTIONS.items()}
        ),
        tabm_runtime_root=root / "tabm",
        lineage=MappingProxyType(dict(active.input_hashes)),
    )


def verify_official_data(
    root: Path,
    contract: E2Contract | None = None,
    *,
    testing: bool = False,
) -> VerifiedOfficialData:
    active = load_e2_contract() if contract is None else contract
    source = Path(root)
    if source.is_symlink() or not source.is_dir():
        raise E2InputError(f"official data root is invalid: {source}")
    resolved = source.resolve()
    if testing and not resolved.is_relative_to(Path(tempfile.gettempdir()).resolve()):
        raise E2InputError("testing hash bypass requires temporary data")
    paths = {name: resolved / name for name in ("train.csv", "trackman_history.csv")}
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise E2InputError("official data files must be regular files at the root")
    train_sha = file_sha256(paths["train.csv"])
    history_sha = file_sha256(paths["trackman_history.csv"])
    if not testing and (
        train_sha != active.input_hashes["official_train_sha256"]
        or history_sha != active.input_hashes["official_history_sha256"]
    ):
        raise E2InputError("official data SHA-256 differs")
    return VerifiedOfficialData(
        root=resolved,
        train=paths["train.csv"],
        history=paths["trackman_history.csv"],
        train_sha256=train_sha,
        history_sha256=history_sha,
    )
