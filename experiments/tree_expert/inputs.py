from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd

from experiments.tabm_campaign.artifacts import verify_review_bundle
from experiments.tabm_campaign.colab_recovery import verify_delivery_bundle

from .contracts import E1Contract


class TreeExpertInputError(ValueError):
    """Raised when E1 input evidence cannot be trusted."""


PREDICTION_COLUMNS = (
    "row_id",
    "target",
    "probability",
    "game_type",
    "game_month",
    "pitcher_id_known",
    "batter_id_known",
)
_PREDICTION_MEMBER = (
    "predictions/"
    "c_final__a__p2__piecewise_linear__bce__plateau__s42__s3407"
    "__tr2023__va2024.csv"
)
_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)
_INPUT_NAMES = {"manifest.json", "baseline_predictions.csv"}
_MAX_MEMBER_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class VerifiedOfficialData:
    root: Path
    train: Path
    history: Path
    train_sha256: str
    history_sha256: str


@dataclass(frozen=True)
class VerifiedE1Input:
    archive_sha256: str
    manifest_sha256: str
    stage_c_delivery_sha256: str
    stage_c_review_sha256: str
    baseline_fold: str
    baseline_predictions: Path


def file_sha256(path: Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise TreeExpertInputError(f"source is not a regular file: {source}")
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
        raise TreeExpertInputError(f"cannot serialize input manifest: {error}") from error


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _write_archive(path: Path, members: Mapping[str, bytes]) -> None:
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


def _safe_input_names(archive: ZipFile) -> list[str]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)) or set(names) != _INPUT_NAMES:
        raise TreeExpertInputError("E1 input member names differ")
    for info in infos:
        pure = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if (
            pure.is_absolute()
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
            or stat.S_ISLNK(mode)
            or info.is_dir()
            or info.file_size > _MAX_MEMBER_BYTES
        ):
            raise TreeExpertInputError(f"unsafe E1 input member: {info.filename}")
    return names


def _validate_predictions(payload: bytes) -> None:
    try:
        frame = pd.read_csv(__import__("io").BytesIO(payload))
    except Exception as error:
        raise TreeExpertInputError(f"baseline predictions are unreadable: {error}") from error
    if tuple(frame.columns) != PREDICTION_COLUMNS:
        raise TreeExpertInputError("baseline prediction columns differ")
    if frame.empty or frame["row_id"].isna().any() or not frame["row_id"].is_unique:
        raise TreeExpertInputError("baseline row_id values are invalid")
    target = pd.to_numeric(frame["target"], errors="coerce")
    probability = pd.to_numeric(frame["probability"], errors="coerce")
    if not target.isin([0, 1]).all() or probability.isna().any() or not probability.between(0, 1).all():
        raise TreeExpertInputError("baseline target or probability values are invalid")
    if frame.loc[:, PREDICTION_COLUMNS[3:]].isna().any().any():
        raise TreeExpertInputError("baseline diagnostic values are missing")


def _review_member(delivery: Path, contract: E1Contract) -> tuple[bytes, str]:
    if file_sha256(delivery) != contract.stage_c_delivery_sha256:
        raise TreeExpertInputError("Stage C delivery SHA-256 differs")
    try:
        verified_delivery = verify_delivery_bundle(delivery)
        review_sha = verified_delivery.member_sha256[
            "tabm_search_stage_C_review_bundle.zip"
        ]
        if review_sha != contract.stage_c_review_sha256:
            raise TreeExpertInputError("Stage C review SHA-256 differs")
        with ZipFile(delivery, "r") as outer:
            review_bytes = outer.read("tabm_search_stage_C_review_bundle.zip")
    except TreeExpertInputError:
        raise
    except Exception as error:
        raise TreeExpertInputError(f"cannot verify Stage C delivery: {error}") from error

    with tempfile.TemporaryDirectory(prefix="tree-e1-review-") as temporary:
        review_path = Path(temporary) / "review.zip"
        review_path.write_bytes(review_bytes)
        try:
            verified_review = verify_review_bundle(review_path)
            if verified_review.version != "C":
                raise TreeExpertInputError("Stage C review version differs")
            if verified_review.member_sha256.get(_PREDICTION_MEMBER) is None:
                raise TreeExpertInputError("bound F3 prediction member is absent")
            with ZipFile(review_path, "r") as review:
                state = json.loads(review.read("stage_state.json"))
                predictions = review.read(_PREDICTION_MEMBER)
        except TreeExpertInputError:
            raise
        except Exception as error:
            raise TreeExpertInputError(f"cannot verify Stage C review: {error}") from error

    if state.get("stage_complete") is not True:
        raise TreeExpertInputError("Stage C state is incomplete")
    predictor_evidence = state.get("predictor_evidence")
    matched_predictor = [
        item
        for item in predictor_evidence
        if isinstance(item, dict)
        and item.get("predictor_id") == contract.baseline_predictor
        and item.get("accepted") is True
    ] if isinstance(predictor_evidence, list) else []
    if len(matched_predictor) != 1:
        raise TreeExpertInputError("bound baseline predictor evidence differs")
    final_members = state.get("final_members")
    matched_member = [
        item
        for item in final_members
        if isinstance(item, dict)
        and item.get("seed") == 3407
        and item.get("status") == "completed"
        and item.get("predictions") == Path(_PREDICTION_MEMBER).name
    ] if isinstance(final_members, list) else []
    if len(matched_member) != 1:
        raise TreeExpertInputError("bound F3 final member differs")
    _validate_predictions(predictions)
    return predictions, verified_review.manifest_sha256


def prepare_e1_input(
    stage_c_delivery: Path,
    output: Path,
    contract: E1Contract,
) -> Path:
    delivery = Path(stage_c_delivery)
    predictions, review_manifest_sha = _review_member(delivery, contract)
    prediction_sha = sha256(predictions).hexdigest()
    manifest = {
        "schema_version": 1,
        "artifact_kind": "tree_expert_e1_input_v1",
        "review_only": True,
        "submission_package": False,
        "stage_c_delivery_sha256": contract.stage_c_delivery_sha256,
        "stage_c_review_sha256": contract.stage_c_review_sha256,
        "stage_c_review_manifest_sha256": review_manifest_sha,
        "baseline_predictor": contract.baseline_predictor,
        "baseline_fold": "2023->2024",
        "original_member": _PREDICTION_MEMBER,
        "members": {
            "baseline_predictions.csv": {
                "size": len(predictions),
                "sha256": prediction_sha,
            }
        },
    }
    destination = Path(output)
    _write_archive(
        destination,
        {
            "manifest.json": _canonical_json(manifest),
            "baseline_predictions.csv": predictions,
        },
    )
    verify_and_extract_e1_input(
        destination,
        destination.parent / f".{destination.name}.verify",
        contract,
    )
    __import__("shutil").rmtree(
        destination.parent / f".{destination.name}.verify", ignore_errors=True
    )
    return destination


def verify_and_extract_e1_input(
    archive_path: Path,
    destination: Path,
    contract: E1Contract,
) -> VerifiedE1Input:
    source = Path(archive_path)
    archive_sha = file_sha256(source)
    try:
        with ZipFile(source, "r") as archive:
            _safe_input_names(archive)
            manifest_bytes = archive.read("manifest.json")
            manifest = json.loads(manifest_bytes)
            predictions = archive.read("baseline_predictions.csv")
    except TreeExpertInputError:
        raise
    except Exception as error:
        raise TreeExpertInputError(f"cannot read E1 input: {error}") from error

    expected_keys = {
        "schema_version",
        "artifact_kind",
        "review_only",
        "submission_package",
        "stage_c_delivery_sha256",
        "stage_c_review_sha256",
        "stage_c_review_manifest_sha256",
        "baseline_predictor",
        "baseline_fold",
        "original_member",
        "members",
    }
    if type(manifest) is not dict or set(manifest) != expected_keys:
        raise TreeExpertInputError("E1 input manifest keys differ")
    if (
        manifest["schema_version"] != 1
        or manifest["artifact_kind"] != "tree_expert_e1_input_v1"
        or manifest["review_only"] is not True
        or manifest["submission_package"] is not False
        or manifest["stage_c_delivery_sha256"] != contract.stage_c_delivery_sha256
        or manifest["stage_c_review_sha256"] != contract.stage_c_review_sha256
        or manifest["baseline_predictor"] != contract.baseline_predictor
        or manifest["baseline_fold"] != "2023->2024"
        or manifest["original_member"] != _PREDICTION_MEMBER
    ):
        raise TreeExpertInputError("E1 input identity differs")
    members = manifest["members"]
    evidence = members.get("baseline_predictions.csv") if type(members) is dict else None
    if (
        type(evidence) is not dict
        or set(evidence) != {"size", "sha256"}
        or evidence["size"] != len(predictions)
        or evidence["sha256"] != sha256(predictions).hexdigest()
    ):
        raise TreeExpertInputError("baseline prediction member differs")
    review_manifest_sha = manifest["stage_c_review_manifest_sha256"]
    if type(review_manifest_sha) is not str or len(review_manifest_sha) != 64:
        raise TreeExpertInputError("Stage C review manifest SHA-256 is invalid")
    _validate_predictions(predictions)

    root = Path(destination)
    if root.exists() and any(root.iterdir()):
        raise TreeExpertInputError("E1 input destination is not empty")
    root.mkdir(parents=True, exist_ok=True)
    output = root / "baseline_predictions.csv"
    output.write_bytes(predictions)
    return VerifiedE1Input(
        archive_sha256=archive_sha,
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
        stage_c_delivery_sha256=contract.stage_c_delivery_sha256,
        stage_c_review_sha256=contract.stage_c_review_sha256,
        baseline_fold="2023->2024",
        baseline_predictions=output,
    )


def verify_official_data(
    root: Path,
    contract: E1Contract,
    *,
    testing: bool = False,
) -> VerifiedOfficialData:
    source = Path(root)
    if source.is_symlink() or not source.is_dir():
        raise TreeExpertInputError(f"official data root is invalid: {source}")
    resolved = source.resolve()
    if testing:
        temporary_root = Path(tempfile.gettempdir()).resolve()
        if not resolved.is_relative_to(temporary_root):
            raise TreeExpertInputError("testing hash bypass requires temporary data")

    found: dict[str, list[Path]] = {"train.csv": [], "trackman_history.csv": []}
    for directory, directory_names, file_names in os.walk(resolved, followlinks=False):
        current = Path(directory)
        for name in tuple(directory_names):
            if (current / name).is_symlink():
                raise TreeExpertInputError("official data contains a symlink directory")
        for name in found:
            if name in file_names:
                candidate = current / name
                if candidate.is_symlink() or not candidate.is_file():
                    raise TreeExpertInputError(f"official {name} is not a regular file")
                found[name].append(candidate)
    for name, candidates in found.items():
        if len(candidates) != 1 or candidates[0].parent != resolved:
            raise TreeExpertInputError(
                f"official {name} must occur exactly once at the data root"
            )

    train = found["train.csv"][0]
    history = found["trackman_history.csv"][0]
    train_sha = file_sha256(train)
    history_sha = file_sha256(history)
    if not testing and (
        train_sha != contract.official_train_sha256
        or history_sha != contract.official_history_sha256
    ):
        raise TreeExpertInputError("official data SHA-256 differs")
    return VerifiedOfficialData(resolved, train, history, train_sha, history_sha)
