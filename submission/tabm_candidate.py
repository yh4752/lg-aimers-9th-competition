"""Verified import of the frozen Version D TabM review artifact."""

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
from typing import Mapping
import zipfile


CANDIDATE_ID = "tabm_hand_matchup_version_d_seed3407_v1"
_INNER_NAME = "tabm_hand_matchup_final_review_bundle.zip"
_MODEL_NAMES = {
    "inference_manifest.json",
    "numeric_embedding_0.json",
    "preprocessing_state.json",
    "tabm_member_0_seed_3407.pt",
}
_OUTER_NAMES = {"delivery_manifest.json", _INNER_NAME, "version_d.log"}
_INNER_NAMES = {
    "final_review.json",
    "frozen_inference/inference_manifest.json",
    "frozen_inference/numeric_embedding_0.json",
    "frozen_inference/preprocessing_state.json",
    "frozen_inference/tabm_member_0_seed_3407.pt",
    "manifest.json",
    "policy/policy.json",
    "policy/policy_review.json",
}
_MAX_EXPANDED_BYTES = 32_000_000_000


class TabMCandidateError(ValueError):
    """Raised when a review delivery cannot become a frozen candidate."""


@dataclass(frozen=True)
class ImportedTabMCandidate:
    candidate_id: str
    root: Path
    model_dir: Path
    delivery_sha256: str
    review_bundle_sha256: str
    model_sha256: str
    member_sha256: Mapping[str, str]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_json(value: bytes, label: str) -> dict[str, object]:
    def unique(items: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, item in items:
            if key in output:
                raise TabMCandidateError(f"duplicate JSON key in {label}: {key}")
            output[key] = item
        return output

    def finite(constant: str) -> object:
        raise TabMCandidateError(f"non-finite JSON value in {label}: {constant}")

    try:
        payload = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=finite,
        )
    except TabMCandidateError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TabMCandidateError(f"invalid JSON: {label}") from error
    if not isinstance(payload, dict):
        raise TabMCandidateError(f"JSON must be an object: {label}")
    return payload


def _safe_archive(archive: zipfile.ZipFile, label: str) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise TabMCandidateError(f"duplicate archive member in {label}")
    if len(names) > 32:
        raise TabMCandidateError(f"too many archive members in {label}")
    total = 0
    output: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if (
            info.is_dir()
            or path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or stat.S_ISLNK(mode)
        ):
            raise TabMCandidateError(f"unsafe archive member in {label}: {info.filename}")
        if info.file_size < 0:
            raise TabMCandidateError(f"invalid archive size in {label}")
        total += info.file_size
        output[info.filename] = info
    if total > _MAX_EXPANDED_BYTES:
        raise TabMCandidateError(f"expanded archive exceeds limit in {label}")
    return output


def _read_archive(value: bytes, label: str) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(value)) as archive:
            infos = _safe_archive(archive, label)
            bad = archive.testzip()
            if bad is not None:
                raise TabMCandidateError(f"corrupt archive member in {label}: {bad}")
            return {name: archive.read(info) for name, info in infos.items()}
    except TabMCandidateError:
        raise
    except zipfile.BadZipFile as error:
        raise TabMCandidateError(f"invalid ZIP: {label}") from error


def _verify_outer(path: Path) -> tuple[bytes, dict[str, object]]:
    try:
        value = path.read_bytes()
    except OSError as error:
        raise TabMCandidateError(f"cannot read delivery: {path}") from error
    members = _read_archive(value, "outer delivery")
    if set(members) != _OUTER_NAMES:
        raise TabMCandidateError("outer delivery member set differs")
    manifest = _load_json(members["delivery_manifest.json"], "delivery_manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_kind") != "tabm_version_D_review_delivery"
        or manifest.get("review_only") is not True
        or manifest.get("submission_package") is not False
    ):
        raise TabMCandidateError("outer delivery contract differs")
    declared = manifest.get("members")
    if not isinstance(declared, dict) or set(declared) != {_INNER_NAME, "version_d.log"}:
        raise TabMCandidateError("outer delivery manifest member set differs")
    for name in (_INNER_NAME, "version_d.log"):
        entry = declared[name]
        if not isinstance(entry, dict) or set(entry) != {"sha256", "size"}:
            raise TabMCandidateError("outer delivery member metadata differs")
        if entry["sha256"] != sha256(members[name]).hexdigest() or entry["size"] != len(members[name]):
            raise TabMCandidateError(f"outer delivery member SHA-256 differs: {name}")
    return members[_INNER_NAME], manifest


def _verify_inner(value: bytes) -> dict[str, bytes]:
    members = _read_archive(value, "review bundle")
    lowered = "\n".join(members).lower()
    if any(token in lowered for token in ("optimizer", "scaler", "rng")):
        raise TabMCandidateError("frozen delivery contains training state")
    if set(members) != _INNER_NAMES:
        raise TabMCandidateError("review bundle member set differs")
    manifest = _load_json(members["manifest.json"], "manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_kind") != "review"
        or manifest.get("version") != "D"
        or manifest.get("review_only") is not True
    ):
        raise TabMCandidateError("review bundle contract differs")
    declared = manifest.get("members")
    if not isinstance(declared, dict) or set(declared) != _INNER_NAMES - {"manifest.json"}:
        raise TabMCandidateError("review bundle manifest member set differs")
    for name, expected in declared.items():
        if not isinstance(expected, str) or sha256(members[name]).hexdigest() != expected:
            raise TabMCandidateError(f"review bundle member SHA-256 differs: {name}")

    review = _load_json(members["final_review.json"], "final_review.json")
    fit = review.get("fit_report")
    if review.get("acceptance_status") != "review_ready":
        raise TabMCandidateError("final review is not review_ready")
    if (
        review.get("version") != "D"
        or review.get("review_only") is not True
        or review.get("fixed_epochs") != 3
        or not isinstance(fit, dict)
        or fit.get("epochs") != 3
        or fit.get("rows") != 1_475_092
        or fit.get("scheduler") != "constant"
        or fit.get("seeds") != [3407]
        or fit.get("status") != "completed"
    ):
        raise TabMCandidateError("final review does not bind three epochs and seed 3407")

    frozen = {
        name.removeprefix("frozen_inference/"): content
        for name, content in members.items()
        if name.startswith("frozen_inference/")
    }
    if set(frozen) != _MODEL_NAMES:
        raise TabMCandidateError("frozen member set differs")
    review_hashes = review.get("artifact_sha256")
    if not isinstance(review_hashes, dict) or set(review_hashes) != _MODEL_NAMES:
        raise TabMCandidateError("final review artifact member set differs")
    for name, content in frozen.items():
        if review_hashes[name] != sha256(content).hexdigest():
            raise TabMCandidateError(f"final review artifact SHA-256 differs: {name}")

    inference = _load_json(frozen["inference_manifest.json"], "inference_manifest.json")
    if (
        inference.get("schema_version") != 1
        or inference.get("epochs") != 3
        or inference.get("seeds") != [3407]
        or inference.get("scheduler") != "constant"
        or inference.get("fit_scope") != "official_train_2019_2024_only"
        or inference.get("row_count") != 1_475_092
        or inference.get("preprocessing_state") != "preprocessing_state.json"
    ):
        raise TabMCandidateError("frozen inference fit scope or identity differs")
    files = inference.get("files")
    expected_files = _MODEL_NAMES - {"inference_manifest.json"}
    if not isinstance(files, dict) or set(files) != expected_files:
        raise TabMCandidateError("frozen inference file manifest differs")
    for name in expected_files:
        if files[name] != sha256(frozen[name]).hexdigest():
            raise TabMCandidateError(f"frozen inference SHA-256 differs: {name}")
    model_members = inference.get("members")
    if not isinstance(model_members, list) or len(model_members) != 1:
        raise TabMCandidateError("frozen inference must contain one model member")
    member = model_members[0]
    if (
        not isinstance(member, dict)
        or member.get("seed") != 3407
        or member.get("weights") != "tabm_member_0_seed_3407.pt"
        or member.get("numeric_state") != "numeric_embedding_0.json"
    ):
        raise TabMCandidateError("frozen inference model member differs")
    return frozen


def _model_digest(members: Mapping[str, bytes]) -> str:
    digest = sha256()
    for name, value in sorted(members.items()):
        digest.update(name.encode("utf-8") + b"\0" + sha256(value).digest())
    return digest.hexdigest()


def import_review_delivery(
    delivery_path: str | Path,
    destination: str | Path,
) -> ImportedTabMCandidate:
    """Verify both ZIP layers and publish a new immutable candidate directory."""

    source = Path(delivery_path).expanduser().resolve(strict=True)
    target = Path(destination).expanduser().absolute()
    if target.exists():
        raise TabMCandidateError(f"destination already exists: {target}")
    inner, _ = _verify_outer(source)
    model_members = _verify_inner(inner)
    member_sha256 = {
        name: sha256(value).hexdigest() for name, value in sorted(model_members.items())
    }
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent)
    )
    try:
        model_dir = temporary / "model"
        model_dir.mkdir()
        for name, value in sorted(model_members.items()):
            with (model_dir / name).open("xb") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
        metadata = {
            "schema_version": 1,
            "artifact_kind": "tabm_submission_validation_candidate",
            "candidate_id": CANDIDATE_ID,
            "delivery_sha256": _file_sha256(source),
            "review_bundle_sha256": sha256(inner).hexdigest(),
            "model_sha256": _model_digest(model_members),
            "members": member_sha256,
        }
        with (temporary / "candidate_manifest.json").open("xb") as handle:
            handle.write(_canonical_json(metadata))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return ImportedTabMCandidate(
        candidate_id=CANDIDATE_ID,
        root=target,
        model_dir=target / "model",
        delivery_sha256=_file_sha256(source),
        review_bundle_sha256=sha256(inner).hexdigest(),
        model_sha256=_model_digest(model_members),
        member_sha256=MappingProxyType(member_sha256),
    )
