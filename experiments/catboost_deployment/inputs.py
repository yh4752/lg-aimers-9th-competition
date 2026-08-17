from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
from types import MappingProxyType
from typing import Sequence
from zipfile import BadZipFile, ZipFile, ZipInfo

from experiments.catboost_tabm_blend.colab import verify_delivery as verify_blend_delivery
from experiments.catboost_tabm_blend.contracts import (
    contract_sha256 as blend_contract_sha256,
    load_contract as load_blend_contract,
)
from experiments.catboost_tabm_blend.inputs import (
    VerifiedStageC,
    VerifiedTrainingInput,
    file_sha256,
    verify_and_extract_stage_c,
    verify_and_extract_training_input,
)

from .contracts import DeploymentContract, contract_sha256


class DeploymentInputError(ValueError):
    """Raised when an uploaded deployment source cannot be trusted."""


@dataclass(frozen=True)
class VerifiedDeploymentInputs:
    training: VerifiedTrainingInput
    stage_c: VerifiedStageC
    source_blend_path: Path
    source_blend_sha256: str
    source_blend_manifest_sha256: str
    source_decision_sha256: str
    source_selected_tabm_weight: float


_TRAINING_MEMBERS = {"input_manifest.json", "train.csv", "trackman_history.csv"}
_STAGE_C_MEMBERS = {
    "delivery_manifest.json",
    "colab_stage_C.log",
    "tabm_search_stage_C_review_bundle.zip",
    "tabm_search_stage_C_resume_bundle.zip",
}
_BLEND_MEMBERS = {
    "blend_campaign.log",
    "catboost_tabm_blend_review.zip",
    "catboost_tabm_blend_resume.zip",
    "delivery_manifest.json",
}
_MAX_MANIFEST = 1024 * 1024
_MAX_MEMBER = 8 * 1024**3
_MAX_TOTAL = 12 * 1024**3
_MAX_RATIO = 200.0


def _safe_info(info: ZipInfo) -> None:
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
        raise DeploymentInputError(f"unsafe ZIP member: {info.filename}")


def _safe_member_inventory(path: Path) -> tuple[set[str], dict[str, object] | None]:
    try:
        with ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise DeploymentInputError("duplicate ZIP member")
            if sum(info.file_size for info in infos) > _MAX_TOTAL:
                raise DeploymentInputError("ZIP total size exceeds limit")
            for info in infos:
                _safe_info(info)
            manifest_name = "manifest.json" if "manifest.json" in names else None
            manifest: dict[str, object] | None = None
            if manifest_name is not None:
                info = archive.getinfo(manifest_name)
                if info.file_size > _MAX_MANIFEST:
                    raise DeploymentInputError("upload manifest exceeds limit")
                parsed = json.loads(archive.read(manifest_name))
                if type(parsed) is not dict:
                    raise DeploymentInputError("upload manifest must be an object")
                manifest = parsed
            return set(names), manifest
    except DeploymentInputError:
        raise
    except (BadZipFile, OSError, KeyError, json.JSONDecodeError) as error:
        raise DeploymentInputError("upload is not a readable ZIP") from error


def _kind_from_exact_members(
    names: set[str], manifest: dict[str, object] | None
) -> str:
    if names == _TRAINING_MEMBERS:
        return "training_input"
    if names == _STAGE_C_MEMBERS:
        return "stage_c_delivery"
    if names == _BLEND_MEMBERS:
        return "blend_delivery"
    if (
        manifest is not None
        and manifest.get("artifact_kind") == "catboost_deployment_resume"
        and {"manifest.json", "contract/contract.json", "state/stage_state.json"}
        <= names
    ):
        return "deployment_resume"
    raise DeploymentInputError("unknown source archive")


def classify_source(path: Path) -> str:
    names, manifest = _safe_member_inventory(Path(path))
    return _kind_from_exact_members(names, manifest)


def _classify_unique_sources(paths: Sequence[Path]) -> dict[str, Path]:
    if len(paths) not in (3, 4):
        raise DeploymentInputError("source count must be three or four")
    grouped: dict[str, Path] = {}
    for source in paths:
        path = Path(source)
        kind = classify_source(path)
        if kind in grouped:
            raise DeploymentInputError(f"duplicate source kind: {kind}")
        grouped[kind] = path
    required = {"training_input", "stage_c_delivery", "blend_delivery"}
    if not required <= set(grouped) or set(grouped) - required - {"deployment_resume"}:
        raise DeploymentInputError("required sources are missing")
    return grouped


def _json_object(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except Exception as error:
        raise DeploymentInputError(f"{label} is unreadable") from error
    if type(value) is not dict:
        raise DeploymentInputError(f"{label} must be an object")
    return value


def _old_blend_bindings(
    training: VerifiedTrainingInput, stage_c: VerifiedStageC, blend_path: Path
) -> dict[str, str]:
    with ZipFile(blend_path) as archive:
        info = archive.getinfo("delivery_manifest.json")
        if info.file_size > _MAX_MANIFEST:
            raise DeploymentInputError("blend delivery manifest exceeds limit")
        manifest = _json_object(archive.read(info), "blend delivery manifest")
    bindings = manifest.get("bindings")
    if type(bindings) is not dict or any(type(value) is not str for value in bindings.values()):
        raise DeploymentInputError("blend delivery bindings differ")
    expected_source_values = {
        "contract_sha256": blend_contract_sha256(),
        "input_manifest_sha256": training.manifest_sha256,
        "train_sha256": training.train_sha256,
        "history_sha256": training.history_sha256,
        "stage_c_delivery_sha256": stage_c.delivery_sha256,
        "stage_c_review_sha256": stage_c.review_sha256,
        "stage_c_state_sha256": stage_c.stage_state_sha256,
    }
    if any(bindings.get(key) != value for key, value in expected_source_values.items()):
        raise DeploymentInputError("blend delivery source binding differs")
    if set(bindings) != {*expected_source_values, "code_sha256"}:
        raise DeploymentInputError("blend delivery binding keys differ")
    code_sha = bindings["code_sha256"]
    if len(code_sha) != 64 or any(character not in "0123456789abcdef" for character in code_sha):
        raise DeploymentInputError("blend delivery code binding differs")
    return dict(bindings)


def _promoted_decision(blend_path: Path) -> tuple[bytes, dict[str, object], bytes]:
    try:
        with ZipFile(blend_path) as outer:
            manifest_bytes = outer.read("delivery_manifest.json")
            review_bytes = outer.read("catboost_tabm_blend_review.zip")
        with ZipFile(BytesIO(review_bytes)) as review:
            info = review.getinfo("decision/blend_decision.json")
            if info.file_size > _MAX_MANIFEST:
                raise DeploymentInputError("blend decision exceeds limit")
            decision_bytes = review.read(info)
    except DeploymentInputError:
        raise
    except (BadZipFile, KeyError, OSError) as error:
        raise DeploymentInputError("blend decision is missing") from error
    decision = _json_object(decision_bytes, "blend decision")
    candidates = decision.get("candidates")
    if (
        decision.get("selected_tabm_weight") != 0.7
        or decision.get("reason") != "fixed_blend_passed"
        or type(candidates) is not list
        or not any(
            type(item) is dict
            and item.get("tabm_weight") == 0.7
            and item.get("passed") is True
            and set(item.get("fold_brier", {})) == {"2022->2023", "2023->2024"}
            for item in candidates
        )
    ):
        raise DeploymentInputError("source blend was not promoted at weight 0.7")
    return decision_bytes, decision, manifest_bytes


def deployment_bindings(
    verified: VerifiedDeploymentInputs,
    *,
    expected_code_sha256: str,
) -> dict[str, str]:
    return {
        "contract_sha256": contract_sha256(),
        "code_sha256": expected_code_sha256,
        "input_manifest_sha256": verified.training.manifest_sha256,
        "train_sha256": verified.training.train_sha256,
        "history_sha256": verified.training.history_sha256,
        "stage_c_delivery_sha256": verified.stage_c.delivery_sha256,
        "stage_c_review_sha256": verified.stage_c.review_sha256,
        "stage_c_state_sha256": verified.stage_c.stage_state_sha256,
        "source_blend_delivery_sha256": verified.source_blend_sha256,
        "source_blend_manifest_sha256": verified.source_blend_manifest_sha256,
        "source_decision_sha256": verified.source_decision_sha256,
    }


def classify_and_verify_sources(
    paths: Sequence[Path],
    *,
    run_root: Path,
    contract: DeploymentContract,
    expected_code_sha256: str,
) -> tuple[VerifiedDeploymentInputs, Path | None]:
    grouped = _classify_unique_sources(paths)
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=False)
    old_contract = load_blend_contract()
    training = verify_and_extract_training_input(
        grouped["training_input"], run_root / "official_data", old_contract
    )
    stage_source = grouped["stage_c_delivery"]
    if file_sha256(stage_source) != contract.source_stage_c_delivery_sha256:
        raise DeploymentInputError("Stage C source SHA-256 differs")
    stage_c = verify_and_extract_stage_c(stage_source, run_root / "stage_c", old_contract)

    blend_source = grouped["blend_delivery"]
    observed_blend_sha = file_sha256(blend_source)
    if observed_blend_sha != contract.source_blend_delivery_sha256:
        raise DeploymentInputError("blend source SHA-256 differs")
    old_bindings = _old_blend_bindings(training, stage_c, blend_source)
    try:
        verify_blend_delivery(blend_source, expected_bindings=old_bindings)
    except Exception as error:
        raise DeploymentInputError(f"blend delivery is invalid: {error}") from error
    decision_bytes, _, manifest_bytes = _promoted_decision(blend_source)
    copied_blend = run_root / "source_blend_delivery.zip"
    shutil.copyfile(blend_source, copied_blend)
    if file_sha256(copied_blend) != observed_blend_sha:
        raise DeploymentInputError("blend source changed while copying")
    verified = VerifiedDeploymentInputs(
        training=training,
        stage_c=stage_c,
        source_blend_path=copied_blend,
        source_blend_sha256=observed_blend_sha,
        source_blend_manifest_sha256=sha256(manifest_bytes).hexdigest(),
        source_decision_sha256=sha256(decision_bytes).hexdigest(),
        source_selected_tabm_weight=0.7,
    )

    resume_source = grouped.get("deployment_resume")
    resume: Path | None = None
    if resume_source is not None:
        try:
            from .artifacts import verify_deployment_resume
        except ImportError as error:
            raise DeploymentInputError("deployment resume verifier is unavailable") from error
        bindings = deployment_bindings(verified, expected_code_sha256=expected_code_sha256)
        verify_deployment_resume(resume_source, expected_bindings=bindings)
        resume = run_root / "uploaded_deployment_resume.zip"
        shutil.copyfile(resume_source, resume)
    return verified, resume
