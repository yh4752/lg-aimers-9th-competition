from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from types import MappingProxyType
from typing import Callable, Mapping
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

from experiments.catboost_tabm_blend.contracts import (
    load_contract as load_blend_contract,
)
from experiments.catboost_tabm_blend.inputs import (
    verify_and_extract_stage_c,
    verify_and_extract_training_input,
)
from experiments.catboost_deployment.artifacts import (
    verify_deployment_resume,
    verify_deployment_review,
)

from .contracts import RealignContract, contract_sha256


class RealignInputError(ValueError):
    """Raised when a realignment handoff cannot be trusted."""


@dataclass(frozen=True)
class RealignSourcePaths:
    training_input: Path
    stage_c_delivery: Path
    deployment_resume: Path
    deployment_review: Path
    oof_audit: Path


@dataclass(frozen=True)
class TrustedSource:
    path: Path
    sha256: str
    size: int


@dataclass(frozen=True)
class VerifiedSources:
    members: Mapping[str, TrustedSource]
    source_sha256: Mapping[str, str]
    fold_keys: tuple[str, ...]
    audit_weight: Decimal
    cleanup_root: Path | None = None


@dataclass(frozen=True)
class PreparedRealignInput:
    path: Path
    sha256: str


@dataclass(frozen=True)
class VerifiedRealignInput:
    root: Path
    data_dir: Path
    tabm_predictions: Mapping[str, Path]
    catboost_models: Mapping[str, Path]
    catboost_states: Mapping[str, Path]
    catboost_predictions: Mapping[str, Path]
    audit_decision: Path
    manifest_sha256: str
    fold_keys: tuple[str, ...]
    audit_weight: Decimal


_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
_MAX_MEMBER = 8 * 1024 * 1024 * 1024
_MAX_TOTAL = 12 * 1024 * 1024 * 1024
_MAX_RATIO = 200.0
_EXPECTED_MEMBER_NAMES = {
    "data/train.csv",
    "data/trackman_history.csv",
    "tabm/2022_2023.csv",
    "tabm/2023_2024.csv",
    "catboost/2022_2023/model.cbm",
    "catboost/2022_2023/preprocessing_state.json",
    "catboost/2022_2023/predictions.csv",
    "catboost/2023_2024/model.cbm",
    "catboost/2023_2024/preprocessing_state.json",
    "catboost/2023_2024/predictions.csv",
    "audit/next_experiment.json",
    "audit/artifact_inventory.json",
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=_ZIP_TIME)
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o600) << 16
    return info


def _safe_info(info: ZipInfo) -> None:
    path = PurePosixPath(info.filename)
    mode = info.external_attr >> 16
    if (
        not info.filename
        or "\\" in info.filename
        or path.is_absolute()
        or ".." in path.parts
        or info.is_dir()
        or stat.S_IFMT(mode) == stat.S_IFLNK
        or info.file_size > _MAX_MEMBER
        or (
            info.file_size >= 64 * 1024
            and info.file_size / max(1, info.compress_size) > _MAX_RATIO
        )
    ):
        raise RealignInputError(f"unsafe ZIP member: {info.filename}")


def _checked_infos(archive: ZipFile) -> dict[str, ZipInfo]:
    infos = archive.infolist()
    for info in infos:
        _safe_info(info)
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise RealignInputError("duplicate ZIP member")
    if sum(info.file_size for info in infos) > _MAX_TOTAL:
        raise RealignInputError("ZIP total size exceeds limit")
    return {info.filename: info for info in infos}


def _trusted(path: Path) -> TrustedSource:
    source = Path(path)
    return TrustedSource(source, file_sha256(source), source.stat().st_size)


def _reject_symlink_ancestors(path: Path) -> None:
    source = Path(os.path.abspath(os.fspath(path)))
    for component in (source, *source.parents):
        if component.is_symlink():
            raise RealignInputError(f"source path has a symlink ancestor: {path}")


def _verify_training_source(path: Path, root: Path) -> Mapping[str, TrustedSource]:
    try:
        verified = verify_and_extract_training_input(path, root, load_blend_contract())
    except Exception as error:
        raise RealignInputError(f"training source is invalid: {error}") from error
    return MappingProxyType(
        {
            "data/train.csv": _trusted(verified.data_dir / "train.csv"),
            "data/trackman_history.csv": _trusted(
                verified.data_dir / "trackman_history.csv"
            ),
        }
    )


def _verify_stage_c_source(path: Path, root: Path) -> Mapping[str, TrustedSource]:
    try:
        verified = verify_and_extract_stage_c(path, root, load_blend_contract())
    except Exception as error:
        raise RealignInputError(f"Stage C source is invalid: {error}") from error
    if set(verified.prediction_paths) != {"2022->2023", "2023->2024"}:
        raise RealignInputError("Stage C prediction fold set differs")
    return MappingProxyType(
        {
            "tabm/2022_2023.csv": _trusted(
                verified.prediction_paths["2022->2023"]
            ),
            "tabm/2023_2024.csv": _trusted(
                verified.prediction_paths["2023->2024"]
            ),
        }
    )


def _verify_deployment_sources(
    resume: Path, review: Path, root: Path
) -> Mapping[str, TrustedSource]:
    def read_manifest(path: Path, kind: str) -> tuple[dict[str, str], dict[str, object]]:
        try:
            with ZipFile(path) as archive:
                infos = _checked_infos(archive)
                if "manifest.json" not in infos:
                    raise RealignInputError("deployment manifest is missing")
                manifest = json.loads(archive.read("manifest.json"))
        except RealignInputError:
            raise
        except Exception as error:
            raise RealignInputError(f"deployment archive is invalid: {error}") from error
        bindings = manifest.get("bindings") if type(manifest) is dict else None
        members = manifest.get("members") if type(manifest) is dict else None
        if (
            manifest.get("artifact_kind") != kind
            or type(bindings) is not dict
            or len(bindings) != 11
            or any(
                type(key) is not str
                or type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for key, value in bindings.items()
            )
            or type(members) is not dict
        ):
            raise RealignInputError("deployment manifest differs")
        return dict(bindings), members

    resume_bindings, resume_members = read_manifest(
        resume, "catboost_deployment_resume"
    )
    review_bindings, review_members = read_manifest(
        review, "catboost_deployment_review"
    )
    if resume_bindings != review_bindings:
        raise RealignInputError("deployment artifact bindings differ")
    try:
        verify_deployment_resume(resume, expected_bindings=resume_bindings)
        verify_deployment_review(review, expected_bindings=resume_bindings)
    except Exception as error:
        raise RealignInputError(f"deployment artifact verification failed: {error}") from error

    requested = {
        "catboost/2022_2023/model.cbm": (
            resume,
            resume_members,
            "jobs/align_2022_2023/model.cbm",
        ),
        "catboost/2022_2023/preprocessing_state.json": (
            resume,
            resume_members,
            "jobs/align_2022_2023/preprocessing_state.json",
        ),
        "catboost/2022_2023/predictions.csv": (
            review,
            review_members,
            "predictions/align_2022_2023.csv",
        ),
        "catboost/2023_2024/model.cbm": (
            resume,
            resume_members,
            "jobs/align_2023_2024/model.cbm",
        ),
        "catboost/2023_2024/preprocessing_state.json": (
            resume,
            resume_members,
            "jobs/align_2023_2024/preprocessing_state.json",
        ),
        "catboost/2023_2024/predictions.csv": (
            review,
            review_members,
            "predictions/align_2023_2024.csv",
        ),
    }
    root.mkdir(parents=True, exist_ok=False)
    output: dict[str, TrustedSource] = {}
    for output_name, (archive_path, evidence_map, source_name) in requested.items():
        evidence = evidence_map.get(source_name)
        if type(evidence) is not dict or set(evidence) != {"size", "sha256"}:
            raise RealignInputError(f"deployment member evidence differs: {source_name}")
        target = root.joinpath(*PurePosixPath(output_name).parts[1:])
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256()
        size = 0
        with ZipFile(archive_path) as archive, archive.open(source_name) as source:
            with target.open("xb") as sink:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                    sink.write(chunk)
        if evidence != {"size": size, "sha256": digest.hexdigest()}:
            raise RealignInputError(f"deployment member content differs: {source_name}")
        output[output_name] = TrustedSource(target, digest.hexdigest(), size)
    return MappingProxyType(output)


def _verify_audit_source(
    path: Path, root: Path
) -> tuple[Mapping[str, TrustedSource], Decimal]:
    expected_names = {
        "artifact_inventory.json",
        "audit_summary.md",
        "calibration_deciles.csv",
        "correlation_matrix.csv",
        "model_comparison.csv",
        "next_experiment.json",
        "paired_comparison.csv",
        "segment_diagnostics.csv",
    }
    try:
        with ZipFile(path) as archive:
            infos = _checked_infos(archive)
            if set(infos) != expected_names:
                raise RealignInputError("audit member set differs")
            decision_bytes = archive.read("next_experiment.json")
            inventory_bytes = archive.read("artifact_inventory.json")
    except RealignInputError:
        raise
    except Exception as error:
        raise RealignInputError(f"audit source is invalid: {error}") from error
    try:
        decision = json.loads(decision_bytes)
        inventory = json.loads(inventory_bytes)
    except Exception as error:
        raise RealignInputError("audit JSON is unreadable") from error
    diverse = decision.get("evidence", {}).get("diverse_blend", {})
    stable = diverse.get("stable_blends", [])
    expected_fold_gains = {
        "2022->2023": Decimal("0.00017567321480116852"),
        "2023->2024": Decimal("0.0001844426727390278"),
    }
    matched = [
        item
        for item in stable
        if type(item) is dict
        and item.get("candidate_model_id") == "catboost_hand_matchup_seed_42"
        and Decimal(str(item.get("candidate_weight"))) == Decimal("0.5")
    ]
    if (
        decision.get("schema_version") != 1
        or "DIVERSE_BLEND" not in decision.get("supported_directions", [])
        or diverse.get("supported") is not True
        or len(matched) != 1
    ):
        raise RealignInputError("audit has no sealed stable 0.50 blend")
    item = matched[0]
    fold_gains = item.get("fold_gains", {})
    if (
        {key: Decimal(str(value)) for key, value in fold_gains.items()}
        != expected_fold_gains
        or Decimal(str(item.get("latest_interval_lower")))
        != Decimal("0.00009215809950985597")
        or Decimal(str(item.get("maximum_eligible_segment_regression")))
        != Decimal("0.0004936941230183067")
        or item.get("eligible_segment_count") != 26
    ):
        raise RealignInputError("audit stable 0.50 blend evidence differs")
    required_inventory = {
        "stage_c_tabm": (
            "tabm_colab_stage_C_delivery",
            "f054e8e819d1053299fef4749a32e923db9136e20fc9c12cda79bbc7ef8c484a",
        ),
        "catboost_deployment": (
            "catboost_deployment_review",
            "fc65693aa9439bdbd808fcd29c421c1cfcfcb23abebd1a2c6512109b034641e8",
        ),
    }
    observed = {
        row.get("role"): (row.get("artifact_kind"), row.get("sha256"))
        for row in inventory
        if type(row) is dict and row.get("status") == "verified"
    }
    if any(observed.get(role) != evidence for role, evidence in required_inventory.items()):
        raise RealignInputError("audit inventory identity differs")
    root.mkdir(parents=True, exist_ok=False)
    output: dict[str, TrustedSource] = {}
    for source_name, target_name, payload in (
        ("next_experiment.json", "audit/next_experiment.json", decision_bytes),
        ("artifact_inventory.json", "audit/artifact_inventory.json", inventory_bytes),
    ):
        target = root / source_name
        target.write_bytes(payload)
        output[target_name] = TrustedSource(
            target, sha256(payload).hexdigest(), len(payload)
        )
    return MappingProxyType(output), Decimal("0.50")


def verify_source_artifacts(
    sources: RealignSourcePaths, contract: RealignContract
) -> VerifiedSources:
    supplied = {
        "training_input": sources.training_input,
        "stage_c_delivery": sources.stage_c_delivery,
        "deployment_resume": sources.deployment_resume,
        "deployment_review": sources.deployment_review,
        "oof_audit": sources.oof_audit,
    }
    for kind, raw_path in supplied.items():
        path = Path(raw_path)
        _reject_symlink_ancestors(path)
        if path.is_symlink() or not path.is_file():
            raise RealignInputError(f"source artifact must be a regular file: {kind}")
        if file_sha256(path) != contract.source_sha256[kind]:
            raise RealignInputError(f"source artifact SHA-256 differs: {kind}")
    cleanup_root = Path(
        tempfile.mkdtemp(
            prefix=".catboost-50-50-sources-",
            dir=Path(sources.training_input).parent,
        )
    )
    try:
        members: dict[str, TrustedSource] = {}
        for group in (
            _verify_training_source(Path(sources.training_input), cleanup_root / "training"),
            _verify_stage_c_source(Path(sources.stage_c_delivery), cleanup_root / "stage_c"),
            _verify_deployment_sources(
                Path(sources.deployment_resume),
                Path(sources.deployment_review),
                cleanup_root / "deployment",
            ),
        ):
            if set(members) & set(group):
                raise RealignInputError("source member names overlap")
            members.update(group)
        audit_members, audit_weight = _verify_audit_source(
            Path(sources.oof_audit), cleanup_root / "audit"
        )
        if set(members) & set(audit_members):
            raise RealignInputError("source member names overlap")
        members.update(audit_members)
        if set(members) != _EXPECTED_MEMBER_NAMES:
            raise RealignInputError("verified source member set differs")
        return VerifiedSources(
            members=MappingProxyType(members),
            source_sha256=MappingProxyType(dict(contract.source_sha256)),
            fold_keys=("2021->2022", "2022->2023", "2023->2024"),
            audit_weight=audit_weight,
            cleanup_root=cleanup_root,
        )
    except Exception:
        shutil.rmtree(cleanup_root, ignore_errors=True)
        raise


def prepare_input_archive(
    sources: RealignSourcePaths,
    output: Path,
    contract: RealignContract,
) -> PreparedRealignInput:
    verified = verify_source_artifacts(sources, contract)
    temporary: Path | None = None
    try:
        if dict(verified.source_sha256) != dict(contract.source_sha256):
            raise RealignInputError("source artifact identity differs")
        if set(verified.members) != _EXPECTED_MEMBER_NAMES:
            raise RealignInputError("handoff member set differs")
        manifest = {
            "schema_version": 1,
            "artifact_kind": "catboost_50_50_realign_input_v1",
            "submission_package": False,
            "contract_sha256": contract_sha256(),
            "source_sha256": dict(verified.source_sha256),
            "fold_keys": list(verified.fold_keys),
            "audit_weight": str(verified.audit_weight),
            "members": {
                name: {"size": member.size, "sha256": member.sha256}
                for name, member in sorted(verified.members.items())
            },
        }
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise RealignInputError("output archive already exists")
        temporary = target.with_suffix(target.suffix + ".tmp")
        if temporary.exists() or temporary.is_symlink():
            raise RealignInputError("temporary output archive already exists")
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr(_zip_info("manifest.json"), _canonical_json(manifest))
            for name, member in sorted(verified.members.items()):
                path = Path(member.path)
                _reject_symlink_ancestors(path)
                digest = sha256()
                size = 0
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                try:
                    descriptor = os.open(path, flags)
                except OSError as error:
                    raise RealignInputError(
                        f"cannot open source member safely: {name}"
                    ) from error
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    os.close(descriptor)
                    raise RealignInputError(f"source member is not a regular file: {name}")
                with os.fdopen(descriptor, "rb") as source, archive.open(
                    _zip_info(name), "w"
                ) as sink:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                        size += len(chunk)
                        sink.write(chunk)
                if size != member.size or digest.hexdigest() != member.sha256:
                    raise RealignInputError(f"source member changed while reading: {name}")
        os.replace(temporary, target)
        temporary = None
        return PreparedRealignInput(target, file_sha256(target))
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    finally:
        if verified.cleanup_root is not None:
            shutil.rmtree(verified.cleanup_root, ignore_errors=True)


def _manifest(archive: ZipFile) -> tuple[dict[str, object], bytes]:
    try:
        payload = archive.read("manifest.json")
        value = json.loads(payload.decode("utf-8"))
    except Exception as error:
        raise RealignInputError(f"cannot read input manifest: {error}") from error
    if type(value) is not dict:
        raise RealignInputError("input manifest must be an object")
    return value, payload


def verify_and_extract_input(
    archive: Path,
    destination: Path,
    contract: RealignContract,
    check_deadline: Callable[[], None] | None = None,
) -> VerifiedRealignInput:
    source_path = Path(archive)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise RealignInputError("input destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    root = temporary_root
    try:
        with ZipFile(source_path) as source:
            infos = _checked_infos(source)
            if "manifest.json" not in infos:
                raise RealignInputError("input manifest is missing")
            manifest, manifest_bytes = _manifest(source)
            expected_keys = {
                "schema_version",
                "artifact_kind",
                "submission_package",
                "contract_sha256",
                "source_sha256",
                "fold_keys",
                "audit_weight",
                "members",
            }
            if (
                set(manifest) != expected_keys
                or manifest["schema_version"] != 1
                or manifest["artifact_kind"] != "catboost_50_50_realign_input_v1"
                or manifest["submission_package"] is not False
                or manifest["contract_sha256"] != contract_sha256()
                or manifest["source_sha256"] != dict(contract.source_sha256)
                or manifest["fold_keys"]
                != ["2021->2022", "2022->2023", "2023->2024"]
                or manifest["audit_weight"] != "0.50"
                or type(manifest["members"]) is not dict
                or set(manifest["members"]) != _EXPECTED_MEMBER_NAMES
                or set(infos) != {"manifest.json", *_EXPECTED_MEMBER_NAMES}
            ):
                raise RealignInputError("input manifest differs")
            for name in sorted(_EXPECTED_MEMBER_NAMES):
                if check_deadline is not None:
                    check_deadline()
                evidence = manifest["members"][name]
                if type(evidence) is not dict or set(evidence) != {"size", "sha256"}:
                    raise RealignInputError(f"member evidence differs: {name}")
                target = root.joinpath(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = sha256()
                size = 0
                temporary = target.with_suffix(target.suffix + ".tmp")
                with source.open(name) as stream, temporary.open("xb") as sink:
                    while chunk := stream.read(1024 * 1024):
                        if check_deadline is not None:
                            check_deadline()
                        digest.update(chunk)
                        size += len(chunk)
                        sink.write(chunk)
                if evidence != {"size": size, "sha256": digest.hexdigest()}:
                    raise RealignInputError(f"member content differs: {name}")
                os.replace(temporary, target)
        os.replace(temporary_root, destination)
        root = destination
    except RealignInputError:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    except (BadZipFile, OSError) as error:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise RealignInputError(f"input ZIP is invalid: {error}") from error
    return VerifiedRealignInput(
        root=root,
        data_dir=root / "data",
        tabm_predictions=MappingProxyType(
            {
                "2022->2023": root / "tabm/2022_2023.csv",
                "2023->2024": root / "tabm/2023_2024.csv",
            }
        ),
        catboost_models=MappingProxyType(
            {
                "2022->2023": root / "catboost/2022_2023/model.cbm",
                "2023->2024": root / "catboost/2023_2024/model.cbm",
            }
        ),
        catboost_states=MappingProxyType(
            {
                "2022->2023": root / "catboost/2022_2023/preprocessing_state.json",
                "2023->2024": root / "catboost/2023_2024/preprocessing_state.json",
            }
        ),
        catboost_predictions=MappingProxyType(
            {
                "2022->2023": root / "catboost/2022_2023/predictions.csv",
                "2023->2024": root / "catboost/2023_2024/predictions.csv",
            }
        ),
        audit_decision=root / "audit/next_experiment.json",
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
        fold_keys=("2021->2022", "2022->2023", "2023->2024"),
        audit_weight=Decimal("0.50"),
    )
