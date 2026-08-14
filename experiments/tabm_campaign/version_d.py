from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from dataclasses import dataclass
from typing import Mapping, Sequence
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from .artifacts import verify_resume_bundle, verify_review_bundle


class VersionDError(RuntimeError):
    pass


_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class VerifiedEmergency:
    path: Path
    epoch: int
    sha256: str
    identity: Mapping[str, str]


@dataclass(frozen=True)
class VerifiedFrozen:
    path: Path
    sha256: str
    identity: Mapping[str, str]


@dataclass(frozen=True)
class RecoverySelection:
    mode: str
    path: Path | None
    epoch: int


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_version_d_contract(path: Path | None = None) -> dict[str, object]:
    source = Path(__file__).with_name("version_d_contract.json") if path is None else path
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except Exception as error:
        raise VersionDError(f"cannot load Version D contract: {error}") from error
    if value.get("schema_version") != 1 or value.get("version") != "D":
        raise VersionDError("Version D contract identity differs")
    if value.get("review_only") is not True or value.get("submission_package") is not False:
        raise VersionDError("Version D policy differs")
    final_fit = value.get("final_fit")
    if not isinstance(final_fit, dict):
        raise VersionDError("Version D final fit policy is missing")
    if final_fit.get("epochs") != 3 or final_fit.get("scheduler") != "constant":
        raise VersionDError("Version D final fit policy differs")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise VersionDError(f"{label} is missing")
    return value


def _safe_member(info: ZipInfo) -> None:
    path = PurePosixPath(info.filename)
    mode = info.external_attr >> 16
    if (
        not info.filename
        or "\\" in info.filename
        or path.is_absolute()
        or ".." in path.parts
        or info.is_dir()
        or (mode & 0o170000) == 0o120000
    ):
        raise VersionDError(f"unsafe ZIP member: {info.filename}")


def _verify_zip_members(
    archive: ZipFile,
    expected: Mapping[str, object],
    *,
    verify_hashes: bool,
) -> dict[str, ZipInfo]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise VersionDError("ZIP member names differ")
    result: dict[str, ZipInfo] = {}
    for info in infos:
        _safe_member(info)
        evidence = _mapping(expected[info.filename], f"ZIP evidence for {info.filename}")
        if info.file_size != int(evidence.get("size", -1)):
            raise VersionDError(f"ZIP member size differs: {info.filename}")
        if verify_hashes and sha256(archive.read(info.filename)).hexdigest() != evidence.get("sha256"):
            raise VersionDError(f"ZIP member SHA-256 differs: {info.filename}")
        result[info.filename] = info
    return result


def _verify_data_archive(path: Path, contract: Mapping[str, object]) -> tuple[Mapping[str, object], dict[str, ZipInfo]]:
    archive_contract = _mapping(contract.get("data_archive"), "data archive contract")
    if file_sha256(path) != archive_contract.get("sha256"):
        raise VersionDError("official data archive SHA-256 differs")
    members = _mapping(archive_contract.get("members"), "official data member contract")
    with ZipFile(path) as archive:
        infos = _verify_zip_members(archive, members, verify_hashes=False)
    return members, infos


def _existing_members_match(root: Path, members: Mapping[str, object], names: tuple[str, ...]) -> bool:
    if not root.is_dir() or set(path.name for path in root.iterdir()) != set(names):
        return False
    for name in names:
        evidence = _mapping(members[name], f"official data evidence for {name}")
        path = root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(evidence.get("size", -1))
            or file_sha256(path) != evidence.get("sha256")
        ):
            return False
    return True


def _extract_selected(
    archive_path: Path,
    destination: Path,
    contract: Mapping[str, object],
    names: tuple[str, ...],
) -> Path:
    members, _ = _verify_data_archive(archive_path, contract)
    if _existing_members_match(destination, members, names):
        return destination
    if destination.exists():
        raise VersionDError(f"existing data directory differs: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        with ZipFile(archive_path) as archive:
            for name in names:
                evidence = _mapping(members[name], f"official data evidence for {name}")
                target = temporary / name
                digest = sha256()
                size = 0
                with archive.open(name) as source, target.open("wb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                        digest.update(block)
                        size += len(block)
                    output.flush()
                    os.fsync(output.fileno())
                if size != int(evidence.get("size", -1)) or digest.hexdigest() != evidence.get("sha256"):
                    raise VersionDError(f"official data file differs: {name}")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def extract_training_input(
    archive_path: Path,
    destination: Path,
    contract: Mapping[str, object],
) -> Path:
    return _extract_selected(archive_path, destination, contract, ("train.csv",))


def extract_review_inputs(
    archive_path: Path,
    destination: Path,
    contract: Mapping[str, object],
) -> Path:
    return _extract_selected(
        archive_path,
        destination,
        contract,
        ("test.csv", "sample_submission.csv"),
    )


def _delivery_members(archive: ZipFile, manifest: Mapping[str, object]) -> dict[str, bytes]:
    expected = _mapping(manifest.get("members"), "Stage C delivery member manifest")
    required = {
        "colab_stage_C.log",
        "tabm_search_stage_C_review_bundle.zip",
        "tabm_search_stage_C_resume_bundle.zip",
    }
    if set(expected) != required:
        raise VersionDError("Stage C delivery member manifest differs")
    infos = {info.filename: info for info in archive.infolist()}
    if set(infos) != required | {"delivery_manifest.json"} or len(infos) != 4:
        raise VersionDError("Stage C delivery member names differ")
    output: dict[str, bytes] = {}
    for name in required:
        _safe_member(infos[name])
        value = archive.read(name)
        evidence = _mapping(expected[name], f"Stage C delivery evidence for {name}")
        if len(value) != int(evidence.get("size", -1)) or sha256(value).hexdigest() != evidence.get("sha256"):
            raise VersionDError(f"Stage C delivery member differs: {name}")
        output[name] = value
    return output


def verify_stage_c_delivery(
    path: Path,
    contract: Mapping[str, object],
) -> dict[str, object]:
    delivery_contract = _mapping(contract.get("stage_c_delivery"), "Stage C delivery contract")
    if file_sha256(path) != delivery_contract.get("sha256"):
        raise VersionDError("Stage C delivery SHA-256 differs")
    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
            if names.count("delivery_manifest.json") != 1:
                raise VersionDError("Stage C delivery manifest is missing")
            manifest = json.loads(archive.read("delivery_manifest.json"))
            if (
                manifest.get("artifact_kind") != "tabm_colab_stage_C_delivery"
                or manifest.get("final_stage_complete") is not True
            ):
                raise VersionDError("Stage C delivery is incomplete")
            members = _delivery_members(archive, manifest)
    except VersionDError:
        raise
    except Exception as error:
        raise VersionDError(f"cannot verify Stage C delivery: {error}") from error

    review_name = "tabm_search_stage_C_review_bundle.zip"
    resume_name = "tabm_search_stage_C_resume_bundle.zip"
    review_bytes = members[review_name]
    resume_bytes = members[resume_name]
    if sha256(review_bytes).hexdigest() != delivery_contract.get("review_sha256"):
        raise VersionDError("Stage C review bundle SHA-256 differs")
    if sha256(resume_bytes).hexdigest() != delivery_contract.get("resume_sha256"):
        raise VersionDError("Stage C resume bundle SHA-256 differs")

    with tempfile.TemporaryDirectory(prefix="version-d-stage-c-") as directory:
        root = Path(directory)
        review_path = root / review_name
        resume_path = root / resume_name
        review_path.write_bytes(review_bytes)
        resume_path.write_bytes(resume_bytes)
        verified_review = verify_review_bundle(review_path)
        verified_resume = verify_resume_bundle(resume_path)
    if (
        verified_review.version != "C"
        or verified_resume.version != "C"
        or verified_review.campaign_config_sha256 != verified_resume.campaign_config_sha256
        or verified_review.prior_manifest_sha256 != verified_resume.prior_manifest_sha256
    ):
        raise VersionDError("Stage C bundle lineage differs")

    try:
        with ZipFile(BytesIO(resume_bytes)) as resume_archive:
            state_bytes = resume_archive.read("stage_state.json")
            state = json.loads(state_bytes)
    except Exception as error:
        raise VersionDError(f"cannot read Stage C state: {error}") from error
    if sha256(state_bytes).hexdigest() != delivery_contract.get("stage_state_sha256"):
        raise VersionDError("Stage C state SHA-256 differs")
    if state.get("version") != "C" or state.get("stage_complete") is not True:
        raise VersionDError("Stage C state is incomplete")
    predictor_id = contract.get("predictor_id")
    if state.get("selected_predictor") != predictor_id:
        raise VersionDError("Stage C selected predictor differs")
    results = state.get("results")
    expected_jobs = delivery_contract.get("completed_jobs")
    if (
        not isinstance(results, list)
        or len(results) != expected_jobs
        or any(result.get("status") != "completed" for result in results if isinstance(result, Mapping))
        or any(not isinstance(result, Mapping) for result in results)
    ):
        raise VersionDError("Stage C completed result set differs")
    final_members = state.get("final_members")
    if not isinstance(final_members, list) or len(final_members) != 1:
        raise VersionDError("Stage C final member count differs")
    final_member = _mapping(final_members[0], "Stage C final member")
    if (
        final_member.get("status") != "completed"
        or final_member.get("seed") != 3407
        or final_member.get("temporal_best_epochs") != [3, 0]
    ):
        raise VersionDError("Stage C final member evidence differs")
    candidate = _mapping(final_member.get("candidate"), "Stage C final candidate")
    selected = _mapping(contract.get("selected_candidate"), "selected candidate contract")
    if any(candidate.get(key) != value for key, value in selected.items()):
        raise VersionDError("Stage C selected candidate differs")
    return {
        "delivery_sha256": file_sha256(path),
        "review_sha256": sha256(review_bytes).hexdigest(),
        "resume_sha256": sha256(resume_bytes).hexdigest(),
        "prior_manifest_sha256": verified_resume.manifest_sha256,
        "selected_predictor": predictor_id,
        "final_member": dict(final_member),
        "stage_state": state,
    }


def _deterministic_zip(members: Mapping[str, bytes]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in sorted(members.items()):
            path = PurePosixPath(name)
            if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
                raise VersionDError(f"unsafe archive member: {name}")
            info = ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value)
    return buffer.getvalue()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _snapshot_manifest(
    kind: str,
    identity: Mapping[str, str],
    members: Mapping[str, bytes],
    *,
    epoch: int,
) -> bytes:
    return canonical_json(
        {
            "schema_version": 1,
            "artifact_kind": kind,
            "epoch": epoch,
            "identity": dict(identity),
            "members": {
                name: sha256(value).hexdigest() for name, value in sorted(members.items())
            },
        }
    )


def write_emergency_snapshot(
    output_dir: Path,
    checkpoint_path: Path,
    preprocessing_path: Path,
    numeric_path: Path,
    epoch: int,
    identity: Mapping[str, str],
) -> Path:
    if epoch not in {1, 2, 3}:
        raise VersionDError("emergency snapshot epoch differs")
    sources = {
        "checkpoint.pt": checkpoint_path,
        "preprocessing_state.json": preprocessing_path,
        "numeric_embedding_0.json": numeric_path,
    }
    if any(path.is_symlink() or not path.is_file() for path in sources.values()):
        raise VersionDError("emergency snapshot source is missing")
    members = {name: path.read_bytes() for name, path in sources.items()}
    members["snapshot_manifest.json"] = _snapshot_manifest(
        "tabm_version_D_emergency", identity, members, epoch=epoch
    )
    value = _deterministic_zip(members)
    digest = sha256(value).hexdigest()
    path = output_dir / f"tabm_version_D_emergency_epoch_{epoch:03d}_{digest[:12]}.zip"
    _atomic_bytes(path, value)
    verify_emergency_snapshot(path, identity)
    return path


def _verified_snapshot_members(
    path: Path,
    manifest_name: str,
) -> tuple[dict[str, object], dict[str, bytes]]:
    try:
        with ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or manifest_name not in names:
                raise VersionDError("snapshot member set differs")
            for info in infos:
                _safe_member(info)
            manifest = json.loads(archive.read(manifest_name))
            expected = _mapping(manifest.get("members"), "snapshot member manifest")
            if set(expected) != set(names) - {manifest_name}:
                raise VersionDError("snapshot member set differs")
            members = {name: archive.read(name) for name in names if name != manifest_name}
    except VersionDError:
        raise
    except Exception as error:
        raise VersionDError(f"cannot verify snapshot: {error}") from error
    if any(sha256(value).hexdigest() != expected[name] for name, value in members.items()):
        raise VersionDError("snapshot member SHA-256 differs")
    return dict(manifest), members


def verify_emergency_snapshot(
    path: Path,
    expected_identity: Mapping[str, str],
) -> VerifiedEmergency:
    manifest, members = _verified_snapshot_members(path, "snapshot_manifest.json")
    if manifest.get("artifact_kind") != "tabm_version_D_emergency":
        raise VersionDError("emergency snapshot kind differs")
    if manifest.get("identity") != dict(expected_identity):
        raise VersionDError("emergency snapshot identity differs")
    epoch = manifest.get("epoch")
    if epoch not in {1, 2, 3}:
        raise VersionDError("emergency snapshot epoch differs")
    if set(members) != {
        "checkpoint.pt",
        "preprocessing_state.json",
        "numeric_embedding_0.json",
    }:
        raise VersionDError("emergency snapshot member set differs")
    return VerifiedEmergency(path, int(epoch), file_sha256(path), dict(expected_identity))


def write_frozen_snapshot(
    output_dir: Path,
    artifact_root: Path,
    identity: Mapping[str, str],
) -> Path:
    manifest_path = artifact_root / "inference_manifest.json"
    try:
        inference_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise VersionDError(f"cannot read frozen inference manifest: {error}") from error
    if (
        inference_manifest.get("epochs") != 3
        or inference_manifest.get("seeds") != [3407]
        or inference_manifest.get("scheduler") != "constant"
        or inference_manifest.get("identity") != dict(identity)
    ):
        raise VersionDError("frozen inference identity differs")
    declared = _mapping(inference_manifest.get("files"), "frozen inference files")
    expected_names = set(declared) | {"inference_manifest.json"}
    actual_names = {
        path.relative_to(artifact_root).as_posix()
        for path in artifact_root.rglob("*")
        if path.is_file()
    }
    if actual_names != expected_names:
        raise VersionDError("frozen artifact member set differs")
    members: dict[str, bytes] = {}
    for name in sorted(expected_names):
        path = artifact_root / name
        if path.is_symlink() or not path.is_file():
            raise VersionDError(f"frozen artifact is missing: {name}")
        value = path.read_bytes()
        if name in declared and sha256(value).hexdigest() != declared[name]:
            raise VersionDError(f"frozen artifact SHA-256 differs: {name}")
        members[name] = value
    members["frozen_manifest.json"] = _snapshot_manifest(
        "tabm_version_D_frozen_model", identity, members, epoch=3
    )
    value = _deterministic_zip(members)
    path = output_dir / "tabm_version_D_frozen_model.zip"
    _atomic_bytes(path, value)
    verify_frozen_snapshot(path, identity)
    return path


def verify_frozen_snapshot(
    path: Path,
    expected_identity: Mapping[str, str],
) -> VerifiedFrozen:
    manifest, members = _verified_snapshot_members(path, "frozen_manifest.json")
    if manifest.get("artifact_kind") != "tabm_version_D_frozen_model":
        raise VersionDError("frozen snapshot kind differs")
    if manifest.get("identity") != dict(expected_identity):
        raise VersionDError("frozen snapshot identity differs")
    if manifest.get("epoch") != 3 or "inference_manifest.json" not in members:
        raise VersionDError("frozen snapshot member set differs")
    lowered = "\n".join(members).lower()
    if any(token in lowered for token in ("optimizer", "scaler", "rng")):
        raise VersionDError("frozen snapshot contains training state")
    try:
        inference_manifest = json.loads(members["inference_manifest.json"])
    except Exception as error:
        raise VersionDError(f"cannot read frozen snapshot manifest: {error}") from error
    declared = _mapping(inference_manifest.get("files"), "frozen inference files")
    if set(members) != set(declared) | {"inference_manifest.json"}:
        raise VersionDError("frozen snapshot member set differs")
    if (
        inference_manifest.get("epochs") != 3
        or inference_manifest.get("seeds") != [3407]
        or inference_manifest.get("scheduler") != "constant"
        or inference_manifest.get("identity") != dict(expected_identity)
    ):
        raise VersionDError("frozen snapshot inference identity differs")
    if any(sha256(members[name]).hexdigest() != digest for name, digest in declared.items()):
        raise VersionDError("frozen snapshot inference hash differs")
    return VerifiedFrozen(path, file_sha256(path), dict(expected_identity))


def restore_emergency_snapshot(
    path: Path,
    checkpoint_path: Path,
    artifact_root: Path,
    expected_identity: Mapping[str, str],
) -> VerifiedEmergency:
    verified = verify_emergency_snapshot(path, expected_identity)
    with ZipFile(path) as archive:
        values = {
            name: archive.read(name)
            for name in (
                "checkpoint.pt",
                "preprocessing_state.json",
                "numeric_embedding_0.json",
            )
        }
    artifact_root.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(checkpoint_path, values["checkpoint.pt"])
    _atomic_bytes(
        artifact_root / "preprocessing_state.json",
        values["preprocessing_state.json"],
    )
    _atomic_bytes(
        artifact_root / "numeric_embedding_0.json",
        values["numeric_embedding_0.json"],
    )
    return verified


def restore_frozen_snapshot(
    path: Path,
    artifact_root: Path,
    expected_identity: Mapping[str, str],
) -> VerifiedFrozen:
    verified = verify_frozen_snapshot(path, expected_identity)
    with ZipFile(path) as archive:
        names = set(archive.namelist()) - {"frozen_manifest.json"}
        values = {name: archive.read(name) for name in names}
    replace_existing = artifact_root.is_dir()
    if replace_existing:
        actual = {
            item.relative_to(artifact_root).as_posix(): item.read_bytes()
            for item in artifact_root.rglob("*")
            if item.is_file()
        }
        if actual == values:
            return verified
    elif artifact_root.exists():
        raise VersionDError("frozen artifact destination is unsafe")
    artifact_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{artifact_root.name}-", dir=artifact_root.parent)
    )
    try:
        for name, value in values.items():
            target = temporary.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            _atomic_bytes(target, value)
        if replace_existing:
            backup = artifact_root.with_name(f".{artifact_root.name}.previous")
            if backup.exists():
                raise VersionDError("stale frozen artifact backup exists")
            os.replace(artifact_root, backup)
            try:
                os.replace(temporary, artifact_root)
            except Exception:
                os.replace(backup, artifact_root)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(temporary, artifact_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verified


def select_recovery_snapshot(
    emergency_paths: Sequence[Path],
    frozen_paths: Sequence[Path],
    expected_identity: Mapping[str, str],
) -> RecoverySelection:
    frozen = [verify_frozen_snapshot(path, expected_identity) for path in frozen_paths]
    if frozen:
        digests = {item.sha256 for item in frozen}
        if len(digests) != 1:
            raise VersionDError("conflicting recovery snapshots")
        return RecoverySelection("frozen", frozen[0].path, 3)
    emergency = [
        verify_emergency_snapshot(path, expected_identity) for path in emergency_paths
    ]
    if not emergency:
        return RecoverySelection("fresh", None, 0)
    latest_epoch = max(item.epoch for item in emergency)
    latest = [item for item in emergency if item.epoch == latest_epoch]
    if len({item.sha256 for item in latest}) != 1:
        raise VersionDError("conflicting recovery snapshots")
    return RecoverySelection("emergency", latest[0].path, latest_epoch)


def write_review_delivery(
    *,
    output_dir: Path,
    review_bundle: Path,
    log_path: Path,
    contract_sha256: str,
    data_archive_sha256: str,
    stage_c_delivery_sha256: str,
    prior_manifest_sha256: str,
    frozen_sha256: str,
    runtime_versions: Mapping[str, str],
) -> Path:
    if not review_bundle.is_file() or not log_path.is_file():
        raise VersionDError("Version D delivery source is missing")
    verified_review = verify_review_bundle(review_bundle)
    if (
        verified_review.version != "D"
        or verified_review.campaign_config_sha256 != contract_sha256
        or verified_review.prior_manifest_sha256 != prior_manifest_sha256
    ):
        raise VersionDError("Version D review bundle lineage differs")
    members = {
        "tabm_hand_matchup_final_review_bundle.zip": review_bundle.read_bytes(),
        "version_d.log": log_path.read_bytes(),
    }
    manifest = {
        "schema_version": 1,
        "artifact_kind": "tabm_version_D_review_delivery",
        "review_only": True,
        "submission_package": False,
        "contract_sha256": contract_sha256,
        "data_archive_sha256": data_archive_sha256,
        "stage_c_delivery_sha256": stage_c_delivery_sha256,
        "prior_manifest_sha256": prior_manifest_sha256,
        "frozen_sha256": frozen_sha256,
        "runtime_versions": dict(runtime_versions),
        "members": {
            name: {"size": len(value), "sha256": sha256(value).hexdigest()}
            for name, value in sorted(members.items())
        },
    }
    members["delivery_manifest.json"] = canonical_json(manifest)
    value = _deterministic_zip(members)
    path = output_dir / "tabm_hand_matchup_stage_D_review_delivery.zip"
    _atomic_bytes(path, value)
    verify_review_delivery(path)
    return path


def verify_review_delivery(path: Path) -> dict[str, object]:
    required = {
        "delivery_manifest.json",
        "tabm_hand_matchup_final_review_bundle.zip",
        "version_d.log",
    }
    try:
        with ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or set(names) != required:
                raise VersionDError("Version D delivery member set differs")
            for info in infos:
                _safe_member(info)
            manifest = json.loads(archive.read("delivery_manifest.json"))
            if (
                manifest.get("artifact_kind") != "tabm_version_D_review_delivery"
                or manifest.get("review_only") is not True
                or manifest.get("submission_package") is not False
            ):
                raise VersionDError("Version D delivery identity differs")
            expected = _mapping(manifest.get("members"), "Version D delivery members")
            if set(expected) != required - {"delivery_manifest.json"}:
                raise VersionDError("Version D delivery manifest differs")
            values = {
                name: archive.read(name) for name in required - {"delivery_manifest.json"}
            }
    except VersionDError:
        raise
    except Exception as error:
        raise VersionDError(f"cannot verify Version D delivery: {error}") from error
    for name, value in values.items():
        evidence = _mapping(expected[name], f"Version D delivery evidence for {name}")
        if len(value) != evidence.get("size") or sha256(value).hexdigest() != evidence.get("sha256"):
            raise VersionDError(f"Version D delivery member differs: {name}")
    with tempfile.TemporaryDirectory(prefix="version-d-review-") as directory:
        review_path = Path(directory) / "tabm_hand_matchup_final_review_bundle.zip"
        review_path.write_bytes(values["tabm_hand_matchup_final_review_bundle.zip"])
        verified_review = verify_review_bundle(review_path)
    if (
        verified_review.version != "D"
        or verified_review.campaign_config_sha256 != manifest.get("contract_sha256")
        or verified_review.prior_manifest_sha256 != manifest.get("prior_manifest_sha256")
    ):
        raise VersionDError("Version D delivery review lineage differs")
    return dict(manifest)
