from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Mapping
from zipfile import ZipFile, ZipInfo

from .artifacts import verify_resume_bundle, verify_review_bundle


class VersionDError(RuntimeError):
    pass


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
