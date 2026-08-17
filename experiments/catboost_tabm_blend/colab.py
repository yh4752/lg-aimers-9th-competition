from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable, Mapping, Sequence
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

from .artifacts import verify_resume_bundle, verify_review_bundle, write_bundles
from .contracts import DEFAULT_CONTRACT, BlendContract, build_jobs, contract_sha256, load_contract
from .inputs import (
    VerifiedStageC,
    VerifiedTrainingInput,
    verify_and_extract_stage_c,
    verify_and_extract_training_input,
)
from .runner import BlendRun, _bindings, _state_payload, run_campaign
from .training import FoldResult


class BlendColabError(RuntimeError):
    """Raised when the direct-upload Colab handoff is not trustworthy."""


_DATA_MEMBERS = {"input_manifest.json", "train.csv", "trackman_history.csv"}
_STAGE_C_MEMBERS = {
    "delivery_manifest.json",
    "colab_stage_C.log",
    "tabm_search_stage_C_review_bundle.zip",
    "tabm_search_stage_C_resume_bundle.zip",
}
_DELIVERY_MEMBERS = {
    "blend_campaign.log",
    "catboost_tabm_blend_review.zip",
    "catboost_tabm_blend_resume.zip",
    "delivery_manifest.json",
}
_ZIP_TIME = (2026, 1, 1, 0, 0, 0)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _checked_names(path: Path) -> set[str]:
    try:
        with ZipFile(path) as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if len(names) != len(set(names)):
                raise BlendColabError("unsafe duplicate ZIP member")
            total = 0
            for item in infos:
                pure = PurePosixPath(item.filename)
                mode = item.external_attr >> 16
                total += item.file_size
                if (
                    not item.filename
                    or "\\" in item.filename
                    or pure.is_absolute()
                    or ".." in pure.parts
                    or item.is_dir()
                    or stat.S_IFMT(mode) == stat.S_IFLNK
                    or item.file_size > 8 * 1024**3
                    or (item.file_size >= 65536 and item.file_size / max(1, item.compress_size) > 200)
                ):
                    raise BlendColabError(f"unsafe ZIP member: {item.filename}")
            if total > 12 * 1024**3:
                raise BlendColabError("unsafe ZIP total size")
            return set(names)
    except BadZipFile as error:
        raise BlendColabError("upload is not a readable ZIP") from error


def classify_upload_kind(path: Path) -> str:
    names = _checked_names(Path(path))
    if names == _DATA_MEMBERS:
        return "training_input"
    if names == _STAGE_C_MEMBERS:
        return "stage_c_delivery"
    if {
        "manifest.json",
        "contract/contract.json",
        "state/stage_state.json",
    } <= names:
        try:
            with ZipFile(path) as archive:
                if archive.getinfo("manifest.json").file_size > 1024 * 1024:
                    raise BlendColabError("resume manifest exceeds limit")
                manifest = json.loads(archive.read("manifest.json"))
        except (KeyError, ValueError, BadZipFile) as error:
            raise BlendColabError("resume manifest is unreadable") from error
        if manifest.get("artifact_kind") == "catboost_tabm_blend_resume":
            return "blend_resume"
    raise BlendColabError("unknown upload kind")


def classify_and_verify_uploads(
    paths: Sequence[Path],
    *,
    run_root: Path,
    contract: BlendContract,
    expected_contract_sha256: str,
    expected_code_sha256: str,
) -> tuple[VerifiedTrainingInput, VerifiedStageC, Path | None]:
    if len(paths) not in (2, 3):
        raise BlendColabError("upload count must be two or three")
    if expected_contract_sha256 != contract_sha256():
        raise BlendColabError("contract identity differs")
    grouped: dict[str, Path] = {}
    for source in paths:
        kind = classify_upload_kind(Path(source))
        if kind in grouped:
            raise BlendColabError(f"duplicate upload kind: {kind}")
        grouped[kind] = Path(source)
    required = {"training_input", "stage_c_delivery"}
    if not required <= set(grouped) or set(grouped) - required - {"blend_resume"}:
        raise BlendColabError("required uploads are missing")
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=False)
    verified_input = verify_and_extract_training_input(
        grouped["training_input"], run_root / "official_data", contract
    )
    verified_stage_c = verify_and_extract_stage_c(
        grouped["stage_c_delivery"], run_root / "stage_c", contract
    )
    resume = grouped.get("blend_resume")
    if resume is not None:
        bindings = _bindings(verified_input, verified_stage_c)
        if bindings["code_sha256"] != expected_code_sha256:
            raise BlendColabError("code identity differs")
        verify_resume_bundle(resume, expected_bindings=bindings)
        copied = run_root / "uploaded_blend_resume.zip"
        shutil.copyfile(resume, copied)
        resume = copied
    return verified_input, verified_stage_c, resume


@dataclass
class EmergencyCadence:
    interval_seconds: float
    started_at: float
    last_published_at: float | None = None
    last_snapshot_sha256: str | None = None

    def should_publish(self, *, now: float, snapshot_sha256: str) -> bool:
        boundary = self.last_published_at or self.started_at
        return (
            now - boundary >= self.interval_seconds
            and snapshot_sha256 != self.last_snapshot_sha256
        )

    def mark_published(self, *, now: float, snapshot_sha256: str) -> None:
        self.last_published_at = now
        self.last_snapshot_sha256 = snapshot_sha256


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=_ZIP_TIME)
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_delivery(
    *,
    destination: Path,
    campaign_log: Path,
    review_bundle: Path,
    resume_bundle: Path,
    bindings: Mapping[str, str],
) -> Path:
    verify_review_bundle(review_bundle, expected_bindings=bindings)
    verify_resume_bundle(resume_bundle, expected_bindings=bindings)
    sources = {
        "blend_campaign.log": Path(campaign_log),
        "catboost_tabm_blend_review.zip": Path(review_bundle),
        "catboost_tabm_blend_resume.zip": Path(resume_bundle),
    }
    evidence = {
        name: {"size": path.stat().st_size, "sha256": file_sha256(path)}
        for name, path in sorted(sources.items())
    }
    manifest = _canonical_json(
        {
            "schema_version": 1,
            "artifact_kind": "catboost_tabm_blend_delivery",
            "review_only": True,
            "submission_package": False,
            "bindings": dict(bindings),
            "members": evidence,
        }
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}-", dir=destination.parent)
    os.close(descriptor)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for name, path in sorted(sources.items()):
                with path.open("rb") as source, archive.open(_zip_info(name), "w") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            archive.writestr(_zip_info("delivery_manifest.json"), manifest)
        verify_delivery(Path(temporary), expected_bindings=bindings, verify_nested=False)
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def verify_delivery(
    path: Path,
    *,
    expected_bindings: Mapping[str, str],
    verify_nested: bool = True,
) -> None:
    if _checked_names(Path(path)) != _DELIVERY_MEMBERS:
        raise BlendColabError("delivery member set differs")
    temporary = Path(tempfile.mkdtemp(prefix="blend-delivery-"))
    try:
        with ZipFile(path) as archive:
            manifest = json.loads(archive.read("delivery_manifest.json"))
            if (
                set(manifest) != {
                    "schema_version", "artifact_kind", "review_only",
                    "submission_package", "bindings", "members",
                }
                or manifest["schema_version"] != 1
                or manifest["artifact_kind"] != "catboost_tabm_blend_delivery"
                or manifest["review_only"] is not True
                or manifest["submission_package"] is not False
                or manifest["bindings"] != dict(expected_bindings)
                or set(manifest["members"]) != _DELIVERY_MEMBERS - {"delivery_manifest.json"}
            ):
                raise BlendColabError("delivery manifest differs")
            extracted: dict[str, Path] = {}
            for name, evidence in manifest["members"].items():
                target = temporary / name
                digest = sha256()
                size = 0
                with archive.open(name) as source, target.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                if evidence != {"size": size, "sha256": digest.hexdigest()}:
                    raise BlendColabError(f"delivery member differs: {name}")
                extracted[name] = target
        if verify_nested:
            verify_review_bundle(
                extracted["catboost_tabm_blend_review.zip"],
                expected_bindings=expected_bindings,
            )
            verify_resume_bundle(
                extracted["catboost_tabm_blend_resume.zip"],
                expected_bindings=expected_bindings,
            )
    except (BlendColabError, BadZipFile):
        raise
    except Exception as error:
        raise BlendColabError(f"delivery verification failed: {error}") from error
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def run_supervised_campaign(
    *,
    verified_input: VerifiedTrainingInput,
    verified_stage_c: VerifiedStageC,
    output_dir: Path,
    snapshot_dir: Path,
    resume_bundle: Path | None,
    wall_deadline: float,
    on_verified_resume: Callable[[Path], None],
    log_path: Path,
) -> BlendRun:
    """Run the two folds while periodically sealing changed native snapshots."""
    Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).touch(exist_ok=True)
    bindings = _bindings(verified_input, verified_stage_c)
    latest: Path | None = None

    def publish(path: Path) -> None:
        nonlocal latest
        digest = file_sha256(path)
        target = Path(snapshot_dir) / f"catboost_tabm_blend_emergency_{digest[:12]}.zip"
        if latest is not None and file_sha256(latest) == digest:
            return
        shutil.copyfile(path, target)
        verify_resume_bundle(target, expected_bindings=bindings)
        latest = target
        print(f"CATBOOST_SNAPSHOT_READY path={target} sha256={digest}", flush=True)
        on_verified_resume(target)

    def completed_directories(jobs_root: Path) -> tuple[list[str], dict[str, Path]]:
        completed: list[str] = []
        directories: dict[str, Path] = {}
        for job in build_jobs(load_contract()):
            directory = jobs_root / job.job_id
            result_path = directory / "worker_result.json"
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if result.get("job_id") == job.job_id and result.get("status") == "completed":
                completed.append(job.job_id)
                directories[job.job_id] = directory
        return completed, directories

    def stable_active_copy(source: Path, destination: Path) -> str:
        destination.mkdir(parents=True, exist_ok=False)
        for name in ("job.json", "worker.log", "experiment.cbsnapshot"):
            path = source / name
            if path.is_symlink() or not path.is_file():
                raise BlendColabError(f"active snapshot source is missing: {name}")
            before = file_sha256(path)
            shutil.copyfile(path, destination / name)
            after = file_sha256(path)
            if before != after or file_sha256(destination / name) != before:
                raise BlendColabError(f"active snapshot changed while copying: {name}")
        return file_sha256(destination / "experiment.cbsnapshot")

    cadence = EmergencyCadence(interval_seconds=1200.0, started_at=time.time())
    sequence = 0

    def seal_active(job_directory: Path, *, force: bool = False) -> None:
        nonlocal sequence
        snapshot = job_directory / "experiment.cbsnapshot"
        if not snapshot.is_file() or snapshot.is_symlink():
            return
        observed = file_sha256(snapshot)
        now = time.time()
        if not force and not cadence.should_publish(now=now, snapshot_sha256=observed):
            return
        publish_root = Path(snapshot_dir) / f"publish_{sequence:04d}"
        sequence += 1
        active_copy = publish_root / "active" / job_directory.name
        snapshot_sha = stable_active_copy(job_directory, active_copy)
        completed, directories = completed_directories(job_directory.parent)
        directories[job_directory.name] = active_copy
        state_path = publish_root / "stage_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_bytes(
            _canonical_json(
                _state_payload(
                    bindings,
                    completed,
                    job_directory.name,
                    complete=False,
                )
            )
        )
        bundles = write_bundles(
            output_dir=publish_root / "bundles",
            contract_path=DEFAULT_CONTRACT,
            bindings=bindings,
            stage_state_path=state_path,
            campaign_log_path=Path(log_path),
            job_directories=directories,
            decision_path=None,
        )
        publish(bundles.resume)
        cadence.mark_published(now=now, snapshot_sha256=snapshot_sha)

    def supervised_fold(**kwargs: object) -> FoldResult:
        job_directory = Path(kwargs["output_dir"])
        job = kwargs["job"]
        command = [
            sys.executable,
            "-m",
            "experiments.catboost_tabm_blend.training",
            "--job-id",
            job.job_id,  # type: ignore[union-attr]
            "--data-dir",
            str(kwargs["data_dir"]),
            "--output-dir",
            str(job_directory),
            "--contract-sha256",
            str(kwargs["contract_sha256"]),
            "--input-manifest-sha256",
            str(kwargs["input_manifest_sha256"]),
            "--code-sha256",
            str(kwargs["code_sha256"]),
            "--absolute-deadline",
            str(kwargs["absolute_deadline"]),
        ]
        environment = dict(os.environ)
        runtime_root = str(Path(__file__).resolve().parents[2])
        environment["PYTHONPATH"] = runtime_root + os.pathsep + environment.get("PYTHONPATH", "")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )

        def relay() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)

        relay_thread = threading.Thread(target=relay, name="catboost-log", daemon=True)
        relay_thread.start()
        killed_for_deadline = False
        while process.poll() is None:
            time.sleep(5.0)
            try:
                seal_active(job_directory)
            except (BlendColabError, OSError):
                # An in-progress CatBoost write can be transient. The prior
                # verified resume remains intact and the next poll retries.
                pass
            if time.time() >= wall_deadline:
                killed_for_deadline = True
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                break
        relay_thread.join(timeout=5)
        if killed_for_deadline:
            try:
                seal_active(job_directory, force=True)
            except (BlendColabError, OSError):
                pass
            return FoldResult(
                job_id=job.job_id,  # type: ignore[union-attr]
                status="budget_inconclusive",
                train_rows=0,
                valid_rows=0,
                brier=None,
                model_path=None,
                predictions_path=None,
                snapshot_path=(
                    job_directory / "experiment.cbsnapshot"
                    if (job_directory / "experiment.cbsnapshot").is_file()
                    else None
                ),
                elapsed_seconds=0.0,
                failure="absolute session deadline reached",
            )
        result_path = job_directory / "worker_result.json"
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise BlendColabError(
                f"CatBoost worker exited without a valid result: {process.returncode}"
            ) from error
        if payload.get("job_id") != job.job_id:  # type: ignore[union-attr]
            raise BlendColabError("CatBoost worker result identity differs")
        return FoldResult(
            job_id=payload["job_id"],
            status=payload["status"],
            train_rows=int(payload["train_rows"]),
            valid_rows=int(payload["valid_rows"]),
            brier=None if payload["brier"] is None else float(payload["brier"]),
            model_path=(
                None if payload["model"] is None else job_directory / payload["model"]
            ),
            predictions_path=(
                None
                if payload["predictions"] is None
                else job_directory / payload["predictions"]
            ),
            snapshot_path=(
                None
                if payload["snapshot"] is None
                else job_directory / payload["snapshot"]
            ),
            elapsed_seconds=float(payload["elapsed_seconds"]),
            failure=payload["failure"],
        )

    try:
        result = run_campaign(
            verified_input=verified_input,
            verified_stage_c=verified_stage_c,
            output_dir=Path(output_dir),
            resume_bundle=resume_bundle,
            absolute_deadline=wall_deadline,
            on_verified_resume=publish,
            runtime=supervised_fold,
        )
    except Exception:
        if latest is not None:
            on_verified_resume(latest)
        raise
    source_log = Path(output_dir) / "blend_campaign.log"
    if source_log.is_file() and source_log != Path(log_path):
        shutil.copyfile(source_log, log_path)
    return result
