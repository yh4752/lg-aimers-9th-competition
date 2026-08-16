from __future__ import annotations

import base64
import gzip
from hashlib import sha256
import io
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tabm_campaign.row_feature_runtime import (
    STAGE_P_RUNTIME_PYTHON_MEMBERS,
)


OUTPUT = ROOT / "experiments/tabm_campaign/COLAB_ROW_FEATURE_PROXY_CELL.py"
_LIMIT_BYTES = 1_000_000


def _source_paths() -> list[Path]:
    paths = [ROOT / member for member in STAGE_P_RUNTIME_PYTHON_MEMBERS]
    paths.extend(
        (
            ROOT / "experiments/tabm_campaign/configs/row_feature_proxy_v1.json",
            ROOT / "experiments/tabm_campaign/requirements-kaggle.txt",
        )
    )
    result = sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix())
    if any(not path.is_file() for path in result):
        raise RuntimeError("embedded runtime inventory contains a missing source")
    return result


def _archive_bytes() -> bytes:
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in _source_paths():
            relative = path.relative_to(ROOT).as_posix()
            value = path.read_bytes()
            info = tarfile.TarInfo(relative)
            info.size = len(value)
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(value))
    compressed = io.BytesIO()
    with gzip.GzipFile(
        fileobj=compressed,
        mode="wb",
        compresslevel=9,
        mtime=0,
        filename="",
    ) as stream:
        stream.write(tar_buffer.getvalue())
    return compressed.getvalue()


def render() -> bytes:
    archive = _archive_bytes()
    encoded = base64.b64encode(archive).decode("ascii")
    runtime_sha = sha256(archive).hexdigest()
    source = f'''# Copy this entire file into one Colab cell and select a T4 or better GPU.
from __future__ import annotations

import base64
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

SESSION_DEADLINE = time.time() + 10800
NEW_JOB_GUARD_SECONDS = 900

import google.colab.files

EMBEDDED_RUNTIME_B64 = "{encoded}"
EXPECTED_RUNTIME_SHA256 = "{runtime_sha}"
WORK_ROOT = Path("/content/tabm_row_feature_proxy")
CODE_ROOT = WORK_ROOT / f"runtime_{{EXPECTED_RUNTIME_SHA256[:12]}}"


def create_run_root(work_root: Path, run_id: str) -> Path:
    if Path(run_id).name != run_id or run_id in {{"", ".", ".."}}:
        raise RuntimeError("run identifier is unsafe")
    root = work_root / "runs" / run_id
    root.mkdir(parents=True, exist_ok=False)
    return root


RUN_ID = f"{{time.time_ns()}}_{{os.getpid()}}"
RUN_ROOT = create_run_root(WORK_ROOT, RUN_ID)
INPUT_ROOT = RUN_ROOT / "uploads"
DATA_ROOT = RUN_ROOT / "official_data"
OUTPUT_ROOT = RUN_ROOT / "stage_P"
SNAPSHOT_ROOT = RUN_ROOT / "snapshots"
LOG_PATH = RUN_ROOT / "row_feature_proxy.log"
DELIVERY_PATH = RUN_ROOT / "tabm_row_feature_stage_P_delivery.zip"
PROGRESS_MARKERS = ("JOB_START", "TRAINING_PROGRESS", "EPOCH_CHECKPOINTED")
latest_verified = [None]


def fail(stage: str, error: Exception) -> None:
    message = str(error).replace(" ", "_").replace("\\n", "_")
    print(
        f"ROW_FEATURE_ERROR stage={{stage}} type={{type(error).__name__}} message={{message}}",
        flush=True,
    )
    traceback.print_exc()


def verify_extracted_runtime(archive_bytes: bytes) -> None:
    if CODE_ROOT.is_symlink() or not CODE_ROOT.is_dir():
        raise RuntimeError("existing runtime path is unsafe")
    expected = {{}}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as runtime:
        for member in runtime.getmembers():
            stream = runtime.extractfile(member)
            if stream is None:
                raise RuntimeError(f"cannot read embedded runtime member: {{member.name}}")
            expected[member.name] = hashlib.sha256(stream.read()).hexdigest()
    observed = {{}}
    for path in CODE_ROOT.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise RuntimeError(f"existing runtime member is unsafe: {{path}}")
        if path.is_file():
            name = path.relative_to(CODE_ROOT).as_posix()
            observed[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise RuntimeError("existing runtime inventory or hash differs")


def extract_runtime(archive_bytes: bytes) -> None:
    temporary = CODE_ROOT.with_name(f".{{CODE_ROOT.name}}-{{RUN_ID}}-extracting")
    if CODE_ROOT.exists() or CODE_ROOT.is_symlink():
        verify_extracted_runtime(archive_bytes)
        return
    if temporary.exists():
        if temporary.is_symlink() or not temporary.is_dir():
            raise RuntimeError("temporary runtime path is unsafe")
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as runtime:
        members = runtime.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or len(names) > 256:
            raise RuntimeError("embedded runtime inventory is invalid")
        if sum(member.size for member in members) > 16 * 1024 * 1024:
            raise RuntimeError("embedded runtime exceeds its extraction limit")
        for member in members:
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not member.isfile()
                or member.size > 4 * 1024 * 1024
            ):
                raise RuntimeError(f"unsafe embedded runtime member: {{member.name}}")
            destination = temporary.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            stream = runtime.extractfile(member)
            if stream is None:
                raise RuntimeError(f"cannot read embedded runtime member: {{member.name}}")
            with destination.open("xb") as output:
                shutil.copyfileobj(stream, output, length=1024 * 1024)
    os.replace(temporary, CODE_ROOT)


def upload_archives() -> list[Path]:
    print("ROW_FEATURE_UPLOAD_REQUIRED data_zip=1 stage_P_resume_zip=0_or_1", flush=True)
    uploaded = google.colab.files.upload()
    if len(uploaded) not in {{1, 2}}:
        raise RuntimeError(f"expected one data ZIP and optional Stage P resume; received={{len(uploaded)}}")
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, value in uploaded.items():
        if Path(name).name != name or not name.lower().endswith(".zip"):
            raise RuntimeError(f"uploaded input must be a flat ZIP filename: {{name}}")
        destination = INPUT_ROOT / name
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"duplicate uploaded path: {{name}}")
        destination.write_bytes(value)
        paths.append(destination)
    uploaded.clear()
    return paths


def request_download(path: Path, *, enforce_deadline: bool = True) -> None:
    if enforce_deadline:
        remaining_seconds()
    if path.is_symlink() or not path.is_file() or WORK_ROOT.resolve() not in path.resolve().parents:
        raise RuntimeError(f"download path is not verified under the work root: {{path}}")
    google.colab.files.download(str(path))
    print(f"ROW_FEATURE_DOWNLOAD_REQUESTED path={{path}}", flush=True)


def remember_snapshot(snapshot) -> None:
    request_download(snapshot.path)
    latest_verified[0] = snapshot.path


def remember_uploaded_resume(path: Path) -> None:
    latest_verified[0] = path
    print(f"ROW_FEATURE_UPLOADED_RESUME_READY path={{path}}", flush=True)


def publish_latest_verified_resume() -> None:
    path = latest_verified[0]
    if path is not None:
        request_download(Path(path), enforce_deadline=False)


def remaining_seconds() -> float:
    remaining = SESSION_DEADLINE - time.time()
    if remaining <= 0:
        raise TimeoutError("absolute 10800-second Colab deadline expired")
    return remaining


stage = "code"
try:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    archive_bytes = base64.b64decode(EMBEDDED_RUNTIME_B64, validate=True)
    actual_runtime_sha = hashlib.sha256(archive_bytes).hexdigest()
    if actual_runtime_sha != EXPECTED_RUNTIME_SHA256:
        raise RuntimeError("embedded runtime SHA-256 mismatch")
    extract_runtime(archive_bytes)
    sys.path.insert(0, str(CODE_ROOT))
    print(
        f"ROW_FEATURE_CODE_READY sha256={{actual_runtime_sha}} size_bytes={{len(archive_bytes)}}",
        flush=True,
    )

    from experiments.tabm_campaign.row_feature_colab import (
        build_delivery,
        classify_and_verify_uploads,
        register_verified_uploaded_resume,
        run_supervised_stage,
        verify_delivery,
    )
    from experiments.tabm_campaign.row_feature_contracts import (
        load_row_feature_proxy_contract,
        row_feature_contract_sha256,
    )
    from experiments.tabm_campaign.row_feature_proxy import _code_sha256

    contract = load_row_feature_proxy_contract()
    contract_sha = row_feature_contract_sha256()
    code_sha = _code_sha256()
    if (
        contract.budget.wall_seconds != 10800
        or contract.budget.new_job_guard_seconds != NEW_JOB_GUARD_SECONDS
    ):
        raise RuntimeError("embedded Stage P deadline contract differs")
    stage = "inputs"
    remaining_seconds()
    upload_paths = upload_archives()
    verified_input, resume_path = classify_and_verify_uploads(
        upload_paths,
        data_destination=DATA_ROOT,
        expected_train_sha256=contract.official_train_sha256,
        campaign_config_sha256=contract_sha,
        expected_code_sha256=code_sha,
    )
    if resume_path is not None:
        register_verified_uploaded_resume(
            resume_path,
            expected_contract_sha256=contract_sha,
            expected_code_sha256=code_sha,
            verified_input=verified_input,
            on_verified_resume=remember_uploaded_resume,
        )
    print(
        f"ROW_FEATURE_INPUTS_VERIFIED manifest_sha256={{verified_input.input_manifest_sha256}} "
        f"resume={{resume_path if resume_path is not None else 'none'}}",
        flush=True,
    )

    stage = "dependencies"
    requirements = CODE_ROOT / "experiments/tabm_campaign/requirements-kaggle.txt"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)],
        check=True,
        timeout=remaining_seconds(),
    )
    print("ROW_FEATURE_DEPENDENCIES_READY", flush=True)

    stage = "gpu"
    remaining_seconds()
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("one T4 or better CUDA GPU is required")
    capability = torch.cuda.get_device_capability(0)
    if capability < (7, 5):
        raise RuntimeError(f"GPU compute capability is below T4: {{capability}}")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"ROW_FEATURE_GPU_READY device_count={{torch.cuda.device_count()}} name={{gpu_name}}", flush=True)
    print("ROW_FEATURE_STAGE_SELECTED version=P", flush=True)

    stage = "training"
    result, latest = run_supervised_stage(
        data_dir=verified_input.data_dir,
        output_dir=OUTPUT_ROOT,
        resume_bundle=resume_path,
        snapshot_dir=SNAPSHOT_ROOT,
        wall_deadline=SESSION_DEADLINE,
        snapshot_interval_seconds=600,
        on_verified_snapshot=remember_snapshot,
        log_path=LOG_PATH,
    )
    if latest is not None:
        latest_verified[0] = latest.path
    print(
        f"ROW_FEATURE_BUNDLE_SUCCESS version={{result.version}} review={{result.bundles.review}} "
        f"resume={{result.bundles.resume}}",
        flush=True,
    )

    stage = "delivery"
    delivery = build_delivery(
        review_bundle=result.bundles.review,
        resume_bundle=result.bundles.resume,
        log_path=LOG_PATH,
        output_path=DELIVERY_PATH,
        input_manifest_sha256=verified_input.input_manifest_sha256,
        embedded_runtime_sha256=EXPECTED_RUNTIME_SHA256,
        campaign_config_sha256=contract_sha,
        code_sha256=code_sha,
        check_deadline=remaining_seconds,
    )
    verified_delivery = verify_delivery(
        delivery,
        input_manifest_sha256=verified_input.input_manifest_sha256,
        embedded_runtime_sha256=EXPECTED_RUNTIME_SHA256,
        campaign_config_sha256=contract_sha,
        code_sha256=code_sha,
        check_deadline=remaining_seconds,
        temporary_dir=RUN_ROOT,
    )
    print(
        f"ROW_FEATURE_DELIVERY_READY path={{verified_delivery.path}} sha256={{verified_delivery.sha256}}",
        flush=True,
    )
    request_download(delivery)
except Exception as error:
    fail(stage, error)
    try:
        publish_latest_verified_resume()
    except Exception as recovery_error:
        fail("emergency_resume", recovery_error)
    raise
'''
    value = source.encode("utf-8")
    if len(value) >= _LIMIT_BYTES:
        raise RuntimeError(
            f"generated Colab cell must be below {_LIMIT_BYTES} bytes; size={len(value)}"
        )
    return value


def main() -> None:
    value = render()
    OUTPUT.write_bytes(value)
    print(
        f"ROW_FEATURE_CELL_READY path={OUTPUT.resolve()} "
        f"sha256={sha256(value).hexdigest()} size_bytes={len(value)}"
    )


if __name__ == "__main__":
    main()
