"""Render the self-contained direct-upload Colab Stage C recovery cell."""

from __future__ import annotations

import base64
import gzip
from hashlib import sha256
import io
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/tabm_campaign/COLAB_STAGE_C_RECOVERY_CELL.py"


def _source_paths() -> list[Path]:
    paths: list[Path] = []
    excluded_cells = {"KAGGLE_CELL.py", "COLAB_STAGE_C_RECOVERY_CELL.py"}
    for directory in (ROOT / "experiments/tabm_campaign", ROOT / "competition_rules"):
        paths.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".json", ".txt"}
            and path.name not in excluded_cells
            and "__pycache__" not in path.parts
        )
    independent = ROOT / "experiments/independent_dl"
    paths.extend(
        independent / name
        for name in (
            "__init__.py",
            "features.py",
            "preprocessing.py",
            "progress.py",
            "training.py",
        )
    )
    paths.extend(sorted((independent / "models").glob("*.py")))
    paths.extend(sorted((independent / "feature_sources").glob("*.py")))
    paths.append(ROOT / "reports/rules/2026-08-13-policy-review.json")
    return sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix())


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
    digest = sha256(archive).hexdigest()
    source = f'''# Copy this entire file into one Colab cell and select a T4 GPU runtime.
from __future__ import annotations

import base64
import gc
import hashlib
import importlib.metadata
import io
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import traceback

from google.colab import files
from IPython.display import FileLink, display

UPLOAD_EMERGENCY_SNAPSHOT = False  # Set True only after a Colab VM reset.
EMBEDDED_RUNTIME_B64 = "{encoded}"
EXPECTED_RUNTIME_SHA256 = "{digest}"
EXPECTED_BASE_RESUME_SHA256 = "6f7cc5b3c8280551f266767f63e71667db05473846e262371803a1fe1272d806"
WORK_ROOT = Path("/content/tabm_stage_c_recovery")
INPUT_ROOT = WORK_ROOT / "inputs"
CODE_ROOT = WORK_ROOT / f"runtime_{{EXPECTED_RUNTIME_SHA256[:12]}}"
DATA_ARCHIVE = INPUT_ROOT / "lg-aimers-9th-data.zip"
BASE_RESUME = INPUT_ROOT / "tabm_search_stage_C_resume_bundle.zip"
DATA_ROOT = WORK_ROOT / "official_data"
EMERGENCY_INPUT_ROOT = INPUT_ROOT / "emergency"


def fail(stage: str, error: Exception) -> None:
    message = str(error).replace(" ", "_").replace("\\n", "_")
    print(
        f"COLAB_STAGE_C_ERROR stage={{stage}} type={{type(error).__name__}} message={{message}}",
        flush=True,
    )
    traceback.print_exc()


def extract_runtime(archive_bytes: bytes) -> None:
    temporary = CODE_ROOT.with_name(f".{{CODE_ROOT.name}}.extracting")
    if CODE_ROOT.is_dir():
        receipt = CODE_ROOT / ".runtime_sha256"
        if receipt.is_file() and receipt.read_text(encoding="utf-8").strip() == EXPECTED_RUNTIME_SHA256:
            return
        raise RuntimeError("existing embedded runtime receipt differs")
    if temporary.exists():
        if temporary.is_symlink() or not temporary.is_dir():
            raise RuntimeError("stale runtime extraction path is unsafe")
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        for member in archive.getmembers():
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or not member.isfile():
                raise RuntimeError(f"unsafe embedded member: {{member.name}}")
            destination = temporary.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"cannot read embedded member: {{member.name}}")
            destination.write_bytes(extracted.read())
    (temporary / ".runtime_sha256").write_text(
        EXPECTED_RUNTIME_SHA256 + "\\n", encoding="utf-8"
    )
    os.replace(temporary, CODE_ROOT)


def upload_one(expected_name: str, destination: Path) -> Path:
    print(f"COLAB_UPLOAD_REQUIRED filename={{expected_name}}", flush=True)
    uploaded = files.upload()
    if set(uploaded) != {{expected_name}}:
        raise RuntimeError(
            f"expected exactly {{expected_name}}; received={{sorted(uploaded)}}"
        )
    source = Path("/content") / expected_name
    if not source.is_file():
        raise RuntimeError(f"uploaded file was not materialized: {{expected_name}}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    uploaded.clear()
    del uploaded
    gc.collect()
    return destination


def upload_emergency() -> Path:
    print("COLAB_UPLOAD_REQUIRED filename=tabm_colab_emergency_epoch_*.zip", flush=True)
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise RuntimeError(f"expected exactly one emergency ZIP; received={{sorted(uploaded)}}")
    name = next(iter(uploaded))
    if not re.fullmatch(r"tabm_colab_emergency_epoch_[0-9]{{3}}_[0-9a-f]{{12}}[.]zip", name):
        raise RuntimeError(f"unexpected emergency snapshot filename: {{name}}")
    source = Path("/content") / name
    destination = EMERGENCY_INPUT_ROOT / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    uploaded.clear()
    del uploaded
    gc.collect()
    return destination


def dependency_versions_ready() -> bool:
    expected = {{"tabm": "0.0.3", "rtdl-num-embeddings": "0.0.12"}}
    try:
        return all(importlib.metadata.version(name) == version for name, version in expected.items())
    except importlib.metadata.PackageNotFoundError:
        return False


def request_download(path: Path, *, emergency: bool) -> None:
    resolved = path.resolve()
    allowed_root = (WORK_ROOT / "snapshots").resolve() if emergency else WORK_ROOT.resolve()
    if not resolved.is_file() or (resolved != allowed_root and allowed_root not in resolved.parents):
        raise RuntimeError(f"download path is outside the verified work root: {{resolved}}")
    display(FileLink(str(resolved)))
    files.download(str(resolved))
    marker = "EMERGENCY_DOWNLOAD_REQUESTED" if emergency else "COLAB_DOWNLOAD_REQUESTED"
    print(f"{{marker}} path={{resolved}}", flush=True)


stage = "setup"
try:
    os.chdir("/content")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    archive_bytes = base64.b64decode(EMBEDDED_RUNTIME_B64, validate=True)
    actual_runtime_sha = hashlib.sha256(archive_bytes).hexdigest()
    if actual_runtime_sha != EXPECTED_RUNTIME_SHA256:
        raise RuntimeError("embedded runtime SHA-256 mismatch")
    extract_runtime(archive_bytes)
    print(
        f"COLAB_CODE_READY sha256={{actual_runtime_sha}} size_bytes={{len(archive_bytes)}}",
        flush=True,
    )
    sys.path.insert(0, str(CODE_ROOT))
    from experiments.tabm_campaign.colab_recovery import (
        file_sha256,
        load_colab_contract,
        verify_and_extract_data_archive,
    )

    stage = "inputs"
    if not DATA_ARCHIVE.is_file():
        upload_one(DATA_ARCHIVE.name, DATA_ARCHIVE)
    if not BASE_RESUME.is_file():
        upload_one(BASE_RESUME.name, BASE_RESUME)
    if file_sha256(BASE_RESUME) != EXPECTED_BASE_RESUME_SHA256:
        raise RuntimeError("base Stage C resume SHA-256 differs")
    contract = load_colab_contract()
    verify_and_extract_data_archive(DATA_ARCHIVE, DATA_ROOT, contract)
    if UPLOAD_EMERGENCY_SNAPSHOT:
        upload_emergency()
    emergency_paths = sorted(EMERGENCY_INPUT_ROOT.glob("tabm_colab_emergency_epoch_*.zip"))
    emergency_paths.extend(
        path
        for path in sorted((WORK_ROOT / "snapshots").glob("tabm_colab_emergency_epoch_*.zip"))
        if path not in emergency_paths
    )
    print(
        f"COLAB_INPUTS_VERIFIED data={{DATA_ROOT}} resume_sha256={{file_sha256(BASE_RESUME)}} "
        f"emergency_count={{len(emergency_paths)}}",
        flush=True,
    )

    stage = "dependencies"
    if not dependency_versions_ready():
        requirements = CODE_ROOT / "experiments/tabm_campaign/requirements-kaggle.txt"
        install = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)],
            capture_output=True,
            text=True,
        )
        if install.returncode:
            raise RuntimeError(f"dependency installation failed: {{install.stderr[-2000:]}}")
    print(
        "COLAB_DEPENDENCIES_READY "
        f"tabm={{importlib.metadata.version('tabm')}} "
        f"rtdl_num_embeddings={{importlib.metadata.version('rtdl-num-embeddings')}}",
        flush=True,
    )

    stage = "gpu"
    import torch

    device_count = int(torch.cuda.device_count())
    gpu_name = str(torch.cuda.get_device_name(0)) if device_count else "none"
    if device_count != 1 or "T4" not in gpu_name:
        raise RuntimeError(f"exactly one T4 GPU is required; device_count={{device_count}} name={{gpu_name}}")
    print(f"COLAB_GPU_READY device_count=1 name={{gpu_name}}", flush=True)

    stage = "campaign"
    command = [
        sys.executable,
        "-m",
        "experiments.tabm_campaign.colab_recovery",
        "run",
        "--data-dir", str(DATA_ROOT),
        "--base-resume", str(BASE_RESUME),
        "--work-root", str(WORK_ROOT),
        "--runtime-sha256", EXPECTED_RUNTIME_SHA256,
        "--gpu-count", "1",
    ]
    for emergency_path in emergency_paths:
        command.extend(["--emergency", str(emergency_path)])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(CODE_ROOT) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
        start_new_session=True,
    )
    requested_downloads = set()

    def handle_line(line: str) -> None:
        print(line.rstrip(), flush=True)
        match = re.search(r"(?:EMERGENCY_SNAPSHOT_READY|COLAB_DELIVERY_READY).*?path=(\\S+)", line)
        if match is None:
            return
        path = Path(match.group(1))
        key = str(path.resolve())
        if key in requested_downloads:
            return
        emergency = "EMERGENCY_SNAPSHOT_READY" in line
        request_download(path, emergency=emergency)
        requested_downloads.add(key)

    assert process.stdout is not None
    try:
        for line in process.stdout:
            handle_line(line)
    except KeyboardInterrupt:
        print("COLAB_INTERRUPT_RECEIVED requesting_epoch_boundary=true", flush=True)
        os.killpg(process.pid, signal.SIGINT)
        for line in process.stdout:
            handle_line(line)
    returncode = process.wait(timeout=210)
    if returncode:
        raise RuntimeError(f"Colab Stage C process failed returncode={{returncode}}")
except Exception as error:
    fail(stage, error)
    raise
'''
    return source.encode("utf-8")


def main() -> None:
    value = render()
    OUTPUT.write_bytes(value)
    print(
        f"rendered={OUTPUT} size_bytes={len(value)} sha256={sha256(value).hexdigest()}"
    )


if __name__ == "__main__":
    main()
