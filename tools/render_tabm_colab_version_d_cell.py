"""Render the self-contained direct-upload Colab Version D cell."""

from __future__ import annotations

import base64
import gzip
from hashlib import sha256
import io
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/tabm_campaign/COLAB_VERSION_D_REVIEW_CELL.py"


def _source_paths() -> list[Path]:
    campaign = ROOT / "experiments/tabm_campaign"
    paths = [
        campaign / name
        for name in (
            "__init__.py",
            "artifacts.py",
            "colab_version_d.py",
            "contracts.py",
            "decisions.py",
            "dependency_probe.py",
            "final_review.py",
            "final_training.py",
            "inference_runtime.py",
            "metrics.py",
            "model_state.py",
            "requirements-kaggle.txt",
            "runner.py",
            "version_d.py",
            "version_d_contract.json",
        )
    ]
    rules = ROOT / "competition_rules"
    paths.extend(
        path
        for path in rules.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".json"}
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
    paths.append(ROOT / "reports/rules/2026-08-14-policy-review.json")
    return sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix())


def _archive_bytes() -> bytes:
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in _source_paths():
            value = path.read_bytes()
            info = tarfile.TarInfo(path.relative_to(ROOT).as_posix())
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
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

from google.colab import files
from IPython.display import FileLink, display

RECOVERY_MODE = "none"  # Use "emergency" or "frozen" only after a VM reset.
EMBEDDED_RUNTIME_B64 = "{encoded}"
EXPECTED_RUNTIME_SHA256 = "{digest}"
EXPECTED_SUCCESS_MARKER = "VERSION_D_DELIVERY_READY"
WORK_ROOT = Path("/content/tabm_version_d")
INPUT_ROOT = WORK_ROOT / "inputs"
CODE_ROOT = WORK_ROOT / f"runtime_{{EXPECTED_RUNTIME_SHA256[:12]}}"
DATA_ARCHIVE = INPUT_ROOT / "lg-aimers-9th-data.zip"
STAGE_C_DELIVERY = INPUT_ROOT / "tabm_colab_stage_C_delivery.zip"
RECOVERY_ROOT = INPUT_ROOT / "recovery"
LOG_PATH = WORK_ROOT / "version_d.log"


class Tee:
    def __init__(self, primary, log):
        self.primary = primary
        self.log = log

    def write(self, value):
        self.primary.write(value)
        self.log.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.primary.flush()
        self.log.flush()

    def isatty(self):
        return self.primary.isatty()


def fail(stage, error):
    message = str(error).replace(" ", "_").replace("\\n", "_")
    print(
        f"VERSION_D_ERROR stage={{stage}} type={{type(error).__name__}} message={{message}}",
        flush=True,
    )
    traceback.print_exc()


def extract_runtime(archive_bytes):
    receipt = CODE_ROOT / ".runtime_sha256"
    if CODE_ROOT.is_dir():
        if receipt.is_file() and receipt.read_text(encoding="utf-8").strip() == EXPECTED_RUNTIME_SHA256:
            return
        raise RuntimeError("existing embedded runtime receipt differs")
    temporary = CODE_ROOT.with_name(f".{{CODE_ROOT.name}}.extracting")
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
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"cannot read embedded member: {{member.name}}")
            destination.write_bytes(stream.read())
    (temporary / ".runtime_sha256").write_text(
        EXPECTED_RUNTIME_SHA256 + "\\n", encoding="utf-8"
    )
    os.replace(temporary, CODE_ROOT)


def upload_one(expected_name, destination):
    print(f"VERSION_D_UPLOAD_REQUIRED filename={{expected_name}}", flush=True)
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


def recovery_input():
    if RECOVERY_MODE == "none":
        return None
    if RECOVERY_MODE not in {{"emergency", "frozen"}}:
        raise RuntimeError(f"unknown RECOVERY_MODE: {{RECOVERY_MODE}}")
    RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)
    existing = list(RECOVERY_ROOT.glob("*.zip"))
    if len(existing) == 1:
        return existing[0]
    if existing:
        raise RuntimeError("multiple recovery archives already exist")
    print(f"VERSION_D_UPLOAD_REQUIRED recovery_mode={{RECOVERY_MODE}}", flush=True)
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise RuntimeError(f"expected exactly one recovery ZIP; received={{sorted(uploaded)}}")
    name = next(iter(uploaded))
    frozen_ok = RECOVERY_MODE == "frozen" and name == "tabm_version_D_frozen_model.zip"
    emergency_ok = RECOVERY_MODE == "emergency" and re.fullmatch(
        r"tabm_version_D_emergency_epoch_[0-9]{{3}}_[0-9a-f]{{12}}[.]zip", name
    )
    if not frozen_ok and not emergency_ok:
        raise RuntimeError(f"recovery filename differs: {{name}}")
    source = Path("/content") / name
    destination = RECOVERY_ROOT / name
    os.replace(source, destination)
    uploaded.clear()
    del uploaded
    gc.collect()
    return destination


def dependencies_ready():
    expected = {{"tabm": "0.0.3", "rtdl-num-embeddings": "0.0.12"}}
    try:
        return all(importlib.metadata.version(name) == value for name, value in expected.items())
    except importlib.metadata.PackageNotFoundError:
        return False


def venv_probe():
    root = Path(tempfile.mkdtemp(prefix="version_d_venv_probe_"))
    try:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "venv",
                "--system-site-packages",
                str(root / "probe"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def ensure_venv_ready():
    probe = venv_probe()
    if probe.returncode == 0:
        print("VERSION_D_VENV_READY mode=existing", flush=True)
        return
    print("VERSION_D_VENV_REPAIR_REQUIRED package=python3.12-venv", flush=True)
    commands = (
        ["apt-get", "update", "-qq"],
        ["apt-get", "install", "-y", "-qq", "python3.12-venv"],
    )
    for command in commands:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode:
            detail = (completed.stdout + "\\n" + completed.stderr)[-2000:]
            raise RuntimeError(f"venv repair failed: {{detail}}")
    probe = venv_probe()
    if probe.returncode:
        detail = (probe.stdout + "\\n" + probe.stderr)[-2000:]
        raise RuntimeError(f"venv remains unavailable after repair: {{detail}}")
    print("VERSION_D_VENV_READY mode=repaired", flush=True)


def runtime_versions():
    import numpy
    import pandas
    import torch

    return {{
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": str(torch.version.cuda),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "tabm": importlib.metadata.version("tabm"),
        "rtdl_num_embeddings": importlib.metadata.version("rtdl-num-embeddings"),
    }}


def request_download(path, kind):
    resolved = path.resolve()
    allowed = WORK_ROOT.resolve()
    if not resolved.is_file() or allowed not in resolved.parents:
        raise RuntimeError(f"download path is outside work root: {{resolved}}")
    display(FileLink(str(resolved)))
    files.download(str(resolved))
    print(f"VERSION_D_DOWNLOAD_REQUESTED kind={{kind}} path={{resolved}}", flush=True)


stage = "setup"
original_stdout = sys.stdout
original_stderr = sys.stderr
log_handle = None
try:
    os.chdir("/content")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    log_handle = LOG_PATH.open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(original_stdout, log_handle)
    sys.stderr = Tee(original_stderr, log_handle)
    archive_bytes = base64.b64decode(EMBEDDED_RUNTIME_B64, validate=True)
    actual_runtime_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    if actual_runtime_sha256 != EXPECTED_RUNTIME_SHA256:
        raise RuntimeError("embedded runtime SHA-256 differs")
    extract_runtime(archive_bytes)
    print(
        f"VERSION_D_CODE_READY sha256={{actual_runtime_sha256}} size_bytes={{len(archive_bytes)}}",
        flush=True,
    )

    stage = "uploads"
    if not DATA_ARCHIVE.is_file():
        upload_one(DATA_ARCHIVE.name, DATA_ARCHIVE)
    if not STAGE_C_DELIVERY.is_file():
        upload_one(STAGE_C_DELIVERY.name, STAGE_C_DELIVERY)
    recovery = recovery_input()

    stage = "dependencies"
    if not dependencies_ready():
        requirements = CODE_ROOT / "experiments/tabm_campaign/requirements-kaggle.txt"
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)],
            capture_output=True,
            text=True,
            timeout=480,
        )
        if completed.returncode:
            raise RuntimeError(f"dependency installation failed: {{completed.stderr[-2000:]}}")
    versions = runtime_versions()
    print(
        "VERSION_D_DEPENDENCIES_READY "
        + " ".join(f"{{name}}={{value}}" for name, value in sorted(versions.items())),
        flush=True,
    )

    stage = "venv"
    ensure_venv_ready()

    stage = "run"
    sys.path.insert(0, str(CODE_ROOT))
    from experiments.tabm_campaign.colab_version_d import run_version_d

    result = run_version_d(
        data_archive=DATA_ARCHIVE,
        stage_c_delivery=STAGE_C_DELIVERY,
        work_root=WORK_ROOT,
        recovery_archive=recovery,
        absolute_deadline=time.time() + 6600,
        runtime_sha256=EXPECTED_RUNTIME_SHA256,
        runtime_versions=versions,
        log_path=LOG_PATH,
        on_download=request_download,
    )
    print(f"VERSION_D_CELL_COMPLETE path={{result}}", flush=True)
except Exception as error:
    fail(stage, error)
    if log_handle is not None:
        log_handle.flush()
    if LOG_PATH.is_file():
        display(FileLink(str(LOG_PATH)))
    rescue = None
    frozen = WORK_ROOT / "snapshots/tabm_version_D_frozen_model.zip"
    emergency = sorted(WORK_ROOT.glob("snapshots/tabm_version_D_emergency_epoch_*.zip"))
    if frozen.is_file():
        rescue = frozen
    elif emergency:
        rescue = emergency[-1]
    if rescue is not None:
        try:
            request_download(rescue, "rescue")
        except Exception as rescue_error:
            print(f"VERSION_D_RESCUE_DOWNLOAD_ERROR message={{rescue_error}}", flush=True)
    raise
finally:
    if log_handle is not None:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()
'''
    return source.encode("utf-8")


def main() -> int:
    value = render()
    OUTPUT.write_bytes(value)
    print(
        f"COLAB_VERSION_D_CELL_READY path={OUTPUT} "
        f"sha256={sha256(value).hexdigest()} size_bytes={len(value)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
