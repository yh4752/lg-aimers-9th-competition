from __future__ import annotations

import base64
import gzip
from hashlib import sha256
import io
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/tabm_campaign/KAGGLE_CELL.py"


def _source_paths() -> list[Path]:
    paths: list[Path] = []
    for directory in (
        ROOT / "experiments/tabm_campaign",
        ROOT / "experiments/independent_dl",
        ROOT / "competition_rules",
    ):
        paths.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".json", ".txt"}
            and path.name != "KAGGLE_CELL.py"
            and "__pycache__" not in path.parts
        )
    paths.append(ROOT / "reports/rules/2026-08-13-policy-review.json")
    return sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix())


def _archive_bytes() -> bytes:
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in _source_paths():
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            info = tarfile.TarInfo(relative)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", compresslevel=9, mtime=0, filename="") as stream:
        stream.write(tar_buffer.getvalue())
    return compressed.getvalue()


def render() -> bytes:
    archive = _archive_bytes()
    encoded = base64.b64encode(archive).decode("ascii")
    digest = sha256(archive).hexdigest()
    source = f'''# Copy this entire file into one Kaggle notebook cell and use Save Version.
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
import traceback

EMBEDDED_RUNTIME_B64 = "{encoded}"
EXPECTED_RUNTIME_SHA256 = "{digest}"
INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/tabm_champion_campaign")
CODE_ROOT = WORK_ROOT / "runtime"
OUTPUT_ROOT = WORK_ROOT / "outputs"


def fail(stage: str, error: Exception) -> None:
    message = str(error).replace(" ", "_").replace("\\n", "_")
    print(f"TABM_CAMPAIGN_ERROR stage={{stage}} type={{type(error).__name__}} message={{message}}", flush=True)
    traceback.print_exc()


try:
    archive_bytes = base64.b64decode(EMBEDDED_RUNTIME_B64, validate=True)
    actual_sha = hashlib.sha256(archive_bytes).hexdigest()
    if actual_sha != EXPECTED_RUNTIME_SHA256:
        raise RuntimeError("embedded runtime SHA-256 mismatch")
    if CODE_ROOT.exists():
        shutil.rmtree(CODE_ROOT)
    CODE_ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        for member in archive.getmembers():
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or not member.isfile():
                raise RuntimeError(f"unsafe embedded member: {{member.name}}")
            destination = CODE_ROOT.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"cannot read embedded member: {{member.name}}")
            destination.write_bytes(extracted.read())
    print(f"CODE_READY sha256={{actual_sha}} size_bytes={{len(archive_bytes)}}", flush=True)

    requirements = CODE_ROOT / "experiments/tabm_campaign/requirements-kaggle.txt"
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)],
        capture_output=True,
        text=True,
    )
    if install.returncode:
        raise RuntimeError(f"dependency installation failed: {{install.stderr[-2000:]}}")
    import importlib.metadata

    print(
        "DEPENDENCIES_READY "
        f"tabm={{importlib.metadata.version('tabm')}} "
        f"rtdl_num_embeddings={{importlib.metadata.version('rtdl-num-embeddings')}}",
        flush=True,
    )

    train_files = sorted(path for path in INPUT_ROOT.rglob("train.csv") if path.is_file())
    if len(train_files) != 1:
        raise RuntimeError(f"official train.csv candidate count must be 1; found={{len(train_files)}}")
    data_root = train_files[0].parent
    if "lg-aimers-9th-data" not in data_root.as_posix().lower():
        raise RuntimeError(f"official data root is not the expected lg-aimers-9th-data input: {{data_root}}")
    print(f"DATA_FOUND path={{data_root}}", flush=True)

    resume_files = sorted(
        path
        for path in INPUT_ROOT.rglob("tabm_search_stage_*_resume_bundle.zip")
        if path.is_file()
    )
    if len(resume_files) > 1:
        raise RuntimeError(f"compatible resume bundle count must be 0 or 1; found={{len(resume_files)}}")
    resume = resume_files[0] if resume_files else None
    print(f"RESUME_FOUND path={{resume}}", flush=True)

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    gpu_lines = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
    if gpu.returncode or len(gpu_lines) != 2:
        raise RuntimeError(f"T4 x2 is required; visible_gpus={{gpu_lines}}")
    print(f"GPU_READY device_count=2 status={{' | '.join(gpu_lines)}}", flush=True)

    sys.path.insert(0, str(CODE_ROOT))
    from experiments.tabm_campaign.runner import run_one_version

    result = run_one_version(data_root, OUTPUT_ROOT, resume_bundle=resume)
    print(
        f"BUNDLE_SUCCESS version={{result.version}} review={{result.bundles.review}} resume={{result.bundles.resume}}",
        flush=True,
    )
except Exception as error:
    fail("runtime", error)
    raise
'''
    return source.encode("utf-8")


def main() -> None:
    value = render()
    OUTPUT.write_bytes(value)
    print(f"rendered={OUTPUT} size_bytes={len(value)} sha256={sha256(value).hexdigest()}")


if __name__ == "__main__":
    main()
