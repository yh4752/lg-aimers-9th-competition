"""Build the self-contained Kaggle launcher for the temporal T1 stage."""
from __future__ import annotations

import base64
import gzip
import importlib.util
import io
import os
from pathlib import Path
import tarfile


class PlatformError(RuntimeError):
    pass


_TEMPLATE = '''from __future__ import annotations
import base64, io, json, os, shutil, sys, tarfile, time, traceback
from pathlib import Path
from zipfile import ZipFile

RUNTIME_B64 = "__RUNTIME_B64__"
WORK_ROOT = Path("/kaggle/working/temporal_portfolio_output")
RUNTIME_ROOT = Path("/kaggle/working/temporal_portfolio_runtime")
INPUT_ROOT = Path("/kaggle/input")

def unpack_runtime():
    if RUNTIME_ROOT.exists():
        shutil.rmtree(RUNTIME_ROOT)
    RUNTIME_ROOT.mkdir(parents=True)
    payload = base64.b64decode(RUNTIME_B64, validate=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if member.name.startswith("/") or ".." in parts or not member.isfile():
                raise RuntimeError("T1 runtime archive contains an unsafe member")
        archive.extractall(RUNTIME_ROOT)
    sys.path.insert(0, str(RUNTIME_ROOT))
    print(f"T1_CODE_READY size_bytes={len(payload)}", flush=True)

def find_data_root():
    required = {"train.csv", "test.csv", "trackman_history.csv", "sample_submission.csv"}
    found = []
    for current, directories, files in os.walk(INPUT_ROOT):
        directories[:] = sorted(directories)
        if set(files) == required:
            found.append(Path(current))
    if len(found) != 1:
        raise RuntimeError(f"official_data_candidate_count_must_be_1 found={len(found)}")
    return found[0]

def artifact_kind(path):
    try:
        if path.is_file() and path.suffix.casefold() == ".zip":
            with ZipFile(path) as archive:
                return json.loads(archive.read("manifest.json")).get("artifact_kind")
        if path.is_dir() and (path / "manifest.json").is_file():
            return json.loads((path / "manifest.json").read_text()).get("artifact_kind")
    except Exception:
        return None
    return None

def find_resume():
    candidates = []
    for path in sorted(INPUT_ROOT.rglob("*")):
        if artifact_kind(path) == "temporal_t1_resume_v1":
            candidates.append(path)
    nested = {path for parent in candidates if parent.is_dir() for path in candidates if path != parent and parent in path.parents}
    candidates = [path for path in candidates if path not in nested]
    if len(candidates) > 1:
        raise RuntimeError(f"T1_resume_candidate_count_must_be_0_or_1 found={len(candidates)}")
    return candidates[0] if candidates else None

def package_emergency(jobs_root, destination):
    from experiments.temporal_portfolio.contracts import build_stage_jobs, load_contract
    from experiments.temporal_portfolio.t1_artifacts import verify_compact_result, write_t1_bundles
    from experiments.temporal_portfolio.worker import verify_worker_result
    completed, pending = [], []
    for spec in build_stage_jobs(load_contract(), "T1"):
        root = jobs_root / spec.job_id
        try:
            if (root / "worker_result.json").is_file():
                verify_worker_result(root)
                completed.append(spec.job_id)
            elif (root / "compact_result.json").is_file():
                verify_compact_result(root)
                completed.append(spec.job_id)
            else:
                pending.append(spec.job_id)
        except Exception:
            pending.append(spec.job_id)
    return write_t1_bundles(destination, jobs_root=jobs_root, completed=tuple(completed), pending=tuple(pending), failed=())

unpack_runtime()
stage = "inputs"
try:
    import tabm, rtdl_num_embeddings
    from experiments.temporal_portfolio.inputs import verify_official_data
    from experiments.temporal_portfolio.t1_artifacts import restore_t1_resume_source, write_t1_bundles
    from experiments.temporal_portfolio.t1_runner import run_t1_stage

    data_root = find_data_root()
    resume = find_resume()
    verified = verify_official_data(data_root)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    working_jobs = WORK_ROOT / "jobs"
    if resume is not None and not (working_jobs.is_dir() and any(working_jobs.iterdir())):
        restore_t1_resume_source(resume, WORK_ROOT)
        print(f"T1_RESUME_READY source={resume}", flush=True)
    elif resume is not None:
        print(f"T1_WORKING_CACHE_REUSED resume_source={resume}", flush=True)
    else:
        print("T1_RESUME_READY source=None", flush=True)
    stage = "training"
    result = run_t1_stage(
        verified=verified,
        output_root=WORK_ROOT,
        deadline=time.time() + 21600,
    )
    stage = "bundles"
    bundles = write_t1_bundles(
        WORK_ROOT / "bundles",
        jobs_root=WORK_ROOT / "jobs",
        completed=result.completed,
        pending=result.pending,
        failed=result.failed,
    )
    print(f"T1_HANDOFF_READY status={result.status} review={bundles.review} resume={bundles.resume}", flush=True)
except Exception as error:
    print(f"T1_PORTFOLIO_ERROR stage={stage} type={type(error).__name__} message={str(error).replace(' ', '_')}", flush=True)
    traceback.print_exc()
    try:
        emergency = package_emergency(WORK_ROOT / "jobs", WORK_ROOT / "emergency")
        print(f"T1_EMERGENCY_READY review={emergency.review} resume={emergency.resume}", flush=True)
    except Exception as bundle_error:
        print(f"T1_EMERGENCY_ERROR type={type(bundle_error).__name__} message={str(bundle_error).replace(' ', '_')}", flush=True)
    raise
'''


def build_kaggle_cell(output: str | Path) -> Path:
    archive = _runtime_archive()
    rendered = _TEMPLATE.replace(
        "__RUNTIME_B64__", base64.b64encode(archive).decode("ascii")
    )
    data = rendered.encode("utf-8")
    if len(data) >= 1_000_000:
        raise PlatformError("generated Kaggle cell exceeds one megabyte")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, destination)
    return destination


def _runtime_archive() -> bytes:
    root = Path(__file__).resolve().parents[2]
    members: list[tuple[str, bytes]] = []
    for relative_root in (
        Path("experiments/temporal_portfolio"),
        Path("experiments/independent_dl"),
    ):
        for path in sorted((root / relative_root).rglob("*")):
            if (
                path.is_file()
                and not path.is_symlink()
                and "__pycache__" not in path.parts
                and path.name not in {
                    "KAGGLE_CELL.py",
                    "T2A_KAGGLE_CELL.py",
                    "COLAB_RECOVERY_CELL.py",
                }
                and path.suffix in {".py", ".json"}
            ):
                members.append((path.relative_to(root).as_posix(), path.read_bytes()))
    experiments_init = root / "experiments" / "__init__.py"
    if experiments_init.is_file():
        members.append(("experiments/__init__.py", experiments_init.read_bytes()))
    for module_name in ("tabm", "rtdl_num_embeddings"):
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise PlatformError(f"runtime dependency source is missing: {module_name}")
        source = Path(spec.origin)
        members.append((source.name, source.read_bytes()))
    members.sort(key=lambda item: item[0])
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, data in members:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mtime = 0
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(data))
    return raw.getvalue()
