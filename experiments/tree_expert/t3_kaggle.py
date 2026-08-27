from __future__ import annotations

import base64
from dataclasses import dataclass
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile
from zipfile import BadZipFile, ZipFile


class T3KaggleError(ValueError):
    pass


@dataclass(frozen=True)
class DiscoveredT3Inputs:
    t3_input: Path
    resume: Path | None


_RUNTIME_MEMBERS = (
    "experiments/tree_expert/__init__.py",
    "experiments/tree_expert/contracts.py",
    "experiments/tree_expert/features.py",
    "experiments/tree_expert/t3_artifacts.py",
    "experiments/tree_expert/t3_contract.json",
    "experiments/tree_expert/t3_contracts.py",
    "experiments/tree_expert/t3_decisions.py",
    "experiments/tree_expert/t3_diagnostics.py",
    "experiments/tree_expert/t3_full_fit.py",
    "experiments/tree_expert/t3_inference.py",
    "experiments/tree_expert/t3_inputs.py",
    "experiments/tree_expert/t3_kaggle.py",
    "experiments/tree_expert/t3_runner.py",
    "experiments/tree_expert/t3_state.py",
    "experiments/tree_expert/t3_temporal.py",
    "experiments/tree_expert/t3_training.py",
    "experiments/temporal_portfolio/__init__.py",
    "experiments/temporal_portfolio/seasonal_features.py",
    "experiments/temporal_portfolio/trackman_pitcher.py",
    "experiments/independent_dl/__init__.py",
    "experiments/independent_dl/feature_sources/__init__.py",
    "experiments/independent_dl/feature_sources/seasonal.py",
    "experiments/independent_dl/feature_sources/trackman.py",
)


def runtime_member_names() -> tuple[str, ...]:
    return _RUNTIME_MEMBERS


def _project_root(root: Path | None = None) -> Path:
    return Path(__file__).resolve().parents[2] if root is None else Path(root)


def runtime_archive(root: Path | None = None) -> bytes:
    project = _project_root(root)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name in _RUNTIME_MEMBERS:
                path = project / name
                if not path.is_file():
                    raise T3KaggleError(f"runtime member is missing: {name}")
                payload = path.read_bytes()
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mode = 0o644
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _kind_from_directory(path: Path) -> str | None:
    try:
        value = json.loads((path / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value.get("artifact_kind") if type(value) is dict else None


def _kind_from_zip(path: Path) -> str | None:
    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
            if names.count("manifest.json") != 1:
                return None
            value = json.loads(archive.read("manifest.json"))
            return value.get("artifact_kind") if type(value) is dict else None
    except (OSError, BadZipFile, KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def discover_t3_inputs(root: Path) -> DiscoveredT3Inputs:
    source = Path(root)
    candidates: dict[str, set[Path]] = {
        "tree_expert_t3_input_v1": set(), "tree_expert_t3_resume_v1": set(),
    }
    for manifest in source.rglob("manifest.json"):
        kind = _kind_from_directory(manifest.parent)
        if kind in candidates:
            candidates[kind].add(manifest.parent)
    for archive in source.rglob("*.zip"):
        kind = _kind_from_zip(archive)
        if kind in candidates:
            candidates[kind].add(archive)
    inputs = sorted(candidates["tree_expert_t3_input_v1"])
    resumes = sorted(candidates["tree_expert_t3_resume_v1"])
    if len(inputs) != 1:
        raise T3KaggleError(f"T3 input count must be one; found={len(inputs)}")
    if len(resumes) > 1:
        raise T3KaggleError(f"resume count must be zero or one; found={len(resumes)}")
    return DiscoveredT3Inputs(inputs[0], resumes[0] if resumes else None)


def build_t3_kaggle_cell(output: Path, root: Path | None = None) -> Path:
    payload = runtime_archive(root)
    encoded = base64.b64encode(payload).decode("ascii")
    digest = sha256(payload).hexdigest()
    source = f'''from __future__ import annotations
import base64, hashlib, importlib.metadata, io, json, os, shutil, subprocess, sys, tarfile, time, traceback
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

RUNTIME_B64 = "{encoded}"
RUNTIME_SHA256 = "{digest}"
STAGE = "setup"

def safe_extract(payload, destination):
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) > 256 or sum(item.size for item in members) > 8 * 1024 * 1024:
            raise RuntimeError("runtime archive exceeds limit")
        for item in members:
            path = PurePosixPath(item.name)
            if item.isdir() or item.issym() or item.islnk() or path.is_absolute() or ".." in path.parts:
                raise RuntimeError("unsafe runtime member")
        archive.extractall(destination, filter="data")

def materialize_resume(source, destination):
    if source is None or source.is_file():
        return source
    with ZipFile(destination, "w") as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            name = path.relative_to(source).as_posix()
            info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    return destination

try:
    RUN_ROOT = Path("/kaggle/working/tree_expert_t3_runs") / str(time.time_ns())
    RUNTIME_ROOT = RUN_ROOT / "runtime"
    RUNTIME_ROOT.mkdir(parents=True)
    runtime = base64.b64decode(RUNTIME_B64, validate=True)
    if hashlib.sha256(runtime).hexdigest() != RUNTIME_SHA256:
        raise RuntimeError("embedded runtime SHA-256 differs")
    safe_extract(runtime, RUNTIME_ROOT)
    sys.path.insert(0, str(RUNTIME_ROOT))
    print(f"TREE_T3_CODE_READY sha256={{RUNTIME_SHA256}} size_bytes={{len(runtime)}}", flush=True)

    STAGE = "dependencies"
    try:
        version = importlib.metadata.version("catboost")
    except importlib.metadata.PackageNotFoundError:
        version = "missing"
    if version != "1.2.10":
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost==1.2.10"])
    print(f"TREE_T3_DEPENDENCIES_READY catboost={{importlib.metadata.version('catboost')}}", flush=True)

    STAGE = "gpu"
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True,
    ).strip().splitlines()
    if len(names) != 2:
        raise RuntimeError(f"T4x2 required; found={{len(names)}}")
    print(f"TREE_T3_GPU_READY count=2 names={{' | '.join(names)}}", flush=True)

    STAGE = "inputs"
    from experiments.tree_expert.t3_kaggle import discover_t3_inputs
    from experiments.tree_expert.t3_inputs import verify_and_extract_t3_input, verify_official_data
    from experiments.tree_expert.t3_runner import run_t3_campaign
    discovered = discover_t3_inputs(Path("/kaggle/input"))
    official_roots = sorted({{path.parent for path in Path("/kaggle/input").rglob("train.csv") if (path.parent / "trackman_history.csv").is_file()}})
    if len(official_roots) != 1:
        raise RuntimeError(f"official data count must be one; found={{len(official_roots)}}")
    official = verify_official_data(official_roots[0])
    verified = verify_and_extract_t3_input(discovered.t3_input, RUN_ROOT / "verified_input")
    resume = materialize_resume(discovered.resume, RUN_ROOT / "uploaded_resume.zip")
    print(f"TREE_T3_INPUTS_VERIFIED input={{discovered.t3_input}} resume={{resume}}", flush=True)

    STAGE = "campaign"
    result = run_t3_campaign(
        verified, official, RUN_ROOT / "campaign", resume_bundle=resume,
        wall_deadline=time.monotonic() + 28_200,
    )
    final = Path("/kaggle/working/tree_expert_t3_handoff.zip")
    shutil.copy2(result.handoff_bundle, final)
    print(f"TREE_T3_SUCCESS handoff={{final}} acceptance={{result.acceptance_status}}", flush=True)
except Exception as error:
    print(f"TREE_T3_ERROR stage={{STAGE}} type={{type(error).__name__}} message={{str(error).replace(' ', '_')}}", flush=True)
    traceback.print_exc()
    snapshots = sorted(Path("/kaggle/working/tree_expert_t3_runs").rglob("tree_expert_t3_resume.zip"), key=lambda path: path.stat().st_mtime_ns)
    if snapshots:
        emergency = Path("/kaggle/working/tree_expert_t3_resume.zip")
        shutil.copy2(snapshots[-1], emergency)
        print(f"TREE_T3_RESUME_READY path={{emergency}}", flush=True)
'''
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8")
    return destination
