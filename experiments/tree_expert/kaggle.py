from __future__ import annotations

import base64
from gzip import GzipFile
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


_RUNTIME_MEMBERS = (
    "experiments/tree_expert/__init__.py",
    "experiments/tree_expert/e1_contract.json",
    "experiments/tree_expert/contracts.py",
    "experiments/tree_expert/inputs.py",
    "experiments/tree_expert/features.py",
    "experiments/tree_expert/failure_labels.py",
    "experiments/tree_expert/training.py",
    "experiments/tree_expert/metrics.py",
    "experiments/tree_expert/artifacts.py",
    "experiments/tree_expert/runner.py",
    "experiments/tree_expert/kaggle.py",
    "experiments/tree_expert/requirements-kaggle.txt",
    "experiments/tabm_campaign/artifacts.py",
    "experiments/tabm_campaign/colab_recovery.py",
    "experiments/temporal_portfolio/__init__.py",
    "experiments/temporal_portfolio/seasonal_features.py",
    "experiments/temporal_portfolio/trackman_pitcher.py",
    "experiments/independent_dl/__init__.py",
    "experiments/independent_dl/feature_sources/__init__.py",
    "experiments/independent_dl/feature_sources/seasonal.py",
    "experiments/independent_dl/feature_sources/trackman.py",
)
_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


def runtime_member_names() -> tuple[str, ...]:
    return _RUNTIME_MEMBERS


def classify_e1_input(path: Path) -> str | None:
    source = Path(path)
    try:
        if source.is_dir():
            for name in ("manifest.json", "handoff_manifest.json"):
                manifest = source / name
                if manifest.is_file():
                    return str(json.loads(manifest.read_text())["artifact_kind"])
            return None
        if source.is_file() and source.suffix.lower() == ".zip":
            with ZipFile(source, "r") as archive:
                for name in ("manifest.json", "handoff_manifest.json"):
                    if name in archive.namelist():
                        return str(json.loads(archive.read(name))["artifact_kind"])
    except Exception:
        return None
    return None


def _runtime_archive(root: Path) -> bytes:
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
        for name in _RUNTIME_MEMBERS:
            source = root / name
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"runtime source is absent: {name}")
            payload = source.read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
    compressed = io.BytesIO()
    with GzipFile(fileobj=compressed, mode="wb", mtime=0) as handle:
        handle.write(tar_buffer.getvalue())
    return compressed.getvalue()


def _template(encoded: str, digest: str) -> str:
    return f'''from __future__ import annotations
import base64, importlib.metadata, io, json, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

RUNTIME_B64 = "{encoded}"
RUNTIME_SHA256 = "{digest}"
WORK = Path("/kaggle/working/tree_expert_e1_runtime")
STAGE = "setup"
SUCCESS_PREFIX = "TREE_E1_HANDOFF_READY"

def safe_extract(payload: bytes, destination: Path) -> None:
    import hashlib
    if hashlib.sha256(payload).hexdigest() != RUNTIME_SHA256:
        raise RuntimeError("runtime_sha256_differs")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()) or not member.isfile():
                raise RuntimeError("unsafe_runtime_member")
        archive.extractall(destination, filter="data")

def zip_directory(source: Path, destination: Path) -> Path:
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            name = path.relative_to(source).as_posix()
            info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    return destination

def artifact_candidates(root: Path, kinds: set[str]) -> list[tuple[Path, str]]:
    from experiments.tree_expert.kaggle import classify_e1_input
    found: list[tuple[Path, str]] = []
    for manifest in sorted(root.rglob("manifest.json")):
        kind = classify_e1_input(manifest.parent)
        if kind in kinds:
            found.append((manifest.parent, kind))
    for archive in sorted(root.rglob("*.zip")):
        kind = classify_e1_input(archive)
        if kind in kinds:
            found.append((archive, kind))
    unique = {{str(path.resolve()): (path, kind) for path, kind in found}}
    return list(unique.values())

try:
    payload = base64.b64decode(RUNTIME_B64, validate=True)
    safe_extract(payload, WORK / "code")
    sys.path.insert(0, str(WORK / "code"))
    print(f"TREE_E1_CODE_READY sha256={{RUNTIME_SHA256}} size_bytes={{len(payload)}}", flush=True)

    STAGE = "dependencies"
    try:
        version = importlib.metadata.version("catboost")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version != "1.2.10":
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "catboost==1.2.10"], check=True)
    print("TREE_E1_DEPENDENCIES_READY catboost=1.2.10", flush=True)

    STAGE = "inputs"
    input_root = Path("/kaggle/input")
    data_roots = sorted({{p.parent for p in input_root.rglob("train.csv") if (p.parent / "trackman_history.csv").is_file()}})
    if len(data_roots) != 1:
        raise RuntimeError(f"official_data_root_count_must_be_1_found={{len(data_roots)}}")
    e1 = artifact_candidates(input_root, {{"tree_expert_e1_input_v1"}})
    if len(e1) != 1:
        raise RuntimeError(f"e1_input_count_must_be_1_found={{len(e1)}}")
    e1_path = e1[0][0]
    if e1_path.is_dir():
        e1_path = zip_directory(e1_path, WORK / "tree_expert_e1_input.zip")
    resumes = artifact_candidates(input_root, {{"tree_expert_e1_resume_v1", "tree_expert_e1_handoff_v1"}})
    if len(resumes) > 1:
        raise RuntimeError(f"resume_count_must_be_0_or_1_found={{len(resumes)}}")
    resume = None
    if resumes:
        resume_path, resume_kind = resumes[0]
        if resume_kind == "tree_expert_e1_handoff_v1":
            if resume_path.is_dir():
                resume = resume_path / "tree_expert_e1_resume.zip"
            else:
                with ZipFile(resume_path) as archive:
                    resume = WORK / "tree_expert_e1_resume.zip"
                    resume.write_bytes(archive.read("tree_expert_e1_resume.zip"))
        elif resume_path.is_dir():
            resume = zip_directory(resume_path, WORK / "tree_expert_e1_resume.zip")
        else:
            resume = resume_path

    from experiments.tree_expert.contracts import load_e1_contract
    from experiments.tree_expert.inputs import verify_and_extract_e1_input, verify_official_data
    contract = load_e1_contract()
    verified_data = verify_official_data(data_roots[0], contract)
    verified_input = verify_and_extract_e1_input(e1_path, WORK / "verified_e1", contract)
    print(f"TREE_E1_INPUTS_VERIFIED data={{verified_data.root}} resume={{resume}}", flush=True)

    STAGE = "gpu"
    import torch
    device_count = torch.cuda.device_count()
    if device_count != 2:
        raise RuntimeError(f"gpu_count_must_be_2_found={{device_count}}")
    print("TREE_E1_GPU_READY device_count=2", flush=True)

    STAGE = "campaign"
    from experiments.tree_expert.runner import run_e1_campaign
    result = run_e1_campaign(
        verified_data=verified_data,
        verified_input=verified_input,
        output_dir=Path("/kaggle/working/tree_expert_e1"),
        resume_bundle=resume,
        absolute_deadline=time.time() + 14400,
        gpu_ids=(0, 1),
    )
    print(f"TREE_E1_CAMPAIGN_DONE status={{result.status}}", flush=True)
except Exception as error:
    print(
        f"TREE_EXPERT_ERROR stage={{STAGE}} type={{type(error).__name__}} "
        f"message={{str(error).replace(' ', '_')}}",
        flush=True,
    )
    emergency = Path("/kaggle/working/tree_expert_e1/bundles/tree_expert_e1_handoff.zip")
    if emergency.is_file():
        print(f"TREE_E1_EMERGENCY_HANDOFF path={{emergency}}", flush=True)
    raise
'''


def build_e1_kaggle_cell(output: Path, root: Path | None = None) -> Path:
    repository = Path(__file__).resolve().parents[2] if root is None else Path(root)
    archive = _runtime_archive(repository)
    source = _template(base64.b64encode(archive).decode("ascii"), sha256(archive).hexdigest())
    if len(source.encode("utf-8")) >= 1_000_000:
        raise ValueError("generated Kaggle cell reaches 1MB")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8")
    return destination
