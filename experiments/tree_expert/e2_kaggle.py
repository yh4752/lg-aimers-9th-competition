from __future__ import annotations

import base64
from dataclasses import dataclass
from gzip import GzipFile
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile
from typing import Mapping
from zipfile import ZipFile

from .e2_contracts import load_e2_contract


class E2KaggleError(ValueError):
    """Raised before a Kaggle E2 campaign can consume ambiguous inputs."""


_RUNTIME_MEMBERS = (
    "experiments/tree_expert/__init__.py",
    "experiments/tree_expert/e1_contract.json",
    "experiments/tree_expert/contracts.py",
    "experiments/tree_expert/inputs.py",
    "experiments/tree_expert/features.py",
    "experiments/tree_expert/failure_labels.py",
    "experiments/tree_expert/training.py",
    "experiments/tree_expert/artifacts.py",
    "experiments/tree_expert/e2_contract.json",
    "experiments/tree_expert/e2_contracts.py",
    "experiments/tree_expert/e2_inputs.py",
    "experiments/tree_expert/e2_baseline.py",
    "experiments/tree_expert/e2_decisions.py",
    "experiments/tree_expert/e2_training.py",
    "experiments/tree_expert/e2_full_fit.py",
    "experiments/tree_expert/e2_inference.py",
    "experiments/tree_expert/e2_artifacts.py",
    "experiments/tree_expert/e2_runner.py",
    "experiments/tree_expert/e2_kaggle.py",
    "experiments/tree_expert/e2_production.py",
    "experiments/tree_expert/requirements-kaggle.txt",
    "experiments/tabm_campaign/__init__.py",
    "experiments/tabm_campaign/artifacts.py",
    "experiments/tabm_campaign/colab_recovery.py",
    "experiments/tabm_campaign/cache.py",
    "experiments/independent_dl/__init__.py",
    "experiments/independent_dl/features.py",
    "experiments/independent_dl/preprocessing.py",
    "experiments/independent_dl/row_features.py",
    "experiments/independent_dl/training.py",
    "experiments/independent_dl/progress.py",
    "experiments/independent_dl/models/__init__.py",
    "experiments/independent_dl/models/common.py",
    "experiments/independent_dl/models/tabm.py",
    "experiments/independent_dl/feature_sources/__init__.py",
    "experiments/independent_dl/feature_sources/seasonal.py",
    "experiments/independent_dl/feature_sources/trackman.py",
    "experiments/temporal_portfolio/__init__.py",
    "experiments/temporal_portfolio/seasonal_features.py",
    "experiments/temporal_portfolio/trackman_pitcher.py",
)


@dataclass(frozen=True)
class DiscoveredInputs:
    official_data: Path
    e2_input: Path
    resume: Path | None


def runtime_member_names() -> tuple[str, ...]:
    return _RUNTIME_MEMBERS


def runtime_identity_sha256(root: Path) -> str:
    source = Path(root)
    digest = sha256()
    for name in _RUNTIME_MEMBERS:
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise E2KaggleError(f"runtime source is absent: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _file_sha(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_kind(path: Path) -> str | None:
    source = Path(path)
    try:
        if source.is_dir():
            for name in ("manifest.json", "handoff_manifest.json"):
                manifest = source / name
                if manifest.is_file():
                    return str(json.loads(manifest.read_text(encoding="utf-8"))["artifact_kind"])
            return None
        if source.is_file() and source.suffix.lower() == ".zip":
            with ZipFile(source) as archive:
                for name in ("manifest.json", "handoff_manifest.json"):
                    if name in archive.namelist():
                        return str(json.loads(archive.read(name))["artifact_kind"])
    except Exception:
        return None
    return None


def _artifact_candidates(root: Path, kinds: set[str]) -> list[tuple[Path, str]]:
    expanded: list[tuple[Path, str]] = []
    for name in ("manifest.json", "handoff_manifest.json"):
        for manifest in sorted(Path(root).rglob(name)):
            kind = _artifact_kind(manifest.parent)
            if kind in kinds:
                expanded.append((manifest.parent, kind))
    expanded_roots = [path.resolve() for path, _ in expanded]
    archives: list[tuple[Path, str]] = []
    for path in sorted(Path(root).rglob("*.zip")):
        resolved = path.resolve()
        if any(resolved.is_relative_to(parent) for parent in expanded_roots):
            continue
        kind = _artifact_kind(path)
        if kind in kinds:
            archives.append((path, kind))
    unique = {str(path.resolve()): (path, kind) for path, kind in (*expanded, *archives)}
    return list(unique.values())


def discover_inputs(
    root: Path,
    *,
    official_hashes: Mapping[str, str] | None = None,
) -> DiscoveredInputs:
    source = Path(root)
    if source.is_symlink() or not source.is_dir():
        raise E2KaggleError("Kaggle input root differs")
    hashes = (
        dict(load_e2_contract().input_hashes)
        if official_hashes is None
        else dict(official_hashes)
    )
    required = {"official_train_sha256", "official_history_sha256"}
    if not required.issubset(hashes):
        raise E2KaggleError("official hash keys differ")
    official: list[Path] = []
    for train in sorted(source.rglob("train.csv")):
        history = train.parent / "trackman_history.csv"
        if (
            not train.is_symlink()
            and history.is_file()
            and not history.is_symlink()
            and _file_sha(train) == hashes["official_train_sha256"]
            and _file_sha(history) == hashes["official_history_sha256"]
        ):
            official.append(train.parent)
    if len(official) != 1:
        raise E2KaggleError(f"official data count must be one; found={len(official)}")
    inputs = _artifact_candidates(source, {"tree_expert_e2_input_v1"})
    if len(inputs) != 1:
        raise E2KaggleError(f"E2 input count must be one; found={len(inputs)}")
    resumes = _artifact_candidates(
        source,
        {"tree_expert_e2_resume_v1", "tree_expert_e2_handoff_v1"},
    )
    if len(resumes) > 1:
        raise E2KaggleError(f"resume count must be zero or one; found={len(resumes)}")
    return DiscoveredInputs(
        official_data=official[0],
        e2_input=inputs[0][0],
        resume=resumes[0][0] if resumes else None,
    )


def verify_gpu(torch_module: object) -> tuple[str, str]:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or cuda.device_count() != 2:
        raise E2KaggleError("E2 requires exactly two CUDA devices")
    names = tuple(str(cuda.get_device_name(index)) for index in range(2))
    return names


def _runtime_archive(root: Path) -> bytes:
    source = Path(root)
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
        for name in _RUNTIME_MEMBERS:
            path = source / name
            if path.is_symlink() or not path.is_file():
                raise E2KaggleError(f"runtime source is absent: {name}")
            payload = path.read_bytes()
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


def _cell_source(encoded: str, archive_sha: str, code_sha: str) -> str:
    return f'''from __future__ import annotations
import base64, hashlib, importlib.metadata, io, json, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

RUNTIME_B64 = "{encoded}"
RUNTIME_ARCHIVE_SHA256 = "{archive_sha}"
RUNTIME_CODE_SHA256 = "{code_sha}"
WORK = Path("/kaggle/working/tree_expert_e2")
CODE = WORK / "runtime"
STAGE = "setup"

def extract_runtime(payload: bytes) -> None:
    if hashlib.sha256(payload).hexdigest() != RUNTIME_ARCHIVE_SHA256:
        raise RuntimeError("runtime_archive_sha256_differs")
    CODE.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            target = (CODE / member.name).resolve()
            if not member.isfile() or not target.is_relative_to(CODE.resolve()):
                raise RuntimeError("unsafe_runtime_member")
        archive.extractall(CODE, filter="data")

def zip_directory(source: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            info = ZipInfo(path.relative_to(source).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    return output

def materialize_artifact(path: Path, name: str) -> Path:
    return path if path.is_file() else zip_directory(path, WORK / "materialized" / name)

try:
    payload = base64.b64decode(RUNTIME_B64, validate=True)
    extract_runtime(payload)
    sys.path.insert(0, str(CODE))
    from experiments.tree_expert.e2_kaggle import runtime_identity_sha256
    if runtime_identity_sha256(CODE) != RUNTIME_CODE_SHA256:
        raise RuntimeError("runtime_code_sha256_differs")
    print(f"TREE_E2_CODE_READY sha256={{RUNTIME_CODE_SHA256}} size_bytes={{len(payload)}}", flush=True)

    STAGE = "dependencies"
    required = {{"catboost": "1.2.10", "tabm": "0.0.3", "rtdl_num_embeddings": "0.0.12"}}
    for package, version in required.items():
        try:
            current = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            current = None
        if current != version:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"{{package}}=={{version}}"], check=True)
    print("TREE_E2_DEPENDENCIES_READY catboost=1.2.10 tabm=0.0.3 rtdl_num_embeddings=0.0.12", flush=True)

    STAGE = "inputs"
    from experiments.tree_expert.e2_kaggle import discover_inputs
    found = discover_inputs(Path("/kaggle/input"))
    compact = materialize_artifact(found.e2_input, "tree_expert_e2_input.zip")
    resume = None
    if found.resume is not None:
        resume_source = materialize_artifact(found.resume, "tree_expert_e2_resume_source.zip")
        from experiments.tree_expert.e2_kaggle import extract_resume_from_source
        resume = extract_resume_from_source(resume_source, WORK / "materialized" / "tree_expert_e2_resume.zip")
    print(f"TREE_E2_INPUTS_FOUND official={{found.official_data}} compact={{compact}} resume={{resume}}", flush=True)

    STAGE = "gpu"
    import torch
    from experiments.tree_expert.e2_kaggle import verify_gpu
    names = verify_gpu(torch)
    print(f"TREE_E2_GPU_READY device_count=2 names={{names}}", flush=True)

    STAGE = "campaign"
    from experiments.tree_expert.e2_kaggle import run_kaggle_campaign
    result = run_kaggle_campaign(
        official_root=found.official_data,
        compact_input=compact,
        resume_bundle=resume,
        output_dir=WORK,
        absolute_deadline=time.time() + 21600,
    )
    print(f"TREE_E2_HANDOFF_READY path={{result.bundles.handoff}} sha256={{result.bundles.handoff_sha256}} status={{result.status}} delivery={{'yes' if result.bundles.delivery else 'no'}}", flush=True)
except Exception as error:
    print(f"TREE_EXPERT_ERROR stage={{STAGE}} type={{type(error).__name__}} message={{str(error).replace(' ', '_')}}", flush=True)
    emergency = WORK / "bundles" / "tree_expert_e2_handoff.zip"
    if emergency.is_file():
        print(f"TREE_E2_EMERGENCY_HANDOFF path={{emergency}}", flush=True)
    raise
'''


def build_e2_kaggle_cell(output: Path, root: Path | None = None) -> Path:
    repository = Path(__file__).resolve().parents[2] if root is None else Path(root)
    archive = _runtime_archive(repository)
    source = _cell_source(
        base64.b64encode(archive).decode("ascii"),
        sha256(archive).hexdigest(),
        runtime_identity_sha256(repository),
    )
    if len(source.encode("utf-8")) >= 1_000_000:
        raise E2KaggleError("generated Kaggle cell reaches one megabyte")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8")
    return destination


def extract_resume_from_source(source: Path, output: Path) -> Path:
    path = Path(source)
    kind = _artifact_kind(path)
    if kind == "tree_expert_e2_resume_v1":
        return path
    if kind != "tree_expert_e2_handoff_v1":
        raise E2KaggleError("resume source identity differs")
    with ZipFile(path) as archive:
        payload = archive.read("tree_expert_e2_resume.zip")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return destination


def run_kaggle_campaign(**kwargs: object) -> object:
    from .e2_production import run_production_campaign

    return run_production_campaign(**kwargs)
