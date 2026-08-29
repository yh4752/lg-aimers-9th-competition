from __future__ import annotations

import base64
from dataclasses import dataclass
from gzip import GzipFile
from hashlib import sha256
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from .s4_contracts import contract_sha256, load_s4_contract


class S4KaggleError(ValueError):
    pass


_RUNTIME_MEMBERS = (
    "experiments/independent_dl/__init__.py",
    "experiments/independent_dl/feature_sources/__init__.py",
    "experiments/independent_dl/feature_sources/seasonal.py",
    "experiments/independent_dl/feature_sources/trackman.py",
    "experiments/temporal_portfolio/__init__.py",
    "experiments/temporal_portfolio/seasonal_features.py",
    "experiments/temporal_portfolio/trackman_pitcher.py",
    "experiments/tree_expert/features.py",
    "experiments/tree_expert/hc_contract.json",
    "experiments/tree_expert/hc_contracts.py",
    "experiments/tree_expert/hc_features.py",
    "experiments/tree_expert/hc_calibration.py",
    "experiments/tree_expert/hc_metrics.py",
    "experiments/tree_expert/s4_contract.json",
    "experiments/tree_expert/s4_contracts.py",
    "experiments/tree_expert/s4_inputs.py",
    "experiments/tree_expert/s4_temporal.py",
    "experiments/tree_expert/s4_features.py",
    "experiments/tree_expert/s4_training.py",
    "experiments/tree_expert/s4_calibration.py",
    "experiments/tree_expert/s4_decisions.py",
    "experiments/tree_expert/s4_state.py",
    "experiments/tree_expert/s4_artifacts.py",
    "experiments/tree_expert/s4_full_fit.py",
    "experiments/tree_expert/s4_inference.py",
    "experiments/tree_expert/s4_runner.py",
    "experiments/tree_expert/s4_production.py",
    "experiments/tree_expert/s4_kaggle.py",
    "experiments/tree_expert/t3_contract.json",
    "experiments/tree_expert/t3_contracts.py",
    "experiments/tree_expert/t3_inputs.py",
)


@dataclass(frozen=True)
class DiscoveredS4Inputs:
    official_data: Path
    s4_input: Path
    previous_handoff: Path | None


def runtime_member_names() -> tuple[str, ...]:
    return _RUNTIME_MEMBERS


def _file_sha(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_identity_sha256(root: Path) -> str:
    digest = sha256()
    for name in _RUNTIME_MEMBERS:
        path = Path(root) / name
        if path.is_symlink() or not path.is_file():
            raise S4KaggleError(f"runtime source is absent: {name}")
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _manifest(path: Path) -> tuple[str, str] | None:
    try:
        if path.is_dir():
            candidate = path / "manifest.json"
            if not candidate.is_file() or candidate.is_symlink():
                return None
            payload = candidate.read_bytes()
        elif path.is_file() and path.suffix.lower() == ".zip":
            with ZipFile(path) as archive:
                if "manifest.json" not in archive.namelist():
                    return None
                payload = archive.read("manifest.json")
        else:
            return None
        value = json.loads(payload)
        return str(value["artifact_kind"]), sha256(payload).hexdigest()
    except Exception:
        return None


def _logical_candidates(root: Path, kind: str) -> tuple[Path, ...]:
    found: dict[str, Path] = {}
    expanded: list[Path] = []
    for manifest in sorted(Path(root).rglob("manifest.json")):
        identity = _manifest(manifest.parent)
        if identity is not None and identity[0] == kind:
            found.setdefault(identity[1], manifest.parent)
            expanded.append(manifest.parent.resolve())
    for archive in sorted(Path(root).rglob("*.zip")):
        resolved = archive.resolve()
        if any(resolved.is_relative_to(parent) for parent in expanded):
            continue
        identity = _manifest(archive)
        if identity is not None and identity[0] == kind:
            found.setdefault(identity[1], archive)
    return tuple(found.values())


def discover_s4_inputs(
    root: Path, *, official_hashes: Mapping[str, str] | None = None
) -> DiscoveredS4Inputs:
    source = Path(root)
    if source.is_symlink() or not source.is_dir():
        raise S4KaggleError("Kaggle input root differs")
    contract = load_s4_contract()
    hashes = dict(contract.inputs if official_hashes is None else official_hashes)
    official = []
    for train in sorted(source.rglob("train.csv")):
        history = train.parent / "trackman_history.csv"
        if (
            train.is_file() and not train.is_symlink() and history.is_file() and not history.is_symlink()
            and _file_sha(train) == hashes["official_train_sha256"]
            and _file_sha(history) == hashes["official_history_sha256"]
        ):
            official.append(train.parent)
    if len(official) != 1:
        raise S4KaggleError(f"official data count must be one; found={len(official)}")
    compact = _logical_candidates(source, "tree_s4_input_v1")
    if len(compact) != 1:
        raise S4KaggleError(f"S4 input count must be one; found={len(compact)}")
    previous = _logical_candidates(source, "tree_s4_handoff_v1")
    if len(previous) > 1:
        raise S4KaggleError(f"S4 handoff count must be zero or one; found={len(previous)}")
    return DiscoveredS4Inputs(official[0], compact[0], previous[0] if previous else None)


def verify_gpu(torch_module: object) -> tuple[str, str]:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or cuda.device_count() != 2:
        raise S4KaggleError("S4 requires exactly two CUDA devices")
    names = tuple(str(cuda.get_device_name(index)) for index in range(2))
    if any("T4" not in name for name in names):
        raise S4KaggleError("S4 requires Tesla T4 x2")
    return names  # type: ignore[return-value]


def _zip_directory(source: Path, destination: Path) -> Path:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(Path(source).rglob("*")):
            if path.is_symlink():
                raise S4KaggleError("expanded handoff contains a symlink")
            if path.is_file():
                name = path.relative_to(source).as_posix()
                pure = PurePosixPath(name)
                if pure.is_absolute() or ".." in pure.parts:
                    raise S4KaggleError("expanded handoff member is unsafe")
                info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
    return output


def materialize_resume_from_handoff(
    source: Path, destination: Path, bindings
) -> Path:
    from .s4_artifacts import verify_s4_handoff

    path = Path(source)
    handoff = path if path.is_file() else _zip_directory(path, Path(destination).with_name("previous_handoff.zip"))
    verify_s4_handoff(handoff, bindings)
    with ZipFile(handoff) as archive:
        payload = archive.read("resume.zip")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return output


def _runtime_archive(root: Path) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name in _RUNTIME_MEMBERS:
            path = Path(root) / name
            if path.is_symlink() or not path.is_file():
                raise S4KaggleError(f"runtime source is absent: {name}")
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
        handle.write(raw.getvalue())
    return compressed.getvalue()


def _cell_source(encoded: str, archive_sha: str, code_sha: str) -> str:
    return f'''from __future__ import annotations
import base64, hashlib, importlib.metadata, io, subprocess, sys, tarfile, time
from pathlib import Path

RUNTIME_B64 = "{encoded}"
RUNTIME_ARCHIVE_SHA256 = "{archive_sha}"
RUNTIME_CODE_SHA256 = "{code_sha}"
CODE_ROOT = Path("/kaggle/working/tree_s4_runtime")
CAMPAIGN_ROOT = Path("/kaggle/working/tree_s4_campaign")
FINAL_HANDOFF = Path("/kaggle/working/anchor_residual_hierarchical_handoff.zip")
STAGE = "setup"

def extract_runtime(payload: bytes) -> None:
    if hashlib.sha256(payload).hexdigest() != RUNTIME_ARCHIVE_SHA256:
        raise RuntimeError("runtime_archive_sha256_differs")
    CODE_ROOT.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            target = (CODE_ROOT / member.name).resolve()
            if not member.isfile() or not target.is_relative_to(CODE_ROOT.resolve()):
                raise RuntimeError("unsafe_runtime_member")
        archive.extractall(CODE_ROOT, filter="data")

try:
    payload = base64.b64decode(RUNTIME_B64, validate=True)
    extract_runtime(payload)
    sys.path.insert(0, str(CODE_ROOT))
    from experiments.tree_expert.s4_kaggle import runtime_identity_sha256
    if runtime_identity_sha256(CODE_ROOT) != RUNTIME_CODE_SHA256:
        raise RuntimeError("runtime_code_sha256_differs")
    print(f"S4_CODE_READY sha256={{RUNTIME_CODE_SHA256}} size_bytes={{len(payload)}}", flush=True)

    STAGE = "dependencies"
    required = {{"catboost": "1.2.10", "xgboost": "3.0.2", "lightgbm": "4.6.0"}}
    for package, version in required.items():
        try:
            current = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            current = None
        if current != version:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"{{package}}=={{version}}"], check=True)
    print("S4_DEPENDENCIES_READY catboost=1.2.10 xgboost=3.0.2 lightgbm=4.6.0", flush=True)

    STAGE = "inputs"
    import torch
    from experiments.tree_expert.s4_kaggle import discover_s4_inputs, materialize_resume_from_handoff, verify_gpu
    from experiments.tree_expert.s4_inputs import verify_and_extract_s4_input
    from experiments.tree_expert.t3_inputs import verify_official_data
    from experiments.tree_expert.s4_artifacts import S4Bindings
    from experiments.tree_expert.s4_contracts import contract_sha256, load_s4_contract
    found = discover_s4_inputs(Path("/kaggle/input"))
    names = verify_gpu(torch)
    verified = verify_and_extract_s4_input(found.s4_input, Path("/kaggle/working/tree_s4_verified_input"))
    official = verify_official_data(found.official_data)
    bindings = S4Bindings(
        contract_sha256(), RUNTIME_CODE_SHA256, verified.manifest_sha256,
        official.train_sha256, official.history_sha256, verified.e2_handoff_sha256,
    )
    resume = None
    if found.previous_handoff is not None:
        resume = materialize_resume_from_handoff(found.previous_handoff, Path("/kaggle/working/tree_s4_resume.zip"), bindings)
    print(f"S4_INPUTS_VERIFIED official={{found.official_data}} input={{found.s4_input}} previous={{found.previous_handoff}}", flush=True)
    print(f"S4_GPU_READY count=2 names={{' | '.join(names)}}", flush=True)

    STAGE = "campaign"
    from experiments.tree_expert.s4_production import ProductionS4Runtime
    from experiments.tree_expert.s4_runner import run_s4_campaign
    runtime = ProductionS4Runtime(verified, official, CAMPAIGN_ROOT)
    deadline = time.monotonic() + load_s4_contract().runtime.wall_seconds
    result = run_s4_campaign(
        bindings, CAMPAIGN_ROOT, runtime=runtime, wall_deadline=deadline,
        resume_bundle=resume, gpu_ids=(0, 1),
    )
    if result.handoff != FINAL_HANDOFF:
        raise RuntimeError("final_handoff_path_differs")
    print(f"S4_SUCCESS status={{result.status}} handoff={{FINAL_HANDOFF}}", flush=True)
except Exception as error:
    if FINAL_HANDOFF.is_file():
        print(f"S4_HANDOFF_READY path={{FINAL_HANDOFF}} status=emergency", flush=True)
    print(f"S4_ERROR stage={{STAGE}} type={{type(error).__name__}} message={{str(error).replace(' ', '_')}}", flush=True)
    raise
'''


def build_s4_kaggle_cell(output: Path, root: Path | None = None) -> Path:
    repository = Path(__file__).resolve().parents[2] if root is None else Path(root)
    archive = _runtime_archive(repository)
    source = _cell_source(
        base64.b64encode(archive).decode("ascii"),
        sha256(archive).hexdigest(),
        runtime_identity_sha256(repository),
    )
    if len(source.encode()) >= 1_000_000:
        raise S4KaggleError("generated Kaggle cell reaches one megabyte")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8")
    return destination
