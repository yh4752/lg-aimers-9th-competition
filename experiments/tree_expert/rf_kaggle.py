from __future__ import annotations

import base64
from dataclasses import dataclass
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile
from typing import Callable
from zipfile import BadZipFile, ZipFile


class RFKaggleError(ValueError):
    pass


@dataclass(frozen=True)
class DiscoveredRFInputs:
    rf_input: Path
    resume: Path | None


@dataclass(frozen=True)
class RFPublishedArtifact:
    kind: str
    path: Path


_RUNTIME_MEMBERS = (
    "experiments/tree_expert/__init__.py",
    "experiments/tree_expert/contracts.py",
    "experiments/tree_expert/features.py",
    "experiments/tree_expert/rf_artifacts.py",
    "experiments/tree_expert/rf_contract.json",
    "experiments/tree_expert/rf_contracts.py",
    "experiments/tree_expert/rf_decisions.py",
    "experiments/tree_expert/rf_diagnostics.py",
    "experiments/tree_expert/rf_full_fit.py",
    "experiments/tree_expert/rf_inference.py",
    "experiments/tree_expert/rf_inputs.py",
    "experiments/tree_expert/rf_kaggle.py",
    "experiments/tree_expert/rf_runner.py",
    "experiments/tree_expert/rf_state.py",
    "experiments/tree_expert/rf_training.py",
    "experiments/temporal_portfolio/__init__.py",
    "experiments/temporal_portfolio/seasonal_features.py",
    "experiments/temporal_portfolio/trackman_pitcher.py",
    "experiments/independent_dl/__init__.py",
    "experiments/independent_dl/feature_sources/__init__.py",
    "experiments/independent_dl/feature_sources/seasonal.py",
    "experiments/independent_dl/feature_sources/trackman.py",
)
_KINDS = {"tree_expert_rf_input_v1", "tree_expert_rf_resume_v1"}


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
                if path.is_symlink() or not path.is_file():
                    raise RFKaggleError(f"runtime member is missing: {name}")
                payload = path.read_bytes()
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mode = 0o644
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _manifest_from_directory(path: Path) -> bytes | None:
    manifest = path / "manifest.json"
    if path.is_symlink() or not manifest.is_file() or manifest.is_symlink():
        return None
    try:
        return manifest.read_bytes()
    except OSError:
        return None


def _manifest_from_zip(path: Path) -> bytes | None:
    try:
        with ZipFile(path) as archive:
            infos = archive.infolist()
            if sum(info.filename == "manifest.json" for info in infos) != 1:
                return None
            return archive.read("manifest.json")
    except (OSError, BadZipFile, KeyError):
        return None


def _candidate(path: Path) -> tuple[str, str] | None:
    payload = _manifest_from_directory(path) if path.is_dir() else _manifest_from_zip(path)
    if payload is None or len(payload) > 4 * 1024 * 1024:
        return None
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if type(manifest) is not dict or manifest.get("artifact_kind") not in _KINDS:
        return None
    members = manifest.get("members")
    if type(members) is not dict:
        return None
    identity_payload = {
        "artifact_kind": manifest["artifact_kind"],
        "campaign_id": manifest.get("campaign_id"),
        "schema_version": manifest.get("schema_version"),
        "bindings": manifest.get("bindings"),
        "identity": manifest.get("identity"),
        "members": members,
    }
    identity = sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return str(manifest["artifact_kind"]), identity


def _choose(candidates: list[tuple[Path, str]], *, label: str, required: bool) -> Path | None:
    by_identity: dict[str, list[Path]] = {}
    for path, identity in candidates:
        by_identity.setdefault(identity, []).append(path)
    if len(by_identity) != (1 if required else min(1, len(by_identity))):
        if required:
            raise RFKaggleError(f"{label} count must be one; found={len(by_identity)}")
        raise RFKaggleError(f"{label} count must be zero or one; found={len(by_identity)}")
    if not by_identity:
        return None
    paths = next(iter(by_identity.values()))
    return sorted(paths, key=lambda path: (path.is_file(), len(path.parts), str(path)))[0]


def discover_rf_inputs(root: Path) -> DiscoveredRFInputs:
    source = Path(root)
    if source.is_symlink() or not source.is_dir():
        raise RFKaggleError("Kaggle input root differs")
    found: dict[str, list[tuple[Path, str]]] = {kind: [] for kind in _KINDS}
    seen: set[Path] = set()
    for manifest in source.rglob("manifest.json"):
        path = manifest.parent
        if path in seen:
            continue
        seen.add(path)
        item = _candidate(path)
        if item is not None:
            found[item[0]].append((path, item[1]))
    for archive in source.rglob("*.zip"):
        if archive in seen:
            continue
        seen.add(archive)
        item = _candidate(archive)
        if item is not None:
            found[item[0]].append((archive, item[1]))
    rf_input = _choose(found["tree_expert_rf_input_v1"], label="RF input", required=True)
    resume = _choose(found["tree_expert_rf_resume_v1"], label="resume", required=False)
    assert rf_input is not None
    return DiscoveredRFInputs(rf_input, resume)


def run_supervised_rf_campaign(
    *,
    run_campaign: Callable[[], object],
    publish: Callable[[RFPublishedArtifact], None],
) -> object:
    result = run_campaign()
    ordered = (
        ("resume", getattr(result, "resume_bundle", None)),
        ("review", getattr(result, "review_bundle", None)),
        ("delivery", getattr(result, "delivery_bundle", None)),
    )
    seen: set[Path] = set()
    for kind, value in ordered:
        if value is None:
            continue
        path = Path(value)
        if path in seen or path.is_symlink() or not path.is_file():
            if path in seen:
                continue
            raise RFKaggleError(f"final {kind} artifact differs")
        seen.add(path)
        publish(RFPublishedArtifact(kind, path))
    return result


def build_rf_kaggle_cell(output: Path, root: Path | None = None) -> Path:
    payload = runtime_archive(root)
    encoded = base64.b64encode(payload).decode("ascii")
    digest = sha256(payload).hexdigest()
    source = f'''from __future__ import annotations
import base64, hashlib, importlib.metadata, io, os, shutil, subprocess, sys, tarfile, time, traceback
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

RUNTIME_B64 = "{encoded}"
RUNTIME_SHA256 = "{digest}"
STAGE = "setup"

def safe_extract(payload, destination):
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) > 128 or sum(item.size for item in members) > 8 * 1024 * 1024:
            raise RuntimeError("runtime archive exceeds limit")
        for item in members:
            path = PurePosixPath(item.name)
            if not item.isfile() or item.issym() or item.islnk() or path.is_absolute() or ".." in path.parts:
                raise RuntimeError("unsafe runtime member")
        archive.extractall(destination, filter="data")

def zip_directory(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w") as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            info = ZipInfo(path.relative_to(source).as_posix(), (2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    return destination

try:
    RUN_ROOT = Path("/kaggle/working/tree_expert_rf_runs") / str(time.time_ns())
    CODE_ROOT = RUN_ROOT / "runtime"
    CODE_ROOT.mkdir(parents=True)
    runtime = base64.b64decode(RUNTIME_B64, validate=True)
    if hashlib.sha256(runtime).hexdigest() != RUNTIME_SHA256:
        raise RuntimeError("embedded runtime SHA-256 differs")
    safe_extract(runtime, CODE_ROOT)
    sys.path.insert(0, str(CODE_ROOT))
    print(f"TREE_RF_CODE_READY sha256={{RUNTIME_SHA256}} size_bytes={{len(runtime)}}", flush=True)

    STAGE = "dependencies"
    try:
        version = importlib.metadata.version("catboost")
    except importlib.metadata.PackageNotFoundError:
        version = "missing"
    if version != "1.2.10":
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "catboost==1.2.10"], check=True)
    print(f"TREE_RF_DEPENDENCIES_READY catboost={{importlib.metadata.version('catboost')}}", flush=True)

    STAGE = "gpu"
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True,
    ).strip().splitlines()
    if len(names) != 2 or any("T4" not in name for name in names):
        raise RuntimeError(f"T4x2 required; found={{names}}")
    print(f"TREE_RF_GPU_READY count=2 names={{' | '.join(names)}}", flush=True)

    STAGE = "inputs"
    from experiments.tree_expert.rf_inputs import verify_and_extract_rf_input, verify_official_data
    from experiments.tree_expert.rf_kaggle import discover_rf_inputs, run_supervised_rf_campaign
    from experiments.tree_expert.rf_runner import run_rf_campaign
    cached = globals().get("_TREE_RF_INPUT_CACHE")
    if cached and all(Path(item).exists() for item in cached if item is not None):
        input_path, resume_path = (Path(item) if item is not None else None for item in cached)
        print("TREE_RF_UPLOAD_CACHE_REUSED", flush=True)
    else:
        discovered = discover_rf_inputs(Path("/kaggle/input"))
        input_path, resume_path = discovered.rf_input, discovered.resume
        globals()["_TREE_RF_INPUT_CACHE"] = (str(input_path), str(resume_path) if resume_path else None)
    official_roots = sorted({{path.parent for path in Path("/kaggle/input").rglob("train.csv") if (path.parent / "trackman_history.csv").is_file()}})
    if len(official_roots) != 1:
        raise RuntimeError(f"official data count must be one; found={{len(official_roots)}}")
    official = verify_official_data(official_roots[0])
    verified = verify_and_extract_rf_input(input_path, RUN_ROOT / "verified_input")
    resume = resume_path
    if resume is not None and resume.is_dir():
        resume = zip_directory(resume, RUN_ROOT / "materialized/tree_expert_rf_resume.zip")
    print(f"TREE_RF_INPUTS_VERIFIED input={{input_path}} resume={{resume}} official={{official.root}}", flush=True)

    STAGE = "campaign"
    published = []
    result = run_supervised_rf_campaign(
        run_campaign=lambda: run_rf_campaign(
            verified, official, RUN_ROOT / "campaign", resume_bundle=resume,
            wall_deadline=time.monotonic() + 19_800,
        ),
        publish=published.append,
    )
    destinations = {{
        "resume": Path("/kaggle/working/tree_expert_rf_resume.zip"),
        "review": Path("/kaggle/working/tree_expert_rf_review.zip"),
        "delivery": Path("/kaggle/working/tree_expert_rf_delivery.zip"),
    }}
    for item in published:
        shutil.copy2(item.path, destinations[item.kind])
        print(f"TREE_RF_ARTIFACT_READY kind={{item.kind}} path={{destinations[item.kind]}}", flush=True)
    print(
        f"TREE_RF_CAMPAIGN_SUCCESS status={{result.status}} acceptance={{result.acceptance_status}} "
        f"review={{destinations['review']}} resume={{destinations['resume']}} "
        f"delivery={{destinations['delivery'] if result.delivery_bundle else None}}",
        flush=True,
    )
except Exception as error:
    print(f"TREE_RF_ERROR stage={{STAGE}} type={{type(error).__name__}} message={{str(error).replace(' ', '_')}}", flush=True)
    traceback.print_exc()
    candidates = sorted(
        Path("/kaggle/working/tree_expert_rf_runs").rglob("tree_expert_rf_resume.zip"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if candidates:
        emergency = Path("/kaggle/working/tree_expert_rf_resume.zip")
        shutil.copy2(candidates[-1], emergency)
        print(f"TREE_RF_ARTIFACT_READY kind=resume path={{emergency}}", flush=True)
    raise
'''
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded_source = source.encode("utf-8")
    if len(encoded_source) >= 900_000:
        raise RFKaggleError("generated Kaggle cell exceeds 900000 bytes")
    destination.write_bytes(encoded_source)
    return destination
