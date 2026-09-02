from __future__ import annotations

import base64
from dataclasses import dataclass
from hashlib import sha256
import gzip
import io
import json
from pathlib import Path
import tarfile
import time
from zipfile import BadZipFile, ZipFile

from competition_rules.code_gate import inspect_inference_source
from experiments.direct_expert.kaggle import _smoke_gpu

from .artifacts import canonical_json
from .contracts import load_contract
from .runner import run_campaign
from .runtime import ProductionE3Runtime
from .runtime_inventory import code_identity_sha256, runtime_members


class E3KaggleError(ValueError):
    pass


@dataclass(frozen=True)
class DiscoveredInputs:
    official_data: Path
    campaign_input: Path
    previous_handoff: Path | None


def _descriptor(path: Path) -> tuple[str, str] | None:
    try:
        if path.is_dir():
            payload = (path / "manifest.json").read_bytes()
        else:
            with ZipFile(path) as archive:
                payload = archive.read("manifest.json")
        manifest = json.loads(payload)
    except (OSError, KeyError, BadZipFile, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if type(manifest) is not dict or type(manifest.get("artifact_kind")) is not str:
        return None
    return manifest["artifact_kind"], sha256(canonical_json(manifest)).hexdigest()


def discover_inputs(root: Path) -> DiscoveredInputs:
    source = Path(root)
    official: list[Path] = []
    campaigns: dict[str, Path] = {}
    handoffs: dict[str, Path] = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            continue
        if path.is_dir() and (path / "train.csv").is_file() and (path / "trackman_history.csv").is_file():
            official.append(path)
        if (path.is_file() and path.suffix.lower() == ".zip") or (path.is_dir() and (path / "manifest.json").is_file()):
            descriptor = _descriptor(path)
            if descriptor is None:
                continue
            kind, identity = descriptor
            if kind == "direct_expert_input_v1":
                campaigns.setdefault(identity, path)
            elif kind == "failure_regime_e3_handoff_v1":
                handoffs.setdefault(identity, path)
    if len(official) != 1:
        raise E3KaggleError(f"official data count must be one; found={len(official)}")
    if len(campaigns) != 1:
        raise E3KaggleError(f"direct expert input count must be one; found={len(campaigns)}")
    if len(handoffs) > 1:
        raise E3KaggleError(f"E3 handoff count must be zero or one; found={len(handoffs)}")
    return DiscoveredInputs(
        official[0],
        next(iter(campaigns.values())),
        next(iter(handoffs.values())) if handoffs else None,
    )


def verify_t4x2(torch_module) -> tuple[str, str]:
    names = tuple(torch_module.cuda.get_device_name(index) for index in range(torch_module.cuda.device_count()))
    if len(names) != 2 or any("T4" not in name for name in names):
        raise E3KaggleError("two Tesla T4 GPUs are required")
    return names


def inference_source_paths(repository_root: Path) -> tuple[Path, ...]:
    root = Path(repository_root)
    return tuple(
        root / name
        for name in (
            "experiments/failure_regime_e3/meta.py",
            "experiments/failure_regime_e3/runtime.py",
            "experiments/failure_regime_e3/selection.py",
            "experiments/direct_expert/features.py",
            "experiments/tree_expert/features.py",
            "experiments/temporal_portfolio/seasonal_features.py",
            "experiments/temporal_portfolio/trackman_pitcher.py",
            "experiments/temporal_portfolio/trackman_batter.py",
            "experiments/independent_dl/feature_sources/seasonal.py",
            "experiments/independent_dl/feature_sources/trackman.py",
        )
    )


def run_kaggle_campaign(input_root: Path, work_root: Path):
    import torch

    found = discover_inputs(input_root)
    print(
        f"E3_INPUTS_FOUND official={found.official_data} input={found.campaign_input} "
        f"resume={found.previous_handoff}",
        flush=True,
    )
    names = verify_t4x2(torch)
    print(f"E3_GPU_READY count=2 names={' | '.join(names)}", flush=True)
    _smoke_gpu(torch)
    print("E3_GPU_SMOKE_SUCCESS", flush=True)
    repository_root = Path(__file__).resolve().parents[2]
    source_gate = inspect_inference_source(
        inference_source_paths(repository_root),
        project_root=repository_root,
    )
    print(
        f"E3_SOURCE_GATE_PASSED files={source_gate['file_count']} sha256={source_gate['source_sha256']}",
        flush=True,
    )
    identity = code_identity_sha256(repository_root)
    root = Path(work_root)
    runtime = ProductionE3Runtime(
        found.official_data,
        found.campaign_input,
        root / "runtime",
        code_sha256=identity,
    )
    print(
        f"E3_PREFLIGHT_SUCCESS input_manifest={runtime.bindings.input_manifest_sha256} "
        f"train={runtime.bindings.train_sha256} history={runtime.bindings.history_sha256}",
        flush=True,
    )
    deadline = time.time() + int(load_contract().runtime["wall_seconds"])
    result = run_campaign(
        runtime,
        root / "campaign",
        absolute_deadline=deadline,
        previous_handoff=found.previous_handoff,
    )
    print(f"E3_CAMPAIGN_STATUS status={result.status} decision={result.decision}", flush=True)
    print(f"E3_REVIEW_READY path={result.review.resolve()}", flush=True)
    print(f"E3_HANDOFF_READY path={result.handoff.resolve()}", flush=True)
    return result


def _runtime_archive(root: Path) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name in runtime_members(root):
                payload = (Path(root) / name).read_bytes()
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mtime = 0
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def build_kaggle_cell(output: Path, *, root: Path | None = None) -> Path:
    repository = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    payload = _runtime_archive(repository)
    encoded = base64.b64encode(payload).decode("ascii")
    source = f'''from __future__ import annotations
import base64, importlib.metadata, io, subprocess, sys, tarfile
from hashlib import sha256
from pathlib import Path

PAYLOAD = base64.b64decode("{encoded}")
print(f"E3_CODE_READY sha256={{sha256(PAYLOAD).hexdigest()}} size_bytes={{len(PAYLOAD)}}", flush=True)
required = {{"catboost": "1.2.10"}}
for package, version in required.items():
    try:
        current = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        current = None
    if current != version:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"{{package}}=={{version}}"], check=True)
runtime_root = Path("/kaggle/working/failure_regime_e3_runtime")
runtime_root.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(PAYLOAD), mode="r:gz") as archive:
    for member in archive.getmembers():
        target = (runtime_root / member.name).resolve()
        if not target.is_relative_to(runtime_root.resolve()) or not member.isfile():
            raise RuntimeError("unsafe embedded runtime member")
    archive.extractall(runtime_root)
sys.path.insert(0, str(runtime_root))
from experiments.failure_regime_e3.kaggle import run_kaggle_campaign
# terminal markers: E3_GPU_SMOKE_SUCCESS E3_PREFLIGHT_SUCCESS E3_REVIEW_READY E3_HANDOFF_READY
try:
    run_kaggle_campaign(Path("/kaggle/input"), Path("/kaggle/working/failure_regime_e3"))
except Exception as error:
    message = str(error).replace(" ", "_").replace("\\n", "_")
    print(f"E3_ERROR type={{type(error).__name__}} message={{message}}", flush=True)
    raise
'''.encode("utf-8")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source)
    return destination
