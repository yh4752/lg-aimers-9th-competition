from __future__ import annotations

import base64
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile
from typing import Callable, Mapping, Sequence


RUNTIME_MEMBERS = (
    "experiments/hierarchical_tabm/__init__.py",
    "experiments/hierarchical_tabm/contract.json",
    "experiments/hierarchical_tabm/contracts.py",
    "experiments/hierarchical_tabm/context_features.py",
    "experiments/hierarchical_tabm/feature_adapter.py",
    "experiments/hierarchical_tabm/calibration.py",
    "experiments/hierarchical_tabm/metrics.py",
    "experiments/hierarchical_tabm/inputs.py",
    "experiments/hierarchical_tabm/training.py",
    "experiments/hierarchical_tabm/artifacts.py",
    "experiments/hierarchical_tabm/runner.py",
    "experiments/hierarchical_tabm/colab.py",
    "experiments/hierarchical_tabm/inference.py",
    "experiments/hierarchical_tabm/runtime_inventory.py",
    "experiments/hierarchical_tabm/requirements-colab.txt",
    "experiments/catboost_preprocessing/features.py",
    "experiments/independent_dl/row_features.py",
    "experiments/independent_dl/preprocessing.py",
    "experiments/independent_dl/feature_sources/trackman.py",
    "experiments/independent_dl/features.py",
    "experiments/independent_dl/progress.py",
    "experiments/independent_dl/training.py",
    "experiments/independent_dl/models/common.py",
    "experiments/independent_dl/models/tabm.py",
    "experiments/catboost_tabm_blend/contract.json",
    "experiments/catboost_tabm_blend/contracts.py",
    "experiments/catboost_tabm_blend/metrics.py",
    "experiments/catboost_tabm_blend/inputs.py",
    "experiments/tabm_campaign/artifacts.py",
)


def runtime_archive(root: Path) -> bytes:
    root = Path(root)
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name in RUNTIME_MEMBERS:
            payload = (root / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0, filename="") as stream:
        stream.write(raw.getvalue())
    return output.getvalue()


def code_identity_sha256(root: Path) -> str:
    digest = sha256()
    root = Path(root)
    for name in RUNTIME_MEMBERS:
        payload = (root / name).read_bytes()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def environment_identity() -> Mapping[str, str]:
    import importlib.metadata
    import platform

    packages = ("numpy", "pandas", "scipy", "torch", "tabm", "rtdl-num-embeddings")
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in packages},
    }


def environment_identity_sha256() -> str:
    payload = json.dumps(
        environment_identity(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def discover_cached_uploads(
    candidates: Sequence[Path],
    *,
    classifier: Callable[[Path], str],
) -> tuple[Path, ...]:
    latest: dict[str, Path] = {}
    allowed = ("training_input", "stage_c_delivery", "campaign_resume")
    for raw_path in candidates:
        path = Path(raw_path)
        if path.is_symlink() or not path.is_file():
            continue
        try:
            kind = classifier(path)
        except Exception:
            continue
        if kind not in allowed:
            continue
        previous = latest.get(kind)
        if previous is None or (path.stat().st_mtime_ns, path.name) > (
            previous.stat().st_mtime_ns, previous.name
        ):
            latest[kind] = path
    required = ("training_input", "stage_c_delivery")
    if not all(kind in latest for kind in required):
        return ()
    return tuple(latest[kind] for kind in (*required, "campaign_resume") if kind in latest)


def render_cell(root: Path) -> bytes:
    root = Path(root)
    archive = runtime_archive(root)
    encoded = base64.b64encode(archive).decode("ascii")
    archive_sha = sha256(archive).hexdigest()
    code_sha = code_identity_sha256(root)
    source = f'''from __future__ import annotations
import base64, hashlib, json, os, pathlib, subprocess, sys, tarfile, time, traceback
from io import BytesIO

SESSION_DEADLINE = time.time() + 10800
RUN_BASE = pathlib.Path("/content/hierarchical_tabm/runs")
RUN_ROOT = RUN_BASE / f"{{time.time_ns()}}_{{os.getpid()}}"
RUN_ROOT.mkdir(parents=True, exist_ok=False)
RUNTIME_ROOT = RUN_ROOT / "runtime"
UPLOAD_ROOT = pathlib.Path("/content/hierarchical_tabm/upload_cache")
OUTPUT_ROOT = RUN_ROOT / "campaign"
SNAPSHOT_ROOT = RUN_ROOT / "snapshots"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
RUNTIME_B64 = "{encoded}"
EXPECTED_RUNTIME_SHA256 = "{archive_sha}"
EXPECTED_CODE_SHA256 = "{code_sha}"

def fail(stage, error):
    message = str(error).replace(" ", "_").replace("\\n", "_")[:500]
    print(f"HIER_ERROR stage={{stage}} type={{type(error).__name__}} message={{message}}", flush=True)
    traceback.print_exc()

stage = "runtime"
try:
    payload = base64.b64decode(RUNTIME_B64, validate=True)
    if hashlib.sha256(payload).hexdigest() != EXPECTED_RUNTIME_SHA256:
        raise RuntimeError("embedded runtime SHA-256 differs")
    RUNTIME_ROOT.mkdir()
    with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (RUNTIME_ROOT / member.name).resolve()
            if not member.isfile() or RUNTIME_ROOT.resolve() not in target.parents:
                raise RuntimeError("unsafe embedded runtime member")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError("embedded runtime member is unreadable")
            with target.open("xb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
    sys.path.insert(0, str(RUNTIME_ROOT))
    from experiments.hierarchical_tabm.runtime_inventory import code_identity_sha256
    if code_identity_sha256(RUNTIME_ROOT) != EXPECTED_CODE_SHA256:
        raise RuntimeError("embedded code identity differs")
    print(f"HIER_CODE_READY sha256={{EXPECTED_CODE_SHA256}} size_bytes={{len(payload)}}", flush=True)

    stage = "dependencies"
    if time.time() >= SESSION_DEADLINE:
        raise RuntimeError("deadline expired before dependency installation")
    requirements = RUNTIME_ROOT / "experiments/hierarchical_tabm/requirements-colab.txt"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)], check=True)
    from experiments.hierarchical_tabm.runtime_inventory import environment_identity_sha256
    import tabm, rtdl_num_embeddings, torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU is required")
    if torch.cuda.get_device_capability(0) < (7, 5):
        raise RuntimeError("Tesla T4-compatible CUDA capability is required")
    print(f"HIER_GPU_READY name={{torch.cuda.get_device_name(0)}}", flush=True)

    stage = "upload"
    from google.colab import files
    from experiments.hierarchical_tabm.inputs import classify_upload
    from experiments.hierarchical_tabm.runtime_inventory import discover_cached_uploads
    cached_candidates = tuple(UPLOAD_ROOT.glob("*.zip")) + tuple(pathlib.Path("/content").glob("*.zip"))
    paths = list(discover_cached_uploads(cached_candidates, classifier=classify_upload))
    if paths:
        print(f"HIER_UPLOAD_CACHE_REUSED count={{len(paths)}}", flush=True)
    else:
        uploaded = files.upload()
        if len(uploaded) not in (2, 3):
            raise RuntimeError("upload count must be two or three")
        for name, value in uploaded.items():
            safe = pathlib.Path(name).name
            if safe != name or not safe.lower().endswith(".zip"):
                raise RuntimeError("every upload must be a top-level ZIP")
            (UPLOAD_ROOT / safe).write_bytes(value)
        paths = list(discover_cached_uploads(
            tuple(UPLOAD_ROOT.glob("*.zip")), classifier=classify_upload
        ))
        if not paths:
            raise RuntimeError("uploaded ZIP kinds are incomplete")
        print(f"HIER_UPLOAD_CACHE_READY count={{len(paths)}}", flush=True)

    stage = "inputs"
    from experiments.hierarchical_tabm.contracts import contract_sha256, load_contract
    from experiments.hierarchical_tabm.inputs import classify_and_verify_uploads
    contract = load_contract()
    verified, resume = classify_and_verify_uploads(
        paths, run_root=RUN_ROOT / "verified", contract=contract,
        expected_contract_sha256=contract_sha256(),
        expected_code_sha256=EXPECTED_CODE_SHA256,
    )
    def file_sha256(path):
        digest = hashlib.sha256()
        with pathlib.Path(path).open("rb") as stream:
            while chunk := stream.read(1024 * 1024): digest.update(chunk)
        return digest.hexdigest()
    bindings = {{
        "contract_sha256": contract_sha256(),
        "code_sha256": EXPECTED_CODE_SHA256,
        "environment_sha256": environment_identity_sha256(),
        "input_manifest_sha256": verified.training.manifest_sha256,
        "train_sha256": verified.training.train_sha256,
        "history_sha256": verified.training.history_sha256,
        "stage_c_delivery_sha256": verified.stage_c.delivery_sha256,
        "stage_c_review_sha256": verified.stage_c.review_sha256,
        "stage_c_state_sha256": verified.stage_c.stage_state_sha256,
        "anchor_2022_2023_sha256": file_sha256(verified.anchor_predictions["2022->2023"]),
        "anchor_2023_2024_sha256": file_sha256(verified.anchor_predictions["2023->2024"]),
    }}
    print(f"HIER_INPUTS_VERIFIED count={{len(paths)}}", flush=True)

    stage = "training"
    from experiments.hierarchical_tabm.colab import SubprocessCampaignRuntime, run_supervised_campaign
    campaign_log = RUN_ROOT / "hierarchical_tabm.log"
    runtime = SubprocessCampaignRuntime(
        data_dir=verified.training.data_dir, bindings=bindings, log_path=campaign_log,
    )
    def download_resume(path):
        print(f"HIER_DOWNLOAD_REQUESTED path={{path}}", flush=True)
        files.download(str(path))
    result = run_supervised_campaign(
        verified, output_dir=OUTPUT_ROOT, snapshot_dir=SNAPSHOT_ROOT,
        resume_bundle=resume, wall_deadline=SESSION_DEADLINE,
        on_verified_resume=download_resume, log_path=campaign_log, runtime=runtime,
        snapshot_interval_seconds=contract.snapshot_interval_seconds,
        download_interval_seconds=contract.download_interval_seconds,
    )
    print(f"HIER_CAMPAIGN_SUCCESS status={{result.state.status}}", flush=True)
    for path in (result.bundles.review, result.bundles.resume, result.candidate_delivery):
        if path is not None:
            print(f"HIER_DOWNLOAD_REQUESTED path={{path}}", flush=True)
            files.download(str(path))
    if result.candidate_delivery is not None:
        print(f"HIER_DELIVERY_READY path={{result.candidate_delivery}}", flush=True)
except Exception as error:
    fail(stage, error)
    raise
'''
    return source.encode("utf-8")
