from __future__ import annotations

import base64
import gzip
from hashlib import sha256
import io
from pathlib import Path
import tarfile


RUNTIME_MEMBERS = (
    "experiments/catboost_preprocessing/features.py",
    "experiments/independent_dl/preprocessing.py",
    "experiments/independent_dl/row_features.py",
    "experiments/tabm_campaign/artifacts.py",
    "experiments/catboost_tabm_blend/contract.json",
    "experiments/catboost_tabm_blend/contracts.py",
    "experiments/catboost_tabm_blend/metrics.py",
    "experiments/catboost_tabm_blend/inputs.py",
    "experiments/catboost_tabm_blend/training.py",
    "experiments/catboost_tabm_blend/artifacts.py",
    "experiments/catboost_tabm_blend/runner.py",
    "experiments/catboost_tabm_blend/colab.py",
    "experiments/catboost_tabm_blend/runtime_inventory.py",
    "experiments/catboost_tabm_blend/requirements-colab.txt",
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


def runtime_sha256(root: Path) -> str:
    return sha256(runtime_archive(root)).hexdigest()


def code_identity_sha256(root: Path) -> str:
    digest = sha256()
    root = Path(root)
    for name in RUNTIME_MEMBERS:
        payload = (root / name).read_bytes()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def render_colab_cell(root: Path) -> bytes:
    root = Path(root)
    archive = runtime_archive(root)
    encoded = base64.b64encode(archive).decode("ascii")
    archive_sha = sha256(archive).hexdigest()
    source = f'''from __future__ import annotations
import base64, hashlib, os, pathlib, subprocess, sys, tarfile, time, traceback
from io import BytesIO

SESSION_DEADLINE = time.time() + 10800
RUN_BASE = pathlib.Path("/content/catboost_tabm_blend/runs")
RUN_ROOT = RUN_BASE / f"{{time.time_ns()}}_{{os.getpid()}}"
RUN_ROOT.mkdir(parents=True, exist_ok=False)
RUNTIME_ROOT = RUN_ROOT / "runtime"
UPLOAD_ROOT = RUN_ROOT / "uploads"
OUTPUT_ROOT = RUN_ROOT / "campaign"
SNAPSHOT_ROOT = RUN_ROOT / "snapshots"
UPLOAD_ROOT.mkdir(); SNAPSHOT_ROOT.mkdir()
RUNTIME_B64 = "{encoded}"
EXPECTED_RUNTIME_SHA256 = "{archive_sha}"

def fail(stage, error):
    message = str(error).replace(" ", "_").replace("\\n", "_")[:500]
    print(f"BLEND_ERROR stage={{stage}} type={{type(error).__name__}} message={{message}}", flush=True)
    traceback.print_exc()

stage = "runtime"
try:
    payload = base64.b64decode(RUNTIME_B64, validate=True)
    observed = hashlib.sha256(payload).hexdigest()
    if observed != EXPECTED_RUNTIME_SHA256:
        raise RuntimeError("embedded runtime SHA-256 differs")
    RUNTIME_ROOT.mkdir()
    with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (RUNTIME_ROOT / member.name).resolve()
            if not member.isfile() or RUNTIME_ROOT.resolve() not in target.parents:
                raise RuntimeError("unsafe embedded runtime member")
        archive.extractall(RUNTIME_ROOT, members=members, filter="data")
    sys.path.insert(0, str(RUNTIME_ROOT))
    print(f"BLEND_CODE_READY sha256={{observed}} size_bytes={{len(payload)}}", flush=True)

    stage = "upload"
    import google.colab.files
    uploaded = google.colab.files.upload()
    paths = []
    for name, value in uploaded.items():
        safe = pathlib.Path(name).name
        if safe != name or not safe.lower().endswith(".zip"):
            raise RuntimeError("every upload must be a top-level ZIP")
        path = UPLOAD_ROOT / safe
        path.write_bytes(value)
        paths.append(path)

    stage = "dependencies"
    if time.time() >= SESSION_DEADLINE:
        raise RuntimeError("deadline expired before dependency installation")
    requirements = RUNTIME_ROOT / "experiments/catboost_tabm_blend/requirements-colab.txt"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)], check=True)
    import catboost
    if catboost.__version__ != "1.2.10":
        raise RuntimeError("CatBoost version differs")
    print("BLEND_DEPENDENCIES_READY catboost=1.2.10", flush=True)

    stage = "gpu"
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("one CUDA GPU is required")
    capability = torch.cuda.get_device_capability(0)
    if capability < (7, 5):
        raise RuntimeError(f"CUDA capability is too old: {{capability}}")
    print(f"BLEND_GPU_READY name={{torch.cuda.get_device_name(0)}} capability={{capability}}", flush=True)

    stage = "inputs"
    from experiments.catboost_tabm_blend.contracts import contract_sha256, load_contract
    from experiments.catboost_tabm_blend.runner import _bindings, code_sha256
    from experiments.catboost_tabm_blend.colab import (
        build_delivery, classify_and_verify_uploads, file_sha256,
        run_supervised_campaign, verify_delivery,
    )
    contract = load_contract()
    verified_input, verified_stage_c, resume = classify_and_verify_uploads(
        paths, run_root=RUN_ROOT / "verified", contract=contract,
        expected_contract_sha256=contract_sha256(), expected_code_sha256=code_sha256(),
    )
    print(f"BLEND_INPUTS_VERIFIED count={{len(paths)}}", flush=True)
    print(f"BLEND_RESUME_READY source={{resume if resume else 'fresh'}}", flush=True)

    def download_resume(path):
        print(f"BLEND_DOWNLOAD_REQUESTED path={{path}}", flush=True)
        google.colab.files.download(str(path))

    stage = "training"
    campaign_log = RUN_ROOT / "blend_campaign.log"
    result = run_supervised_campaign(
        verified_input=verified_input, verified_stage_c=verified_stage_c,
        output_dir=OUTPUT_ROOT, snapshot_dir=SNAPSHOT_ROOT,
        resume_bundle=resume, wall_deadline=SESSION_DEADLINE,
        on_verified_resume=download_resume, log_path=campaign_log,
    )
    if not result.stage_complete or result.bundles.review is None:
        print("BLEND_BUNDLE_SUCCESS stage_complete=false resume_only=true", flush=True)
    else:
        stage = "delivery"
        bindings = _bindings(verified_input, verified_stage_c)
        delivery = build_delivery(
            destination=RUN_ROOT / "catboost_tabm_blend_delivery.zip",
            campaign_log=campaign_log, review_bundle=result.bundles.review,
            resume_bundle=result.bundles.resume, bindings=bindings,
        )
        verify_delivery(delivery, expected_bindings=bindings)
        digest = file_sha256(delivery)
        print(f"BLEND_BUNDLE_SUCCESS review={{result.bundles.review}} resume={{result.bundles.resume}}", flush=True)
        print(f"BLEND_DELIVERY_READY path={{delivery}} sha256={{digest}}", flush=True)
        print(f"BLEND_DOWNLOAD_REQUESTED path={{delivery}}", flush=True)
        google.colab.files.download(str(delivery))
except Exception as error:
    fail(stage, error)
    raise
'''
    return source.encode("utf-8")
