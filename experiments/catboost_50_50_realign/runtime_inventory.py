from __future__ import annotations

import base64
import gzip
from hashlib import sha256
import io
from pathlib import Path
import tarfile


RUNTIME_MEMBERS = (
    "experiments/catboost_50_50_realign/artifacts.py",
    "experiments/catboost_50_50_realign/colab.py",
    "experiments/catboost_50_50_realign/contract.json",
    "experiments/catboost_50_50_realign/contracts.py",
    "experiments/catboost_50_50_realign/inputs.py",
    "experiments/catboost_50_50_realign/metrics.py",
    "experiments/catboost_50_50_realign/requirements-colab.txt",
    "experiments/catboost_50_50_realign/runner.py",
    "experiments/catboost_50_50_realign/runtime_inventory.py",
    "experiments/catboost_50_50_realign/state.py",
    "experiments/catboost_50_50_realign/tabm_fold.py",
    "experiments/catboost_50_50_realign/training.py",
    "experiments/catboost_deployment/artifacts.py",
    "experiments/catboost_deployment/contract.json",
    "experiments/catboost_deployment/contracts.py",
    "experiments/catboost_deployment/metrics.py",
    "experiments/catboost_deployment/state.py",
    "experiments/catboost_preprocessing/features.py",
    "experiments/catboost_tabm_blend/contract.json",
    "experiments/catboost_tabm_blend/contracts.py",
    "experiments/catboost_tabm_blend/inputs.py",
    "experiments/catboost_tabm_blend/metrics.py",
    "experiments/independent_dl/feature_sources/trackman.py",
    "experiments/independent_dl/features.py",
    "experiments/independent_dl/models/common.py",
    "experiments/independent_dl/models/tabm.py",
    "experiments/independent_dl/preprocessing.py",
    "experiments/independent_dl/progress.py",
    "experiments/independent_dl/row_features.py",
    "experiments/independent_dl/training.py",
    "experiments/oof_reset_audit/metrics.py",
    "experiments/oof_reset_audit/types.py",
    "experiments/tabm_campaign/artifacts.py",
    "experiments/tabm_campaign/cache.py",
    "experiments/tabm_campaign/row_feature_contracts.py",
    "experiments/tabm_campaign/row_feature_decisions.py",
    "experiments/tabm_campaign/row_feature_proxy.py",
    "experiments/tabm_campaign/row_feature_runtime.py",
    "experiments/tabm_campaign/sampling.py",
    "experiments/tabm_campaign/training.py",
    "experiments/tabm_campaign/worker.py",
    "experiments/tabm_campaign/configs/row_feature_proxy_v1.json",
    "experiments/tabm_campaign/requirements-kaggle.txt",
)


def code_identity_sha256(root: Path) -> str:
    digest = sha256()
    for name in RUNTIME_MEMBERS:
        digest.update(name.encode("utf-8"))
        digest.update((Path(root) / name).read_bytes())
    return digest.hexdigest()


def runtime_archive(root: Path) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name in RUNTIME_MEMBERS:
                payload = (Path(root) / name).read_bytes()
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mode = 0o600
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def render_colab_cell(root: Path) -> bytes:
    encoded = base64.b64encode(runtime_archive(root)).decode("ascii")
    template = f'''from __future__ import annotations
import base64, io, os, subprocess, sys, tarfile, time
from pathlib import Path

SESSION_DEADLINE = time.time() + 10800
BASE_ROOT = Path("/content/catboost_50_50_realign")
RUN_ROOT = BASE_ROOT / "runs" / f"{{time.time_ns()}}_{{os.getpid()}}"
CODE_ROOT = RUN_ROOT / "runtime"
CODE_ROOT.mkdir(parents=True, exist_ok=False)

ARCHIVE_B64 = "{encoded}"
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(ARCHIVE_B64)), mode="r:gz") as archive:
    archive.extractall(CODE_ROOT, filter="data")
sys.path.insert(0, str(CODE_ROOT))
for module_name in tuple(sys.modules):
    if module_name == "experiments" or module_name.startswith("experiments."):
        del sys.modules[module_name]

try:
    from google.colab import files
    cache = globals().get("_REALIGN_UPLOAD_CACHE")
    if cache and all(Path(path).is_file() for path in cache):
        upload_paths = [Path(path) for path in cache]
        print(f"REALIGN_UPLOAD_CACHE_REUSED count={{len(upload_paths)}}", flush=True)
    else:
        uploaded = files.upload()
        upload_paths = [Path(name).resolve() for name in uploaded]
        globals()["_REALIGN_UPLOAD_CACHE"] = [str(path) for path in upload_paths]

    requirements = CODE_ROOT / "experiments/catboost_50_50_realign/requirements-colab.txt"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)], check=True)
    import catboost, torch
    if catboost.__version__ != "1.2.10":
        raise RuntimeError(f"catboost version differs: {{catboost.__version__}}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one CUDA GPU is required")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name:
        raise RuntimeError(f"Tesla T4 is required: {{gpu_name}}")
    print(f"REALIGN_GPU_READY name={{gpu_name}}", flush=True)

    from experiments.catboost_50_50_realign.colab import _artifact_kind, classify_and_verify_uploads, run_supervised_campaign
    verified, resume = classify_and_verify_uploads(upload_paths, run_root=RUN_ROOT / "inputs")
    latest = {{"path": resume}}
    def download_event(event):
        print(f"REALIGN_DOWNLOAD_REQUESTED phase={{event.phase}} path={{event.path}}", flush=True)
        files.download(str(event.path))
        if "resume" in event.phase or event.phase == "f1_complete":
            latest["path"] = event.path
            input_paths = [path for path in upload_paths if _artifact_kind(path) == "catboost_50_50_realign_input_v1"]
            globals()["_REALIGN_UPLOAD_CACHE"] = [str(input_paths[0]), str(event.path)]
            print(f"REALIGN_RESUME_CACHE_READY path={{event.path}}", flush=True)
    result = run_supervised_campaign(
        campaign_kwargs={{
            "verified": verified,
            "output_dir": RUN_ROOT / "campaign",
            "resume_bundle": resume,
            "absolute_deadline": SESSION_DEADLINE - 300,
        }},
        download=download_event,
        latest_verified_resume=lambda: latest["path"],
    )
    selected = "none" if result.selected_tree_count is None else result.selected_tree_count
    print(f"REALIGN_CAMPAIGN_SUCCESS status={{result.status}} selected_tree_count={{selected}}", flush=True)
except Exception as error:
    print(f"REALIGN_ERROR stage=campaign type={{type(error).__name__}} message={{str(error).replace(' ', '_')}}", flush=True)
    raise
'''
    return template.encode("utf-8")
