"""Build the self-contained Kaggle launcher for temporal T2-B."""
from __future__ import annotations

import base64
import os
from pathlib import Path

from .platform import _runtime_archive


class T2BPlatformError(RuntimeError):
    pass


_TEMPLATE = '''from __future__ import annotations
import base64, io, json, os, shutil, subprocess, sys, tarfile, time, traceback
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

RUNTIME_B64 = "__RUNTIME_B64__"
WORK_ROOT = Path("/kaggle/working/temporal_portfolio_t2b")
RUNTIME_ROOT = Path("/kaggle/working/temporal_portfolio_t2b_runtime")
INPUT_ROOT = Path("/kaggle/input")
LOG_PATH = WORK_ROOT / "run.log"

class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, value):
        for stream in self.streams:
            stream.write(value); stream.flush()
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()

def unpack_runtime():
    if RUNTIME_ROOT.exists(): shutil.rmtree(RUNTIME_ROOT)
    RUNTIME_ROOT.mkdir(parents=True)
    payload = base64.b64decode(RUNTIME_B64, validate=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if member.name.startswith("/") or ".." in parts or not member.isfile():
                raise RuntimeError("T2B_runtime_archive_contains_unsafe_member")
        archive.extractall(RUNTIME_ROOT)
    sys.path.insert(0, str(RUNTIME_ROOT))
    print(f"T2B_CODE_READY size_bytes={len(payload)}", flush=True)

def ensure_dependencies():
    try:
        import tabm, rtdl_num_embeddings
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "tabm==0.0.3", "rtdl-num-embeddings==0.0.12"],
            check=True,
        )
        import tabm, rtdl_num_embeddings
    if getattr(tabm, "__version__", None) != "0.0.3" or getattr(rtdl_num_embeddings, "__version__", None) != "0.0.12":
        raise RuntimeError("T2B_dependency_versions_differ")
    print("T2B_DEPENDENCIES_READY tabm=0.0.3 rtdl_num_embeddings=0.0.12", flush=True)

def artifact_kind(path):
    try:
        if path.is_file() and path.suffix.casefold() == ".zip":
            with ZipFile(path) as archive:
                names = set(archive.namelist())
                marker = "handoff_manifest.json" if "handoff_manifest.json" in names else "manifest.json"
                return json.loads(archive.read(marker)).get("artifact_kind")
        for marker in ("handoff_manifest.json", "manifest.json"):
            candidate = path / marker
            if path.is_dir() and candidate.is_file():
                return json.loads(candidate.read_text()).get("artifact_kind")
    except Exception: return None
    return None

def artifact_candidates():
    candidates = set(INPUT_ROOT.rglob("*.zip"))
    candidates.update(path.parent for path in INPUT_ROOT.rglob("manifest.json"))
    candidates.update(path.parent for path in INPUT_ROOT.rglob("handoff_manifest.json"))
    return sorted(candidates)

def find_kind(kind, *, required):
    found = [path for path in artifact_candidates() if artifact_kind(path) == kind]
    nested = {path for parent in found if parent.is_dir() for path in found if path != parent and parent in path.parents}
    found = [path for path in found if path not in nested]
    expected = "1" if required else "0_or_1"
    if (required and len(found) != 1) or (not required and len(found) > 1):
        raise RuntimeError(f"{kind}_candidate_count_must_be_{expected} found={len(found)}")
    return found[0] if found else None

def find_data_root():
    required = {"train.csv", "test.csv", "trackman_history.csv", "sample_submission.csv"}
    found = []
    for current, directories, files in os.walk(INPUT_ROOT):
        directories[:] = sorted(directories)
        if required.issubset(files): found.append(Path(current))
    if len(found) != 1:
        raise RuntimeError(f"official_data_candidate_count_must_be_1 found={len(found)}")
    return found[0]

def normalize_input(path):
    if path.is_file(): return path
    manifest = json.loads((path / "manifest.json").read_text())
    names = ["manifest.json", *sorted(manifest.get("members", {}))]
    destination = WORK_ROOT / "temporal_portfolio_t2b_input.zip"
    temporary = destination.with_suffix(".zip.tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        for name in names: archive.write(path / name, name)
    os.replace(temporary, destination)
    return destination

def find_resume():
    direct = find_kind("temporal_t2b_resume_v1", required=False)
    handoff = find_kind("temporal_t2b_handoff_v1", required=False)
    if direct is not None and handoff is not None:
        if handoff.is_dir() and handoff in direct.parents: return direct
        raise RuntimeError("T2B_resume_sources_are_ambiguous")
    if direct is not None or handoff is None: return direct
    if handoff.is_dir():
        candidate = handoff / "resume.zip"
        if not candidate.is_file(): raise RuntimeError("T2B_handoff_resume_is_missing")
        return candidate
    destination = WORK_ROOT / "resume_from_handoff.zip"
    with ZipFile(handoff) as archive: destination.write_bytes(archive.read("resume.zip"))
    return destination

def handoff(review, resume, state, destination):
    members = {
        "review.zip": Path(review).read_bytes(),
        "resume.zip": Path(resume).read_bytes(),
        "run.log": LOG_PATH.read_bytes(),
        "stage_summary.json": Path(state).read_bytes(),
    }
    records = {
        name: {"size_bytes": len(data), "sha256": __import__("hashlib").sha256(data).hexdigest()}
        for name, data in sorted(members.items())
    }
    marker = json.dumps(
        {"artifact_kind": "temporal_t2b_handoff_v1", "members": records},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    temporary = destination.with_suffix(".zip.tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        archive.writestr("handoff_manifest.json", marker)
        for name, data in sorted(members.items()): archive.writestr(name, data)
    os.replace(temporary, destination)
    return destination

WORK_ROOT.mkdir(parents=True, exist_ok=True)
log_stream = LOG_PATH.open("a", encoding="utf-8", buffering=1)
sys.stdout = Tee(sys.__stdout__, log_stream)
sys.stderr = Tee(sys.__stderr__, log_stream)
unpack_runtime()
stage = "dependencies"
prepared = None
try:
    ensure_dependencies()
    stage = "inputs"
    from experiments.temporal_portfolio.inputs import verify_official_data
    from experiments.temporal_portfolio.t2b_artifacts import restore_t2b_resume_source, write_t2b_bundles
    from experiments.temporal_portfolio.t2b_input import verify_t2b_input
    from experiments.temporal_portfolio.t2b_runner import run_t2b_stage

    data_root = find_data_root()
    prepared = normalize_input(find_kind("temporal_t2b_input_v1", required=True))
    resume = find_resume()
    verified = verify_official_data(data_root)
    bound = verify_t2b_input(prepared)
    print(f"T2B_INPUTS_VERIFIED data={data_root} input_sha256={bound.archive_sha256}", flush=True)
    if resume is not None and not ((WORK_ROOT / "jobs").is_dir() and any((WORK_ROOT / "jobs").iterdir())):
        restore_t2b_resume_source(resume, WORK_ROOT)
        print(f"T2B_RESUME_READY source={resume}", flush=True)
    elif resume is not None:
        print(f"T2B_WORKING_CACHE_REUSED resume_source={resume}", flush=True)
    else:
        print("T2B_RESUME_READY source=None", flush=True)
    stage = "training"
    result = run_t2b_stage(
        verified=verified, t2b_input=prepared, output_root=WORK_ROOT,
        deadline=time.time() + 14400,
    )
    stage = "bundles"
    bundles = write_t2b_bundles(
        WORK_ROOT / "bundles", jobs_root=WORK_ROOT / "jobs",
        completed=result.completed, pending=result.pending, failed=result.failed,
        evidence=result.evidence, parent_sha256=bound.t2a_decision_sha256,
    )
    delivery = handoff(
        bundles.review, bundles.resume, WORK_ROOT / "t2b_stage_result.json",
        Path("/kaggle/working/temporal_t2b_handoff.zip"),
    )
    print(f"T2B_HANDOFF_READY status={result.status} path={delivery}", flush=True)
except Exception as error:
    print(f"T2B_PORTFOLIO_ERROR stage={stage} type={type(error).__name__} message={str(error).replace(' ', '_')}", flush=True)
    traceback.print_exc()
    try:
        if prepared is None: raise RuntimeError("T2B_verified_input_is_unavailable")
        from experiments.temporal_portfolio.t1_artifacts import verify_compact_result
        from experiments.temporal_portfolio.t2b_artifacts import write_t2b_bundles
        from experiments.temporal_portfolio.t2b_input import verify_t2b_input
        from experiments.temporal_portfolio.worker import verify_worker_result
        completed, pending = [], []
        roots = sorted((WORK_ROOT / "jobs").glob("t2b__*")) if (WORK_ROOT / "jobs").is_dir() else []
        for root in roots:
            try:
                if (root / "worker_result.json").is_file(): verify_worker_result(root)
                else: verify_compact_result(root)
                completed.append(root.name)
            except Exception: pending.append(root.name)
        bound = verify_t2b_input(prepared)
        state = WORK_ROOT / "t2b_emergency_state.json"
        state.write_text(json.dumps(
            {"schema_version": 1, "stage": "T2B", "status": "failed", "completed": completed,
             "pending": pending, "failed": [], "error": type(error).__name__, "evidence": {}},
            sort_keys=True, separators=(",", ":"),
        ))
        bundles = write_t2b_bundles(
            WORK_ROOT / "emergency", jobs_root=WORK_ROOT / "jobs",
            completed=tuple(completed), pending=tuple(pending), failed=(), evidence={},
            parent_sha256=bound.t2a_decision_sha256,
        )
        delivery = handoff(
            bundles.review, bundles.resume, state,
            Path("/kaggle/working/temporal_t2b_emergency_handoff.zip"),
        )
        print(f"T2B_EMERGENCY_HANDOFF_READY path={delivery}", flush=True)
    except Exception as bundle_error:
        print(f"T2B_EMERGENCY_ERROR type={type(bundle_error).__name__} message={str(bundle_error).replace(' ', '_')}", flush=True)
    raise
finally:
    log_stream.flush()
'''


def build_t2b_kaggle_cell(output: str | Path) -> Path:
    archive = _runtime_archive()
    rendered = _TEMPLATE.replace(
        "__RUNTIME_B64__", base64.b64encode(archive).decode("ascii")
    )
    data = rendered.encode("utf-8")
    if len(data) >= 1_000_000:
        raise T2BPlatformError("generated T2-B Kaggle cell exceeds one megabyte")
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, destination)
    return destination
