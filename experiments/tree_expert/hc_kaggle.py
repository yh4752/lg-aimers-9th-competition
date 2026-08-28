from __future__ import annotations

import base64
from dataclasses import dataclass
from gzip import GzipFile
from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
import tarfile
import time
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from .e2_kaggle import runtime_member_names as e2_runtime_member_names
from .hc_contracts import contract_sha256, load_hc_contract


class HCKaggleError(ValueError):
    """Raised before ambiguous Kaggle inputs can start a campaign."""


_HC_MEMBERS = (
    "experiments/tree_expert/hc_contract.json",
    "experiments/tree_expert/hc_contracts.py",
    "experiments/tree_expert/hc_inputs.py",
    "experiments/tree_expert/hc_base.py",
    "experiments/tree_expert/hc_features.py",
    "experiments/tree_expert/hc_training.py",
    "experiments/tree_expert/hc_calibration.py",
    "experiments/tree_expert/hc_metrics.py",
    "experiments/tree_expert/hc_evaluation.py",
    "experiments/tree_expert/hc_decisions.py",
    "experiments/tree_expert/hc_state.py",
    "experiments/tree_expert/hc_artifacts.py",
    "experiments/tree_expert/hc_runner.py",
    "experiments/tree_expert/hc_full_fit.py",
    "experiments/tree_expert/hc_inference.py",
    "experiments/tree_expert/hc_production.py",
    "experiments/tree_expert/hc_kaggle.py",
)
_RUNTIME_MEMBERS = tuple(dict.fromkeys((*e2_runtime_member_names(), *_HC_MEMBERS)))


@dataclass(frozen=True)
class DiscoveredInputs:
    official_data: Path
    hc_input: Path
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
    source = Path(root)
    digest = sha256()
    for name in _RUNTIME_MEMBERS:
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise HCKaggleError(f"runtime source is absent: {name}")
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _manifest(path: Path) -> tuple[str, str] | None:
    source = Path(path)
    try:
        if source.is_dir():
            for name in ("manifest.json", "handoff_manifest.json"):
                candidate = source / name
                if candidate.is_file() and not candidate.is_symlink():
                    payload = candidate.read_bytes()
                    return str(json.loads(payload)["artifact_kind"]), sha256(payload).hexdigest()
            return None
        if source.is_file() and source.suffix.lower() == ".zip":
            with ZipFile(source) as archive:
                for name in ("manifest.json", "handoff_manifest.json"):
                    if name in archive.namelist():
                        payload = archive.read(name)
                        return str(json.loads(payload)["artifact_kind"]), sha256(payload).hexdigest()
    except Exception:
        return None
    return None


def _candidates(root: Path, kinds: set[str]) -> list[tuple[Path, str, str]]:
    found: list[tuple[Path, str, str]] = []
    expanded_roots: list[Path] = []
    for name in ("manifest.json", "handoff_manifest.json"):
        for path in sorted(Path(root).rglob(name)):
            identity = _manifest(path.parent)
            if identity is not None and identity[0] in kinds:
                found.append((path.parent, *identity))
                expanded_roots.append(path.parent.resolve())
    for path in sorted(Path(root).rglob("*.zip")):
        resolved = path.resolve()
        if any(resolved.is_relative_to(parent) for parent in expanded_roots):
            continue
        identity = _manifest(path)
        if identity is not None and identity[0] in kinds:
            found.append((path, *identity))
    unique: dict[tuple[str, str], tuple[Path, str, str]] = {}
    for item in found:
        unique.setdefault((item[1], item[2]), item)
    return list(unique.values())


def discover_inputs(
    root: Path,
    *,
    official_hashes: Mapping[str, str] | None = None,
) -> DiscoveredInputs:
    source = Path(root)
    if source.is_symlink() or not source.is_dir():
        raise HCKaggleError("Kaggle input root differs")
    contract = load_hc_contract()
    hashes = dict(contract.inputs if official_hashes is None else official_hashes)
    required = {"official_train_sha256", "official_history_sha256"}
    if not required.issubset(hashes):
        raise HCKaggleError("official hash keys differ")
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
        raise HCKaggleError(f"official data count must be one; found={len(official)}")
    compact = _candidates(source, {"tree_hierarchical_input_v1"})
    if len(compact) != 1:
        raise HCKaggleError(f"HC input count must be one; found={len(compact)}")
    previous = _candidates(source, {"tree_hierarchical_handoff_v1"})
    if len(previous) > 1:
        raise HCKaggleError(f"handoff count must be zero or one; found={len(previous)}")
    return DiscoveredInputs(
        official[0], compact[0][0], previous[0][0] if previous else None
    )


def verify_gpu(torch_module: object) -> tuple[str, ...]:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or cuda.device_count() not in {1, 2}:
        raise HCKaggleError("HC requires one or two CUDA devices")
    names = tuple(str(cuda.get_device_name(index)) for index in range(cuda.device_count()))
    if any("T4" not in name for name in names):
        raise HCKaggleError("HC requires Tesla T4 GPU devices")
    return names


def _zip_directory(source: Path, output: Path) -> Path:
    root = Path(source)
    if root.is_symlink() or not root.is_dir():
        raise HCKaggleError("expanded artifact differs")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise HCKaggleError("expanded artifact contains a symlink")
            if path.is_file():
                info = ZipInfo(path.relative_to(root).as_posix(), (2026, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
    temporary.replace(destination)
    return destination


def materialize_resume_from_handoff(source: Path, output: Path) -> Path:
    from .hc_artifacts import verify_handoff

    path = Path(source)
    handoff = path if path.is_file() else _zip_directory(path, output.with_name("handoff.zip"))
    verify_handoff(handoff)
    with ZipFile(handoff) as archive:
        payload = archive.read("tree_hierarchical_resume.zip")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return destination


def _runtime_archive(root: Path) -> bytes:
    source = Path(root)
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name in _RUNTIME_MEMBERS:
            path = source / name
            if path.is_symlink() or not path.is_file():
                raise HCKaggleError(f"runtime source is absent: {name}")
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
import base64, hashlib, importlib.metadata, io, shutil, subprocess, sys, tarfile, time
from pathlib import Path

RUNTIME_B64 = "{encoded}"
RUNTIME_ARCHIVE_SHA256 = "{archive_sha}"
RUNTIME_CODE_SHA256 = "{code_sha}"
TEMP_ROOT = Path("/kaggle/temp/tree_hierarchical")
CODE_ROOT = TEMP_ROOT / "runtime"
FINAL_HANDOFF = Path("/kaggle/working/tree_hierarchical_handoff.zip")
STAGE = "setup"
LOG_MARKERS = ("TREE_HC_CODE_READY", "TREE_HC_STAGE_SELECTED", "TREE_HC_JOB_START", "TREE_HC_TRAINING_PROGRESS", "TREE_HC_JOB_END", "TREE_HC_DECISION", "TREE_HC_ARTIFACT_READY", "TREE_HC_SUCCESS", "TREE_HC_ERROR")

def extract_runtime(payload: bytes) -> None:
    if hashlib.sha256(payload).hexdigest() != RUNTIME_ARCHIVE_SHA256:
        raise RuntimeError("runtime_archive_sha256_differs")
    CODE_ROOT.mkdir(parents=True, exist_ok=True)
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
    from experiments.tree_expert.hc_kaggle import runtime_identity_sha256
    if runtime_identity_sha256(CODE_ROOT) != RUNTIME_CODE_SHA256:
        raise RuntimeError("runtime_code_sha256_differs")
    print(f"TREE_HC_CODE_READY sha256={{RUNTIME_CODE_SHA256}} size_bytes={{len(payload)}}", flush=True)

    STAGE = "dependencies"
    required = {{"catboost": "1.2.10", "tabm": "0.0.3", "rtdl_num_embeddings": "0.0.12"}}
    for package, version in required.items():
        try:
            current = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            current = None
        if current != version:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"{{package}}=={{version}}"], check=True)
    print("TREE_HC_DEPENDENCIES_READY catboost=1.2.10 tabm=0.0.3 rtdl_num_embeddings=0.0.12", flush=True)

    STAGE = "inputs"
    from experiments.tree_expert.hc_kaggle import discover_inputs, verify_gpu
    found = discover_inputs(Path("/kaggle/input"))
    print(f"TREE_HC_INPUTS_FOUND official={{found.official_data}} input={{found.hc_input}} previous={{found.previous_handoff}}", flush=True)
    import torch
    names = verify_gpu(torch)
    print(f"TREE_HC_GPU_READY count={{len(names)}} names={{' | '.join(names)}}", flush=True)

    STAGE = "campaign"
    from experiments.tree_expert.hc_kaggle import run_hc_kaggle_campaign
    result = run_hc_kaggle_campaign(
        official_root=found.official_data,
        hc_input=found.hc_input,
        previous_handoff=found.previous_handoff,
        work_root=TEMP_ROOT,
        absolute_deadline=time.time() + 21600,
        gpu_count=len(names),
    )
    shutil.copy2(result.handoff, FINAL_HANDOFF)
    print(f"TREE_HC_SUCCESS stage={{result.completed_stage}} next={{result.next_stage}} status={{result.status}} handoff={{FINAL_HANDOFF}}", flush=True)
except Exception as error:
    emergency = TEMP_ROOT / "bundles/tree_hierarchical_handoff.zip"
    if emergency.is_file():
        shutil.copy2(emergency, FINAL_HANDOFF)
        print(f"TREE_HC_ARTIFACT_READY path={{FINAL_HANDOFF}} status=emergency", flush=True)
    print(f"TREE_HC_ERROR stage={{STAGE}} type={{type(error).__name__}} message={{str(error).replace(' ', '_')}}", flush=True)
    raise
'''


def build_hc_kaggle_cell(output: Path, root: Path | None = None) -> Path:
    repository = Path(__file__).resolve().parents[2] if root is None else Path(root)
    archive = _runtime_archive(repository)
    source = _cell_source(
        base64.b64encode(archive).decode("ascii"),
        sha256(archive).hexdigest(),
        runtime_identity_sha256(repository),
    )
    if len(source.encode()) >= 1_000_000:
        raise HCKaggleError("generated Kaggle cell reaches one megabyte")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8")
    return destination


def run_hc_kaggle_campaign(
    *,
    official_root: Path,
    hc_input: Path,
    previous_handoff: Path | None,
    work_root: Path,
    absolute_deadline: float,
    gpu_count: int,
) -> object:
    from .e2_inputs import verify_official_data
    from .hc_artifacts import (
        create_handoff,
        create_resume_bundle,
        create_review_bundle,
        restore_resume_bundle,
        verify_resume_bundle,
    )
    from .hc_inputs import verify_and_extract_hc_input
    from .hc_production import HCProductionRuntime
    from .hc_runner import run_campaign
    from .hc_state import HCBindings, initial_state, state_from_payload

    root = Path(work_root)
    campaign = root / "campaign"
    bundles = root / "bundles"
    verified_data = verify_official_data(Path(official_root))
    verified_input = verify_and_extract_hc_input(
        Path(hc_input), root / "verified_input"
    )
    bindings = HCBindings(
        contract_sha256=contract_sha256(),
        code_sha256=runtime_identity_sha256(Path(__file__).resolve().parents[2]),
        input_manifest_sha256=verified_input.manifest_sha256,
        train_sha256=verified_data.train_sha256,
        history_sha256=verified_data.history_sha256,
        e2_handoff_sha256=verified_input.e2_handoff_sha256,
    )
    if previous_handoff is None:
        state = initial_state(bindings)
    else:
        resume = materialize_resume_from_handoff(
            Path(previous_handoff), root / "inputs/tree_hierarchical_resume.zip"
        )
        verify_resume_bundle(resume, bindings)
        restore_resume_bundle(resume, campaign, bindings)
        state = state_from_payload(
            json.loads((campaign / "state/state.json").read_text(encoding="utf-8"))
        )
    log = campaign / "tree_hierarchical.log"
    runtime = HCProductionRuntime(
        data=verified_data,
        evidence=verified_input,
        output=campaign,
        contract=load_hc_contract(),
    )
    try:
        return run_campaign(
            runtime=runtime,
            state=state,
            campaign_root=campaign,
            bundle_root=bundles,
            log_path=log,
            wall_deadline=absolute_deadline,
            gpu_count=gpu_count,
        )
    except Exception:
        state_path = campaign / "states/state.json"
        if state_path.is_file():
            latest = state_from_payload(json.loads(state_path.read_text(encoding="utf-8")))
            review = create_review_bundle(
                sources={}, state=latest, output=bundles / "tree_hierarchical_review.zip"
            )
            resume = create_resume_bundle(
                campaign,
                bundles / "tree_hierarchical_resume.zip",
                bindings,
                latest,
            )
            create_handoff(
                review=review,
                resume=resume,
                log=log,
                output=bundles / "tree_hierarchical_handoff.zip",
                status="error",
            )
        raise
