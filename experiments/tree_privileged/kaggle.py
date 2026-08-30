"""Kaggle discovery, resume restoration and deterministic one-cell renderer."""
from __future__ import annotations

import ast
import base64
from dataclasses import dataclass
from gzip import GzipFile
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from experiments.tree_expert.inputs import VerifiedOfficialData

from .artifacts import verify_bundle
from .contracts import load_contract


class PrivilegedKaggleError(ValueError):
    pass


@dataclass(frozen=True)
class DiscoveredInputs:
    official_data: Path
    campaign_input: Path
    previous_handoff: Path | None


def _file_sha(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def verify_official_data(root: Path, *, hashes: Mapping[str, str] | None = None) -> VerifiedOfficialData:
    source = Path(root)
    expected = dict(load_contract().inputs if hashes is None else hashes)
    names = ("train.csv", "test.csv", "trackman_history.csv", "sample_submission.csv")
    if source.is_symlink() or not source.is_dir() or any(not (source / name).is_file() or (source / name).is_symlink() for name in names):
        raise PrivilegedKaggleError("official data member set differs")
    train, history = source / "train.csv", source / "trackman_history.csv"
    train_sha, history_sha = _file_sha(train), _file_sha(history)
    if train_sha != expected["official_train_sha256"] or history_sha != expected["official_history_sha256"]:
        raise PrivilegedKaggleError("official data SHA-256 differs")
    return VerifiedOfficialData(source.resolve(), train, history, train_sha, history_sha)


def _manifest(path: Path) -> tuple[str, str] | None:
    try:
        if path.is_dir() and not path.is_symlink(): payload = (path / "manifest.json").read_bytes()
        elif path.is_file() and path.suffix.lower() == ".zip":
            with ZipFile(path) as archive: payload = archive.read("manifest.json")
        else: return None
        value = json.loads(payload)
        return str(value["artifact_kind"]), sha256(payload).hexdigest()
    except Exception: return None


def _logical(root: Path, kind: str) -> tuple[Path, ...]:
    identities: dict[str, Path] = {}; expanded = []
    for manifest in sorted(Path(root).rglob("manifest.json")):
        identity = _manifest(manifest.parent)
        if identity is not None and identity[0] == kind:
            identities.setdefault(identity[1], manifest.parent); expanded.append(manifest.parent.resolve())
    for archive in sorted(Path(root).rglob("*.zip")):
        resolved = archive.resolve()
        if any(resolved.is_relative_to(parent) for parent in expanded): continue
        identity = _manifest(archive)
        if identity is not None and identity[0] == kind: identities.setdefault(identity[1], archive)
    return tuple(identities.values())


def discover_inputs(root: Path, *, official_hashes: Mapping[str, str] | None = None) -> DiscoveredInputs:
    source = Path(root); expected = dict(load_contract().inputs if official_hashes is None else official_hashes)
    if source.is_symlink() or not source.is_dir(): raise PrivilegedKaggleError("Kaggle input root differs")
    official = []
    for train in sorted(source.rglob("train.csv")):
        parent = train.parent
        try: verify_official_data(parent, hashes=expected)
        except PrivilegedKaggleError: continue
        official.append(parent)
    if len(official) != 1: raise PrivilegedKaggleError(f"official data count must be one; found={len(official)}")
    compact = _logical(source, "tree_privileged_input_v1")
    if len(compact) != 1: raise PrivilegedKaggleError(f"campaign input count must be one; found={len(compact)}")
    previous = _logical(source, "tree_privileged_handoff_v1")
    if len(previous) > 1: raise PrivilegedKaggleError(f"previous handoff count must be zero or one; found={len(previous)}")
    return DiscoveredInputs(official[0], compact[0], previous[0] if previous else None)


def verify_gpu(torch_module: object) -> tuple[str, str]:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or cuda.device_count() != 2: raise PrivilegedKaggleError("campaign requires exactly two CUDA devices")
    names = tuple(str(cuda.get_device_name(index)) for index in range(2))
    if any("T4" not in name for name in names): raise PrivilegedKaggleError("campaign requires Tesla T4 x2")
    return names  # type: ignore[return-value]


def _zip_directory(source: Path, destination: Path) -> Path:
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(Path(source).rglob("*")):
            if path.is_symlink(): raise PrivilegedKaggleError("expanded artifact contains a symlink")
            if path.is_file():
                name = path.relative_to(source).as_posix(); info = ZipInfo(name, (2026, 1, 1, 0, 0, 0)); info.compress_type = ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())
    return destination


def _materialize(path: Path, destination: Path) -> Path:
    return Path(path) if Path(path).is_file() else _zip_directory(Path(path), Path(destination))


def restore_previous_handoff(source: Path, campaign_root: Path, bindings: Mapping[str, str]) -> None:
    temporary_root = Path(tempfile.mkdtemp(prefix="tree_priv_restore_"))
    try:
        handoff = _materialize(Path(source), temporary_root / "handoff.zip")
        verify_bundle(handoff, kind="tree_privileged_handoff_v1", expected_bindings=bindings)
        with ZipFile(handoff) as archive: resume_payload = archive.read("resume.bundle")
        resume = temporary_root / "resume.zip"; resume.write_bytes(resume_payload)
        verify_bundle(resume, kind="tree_privileged_resume_v1", expected_bindings=bindings)
        with ZipFile(resume) as archive:
            for info in archive.infolist():
                if info.filename == "manifest.json": continue
                name = PurePosixPath(info.filename); target = (Path(campaign_root) / name).resolve()
                if name.is_absolute() or any(part in {"", ".", ".."} for part in name.parts) or not target.is_relative_to(Path(campaign_root).resolve()):
                    raise PrivilegedKaggleError("unsafe resume member")
                target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(archive.read(info))
    finally:
        shutil.rmtree(temporary_root)


def _module_path(root: Path, module: str) -> Path | None:
    file_path = root / (module.replace(".", "/") + ".py")
    if file_path.is_file(): return file_path
    init = root / module.replace(".", "/") / "__init__.py"
    return init if init.is_file() else None


def runtime_member_names(root: Path) -> tuple[str, ...]:
    root = Path(root)
    seeds = list((root / "experiments/tree_privileged").glob("*.py"))
    queue = [path for path in seeds if path.name != "KAGGLE_CELL.py"]
    found = {path.resolve() for path in queue}
    while queue:
        path = queue.pop()
        try: tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as error: raise PrivilegedKaggleError(f"runtime source cannot be parsed: {path}") from error
        package = path.relative_to(root).with_suffix("").parts
        if package[-1] == "__init__": package = package[:-1]
        else: package = package[:-1]
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                base = list(package)
                if node.level:
                    base = base[:len(base) - node.level + 1]
                    module = ".".join([*base, *(node.module or "").split(".")])
                else: module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("experiments."):
                        target = _module_path(root, alias.name)
                        if target is not None and target.resolve() not in found: found.add(target.resolve()); queue.append(target)
                continue
            if module and module.startswith("experiments"):
                target = _module_path(root, module)
                if target is not None and target.resolve() not in found: found.add(target.resolve()); queue.append(target)
    for name in ("experiments/__init__.py", "experiments/tree_privileged/contract.json",
                 "experiments/tree_expert/e1_contract.json", "experiments/tree_expert/e2_contract.json"):
        path = root / name
        if path.is_file(): found.add(path.resolve())
    return tuple(sorted(path.relative_to(root).as_posix() for path in found))


def runtime_identity_sha256(root: Path) -> str:
    digest = sha256()
    for name in runtime_member_names(root): digest.update(name.encode()); digest.update(b"\0"); digest.update((Path(root) / name).read_bytes())
    return digest.hexdigest()


def _runtime_archive(root: Path) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name in runtime_member_names(root):
            payload = (Path(root) / name).read_bytes(); info = tarfile.TarInfo(name); info.size = len(payload); info.mtime = 0; info.mode = 0o644; info.uid = info.gid = 0; info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
    compressed = io.BytesIO()
    with GzipFile(fileobj=compressed, mode="wb", mtime=0) as handle: handle.write(raw.getvalue())
    return compressed.getvalue()


def _cell_source(encoded: str, archive_sha: str, code_sha: str) -> str:
    return f'''from __future__ import annotations
import base64, hashlib, importlib.metadata, io, json, shutil, subprocess, sys, tarfile, time
from pathlib import Path

RUNTIME_B64 = "{encoded}"
RUNTIME_SHA256 = "{archive_sha}"
CODE_SHA256 = "{code_sha}"
CODE_ROOT = Path("/kaggle/working/tree_privileged_runtime_{code_sha[:12]}")
CAMPAIGN_ROOT = Path("/kaggle/working/tree_privileged") / CODE_SHA256[:12]
STAGE = "setup"

try:
    payload = base64.b64decode(RUNTIME_B64, validate=True)
    if hashlib.sha256(payload).hexdigest() != RUNTIME_SHA256:
        raise RuntimeError("runtime_archive_sha256_differs")
    if CODE_ROOT.exists():
        shutil.rmtree(CODE_ROOT)
    CODE_ROOT.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            target = (CODE_ROOT / member.name).resolve()
            if not member.isfile() or not target.is_relative_to(CODE_ROOT.resolve()):
                raise RuntimeError("unsafe_runtime_member")
        archive.extractall(CODE_ROOT, filter="data")
    STAGE = "dependencies"
    for package, version in {{"catboost": "1.2.10", "scipy": "1.16.3"}}.items():
        try: current = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: current = None
        if current != version:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"{{package}}=={{version}}"], check=True)
    print("TREE_PRIV_DEPENDENCIES_READY catboost=1.2.10 scipy=1.16.3", flush=True)

    sys.path.insert(0, str(CODE_ROOT))
    from experiments.tree_privileged.kaggle import runtime_identity_sha256
    if runtime_identity_sha256(CODE_ROOT) != CODE_SHA256:
        raise RuntimeError("runtime_code_sha256_differs")
    print(f"TREE_PRIV_CODE_READY sha256={{CODE_SHA256}} size_bytes={{len(payload)}}", flush=True)

    STAGE = "inputs"
    import torch
    from experiments.tree_privileged.contracts import load_contract
    from experiments.tree_privileged.inputs import verify_and_extract_input
    from experiments.tree_privileged.kaggle import discover_inputs, restore_previous_handoff, verify_gpu, verify_official_data
    from experiments.tree_privileged.runner import VerifiedCampaignInputs, _bindings, run_campaign
    found = discover_inputs(Path("/kaggle/input"))
    names = verify_gpu(torch)
    official = verify_official_data(found.official_data)
    verified_root = Path("/kaggle/working/tree_privileged_verified_input")
    if verified_root.exists():
        shutil.rmtree(verified_root)
    verified_input = verify_and_extract_input(found.campaign_input, verified_root)
    verified = VerifiedCampaignInputs(official, verified_input)
    if found.previous_handoff is not None:
        restore_previous_handoff(found.previous_handoff, CAMPAIGN_ROOT, _bindings(verified))
        print(f"TREE_PRIV_RESUME_READY source={{found.previous_handoff}}", flush=True)
    print(f"TREE_PRIV_INPUTS_VERIFIED official={{found.official_data}} input={{found.campaign_input}}", flush=True)
    print(f"TREE_PRIV_GPU_READY count=2 names={{' | '.join(names)}}", flush=True)

    STAGE = "campaign"
    deadline = time.time() + load_contract().runtime.wall_seconds
    result = run_campaign(verified, CAMPAIGN_ROOT, wall_deadline=deadline, gpu_ids=(0, 1))
    handoff = result.bundles.handoff
    print(f"TREE_PRIV_HANDOFF_READY path={{handoff}} sha256={{hashlib.sha256(handoff.read_bytes()).hexdigest()}}", flush=True)
    print(f"TREE_PRIV_CAMPAIGN_SUCCESS status={{result.status}} phase={{result.phase}} accepted={{','.join(result.accepted_candidates) or 'none'}}", flush=True)
    from IPython.display import FileLink, display
    class _KaggleFiles:
        @staticmethod
        def download(path):
            display(FileLink(path))
    files = _KaggleFiles()
    files.download(str(handoff))
except Exception as error:
    print(f"TREE_PRIV_ERROR stage={{STAGE}} type={{type(error).__name__}} message={{str(error).replace(' ', '_')}}", flush=True)
    raise
'''


def build_kaggle_cell(output: Path, *, root: Path | None = None) -> Path:
    repository = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    archive = _runtime_archive(repository); code_sha = runtime_identity_sha256(repository)
    source = _cell_source(base64.b64encode(archive).decode("ascii"), sha256(archive).hexdigest(), code_sha)
    if len(source.encode()) >= 1_000_000: raise PrivilegedKaggleError("generated Kaggle cell exceeds one megabyte")
    destination = Path(output); destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(source, encoding="utf-8")
    return destination
