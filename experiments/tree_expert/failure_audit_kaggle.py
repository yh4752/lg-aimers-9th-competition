from __future__ import annotations

import base64
import gzip
from hashlib import sha256
import io
from pathlib import Path
import tarfile

from .failure_audit_contracts import FailureAuditContract, load_failure_audit_contract


class FailureAuditKaggleError(ValueError):
    pass


_RUNTIME_MEMBERS = (
    "experiments/tree_expert/__init__.py",
    "experiments/tree_expert/contracts.py",
    "experiments/tree_expert/failure_labels.py",
    "experiments/tree_expert/failure_audit_contract.json",
    "experiments/tree_expert/failure_audit_contracts.py",
    "experiments/tree_expert/failure_audit.py",
    "experiments/tree_expert/failure_audit_artifacts.py",
    "experiments/tree_expert/failure_audit_kaggle.py",
)


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
                    raise FailureAuditKaggleError(f"runtime member is missing: {name}")
                payload = path.read_bytes()
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mode = 0o644
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def file_sha256(path: Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FailureAuditKaggleError(f"official file is not regular: {source}")
    digest = sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_official_hashes(root: Path, contract: FailureAuditContract) -> None:
    if file_sha256(root / "train.csv") != contract.official_train_sha256:
        raise FailureAuditKaggleError("official train SHA-256 differs")
    if file_sha256(root / "trackman_history.csv") != contract.official_history_sha256:
        raise FailureAuditKaggleError("official history SHA-256 differs")


def discover_official_data(root: Path, *, testing: bool = False) -> Path:
    source = Path(root)
    if source.is_symlink() or not source.is_dir():
        raise FailureAuditKaggleError("Kaggle input root differs")
    candidates = sorted(
        {
            train.parent
            for train in source.rglob("train.csv")
            if not train.is_symlink()
            and train.is_file()
            and not train.parent.is_symlink()
            and (train.parent / "trackman_history.csv").is_file()
            and not (train.parent / "trackman_history.csv").is_symlink()
        },
        key=str,
    )
    if len(candidates) != 1:
        raise FailureAuditKaggleError(
            f"official data count must be one; found={len(candidates)}"
        )
    if not testing:
        _verify_official_hashes(candidates[0], load_failure_audit_contract())
    return candidates[0]


def build_failure_audit_cell(output: Path, root: Path | None = None) -> Path:
    payload = runtime_archive(root)
    encoded = base64.b64encode(payload).decode("ascii")
    digest = sha256(payload).hexdigest()
    source = f'''from __future__ import annotations
import base64, hashlib, io, resource, shutil, sys, tarfile, time, traceback
from pathlib import Path, PurePosixPath

RUNTIME_B64 = "{encoded}"
RUNTIME_SHA256 = "{digest}"
STAGE = "setup"
OUTPUT = Path("/kaggle/working/failure_expert_label_audit_review.zip")

def safe_extract(payload, destination):
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) > 32 or sum(item.size for item in members) > 4 * 1024 * 1024:
            raise RuntimeError("runtime archive exceeds limit")
        for item in members:
            path = PurePosixPath(item.name)
            if not item.isfile() or item.issym() or item.islnk() or path.is_absolute() or ".." in path.parts:
                raise RuntimeError("unsafe runtime member")
        archive.extractall(destination, filter="data")

try:
    OUTPUT.unlink(missing_ok=True)
    RUN_ROOT = Path("/kaggle/working/failure_expert_label_audit_run")
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    CODE_ROOT = RUN_ROOT / "runtime"
    CODE_ROOT.mkdir(parents=True)
    runtime = base64.b64decode(RUNTIME_B64, validate=True)
    if hashlib.sha256(runtime).hexdigest() != RUNTIME_SHA256:
        raise RuntimeError("embedded runtime SHA-256 differs")
    safe_extract(runtime, CODE_ROOT)
    sys.path.insert(0, str(CODE_ROOT))
    print(f"FAIL_AUDIT_CODE_READY sha256={{RUNTIME_SHA256}} size_bytes={{len(runtime)}}", flush=True)

    STAGE = "inputs"
    import pandas as pd
    from experiments.tree_expert.failure_audit import run_failure_label_audit
    from experiments.tree_expert.failure_audit_artifacts import FailureAuditBindings, create_failure_audit_review, verify_failure_audit_review
    from experiments.tree_expert.failure_audit_contracts import contract_sha256, load_failure_audit_contract
    from experiments.tree_expert.failure_audit_kaggle import discover_official_data, file_sha256
    contract = load_failure_audit_contract()
    official = discover_official_data(Path("/kaggle/input"))
    train_sha = file_sha256(official / "train.csv")
    history_sha = file_sha256(official / "trackman_history.csv")
    print(f"FAIL_AUDIT_DATA_VERIFIED root={{official}} train_sha256={{train_sha}} history_sha256={{history_sha}}", flush=True)

    STAGE = "audit"
    train = pd.read_csv(official / "train.csv", low_memory=False)
    peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if peak_bytes > contract.maximum_rss_bytes:
        raise RuntimeError(f"memory limit exceeded before audit: {{peak_bytes}}")
    def audit_log(message):
        if not (message.startswith("FAIL_AUDIT_CUTOFF_START") or message.startswith("FAIL_AUDIT_CUTOFF_RESULT")):
            raise RuntimeError("audit log marker differs")
        print(message, flush=True)
    result = run_failure_label_audit(
        train,
        contract=contract,
        wall_deadline=time.monotonic() + contract.wall_seconds,
        log=audit_log,
    )
    peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if peak_bytes > contract.maximum_rss_bytes:
        raise RuntimeError(f"memory limit exceeded after audit: {{peak_bytes}}")

    STAGE = "artifact"
    bindings = FailureAuditBindings(
        contract_sha256=contract_sha256(),
        code_sha256=RUNTIME_SHA256,
        official_train_sha256=train_sha,
        official_history_sha256=history_sha,
    )
    review = create_failure_audit_review(result, RUN_ROOT / "failure_expert_label_audit_review.zip", bindings)
    verify_failure_audit_review(review, bindings)
    shutil.copy2(review, OUTPUT)
    verified = verify_failure_audit_review(OUTPUT, bindings)
    for failure_type, decision in result.type_decisions.items():
        print(f"FAIL_AUDIT_DECISION type={{failure_type}} status={{decision.status}} reason={{decision.reason}}", flush=True)
    print(f"FAIL_AUDIT_SUCCESS review={{OUTPUT}} sha256={{verified.archive_sha256}}", flush=True)
except Exception as error:
    OUTPUT.unlink(missing_ok=True)
    print(f"FAIL_AUDIT_ERROR stage={{STAGE}} type={{type(error).__name__}} message={{str(error).replace(' ', '_')}}", flush=True)
    traceback.print_exc()
    raise
'''
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8")
    return destination
