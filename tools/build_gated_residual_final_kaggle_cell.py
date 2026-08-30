from __future__ import annotations

import argparse
import base64
import gzip
import io
from pathlib import Path
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gated_residual_final.runtime_inventory import (  # noqa: E402
    code_identity_sha256,
    runtime_members,
)


class CellBuildError(ValueError):
    pass


def _runtime_archive(root: Path) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name in runtime_members(root):
                payload = (root / name).read_bytes()
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mtime = 0
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def build_cell(output: Path, *, root: Path | None = None) -> Path:
    repository = Path(root) if root is not None else ROOT
    payload = _runtime_archive(repository)
    encoded = base64.b64encode(payload).decode("ascii")
    identity = code_identity_sha256(repository)
    source = f'''from __future__ import annotations
import base64, importlib.metadata, io, os, shutil, subprocess, sys, tarfile, time, traceback
from hashlib import sha256
from pathlib import Path

STAGE = "bootstrap"
RUNTIME_B64 = "{encoded}"
PAYLOAD = base64.b64decode(RUNTIME_B64)
EXPECTED_CODE_SHA256 = "{identity}"
print(f"FINAL_CANDIDATE_CODE_READY archive_sha256={{sha256(PAYLOAD).hexdigest()}} code_sha256={{EXPECTED_CODE_SHA256}} size_bytes={{len(PAYLOAD)}}", flush=True)
try:
    STAGE = "runtime"
    runtime_root = Path("/kaggle/working/gated_residual_final_runtime")
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(PAYLOAD), mode="r:gz") as archive:
        for member in archive.getmembers():
            target = (runtime_root / member.name).resolve()
            if not member.isfile() or not target.is_relative_to(runtime_root.resolve()):
                raise RuntimeError(f"unsafe embedded runtime member: {{member.name}}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source_file = archive.extractfile(member)
            if source_file is None:
                raise RuntimeError(f"embedded runtime member is unreadable: {{member.name}}")
            with source_file, target.open("wb") as sink:
                shutil.copyfileobj(source_file, sink)
    sys.path.insert(0, str(runtime_root))

    STAGE = "inputs"
    from experiments.gated_residual_final.kaggle import discover_inputs
    found = discover_inputs(Path("/kaggle/input"))
    print(f"FINAL_CANDIDATE_INPUTS_FOUND official={{found.official_data}} input={{found.final_input}} resume={{found.previous_handoff}}", flush=True)

    STAGE = "dependencies"
    try:
        catboost_version = importlib.metadata.version("catboost")
    except importlib.metadata.PackageNotFoundError:
        catboost_version = None
    if catboost_version != "1.2.10":
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "catboost==1.2.10"], check=True)
    print(f"FINAL_CANDIDATE_DEPENDENCIES_READY catboost={{importlib.metadata.version('catboost')}}", flush=True)

    STAGE = "campaign"
    from experiments.gated_residual_final.kaggle import run_kaggle_campaign
    run_root = Path("/kaggle/working/gated_residual_final_runs") / f"{{time.time_ns()}}_{{os.getpid()}}"
    result = run_kaggle_campaign(Path("/kaggle/input"), run_root)
    for artifact in (result.review, result.handoff, result.delivery):
        if artifact is None:
            continue
        destination = Path("/kaggle/working") / artifact.name
        shutil.copy2(artifact, destination)
        print(f"FINAL_CANDIDATE_OUTPUT_READY path={{destination}}", flush=True)
    # terminal markers: FINAL_CANDIDATE_REVIEW_READY FINAL_CANDIDATE_HANDOFF_READY FINAL_CANDIDATE_DELIVERY_READY
except Exception as error:
    print(f"FINAL_CANDIDATE_ERROR stage={{STAGE}} type={{type(error).__name__}} message={{str(error).replace(' ', '_')}}", flush=True)
    traceback.print_exc()
    raise
'''.encode("utf-8")
    if len(source) >= 1_000_000:
        raise CellBuildError("Kaggle cell exceeds one megabyte")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the final gated-residual Kaggle cell.")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/gated_residual_final/KAGGLE_CELL.py")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        expected = args.output.read_bytes()
        temporary = args.output.with_suffix(".check.tmp")
        try:
            build_cell(temporary)
            if temporary.read_bytes() != expected:
                raise CellBuildError("checked-in Kaggle cell differs")
        finally:
            temporary.unlink(missing_ok=True)
        print("GATED_RESIDUAL_CELL_CHECK_SUCCESS", flush=True)
        return 0
    output = build_cell(args.output)
    print(
        f"GATED_RESIDUAL_CELL_READY path={output.resolve()} size_bytes={output.stat().st_size}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
