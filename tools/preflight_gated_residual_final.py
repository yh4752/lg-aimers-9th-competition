from __future__ import annotations

import argparse
import base64
import gzip
import io
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from typing import Mapping
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gated_residual_final.contracts import load_contract  # noqa: E402
from experiments.gated_residual_final.inputs import verify_and_extract_final_input  # noqa: E402


class PreflightError(ValueError):
    pass


def _extract_runtime(cell_path: Path, destination: Path) -> Path:
    source = Path(cell_path).read_text(encoding="utf-8")
    match = re.search(r'^RUNTIME_B64 = "([A-Za-z0-9+/=]+)"$', source, re.MULTILINE)
    if match is None:
        raise PreflightError("embedded runtime payload is absent")
    payload = base64.b64decode(match.group(1), validate=True)
    destination.mkdir(parents=True)
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r:") as archive:
            for member in archive.getmembers():
                target = (destination / member.name).resolve()
                if not member.isfile() or not target.is_relative_to(destination.resolve()):
                    raise PreflightError(f"unsafe runtime member: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                source_file = archive.extractfile(member)
                if source_file is None:
                    raise PreflightError(f"runtime member is unreadable: {member.name}")
                target.write_bytes(source_file.read())
    return destination


def _expected_sources() -> Mapping[str, str]:
    contract = load_contract()
    return {
        "e2_input": contract.expected_hashes["direct_expert_input"],
        "stage_a": contract.expected_hashes["stage_a_handoff"],
        "stage_b": contract.expected_hashes["stage_b_handoff"],
    }


def run_preflight(
    *,
    input_path: Path,
    cell_path: Path,
    work_root: Path,
    enforce_contract_hashes: bool = True,
) -> dict[str, object]:
    source = Path(input_path)
    if not source.is_file() or source.suffix.lower() != ".zip":
        raise PreflightError("preflight input must be a ZIP")
    root = Path(work_root)
    if root.exists() or root.is_symlink():
        raise PreflightError("preflight output already exists")
    root.mkdir(parents=True)
    zipped = verify_and_extract_final_input(source, root / "verified_zip")
    expanded = root / "expanded_input"
    with ZipFile(source) as archive:
        archive.extractall(expanded)
    directory = verify_and_extract_final_input(expanded, root / "verified_expanded")
    if zipped.manifest_sha256 != directory.manifest_sha256 or dict(zipped.source_hashes) != dict(directory.source_hashes):
        raise PreflightError("ZIP and expanded input identities differ")
    if enforce_contract_hashes and dict(zipped.source_hashes) != dict(_expected_sources()):
        raise PreflightError("source artifact bindings differ")
    runtime = _extract_runtime(Path(cell_path), root / "runtime")
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import experiments.gated_residual_final.kaggle; "
        "import experiments.gated_residual_final.runtime; print('IMPORT_OK')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(runtime)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or result.stdout.strip() != "IMPORT_OK":
        raise PreflightError(f"isolated runtime import failed: {result.stderr.strip()}")
    submission_created = any(path.name == "submission.zip" for path in root.rglob("*"))
    if submission_created:
        raise PreflightError("submission package was created")
    return {"layouts": ("zip", "expanded"), "imports": "isolated", "submission_created": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight the final Kaggle experiment without training.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_preflight(input_path=args.input, cell_path=args.cell, work_root=args.work_root)
    print(
        "GATED_RESIDUAL_PREFLIGHT_SUCCESS "
        f"layouts={','.join(report['layouts'])} imports={report['imports']} "
        f"submission_created={str(report['submission_created']).lower()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
