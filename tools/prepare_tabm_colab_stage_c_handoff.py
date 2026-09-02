"""Build the verified local upload directory for Colab Stage C recovery."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tabm_campaign.artifacts import verify_resume_bundle
from experiments.tabm_campaign.colab_recovery import (
    ColabRecoveryError,
    canonical_json,
    file_sha256,
    load_colab_contract,
    verify_and_extract_data_archive,
)


DEFAULT_CELL = ROOT / "experiments/tabm_campaign/COLAB_STAGE_C_RECOVERY_CELL.py"
_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class HandoffPaths:
    root: Path
    data_zip: Path
    resume_zip: Path
    cell: Path
    manifest: Path


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_copy(source: Path, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-", dir=destination.parent
    )
    try:
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output:
            shutil.copyfileobj(input_stream, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_deterministic_data_zip(
    data_dir: Path,
    destination: Path,
    members: Mapping[str, object],
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-", dir=destination.parent
    )
    os.close(descriptor)
    try:
        with ZipFile(
            temporary_name, "w", compression=ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in sorted(members):
                info = ZipInfo(name, date_time=_ZIP_TIMESTAMP)
                info.compress_type = ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                with (data_dir / name).open("rb") as source, archive.open(
                    info, "w", force_zip64=True
                ) as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def prepare_handoff(
    data_dir: Path,
    base_resume: Path,
    output_dir: Path,
    *,
    contract: Mapping[str, object] | None = None,
    cell_path: Path = DEFAULT_CELL,
) -> HandoffPaths:
    contract = load_colab_contract() if contract is None else contract
    archive_contract = contract.get("data_archive")
    if not isinstance(archive_contract, Mapping):
        raise ColabRecoveryError("data archive contract is missing")
    declared = archive_contract.get("members")
    if not isinstance(declared, Mapping):
        raise ColabRecoveryError("data member contract is missing")
    actual_names = {path.name for path in data_dir.iterdir()}
    if actual_names != set(declared):
        raise ColabRecoveryError("official data member names differ")
    for name, evidence in declared.items():
        if not isinstance(evidence, Mapping):
            raise ColabRecoveryError(f"official data contract is invalid: {name}")
        source = data_dir / str(name)
        if (
            source.is_symlink()
            or not source.is_file()
            or source.stat().st_size != int(evidence["size"])
            or file_sha256(source) != str(evidence["sha256"])
        ):
            raise ColabRecoveryError(f"official data file differs: {name}")
    base_contract = contract.get("base_resume")
    if not isinstance(base_contract, Mapping):
        raise ColabRecoveryError("base resume contract is missing")
    if file_sha256(base_resume) != str(base_contract.get("sha256")):
        raise ColabRecoveryError("base Stage C resume SHA-256 differs")
    verified_resume = verify_resume_bundle(base_resume)
    if verified_resume.version != "C":
        raise ColabRecoveryError("base resume is not Stage C")
    if not cell_path.is_file():
        raise ColabRecoveryError(f"generated Colab cell is missing: {cell_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    data_zip = output_dir / str(archive_contract["filename"])
    resume_zip = output_dir / str(base_contract["filename"])
    cell = output_dir / "COLAB_STAGE_C_RECOVERY_CELL.py"
    manifest = output_dir / "handoff_manifest.json"
    _write_deterministic_data_zip(data_dir, data_zip, declared)
    with tempfile.TemporaryDirectory(prefix="tabm-handoff-verify-") as temporary:
        verify_and_extract_data_archive(
            data_zip, Path(temporary) / "official_data", contract
        )
    _atomic_copy(base_resume, resume_zip)
    _atomic_copy(cell_path, cell)
    payloads = (data_zip, resume_zip, cell)
    manifest_value = {
        "schema_version": 1,
        "files": {
            path.name: {
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in payloads
        },
    }
    _atomic_write(manifest, canonical_json(manifest_value))
    return HandoffPaths(output_dir, data_zip, resume_zip, cell, manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-resume", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = prepare_handoff(args.data_dir, args.base_resume, args.output_dir)
    print(
        f"COLAB_HANDOFF_READY root={paths.root.resolve()} "
        f"data_sha256={file_sha256(paths.data_zip)} "
        f"resume_sha256={file_sha256(paths.resume_zip)} "
        f"cell_sha256={file_sha256(paths.cell)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
