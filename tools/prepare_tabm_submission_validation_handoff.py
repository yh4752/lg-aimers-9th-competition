#!/usr/bin/env python3
"""Prepare the upload directory for TabM evaluator validation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission.tabm_candidate import import_review_delivery  # noqa: E402


REQUIREMENTS = b"tabm==0.0.3\nrtdl-num-embeddings==0.0.12\n"
RUNTIME_FILES = (
    "competition_rules/__init__.py",
    "competition_rules/code_gate.py",
    "competition_rules/contract.py",
    "competition_rules/evidence_gate.py",
    "competition_rules/policy.json",
    "submission/__init__.py",
    "submission/tabm_candidate.py",
    "submission/tabm_validation.py",
    "submission/tabm_version_d_script.py",
)


@dataclass(frozen=True)
class HandoffResult:
    root: Path
    manifest_sha256: str
    model_sha256: str


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def prepare_handoff(delivery: str | Path, output_dir: str | Path) -> HandoffResult:
    """Verify the candidate and publish a validation-only directory."""

    source = Path(delivery).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().absolute()
    if destination.exists():
        raise FileExistsError(f"handoff output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        candidate = import_review_delivery(source, temporary / "candidate")
        runtime_root = temporary / "runtime"
        for relative in RUNTIME_FILES:
            incoming = ROOT / relative
            target = runtime_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(incoming, target)
        (temporary / "requirements.txt").write_bytes(REQUIREMENTS)
        members = {
            path.relative_to(temporary).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "artifact_kind": "tabm_submission_validation_handoff_v1",
            "candidate_id": candidate.candidate_id,
            "delivery_sha256": candidate.delivery_sha256,
            "model_sha256": candidate.model_sha256,
            "requirements_sha256": sha256(REQUIREMENTS).hexdigest(),
            "members": members,
            "submission_package": False,
        }
        manifest_value = _canonical_json(manifest)
        (temporary / "handoff_manifest.json").write_bytes(manifest_value)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return HandoffResult(
        root=destination,
        manifest_sha256=sha256(manifest_value).hexdigest(),
        model_sha256=candidate.model_sha256,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a Kaggle upload directory for TabM validation"
    )
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare_handoff(args.delivery, args.output_dir)
    print(
        f"TABM_SUBMISSION_VALIDATION_HANDOFF_READY root={result.root} "
        f"manifest_sha256={result.manifest_sha256} "
        f"model_sha256={result.model_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
