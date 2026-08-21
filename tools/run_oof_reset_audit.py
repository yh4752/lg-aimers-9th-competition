from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.oof_reset_audit.artifacts import AuditArtifactError, load_artifacts
from experiments.oof_reset_audit.run import run_audit
from experiments.oof_reset_audit.types import ArtifactRole


EXPECTED_ROLES = (
    ArtifactRole.STAGE_C_TABM,
    ArtifactRole.ROW_FEATURE,
    ArtifactRole.CATBOOST_BLEND,
    ArtifactRole.CATBOOST_DEPLOYMENT,
    ArtifactRole.HIERARCHICAL,
    ArtifactRole.QUARANTINED_XGBOOST,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit existing temporal OOF evidence without training models.")
    parser.add_argument("--artifact", action="append", type=Path, required=True,
                        help="Existing review/delivery OOF ZIP; repeat for each artifact.")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Directory under which a new immutable audit run is created.")
    return parser


def _file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    stage = "inputs"
    try:
        paths = [path.resolve(strict=True) for path in args.artifact]
        if any(path.is_symlink() or not path.is_file() for path in paths):
            raise AuditArtifactError("every artifact must be a regular non-symlink file")
        loaded = load_artifacts(paths, expected_roles=EXPECTED_ROLES)
        print(
            f"OOF_RESET_AUDIT_INPUTS_VERIFIED artifacts={len(paths)} "
            f"predictions={len(loaded.predictions)}",
            flush=True,
        )
        stage = "reports"
        result = run_audit(
            predictions=loaded.predictions,
            inventory=loaded.inventory,
            output_root=args.output_root.resolve(),
        )
        print(
            f"OOF_RESET_AUDIT_SUCCESS output_dir={result.output_dir.resolve()} "
            f"result_zip={result.result_zip.resolve()} sha256={_file_sha(result.result_zip)}",
            flush=True,
        )
        return 0
    except Exception as error:
        message = "_".join(str(error).split()) or "unknown"
        print(
            f"OOF_RESET_AUDIT_ERROR stage={stage} type={type(error).__name__} message={message}",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
