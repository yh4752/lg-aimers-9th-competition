from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import secrets
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.experiment_registry import (
    RegistryError,
    audit_registry,
    file_sha256,
    load_registry,
    render_audit_markdown,
)


_HASH = re.compile(r"^[0-9a-f]{64}$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic audit from the curated experiment registry."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("reports/experiment_registry.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="ID=SHA256:PATH",
        help="Explicitly verify a local evidence artifact; may be repeated.",
    )
    return parser


def _artifact_spec(value: str) -> tuple[str, str, Path]:
    try:
        experiment_id, remainder = value.split("=", 1)
        expected, raw_path = remainder.split(":", 1)
    except ValueError as error:
        raise RegistryError("artifact_spec_must_be_ID=SHA256:PATH") from error
    if not experiment_id or _HASH.fullmatch(expected) is None or not raw_path:
        raise RegistryError("artifact_spec_must_be_ID=SHA256:PATH")
    return experiment_id, expected, Path(raw_path)


def _verify_artifacts(payload: dict[str, object], specs: list[str]) -> None:
    experiments = payload["experiments"]
    assert isinstance(experiments, list)
    indexed = {str(item["experiment_id"]): item for item in experiments}
    seen: set[str] = set()
    for raw_spec in specs:
        experiment_id, expected, path = _artifact_spec(raw_spec)
        if experiment_id not in indexed:
            raise RegistryError(f"unknown_experiment_id:{experiment_id}")
        if experiment_id in seen:
            raise RegistryError(f"duplicate_artifact_experiment_id:{experiment_id}")
        seen.add(experiment_id)
        if path.is_symlink():
            raise RegistryError(f"artifact_must_not_be_symlink:{path}")
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise RegistryError(f"artifact_is_absent:{path}") from error
        if not resolved.is_file():
            raise RegistryError(f"artifact_must_be_regular_file:{path}")
        observed = file_sha256(resolved)
        if observed != expected:
            raise RegistryError(
                f"artifact_sha256_differs:{experiment_id}:expected={expected}:observed={observed}"
            )
        registered = indexed[experiment_id]["artifact_sha256"]
        assert isinstance(registered, list)
        if expected not in registered:
            raise RegistryError(f"artifact_sha256_not_registered:{experiment_id}")


def _atomic_write(path: Path, payload: bytes) -> None:
    path = Path(path)
    if path.is_symlink():
        raise RegistryError("output_must_not_be_symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise RegistryError("output_parent_must_be_regular_directory")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    stage = "registry"
    try:
        registry = load_registry(args.registry)
        stage = "artifacts"
        _verify_artifacts(registry, args.artifact)
        stage = "render"
        report = render_audit_markdown(audit_registry(registry)).encode("utf-8")
        _atomic_write(args.output, report)
        output = args.output.resolve(strict=True)
        print(
            f"EXPERIMENT_AUDIT_SUCCESS experiments={len(registry['experiments'])} "
            f"evidence_gaps={len(registry['evidence_gaps'])} output={output} "
            f"sha256={file_sha256(output)}",
            flush=True,
        )
        return 0
    except Exception as error:
        message = "_".join(str(error).split()) or "unknown"
        print(
            f"EXPERIMENT_AUDIT_ERROR stage={stage} type={type(error).__name__} "
            f"message={message}",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
