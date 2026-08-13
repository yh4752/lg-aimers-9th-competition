"""Immutable input and output contracts for the sole packager."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class PackageRequest:
    project_root: Path
    policy_path: Path
    policy_review_path: Path
    acceptance_path: Path
    full_audit_manifest_path: Path
    runtime_benchmark_path: Path
    model_dir: Path
    requirements_path: Path
    adapter_id: str
    archive_path: Path
    receipt_path: Path
    package_time: datetime


@dataclass(frozen=True)
class PackageResult:
    archive_path: Path
    receipt_path: Path
    archive_sha256: str
    archive_bytes: int

