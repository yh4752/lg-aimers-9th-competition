from __future__ import annotations

import importlib.metadata
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DependencyProbe:
    status: str
    command: tuple[str, ...]
    return_code: int
    elapsed_seconds: float
    versions: dict[str, str]
    output_tail: str


def installed_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("tabm", "rtdl-num-embeddings", "torch", "numpy", "pandas"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "missing"
    return result


def run_clean_install_probe(
    requirements: str | Path,
    output_dir: str | Path,
    *,
    timeout_seconds: int = 480,
) -> DependencyProbe:
    """Probe the two pinned packages in a disposable, base-package-sharing venv."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    venv = root / "dependency_probe_venv"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], check=True)
    python = venv / "bin" / "python"
    command = (str(python), "-m", "pip", "install", "-r", str(Path(requirements).resolve()))
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds)
    elapsed = time.monotonic() - started
    probe = subprocess.run(
        [str(python), "-c", "import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ['tabm','rtdl-num-embeddings']}))"],
        capture_output=True,
        text=True,
    )
    versions = {}
    if probe.returncode == 0:
        import json

        versions = json.loads(probe.stdout)
    output = (completed.stdout + "\n" + completed.stderr)[-12000:]
    return DependencyProbe(
        "passed" if completed.returncode == 0 and elapsed <= timeout_seconds and len(versions) == 2 else "failed",
        command,
        completed.returncode,
        elapsed,
        versions,
        output,
    )
