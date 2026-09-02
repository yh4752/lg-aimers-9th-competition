from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from experiments.direct_expert.runtime_inventory import runtime_members as direct_runtime_members


def runtime_members(root: Path) -> tuple[str, ...]:
    source = Path(root)
    e3 = tuple(
        path.relative_to(source).as_posix()
        for path in sorted((source / "experiments/failure_regime_e3").iterdir())
        if path.is_file() and path.suffix in {".py", ".json"} and path.name != "KAGGLE_CELL.py"
    )
    rules = tuple(
        path.relative_to(source).as_posix()
        for path in sorted((source / "competition_rules").iterdir())
        if path.is_file() and path.suffix in {".py", ".json"}
    )
    members = tuple(sorted({*direct_runtime_members(source), *rules, *e3}))
    for name in members:
        if not (source / name).is_file():
            raise FileNotFoundError(f"runtime member is absent: {name}")
    return members


def code_identity_sha256(root: Path) -> str:
    digest = sha256()
    for name in runtime_members(root):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((Path(root) / name).read_bytes())
    return digest.hexdigest()
