from __future__ import annotations

from hashlib import sha256
from pathlib import Path


_SHARED = (
    "experiments/tree_expert/__init__.py",
    "experiments/tree_expert/contracts.py",
    "experiments/tree_expert/features.py",
    "experiments/temporal_portfolio/__init__.py",
    "experiments/temporal_portfolio/seasonal_features.py",
    "experiments/temporal_portfolio/trackman_pitcher.py",
    "experiments/temporal_portfolio/trackman_batter.py",
    "experiments/independent_dl/__init__.py",
    "experiments/independent_dl/feature_sources/__init__.py",
    "experiments/independent_dl/feature_sources/seasonal.py",
    "experiments/independent_dl/feature_sources/trackman.py",
)


def runtime_members(root: Path) -> tuple[str, ...]:
    source = Path(root)
    direct = tuple(
        path.relative_to(source).as_posix()
        for path in sorted((source / "experiments/direct_expert").iterdir())
        if path.is_file() and path.suffix in {".py", ".json"} and not path.name.startswith("KAGGLE_")
    )
    local = tuple(
        path.relative_to(source).as_posix()
        for path in sorted((source / "experiments/gated_residual_final").iterdir())
        if path.is_file() and path.suffix in {".py", ".json"} and not path.name.startswith("KAGGLE_")
    )
    members = tuple(sorted({*_SHARED, *direct, *local}))
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
