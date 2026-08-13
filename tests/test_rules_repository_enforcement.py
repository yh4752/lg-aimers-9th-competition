from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_official_submission_archive_has_one_writer() -> None:
    offenders = []
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith((".worktrees/", ".venv/", "tests/")):
            continue
        text = path.read_text(encoding="utf-8")
        if "script.py" in text and "requirements.txt" in text and "model/" in text:
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            if any(
                isinstance(node, ast.Attribute) and node.attr in {"writestr", "write"}
                for node in ast.walk(tree)
            ):
                offenders.append(relative)
    assert offenders == ["submission/package.py"]
