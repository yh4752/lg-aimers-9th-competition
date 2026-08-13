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


def test_rules_contract_names_all_four_transition_gates() -> None:
    text = (ROOT / "docs/EXPERIMENT_CONTRACT.md").read_text(encoding="utf-8")
    positions = [
        text.index(phrase)
        for phrase in (
            "실험 시작 gate",
            "사용자 실행 gate",
            "후보 수용 gate",
            "패키징 gate",
        )
    ]
    assert positions == sorted(positions)
    assert "자동 업로드" in text
    assert "평가 행의 수나 분포" in text
