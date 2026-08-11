from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def load_json(path: str) -> dict[str, object]:
    with (ROOT / path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    assert isinstance(value, dict)
    return value


def sha256(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_required_root_files_exist() -> None:
    for relative in (
        "README.md",
        "AGENTS.md",
        "docs/EXPERIMENT_CONTRACT.md",
        "docs/ROADMAP.md",
        "reports/EXPERIMENT_LEDGER.md",
    ):
        assert (ROOT / relative).is_file(), relative


def test_gitignore_blocks_large_and_secret_inputs() -> None:
    text = read_text(".gitignore")
    for pattern in (
        "data/",
        "artifacts/",
        "*.npy",
        "*.zip",
        ".env",
        "*.pem",
        ".venv/",
        "__pycache__/",
    ):
        assert pattern in text


def test_agents_enforces_execution_ownership_and_mutation_boundaries() -> None:
    agents = read_text("AGENTS.md")
    required = (
        "Codex는 코드",
        "전체 데이터",
        "사용자가 수행",
        "비용만으로 후보를 제외하지 않는다",
        "사용자 요청 없이 push하지 않는다",
        "노트북",
        "패키징",
        "제출",
    )
    for phrase in required:
        assert phrase in agents


def test_experiment_contract_preserves_temporal_and_package_gates() -> None:
    contract = read_text("docs/EXPERIMENT_CONTRACT.md")
    required = (
        "planned → code_ready → waiting_for_user_run → passed → package_ready",
        "rejected",
        "failed",
        "시간 전이",
        "행 순서",
        "SHA-256",
        "acceptance",
        "제출 패키지를 만들지 않는다",
    )
    for phrase in required:
        assert phrase in contract
