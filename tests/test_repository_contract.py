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


IMMUTABLE_REPORTS = {
    "reports/acceptances/core_acceptance.json": (
        "0dc7bce3a730523a3ce7ffd5f9f6c1c90d1694563b057c827d6aadf4d578c471",
        "passed",
    ),
    "reports/acceptances/round9_temporal_oof_acceptance.json": (
        "e56f87dd8800e8a4d899e3d8cdb5830c1e961d74e1298a60505570e5d5e3621a",
        "passed",
    ),
    "reports/acceptances/anchor_brier_audit_acceptance.json": (
        "9ccffd3fe983a0a125efed0822ba1722a2887a1505bb424ea02ed3e6a1283e0f",
        "passed",
    ),
    "reports/rejections/fwfm_standalone_rejection.json": (
        "e9066396ca6db12dd69f823bc6610cb457653d5859317073aed88c3f43d52606",
        "rejected",
    ),
    "reports/rejections/r9_fwfm_game_type_f_blend_rejection.json": (
        "0ae787f74a798077e136a43902d95b96f00b3dc67008eb00d138699ed3b68e66",
        "rejected",
    ),
    "reports/rejections/r9_fwfm_game_type_f_blend_w080_rejection.json": (
        "43499ceb6b447e6f6acd5a1d5cf9c8bfe4aba46cd233be097e295c011b302332",
        "rejected",
    ),
    "reports/rejections/tabm_residual_rejection.json": (
        "30839509418e0ae46d7e0acf426b98d25280ab1f0a246a2d77a8c7042eed427d",
        "rejected",
    ),
    "reports/diagnostics/fwfm_failure_boundary_exit_audit.json": (
        "e0d8fdbefcba40ad77f1272aef819e83b992ee6bd4aea1b44ec8cfec3c32ca34",
        "read_only_diagnostic",
    ),
}


def test_immutable_reports_match_original_bytes_and_status() -> None:
    for path, (expected_hash, expected_status) in IMMUTABLE_REPORTS.items():
        assert sha256(path) == expected_hash, path
        assert load_json(path)["status"] == expected_status, path
