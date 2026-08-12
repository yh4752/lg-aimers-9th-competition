from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def load_json(path: str) -> dict[str, object]:
    with (ROOT / path).open(encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_nonfinite,
        )
    assert isinstance(value, dict)
    return value


def sha256(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "payload",
    ('{"status":"passed","status":"rejected"}', '{"metric":NaN}'),
)
def test_strict_json_helpers_reject_ambiguous_payloads(payload: str) -> None:
    with pytest.raises(ValueError):
        json.loads(
            payload,
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_nonfinite,
        )


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


def test_calibration_rejection_records_open_family() -> None:
    report = load_json("reports/rejections/calibration_blending_rejection.json")
    assert report["status"] == "rejected"
    assert report["candidate_id"] == "calibration_blending_exact_variants"
    assert report["family_closed"] is False
    assert report["best_candidate"] == "game_type_temperature"
    assert report["best_global_brier"] == 0.24789890676984772
    assert report["failed_gates"] == [
        "global_calibration_gap_not_higher",
        "max_fold_delta",
    ]


def test_xgboost_v3_exploratory_acceptance_is_not_public_score() -> None:
    report = load_json("reports/acceptances/xgboost_v3_exploratory_acceptance.json")
    assert report["status"] == "accepted_for_exploratory_submission"
    assert report["run_id"] == "f224a534242a41fea3b88218086e795c"
    assert report["mean_temporal_brier"] == 0.24701737756648098
    assert report["final_2024_brier"] == 0.248274358430683
    assert report["public_score"] is None


def test_xgboost_rescue_is_technically_verified_not_submitted() -> None:
    report = load_json(
        "reports/acceptances/xgboost_original_preproc_rescue_acceptance.json"
    )
    assert report["status"] == "verified_ready"
    assert report["run_id"] == "7c3820cb6b904193b9cf337ad602dff1"
    assert report["selected_candidate"] == "lossguide_l31"
    assert report["final_2024_brier"] == 0.24826687414041645
    assert report["archive_sha256"] == (
        "075e38e355462953543da4532b658568898a6458c6d30e604d5fcbb6b4772006"
    )
    assert report["public_score"] is None


def test_xgboost_aggressive_capacity_public_result_is_bound() -> None:
    report = load_json(
        "reports/acceptances/xgboost_aggressive_capacity_public_result.json"
    )
    assert report["status"] == "public_scored"
    assert report["candidate_id"] == "xgboost_aggressive_capacity_v1"
    assert report["run_id"] == "a0b99dd0e7eb41fba2b5ff729b11aeb1"
    assert report["public_score"] == 820.9583317093
    assert report["local_brier"] == 0.2479270213638507
    assert report["local_score"] == 752.5433411090132
    assert report["archive_sha256"] == (
        "f81b5df770733898535d9a1d4a019b7c339aa72d7cbce0cb67a3e49ccb41f43a"
    )
    assert report["scale"] == 1.05
    assert report["mean_shift"] == "linear_extrapolated"
    assert report["members"] == [
        {"structure": "depthwise_d6", "seed": 42, "rounds": 119},
        {"structure": "depthwise_d6", "seed": 2026, "rounds": 134},
        {"structure": "lossguide_l63", "seed": 42, "rounds": 119},
        {"structure": "lossguide_l63", "seed": 2026, "rounds": 106},
    ]


def test_agents_default_to_broad_performance_exploration() -> None:
    agents = read_text("AGENTS.md")
    for phrase in (
        "모델 깊이",
        "GPU 사용량을 사전에 제한하지 않는다",
        "비용과 실행 시간은 안내와 실행 순서에만 사용한다",
        "탐색을 막지 않고",
        "새 폴더·문서·자동화",
    ):
        assert phrase in agents


def test_roadmap_keeps_dl_primary_without_discarding_ml() -> None:
    agents = read_text("AGENTS.md")
    roadmap = read_text("docs/ROADMAP.md")

    for phrase in (
        "다음 주력 연구는 독립 DL",
        "ML 단독 미세 조정",
        "독립 DL 준비나 실행을 지연시킬 수 없다",
        "TabM residual",
        "DL 계열 전체",
        "CatBoost와 XGBoost",
    ):
        assert phrase in agents

    stages = ("독립 DL 탐색", "DL 내부 앙상블", "ML+DL 앙상블")
    positions = [roadmap.index(stage) for stage in stages]
    assert positions == sorted(positions)
    assert "보조 트랙" in roadmap
    assert "유망 후보" in roadmap
    assert "정렬된 OOF" in roadmap
    for retained_context in (
        "R25 TabM 잔차",
        "R32 예측 분모 보정",
        "검증 프로토콜이 다르므로",
        "기존 세 변형의 평균 개선과 실패 gate",
    ):
        assert retained_context in roadmap


def test_every_experiment_requires_performance_first_precheck() -> None:
    agents = read_text("AGENTS.md")
    contract = read_text("docs/EXPERIMENT_CONTRACT.md")

    for phrase in (
        "[성능 우선 확인]",
        "smoke 결과를 성능 근거로 사용하지 않았는가",
        "첫 본 실험에 큰 모델·긴 학습·넓은 탐색",
        "최고 설정이 탐색 경계에 있으면",
        "OOM·시간·비용 또는 단일 설정 실패",
        "실험 설계는 미완성",
    ):
        assert phrase in agents

    for phrase in (
        "기술 확인 전용",
        "성능을 판단할 수 있는 충분한",
        "다음 범위를 확장",
        "mixed precision",
        "gradient accumulation",
        "모델 계열 기각 근거가 아니다",
        "해당 설정만 종료",
    ):
        assert phrase in contract


def test_xgboost_public_result_is_visible_in_existing_documents() -> None:
    readme = read_text("README.md")
    ledger = read_text("reports/EXPERIMENT_LEDGER.md")
    round_doc = read_text("docs/rounds/05-xgboost.md")
    roadmap = read_text("docs/ROADMAP.md")

    assert "820.9583317093" in readme
    assert "xgboost_aggressive_capacity_v1" in ledger
    assert "820.9583317093" in ledger
    for phrase in (
        "depthwise_d6",
        "lossguide_l63",
        "depthwise_d8",
        "lossguide_l255",
        "714.8814792915847",
        "750.5344761643662",
        "752.5433411090132",
    ):
        assert phrase in round_doc
    assert "정렬된 OOF" in roadmap
    assert "CatBoost" in roadmap


def test_ledger_records_every_completed_family_and_protocol() -> None:
    ledger = read_text("reports/EXPERIMENT_LEDGER.md")
    for experiment in (
        "catboost_smooth_v1",
        "round9_temporal_oof",
        "fwfm_standalone",
        "tabm_residual",
        "calibration_blending",
        "xgboost_score_push_v3",
        "xgboost_original_preproc_rescue",
    ):
        assert experiment in ledger
    assert "검증 프로토콜" in ledger
    assert "Public" in ledger


def test_round_index_links_to_all_round_documents() -> None:
    index = read_text("docs/rounds/README.md")
    for name in (
        "01-r9-foundation.md",
        "02-fwfm.md",
        "03-tabm-residual.md",
        "04-calibration.md",
        "05-xgboost.md",
    ):
        assert f"]({name})" in index
        assert (ROOT / "docs/rounds" / name).is_file()


def test_readme_is_a_competition_dashboard() -> None:
    readme = read_text("README.md")
    for phrase in (
        "LG Aimers 9th",
        "현재 기준선",
        "실험 장부",
        "검증 프로토콜",
        "다음 후보",
        "Google Drive",
    ):
        assert phrase in readme
    assert "31개 LG Aimers VOD" not in readme


def test_roadmap_keeps_cost_and_independent_candidates_open() -> None:
    roadmap = read_text("docs/ROADMAP.md")
    assert "비용만으로" in roadmap
    assert "독립 후보" in roadmap
    assert "동료 저장소" in roadmap
    assert "calibration" in roadmap.lower()
    assert "XGBoost" in roadmap


def test_tracked_text_contains_no_secret_or_private_mount_path() -> None:
    secret_markers = ("GITHUB_TOKEN=", "AIMERS_REPO_URL=")
    all_operational_paths = [ROOT / "README.md", ROOT / "AGENTS.md"]
    all_operational_paths.extend((ROOT / "docs/rounds").glob("*.md"))
    all_operational_paths.extend((ROOT / "reports").rglob("*.json"))
    all_operational_paths.extend(
        (ROOT / name)
        for name in ("docs/EXPERIMENT_CONTRACT.md", "docs/ROADMAP.md")
    )
    for path in all_operational_paths:
        text = path.read_text(encoding="utf-8")
        for value in secret_markers:
            assert value not in text, f"{value!r} in {path.relative_to(ROOT)}"

    human_docs = [ROOT / "README.md", ROOT / "AGENTS.md"]
    human_docs.extend((ROOT / "docs/rounds").glob("*.md"))
    human_docs.extend(
        (ROOT / name)
        for name in ("docs/EXPERIMENT_CONTRACT.md", "docs/ROADMAP.md")
    )
    for path in human_docs:
        assert "/content/drive/MyDrive/" not in path.read_text(encoding="utf-8")


def test_relative_markdown_links_resolve() -> None:
    pattern = re.compile(r"\[[^]]+\]\((?!https?://|#)([^)]+)\)")
    for path in ROOT.rglob("*.md"):
        if ".git" in path.parts:
            continue
        for target in pattern.findall(path.read_text(encoding="utf-8")):
            clean = target.split("#", 1)[0]
            if clean:
                assert (path.parent / clean).resolve().exists(), (path, target)
