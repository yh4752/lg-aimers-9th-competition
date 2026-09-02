from __future__ import annotations

import json
from pathlib import Path


def test_repository_registers_failed_campaign_without_packager() -> None:
    contract = json.loads(Path("experiments/failure_regime_e3/contract.json").read_text(encoding="utf-8"))
    roadmap = Path("docs/ROADMAP.md").read_text(encoding="utf-8")
    ledger = Path("reports/EXPERIMENT_LEDGER.md").read_text(encoding="utf-8")
    evidence = json.loads(
        Path("reports/evidence/failure_regime_e3_20260902.json").read_text(encoding="utf-8")
    )
    package = Path("experiments/failure_regime_e3")

    assert contract["policy_version"] == "dacon-236743-2026-08-15"
    assert contract["campaign_id"] == "failure_regime_e3_v1"
    assert "failure_regime_e3_v1" in roadmap and "failed" in roadmap
    assert "failure_regime_e3_v1" in ledger and "resource memory" in ledger
    assert evidence["status"] == "failed"
    assert evidence["campaign_state"]["oof_jobs_completed"] == 53
    assert evidence["submission_eligible"] is False
    assert (package / "KAGGLE_CELL.py").is_file()
    assert not any("submission" in path.name.lower() or "delivery" in path.name.lower() for path in package.iterdir())
    assert "create_submission" not in "\n".join(
        path.read_text(encoding="utf-8") for path in package.glob("*.py")
    )
