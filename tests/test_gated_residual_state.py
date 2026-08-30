from __future__ import annotations

from pathlib import Path

from experiments.gated_residual_final.state import CampaignState, load_state, save_state


def test_atomic_state_survives_reload(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    expected = CampaignState(
        phase="P3", status="running", completed_jobs=("D0_s42",),
        decision_status="accepted", candidate_id="G1_x",
    )

    save_state(path, expected)

    assert load_state(path) == expected
    assert not path.with_suffix(".json.tmp").exists()


def test_second_save_atomically_replaces_state(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    save_state(path, CampaignState("P1", "running", (), None, None))
    save_state(path, CampaignState("P2", "running", (), "rejected", "G0_x"))

    assert load_state(path).phase == "P2"
    assert load_state(path).decision_status == "rejected"
