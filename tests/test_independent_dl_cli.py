from __future__ import annotations

import json
from pathlib import Path

from experiments.independent_dl import run_campaign as module
from experiments.independent_dl.campaign import CampaignSummary


def test_run_cli_forwards_family_and_single_candidate_limit(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    captured: dict[str, object] = {}
    campaign = object()
    summary = CampaignSummary(
        campaign_id="tiny",
        completed=(),
        failed=(),
        pending=("next",),
        registered=(),
        output_root=tmp_path,
    )
    monkeypatch.setattr(module, "load_campaign", lambda path: campaign)
    monkeypatch.setattr(module, "OfficialCampaignRuntime", lambda *args, **kwargs: object())

    def fake_run_campaign(*args, **kwargs):
        captured.update(kwargs)
        return summary

    monkeypatch.setattr(module, "run_campaign", fake_run_campaign)

    assert module.main(
        [
            "run",
            "--config",
            str(tmp_path / "config.json"),
            "--data-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "output"),
            "--family",
            "tabm",
            "--max-candidates",
            "1",
        ]
    ) == 0

    assert captured == {"family": "tabm", "max_candidates": 1}
    assert json.loads(capsys.readouterr().out)["pending"] == 1


def test_run_cli_forwards_explicit_failed_candidate_retry(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    captured: dict[str, object] = {}
    summary = CampaignSummary(
        campaign_id="tiny",
        completed=(),
        failed=(),
        pending=("next",),
        registered=(),
        output_root=tmp_path,
    )
    monkeypatch.setattr(module, "load_campaign", lambda path: object())
    monkeypatch.setattr(module, "OfficialCampaignRuntime", lambda *args, **kwargs: object())

    def fake_run_campaign(*args, **kwargs):
        captured.update(kwargs)
        return summary

    monkeypatch.setattr(module, "run_campaign", fake_run_campaign)

    assert module.main(
        [
            "run",
            "--config",
            str(tmp_path / "config.json"),
            "--data-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "output"),
            "--family",
            "ft_transformer",
            "--max-candidates",
            "1",
            "--retry-candidate",
            "ft_transformer__raw_typed__p3__s42",
        ]
    ) == 0

    assert captured["retry_candidate_id"] == "ft_transformer__raw_typed__p3__s42"


def test_status_cli_lists_next_candidate_and_remaining_count_by_family(
    tmp_path: Path, capsys
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    manifest = {
        "campaign_id": "tiny",
        "candidates": {
            "tabm_done": {"candidate": {"family": "tabm"}, "state": "completed"},
            "tabm_p3": {"candidate": {"family": "tabm"}, "state": "pending"},
            "tabm_p4": {"candidate": {"family": "tabm"}, "state": "pending"},
            "tabm_expand": {"candidate": {"family": "tabm"}, "state": "pending"},
            "tabm_confirm": {"candidate": {"family": "tabm"}, "state": "pending"},
            "resnet_next": {"candidate": {"family": "mlp_resnet"}, "state": "pending"},
        },
    }
    (output / "campaign_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    assert module.main(["status", "--output-dir", str(output)]) == 0

    payload = json.loads(capsys.readouterr().out)
    tabm = payload["family_status"]["tabm"]
    assert tabm["next_candidate"] == "tabm_p3"
    assert tabm["remaining_count"] == 4
    assert tabm["pending"] == [
        "tabm_p3",
        "tabm_p4",
        "tabm_expand",
        "tabm_confirm",
    ]
    assert payload["family_status"]["mlp_resnet"]["next_candidate"] == "resnet_next"


def test_status_cli_does_not_modify_manifest(tmp_path: Path, capsys) -> None:
    output = tmp_path / "output"
    output.mkdir()
    path = output / "campaign_manifest.json"
    source = json.dumps(
        {
            "campaign_id": "tiny",
            "candidates": {
                "next": {"candidate": {"family": "tabr"}, "state": "pending"},
            },
        },
        separators=(",", ":"),
    )
    path.write_text(source, encoding="utf-8")

    assert module.main(["status", "--output-dir", str(output)]) == 0

    capsys.readouterr()
    assert path.read_text(encoding="utf-8") == source
