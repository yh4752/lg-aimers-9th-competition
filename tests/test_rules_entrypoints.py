from __future__ import annotations

import json
from pathlib import Path

import pytest

from competition_rules.contract import (
    RulesContractError,
    assert_experiment_runnable,
)
from experiments.independent_dl import run_campaign as independent_cli
from experiments.preprocessing_campaign import run_budgeted_campaign as budgeted_cli
from experiments.preprocessing_campaign import run_campaign as preprocessing_cli


ROOT = Path(__file__).resolve().parents[1]


def _blocked(**_: object) -> dict[str, object]:
    raise RulesContractError("blocked before data")


def test_real_runnable_assertion_binds_source_and_candidates() -> None:
    report = assert_experiment_runnable(
        project_root=ROOT,
        contract_path=ROOT / "experiments/independent_dl/experiment_contract.json",
        config_path=ROOT / "experiments/independent_dl/configs/campaign_v1.json",
        candidate_ids=["tabm__raw_typed__p1__s42"],
    )

    assert report["status"] == "passed"
    assert report["candidate_count"] == 64
    assert report["source_gate"]["status"] == "passed"


def test_independent_run_gates_before_runtime_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_calls: list[str] = []
    monkeypatch.setattr(independent_cli, "assert_experiment_runnable", _blocked, raising=False)
    monkeypatch.setattr(
        independent_cli,
        "OfficialCampaignRuntime",
        lambda *args, **kwargs: runtime_calls.append("runtime"),
    )

    with pytest.raises(RulesContractError, match="blocked before data"):
        independent_cli.main(
            [
                "run",
                "--config",
                str(tmp_path / "missing.json"),
                "--data-dir",
                str(tmp_path / "data"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
    assert runtime_calls == []


def test_independent_status_is_read_only_and_does_not_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "campaign_manifest.json").write_text(
        json.dumps({"campaign_id": "tiny", "candidates": {}}), encoding="utf-8"
    )
    monkeypatch.setattr(independent_cli, "assert_experiment_runnable", _blocked, raising=False)

    assert independent_cli.main(["status", "--output-dir", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["campaign_id"] == "tiny"


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "run", "--wave", "a", "--config", "missing.json",
            "--data-dir", "missing", "--output-dir", "output",
        ],
        [
            "promote", "--from-wave", "a", "--config", "missing.json",
            "--output-dir", "output",
        ],
    ],
)
def test_preprocessing_mutations_gate_before_campaign_or_data(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    monkeypatch.setattr(preprocessing_cli, "assert_experiment_runnable", _blocked, raising=False)
    monkeypatch.setattr(
        preprocessing_cli,
        "load_preprocessing_campaign",
        lambda *_: (_ for _ in ()).throw(AssertionError("campaign loaded")),
    )

    with pytest.raises(RulesContractError, match="blocked before data"):
        preprocessing_cli.main(arguments)


def test_budgeted_auto_gates_before_environment_or_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(budgeted_cli, "assert_experiment_runnable", _blocked, raising=False)
    monkeypatch.setattr(
        budgeted_cli,
        "run_auto",
        lambda **_: (_ for _ in ()).throw(AssertionError("auto started")),
    )

    with pytest.raises(RulesContractError, match="blocked before data"):
        budgeted_cli.main(
            [
                "auto",
                "--config", str(tmp_path / "missing.json"),
                "--input-root", str(tmp_path / "input"),
                "--data-dir", str(tmp_path / "data"),
                "--output-root", str(tmp_path / "output"),
                "--session-started-unix", "0",
            ]
        )


def test_uncontracted_tabicl_is_filtered_without_blocking_tabm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_run(campaign, output, runtime, **kwargs):
        captured["ids"] = [candidate.candidate_id for candidate in campaign.candidates]
        return type(
            "Summary",
            (),
            {
                "campaign_id": "independent_dl_campaign_v1",
                "completed": (),
                "failed": (),
                "pending": (),
                "output_root": output,
            },
        )()

    monkeypatch.setattr(independent_cli, "run_campaign", fake_run)
    monkeypatch.setattr(independent_cli, "OfficialCampaignRuntime", lambda *a, **k: object())

    independent_cli.main(
        [
            "run",
            "--config", str(ROOT / "experiments/independent_dl/configs/campaign_v1.json"),
            "--data-dir", str(tmp_path / "unread"),
            "--output-dir", str(tmp_path / "output"),
        ]
    )

    assert "tabm__raw_typed__p1__s42" in captured["ids"]
    assert all("tabicl_v2" not in candidate_id for candidate_id in captured["ids"])


def test_renderers_embed_the_rules_package() -> None:
    for relative in (
        "tools/render_preprocessing_kaggle_cell.py",
        "tools/render_budgeted_preprocessing_kaggle_cell.py",
        "tools/render_independent_dl_kaggle_notebook.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert '"competition_rules"' in text
