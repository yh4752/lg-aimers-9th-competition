from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.catboost_50_50_realign.contracts import load_contract
from experiments.catboost_50_50_realign.metrics import TABM_COLUMNS
from experiments.catboost_50_50_realign.tabm_fold import (
    RealignTabMFoldError,
    build_tabm_f1_job,
    run_tabm_f1,
)
from experiments.tabm_campaign.row_feature_runtime import CampaignJobResult


def _data(root: Path) -> Path:
    root.mkdir()
    pd.DataFrame(
        {
            "row_id": ["tr1", "tr2", "va1", "va2"],
            "season": [2021, 2021, 2022, 2022],
            "control_success": [0, 1, 0, 1],
            "game_type": ["R", "R", "F", "R"],
            "game_month": ["04", "05", "06", "07"],
            "pitcher_id": ["p1", "p2", "p1", "p3"],
            "batter_id": ["b1", "b2", "b3", "b2"],
        }
    ).to_csv(root / "train.csv", index=False)
    return root


def _completed(output: Path) -> CampaignJobResult:
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best_checkpoint.pt"
    checkpoint.write_bytes(b"best")
    (output / "checkpoint.pt").write_bytes(b"resume")
    (output / "checkpoint_meta.json").write_text("{}", encoding="utf-8")
    predictions = output / "predictions.csv"
    pd.DataFrame(
        {
            "row_id": ["va1", "va2"],
            "target": [0, 1],
            "probability": [0.25, 0.75],
            "game_type": ["F", "R"],
            "game_month": ["06", "07"],
            "pitcher_id_known": ["known", "oov"],
            "batter_id_known": ["oov", "known"],
            "count_state": ["0-0", "0-0"],
        }
    ).to_csv(predictions, index=False)
    return CampaignJobResult(
        candidate_id="realign_tabm__tr2021__va2022__s3407",
        status="completed",
        brier=0.0625,
        best_epoch=2,
        completed_epochs=3,
        checkpoint=checkpoint,
        predictions_path=predictions,
        resource_evidence={"cache_digest": "c" * 64},
        failure=None,
    )


def _accept_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    import experiments.catboost_50_50_realign.tabm_fold as module

    monkeypatch.setattr(module, "_valid_completed_result", lambda *args: True)
    monkeypatch.setattr(module, "_validate_checkpoint_meta", lambda *args, **kwargs: None)


def test_build_tabm_f1_job_is_exact() -> None:
    assert asdict(build_tabm_f1_job(load_contract())) == {
        "candidate_id": "realign_tabm__tr2021__va2022__s3407",
        "capacity": "p2",
        "k": 32,
        "width": 512,
        "blocks": 4,
        "dropout": 0.1,
        "num_embedding": "piecewise_linear",
        "loss": "bce",
        "scheduler": "plateau",
        "learning_rate": 0.0006,
        "seed": 3407,
        "train_end_year": 2021,
        "valid_year": 2022,
        "sample_mode": "full",
        "max_epochs": 40,
        "min_epochs": 3,
        "patience": 10,
        "feature_bundle": None,
    }


def test_run_tabm_f1_wraps_existing_worker_and_normalizes_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _accept_provenance(monkeypatch)
    data_dir = _data(tmp_path / "data")
    output = tmp_path / "output"
    calls: list[object] = []

    def worker(job, actual_data, actual_output, cache, deadline):
        calls.append((job, actual_data, actual_output, cache, deadline))
        return _completed(actual_output)

    result = run_tabm_f1(
        data_dir=data_dir,
        output_dir=output,
        cache_root=tmp_path / "cache",
        absolute_deadline=1234.0,
        worker=worker,
    )

    assert len(calls) == 1
    assert calls[0][1:] == (data_dir, output, tmp_path / "cache", 1234.0)
    assert result.status == "completed"
    assert result.completed_epochs == 3
    assert result.brier == 0.0625
    assert tuple(pd.read_csv(result.predictions_path).columns) == TABM_COLUMNS
    assert result.predictions_path.name == "tabm_f1_predictions.csv"


@pytest.mark.parametrize("status", ["failed", "inconclusive"])
def test_noncompleted_worker_result_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    _accept_provenance(monkeypatch)
    data_dir = _data(tmp_path / "data")

    def worker(job, actual_data, actual_output, cache, deadline):
        return replace(_completed(actual_output), status=status)

    with pytest.raises(RealignTabMFoldError, match="completed result is required"):
        run_tabm_f1(
            data_dir=data_dir,
            output_dir=tmp_path / "output",
            cache_root=tmp_path / "cache",
            absolute_deadline=1234.0,
            worker=worker,
        )


def test_candidate_mismatch_is_rejected_before_artifact_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _accept_provenance(monkeypatch)
    data_dir = _data(tmp_path / "data")

    def worker(job, actual_data, actual_output, cache, deadline):
        return replace(_completed(actual_output), candidate_id="different")

    with pytest.raises(RealignTabMFoldError, match="candidate differs"):
        run_tabm_f1(
            data_dir=data_dir,
            output_dir=tmp_path / "output",
            cache_root=tmp_path / "cache",
            absolute_deadline=1234.0,
            worker=worker,
        )


def test_worker_artifact_binding_failure_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.catboost_50_50_realign.tabm_fold as module

    monkeypatch.setattr(module, "_valid_completed_result", lambda *args: False)
    data_dir = _data(tmp_path / "data")

    with pytest.raises(RealignTabMFoldError, match="artifact binding differs"):
        run_tabm_f1(
            data_dir=data_dir,
            output_dir=tmp_path / "output",
            cache_root=tmp_path / "cache",
            absolute_deadline=1234.0,
            worker=lambda job, data, output, cache, deadline: _completed(output),
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("row_id", "different", "row alignment differs"),
        ("target", 1, "target differs"),
        ("game_type", "different", "segment differs"),
        ("pitcher_id_known", "oov", "segment differs"),
    ],
)
def test_prediction_source_alignment_is_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    value: object,
    message: str,
) -> None:
    _accept_provenance(monkeypatch)
    data_dir = _data(tmp_path / "data")

    def worker(job, actual_data, actual_output, cache, deadline):
        result = _completed(actual_output)
        frame = pd.read_csv(result.predictions_path)
        frame.loc[0, column] = value
        frame.to_csv(result.predictions_path, index=False)
        return result

    with pytest.raises(RealignTabMFoldError, match=message):
        run_tabm_f1(
            data_dir=data_dir,
            output_dir=tmp_path / "output",
            cache_root=tmp_path / "cache",
            absolute_deadline=1234.0,
            worker=worker,
        )


def test_checkpoint_semantic_validator_failure_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.catboost_50_50_realign.tabm_fold as module

    monkeypatch.setattr(module, "_valid_completed_result", lambda *args: True)
    monkeypatch.setattr(
        module,
        "_validate_checkpoint_meta",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("invalid model state")),
    )
    data_dir = _data(tmp_path / "data")
    with pytest.raises(RealignTabMFoldError, match="checkpoint is invalid"):
        run_tabm_f1(
            data_dir=data_dir,
            output_dir=tmp_path / "output",
            cache_root=tmp_path / "cache",
            absolute_deadline=1234.0,
            worker=lambda job, data, output, cache, deadline: _completed(output),
        )


def test_existing_active_checkpoint_is_validated_before_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.catboost_50_50_realign.tabm_fold as module

    data_dir = _data(tmp_path / "data")
    output = tmp_path / "output"
    output.mkdir()
    (output / "checkpoint.pt").write_bytes(b"forged")
    (output / "checkpoint_meta.json").write_text(
        json.dumps({"epoch": 2}), encoding="utf-8"
    )
    worker_called = False

    def reject_checkpoint(*args, **kwargs):
        raise ValueError("invalid model state")

    def worker(*args, **kwargs):
        nonlocal worker_called
        worker_called = True
        raise AssertionError("worker must not start")

    monkeypatch.setattr(module, "_validate_checkpoint_meta", reject_checkpoint)
    with pytest.raises(RealignTabMFoldError, match="existing checkpoint is invalid"):
        run_tabm_f1(
            data_dir=data_dir,
            output_dir=output,
            cache_root=tmp_path / "cache",
            absolute_deadline=1234.0,
            worker=worker,
        )
    assert worker_called is False
