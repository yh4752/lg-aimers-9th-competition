from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

from experiments.hierarchical_tabm.contracts import build_jobs, load_contract
from experiments.hierarchical_tabm.training import (
    HierarchicalTrainingError,
    PREDICTION_COLUMNS,
    _training_config,
    run_training_job,
    training_result_payload,
)


def _data(path: Path) -> Path:
    rows = []
    for season in range(2019, 2025):
        for position in range(4):
            rows.append(
                {
                    "row_id": f"{season}-{position}",
                    "season": season,
                    "game_month": 4 + position,
                    "game_type": "R" if position < 3 else "P",
                    "balls_before": position % 4,
                    "strikes_before": position % 3,
                    "pitcher_hand": "R" if position % 2 else "L",
                    "batter_hand": "L" if position % 2 else "R",
                    "base_state": f"{position:03b}",
                    "outs_before": position % 3,
                    "pitcher_id": f"p{position % 2}",
                    "batter_id": f"b{position % 3}",
                    "asof_pitcher_n": 10 + position,
                    "asof_pitcher_success_rate": 0.4 + position * 0.05,
                    "asof_batter_n": 20 + position,
                    "asof_batter_success_rate": 0.5 - position * 0.05,
                    "li": 0.2 + position,
                    "control_success": position % 2,
                }
            )
    path.mkdir()
    pd.DataFrame(rows).to_csv(path / "train.csv", index=False)
    return path


class FakeFit:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        output = kwargs["output_dir"]
        checkpoint = output / "checkpoint.pt"
        checkpoint.write_bytes(b"verified-checkpoint")
        if kwargs["job"].kind == "full_fit":
            model = output / "final_checkpoint.pt"
            model.write_bytes(b"final-model")
            return {
                "status": "completed",
                "completed_epochs": kwargs["final_epochs"],
                "checkpoint_path": checkpoint,
                "final_model_path": model,
            }
        target = kwargs["valid_rows"]["control_success"].to_numpy(dtype="float64")
        probability = 0.2 + 0.6 * target
        return {
            "status": "completed",
            "best_epoch": 2,
            "best_brier": float(np.mean((target - probability) ** 2)),
            "completed_epochs": 3,
            "probability": probability,
            "checkpoint_path": checkpoint,
        }


def _identity() -> dict[str, str]:
    return {"contract_sha256": "a" * 64, "code_sha256": "b" * 64}


def test_oof_worker_uses_only_fold_training_for_context(tmp_path: Path) -> None:
    fit = FakeFit()
    job = build_jobs(load_contract())[0]
    result = run_training_job(
        job,
        data_dir=_data(tmp_path / "data"),
        output_dir=tmp_path / "out",
        selected_k=128.0,
        absolute_deadline=time.time() + 60,
        identity=_identity(),
        fit=fit,
    )
    call = fit.calls[0]
    assert call["fit_rows"]["season"].max() == 2022
    assert call["valid_rows"]["season"].unique().tolist() == [2023]
    assert call["context_state"].row_count == len(call["fit_rows"])
    assert result.status == "completed"
    assert result.predictions_path.name == "predictions.csv"


def test_worker_emits_exact_anchor_aligned_prediction_schema(tmp_path: Path) -> None:
    result = run_training_job(
        build_jobs(load_contract())[0],
        data_dir=_data(tmp_path / "data"),
        output_dir=tmp_path / "out",
        selected_k=32.0,
        absolute_deadline=time.time() + 60,
        identity=_identity(),
        fit=FakeFit(),
    )
    frame = pd.read_csv(result.predictions_path)
    assert tuple(frame) == PREDICTION_COLUMNS
    assert frame["row_id"].tolist() == [f"2023-{i}" for i in range(4)]
    assert result.best_epoch == 2
    assert result.best_brier == pytest.approx(0.04)
    assert json.loads((tmp_path / "out/worker_result.json").read_text())["status"] == "completed"


def test_expired_deadline_returns_incomplete_without_starting_fit(tmp_path: Path) -> None:
    fit = FakeFit()
    result = run_training_job(
        build_jobs(load_contract())[0],
        data_dir=_data(tmp_path / "data"),
        output_dir=tmp_path / "out",
        selected_k=32.0,
        absolute_deadline=time.time() - 1,
        identity=_identity(),
        fit=fit,
    )
    assert result.status == "incomplete"
    assert fit.calls == []
    assert result.failure == "deadline_reached_before_training"


def test_resume_rejects_identity_mismatch_before_fit(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint.with_suffix(".pt.meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_id": build_jobs(load_contract())[0].job_id,
                "identity": {"contract_sha256": "0" * 64, "code_sha256": "b" * 64},
                "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
                "completed_epochs": 2,
            }
        )
    )
    with pytest.raises(HierarchicalTrainingError, match="identity differs"):
        run_training_job(
            build_jobs(load_contract())[0],
            data_dir=_data(tmp_path / "data"),
            output_dir=tmp_path / "out",
            selected_k=32.0,
            absolute_deadline=time.time() + 60,
            identity=_identity(),
            resume_checkpoint=checkpoint,
            fit=FakeFit(),
        )


def test_full_fit_has_no_validation_and_exact_epoch_count(tmp_path: Path) -> None:
    fit = FakeFit()
    result = run_training_job(
        build_jobs(load_contract())[-1],
        data_dir=_data(tmp_path / "data"),
        output_dir=tmp_path / "out",
        selected_k=512.0,
        absolute_deadline=time.time() + 60,
        identity=_identity(),
        final_epochs=5,
        fit=fit,
    )
    assert fit.calls[0]["valid_rows"] is None
    assert fit.calls[0]["prepared"].valid is None
    assert result.valid_rows is None
    assert result.completed_epochs == 5
    assert result.final_model_path.is_file()


def test_result_payload_hashes_every_referenced_file(tmp_path: Path) -> None:
    result = run_training_job(
        build_jobs(load_contract())[0],
        data_dir=_data(tmp_path / "data"), output_dir=tmp_path / "out",
        selected_k=32.0, absolute_deadline=time.time() + 60,
        identity=_identity(), fit=FakeFit(),
    )
    payload = training_result_payload(result)
    assert payload["checkpoint_sha256"] == sha256(result.checkpoint_path.read_bytes()).hexdigest()
    assert payload["predictions_sha256"] == sha256(result.predictions_path.read_bytes()).hexdigest()
    assert payload["feature_state_sha256"] == sha256(result.feature_state_path.read_bytes()).hexdigest()


def test_full_fit_uses_constant_schedule() -> None:
    assert _training_config(epochs=5, full_fit=True)["scheduler"] == "constant"
    assert _training_config(epochs=40, full_fit=False)["scheduler"] == "plateau"
