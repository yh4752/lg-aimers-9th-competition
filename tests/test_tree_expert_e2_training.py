from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.tree_expert.contracts import E1Job
from experiments.tree_expert.e2_contracts import build_structure_jobs, load_e2_contract
from experiments.tree_expert.e2_training import (
    E2TrainingError,
    expected_job_identity,
    reusable_completed_job,
    run_e2_fold_job,
)
from experiments.tree_expert.inputs import PREDICTION_COLUMNS, VerifiedOfficialData
from experiments.tree_expert.training import FoldResult


def _baseline() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["r1"],
            "target": [1],
            "probability": [0.6],
            "game_type": ["R"],
            "game_month": [4],
            "pitcher_id_known": ["known"],
            "batter_id_known": ["known"],
        }
    ).loc[:, PREDICTION_COLUMNS]


def _data(tmp_path: Path) -> VerifiedOfficialData:
    train = tmp_path / "train.csv"
    history = tmp_path / "trackman_history.csv"
    train.write_text("row_id,season,control_success\nr1,2022,1\n")
    history.write_text("pitcher_id,season\np1,2022\n")
    return VerifiedOfficialData(tmp_path, train, history, "a" * 64, "b" * 64)


def _job():
    return build_structure_jobs(
        load_e2_contract(), seed=3407, folds=((2021, 2022),)
    )[0]


def test_wrapper_maps_e2_job_to_e1_without_changing_semantics(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fold_runner(**kwargs: object) -> FoldResult:
        captured.update(kwargs)
        output = Path(kwargs["output_dir"])
        model = output / "model.cbm"
        predictions = output / "predictions.csv"
        model.write_bytes(b"model")
        _baseline().to_csv(predictions, index=False)
        return FoldResult(
            job_id=kwargs["job"].job_id,
            candidate_id=kwargs["job"].candidate_id,
            status="completed",
            brier=0.24,
            model_path=model,
            predictions_path=predictions,
            snapshot_path=None,
            failure=None,
        )

    result = run_e2_fold_job(
        job=_job(),
        data=_data(tmp_path),
        baseline=_baseline(),
        output_dir=tmp_path / "job",
        absolute_deadline=10**12,
        gpu_id=0,
        input_manifest_sha256="c" * 64,
        fold_runner=fold_runner,
    )

    mapped = captured["job"]
    assert isinstance(mapped, E1Job)
    assert mapped.train_end_year == 2021
    assert mapped.valid_year == 2022
    assert mapped.seed == 3407
    assert mapped.objective == "residual"
    assert mapped.use_trackman is False
    assert result.status == "completed"
    assert (tmp_path / "job" / "e2_job_binding.json").is_file()


@pytest.mark.parametrize("field", ["candidate_id", "fold", "seed", "input"])
def test_reuse_rejects_changed_identity(tmp_path: Path, field: str) -> None:
    job = _job()
    data = _data(tmp_path)
    identity = expected_job_identity(job, data, "c" * 64)
    output = tmp_path / "job"
    output.mkdir()
    model = output / "model.cbm"
    prediction = output / "predictions.csv"
    model.write_bytes(b"model")
    _baseline().to_csv(prediction, index=False)
    binding = {
        "schema_version": 1,
        "identity": identity,
        "artifacts": {
            "model.cbm": {"sha256": _sha(model), "size": model.stat().st_size},
            "predictions.csv": {
                "sha256": _sha(prediction),
                "size": prediction.stat().st_size,
            },
        },
    }
    (output / "e2_job_binding.json").write_text(json.dumps(binding))
    (output / "worker_result.json").write_text(
        json.dumps(
            {
                "job_id": job.job_id,
                "candidate_id": job.candidate_id,
                "status": "completed",
                "brier": 0.24,
                "model": "model.cbm",
                "predictions": "predictions.csv",
                "snapshot": None,
                "failure": None,
            }
        )
    )
    changed = dict(identity)
    if field == "candidate_id":
        changed["candidate_id"] = "wrong"
    elif field == "fold":
        changed["valid_year"] = 2023
    elif field == "seed":
        changed["seed"] = 42
    else:
        changed["input_manifest_sha256"] = "d" * 64

    assert reusable_completed_job(output, job, changed) is None


def test_reuse_requires_all_terminal_files(tmp_path: Path) -> None:
    job = _job()
    data = _data(tmp_path)
    identity = expected_job_identity(job, data, "c" * 64)
    output = tmp_path / "job"

    def fold_runner(**kwargs: object) -> FoldResult:
        target = Path(kwargs["output_dir"])
        model = target / "model.cbm"
        predictions = target / "predictions.csv"
        model.write_bytes(b"model")
        _baseline().to_csv(predictions, index=False)
        return FoldResult(job.job_id, job.candidate_id, "completed", 0.24, model, predictions, None, None)

    run_e2_fold_job(
        job=job,
        data=data,
        baseline=_baseline(),
        output_dir=output,
        absolute_deadline=10**12,
        gpu_id=0,
        input_manifest_sha256="c" * 64,
        fold_runner=fold_runner,
    )
    (output / "predictions.csv").unlink()

    assert reusable_completed_job(output, job, identity) is None


def _sha(path: Path) -> str:
    from hashlib import sha256

    return sha256(path.read_bytes()).hexdigest()


def test_wrapper_rejects_result_for_another_job(tmp_path: Path) -> None:
    job = _job()

    def fold_runner(**kwargs: object) -> FoldResult:
        return FoldResult("wrong", "wrong", "failed", None, None, None, None, "x")

    with pytest.raises(E2TrainingError, match="result identity differs"):
        run_e2_fold_job(
            job=job,
            data=_data(tmp_path),
            baseline=_baseline(),
            output_dir=tmp_path / "job",
            absolute_deadline=10**12,
            gpu_id=0,
            input_manifest_sha256="c" * 64,
            fold_runner=fold_runner,
        )
