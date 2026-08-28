from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.independent_dl.features import FeatureBatch
from experiments.independent_dl.models.common import ModelMetadata
from experiments.tree_expert.hc_base import (
    HCBaseError,
    ensemble_source_predictions,
    make_source_request,
    run_source_baseline,
    run_source_residual_job,
    source_e2_jobs,
)
from experiments.tree_expert.e2_contracts import load_e2_contract
from experiments.tree_expert.inputs import PREDICTION_COLUMNS, VerifiedOfficialData


def _feature_batch(row_ids: tuple[str, ...], target: tuple[int, ...] | None):
    size = len(row_ids)
    return FeatureBatch(
        row_id=np.asarray(row_ids),
        season=np.full(size, 2021),
        game_type=np.full(size, "R"),
        x_num=np.ones((size, 2), dtype="float32"),
        x_cat=np.ones((size, 1), dtype="int64"),
        y=None if target is None else np.asarray(target, dtype="float32"),
    )


def _cache():
    train = _feature_batch(("a", "b"), (1, 0))
    valid = _feature_batch(("c", "d"), (1, 0))
    return SimpleNamespace(
        train=train,
        valid=valid,
        model_metadata=ModelMetadata(2, (3,), train.x_num),
        identity=SimpleNamespace(digest=lambda: "1" * 64),
        state=SimpleNamespace(category_maps=MappingProxyType({})),
    )


def _prediction(probability: tuple[float, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["c", "d"],
            "target": [1, 0],
            "probability": probability,
            "game_type": ["R", "R"],
            "game_month": [4, 4],
            "pitcher_id_known": ["known", "known"],
            "batter_id_known": ["known", "known"],
        }
    ).loc[:, PREDICTION_COLUMNS]


def test_source_request_is_2020_to_2021_and_frozen():
    request = make_source_request(
        _cache(), load_e2_contract(), checkpoint_binding={"config_sha256": "a" * 64}
    )
    assert request.seed == 3407
    assert request.model_config["architecture"] == "tabm"
    assert request.model_config["num_embedding"] == "piecewise_linear"
    assert request.candidate_id.endswith("__tr2020__va2021__s3407")
    assert request.epochs == 40


def test_source_residual_jobs_are_exactly_three_registered_seeds():
    jobs = source_e2_jobs()
    assert [job.seed for job in jobs] == [3407, 42, 2026]
    assert {(job.train_end_year, job.valid_year) for job in jobs} == {(2020, 2021)}
    assert {job.candidate_id for job in jobs} == {"c1_anchor_residual"}
    assert all(job.objective == "residual" and not job.use_trackman for job in jobs)


def test_source_ensemble_is_row_aligned_arithmetic_mean(tmp_path: Path):
    paths = {}
    for seed, probabilities in {
        3407: (0.3, 0.7),
        42: (0.4, 0.6),
        2026: (0.5, 0.5),
    }.items():
        path = tmp_path / f"{seed}.csv"
        _prediction(probabilities).to_csv(path, index=False)
        paths[seed] = path
    result = ensemble_source_predictions(paths)
    assert result["probability"].tolist() == pytest.approx([0.4, 0.6])
    assert result["row_id"].tolist() == ["c", "d"]


def test_source_ensemble_rejects_reordered_seed_output(tmp_path: Path):
    paths = {}
    for seed in (3407, 42, 2026):
        frame = _prediction((0.4, 0.6))
        if seed == 42:
            frame = frame.iloc[::-1]
        path = tmp_path / f"{seed}.csv"
        frame.to_csv(path, index=False)
        paths[seed] = path
    with pytest.raises(HCBaseError, match="row alignment differs"):
        ensemble_source_predictions(paths)


def test_source_baseline_uses_only_rows_through_2020_for_2021(tmp_path: Path):
    rows = pd.DataFrame(
        {
            "row_id": ["a", "b", "c", "d", "e"],
            "season": [2019, 2020, 2021, 2021, 2022],
            "control_success": [0, 1, 1, 0, 1],
            "game_type": ["R", "R", "R", "R", "F"],
            "game_month": [3, 4, 4, 4, 5],
            "pitcher_id": ["p1", "p2", "p1", "p3", "p4"],
            "batter_id": ["b1", "b2", "b3", "b4", "b5"],
        }
    )
    train = tmp_path / "train.csv"
    history = tmp_path / "history.csv"
    rows.to_csv(train, index=False)
    history.write_text("pitcher_id,season\np1,2020\n")
    data = VerifiedOfficialData(tmp_path, train, history, "a" * 64, "b" * 64)
    cache = _cache()
    captured = {}

    def cache_builder(cache_root, **kwargs):
        captured.update(kwargs)
        return cache

    def trainer(request, adapter, output_dir):
        checkpoint = output_dir / "best_checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint")
        return SimpleNamespace(
            predictions=np.asarray([0.4, 0.6]),
            best_epoch=2,
            completed_epochs=3,
            checkpoint=checkpoint,
            budget_reached=False,
            hardware={},
        )

    result = run_source_baseline(
        data=data,
        output_dir=tmp_path / "job",
        cache_root=tmp_path / "cache",
        absolute_deadline=10**12,
        cache_builder=cache_builder,
        trainer=trainer,
        adapter_factory=lambda: object(),
    )
    assert result.status == "completed"
    assert captured["train"]["season"].tolist() == [2019, 2020]
    assert captured["valid"]["season"].tolist() == [2021, 2021]
    assert pd.read_csv(result.predictions_path)["row_id"].tolist() == ["c", "d"]


def test_source_residual_wrapper_rejects_non_source_fold(tmp_path: Path):
    wrong = source_e2_jobs()[0]
    wrong = type(wrong)(
        wrong.job_id,
        wrong.candidate_id,
        2021,
        2022,
        wrong.seed,
        wrong.objective,
        wrong.use_trackman,
    )
    with pytest.raises(HCBaseError, match="source residual job differs"):
        run_source_residual_job(
            job=wrong,
            data=object(),
            baseline=_prediction((0.4, 0.6)),
            output_dir=tmp_path,
            absolute_deadline=10**12,
            gpu_id=0,
            input_manifest_sha256="a" * 64,
        )
