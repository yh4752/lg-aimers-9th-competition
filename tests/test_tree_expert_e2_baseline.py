from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.independent_dl.features import FeatureBatch
from experiments.independent_dl.models.common import ModelMetadata
from experiments.tree_expert.e2_baseline import (
    E2BaselineError,
    align_baseline,
    build_baseline_plan,
    load_reused_baselines,
    make_f1_request,
    run_f1_baseline,
)
from experiments.tree_expert.e2_contracts import load_e2_contract
from experiments.tree_expert.e2_inputs import VerifiedE2Input
from experiments.tree_expert.inputs import PREDICTION_COLUMNS, VerifiedOfficialData


def _official_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["a", "b", "c", "d"],
            "season": [2021, 2022, 2022, 2023],
            "control_success": [1, 0, 1, 0],
            "game_type": ["R", "R", "F", "R"],
            "game_month": [4, 5, 6, 7],
            "pitcher_id": ["p1", "p1", "p2", "p3"],
            "batter_id": ["b1", "b2", "b3", "b4"],
        }
    )


def _prediction(rows: pd.DataFrame, probability: tuple[float, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": rows["row_id"].astype(str).to_numpy(),
            "target": rows["control_success"].astype(int).to_numpy(),
            "probability": probability,
            "game_type": rows["game_type"].to_numpy(),
            "game_month": rows["game_month"].to_numpy(),
            "pitcher_id_known": "known",
            "batter_id_known": "known",
        }
    ).loc[:, PREDICTION_COLUMNS]


def _feature_batch(row_ids: tuple[str, ...], target: tuple[int, ...] | None) -> FeatureBatch:
    size = len(row_ids)
    return FeatureBatch(
        row_id=np.asarray(row_ids),
        season=np.full(size, 2022),
        game_type=np.full(size, "R"),
        x_num=np.ones((size, 2), dtype="float32"),
        x_cat=np.ones((size, 1), dtype="int64"),
        y=None if target is None else np.asarray(target, dtype="float32"),
    )


def _fake_cache() -> SimpleNamespace:
    train = _feature_batch(("a", "b"), (1, 0))
    valid = _feature_batch(("c", "d"), (1, 0))
    return SimpleNamespace(
        train=train,
        valid=valid,
        model_metadata=ModelMetadata(2, (3,), train.x_num),
        identity=SimpleNamespace(digest=lambda: "1" * 64),
        state=SimpleNamespace(
            category_maps=MappingProxyType(
                {
                    "pitcher_id": MappingProxyType({"p1": 1}),
                    "batter_id": MappingProxyType({"b1": 1}),
                }
            )
        ),
    )


def test_baseline_plan_reuses_f2_f3_and_trains_only_f1(tmp_path: Path) -> None:
    evidence = SimpleNamespace(
        tabm_predictions={
            "2022->2023": tmp_path / "f2.csv",
            "2023->2024": tmp_path / "f3.csv",
        }
    )

    plan = build_baseline_plan(load_e2_contract(), evidence)

    assert [(item.fold, item.action) for item in plan] == [
        ((2021, 2022), "train"),
        ((2022, 2023), "reuse"),
        ((2023, 2024), "reuse"),
    ]


def test_f1_request_matches_stage_c_configuration() -> None:
    request = make_f1_request(
        _fake_cache(),
        load_e2_contract(),
        checkpoint_binding={"config_sha256": "a" * 64},
    )

    assert request.candidate_id == (
        "e2_baseline__tabm_p2_piecewise_linear_bce_plateau"
        "__tr2021__va2022__s3407"
    )
    assert request.seed == 3407
    assert request.epochs == 40
    assert request.min_epochs == 3
    assert request.model_config == {
        "architecture": "tabm",
        "k": 32,
        "width": 512,
        "blocks": 4,
        "dropout": 0.1,
        "num_embedding": "piecewise_linear",
    }
    assert request.training_config["scheduler"] == "plateau"
    assert request.training_config["learning_rate"] == 0.0006
    assert request.training_config["effective_batch_size"] == 4096


@pytest.mark.parametrize("mutation", ["reverse", "target", "fold"])
def test_alignment_rejects_changed_evidence(mutation: str) -> None:
    official = _official_rows()
    valid = official.loc[official["season"].eq(2022)].copy()
    baseline = _prediction(valid, (0.4, 0.6))
    if mutation == "reverse":
        baseline = baseline.iloc[::-1].reset_index(drop=True)
    elif mutation == "target":
        baseline.loc[0, "target"] = 1
    else:
        official.loc[official["season"].eq(2022), "season"] = 2021

    with pytest.raises(E2BaselineError, match="alignment|target|fold"):
        align_baseline(baseline, official, fold=(2021, 2022))


def test_reused_baselines_are_aligned_to_official_rows(tmp_path: Path) -> None:
    official = _official_rows()
    train_path = tmp_path / "train.csv"
    history_path = tmp_path / "trackman_history.csv"
    official.to_csv(train_path, index=False)
    history_path.write_text("pitcher_id,season\n")
    f2 = _prediction(official.loc[official["season"].eq(2023)], (0.5,))
    f3_rows = official.loc[official["season"].eq(2022)].copy()
    f3_rows["season"] = 2024
    official = pd.concat([official, f3_rows], ignore_index=True)
    official.to_csv(train_path, index=False)
    f3 = _prediction(f3_rows, (0.4, 0.6))
    f2_path = tmp_path / "f2.csv"
    f3_path = tmp_path / "f3.csv"
    f2.to_csv(f2_path, index=False)
    f3.to_csv(f3_path, index=False)
    evidence = VerifiedE2Input(
        root=tmp_path,
        archive_sha256="a" * 64,
        manifest_sha256="b" * 64,
        e1_predictions=MappingProxyType({}),
        tabm_predictions=MappingProxyType(
            {"2022->2023": f2_path, "2023->2024": f3_path}
        ),
        tabm_runtime_root=tmp_path,
        lineage=MappingProxyType({}),
    )
    data = VerifiedOfficialData(tmp_path, train_path, history_path, "c" * 64, "d" * 64)

    loaded = load_reused_baselines(evidence, data)

    assert tuple(loaded) == ((2022, 2023), (2023, 2024))
    assert loaded[(2023, 2024)]["row_id"].tolist() == ["b", "c"]


def test_budget_reached_f1_is_inconclusive_and_uses_all_fit_rows(tmp_path: Path) -> None:
    official = _official_rows()
    train_path = tmp_path / "train.csv"
    history_path = tmp_path / "trackman_history.csv"
    official.to_csv(train_path, index=False)
    history_path.write_text("pitcher_id,season\n")
    data = VerifiedOfficialData(tmp_path, train_path, history_path, "c" * 64, "d" * 64)
    train_batch = _feature_batch(("a",), (1,))
    cache = SimpleNamespace(
        train=train_batch,
        valid=_feature_batch(("b", "c"), (0, 1)),
        model_metadata=ModelMetadata(2, (3,), train_batch.x_num),
        identity=SimpleNamespace(digest=lambda: "1" * 64),
        state=SimpleNamespace(
            category_maps=MappingProxyType(
                {
                    "pitcher_id": MappingProxyType({"p1": 1}),
                    "batter_id": MappingProxyType({"b1": 1}),
                }
            )
        ),
    )
    captured: dict[str, object] = {}

    def cache_builder(cache_root: Path, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return cache

    def trainer(request: object, adapter: object, output_dir: Path) -> SimpleNamespace:
        checkpoint = output_dir / "best_checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint")
        return SimpleNamespace(
            predictions=np.asarray([0.4, 0.6]),
            best_brier=0.25,
            best_epoch=0,
            completed_epochs=1,
            checkpoint=checkpoint,
            budget_reached=True,
            hardware={},
        )

    result = run_f1_baseline(
        data=data,
        output_dir=tmp_path / "job",
        cache_root=tmp_path / "cache",
        absolute_deadline=10**12,
        cache_builder=cache_builder,
        trainer=trainer,
        adapter_factory=lambda: object(),
    )

    assert result.status == "inconclusive"
    assert result.ready_for_structure is False
    assert tuple(captured["sample_ids"]) == ("a",)
