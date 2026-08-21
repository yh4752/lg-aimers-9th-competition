from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import pandas as pd

from experiments.catboost_50_50_realign.inputs import VerifiedRealignInput
from experiments.catboost_50_50_realign.metrics import PrefixEvidence, RealignDecision
from experiments.catboost_50_50_realign.runner import run_campaign
from experiments.catboost_50_50_realign.runner import (
    RealignRunnerError,
    build_existing_fold_predictions,
)
from experiments.catboost_50_50_realign.tabm_fold import TabMFoldResult
from experiments.catboost_50_50_realign.training import CatBoostJobResult
from experiments.catboost_deployment.state import serialize_feature_state
from experiments.catboost_preprocessing.features import fit_catboost_features
import pytest


FOLDS = ("2021->2022", "2022->2023", "2023->2024")


def _verified(tmp_path: Path) -> VerifiedRealignInput:
    root = tmp_path / "input"
    data = root / "data"
    data.mkdir(parents=True)
    pd.DataFrame(
        {
            "row_id": ["r1", "r2", "r3"],
            "season": [2022, 2023, 2024],
            "control_success": [0, 1, 0],
            "game_type": ["R", "R", "F"],
            "game_month": [4, 5, 6],
            "pitcher_id": ["p1", "p2", "p3"],
            "batter_id": ["b1", "b2", "b3"],
        }
    ).to_csv(data / "train.csv", index=False)
    (data / "trackman_history.csv").write_text("x\n1\n", encoding="utf-8")
    decision = root / "audit/next_experiment.json"
    decision.parent.mkdir(parents=True)
    decision.write_text("{}", encoding="utf-8")
    empty = {fold: root / f"{fold.replace('->', '_')}.csv" for fold in FOLDS[1:]}
    for path in empty.values():
        path.write_text("fixture", encoding="utf-8")
    models = {fold: root / f"{fold.replace('->', '_')}.cbm" for fold in FOLDS[1:]}
    states = {fold: root / f"{fold.replace('->', '_')}.json" for fold in FOLDS[1:]}
    for path in (*models.values(), *states.values()):
        path.write_text("fixture", encoding="utf-8")
    return VerifiedRealignInput(
        root=root,
        data_dir=data,
        tabm_predictions=empty,
        catboost_models=models,
        catboost_states=states,
        catboost_predictions=empty,
        audit_decision=decision,
        manifest_sha256="a" * 64,
        fold_keys=FOLDS,
        audit_weight=__import__("decimal").Decimal("0.50"),
    )


def _decision(promoted: bool) -> RealignDecision:
    evidence = PrefixEvidence(
        tree_count=16,
        fold_gain={fold: 0.001 for fold in FOLDS},
        weighted_gain=0.001,
        latest_bootstrap_lower=0.0001,
        maximum_segment_regression=0.0,
        eligible_segment_count=3,
        bootstrap_status="completed",
        passed=promoted,
    )
    return RealignDecision(
        status="promoted" if promoted else "rejected",
        selected_tree_count=16 if promoted else None,
        candidates=(evidence,),
        reason=(
            "selected_by_preregistered_order"
            if promoted
            else "no_prefix_passed_all_preregistered_gates"
        ),
    )


class Runtime:
    def __init__(self, root: Path, *, promoted: bool = True, tabm_status: str = "completed"):
        self.root = root
        self.promoted = promoted
        self.tabm_status = tabm_status
        self.calls: list[str] = []

    def tabm(self, **kwargs):
        self.calls.append("tabm_f1_2022")
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / "best_checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint")
        predictions = output / "tabm_f1_predictions.csv"
        predictions.write_text("row_id,target,probability\nr1,0,0.4\n", encoding="utf-8")
        return TabMFoldResult(
            self.tabm_status,
            checkpoint,
            predictions,
            3,
            0.16,
        )

    def catboost(self, **kwargs):
        self.calls.append("catboost_f1_2022")
        return self._catboost_result(Path(kwargs["output_dir"]), "catboost_f1_2022", True)

    def existing(self, verified):
        self.calls.append("existing_folds_reused")
        return ({fold: pd.DataFrame() for fold in FOLDS[1:]}, {fold: pd.DataFrame() for fold in FOLDS[1:]})

    def evaluate(self, tabm, catboost, contract):
        self.calls.append("evaluate_prefixes")
        return _decision(self.promoted)

    def full(self, **kwargs):
        self.calls.append("catboost_full_2024")
        return self._catboost_result(Path(kwargs["output_dir"]), "catboost_full_2024", False)

    @staticmethod
    def _catboost_result(output: Path, job_id: str, predictions: bool) -> CatBoostJobResult:
        output.mkdir(parents=True, exist_ok=True)
        model = output / "model.cbm"
        state = output / "preprocessing_state.json"
        model.write_bytes(b"model")
        state.write_bytes(b"state")
        prediction_path = output / "predictions.csv" if predictions else None
        if prediction_path is not None:
            prediction_path.write_text("row_id,target,p_4\nr1,0,0.4\n", encoding="utf-8")
        return CatBoostJobResult(job_id, "completed", model, state, prediction_path, None)


def _run(
    tmp_path: Path,
    runtime: Runtime,
    *,
    resume: Path | None = None,
    callback=None,
):
    return run_campaign(
        verified=_verified(tmp_path),
        output_dir=tmp_path / "run",
        resume_bundle=resume,
        absolute_deadline=time.time() + 60,
        on_phase_resume=callback,
        tabm_runtime=runtime.tabm,
        catboost_runtime=runtime.catboost,
        full_runtime=runtime.full,
        existing_fold_runtime=runtime.existing,
        evaluation_runtime=runtime.evaluate,
    )


def test_promoted_path_is_strictly_sequential_and_writes_delivery(tmp_path: Path) -> None:
    runtime = Runtime(tmp_path)
    result = _run(tmp_path, runtime)
    assert runtime.calls == [
        "tabm_f1_2022",
        "catboost_f1_2022",
        "existing_folds_reused",
        "evaluate_prefixes",
        "catboost_full_2024",
    ]
    assert result.status == "completed"
    assert result.selected_tree_count == 16
    assert result.review_bundle.is_file()
    assert result.resume_bundle.is_file()
    assert result.delivery_bundle.is_file()


def test_rejected_path_never_calls_full_fit_or_writes_delivery(tmp_path: Path) -> None:
    runtime = Runtime(tmp_path, promoted=False)
    result = _run(tmp_path, runtime)
    assert "catboost_full_2024" not in runtime.calls
    assert result.status == "deployment_blocked"
    assert result.delivery_bundle is None


def test_inconclusive_tabm_stops_before_catboost_and_is_resumable(tmp_path: Path) -> None:
    runtime = Runtime(tmp_path, tabm_status="budget_inconclusive")
    result = _run(tmp_path, runtime)
    assert runtime.calls == ["tabm_f1_2022"]
    assert result.status == "budget_inconclusive"
    assert result.resume_bundle.is_file()
    assert result.delivery_bundle is None


def test_verified_resume_reuses_completed_tabm_and_runs_only_remaining_jobs(
    tmp_path: Path,
) -> None:
    captured: dict[str, Path] = {}
    first_runtime = Runtime(tmp_path / "first")
    _run(
        tmp_path / "first",
        first_runtime,
        callback=lambda path, phase: captured.setdefault(phase, path),
    )
    second_runtime = Runtime(tmp_path / "second")
    result = _run(
        tmp_path / "second", second_runtime, resume=captured["f1_complete"]
    )
    assert second_runtime.calls == [
        "existing_folds_reused",
        "evaluate_prefixes",
        "catboost_full_2024",
    ]
    assert result.status == "completed"


def test_existing_models_generate_new_prefixes_and_check_old_prefix_parity(
    tmp_path: Path, preprocessing_frame: pd.DataFrame
) -> None:
    root = tmp_path / "existing"
    data = root / "data"
    data.mkdir(parents=True)
    full = preprocessing_frame.copy()
    full["season"] = [2021, 2022, 2023, 2024, 2021]
    full["row_id"] = [f"row-{index}" for index in range(len(full))]
    full["control_success"] = [0, 1, 0, 1, 0]
    full["game_month"] = [4, 5, 6, 7, 8]
    full.to_csv(data / "train.csv", index=False)
    (data / "trackman_history.csv").write_text("x\n1\n")
    state, _ = fit_catboost_features(
        full.loc[full["season"].le(2022)].reset_index(drop=True),
        components=("hand_matchup",),
    )

    class ExistingModel:
        def predict(self, frame, *, ntree_end):
            return np.full(len(frame), 0.4 + ntree_end / 1000)

    tabm_paths = {}
    model_paths = {}
    state_paths = {}
    prior_paths = {}
    for fold, year in (("2022->2023", 2023), ("2023->2024", 2024)):
        valid = full.loc[full["season"].eq(year)].reset_index(drop=True)
        tabm_path = root / f"tabm-{year}.csv"
        pd.DataFrame(
            {
                "row_id": valid["row_id"],
                "target": valid["control_success"],
                "probability": 0.5,
                "game_type": valid["game_type"],
                "game_month": valid["game_month"],
                "pitcher_id_known": "known",
                "batter_id_known": "known",
            }
        ).to_csv(tabm_path, index=False)
        model_path = root / f"model-{year}.cbm"
        model_path.write_bytes(b"model")
        state_path = root / f"state-{year}.json"
        state_path.write_bytes(serialize_feature_state(state))
        prior_path = root / f"prior-{year}.csv"
        pd.DataFrame({"p_4": [0.404], "p_32": [0.432]}).to_csv(prior_path, index=False)
        tabm_paths[fold] = tabm_path
        model_paths[fold] = model_path
        state_paths[fold] = state_path
        prior_paths[fold] = prior_path
    verified = VerifiedRealignInput(
        root=root,
        data_dir=data,
        tabm_predictions=tabm_paths,
        catboost_models=model_paths,
        catboost_states=state_paths,
        catboost_predictions=prior_paths,
        audit_decision=root / "audit.json",
        manifest_sha256="a" * 64,
        fold_keys=FOLDS,
        audit_weight=__import__("decimal").Decimal("0.50"),
    )
    _, catboost = build_existing_fold_predictions(
        verified, model_loader=lambda path: ExistingModel()
    )
    assert tuple(catboost["2022->2023"].columns) == (
        "row_id",
        "target",
        "p_4",
        "p_8",
        "p_12",
        "p_16",
        "p_20",
        "p_24",
        "p_28",
        "p_32",
        "game_type",
        "game_month",
        "pitcher_id_known",
        "batter_id_known",
    )
    pd.DataFrame({"p_4": [0.9], "p_32": [0.432]}).to_csv(
        prior_paths["2022->2023"], index=False
    )
    with pytest.raises(RealignRunnerError, match="parity differs"):
        build_existing_fold_predictions(verified, model_loader=lambda path: ExistingModel())
