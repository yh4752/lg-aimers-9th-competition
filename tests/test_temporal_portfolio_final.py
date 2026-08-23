from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from dataclasses import dataclass

from experiments.temporal_portfolio.features import PortfolioFeatureSpec
from experiments.temporal_portfolio.final_training import (
    FinalTrainingError,
    T4Decision,
    benchmark_inference,
    build_full_fit_plan,
    resolve_final_epoch,
)


def _decision(status: str = "accepted") -> T4Decision:
    return T4Decision(
        status=status,
        decision_sha256="a" * 64,
        decay=Decimal("0.55"),
        recent_weight=Decimal("0.75"),
        best_epoch_by_year={2022: 8, 2023: 5, 2024: 3},
        feature_spec=PortfolioFeatureSpec(("base", "S1", "P2"), "dl_standard"),
        model_spec={"family": "tabm", "profile": "p2", "loss": "bce"},
        catboost_prefix=None,
    )


def test_final_epoch_is_temporally_weighted_median() -> None:
    assert resolve_final_epoch({2022: 8, 2023: 5, 2024: 3}) == 3
    with pytest.raises(FinalTrainingError, match="unstable"):
        resolve_final_epoch({2022: 2, 2023: 3, 2024: 40})


def test_full_fit_uses_2024_recent_and_2021_2024_multi() -> None:
    train = pd.DataFrame({"season": [2021, 2022, 2023, 2024]})
    plan = build_full_fit_plan(train, _decision())
    assert plan.recent_years == (2024,)
    assert plan.multi_years == (2021, 2022, 2023, 2024)
    np.testing.assert_allclose(
        list(plan.multi_weight_by_year.values()),
        [0.55**3, 0.55**2, 0.55, 1.0],
    )


def test_rejected_or_unconfirmed_candidate_cannot_full_fit() -> None:
    train = pd.DataFrame({"season": [2021, 2022, 2023, 2024]})
    with pytest.raises(FinalTrainingError, match="acceptance"):
        build_full_fit_plan(train, _decision("rejected"))


class _Predictor:
    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.full(len(rows), 0.5)


class _Clock:
    def __init__(self, elapsed: float) -> None:
        self.values = iter((100.0, 100.0 + elapsed))

    def __call__(self) -> float:
        return next(self.values)


def test_full_test_inference_must_finish_inside_eight_minute_safety_gate() -> None:
    rows = pd.DataFrame({"row_id": ["a", "b"]})
    report = benchmark_inference(_Predictor(), rows, clock=_Clock(479.0))
    assert report.accepted is True
    with pytest.raises(FinalTrainingError, match="inference budget"):
        benchmark_inference(_Predictor(), rows, clock=_Clock(481.0))


class _RowLocalPredictor:
    def predict(self, rows: pd.DataFrame, *, batch_size: int = 512) -> np.ndarray:
        if "control_success" in rows:
            from experiments.temporal_portfolio.inference import InferenceRuleError

            raise InferenceRuleError("evaluation rows contain target")
        return 1.0 / (1.0 + np.exp(-rows["x"].to_numpy(dtype="float64")))


def test_frozen_prediction_is_invariant_to_order_batch_and_subset() -> None:
    from experiments.temporal_portfolio.row_independence import predict_by_row_id

    rows = pd.DataFrame(
        {"row_id": ["r0", "r1", "r2"], "x": [-2.0, 0.0, 2.0]}
    )
    predictor = _RowLocalPredictor()
    expected = predict_by_row_id(predictor, rows, batch_size=32)
    assert expected == predict_by_row_id(
        predictor, rows.sample(frac=1, random_state=3), batch_size=1
    )
    assert expected == predict_by_row_id(predictor, rows, batch_size=512)
    subset = rows.iloc[[0, 2]]
    assert {key: expected[key] for key in subset.row_id} == predict_by_row_id(
        predictor, subset, batch_size=32
    )


@dataclass(frozen=True)
class _State:
    inference_mode: bool


class _BatchModel:
    def predict(self, batch: pd.DataFrame) -> np.ndarray:
        return np.full(len(batch), 0.5)


def test_inference_rejects_target_and_unfitted_state() -> None:
    from experiments.temporal_portfolio.ensembles import Recipe
    from experiments.temporal_portfolio.inference import (
        FrozenManifest,
        FrozenPredictor,
        InferenceRuleError,
    )

    recipe = Recipe("single__main", ("main",), (Decimal("1"),), "probability")
    manifest = FrozenManifest(("main",), recipe)
    frozen = FrozenPredictor(
        manifest,
        {"main": _BatchModel()},
        {"main": _State(True)},
        transformer=lambda rows, _state: rows,
    )
    rows = pd.DataFrame({"row_id": ["r0"], "x": [1.0]})
    with pytest.raises(InferenceRuleError, match="target"):
        frozen.predict(rows.assign(control_success=0))
    with pytest.raises(InferenceRuleError, match="frozen"):
        FrozenPredictor(
            manifest,
            {"main": _BatchModel()},
            {"main": _State(False)},
            transformer=lambda values, _state: values,
        )


def test_frozen_anchor_recipe_produces_bounded_probabilities() -> None:
    from experiments.temporal_portfolio.ensembles import Recipe
    from experiments.temporal_portfolio.inference import FrozenManifest, FrozenPredictor

    recipe = Recipe(
        "single__main__anchor",
        ("main",),
        (Decimal("1"),),
        "probability",
        anchor=Decimal("0.05"),
    )
    predictor = FrozenPredictor(
        FrozenManifest(("main",), recipe, anchor_rate=0.55),
        {"main": _BatchModel()},
        {"main": _State(True)},
        transformer=lambda rows, _state: rows,
    )
    result = predictor.predict(pd.DataFrame({"row_id": ["r0"], "x": [1.0]}))
    assert result.shape == (1,)
    assert 0 < result[0] < 1


def test_training_delivery_contains_frozen_evidence_without_submission(
    tmp_path,
) -> None:
    from experiments.temporal_portfolio.final_delivery import (
        AcceptedFullFit,
        verify_training_delivery,
        write_training_delivery,
    )
    from experiments.temporal_portfolio.state import Bindings

    model = tmp_path / "model.pt"
    state = tmp_path / "state.json"
    model.write_bytes(b"weights")
    state.write_bytes(b"state")
    accepted = AcceptedFullFit(
        bindings=Bindings("temporal_portfolio_v1", "b" * 64, "c" * 64),
        decision_sha256="a" * 64,
        confirmation_sha256="d" * 64,
        frozen_members={"models/main.pt": model, "states/main.json": state},
        acceptance={"status": "accepted", "candidate_id": "main"},
        row_independence={"accepted": True, "maximum_absolute_difference": 0.0},
        runtime={"python": "3.11", "inference_seconds": 10.0},
    )
    delivery = write_training_delivery(tmp_path / "delivery", accepted)
    verified = verify_training_delivery(delivery)
    assert verified.policy["submission_package"] is False
    assert "submit.zip" not in verified.members
    assert {
        "frozen/manifest.json",
        "policy/policy.json",
        "review/acceptance.json",
    }.issubset(verified.members)


def test_packaging_is_not_authorized_by_training_delivery() -> None:
    from experiments.temporal_portfolio.final_delivery import (
        SubmissionNotAuthorized,
        assert_submission_packaging_authorized,
    )

    with pytest.raises(SubmissionNotAuthorized, match="separate reviewed step"):
        assert_submission_packaging_authorized(object())
