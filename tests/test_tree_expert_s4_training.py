import numpy as np

from experiments.tree_expert.s4_contracts import load_s4_contract
from experiments.tree_expert.s4_training import (
    corrected_probability,
    fit_residual_estimator,
    model_parameters,
    residual_target,
    route_rf,
)


class _FakeRegressor:
    best_iteration_ = 7

    def fit(self, x, y, **kwargs):
        self.mean = float(np.mean(y))
        self.kwargs = kwargs
        return self

    def predict(self, x):
        return np.full(len(x), self.mean)


def test_residual_target_is_recomputed_for_each_anchor() -> None:
    target = np.array([0.0, 1.0])
    np.testing.assert_allclose(residual_target(target, [0.2, 0.7]), [-0.2, 0.3])
    np.testing.assert_allclose(residual_target(target, [0.4, 0.6]), [-0.4, 0.4])


def test_model_factory_parameters_register_full_families() -> None:
    contract = load_s4_contract()
    assert model_parameters(contract, "catboost", gpu_id=0)["task_type"] == "GPU"
    assert model_parameters(contract, "xgboost", gpu_id=1)["device"] == "cuda:1"
    assert model_parameters(contract, "lightgbm", gpu_id=0)["num_threads"] >= 4


def test_rf_routing_is_current_row_only() -> None:
    actual = route_rf(["R", "F", "R"], [0.1, 0.2, 0.3], [0.7, 0.8, 0.9])
    np.testing.assert_allclose(actual, [0.1, 0.8, 0.3])


def test_fit_residual_estimator_accepts_weights_and_returns_prediction() -> None:
    result = fit_residual_estimator(
        family="catboost", train_matrix=np.ones((4, 2)), train_target=[-0.2, 0.1, 0.2, -0.1],
        valid_matrix=np.ones((2, 2)), gpu_id=0, seed=3407,
        sample_weight=np.array([1.0, 0.5, 1.0, 0.5]),
        model_factory=lambda family, parameters: _FakeRegressor(),
    )
    np.testing.assert_allclose(result.prediction, [0.0, 0.0])
    assert result.best_iteration == 7


def test_corrected_probability_clips_range() -> None:
    actual = corrected_probability([0.01, 0.99], [-1.0, 1.0], alpha=1.0)
    assert np.all((actual > 0.0) & (actual < 1.0))
