from __future__ import annotations

from collections import OrderedDict
import threading

from experiments.failure_regime_e3.runtime import ProductionE3Runtime


def _runtime() -> ProductionE3Runtime:
    runtime = ProductionE3Runtime.__new__(ProductionE3Runtime)
    runtime._folds = OrderedDict()
    runtime._fold_lock = threading.Lock()
    runtime._full_lock = threading.Lock()
    runtime._full_cache = object()
    return runtime


def test_large_fold_cache_keeps_only_two_most_recent_years() -> None:
    runtime = _runtime()

    runtime._store_fold(2022, "fold-2022")
    runtime._store_fold(2023, "fold-2023")
    runtime._store_fold(2024, "fold-2024")

    assert list(runtime._folds) == [2023, 2024]


def test_old_fold_is_evicted_before_building_a_third_large_fold() -> None:
    runtime = _runtime()
    runtime._store_fold(2022, "fold-2022")
    runtime._store_fold(2023, "fold-2023")

    runtime._prepare_fold_slot(2024)

    assert list(runtime._folds) == [2023]


def test_training_caches_can_be_released_before_model_audit() -> None:
    runtime = _runtime()
    runtime._store_fold(2024, "fold-2024")

    runtime.release_training_cache()

    assert not runtime._folds
    assert runtime._full_cache is None
