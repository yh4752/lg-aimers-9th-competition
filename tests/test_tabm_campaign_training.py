from __future__ import annotations

from pathlib import Path

import pytest

from experiments.tabm_campaign.training import (
    CheckpointBindingError,
    EarlyStopper,
    EpochResult,
    RestartableRequest,
    preflight,
    train_restartable,
)


class _Clock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


class _Backend:
    def __init__(self, curve: list[float]) -> None:
        self.curve = curve
        self.epochs: list[int] = []

    def run_epoch(self, epoch: int) -> EpochResult:
        self.epochs.append(epoch)
        return EpochResult(metric=self.curve[epoch], checkpoint_bytes=f"epoch-{epoch}".encode())


def _request(**changes: object) -> RestartableRequest:
    values = dict(
        candidate_id="candidate",
        max_epochs=8,
        min_epochs=3,
        patience=2,
        absolute_deadline=100.0,
        config_sha256="1" * 64,
        source_sha256="2" * 64,
        cache_sha256="3" * 64,
    )
    values.update(changes)
    return RestartableRequest(**values)


def test_deadline_after_checkpoint_returns_inconclusive_not_pending(tmp_path: Path) -> None:
    backend = _Backend([0.25, 0.24, 0.23])
    result = train_restartable(
        _request(),
        tmp_path,
        backend=backend,
        clock=_Clock([0.0, 10.0, 20.0, 101.0]),
    )
    assert result.status == "inconclusive"
    assert result.completed_epochs == 3
    assert result.checkpoint_sha256


def test_patience_cannot_stop_before_minimum_epochs() -> None:
    stop = EarlyStopper(min_epochs=3, patience=1)
    assert [stop.update(value) for value in (0.2, 0.3)] == [False, False]
    assert stop.update(0.4) is True


def test_preflight_failure_rejects_only_candidate() -> None:
    def oom() -> object:
        raise RuntimeError("CUDA out of memory")

    result = preflight("candidate", oom)
    assert result.status == "failed"
    assert result.failure_type == "cuda_oom"


def test_resume_refuses_different_cache_binding(tmp_path: Path) -> None:
    train_restartable(
        _request(max_epochs=1, min_epochs=1, absolute_deadline=None),
        tmp_path,
        backend=_Backend([0.25]),
        clock=lambda: 0.0,
    )
    with pytest.raises(CheckpointBindingError, match="binding"):
        train_restartable(
            _request(max_epochs=2, min_epochs=1, absolute_deadline=None, cache_sha256="4" * 64),
            tmp_path,
            backend=_Backend([0.25, 0.24]),
            clock=lambda: 0.0,
        )
