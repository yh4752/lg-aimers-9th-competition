from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.independent_dl.progress import (
    ProgressReporter,
    ProgressTimeoutError,
)


class _Clock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _reporter(tmp_path: Path, clock: _Clock) -> ProgressReporter:
    return ProgressReporter(
        candidate_id="tabm__raw_typed__p1__s42",
        family="tabm",
        fold="2023->2024",
        output_dir=tmp_path,
        clock=clock,
        process_id=4321,
        run_id="run-test",
    )


def test_progress_is_identical_on_stdout_and_append_only_jsonl(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clock = _Clock()
    reporter = _reporter(tmp_path, clock)

    first = reporter.progress(
        "TRAINING_PROGRESS",
        completed_rows=4096,
        total_rows=1_221_585,
        started_at=92.0,
        gpu={"allocated_bytes": 10, "reserved_bytes": 20, "peak_bytes": 30},
        epoch=0,
        batch=8,
        loss=0.61,
    )
    clock.value = 104.0
    second = reporter.emit("EPOCH_CHECKPOINTED", epoch=0, brier=0.24)

    stdout_lines = capsys.readouterr().out.strip().splitlines()
    file_lines = (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    assert stdout_lines == file_lines
    assert [json.loads(line) for line in file_lines] == [first, second]
    assert first["run_id"] == "run-test"
    assert first["candidate_id"] == "tabm__raw_typed__p1__s42"
    assert first["family"] == "tabm"
    assert first["fold"] == "2023->2024"
    assert first["process_id"] == 4321
    assert str(first["timestamp_utc"]).endswith("Z")
    assert first["rows_per_second"] == 512.0
    assert first["eta_seconds"] == pytest.approx((1_221_585 - 4096) / 512.0)
    assert first["gpu"] == {
        "allocated_bytes": 10,
        "reserved_bytes": 20,
        "peak_bytes": 30,
    }


def test_progress_file_is_preserved_across_reporter_restarts(tmp_path: Path) -> None:
    clock = _Clock()
    _reporter(tmp_path, clock).emit("CANDIDATE_RUNTIME_READY", device="cuda:0")
    _reporter(tmp_path, clock).emit("CANDIDATE_COMPLETED", brier=0.24)

    events = [
        json.loads(line)
        for line in (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "CANDIDATE_RUNTIME_READY",
        "CANDIDATE_COMPLETED",
    ]


def test_training_and_validation_progress_use_independent_streams(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    reporter = _reporter(tmp_path, clock)
    gpu = {"allocated_bytes": 0, "reserved_bytes": 0, "peak_bytes": 0}
    reporter.progress(
        "TRAINING_PROGRESS",
        completed_rows=1_000,
        total_rows=1_000,
        started_at=99.0,
        gpu=gpu,
        stream="training_epoch_0",
    )

    assert reporter.should_emit(completed_rows=1, stream="validation_epoch_0") is True
    reporter.progress(
        "VALIDATION_PROGRESS",
        completed_rows=1,
        total_rows=100,
        started_at=99.0,
        gpu=gpu,
        stream="validation_epoch_0",
    )
    assert reporter.should_emit(completed_rows=1, stream="validation_epoch_0") is False


def test_emit_interval_stall_warning_and_initial_progress_timeout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clock = _Clock()
    reporter = _reporter(tmp_path, clock)

    assert reporter.should_emit(completed_rows=0) is False
    assert reporter.should_emit(completed_rows=1) is True
    clock.value += 60.0
    assert reporter.should_emit(completed_rows=0) is True

    reporter.progress(
        "TRAINING_PROGRESS",
        completed_rows=100,
        total_rows=1_000,
        started_at=100.0,
        gpu={"allocated_bytes": 0, "reserved_bytes": 0, "peak_bytes": 0},
    )
    clock.value += 299.0
    assert reporter.should_emit(completed_rows=100) is True
    assert "TRAINING_STALL_WARNING" not in capsys.readouterr().out

    clock.value += 1.0
    assert reporter.should_emit(completed_rows=100) is True
    events = [
        json.loads(line)
        for line in (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["event"] == "TRAINING_STALL_WARNING"
    assert events[-1]["completed_rows"] == 100
    assert events[-1]["stalled_seconds"] == 300.0

    timeout_clock = _Clock()
    timeout_reporter = ProgressReporter(
        candidate_id="tabr__raw_typed__p1__s42",
        family="tabr",
        fold="2023->2024",
        output_dir=tmp_path / "timeout",
        clock=timeout_clock,
        process_id=4321,
        run_id="timeout-run",
    )
    timeout_clock.value += 600.0
    with pytest.raises(ProgressTimeoutError, match="no substantive progress"):
        timeout_reporter.should_emit(completed_rows=0)
    with pytest.raises(ProgressTimeoutError, match="no substantive progress"):
        timeout_reporter.should_emit(completed_rows=1)


@pytest.mark.parametrize(
    ("method", "match"),
    [
        (lambda reporter: reporter.emit("UNKNOWN_EVENT"), "unknown progress event"),
        (
            lambda reporter: reporter.progress(
                "TRAINING_PROGRESS",
                completed_rows=True,
                total_rows=10,
                started_at=1.0,
                gpu={"allocated_bytes": 0, "reserved_bytes": 0, "peak_bytes": 0},
            ),
            "completed_rows",
        ),
        (
            lambda reporter: reporter.progress(
                "TRAINING_PROGRESS",
                completed_rows=1,
                total_rows=10,
                started_at=1.0,
                gpu={"allocated_bytes": 0, "reserved_bytes": float("nan"), "peak_bytes": 0},
            ),
            "finite",
        ),
        (
            lambda reporter: reporter.progress(
                "TRAINING_PROGRESS",
                completed_rows=-1,
                total_rows=10,
                started_at=1.0,
                gpu={"allocated_bytes": 0, "reserved_bytes": 0, "peak_bytes": 0},
            ),
            "completed_rows",
        ),
    ],
)
def test_reporter_rejects_unsafe_events_and_numbers(
    tmp_path: Path, method: object, match: str
) -> None:
    reporter = _reporter(tmp_path, _Clock())

    with pytest.raises(ValueError, match=match):
        method(reporter)
