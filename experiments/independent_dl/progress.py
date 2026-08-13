"""Append-only progress events for user-owned independent-DL runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Callable
import uuid


class ProgressTimeoutError(RuntimeError):
    """Raised when a candidate produces no substantive work for ten minutes."""


@dataclass
class _StreamState:
    started_at: float
    last_emit_at: float
    last_progress_at: float
    last_completed_rows: int = 0
    stall_warning_at: float | None = None


_EVENTS = frozenset(
    {
        "CANDIDATE_RUNTIME_READY",
        "TRAINING_PROGRESS",
        "VALIDATION_PROGRESS",
        "EPOCH_CHECKPOINTED",
        "CANDIDATE_COMPLETED",
        "CANDIDATE_FAILED",
        "TRAINING_TIME_BUDGET_REACHED",
        "TRAINING_STALL_WARNING",
        "TABR_CONTEXT_ENCODING_PROGRESS",
        "TABR_INDEX_READY",
        "TABR_SEARCH_PROGRESS",
        "TABR_CONTEXTS_FROZEN",
    }
)
_COUNT_FIELDS = frozenset(
    {
        "completed_rows",
        "total_rows",
        "allocated_bytes",
        "reserved_bytes",
        "peak_bytes",
    }
)


def _json_value(value: object, *, field: str) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        if field in _COUNT_FIELDS and value < 0:
            raise ValueError(f"{field} must not be negative")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
        if field in _COUNT_FIELDS and value < 0:
            raise ValueError(f"{field} must not be negative")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item, field=str(key))
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item, field=field) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_value(item(), field=field)
    raise ValueError(f"{field} is not JSON serializable")


def _nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


class ProgressReporter:
    """Write the same strict JSON event to stdout and a candidate-local file."""

    def __init__(
        self,
        candidate_id: str,
        family: str,
        fold: str,
        output_dir: str | Path,
        *,
        clock: Callable[[], float] = time.monotonic,
        process_id: int | None = None,
        run_id: str | None = None,
    ) -> None:
        if not candidate_id or not family or not fold:
            raise ValueError("candidate_id, family, and fold must not be empty")
        self.candidate_id = candidate_id
        self.family = family
        self.fold = fold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "progress.jsonl"
        self._clock = clock
        self.process_id = int(os.getpid() if process_id is None else process_id)
        self.run_id = run_id or uuid.uuid4().hex
        self._started_at = float(clock())
        self._streams = {
            "default": _StreamState(
                self._started_at, self._started_at, self._started_at
            )
        }

    def _stream_state(self, stream: str) -> _StreamState:
        if not stream:
            raise ValueError("stream must not be empty")
        state = self._streams.get(stream)
        if state is None:
            now = float(self._clock())
            state = _StreamState(now, now, now)
            self._streams[stream] = state
        return state

    def emit(self, event: str, **fields: object) -> dict[str, object]:
        if event not in _EVENTS:
            raise ValueError(f"unknown progress event: {event}")
        payload: dict[str, object] = {
            "timestamp_utc": datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "family": self.family,
            "fold": self.fold,
            "process_id": self.process_id,
            "event": event,
        }
        payload.update(
            {key: _json_value(value, field=key) for key, value in fields.items()}
        )
        line = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        print(line, flush=True)
        return payload

    def progress(
        self,
        event: str,
        *,
        completed_rows: int,
        total_rows: int,
        started_at: float,
        gpu: Mapping[str, int],
        stream: str = "default",
        **fields: object,
    ) -> dict[str, object]:
        completed = _nonnegative_integer(completed_rows, "completed_rows")
        total = _nonnegative_integer(total_rows, "total_rows")
        if completed > total:
            raise ValueError("completed_rows must not exceed total_rows")
        now = float(self._clock())
        elapsed = now - float(started_at)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("elapsed_seconds must be finite and positive")
        safe_gpu = {
            name: _nonnegative_integer(gpu.get(name), name)
            for name in ("allocated_bytes", "reserved_bytes", "peak_bytes")
        }
        rows_per_second = completed / elapsed
        eta_seconds = (
            (total - completed) / rows_per_second if rows_per_second > 0 else None
        )
        state = self._stream_state(stream)
        if completed > state.last_completed_rows:
            state.last_completed_rows = completed
            state.last_progress_at = now
            state.stall_warning_at = None
        payload = self.emit(
            event,
            completed_rows=completed,
            total_rows=total,
            elapsed_seconds=elapsed,
            rows_per_second=rows_per_second,
            eta_seconds=eta_seconds,
            gpu=safe_gpu,
            stream=stream,
            **fields,
        )
        state.last_emit_at = float(self._clock())
        return payload

    def should_emit(
        self,
        *,
        completed_rows: int,
        now: float | None = None,
        stream: str = "default",
    ) -> bool:
        completed = _nonnegative_integer(completed_rows, "completed_rows")
        checked_at = float(self._clock() if now is None else now)
        state = self._stream_state(stream)
        first_substantive_progress = (
            state.last_completed_rows == 0 and completed > 0
        )
        if (
            state.last_completed_rows == 0
            and checked_at - state.started_at >= 600.0
        ):
            raise ProgressTimeoutError(
                "no substantive progress was recorded within 600 seconds"
            )
        stalled_seconds = checked_at - state.last_progress_at
        emitted_stall_warning = False
        if (
            completed <= state.last_completed_rows
            and state.last_completed_rows > 0
            and stalled_seconds >= 300.0
            and state.stall_warning_at is None
        ):
            self.emit(
                "TRAINING_STALL_WARNING",
                completed_rows=state.last_completed_rows,
                stalled_seconds=stalled_seconds,
                stream=stream,
            )
            state.stall_warning_at = checked_at
            state.last_emit_at = checked_at
            emitted_stall_warning = True
        return (
            first_substantive_progress
            or emitted_stall_warning
            or checked_at - state.last_emit_at >= 60.0
        )
