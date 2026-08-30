from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

from .inputs import canonical_json


class FinalStateError(ValueError):
    pass


@dataclass(frozen=True)
class CampaignState:
    phase: str
    status: str
    completed_jobs: tuple[str, ...]
    decision_status: str | None
    candidate_id: str | None


def save_state(path: Path, state: CampaignState) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = {**asdict(state), "completed_jobs": list(state.completed_jobs)}
    with temporary.open("wb") as handle:
        handle.write(canonical_json(payload))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def load_state(path: Path) -> CampaignState:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FinalStateError("campaign state is unreadable") from error
    expected = {"phase", "status", "completed_jobs", "decision_status", "candidate_id"}
    if type(payload) is not dict or set(payload) != expected:
        raise FinalStateError("campaign state keys differ")
    if (
        type(payload["phase"]) is not str or type(payload["status"]) is not str
        or type(payload["completed_jobs"]) is not list
        or any(type(item) is not str for item in payload["completed_jobs"])
        or payload["decision_status"] is not None and type(payload["decision_status"]) is not str
        or payload["candidate_id"] is not None and type(payload["candidate_id"]) is not str
    ):
        raise FinalStateError("campaign state values differ")
    return CampaignState(
        phase=payload["phase"], status=payload["status"],
        completed_jobs=tuple(payload["completed_jobs"]),
        decision_status=payload["decision_status"], candidate_id=payload["candidate_id"],
    )
