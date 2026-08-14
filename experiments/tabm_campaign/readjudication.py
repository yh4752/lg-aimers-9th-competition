"""Deterministically correct Version B selection without retraining models."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from zipfile import ZipFile

from .artifacts import (
    BundlePaths,
    StageEvidence,
    verify_review_bundle,
    write_stage_bundles,
)
from .decisions import choose_temporal_champion
from .runner import VERSION_B_REFERENCE_CANDIDATE_ID


class ReadjudicationError(ValueError):
    """Raised when supplied review evidence cannot support a safe correction."""


@dataclass(frozen=True)
class ReadjudicationResult:
    bundles: BundlePaths
    champion_id: str
    source_review_manifest_sha256: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _job_id(candidate_id: str, fold: str) -> str:
    if not candidate_id.startswith("a__"):
        raise ReadjudicationError(f"invalid Version A candidate id: {candidate_id}")
    return f"b__{candidate_id[3:]}__{fold}"


def readjudicate_version_b(
    review_bundle: str | Path,
    output_dir: str | Path,
) -> ReadjudicationResult:
    """Rebuild Version B bundles after applying the declared temporal reference."""

    verified = verify_review_bundle(review_bundle)
    if verified.version != "B" or verified.prior_manifest_sha256 is None:
        raise ReadjudicationError("readjudication requires a hash-bound Version B review")
    required = {"stage_state.json", "metrics/job_results.json", "logs/stage.log"}
    if not required.issubset(verified.member_sha256):
        raise ReadjudicationError("Version B review is missing required evidence")

    with ZipFile(verified.path, "r") as archive:
        members = {
            name: archive.read(name) for name in verified.member_sha256
        }
    try:
        state = json.loads(members["stage_state.json"])
        metrics = json.loads(members["metrics/job_results.json"])
    except json.JSONDecodeError as error:
        raise ReadjudicationError(f"Version B evidence JSON is invalid: {error}") from error
    if (
        not isinstance(state, dict)
        or state.get("version") != "B"
        or state.get("stage_complete") is not True
        or not isinstance(metrics, list)
        or state.get("results") != metrics
    ):
        raise ReadjudicationError("Version B state and metric evidence are inconsistent")
    survivors = state.get("survivors")
    if not isinstance(survivors, list) or not all(isinstance(item, dict) for item in survivors):
        raise ReadjudicationError("Version B survivor evidence is invalid")
    candidate_by_id = {
        str(item.get("candidate_id")): item for item in survivors
    }
    if VERSION_B_REFERENCE_CANDIDATE_ID not in candidate_by_id:
        raise ReadjudicationError("declared Version B reference is missing")

    result_by_id: dict[str, dict[str, object]] = {}
    for item in metrics:
        if not isinstance(item, dict) or item.get("status") != "completed":
            raise ReadjudicationError("readjudication requires completed Version B results")
        candidate_id = str(item.get("candidate_id"))
        brier = item.get("brier")
        if candidate_id in result_by_id or isinstance(brier, bool) or not isinstance(brier, (int, float)) or not math.isfinite(float(brier)):
            raise ReadjudicationError("Version B result identity or Brier is invalid")
        result_by_id[candidate_id] = item

    fold_briers: dict[str, tuple[float, float]] = {}
    for candidate_id in candidate_by_id:
        primary = result_by_id.get(_job_id(candidate_id, "tr2023__va2024"))
        older = result_by_id.get(_job_id(candidate_id, "tr2022__va2023"))
        if primary is not None and older is not None:
            fold_briers[candidate_id] = (
                float(primary["brier"]),
                float(older["brier"]),
            )
    champion_id, selection_delta = choose_temporal_champion(
        fold_briers,
        reference_id=VERSION_B_REFERENCE_CANDIDATE_ID,
    )
    original_champion = state.get("champion")
    original_champion_id = (
        str(original_champion.get("candidate_id"))
        if isinstance(original_champion, dict)
        else None
    )
    corrected_state = dict(state)
    corrected_state.update(
        {
            "champion": candidate_by_id[champion_id],
            "selection_delta": selection_delta,
            "champion_fold_briers": {
                "2024": fold_briers[champion_id][0],
                "2023": fold_briers[champion_id][1],
            },
            "adjudication": {
                "schema_version": 1,
                "reason": "declared_temporal_reference_bug_fix",
                "reference_candidate_id": VERSION_B_REFERENCE_CANDIDATE_ID,
                "original_champion_candidate_id": original_champion_id,
                "source_review_manifest_sha256": verified.manifest_sha256,
                "retraining_performed": False,
            },
        }
    )
    state_bytes = _canonical_json(corrected_state)
    members["stage_state.json"] = state_bytes
    members["logs/stage.log"] = (
        "version=B readjudicated=true retraining=false "
        f"reference={VERSION_B_REFERENCE_CANDIDATE_ID} champion={champion_id}\n"
    ).encode("utf-8")
    bundles = write_stage_bundles(
        output_dir,
        StageEvidence(
            "B",
            verified.campaign_config_sha256,
            verified.prior_manifest_sha256,
            members,
            {"stage_state.json": state_bytes},
        ),
    )
    return ReadjudicationResult(bundles, champion_id, verified.manifest_sha256)
