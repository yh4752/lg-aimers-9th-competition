"""Sealed lineage input for the temporal T2-C confirmation stage."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from .t1_review import _file_sha256, _json_bytes, _zip_info
from .t2b_input import VerifiedT2BInput, verify_t2b_input
from .uncertainty import SEGMENT_COLUMNS


class T2CInputError(ValueError):
    pass


@dataclass(frozen=True)
class VerifiedT2CInput:
    path: Path
    archive_sha256: str
    data_rows_sha256: str
    t2b_input_sha256: str
    t2b_handoff_sha256: str
    t2b_review_sha256: str
    decision_sha256: str
    candidate_id: str


CANDIDATE_ID = "s1_game_type_f_fallback_v1"
YEARS = (2022, 2023, 2024)
ALL_SEEDS = (3407, 42, 2026)
NEW_SEEDS = (42, 2026)
EXPECTED_T2C_MEMBERS = {
    "manifest.json",
    "decision.json",
    *(f"t1_anchor_{year}.csv" for year in YEARS),
    *(f"t1_multi_{year}.csv" for year in YEARS),
    *(f"s1_recent_s3407_{year}.csv" for year in YEARS),
}
_EXPECTED_COLUMNS = (
    "row_id",
    "valid_year",
    "target",
    "probability",
    "pitcher_id",
    "batter_id",
    *SEGMENT_COLUMNS,
)
_NON_PROBABILITY_COLUMNS = tuple(
    column for column in _EXPECTED_COLUMNS if column != "probability"
)
_T2B_COMPLETED = (
    "t2b__f__s1__va2022__s3407",
    "t2b__f__p3__va2022__s3407",
    "t2b__f__p2__va2022__s3407",
    "t2b__f__s1__va2023__s3407",
    "t2b__f__p3__va2023__s3407",
    "t2b__f__p2__va2023__s3407",
    "t2b__c__s1_p3__va2024__s3407",
    "t2b__c__s1_p2__va2024__s3407",
    "t2b__h__s1_p2__va2023__s3407",
    "t2b__h__s1_p2__va2022__s3407",
)


def prepare_t2c_input(
    t2b_input: str | Path,
    t2b_handoff: str | Path,
    output: str | Path,
) -> Path:
    """Verify T2-B parents and publish the minimal deterministic T2-C input."""

    bound = verify_t2b_input(t2b_input)
    handoff = _verify_t2b_handoff(Path(t2b_handoff), bound)
    archive_members: dict[str, bytes] = {}
    with ZipFile(bound.path) as source:
        for year in YEARS:
            for prefix in ("t1_anchor", "t1_multi"):
                name = f"{prefix}_{year}.csv"
                archive_members[name] = source.read(name)
        archive_members["s1_recent_s3407_2024.csv"] = source.read(
            "t2a_recent_s1_2024.csv"
        )
    with ZipFile(Path(t2b_handoff)) as outer:
        with ZipFile(io.BytesIO(outer.read("review.zip"))) as review:
            for year in (2022, 2023):
                source_name = (
                    f"jobs/t2b__f__s1__va{year}__s3407/predictions.csv"
                )
                frame = pd.read_csv(review.open(source_name))
                if "valid_year" not in frame:
                    frame.insert(1, "valid_year", year)
                frame = frame.loc[:, _EXPECTED_COLUMNS]
                archive_members[f"s1_recent_s3407_{year}.csv"] = frame.to_csv(
                    index=False
                ).encode("utf-8")
    decision = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "posthoc_candidate": True,
        "fallback_rule": {"column": "game_type", "operator": "eq", "value": "F"},
        "recent_features": ["base", "S1"],
        "multi_features": ["base"],
        "recent_weight": "0.50",
        "blend_mode": "logit",
        "all_seeds": list(ALL_SEEDS),
        "new_seeds": list(NEW_SEEDS),
        "valid_years": list(YEARS),
        "t2b_input_sha256": bound.archive_sha256,
        "t2b_handoff_sha256": handoff["handoff_sha256"],
        "t2b_review_sha256": handoff["review_sha256"],
        "data_rows_sha256": bound.data_rows_sha256,
    }
    archive_members["decision.json"] = _json_bytes(decision)
    _validate_reference_bytes(archive_members)
    records = {
        name: {"size_bytes": len(data), "sha256": sha256(data).hexdigest()}
        for name, data in sorted(archive_members.items())
    }
    manifest = {
        "schema_version": 1,
        "artifact_kind": "temporal_t2c_input_v1",
        "candidate_id": CANDIDATE_ID,
        "data_rows_sha256": bound.data_rows_sha256,
        "t2b_input_sha256": bound.archive_sha256,
        "t2b_handoff_sha256": handoff["handoff_sha256"],
        "t2b_review_sha256": handoff["review_sha256"],
        "decision_sha256": sha256(archive_members["decision.json"]).hexdigest(),
        "members": records,
    }
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with ZipFile(temporary, "w") as archive:
        archive.writestr(_zip_info("manifest.json"), _json_bytes(manifest))
        for name, data in sorted(archive_members.items()):
            archive.writestr(_zip_info(name), data)
    os.replace(temporary, destination)
    verify_t2c_input(destination)
    return destination


def verify_t2c_input(path: str | Path) -> VerifiedT2CInput:
    """Reject schema, hash, lineage, row alignment, or candidate drift."""

    source = Path(path)
    if not source.is_file() or source.is_symlink() or source.stat().st_size > 512_000_000:
        raise T2CInputError("T2-C input file is invalid")
    try:
        with ZipFile(source) as archive:
            names = archive.namelist()
            if set(names) != EXPECTED_T2C_MEMBERS or len(names) != len(
                EXPECTED_T2C_MEMBERS
            ):
                raise T2CInputError("T2-C input members differ")
            manifest_raw = archive.read("manifest.json")
            manifest = json.loads(manifest_raw)
            if (
                manifest_raw != _json_bytes(manifest)
                or manifest.get("schema_version") != 1
                or manifest.get("artifact_kind") != "temporal_t2c_input_v1"
                or manifest.get("candidate_id") != CANDIDATE_ID
            ):
                raise T2CInputError("T2-C input manifest differs")
            records = manifest.get("members")
            if type(records) is not dict or set(records) != EXPECTED_T2C_MEMBERS - {
                "manifest.json"
            }:
                raise T2CInputError("T2-C input member manifest differs")
            payloads = {}
            for name, record in records.items():
                data = archive.read(name)
                if (
                    type(record) is not dict
                    or set(record) != {"size_bytes", "sha256"}
                    or record["size_bytes"] != len(data)
                    or record["sha256"] != sha256(data).hexdigest()
                ):
                    raise T2CInputError(f"T2-C input member differs: {name}")
                payloads[name] = data
            decision_raw = payloads["decision.json"]
            decision = json.loads(decision_raw)
            _validate_decision(decision, manifest)
            if sha256(decision_raw).hexdigest() != manifest.get("decision_sha256"):
                raise T2CInputError("T2-C decision binding differs")
            _validate_reference_bytes(payloads)
            return VerifiedT2CInput(
                source.resolve(),
                _file_sha256(source),
                str(manifest["data_rows_sha256"]),
                str(manifest["t2b_input_sha256"]),
                str(manifest["t2b_handoff_sha256"]),
                str(manifest["t2b_review_sha256"]),
                str(manifest["decision_sha256"]),
                CANDIDATE_ID,
            )
    except T2CInputError:
        raise
    except (OSError, BadZipFile, KeyError, json.JSONDecodeError, ValueError) as error:
        raise T2CInputError("cannot verify T2-C input") from error


def load_t2c_references(
    verified: VerifiedT2CInput,
) -> tuple[
    Mapping[int, pd.DataFrame],
    Mapping[int, pd.DataFrame],
    Mapping[int, pd.DataFrame],
    Mapping[str, object],
]:
    """Return anchor, fixed multi, seed-3407 S1, and sealed decision."""

    if type(verified) is not VerifiedT2CInput:
        raise T2CInputError("verified T2-C input identity differs")
    with ZipFile(verified.path) as archive:
        anchors = {
            year: pd.read_csv(archive.open(f"t1_anchor_{year}.csv")) for year in YEARS
        }
        multi = {
            year: pd.read_csv(archive.open(f"t1_multi_{year}.csv")) for year in YEARS
        }
        recent = {
            year: pd.read_csv(archive.open(f"s1_recent_s3407_{year}.csv"))
            for year in YEARS
        }
        decision = json.loads(archive.read("decision.json"))
    return (
        MappingProxyType(anchors),
        MappingProxyType(multi),
        MappingProxyType(recent),
        MappingProxyType(decision),
    )


def _verify_t2b_handoff(path: Path, bound: VerifiedT2BInput) -> dict[str, str]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 768_000_000:
        raise T2CInputError("T2-B handoff file is invalid")
    try:
        with ZipFile(path) as archive:
            expected = {
                "handoff_manifest.json",
                "review.zip",
                "resume.zip",
                "run.log",
                "stage_summary.json",
            }
            if set(archive.namelist()) != expected or len(archive.infolist()) != 5:
                raise T2CInputError("T2-B handoff members differ")
            raw = archive.read("handoff_manifest.json")
            manifest = json.loads(raw)
            if (
                raw != _json_bytes(manifest)
                or manifest.get("artifact_kind") != "temporal_t2b_handoff_v1"
            ):
                raise T2CInputError("T2-B handoff manifest differs")
            records = manifest.get("members")
            if type(records) is not dict or set(records) != expected - {
                "handoff_manifest.json"
            }:
                raise T2CInputError("T2-B handoff member manifest differs")
            payloads = {}
            for name, record in records.items():
                data = archive.read(name)
                if (
                    type(record) is not dict
                    or set(record) != {"size_bytes", "sha256"}
                    or record["size_bytes"] != len(data)
                    or record["sha256"] != sha256(data).hexdigest()
                ):
                    raise T2CInputError(f"T2-B handoff member differs: {name}")
                payloads[name] = data
        review_manifest, evidence = _verify_t2b_review(payloads["review.zip"])
        stage = json.loads(payloads["stage_summary.json"])
        if (
            type(stage) is not dict
            or stage.get("stage") != "T2B"
            or stage.get("status") != "completed"
            or tuple(stage.get("completed", ())) != _T2B_COMPLETED
            or stage.get("pending") != []
            or stage.get("failed") != []
            or stage.get("data_rows_sha256") != bound.data_rows_sha256
            or stage.get("parent_sha256") != bound.t2a_decision_sha256
            or stage.get("evidence") != evidence
            or tuple(review_manifest.get("completed", ())) != _T2B_COMPLETED
            or review_manifest.get("pending") != []
            or review_manifest.get("failed") != []
            or review_manifest.get("parent_sha256") != bound.t2a_decision_sha256
        ):
            raise T2CInputError("T2-B completed evidence differs")
        decisions = evidence.get("decisions")
        if type(decisions) is not list or not any(
            type(item) is dict
            and item.get("candidate") == "S1"
            and item.get("status") == "rejected"
            for item in decisions
        ):
            raise T2CInputError("T2-B S1 decision differs")
        return {
            "handoff_sha256": _file_sha256(path),
            "review_sha256": sha256(payloads["review.zip"]).hexdigest(),
        }
    except T2CInputError:
        raise
    except (OSError, BadZipFile, KeyError, json.JSONDecodeError) as error:
        raise T2CInputError("cannot verify T2-B handoff") from error


def _verify_t2b_review(data: bytes) -> tuple[dict[str, object], dict[str, object]]:
    with ZipFile(io.BytesIO(data)) as archive:
        raw = archive.read("manifest.json")
        manifest = json.loads(raw)
        if (
            raw != _json_bytes(manifest)
            or manifest.get("artifact_kind") != "temporal_t2b_review_v1"
        ):
            raise T2CInputError("T2-B review manifest differs")
        declared = manifest.get("members")
        names = set(archive.namelist())
        if type(declared) is not dict or set(declared) != names - {"manifest.json"}:
            raise T2CInputError("T2-B review member manifest differs")
        for name, record in declared.items():
            payload = archive.read(name)
            if (
                type(record) is not dict
                or record.get("size_bytes") != len(payload)
                or record.get("sha256") != sha256(payload).hexdigest()
            ):
                raise T2CInputError(f"T2-B review member differs: {name}")
        state = json.loads(archive.read("t2b_evidence.json"))
        evidence = state.get("evidence")
        if type(evidence) is not dict:
            raise T2CInputError("T2-B review evidence differs")
        return manifest, evidence


def _validate_reference_bytes(payloads: Mapping[str, bytes]) -> None:
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    for year in YEARS:
        for prefix in ("t1_anchor", "t1_multi", "s1_recent_s3407"):
            name = f"{prefix}_{year}.csv"
            frames[(prefix, year)] = _read_oof_bytes(payloads[name], name, year)
        anchor = frames[("t1_anchor", year)]
        for prefix in ("t1_multi", "s1_recent_s3407"):
            other = frames[(prefix, year)]
            if not anchor.loc[:, _NON_PROBABILITY_COLUMNS].equals(
                other.loc[:, _NON_PROBABILITY_COLUMNS]
            ):
                raise T2CInputError(f"T2-C paired OOF evidence differs: {year}")


def _read_oof_bytes(data: bytes, name: str, year: int) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(data))
    probability = frame.get("probability")
    if (
        frame.empty
        or tuple(frame.columns) != _EXPECTED_COLUMNS
        or frame["valid_year"].ne(year).any()
        or frame["row_id"].duplicated().any()
        or probability is None
        or not np.isfinite(probability.to_numpy(dtype="float64")).all()
        or not probability.between(0, 1).all()
    ):
        raise T2CInputError(f"T2-C OOF evidence differs: {name}")
    return frame


def _validate_decision(
    decision: object, manifest: Mapping[str, object]
) -> None:
    if type(decision) is not dict:
        raise T2CInputError("T2-C decision differs")
    expected = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "posthoc_candidate": True,
        "fallback_rule": {"column": "game_type", "operator": "eq", "value": "F"},
        "recent_features": ["base", "S1"],
        "multi_features": ["base"],
        "recent_weight": "0.50",
        "blend_mode": "logit",
        "all_seeds": list(ALL_SEEDS),
        "new_seeds": list(NEW_SEEDS),
        "valid_years": list(YEARS),
        "t2b_input_sha256": manifest.get("t2b_input_sha256"),
        "t2b_handoff_sha256": manifest.get("t2b_handoff_sha256"),
        "t2b_review_sha256": manifest.get("t2b_review_sha256"),
        "data_rows_sha256": manifest.get("data_rows_sha256"),
    }
    if decision != expected or not all(
        _is_sha256(manifest.get(key))
        for key in (
            "data_rows_sha256",
            "t2b_input_sha256",
            "t2b_handoff_sha256",
            "t2b_review_sha256",
        )
    ):
        raise T2CInputError("T2-C decision differs")


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
