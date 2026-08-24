"""Verified lineage input for the temporal T2-B confirmation stage."""
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

from .t1_review import (
    _align_pair,
    _blend,
    _decision_payload,
    _file_sha256,
    _json_bytes,
    _zip_info,
    evaluate_t1_review,
)
from .uncertainty import SEGMENT_COLUMNS


class T2BInputError(ValueError):
    pass


@dataclass(frozen=True)
class VerifiedT2BInput:
    path: Path
    archive_sha256: str
    data_rows_sha256: str
    t1_decision_sha256: str
    t2a_decision_sha256: str
    t2a_review_sha256: str
    promoted: tuple[str, ...]


_PROMOTED = ("S1", "P3", "P2")
_VARIANTS = {
    "S1": "both_experts",
    "P3": "recent_only",
    "P2": "recent_only",
}
_YEARS = (2022, 2023, 2024)
_MEMBERS = {
    "manifest.json",
    "t1_decision.json",
    "t2a_decision.json",
    *(f"t1_anchor_{year}.csv" for year in _YEARS),
    *(f"t1_multi_{year}.csv" for year in _YEARS),
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


def prepare_t2b_input(
    t1_review: str | Path,
    t2a_handoff: str | Path,
    output: str | Path,
) -> Path:
    decision = evaluate_t1_review(t1_review)
    t2a = _verify_t2a_handoff(Path(t2a_handoff), decision.sha256)
    archive_members: dict[str, bytes] = {
        "t1_decision.json": _json_bytes(_decision_payload(decision)),
        "t2a_decision.json": _json_bytes(t2a),
    }
    with ZipFile(Path(t1_review)) as archive:
        decay_id = str(decision.decay).replace(".", "p")
        for year in _YEARS:
            recent_name = f"jobs/t1__recent__va{year}__s3407/predictions.csv"
            multi_name = (
                f"jobs/t1__multi_d{decay_id}__va{year}__s3407/predictions.csv"
            )
            recent = pd.read_csv(io.BytesIO(archive.read(recent_name)))
            multi = pd.read_csv(io.BytesIO(archive.read(multi_name)))
            aligned = _align_pair(recent, multi, year)
            probability = _blend(
                aligned["recent"].to_numpy(),
                aligned["multi"].to_numpy(),
                decision,
                anchor_rate=None,
            )
            base_columns = (
                "row_id",
                "valid_year",
                "target",
                "pitcher_id",
                "batter_id",
                *SEGMENT_COLUMNS,
            )
            anchor = aligned.loc[:, base_columns].copy(deep=True)
            anchor.insert(3, "probability", probability)
            fixed_multi = aligned.loc[:, base_columns].copy(deep=True)
            fixed_multi.insert(3, "probability", aligned["multi"].to_numpy())
            archive_members[f"t1_anchor_{year}.csv"] = anchor.to_csv(
                index=False
            ).encode("utf-8")
            archive_members[f"t1_multi_{year}.csv"] = fixed_multi.to_csv(
                index=False
            ).encode("utf-8")
    records = {
        name: {"size_bytes": len(data), "sha256": sha256(data).hexdigest()}
        for name, data in sorted(archive_members.items())
    }
    manifest = {
        "schema_version": 1,
        "artifact_kind": "temporal_t2b_input_v1",
        "data_rows_sha256": decision.data_rows_sha256,
        "t1_review_sha256": decision.review_sha256,
        "t1_decision_sha256": decision.sha256,
        "t2a_handoff_sha256": t2a["handoff_sha256"],
        "t2a_review_sha256": t2a["review_sha256"],
        "t2a_decision_sha256": sha256(archive_members["t2a_decision.json"]).hexdigest(),
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
    verify_t2b_input(destination)
    return destination


def verify_t2b_input(path: str | Path) -> VerifiedT2BInput:
    source = Path(path)
    if (
        not source.is_file()
        or source.is_symlink()
        or source.stat().st_size > 768_000_000
    ):
        raise T2BInputError("T2-B input file is invalid")
    try:
        with ZipFile(source) as archive:
            if set(archive.namelist()) != _MEMBERS or len(archive.infolist()) != len(
                _MEMBERS
            ):
                raise T2BInputError("T2-B input members differ")
            manifest_raw = archive.read("manifest.json")
            manifest = json.loads(manifest_raw)
            if (
                manifest_raw != _json_bytes(manifest)
                or manifest.get("artifact_kind") != "temporal_t2b_input_v1"
            ):
                raise T2BInputError("T2-B input manifest differs")
            records = manifest.get("members")
            if type(records) is not dict or set(records) != _MEMBERS - {
                "manifest.json"
            }:
                raise T2BInputError("T2-B input member manifest differs")
            for name, record in records.items():
                data = archive.read(name)
                if (
                    type(record) is not dict
                    or set(record) != {"size_bytes", "sha256"}
                    or record["size_bytes"] != len(data)
                    or record["sha256"] != sha256(data).hexdigest()
                ):
                    raise T2BInputError(f"T2-B input member differs: {name}")
            t1_raw = archive.read("t1_decision.json")
            t1 = json.loads(t1_raw)
            t2a_raw = archive.read("t2a_decision.json")
            t2a = json.loads(t2a_raw)
            if (
                sha256(t1_raw).hexdigest() != manifest.get("t1_decision_sha256")
                or t1.get("status") != "champion"
                or t1.get("decay") != "0.55"
                or t1.get("recent_weight") != "0.50"
                or t1.get("mode") != "logit"
                or t1.get("beta") != "0"
                or t1.get("data_rows_sha256") != manifest.get("data_rows_sha256")
                or t1.get("review_sha256") != manifest.get("t1_review_sha256")
            ):
                raise T2BInputError("T2-B fixed T1 decision differs")
            promoted = _promoted_tuple(t2a.get("promoted"))
            if (
                sha256(t2a_raw).hexdigest() != manifest.get("t2a_decision_sha256")
                or t2a.get("status") != "completed"
                or t2a.get("data_rows_sha256") != manifest.get("data_rows_sha256")
                or t2a.get("t1_decision_sha256")
                != manifest.get("t1_decision_sha256")
                or t2a.get("handoff_sha256") != manifest.get("t2a_handoff_sha256")
                or t2a.get("review_sha256") != manifest.get("t2a_review_sha256")
                or promoted != _PROMOTED
            ):
                raise T2BInputError("T2-B T2-A decision differs")
            recent = t2a.get("recent_evidence")
            if type(recent) is not dict or set(recent) != set(_PROMOTED):
                raise T2BInputError("T2-B latest-fold evidence differs")
            for bundle in _PROMOTED:
                item = recent[bundle]
                if type(item) is not dict or item.get("status") != "completed":
                    raise T2BInputError("T2-B latest-fold evidence differs")
            for year in _YEARS:
                first = _read_oof(archive, f"t1_anchor_{year}.csv", year)
                second = _read_oof(archive, f"t1_multi_{year}.csv", year)
                columns = tuple(
                    column for column in _EXPECTED_COLUMNS if column != "probability"
                )
                if not first.loc[:, columns].equals(second.loc[:, columns]):
                    raise T2BInputError(f"T2-B paired OOF evidence differs: {year}")
            return VerifiedT2BInput(
                source.resolve(),
                _file_sha256(source),
                str(manifest["data_rows_sha256"]),
                str(manifest["t1_decision_sha256"]),
                str(manifest["t2a_decision_sha256"]),
                str(manifest["t2a_review_sha256"]),
                promoted,
            )
    except T2BInputError:
        raise
    except (OSError, BadZipFile, KeyError, json.JSONDecodeError) as error:
        raise T2BInputError("cannot verify T2-B input") from error


def load_t2b_references(
    verified: VerifiedT2BInput,
) -> tuple[Mapping[int, pd.DataFrame], Mapping[int, pd.DataFrame], Mapping[str, object]]:
    if type(verified) is not VerifiedT2BInput:
        raise T2BInputError("verified T2-B input identity differs")
    with ZipFile(verified.path) as archive:
        anchors = {
            year: pd.read_csv(archive.open(f"t1_anchor_{year}.csv"))
            for year in _YEARS
        }
        multi = {
            year: pd.read_csv(archive.open(f"t1_multi_{year}.csv"))
            for year in _YEARS
        }
        t2a = json.loads(archive.read("t2a_decision.json"))
    return MappingProxyType(anchors), MappingProxyType(multi), MappingProxyType(t2a)


def _verify_t2a_handoff(path: Path, t1_decision_sha256: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 512_000_000:
        raise T2BInputError("T2-A handoff file is invalid")
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
                raise T2BInputError("T2-A handoff members differ")
            raw = archive.read("handoff_manifest.json")
            manifest = json.loads(raw)
            if (
                raw != _json_bytes(manifest)
                or manifest.get("artifact_kind") != "temporal_t2a_handoff_v1"
            ):
                raise T2BInputError("T2-A handoff manifest differs")
            records = manifest.get("members")
            if type(records) is not dict or set(records) != expected - {
                "handoff_manifest.json"
            }:
                raise T2BInputError("T2-A handoff member manifest differs")
            payloads = {}
            for name, record in records.items():
                data = archive.read(name)
                if (
                    type(record) is not dict
                    or set(record) != {"size_bytes", "sha256"}
                    or record["size_bytes"] != len(data)
                    or record["sha256"] != sha256(data).hexdigest()
                ):
                    raise T2BInputError(f"T2-A handoff member differs: {name}")
                payloads[name] = data
        review_manifest, evidence = _verify_t2a_review(payloads["review.zip"])
        stage = json.loads(payloads["stage_summary.json"])
        if (
            type(stage) is not dict
            or stage.get("stage") != "T2A"
            or stage.get("status") != "completed"
            or stage.get("pending") != []
            or stage.get("failed") != []
            or stage.get("t1_decision_sha256") != t1_decision_sha256
            or review_manifest.get("t1_decision_sha256") != t1_decision_sha256
            or stage.get("evidence") != evidence
        ):
            raise T2BInputError("T2-A completed evidence differs")
        promoted = _promoted_tuple(evidence.get("promoted"))
        recent = evidence.get("phase_r")
        if promoted != _PROMOTED or type(recent) is not dict:
            raise T2BInputError("T2-A promoted evidence differs")
        latest = {bundle: recent.get(bundle) for bundle in _PROMOTED}
        if any(type(item) is not dict or item.get("status") != "completed" for item in latest.values()):
            raise T2BInputError("T2-A latest-fold evidence differs")
        return {
            "schema_version": 1,
            "status": "completed",
            "handoff_sha256": _file_sha256(path),
            "review_sha256": sha256(payloads["review.zip"]).hexdigest(),
            "stage_summary_sha256": sha256(payloads["stage_summary.json"]).hexdigest(),
            "data_rows_sha256": stage.get("data_rows_sha256"),
            "t1_decision_sha256": t1_decision_sha256,
            "promoted": list(promoted),
            "variants": dict(_VARIANTS),
            "recent_evidence": latest,
        }
    except T2BInputError:
        raise
    except (OSError, BadZipFile, KeyError, json.JSONDecodeError) as error:
        raise T2BInputError("cannot verify T2-A handoff") from error


def _verify_t2a_review(data: bytes) -> tuple[dict[str, object], dict[str, object]]:
    with ZipFile(io.BytesIO(data)) as archive:
        raw = archive.read("manifest.json")
        manifest = json.loads(raw)
        if (
            raw != _json_bytes(manifest)
            or manifest.get("artifact_kind") != "temporal_t2a_review_v1"
        ):
            raise T2BInputError("T2-A review manifest differs")
        declared = manifest.get("members")
        names = set(archive.namelist())
        if type(declared) is not dict or set(declared) != names - {"manifest.json"}:
            raise T2BInputError("T2-A review member manifest differs")
        for name, record in declared.items():
            payload = archive.read(name)
            if (
                type(record) is not dict
                or record.get("size_bytes") != len(payload)
                or record.get("sha256") != sha256(payload).hexdigest()
            ):
                raise T2BInputError(f"T2-A review member differs: {name}")
        state = json.loads(archive.read("t2a_evidence.json"))
        evidence = state.get("evidence")
        if type(evidence) is not dict:
            raise T2BInputError("T2-A review evidence differs")
        return manifest, evidence


def _promoted_tuple(value: object) -> tuple[str, ...]:
    if type(value) is not list or len(value) != 3:
        raise T2BInputError("T2-A promoted evidence differs")
    if all(type(item) is str for item in value):
        promoted = tuple(value)
    elif all(type(item) is dict for item in value):
        promoted = tuple(str(item.get("bundle")) for item in value)
        variants = {str(item.get("bundle")): item.get("variant") for item in value}
        if variants != _VARIANTS:
            raise T2BInputError("T2-A promoted variants differ")
    else:
        raise T2BInputError("T2-A promoted evidence differs")
    if promoted != _PROMOTED:
        raise T2BInputError("T2-A promoted order differs")
    return promoted


def _read_oof(archive: ZipFile, name: str, year: int) -> pd.DataFrame:
    frame = pd.read_csv(archive.open(name))
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
        raise T2BInputError(f"T2-B OOF evidence differs: {name}")
    return frame

