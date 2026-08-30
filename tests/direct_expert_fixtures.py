from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd

from tests.test_tree_expert_features import _history, _rows


_TIME = (2026, 1, 1, 0, 0, 0)


def make_train_rows() -> pd.DataFrame:
    return _rows().copy(deep=True)


def make_history_rows() -> pd.DataFrame:
    return _history().copy(deep=True)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _info(name: str) -> ZipInfo:
    info = ZipInfo(name, _TIME)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _archive(path: Path, members: dict[str, bytes]) -> Path:
    with ZipFile(path, "w") as archive:
        for name, payload in sorted(members.items()):
            archive.writestr(_info(name), payload)
    return path


def _manifest(kind: str, members: dict[str, bytes], **extra: object) -> bytes:
    payload = {
        "schema_version": 1,
        "artifact_kind": kind,
        "members": {
            name: {"sha256": sha256(value).hexdigest(), "size": len(value)}
            for name, value in sorted(members.items())
        },
        **extra,
    }
    return _json(payload)


def make_s4_handoff(root: Path) -> tuple[Path, str]:
    logical_e2 = "4" * 64
    oof = {}
    for year in (2021, 2022, 2023, 2024):
        oof[f"anchors/e2/{year}.csv"] = (
            "row_id,game_type,pitcher_id,batter_id,pitcher_hand,batter_hand,balls_before,"
            "strikes_before,outs_before,base_state,target,p_anchor,oof_year\n"
            f"r{year}a,R,10,20,1,0,0,1,0,___,1,0.6,{year}\n"
            f"r{year}b,F,11,21,0,1,2,2,1,1__,0,0.4,{year}\n"
        ).encode()
    bindings = {
        "code_sha256": "1" * 64,
        "contract_sha256": "2" * 64,
        "e2_handoff_sha256": logical_e2,
        "history_sha256": "3" * 64,
        "input_manifest_sha256": "5" * 64,
        "train_sha256": "6" * 64,
    }
    resume_members = dict(oof)
    resume_members["manifest.json"] = _manifest(
        "tree_s4_resume_v1", oof, bindings=bindings
    )
    resume = _archive(root / "resume.zip", resume_members)
    outer_payloads = {
        "campaign.log": b"fixture\n",
        "resume.zip": resume.read_bytes(),
        "review.zip": _archive(root / "review.zip", {"review.txt": b"fixture"}).read_bytes(),
    }
    outer_members = dict(outer_payloads)
    outer_members["manifest.json"] = _manifest(
        "tree_s4_handoff_v1",
        outer_payloads,
        bindings=bindings,
        delivery=False,
        status="completed_no_candidate",
        submission_package=False,
    )
    path = _archive(root / "s4.zip", outer_members)
    return path, logical_e2


def make_e2_submission(root: Path) -> tuple[Path, Path]:
    members = {
        "script.py": b"def predict(test):\n    return [0.5] * len(test)\n",
        "requirements.txt": b"catboost==1.2.10\n",
        "model/frozen_state/feature_state.json": b"{}",
        "model/frozen_state/s1_batter.csv": b"batter_id\n1\n",
        "model/frozen_state/s1_pitcher.csv": b"pitcher_id\n1\n",
        "model/models/catboost_seed_42.cbm": b"seed42",
        "model/models/catboost_seed_2026.cbm": b"seed2026",
        "model/models/catboost_seed_3407.cbm": b"seed3407",
    }
    archive = _archive(root / "e2.zip", members)
    digest = sha256(archive.read_bytes()).hexdigest()
    receipt = root / "submission_receipt.json"
    receipt.write_bytes(
        _json(
            {
                "schema_version": 1,
                "status": "packaged",
                "candidate_id": "c1_anchor_residual",
                "adapter_id": "tree_expert_e2_c1_catboost_v1",
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": digest,
                "policy_version": "dacon-236743-2026-08-15",
                "identity": {"candidate_id": "c1_anchor_residual"},
            }
        )
    )
    return archive, receipt


def rewrite_member(source: Path, member: str, payload: bytes) -> Path:
    output = source.with_name(f"changed-{source.name}")
    with ZipFile(source) as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    members[member] = payload
    return _archive(output, members)
