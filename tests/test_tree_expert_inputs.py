from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd
import pytest

from experiments.tree_expert.contracts import E1Contract, load_e1_contract
from experiments.tree_expert.inputs import (
    PREDICTION_COLUMNS,
    TreeExpertInputError,
    file_sha256,
    prepare_e1_input,
    verify_and_extract_e1_input,
    verify_official_data,
)


PREDICTION_MEMBER = (
    "predictions/"
    "c_final__a__p2__piecewise_linear__bce__plateau__s42__s3407"
    "__tr2023__va2024.csv"
)


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name, payload in sorted(members.items()):
            info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    return buffer.getvalue()


def _stage_c_delivery(path: Path) -> tuple[Path, E1Contract]:
    predictions = pd.DataFrame(
        {
            "row_id": ["TRAIN_3", "TRAIN_4"],
            "target": [0, 1],
            "probability": [0.25, 0.75],
            "game_type": ["R", "R"],
            "game_month": [3, 3],
            "pitcher_id_known": ["known", "oov"],
            "batter_id_known": ["known", "known"],
        }
    ).loc[:, PREDICTION_COLUMNS]
    prediction_bytes = predictions.to_csv(index=False).encode("utf-8")
    review_payloads = {
        PREDICTION_MEMBER: prediction_bytes,
        "stage_state.json": _json_bytes(
            {
                "stage_complete": True,
                "final_members": [
                    {
                        "candidate_id": PREDICTION_MEMBER.removeprefix("predictions/").removesuffix(".csv"),
                        "seed": 3407,
                        "predictions": Path(PREDICTION_MEMBER).name,
                        "status": "completed",
                    }
                ],
                "predictor_evidence": [
                    {
                        "accepted": True,
                        "predictor_id": "single_s3407",
                        "members": [{"seed": 3407}],
                    }
                ],
            }
        ),
    }
    review_manifest = {
        "schema_version": 1,
        "artifact_kind": "review",
        "review_only": True,
        "version": "C",
        "campaign_config_sha256": "1" * 64,
        "prior_manifest_sha256": "2" * 64,
        "members": {
            name: sha256(payload).hexdigest()
            for name, payload in sorted(review_payloads.items())
        },
    }
    review = _zip_bytes({**review_payloads, "manifest.json": _json_bytes(review_manifest)})
    outer_payloads = {
        "tabm_search_stage_C_review_bundle.zip": review,
        "tabm_search_stage_C_resume_bundle.zip": b"resume",
        "colab_stage_C.log": b"complete\n",
    }
    outer_manifest = {
        "schema_version": 1,
        "artifact_kind": "tabm_colab_stage_C_delivery",
        "final_stage_complete": True,
        "members": {
            name: {"size": len(payload), "sha256": sha256(payload).hexdigest()}
            for name, payload in sorted(outer_payloads.items())
        },
    }
    path.write_bytes(_zip_bytes({**outer_payloads, "delivery_manifest.json": _json_bytes(outer_manifest)}))
    contract = replace(
        load_e1_contract(),
        stage_c_delivery_sha256=file_sha256(path),
        stage_c_review_sha256=sha256(review).hexdigest(),
    )
    return path, contract


def test_official_data_requires_exact_top_level_train_and_history(tmp_path: Path) -> None:
    root = tmp_path / "official"
    root.mkdir()
    (root / "train.csv").write_text("row_id,control_success\nTRAIN_1,1\n", encoding="utf-8")
    (root / "trackman_history.csv").write_text("pitcher_id,season\np1,2023\n", encoding="utf-8")

    verified = verify_official_data(root, load_e1_contract(), testing=True)

    assert verified.train.name == "train.csv"
    assert verified.history.name == "trackman_history.csv"
    assert verified.train_sha256 == file_sha256(verified.train)


def test_e1_input_extracts_only_bound_2023_to_2024_single_seed(tmp_path: Path) -> None:
    delivery, contract = _stage_c_delivery(tmp_path / "delivery.zip")

    archive = prepare_e1_input(delivery, tmp_path / "input.zip", contract)
    verified = verify_and_extract_e1_input(archive, tmp_path / "verified", contract)

    assert verified.baseline_fold == "2023->2024"
    frame = pd.read_csv(verified.baseline_predictions)
    assert tuple(frame.columns) == PREDICTION_COLUMNS
    assert frame["row_id"].is_unique
    with ZipFile(archive) as source:
        assert set(source.namelist()) == {"manifest.json", "baseline_predictions.csv"}


def test_e1_input_rejects_same_named_unbound_delivery(tmp_path: Path) -> None:
    delivery, contract = _stage_c_delivery(tmp_path / "delivery.zip")
    changed = tmp_path / "changed.zip"
    changed.write_bytes(delivery.read_bytes() + b"changed")

    with pytest.raises(TreeExpertInputError, match="Stage C delivery SHA-256 differs"):
        prepare_e1_input(changed, tmp_path / "input.zip", contract)
