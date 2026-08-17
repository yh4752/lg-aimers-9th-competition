from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from types import MappingProxyType
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd
import pytest

from experiments.catboost_tabm_blend.contracts import contract_sha256, load_contract
from experiments.catboost_tabm_blend.inputs import (
    BlendInputError,
    file_sha256,
    prepare_input_archive,
    verify_and_extract_stage_c,
    verify_and_extract_training_input,
)
from experiments.tabm_campaign.artifacts import StageEvidence, write_stage_bundles


def _sha(value: bytes) -> str:
    return sha256(value).hexdigest()


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in sorted(members.items()):
            info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value)
    return buffer.getvalue()


def _fixture_contract(train: bytes, history: bytes):
    return replace(
        load_contract(),
        official_train_sha256=_sha(train),
        official_history_sha256=_sha(history),
    )


def test_prepare_and_verify_training_input(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    train = b"row_id,season,control_success\nA,2022,1\n"
    history = b"pitcher_id,season\nP,2022\n"
    (source / "train.csv").write_bytes(train)
    (source / "trackman_history.csv").write_bytes(history)
    (source / "test.csv").write_bytes(b"must not be included")
    contract = _fixture_contract(train, history)
    output = tmp_path / "input.zip"

    prepare_input_archive(source, output, contract)
    verified = verify_and_extract_training_input(
        output, tmp_path / "extracted", contract
    )

    with ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "input_manifest.json",
            "train.csv",
            "trackman_history.csv",
        }
    assert verified.train_sha256 == _sha(train)
    assert verified.history_sha256 == _sha(history)
    assert (verified.data_dir / "train.csv").read_bytes() == train
    assert not (verified.data_dir / "test.csv").exists()


def test_input_preparation_is_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    train = b"train"
    history = b"history"
    (source / "train.csv").write_bytes(train)
    (source / "trackman_history.csv").write_bytes(history)
    contract = _fixture_contract(train, history)
    first = prepare_input_archive(source, tmp_path / "first.zip", contract)
    second = prepare_input_archive(source, tmp_path / "second.zip", contract)

    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(BlendInputError, match="exists"):
        prepare_input_archive(source, first, contract)


def test_training_input_rejects_extra_or_unsafe_member(tmp_path: Path) -> None:
    train = b"train"
    history = b"history"
    contract = _fixture_contract(train, history)
    manifest = json.dumps(
        {
            "schema_version": 1,
            "artifact_kind": "catboost_tabm_blend_input",
            "campaign_config_sha256": contract_sha256(),
            "members": {
                "train.csv": {"size": len(train), "sha256": _sha(train)},
                "trackman_history.csv": {
                    "size": len(history),
                    "sha256": _sha(history),
                },
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    archive = tmp_path / "bad.zip"
    archive.write_bytes(
        _zip_bytes(
            {
                "input_manifest.json": manifest,
                "train.csv": train,
                "trackman_history.csv": history,
                "../test.csv": b"bad",
            }
        )
    )

    with pytest.raises(BlendInputError):
        verify_and_extract_training_input(archive, tmp_path / "out", contract)


def _prediction_bytes(prefix: str) -> bytes:
    frame = pd.DataFrame(
        {
            "row_id": [f"{prefix}1", f"{prefix}2"],
            "target": [0, 1],
            "probability": [0.4, 0.6],
            "game_type": ["R", "R"],
            "game_month": [4, 4],
            "pitcher_id_known": ["known", "oov"],
            "batter_id_known": ["known", "known"],
        }
    )
    return frame.to_csv(index=False).encode()


def _stage_c_fixture(tmp_path: Path):
    base = load_contract()
    prediction_members = dict(base.stage_c["prediction_members"])
    state = {
        "version": "C",
        "stage_complete": True,
        "selected_predictor": "single_s3407",
        "final_members": [
            {
                "status": "completed",
                "seed": 3407,
                "temporal_best_epochs": [3, 0],
            }
        ],
    }
    state_bytes = json.dumps(state, sort_keys=True).encode()
    review_members = {
        "stage_state.json": state_bytes,
        prediction_members["2022->2023"]: _prediction_bytes("A"),
        prediction_members["2023->2024"]: _prediction_bytes("B"),
    }
    evidence = StageEvidence(
        version="C",
        campaign_config_sha256="c" * 64,
        prior_manifest_sha256="d" * 64,
        review_members=review_members,
        resume_members={"stage_state.json": state_bytes},
    )
    bundles = write_stage_bundles(tmp_path / "nested", evidence, bundle_prefix="tabm_search_stage")
    review = bundles.review.read_bytes()
    assert bundles.resume is not None
    resume = bundles.resume.read_bytes()
    log = b"BUNDLE_SUCCESS version=C\n"
    payloads = {
        "colab_stage_C.log": log,
        "tabm_search_stage_C_review_bundle.zip": review,
        "tabm_search_stage_C_resume_bundle.zip": resume,
    }
    delivery_manifest = {
        "schema_version": 1,
        "artifact_kind": "tabm_colab_stage_C_delivery",
        "run_uuid": "fixture-run",
        "runtime_identity": {"fixture": True},
        "final_stage_complete": True,
        "members": {
            name: {"size": len(value), "sha256": _sha(value)}
            for name, value in sorted(payloads.items())
        },
    }
    outer = _zip_bytes(
        {
            **payloads,
            "delivery_manifest.json": json.dumps(
                delivery_manifest, sort_keys=True, separators=(",", ":")
            ).encode(),
        }
    )
    delivery = tmp_path / "delivery.zip"
    delivery.write_bytes(outer)
    stage_c = dict(base.stage_c)
    stage_c.update(
        delivery_sha256=_sha(outer),
        review_sha256=_sha(review),
        resume_sha256=_sha(resume),
        stage_state_sha256=_sha(state_bytes),
        prediction_members=MappingProxyType(prediction_members),
    )
    contract = replace(base, stage_c=MappingProxyType(stage_c))
    return delivery, contract


def test_stage_c_delivery_exposes_only_selected_two_fold_predictions(tmp_path: Path) -> None:
    delivery, contract = _stage_c_fixture(tmp_path)

    verified = verify_and_extract_stage_c(
        delivery, tmp_path / "stage-c-extracted", contract
    )

    assert verified.delivery_sha256 == file_sha256(delivery)
    assert set(verified.prediction_paths) == {"2022->2023", "2023->2024"}
    assert all(path.is_file() for path in verified.prediction_paths.values())


def test_stage_c_delivery_hash_drift_is_rejected(tmp_path: Path) -> None:
    delivery, contract = _stage_c_fixture(tmp_path)
    contract = replace(
        contract,
        stage_c=MappingProxyType({**contract.stage_c, "delivery_sha256": "0" * 64}),
    )

    with pytest.raises(BlendInputError, match="delivery SHA"):
        verify_and_extract_stage_c(delivery, tmp_path / "stage-c-extracted", contract)
