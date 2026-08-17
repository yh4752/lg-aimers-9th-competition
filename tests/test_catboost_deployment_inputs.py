from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from types import MappingProxyType
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from experiments.catboost_deployment.contracts import load_contract
from experiments.catboost_deployment.inputs import (
    DeploymentInputError,
    classify_and_verify_sources,
    classify_source,
)
from experiments.catboost_tabm_blend.inputs import VerifiedStageC, VerifiedTrainingInput


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, value in members.items():
            info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value)
    return path


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    return buffer.getvalue()


def _source_archives(tmp_path: Path) -> tuple[Path, Path, Path]:
    training = _zip(
        tmp_path / "renamed-a.zip",
        {
            "input_manifest.json": b"{}",
            "train.csv": b"x",
            "trackman_history.csv": b"x",
        },
    )
    stage_c = _zip(
        tmp_path / "renamed-b.zip",
        {
            "delivery_manifest.json": b"{}",
            "colab_stage_C.log": b"x",
            "tabm_search_stage_C_review_bundle.zip": b"x",
            "tabm_search_stage_C_resume_bundle.zip": b"x",
        },
    )
    decision = json.dumps(
        {
            "baseline_fold_brier": {"2022->2023": 0.25, "2023->2024": 0.24},
            "baseline_weighted_brier": 0.245,
            "prediction_correlation": {"2022->2023": 0.5, "2023->2024": 0.5},
            "residual_correlation": {"2022->2023": 0.5, "2023->2024": 0.5},
            "segment_diagnostics": {},
            "candidates": [
                {
                    "tabm_weight": 0.7,
                    "fold_brier": {"2022->2023": 0.249, "2023->2024": 0.239},
                    "weighted_brier": 0.244,
                    "weighted_gain": 0.001,
                    "fold_regression": {"2022->2023": -0.001, "2023->2024": -0.001},
                    "passed": True,
                }
            ],
            "selected_tabm_weight": 0.7,
            "reason": "fixed_blend_passed",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    review = _zip_bytes({"decision/blend_decision.json": decision})
    blend = _zip(
        tmp_path / "renamed-c.zip",
        {
            "blend_campaign.log": b"BLEND_DECISION\n",
            "catboost_tabm_blend_review.zip": review,
            "catboost_tabm_blend_resume.zip": b"resume",
            "delivery_manifest.json": b"{}",
        },
    )
    return training, stage_c, blend


def test_classifies_three_required_sources_by_content(tmp_path: Path) -> None:
    paths = _source_archives(tmp_path)

    assert {classify_source(path) for path in paths} == {
        "training_input",
        "stage_c_delivery",
        "blend_delivery",
    }


def test_classifies_deployment_resume_by_manifest(tmp_path: Path) -> None:
    resume = _zip(
        tmp_path / "anything.zip",
        {
            "manifest.json": b'{"artifact_kind":"catboost_deployment_resume"}',
            "contract/contract.json": b"{}",
            "state/stage_state.json": b"{}",
        },
    )

    assert classify_source(resume) == "deployment_resume"


@pytest.mark.parametrize(
    "members",
    [
        {"test.csv": b"x"},
        {"sample_submission.csv": b"x"},
        {"../train.csv": b"x"},
    ],
)
def test_rejects_unknown_test_or_submission_archive(
    tmp_path: Path, members: dict[str, bytes]
) -> None:
    source = _zip(tmp_path / "x.zip", members)

    with pytest.raises(DeploymentInputError):
        classify_source(source)


def test_rejects_duplicate_members(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with ZipFile(source, "w") as archive:
            archive.writestr("train.csv", b"a")
            archive.writestr("train.csv", b"b")

    with pytest.raises(DeploymentInputError, match="duplicate"):
        classify_source(source)


def test_verifies_sources_and_seals_promoted_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training_path, stage_c_path, blend_path = _source_archives(tmp_path)
    blend_sha = sha256(blend_path.read_bytes()).hexdigest()
    stage_sha = sha256(stage_c_path.read_bytes()).hexdigest()
    contract = replace(
        load_contract(),
        source_blend_delivery_sha256=blend_sha,
        source_stage_c_delivery_sha256=stage_sha,
    )
    verified_training = VerifiedTrainingInput(
        data_dir=tmp_path / "official",
        manifest_sha256="1" * 64,
        train_sha256="2" * 64,
        history_sha256="3" * 64,
    )
    verified_stage_c = VerifiedStageC(
        delivery_sha256=stage_sha,
        review_sha256="4" * 64,
        resume_sha256="5" * 64,
        stage_state_sha256="6" * 64,
        prediction_paths=MappingProxyType({}),
    )
    monkeypatch.setattr(
        "experiments.catboost_deployment.inputs.verify_and_extract_training_input",
        lambda *args, **kwargs: verified_training,
    )
    monkeypatch.setattr(
        "experiments.catboost_deployment.inputs.verify_and_extract_stage_c",
        lambda *args, **kwargs: verified_stage_c,
    )
    monkeypatch.setattr(
        "experiments.catboost_deployment.inputs.verify_blend_delivery",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "experiments.catboost_deployment.inputs._old_blend_bindings",
        lambda *args, **kwargs: {"code_sha256": "8" * 64},
    )

    verified, resume = classify_and_verify_sources(
        [blend_path, training_path, stage_c_path],
        run_root=tmp_path / "run",
        contract=contract,
        expected_code_sha256="7" * 64,
    )

    assert resume is None
    assert verified.source_blend_sha256 == blend_sha
    assert verified.source_selected_tabm_weight == 0.7
    assert len(verified.source_decision_sha256) == 64
    assert verified.training is verified_training
    assert verified.stage_c is verified_stage_c


def test_rejects_duplicate_source_kind(tmp_path: Path) -> None:
    training, stage_c, blend = _source_archives(tmp_path)
    duplicate = tmp_path / "duplicate-training.zip"
    duplicate.write_bytes(training.read_bytes())

    with pytest.raises(DeploymentInputError, match="duplicate"):
        classify_and_verify_sources(
            [training, duplicate, stage_c, blend],
            run_root=tmp_path / "run",
            contract=load_contract(),
            expected_code_sha256="7" * 64,
        )
