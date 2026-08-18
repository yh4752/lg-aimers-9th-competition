from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import stat
from types import MappingProxyType
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from experiments.catboost_tabm_blend.inputs import (
    VerifiedStageC,
    VerifiedTrainingInput,
)
from experiments.catboost_tabm_blend.contracts import load_contract as load_blend_contract
from experiments.hierarchical_tabm.contracts import contract_sha256, load_contract
from experiments.hierarchical_tabm.inputs import (
    EXPECTED_BINDING_KEYS,
    HierarchicalInputError,
    classify_and_verify_uploads,
    classify_upload,
)


TRAINING_MEMBERS = {"input_manifest.json", "train.csv", "trackman_history.csv"}
STAGE_C_MEMBERS = {
    "delivery_manifest.json",
    "colab_stage_C.log",
    "tabm_search_stage_C_review_bundle.zip",
    "tabm_search_stage_C_resume_bundle.zip",
}


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, value in members.items():
            info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value)
    return path


def _members(names: set[str]) -> dict[str, bytes]:
    return {name: b"fixture" for name in names}


def test_classifies_renamed_required_uploads_by_exact_members(tmp_path: Path) -> None:
    training = _zip(tmp_path / "anything.zip", _members(TRAINING_MEMBERS))
    stage_c = _zip(tmp_path / "other.zip", _members(STAGE_C_MEMBERS))
    assert classify_upload(training) == "training_input"
    assert classify_upload(stage_c) == "stage_c_delivery"


def test_classifies_future_resume_by_manifest_identity(tmp_path: Path) -> None:
    manifest = json.dumps(
        {"schema_version": 1, "artifact_kind": "hierarchical_tabm_resume_v1"}
    ).encode()
    resume = _zip(
        tmp_path / "renamed.zip",
        {"manifest.json": manifest, "stage_state.json": b"{}"},
    )
    assert classify_upload(resume) == "campaign_resume"


@pytest.mark.parametrize("name", ["test.csv", "submission.csv", "../train.csv"])
def test_rejects_test_submission_or_traversal_members(tmp_path: Path, name: str) -> None:
    bad = _zip(tmp_path / "bad.zip", {name: b"x"})
    with pytest.raises(HierarchicalInputError, match="unknown|unsafe"):
        classify_upload(bad)


def test_rejects_duplicate_and_symlink_members(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.zip"
    with ZipFile(duplicate, "w") as archive:
        archive.writestr("train.csv", b"a")
        archive.writestr("train.csv", b"b")
    with pytest.raises(HierarchicalInputError, match="duplicate"):
        classify_upload(duplicate)

    symlink = tmp_path / "symlink.zip"
    with ZipFile(symlink, "w") as archive:
        info = ZipInfo("train.csv")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, b"target")
    with pytest.raises(HierarchicalInputError, match="unsafe"):
        classify_upload(symlink)


def test_classifies_and_verifies_two_required_uploads(
    tmp_path: Path, monkeypatch
) -> None:
    training_path = _zip(tmp_path / "training.zip", _members(TRAINING_MEMBERS))
    stage_path = _zip(tmp_path / "stage.zip", _members(STAGE_C_MEMBERS))
    contract = replace(
        load_contract(),
        source_stage_c_delivery_sha256=sha256(stage_path.read_bytes()).hexdigest(),
    )
    blend = load_blend_contract()
    monkeypatch.setattr(
        "experiments.hierarchical_tabm.inputs.load_blend_contract",
        lambda: replace(
            blend,
            stage_c=MappingProxyType(
                {
                    **blend.stage_c,
                    "delivery_sha256": contract.source_stage_c_delivery_sha256,
                }
            ),
        ),
    )

    def verify_training(source, destination, blend_contract):
        destination.mkdir(parents=True)
        (destination / "train.csv").write_bytes(b"train")
        (destination / "trackman_history.csv").write_bytes(b"history")
        return VerifiedTrainingInput(
            destination,
            "a" * 64,
            contract.official_train_sha256,
            contract.official_history_sha256,
        )

    def verify_stage(source, destination, blend_contract):
        destination.mkdir(parents=True)
        paths = {}
        for fold in ("2022->2023", "2023->2024"):
            path = destination / f"{fold}.csv"
            path.write_text(
                "row_id,target,probability,game_type,game_month,pitcher_id_known,batter_id_known\n"
                "r,0,0.5,R,4,known,known\n"
            )
            paths[fold] = path
        return VerifiedStageC(
            contract.source_stage_c_delivery_sha256,
            "b" * 64,
            "c" * 64,
            "d" * 64,
            MappingProxyType(paths),
        )

    monkeypatch.setattr(
        "experiments.hierarchical_tabm.inputs.verify_and_extract_training_input",
        verify_training,
    )
    monkeypatch.setattr(
        "experiments.hierarchical_tabm.inputs.verify_and_extract_stage_c",
        verify_stage,
    )
    verified, resume = classify_and_verify_uploads(
        [stage_path, training_path],
        run_root=tmp_path / "run",
        contract=contract,
        expected_contract_sha256=contract_sha256(),
        expected_code_sha256="e" * 64,
    )
    assert resume is None
    assert verified.source_stage_c_path == stage_path.resolve()
    assert verified.source_stage_c_sha256 == contract.source_stage_c_delivery_sha256
    assert tuple(verified.anchor_predictions) == ("2022->2023", "2023->2024")


def test_upload_set_rejects_duplicate_kinds_or_wrong_count(tmp_path: Path) -> None:
    first = _zip(tmp_path / "first.zip", _members(TRAINING_MEMBERS))
    second = _zip(tmp_path / "second.zip", _members(TRAINING_MEMBERS))
    contract = load_contract()
    with pytest.raises(HierarchicalInputError, match="duplicate"):
        classify_and_verify_uploads(
            [first, second], run_root=tmp_path / "run", contract=contract,
            expected_contract_sha256=contract_sha256(), expected_code_sha256="a" * 64,
        )
    with pytest.raises(HierarchicalInputError, match="two or three"):
        classify_and_verify_uploads(
            [first], run_root=tmp_path / "other", contract=contract,
            expected_contract_sha256=contract_sha256(), expected_code_sha256="a" * 64,
        )


def test_identity_is_checked_before_extraction(tmp_path: Path) -> None:
    training = _zip(tmp_path / "training.zip", _members(TRAINING_MEMBERS))
    stage_c = _zip(tmp_path / "stage.zip", _members(STAGE_C_MEMBERS))
    with pytest.raises(HierarchicalInputError, match="contract identity"):
        classify_and_verify_uploads(
            [training, stage_c], run_root=tmp_path / "run", contract=load_contract(),
            expected_contract_sha256="0" * 64, expected_code_sha256="a" * 64,
        )


def test_binding_key_contract_is_exact() -> None:
    assert EXPECTED_BINDING_KEYS == {
        "contract_sha256", "code_sha256", "environment_sha256",
        "input_manifest_sha256", "train_sha256", "history_sha256",
        "stage_c_delivery_sha256", "stage_c_review_sha256",
        "stage_c_state_sha256", "anchor_2022_2023_sha256",
        "anchor_2023_2024_sha256",
    }
