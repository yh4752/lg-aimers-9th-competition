from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from types import MappingProxyType
import warnings
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd
import pytest

from experiments.tree_expert.e2_contracts import E2Contract, load_e2_contract
from experiments.tree_expert.e2_inputs import (
    E2_INPUT_MEMBERS,
    E2InputError,
    verify_and_extract_e2_input,
)


PREDICTION_MEMBERS = (
    "e1/c1_f3_predictions.csv",
    "e1/c2_f3_predictions.csv",
    "stage_c/tabm_f2_predictions.csv",
    "stage_c/tabm_f3_predictions.csv",
)


def _prediction_bytes() -> bytes:
    return pd.DataFrame(
        {
            "row_id": ["r1", "r2"],
            "target": [0, 1],
            "probability": [0.4, 0.6],
            "game_type": ["R", "F"],
            "game_month": [4, 5],
            "pitcher_id_known": ["known", "oov"],
            "batter_id_known": ["known", "known"],
        }
    ).to_csv(index=False).encode("utf-8")


def _member_payloads() -> dict[str, bytes]:
    predictions = _prediction_bytes()
    return {
        "e1/decision.json": json.dumps(
            {
                "status": "completed",
                "promoted": ["c1_anchor_residual", "c2_trackman_residual"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
        "e1/c1_f3_metrics.json": json.dumps(
            {"objective": "residual", "best_iteration": 77, "row_count": 2}
        ).encode("utf-8"),
        "e1/c2_f3_metrics.json": json.dumps(
            {"objective": "residual", "best_iteration": 81, "row_count": 2}
        ).encode("utf-8"),
        **{name: predictions for name in PREDICTION_MEMBERS},
        "tabm/script.py": b"def predict(rows):\n    return rows\n",
        "tabm/requirements.txt": b"tabm==0.0.3\n",
        "tabm/model/inference_manifest.json": b"{}",
        "tabm/model/numeric_embedding_0.json": b"{}",
        "tabm/model/preprocessing_state.json": b"{}",
        "tabm/model/tabm_member_0_seed_3407.pt": b"fake-weight",
    }


def _fixture_contract(members: dict[str, bytes]) -> E2Contract:
    contract = load_e2_contract()
    hashes = dict(contract.input_hashes)
    hashes["tabm_weight_sha256"] = sha256(
        members["tabm/model/tabm_member_0_seed_3407.pt"]
    ).hexdigest()
    return replace(contract, input_hashes=MappingProxyType(hashes))


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _write_input(
    path: Path,
    members: dict[str, bytes],
    contract: E2Contract,
    *,
    decision_status: str = "completed",
) -> Path:
    payloads = dict(members)
    if decision_status != "completed":
        payloads["e1/decision.json"] = json.dumps(
            {
                "status": decision_status,
                "promoted": ["c1_anchor_residual", "c2_trackman_residual"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    reference = pd.read_csv(BytesIO(_prediction_bytes()))
    row_target_sha256 = sha256(
        reference.loc[:, ["row_id", "target"]].to_csv(index=False).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "artifact_kind": "tree_expert_e2_input_v1",
        "review_only": False,
        "submission_package": False,
        "campaign_id": "tree_expert_e2_v1",
        "lineage": dict(contract.input_hashes),
        "prediction_bindings": {
            name: {"row_target_sha256": row_target_sha256}
            for name in PREDICTION_MEMBERS
        },
        "members": {
            name: {"size": len(value), "sha256": sha256(value).hexdigest()}
            for name, value in sorted(payloads.items())
        },
    }
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            _zip_info("manifest.json"),
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        )
        for name, value in sorted(payloads.items()):
            archive.writestr(_zip_info(name), value)
    return path


def test_extracts_only_the_sealed_member_set(tmp_path: Path) -> None:
    members = _member_payloads()
    contract = _fixture_contract(members)
    source = _write_input(tmp_path / "input.zip", members, contract)

    verified = verify_and_extract_e2_input(
        source,
        tmp_path / "extracted",
        contract=contract,
    )

    assert set(E2_INPUT_MEMBERS) == {"manifest.json", *members}
    assert set(verified.e1_predictions) == {
        "c1_anchor_residual",
        "c2_trackman_residual",
    }
    assert set(verified.e1_metrics) == {
        "c1_anchor_residual",
        "c2_trackman_residual",
    }
    assert set(verified.tabm_predictions) == {"2022->2023", "2023->2024"}
    assert verified.tabm_runtime_root == tmp_path / "extracted" / "tabm"
    assert verified.lineage == dict(contract.input_hashes)


def test_rejects_changed_e1_decision(tmp_path: Path) -> None:
    members = _member_payloads()
    contract = _fixture_contract(members)
    source = _write_input(
        tmp_path / "input.zip",
        members,
        contract,
        decision_status="rejected",
    )

    with pytest.raises(E2InputError, match="E1 decision differs"):
        verify_and_extract_e2_input(source, tmp_path / "out", contract=contract)


@pytest.mark.parametrize("mutation", ["duplicate", "traversal", "symlink"])
def test_rejects_unsafe_zip_members(tmp_path: Path, mutation: str) -> None:
    members = _member_payloads()
    contract = _fixture_contract(members)
    valid = _write_input(tmp_path / "valid.zip", members, contract)
    changed = tmp_path / f"{mutation}.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with ZipFile(valid) as source, ZipFile(changed, "w") as target:
            for info in source.infolist():
                target.writestr(info, source.read(info.filename))
            if mutation == "duplicate":
                target.writestr(_zip_info("manifest.json"), source.read("manifest.json"))
            elif mutation == "traversal":
                target.writestr(_zip_info("../escape"), b"x")
            else:
                info = _zip_info("link")
                info.external_attr = 0o120777 << 16
                target.writestr(info, b"target")

    with pytest.raises(E2InputError, match="unsafe|duplicate|member names"):
        verify_and_extract_e2_input(changed, tmp_path / "out", contract=contract)


@pytest.mark.parametrize("mutation", ["duplicate_row", "reverse_order", "wrong_target"])
def test_rejects_changed_fold_prediction_evidence(
    tmp_path: Path,
    mutation: str,
) -> None:
    members = _member_payloads()
    frame = pd.read_csv(BytesIO(members["stage_c/tabm_f2_predictions.csv"]))
    if mutation == "duplicate_row":
        frame.loc[1, "row_id"] = frame.loc[0, "row_id"]
    elif mutation == "reverse_order":
        frame = frame.iloc[::-1]
    else:
        frame.loc[0, "target"] = 1
    members["stage_c/tabm_f2_predictions.csv"] = frame.to_csv(index=False).encode()
    contract = _fixture_contract(members)
    source = _write_input(tmp_path / "input.zip", members, contract)

    if mutation == "duplicate_row":
        with pytest.raises(E2InputError, match="row_id"):
            verify_and_extract_e2_input(source, tmp_path / "out", contract=contract)
    else:
        with pytest.raises(E2InputError, match="fold prediction alignment"):
            verify_and_extract_e2_input(source, tmp_path / "out", contract=contract)


def test_rejects_changed_tabm_weight_binding(tmp_path: Path) -> None:
    members = _member_payloads()
    contract = load_e2_contract()
    source = _write_input(tmp_path / "input.zip", members, contract)

    with pytest.raises(E2InputError, match="TabM weight SHA-256 differs"):
        verify_and_extract_e2_input(source, tmp_path / "out", contract=contract)


def test_rejects_nonempty_destination(tmp_path: Path) -> None:
    members = _member_payloads()
    contract = _fixture_contract(members)
    source = _write_input(tmp_path / "input.zip", members, contract)
    destination = tmp_path / "out"
    destination.mkdir()
    (destination / "keep.txt").write_text("user data")

    with pytest.raises(E2InputError, match="destination is not empty"):
        verify_and_extract_e2_input(source, destination, contract=contract)
