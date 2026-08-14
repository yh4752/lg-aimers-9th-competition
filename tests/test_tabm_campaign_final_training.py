from __future__ import annotations

from pathlib import Path
import inspect

import pytest
import torch

from experiments.tabm_campaign.final_training import (
    build_inference_manifest,
    FinalTrainingError,
    load_epoch_checkpoint,
    resolve_final_candidate,
    resume_start_epoch,
    save_epoch_checkpoint,
    validate_final_fit_policy,
    fit_final_member,
)


IDENTITY = {
    "contract_sha256": "1" * 64,
    "data_archive_sha256": "2" * 64,
    "train_sha256": "3" * 64,
    "runtime_sha256": "4" * 64,
    "training_source_sha256": "7" * 64,
}


def _checkpoint(completed_epochs: int = 1) -> dict[str, object]:
    return {
        "schema_version": 1,
        "identity": IDENTITY,
        "completed_epochs": completed_epochs,
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {"state": {}, "param_groups": []},
        "scaler": {},
        "python_rng": None,
        "numpy_rng": None,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": [],
        "preprocessing_sha256": "5" * 64,
        "numeric_embedding_sha256": "6" * 64,
    }


def test_epoch_checkpoint_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "epoch.pt"
    save_epoch_checkpoint(path, _checkpoint())
    restored = load_epoch_checkpoint(path, IDENTITY)
    assert restored["completed_epochs"] == 1
    assert restored["model"]["weight"].tolist() == [1.0]
    assert list(tmp_path.glob(".epoch.pt-*")) == []


def test_epoch_checkpoint_rejects_identity_change(tmp_path: Path) -> None:
    path = tmp_path / "epoch.pt"
    save_epoch_checkpoint(path, _checkpoint())
    changed = {**IDENTITY, "train_sha256": "9" * 64}
    with pytest.raises(FinalTrainingError, match="checkpoint identity differs"):
        load_epoch_checkpoint(path, changed)


def test_resume_epoch_is_next_completed_epoch() -> None:
    assert resume_start_epoch(None, 3) == 0
    assert resume_start_epoch({"completed_epochs": 1}, 3) == 1
    assert resume_start_epoch({"completed_epochs": 3}, 3) == 3
    with pytest.raises(FinalTrainingError, match="outside final epoch range"):
        resume_start_epoch({"completed_epochs": 4}, 3)


def test_final_fit_policy_is_single_seed_constant_lr() -> None:
    candidate = {
        "family": "tabm",
        "capacity": "p2",
        "k": 32,
        "width": 512,
        "blocks": 4,
        "dropout": 0.1,
        "num_embedding": "piecewise_linear",
        "loss": "bce",
        "scheduler": "constant",
        "learning_rate": 0.0006,
    }
    validate_final_fit_policy(candidate, seed=3407, epochs=3)
    with pytest.raises(FinalTrainingError, match="seed differs"):
        validate_final_fit_policy(candidate, seed=42, epochs=3)
    with pytest.raises(FinalTrainingError, match="epoch count differs"):
        validate_final_fit_policy(candidate, seed=3407, epochs=2)
    with pytest.raises(FinalTrainingError, match="scheduler differs"):
        validate_final_fit_policy({**candidate, "scheduler": "plateau"}, seed=3407, epochs=3)


def test_final_candidate_changes_only_sealed_fit_fields() -> None:
    selected = {
        "family": "tabm",
        "capacity": "p2",
        "k": 32,
        "width": 512,
        "blocks": 4,
        "dropout": 0.1,
        "num_embedding": "piecewise_linear",
        "loss": "bce",
        "scheduler": "plateau",
        "learning_rate": 0.0006,
        "seed": 3407,
    }
    final_fit = {
        "epochs": 3,
        "scheduler": "constant",
        "learning_rate": 0.0006,
        "weight_decay": 0.0001,
        "effective_batch_size": 4096,
        "micro_batch_size": 512,
    }
    resolved = resolve_final_candidate(selected, final_fit)
    assert resolved == {**selected, "scheduler": "constant"}
    with pytest.raises(FinalTrainingError, match="learning rate differs"):
        resolve_final_candidate(selected, {**final_fit, "learning_rate": 0.0009})


def test_inference_manifest_contains_no_training_state() -> None:
    manifest = build_inference_manifest(
        row_count=10,
        preprocessing_state="preprocessing_state.json",
        member={
            "seed": 3407,
            "weights": "tabm_member_0_seed_3407.pt",
            "numeric_state": "numeric_embedding_0.json",
            "model_config": {"architecture": "tabm"},
        },
        files={"preprocessing_state.json": "5" * 64},
        identity=IDENTITY,
    )
    assert manifest["fit_scope"] == "official_train_2019_2024_only"
    assert manifest["epochs"] == 3
    assert manifest["seeds"] == [3407]
    assert manifest["scheduler"] == "constant"
    encoded = repr(manifest).lower()
    assert "optimizer" not in encoded
    assert "scaler" not in encoded
    assert "rng" not in encoded


def test_final_fit_api_requires_checkpoint_identity() -> None:
    parameters = inspect.signature(fit_final_member).parameters
    assert tuple(parameters) == (
        "data_dir",
        "artifact_root",
        "candidate",
        "seed",
        "epochs",
        "absolute_deadline",
        "checkpoint_path",
        "checkpoint_identity",
        "on_epoch_checkpoint",
    )
