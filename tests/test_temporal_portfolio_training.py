from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.independent_dl.models.tabm import TabMAdapter
from experiments.temporal_portfolio.catboost_training import (
    CATBOOST_PREFIXES,
    TemporalTrainingJob,
    run_catboost_job,
    validate_catboost_result,
)
from experiments.temporal_portfolio.identity import TrainingIdentity
from experiments.temporal_portfolio.lupi_teacher import TeacherOOF
from experiments.temporal_portfolio.tabm_training import TemporalTabMAdapter
from experiments.temporal_portfolio.worker import (
    WorkerPublicationError,
    WorkerBackendDispatcher,
    publish_worker_result,
    run_worker,
    verify_worker_result,
)


class _FixedMembers(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        del x_cat
        return self.logits[: len(x_num)].unsqueeze(-1)


def _features(rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros((rows, 2)), torch.zeros((rows, 0), dtype=torch.long)


def test_decay_weighted_bce_uses_complete_window_denominator() -> None:
    logits = torch.tensor([[0.0, 1.0], [-1.0, 2.0], [2.0, -2.0]])
    model = _FixedMembers(logits)
    adapter = TemporalTabMAdapter(
        sample_weight=np.array([1.0, 0.5, 0.25]), loss_name="bce"
    )
    adapter.bind_device("cpu")
    x_num, x_cat = _features(2)
    target = torch.tensor([1.0, 0.0])

    loss = adapter.loss_for_window(
        model,
        x_num,
        x_cat,
        target,
        row_indices=torch.tensor([0, 1]),
        window_indices=torch.tensor([0, 1, 2]),
    )

    member_targets = target.unsqueeze(1).expand_as(logits[:2])
    per_row = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:2], member_targets, reduction="none"
    ).mean(dim=1)
    expected = (per_row * torch.tensor([1.0, 0.5])).sum() / 1.75
    torch.testing.assert_close(loss, expected)


def test_teacher_changes_only_finite_matched_row_targets() -> None:
    logits = torch.tensor([[0.0, 1.0], [-1.0, 2.0], [2.0, -2.0]])
    adapter = TemporalTabMAdapter(
        sample_weight=np.ones(3),
        loss_name="bce",
        teacher_probability=np.array([0.9, np.nan, 0.1]),
        teacher_lambda=0.25,
    )
    adapter.bind_device("cpu")
    hard_target = torch.tensor([1.0, 0.0, 0.0])

    per_row = adapter.debug_per_row_loss(logits, hard_target)
    hard = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        hard_target.unsqueeze(1).expand_as(logits),
        reduction="none",
    ).mean(dim=1)
    blended_target = torch.tensor([0.975, 0.0, 0.025])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        blended_target.unsqueeze(1).expand_as(logits),
        reduction="none",
    ).mean(dim=1)

    torch.testing.assert_close(per_row[[0, 2]], expected[[0, 2]])
    assert torch.equal(per_row[1], hard[1])
    assert not torch.equal(per_row[0], hard[0])


def test_brier_uses_member_mean_probability_before_square_and_weight() -> None:
    logits = torch.tensor([[0.0, 2.0], [-2.0, 1.0], [1.0, 1.5]])
    model = _FixedMembers(logits)
    adapter = TemporalTabMAdapter(
        sample_weight=np.array([1.0, 0.5, 0.25]), loss_name="brier"
    )
    adapter.bind_device("cpu")
    x_num, x_cat = _features(2)
    target = torch.tensor([1.0, 0.0])

    loss = adapter.loss_for_window(
        model,
        x_num,
        x_cat,
        target,
        row_indices=torch.tensor([0, 1]),
        window_indices=torch.tensor([0, 1, 2]),
    )

    per_row = (logits[:2].sigmoid().mean(dim=1) - target).square()
    expected = (per_row * torch.tensor([1.0, 0.5])).sum() / 1.75
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize(
    "sample_weight",
    [
        np.array([]),
        np.array([[1.0]]),
        np.array([1.0, 0.0]),
        np.array([1.0, -0.1]),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.array([True, False]),
        np.array(["1.0", "0.5"]),
        np.array([1.0 + 0.0j]),
        np.array([np.finfo(np.float64).max]),
        np.array([np.nextafter(0.0, 1.0)]),
    ],
)
def test_invalid_weights_are_rejected(sample_weight: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError), match="weight"):
        TemporalTabMAdapter(sample_weight=sample_weight, loss_name="bce")


def test_array_inputs_do_not_silently_coerce_python_sequences() -> None:
    with pytest.raises(TypeError, match="weight"):
        TemporalTabMAdapter(sample_weight=[1.0, 0.5], loss_name="bce")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="teacher"):
        TemporalTabMAdapter(
            sample_weight=np.ones(2),
            loss_name="bce",
            teacher_probability=[0.5, np.nan],  # type: ignore[arg-type]
            teacher_lambda=0.25,
        )


@pytest.mark.parametrize(
    "teacher_probability",
    [
        np.array([]),
        np.array([[0.5, 0.5]]),
        np.array([0.5]),
        np.array([0.5, -0.1]),
        np.array([0.5, 1.1]),
        np.array([0.5, np.inf]),
        np.array([True, False]),
        np.array(["0.5", "nan"]),
        np.array([0.5 + 0.0j, 0.2 + 0.0j]),
    ],
)
def test_invalid_teacher_probabilities_are_rejected(
    teacher_probability: np.ndarray,
) -> None:
    with pytest.raises((TypeError, ValueError), match="teacher"):
        TemporalTabMAdapter(
            sample_weight=np.ones(2),
            loss_name="bce",
            teacher_probability=teacher_probability,
            teacher_lambda=0.25,
        )


@pytest.mark.parametrize("teacher_lambda", [True, "0.25", np.nan, np.inf, -0.1, 1.1])
def test_invalid_teacher_lambdas_are_rejected(teacher_lambda: object) -> None:
    with pytest.raises((TypeError, ValueError), match="lambda"):
        TemporalTabMAdapter(
            sample_weight=np.ones(2),
            loss_name="bce",
            teacher_probability=np.array([0.5, np.nan]),
            teacher_lambda=teacher_lambda,
        )


def test_teacher_contract_rejects_ambiguous_combinations() -> None:
    with pytest.raises(ValueError, match="teacher_lambda"):
        TemporalTabMAdapter(
            sample_weight=np.ones(2), loss_name="bce", teacher_lambda=0.25
        )
    with pytest.raises(ValueError, match="Brier"):
        TemporalTabMAdapter(
            sample_weight=np.ones(2),
            loss_name="brier",
            teacher_probability=np.array([0.5, np.nan]),
            teacher_lambda=0.0,
        )


def test_constructor_detaches_and_protects_caller_arrays() -> None:
    sample_weight = np.array([1.0, 0.5])
    teacher_probability = np.array([0.9, np.nan])
    adapter = TemporalTabMAdapter(
        sample_weight=sample_weight,
        loss_name="bce",
        teacher_probability=teacher_probability,
        teacher_lambda=0.25,
    )

    sample_weight[:] = 99.0
    teacher_probability[:] = 0.1

    np.testing.assert_array_equal(adapter.sample_weight_np, [1.0, 0.5])
    np.testing.assert_allclose(adapter.teacher_probability_np, [0.9, np.nan], equal_nan=True)
    assert not adapter.sample_weight_np.flags.writeable
    assert adapter.teacher_probability_np is not None
    assert not adapter.teacher_probability_np.flags.writeable
    with pytest.raises(ValueError):
        adapter.sample_weight_np.setflags(write=True)
    with pytest.raises(ValueError):
        adapter.teacher_probability_np.setflags(write=True)


@pytest.mark.parametrize("scale", [1e-20, np.finfo(np.float32).max])
def test_weight_scale_does_not_change_loss_or_gradients(scale: float) -> None:
    model = _FixedMembers(torch.zeros((2, 2)))
    adapter = TemporalTabMAdapter(
        sample_weight=np.array([scale, scale], dtype="float32"),
        loss_name="bce",
    )
    adapter.bind_device("cpu")
    x_num, x_cat = _features(2)
    loss = adapter.loss_for_window(
        model,
        x_num,
        x_cat,
        torch.tensor([1.0, 0.0]),
        row_indices=torch.tensor([0, 1]),
        window_indices=torch.tensor([0, 1]),
    )
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(np.log(2.0), dtype=torch.float32))
    assert torch.isfinite(model.logits.grad).all()


def test_build_calls_parent_and_binds_requested_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    calls: list[tuple[object, object, str]] = []

    def fake_build(
        self: TabMAdapter, model_config: object, metadata: object, device: str
    ) -> object:
        calls.append((model_config, metadata, device))
        return sentinel

    monkeypatch.setattr(TabMAdapter, "build", fake_build)
    monkeypatch.setattr(
        "experiments.temporal_portfolio.tabm_training.import_runtime_module",
        lambda name: torch if name == "torch" else None,
    )
    adapter = TemporalTabMAdapter(
        sample_weight=np.array([1.0, 0.5]), loss_name="bce"
    )
    model_config: dict[str, object] = {}
    metadata = object()

    result = adapter.build(model_config, metadata, "cpu")

    assert result is sentinel
    assert calls == [(model_config, metadata, "cpu")]
    assert adapter._weight is not None
    assert adapter._weight.device == torch.device("cpu")


def test_unbound_loss_fails_clearly() -> None:
    model = _FixedMembers(torch.zeros((2, 2)))
    adapter = TemporalTabMAdapter(sample_weight=np.ones(2), loss_name="bce")
    x_num, x_cat = _features(2)

    with pytest.raises(RuntimeError, match="bind_device"):
        adapter.loss_for_window(
            model,
            x_num,
            x_cat,
            torch.tensor([0.0, 1.0]),
            row_indices=torch.tensor([0, 1]),
            window_indices=torch.tensor([0, 1]),
        )


def test_weighted_loss_gradients_remain_finite() -> None:
    model = _FixedMembers(torch.tensor([[0.0, 1.0], [-1.0, 2.0]]))
    adapter = TemporalTabMAdapter(
        sample_weight=np.array([1.0, 0.5]),
        loss_name="bce",
        teacher_probability=np.array([0.9, np.nan]),
        teacher_lambda=0.25,
    )
    adapter.bind_device("cpu")
    x_num, x_cat = _features(2)
    loss = adapter.loss_for_window(
        model,
        x_num,
        x_cat,
        torch.tensor([1.0, 0.0]),
        row_indices=torch.tensor([0, 1]),
        window_indices=torch.tensor([0, 1]),
    )

    loss.backward()

    assert model.logits.grad is not None
    assert torch.isfinite(model.logits.grad).all()


@pytest.mark.parametrize(
    ("row_indices", "window_indices", "message"),
    [
        (torch.tensor([0.0, 1.0]), torch.tensor([0, 1]), "dtype"),
        (torch.tensor([[0, 1]]), torch.tensor([0, 1]), "one-dimensional"),
        (torch.tensor([0, 2]), torch.tensor([0, 1]), "range"),
        (torch.tensor([0, 1]), torch.tensor([-1, 1]), "range"),
    ],
)
def test_loss_rejects_invalid_index_tensors(
    row_indices: torch.Tensor, window_indices: torch.Tensor, message: str
) -> None:
    model = _FixedMembers(torch.zeros((2, 2)))
    adapter = TemporalTabMAdapter(sample_weight=np.ones(2), loss_name="bce")
    adapter.bind_device("cpu")
    x_num, x_cat = _features(2)

    with pytest.raises((TypeError, ValueError, IndexError, RuntimeError), match=message):
        adapter.loss_for_window(
            model,
            x_num,
            x_cat,
            torch.tensor([0.0, 1.0]),
            row_indices=row_indices,
            window_indices=window_indices,
        )


def test_loss_rejects_index_device_mismatch() -> None:
    model = _FixedMembers(torch.zeros((2, 2)))
    adapter = TemporalTabMAdapter(sample_weight=np.ones(2), loss_name="bce")
    adapter.bind_device("cpu")
    x_num, x_cat = _features(2)

    with pytest.raises(ValueError, match="device"):
        adapter.loss_for_window(
            model,
            x_num,
            x_cat,
            torch.tensor([0.0, 1.0]),
            row_indices=torch.tensor([0, 1], device="meta"),
            window_indices=torch.tensor([0, 1]),
        )


def _identity(expert: str = "catboost") -> TrainingIdentity:
    return TrainingIdentity.from_payload(
        {
            "data_rows": "a" * 64,
            "train_seasons": [2021],
            "valid_year": 2022,
            "decay": None,
            "features": ["S1"],
            "model": {"expert": expert},
            "loss": "bce",
            "seed": 3407,
        }
    )


def _audit_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["r2", "r1"],
            "target": [1, 0],
            "pitcher_id": [11, 12],
            "batter_id": [21, 22],
            "game_type": ["regular", "postseason"],
            "pitcher_id_known": ["known", "oov"],
            "batter_id_known": ["oov", "known"],
            "trackman_available": [1, 0],
            "segment_hand_matchup": ["R_R", "L_R"],
        }
    )


def _catboost_job() -> TemporalTrainingJob:
    return TemporalTrainingJob(
        job_id="catboost_2022",
        expert="catboost",
        identity=_identity(),
        sample_weight=np.array([1.0, 0.5, 0.25]),
        seed=3407,
        audit_frame=_audit_frame(),
        segment_columns=("segment_hand_matchup",),
        train_frame=pd.DataFrame({"feature": [3.0, 2.0, 1.0]}),
        valid_frame=pd.DataFrame({"feature": [20.0, 10.0]}),
        target=np.array([1, 0, 1]),
        valid_row_id=np.array(["r2", "r1"]),
    )


class _CatBoostBackend:
    def __init__(self) -> None:
        self.fit_calls: list[dict[str, object]] = []
        self.predict_calls: list[tuple[pd.DataFrame, int]] = []

    def fit(self, frame, target, *, sample_weight, iterations, seed):
        self.fit_calls.append(
            {
                "frame": frame.copy(),
                "target": np.array(target, copy=True),
                "sample_weight": np.array(sample_weight, copy=True),
                "iterations": iterations,
                "seed": seed,
            }
        )
        return type("Model", (), {"tree_count_": 384})()

    def predict(self, model, frame, *, ntree_end):
        del model
        self.predict_calls.append((frame.copy(), ntree_end))
        return np.array([ntree_end / 1000.0, ntree_end / 1000.0 + 0.1])

    def save_model(self, model, path):
        del model
        Path(path).write_bytes(b"fixture-catboost-model")


def test_catboost_job_predicts_all_and_only_preregistered_prefixes() -> None:
    job = _catboost_job()
    backend = _CatBoostBackend()

    result = run_catboost_job(job, backend=backend)

    assert tuple(result.prefix_predictions) == CATBOOST_PREFIXES
    assert [trees for _, trees in backend.predict_calls] == list(CATBOOST_PREFIXES)
    assert backend.fit_calls[0]["iterations"] == 384
    assert backend.fit_calls[0]["seed"] == 3407
    np.testing.assert_array_equal(
        backend.fit_calls[0]["sample_weight"], [1.0, 0.5, 0.25]
    )
    assert result.valid_row_ids == ("r2", "r1")
    assert result.backend_calls["early_stopping"] is False
    for prefix, probability in result.prefix_predictions.items():
        assert probability.dtype == np.dtype("float64")
        assert probability.shape == (job.valid_rows,)
        np.testing.assert_allclose(probability, [prefix / 1000, prefix / 1000 + 0.1])


@pytest.mark.parametrize(
    "predictions",
    [
        {16: [0.1, 0.2], 64: [0.1, 0.2], 192: [0.1, 0.2]},
        {
            16: [0.1, 0.2],
            64: [0.1, 0.2],
            192: [0.1, 0.2],
            384: [0.1, 0.2],
            385: [0.1, 0.2],
        },
        {16: [0.1], 64: [0.1, 0.2], 192: [0.1, 0.2], 384: [0.1, 0.2]},
        {16: [np.nan, 0.2], 64: [0.1, 0.2], 192: [0.1, 0.2], 384: [0.1, 0.2]},
        {16: [1.1, 0.2], 64: [0.1, 0.2], 192: [0.1, 0.2], 384: [0.1, 0.2]},
    ],
)
def test_catboost_result_rejects_incomplete_or_malformed_prefixes(
    predictions: dict[int, list[float]],
) -> None:
    with pytest.raises(ValueError, match="prefix|probabilit|shape"):
        validate_catboost_result(object(), predictions, np.array(["r2", "r1"]))


def test_catboost_job_detaches_inputs_and_predictions_from_callers() -> None:
    job = _catboost_job()
    backend = _CatBoostBackend()
    result = run_catboost_job(job, backend=backend)
    returned = backend.predict(None, job.valid_frame, ntree_end=16)
    returned[:] = 0.99
    job.train_frame.iloc[0, 0] = -999
    job.sample_weight[0] = 99

    np.testing.assert_allclose(result.prefix_predictions[16], [0.016, 0.116])
    assert backend.fit_calls[0]["frame"].iloc[0, 0] == 3.0
    assert backend.fit_calls[0]["sample_weight"][0] == 1.0
    assert not result.prefix_predictions[16].flags.writeable


def test_worker_publishes_bound_artifacts_and_row_aligned_audit_csv(
    tmp_path: Path,
) -> None:
    job = _catboost_job()

    result_path = run_worker(job, tmp_path, backend=_CatBoostBackend())

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["job_id"] == job.job_id
    assert payload["status"] == "completed"
    assert payload["training_identity_sha256"] == job.identity.sha256
    assert {item["path"] for item in payload["artifacts"]} == {
        "checkpoint_meta.json",
        "metrics.json",
        "model.cbm",
        "predictions.csv",
    }
    for item in payload["artifacts"]:
        path = tmp_path / item["path"]
        assert path.stat().st_size == item["size_bytes"]
        assert sha256(path.read_bytes()).hexdigest() == item["sha256"]
    predictions = pd.read_csv(tmp_path / "predictions.csv")
    assert predictions["row_id"].tolist() == ["r2", "r1"]
    assert set(_audit_frame().columns).issubset(predictions.columns)
    assert predictions["probability"].tolist() == pytest.approx([0.384, 0.484])
    assert verify_worker_result(tmp_path)["job_id"] == job.job_id


def test_worker_verifier_rejects_tamper_missing_and_unbound_files(tmp_path: Path) -> None:
    tampered = tmp_path / "tampered"
    run_worker(_catboost_job(), tampered, backend=_CatBoostBackend())
    (tampered / "metrics.json").write_bytes(b"tampered")
    with pytest.raises(WorkerPublicationError, match="SHA-256|size"):
        verify_worker_result(tampered)

    missing = tmp_path / "missing"
    run_worker(_catboost_job(), missing, backend=_CatBoostBackend())
    (missing / "metrics.json").unlink()
    with pytest.raises(WorkerPublicationError, match="missing"):
        verify_worker_result(missing)

    unbound = tmp_path / "unbound"
    run_worker(_catboost_job(), unbound, backend=_CatBoostBackend())
    (unbound / "extra.txt").write_text("unbound", encoding="utf-8")
    with pytest.raises(WorkerPublicationError, match="unbound|artifact set"):
        verify_worker_result(unbound)


def test_worker_verifier_rejects_symlink_and_traversal(tmp_path: Path) -> None:
    run_worker(_catboost_job(), tmp_path, backend=_CatBoostBackend())
    payload_path = tmp_path / "worker_result.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["artifacts"][0]["path"] = "../escape"
    payload_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    with pytest.raises(WorkerPublicationError, match="path"):
        verify_worker_result(tmp_path)

    payload_path.unlink()
    (tmp_path / "predictions.csv").unlink()
    (tmp_path / "predictions.csv").symlink_to(tmp_path / "metrics.json")
    with pytest.raises(WorkerPublicationError, match="symlink|regular"):
        publish_worker_result(
            tmp_path,
            _catboost_job(),
            [
                tmp_path / "metrics.json",
                tmp_path / "predictions.csv",
                tmp_path / "checkpoint_meta.json",
                tmp_path / "model.cbm",
            ],
            status="completed",
        )


def test_atomic_manifest_failure_leaves_no_completed_worker_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("metrics.json", "predictions.csv", "checkpoint_meta.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr("experiments.temporal_portfolio.worker.os.replace", lambda *_: (_ for _ in ()).throw(OSError("boom")))

    with pytest.raises(OSError, match="boom"):
        publish_worker_result(
            tmp_path,
            _catboost_job(),
            [tmp_path / "metrics.json", tmp_path / "predictions.csv", tmp_path / "checkpoint_meta.json"],
            status="completed",
        )

    assert not (tmp_path / "worker_result.json").exists()


def _tabm_job(*, expert: str, teacher: TeacherOOF | None = None) -> TemporalTrainingJob:
    request = SimpleNamespace(
        train=SimpleNamespace(row_id=np.array(["t1", "t2"])),
        valid=SimpleNamespace(row_id=np.array(["r2", "r1"])),
    )
    return TemporalTrainingJob(
        job_id=f"{expert}_2022",
        expert=expert,
        identity=_identity(expert),
        sample_weight=np.array([1.0, 0.25]),
        seed=3407,
        audit_frame=_audit_frame(),
        segment_columns=("segment_hand_matchup",),
        train_request=request,
        teacher_oof=teacher,
        teacher_oof_sha256=None if teacher is None else teacher._sha256,
        teacher_lambda=0.0 if teacher is None else 0.25,
    )


def _teacher() -> TeacherOOF:
    return TeacherOOF._from_validated(
        row_ids=("t1", "t2"),
        probabilities=np.array([0.9, 0.1]),
        fold_items=(("t1", 0), ("t2", 1)),
        predicted_items=(("t1", 0), ("t2", 1)),
        metadata={"fixture": True},
        backend_evidence={"backend": "fixture"},
    )


def test_worker_dispatches_weighted_tabm_through_existing_fit_entrypoint(
    tmp_path: Path,
) -> None:
    calls: list[tuple[object, TemporalTabMAdapter, object]] = []

    def fake_fit(request, adapter, output_dir, *, backend):
        calls.append((request, adapter, backend))
        checkpoint = Path(output_dir) / "checkpoint.pt"
        checkpoint.write_bytes(b"student")
        return SimpleNamespace(
            predictions=np.array([0.6, 0.4]),
            checkpoint=checkpoint,
            best_epoch=2,
            best_brier=0.24,
        )

    runtime = object()
    job = _tabm_job(expert="tabm")
    run_worker(
        job,
        tmp_path,
        backend=WorkerBackendDispatcher(
            tabm_backend=runtime, fit_function=fake_fit
        ),
    )

    assert len(calls) == 1
    assert calls[0][0] is job.train_request
    assert calls[0][2] is runtime
    np.testing.assert_array_equal(calls[0][1].sample_weight_np, [1.0, 0.25])
    assert calls[0][1].teacher_probability_np is None


def test_lupi_verifies_teacher_hash_before_weighted_tabm_dispatch(
    tmp_path: Path,
) -> None:
    teacher = _teacher()
    calls: list[TemporalTabMAdapter] = []

    def fake_fit(request, adapter, output_dir, *, backend):
        del request, backend
        calls.append(adapter)
        checkpoint = Path(output_dir) / "checkpoint.pt"
        checkpoint.write_bytes(b"student-only")
        return SimpleNamespace(
            predictions=np.array([0.55, 0.45]),
            checkpoint=checkpoint,
            best_epoch=1,
            best_brier=0.245,
        )

    job = _tabm_job(expert="lupi", teacher=teacher)
    run_worker(
        job,
        tmp_path / "ok",
        backend=WorkerBackendDispatcher(tabm_backend=object(), fit_function=fake_fit),
    )
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0].teacher_probability_np, [0.9, 0.1])
    assert not any("teacher" in item["path"] for item in verify_worker_result(tmp_path / "ok")["artifacts"])

    forged = object.__new__(TeacherOOF)
    for name in (
        "_row_ids",
        "_probability_bytes",
        "_fold_items",
        "_predicted_items",
        "_metadata_json",
        "_backend_evidence_json",
        "_sha256",
    ):
        object.__setattr__(forged, name, getattr(teacher, name))
    object.__setattr__(forged, "_probability_bytes", np.array([0.2, 0.8]).tobytes())
    forged_job = _tabm_job(expert="lupi", teacher=forged)
    with pytest.raises(WorkerPublicationError, match="integrity"):
        run_worker(
            forged_job,
            tmp_path / "forged",
            backend=WorkerBackendDispatcher(tabm_backend=object(), fit_function=fake_fit),
        )
    assert len(calls) == 1
    assert not (tmp_path / "forged" / "worker_result.json").exists()
