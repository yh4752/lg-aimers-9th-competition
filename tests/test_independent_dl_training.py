from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from experiments.independent_dl.features import FeatureBatch
from experiments.independent_dl.models.ft_transformer import ft_transformer_kwargs
from experiments.independent_dl.models.mlp_resnet import mlp_resnet_kwargs
from experiments.independent_dl.models.common import quantile_bin_edges
from experiments.independent_dl.models import common as model_common
from experiments.independent_dl.models.tabm import SUPPORTED_NUM_EMBEDDINGS
from experiments.independent_dl.training import (
    BackendAttemptResult,
    TrainRequest,
    call_adapter_loss,
    fit_candidate,
    inspect_cuda_hardware,
    prepare_adapter_context,
    refresh_retrieval_cache,
)


def test_cuda_hardware_reports_visible_devices_without_claiming_multi_gpu() -> None:
    properties = SimpleNamespace(total_memory=40 * 1024**3)
    cuda = SimpleNamespace(
        device_count=lambda: 2,
        get_device_name=lambda index: f"NVIDIA A100 #{index}",
        get_device_properties=lambda index: properties,
    )

    hardware = inspect_cuda_hardware(SimpleNamespace(cuda=cuda))

    assert hardware == {
        "device_count": 2,
        "devices": (
            {"index": 0, "name": "NVIDIA A100 #0", "vram_bytes": 40 * 1024**3},
            {"index": 1, "name": "NVIDIA A100 #1", "vram_bytes": 40 * 1024**3},
        ),
        "training_mode": "single_gpu",
        "training_device_indices": (0,),
    }


def test_cuda_hardware_does_not_claim_gpu_when_none_is_visible() -> None:
    cuda = SimpleNamespace(device_count=lambda: 0)

    hardware = inspect_cuda_hardware(SimpleNamespace(cuda=cuda))

    assert hardware["training_mode"] == "cpu"
    assert hardware["training_device_indices"] == ()


class _RecordingBackend:
    def __init__(self, *, oom_once: bool = False) -> None:
        self.oom_once = oom_once
        self.calls: list[dict[str, object]] = []

    def run_attempt(self, **kwargs: object) -> BackendAttemptResult:
        self.calls.append(dict(kwargs))
        if self.oom_once and len(self.calls) == 1:
            raise RuntimeError("CUDA out of memory")
        output_dir = Path(kwargs["output_dir"])
        checkpoint = output_dir / "checkpoint.pt"
        checkpoint.write_bytes(b"fake-checkpoint")
        return BackendAttemptResult(
            best_epoch=11,
            best_brier=0.24,
            checkpoint=checkpoint,
            predictions=np.array([0.4, 0.6], dtype="float64"),
        )


class _FakeAdapter:
    pass


def _batch(prefix: str, *, with_target: bool) -> FeatureBatch:
    return FeatureBatch(
        row_id=np.array([f"{prefix}-1", f"{prefix}-2"]),
        season=np.array([2024, 2024], dtype="int64"),
        game_type=np.array(["R", "F"]),
        x_num=np.array([[0.0, 1.0], [1.0, 0.0]], dtype="float32"),
        x_cat=np.array([[1], [2]], dtype="int64"),
        y=(np.array([0.0, 1.0], dtype="float32") if with_target else None),
    )


def _request() -> TrainRequest:
    return TrainRequest(
        candidate_id="tabm__raw_typed__p2__s42",
        family="tabm",
        seed=42,
        epochs=20,
        model_config={
            "architecture": "tabm",
            "k": 32,
            "width": 512,
            "blocks": 4,
            "dropout": 0.1,
            "num_embedding": "piecewise_linear",
        },
        training_config={
            "optimizer": "adamw",
            "scheduler": "plateau",
            "learning_rate": 0.0006,
            "weight_decay": 0.0001,
            "effective_batch_size": 4096,
            "micro_batch_size": 512,
            "amp": True,
            "patience": 40,
        },
        train=_batch("train", with_target=True),
        valid=_batch("valid", with_target=True),
    )


def test_oom_reduces_microbatch_without_changing_model_or_effective_batch(
    tmp_path: Path,
) -> None:
    request = _request()
    backend = _RecordingBackend(oom_once=True)

    result = fit_candidate(request, _FakeAdapter(), tmp_path, backend=backend)

    assert result.model_config == request.model_config
    assert result.attempted_micro_batches == (512, 256)
    assert result.effective_batch_size == 4096
    assert [call["accumulation_steps"] for call in backend.calls] == [8, 16]
    assert all(call["request"].model_config == request.model_config for call in backend.calls)


def test_resume_starts_after_last_complete_checkpoint(tmp_path: Path) -> None:
    request = _request()
    (tmp_path / "checkpoint.pt").write_bytes(b"checkpoint")
    (tmp_path / "checkpoint_meta.json").write_text(
        json.dumps(
            {
                "candidate_id": request.candidate_id,
                "epoch": 7,
                "checkpoint": "checkpoint.pt",
            }
        ),
        encoding="utf-8",
    )
    backend = _RecordingBackend()

    result = fit_candidate(request, _FakeAdapter(), tmp_path, backend=backend)

    assert result.started_epoch == 8
    assert backend.calls[0]["resume_epoch"] == 8


def test_model_adapter_modules_import_without_torch_site_packages() -> None:
    command = (
        "import sys; "
        "import experiments.independent_dl.models.mlp_resnet; "
        "import experiments.independent_dl.models.ft_transformer; "
        "import experiments.independent_dl.models.tabm; "
        "assert 'torch' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_all_sealed_tabm_numerical_embeddings_are_supported() -> None:
    assert SUPPORTED_NUM_EMBEDDINGS == {
        "linear_relu",
        "piecewise_linear",
        "periodic",
    }


def test_model_kwargs_preserve_sealed_capacity_and_embedding_width() -> None:
    metadata = type(
        "Metadata",
        (),
        {"n_num_features": 30, "categorical_cardinalities": (3, 9)},
    )()
    mlp = mlp_resnet_kwargs(
        {
            "architecture": "resnet",
            "width": 2048,
            "blocks": 16,
            "dropout": 0.1,
            "embedding_dim": 96,
        },
        metadata,
    )
    transformer = ft_transformer_kwargs(
        {
            "architecture": "ft_transformer",
            "dim": 512,
            "layers": 12,
            "heads": 16,
            "attention_dropout": 0.1,
            "ffn_dropout": 0.1,
        },
        metadata,
    )

    assert mlp["d_in"] == 30 + 2 * 96
    assert mlp["d_block"] == 2048
    assert mlp["n_blocks"] == 16
    assert transformer["d_block"] == 512
    assert transformer["n_blocks"] == 12
    assert transformer["attention_n_heads"] == 16
    assert transformer["ffn_d_hidden_multiplier"] == 4 / 3
    assert transformer["residual_dropout"] == 0.0


def test_piecewise_edges_retain_constant_numeric_columns() -> None:
    edges = quantile_bin_edges(
        np.array([[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]], dtype="float32")
    )

    assert len(edges[0]) == 2
    assert edges[0][0] < 0.0 < edges[0][1]
    assert all(len(item) >= 2 for item in edges)


def test_activation_checkpoint_module_is_imported_lazily(monkeypatch) -> None:
    class FakeModule:
        def __init__(self) -> None:
            self.training = True

        def __call__(self, *args):
            return self.forward(*args)

    class FakeModel:
        def __call__(self, x_num, x_cat):
            return (x_num, x_cat)

    calls: list[tuple[object, object]] = []
    checkpoint_module = SimpleNamespace(
        checkpoint=lambda model, x_num, x_cat, use_reentrant: calls.append(
            (x_num, x_cat)
        )
    )
    monkeypatch.setattr(
        model_common,
        "import_runtime_module",
        lambda name: checkpoint_module
        if name == "torch.utils.checkpoint"
        else None,
    )
    fake_torch = SimpleNamespace(nn=SimpleNamespace(Module=FakeModule))
    wrapped = model_common.make_two_input_checkpoint_wrapper(fake_torch, FakeModel())
    wrapped.enable_activation_checkpointing()

    wrapped("num", "cat")

    assert calls == [("num", "cat")]


def test_training_loss_passes_fold_training_row_indices() -> None:
    class IndexRecordingAdapter:
        def loss(self, model, x_num, x_cat, y, *, row_indices):
            self.row_indices = row_indices
            return "loss"

    adapter = IndexRecordingAdapter()
    indices = np.array([7, 3], dtype="int64")

    result = call_adapter_loss(
        adapter, object(), "num", "cat", "target", row_indices=indices
    )

    assert result == "loss"
    assert adapter.row_indices.tolist() == [7, 3]


def test_training_prepares_optional_retrieval_context() -> None:
    class ContextAdapter:
        def fit_context(self, batch):
            self.batch = batch

    adapter = ContextAdapter()
    batch = _batch("train", with_target=True)

    prepare_adapter_context(adapter, batch)

    assert adapter.batch is batch


def test_training_refreshes_optional_retrieval_cache() -> None:
    class CacheAdapter:
        def refresh_retrieval_cache(self, model, device):
            self.call = (model, device)

    adapter = CacheAdapter()
    model = object()

    refresh_retrieval_cache(adapter, model, "cuda")

    assert adapter.call == (model, "cuda")


def test_oom_retry_resumes_newly_completed_checkpoint(tmp_path: Path) -> None:
    request = _request()

    class CheckpointThenOOM(_RecordingBackend):
        def run_attempt(self, **kwargs: object) -> BackendAttemptResult:
            self.calls.append(dict(kwargs))
            if len(self.calls) == 1:
                output_dir = Path(kwargs["output_dir"])
                (output_dir / "checkpoint.pt").write_bytes(b"checkpoint")
                (output_dir / "checkpoint_meta.json").write_text(
                    json.dumps(
                        {
                            "candidate_id": request.candidate_id,
                            "epoch": 3,
                            "checkpoint": "checkpoint.pt",
                        }
                    ),
                    encoding="utf-8",
                )
                raise RuntimeError("CUDA out of memory")
            output_dir = Path(kwargs["output_dir"])
            checkpoint = output_dir / "checkpoint.pt"
            return BackendAttemptResult(
                best_epoch=11,
                best_brier=0.24,
                checkpoint=checkpoint,
                predictions=np.array([0.4, 0.6], dtype="float64"),
            )

    backend = CheckpointThenOOM()

    result = fit_candidate(request, _FakeAdapter(), tmp_path, backend=backend)

    assert result.started_epoch == 0
    assert [call["resume_epoch"] for call in backend.calls] == [0, 4]
