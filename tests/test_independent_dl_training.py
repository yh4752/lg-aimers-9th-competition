from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.independent_dl.features import FeatureBatch
from experiments.independent_dl.models.ft_transformer import ft_transformer_kwargs
from experiments.independent_dl.models.mlp_resnet import mlp_resnet_kwargs
from experiments.independent_dl.models.common import quantile_bin_edges
from experiments.independent_dl.models import common as model_common
from experiments.independent_dl.models.common import attach_progress_reporter
from experiments.independent_dl.models.tabm import SUPPORTED_NUM_EMBEDDINGS
from experiments.independent_dl.progress import ProgressReporter
from experiments.independent_dl.training import (
    BackendAttemptResult,
    TrainRequest,
    call_adapter_loss,
    fit_candidate,
    inspect_cuda_hardware,
    prepare_adapter_context,
    refresh_retrieval_cache,
    session_deadline_reached,
    TrainingTimeBudgetReached,
    enforce_session_deadline,
    progress_message,
    require_finite_validation_brier,
    load_resume_payload,
    report_training_window,
    TorchTrainingBackend,
    adapter_checkpoint_state,
    configure_adapter_output,
    on_adapter_epoch_end,
    on_adapter_epoch_start,
    prepare_initial_retrieval_cache,
    restore_adapter_checkpoint_state,
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


def test_backend_deadline_outcome_is_preserved_in_train_result(tmp_path: Path) -> None:
    class DeadlineBackend(_RecordingBackend):
        def run_attempt(self, **kwargs: object) -> BackendAttemptResult:
            base = super().run_attempt(**kwargs)
            return BackendAttemptResult(
                best_epoch=base.best_epoch,
                best_brier=base.best_brier,
                checkpoint=base.checkpoint,
                predictions=base.predictions,
                completed_epochs=3,
                budget_reached=True,
            )

    result = fit_candidate(_request(), _FakeAdapter(), tmp_path, backend=DeadlineBackend())
    assert result.budget_reached is True


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


def test_resume_checkpoint_is_loaded_on_cpu_before_rng_restore(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    class FakeTorch:
        @staticmethod
        def load(path, *, map_location, weights_only):
            assert path == checkpoint
            assert map_location == "cpu"
            assert weights_only is False
            return {"torch_rng": "cpu-byte-tensor"}

    assert load_resume_payload(FakeTorch(), checkpoint) == {
        "torch_rng": "cpu-byte-tensor"
    }


def test_resume_rejects_changed_campaign_checkpoint_binding(tmp_path: Path) -> None:
    request = _request()
    request = TrainRequest(
        **{
            **request.__dict__,
            "checkpoint_binding": {"config_sha256": "a" * 64, "cache_sha256": "b" * 64},
        }
    )
    (tmp_path / "checkpoint.pt").write_bytes(b"checkpoint")
    (tmp_path / "checkpoint_meta.json").write_text(
        json.dumps(
            {
                "candidate_id": request.candidate_id,
                "epoch": 1,
                "checkpoint": "checkpoint.pt",
                "checkpoint_binding": {"config_sha256": "c" * 64, "cache_sha256": "b" * 64},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="binding"):
        fit_candidate(request, _FakeAdapter(), tmp_path, backend=_RecordingBackend())


def test_expired_session_deadline_stops_at_a_safe_boundary() -> None:
    with pytest.raises(TrainingTimeBudgetReached, match="session time budget"):
        enforce_session_deadline(time.time() - 1, boundary="epoch_3_batch_20")


def test_deadline_probe_allows_backend_to_finish_from_last_complete_epoch() -> None:
    assert session_deadline_reached(time.time() - 1) is True
    assert session_deadline_reached(time.time() + 60) is False
    assert session_deadline_reached(None) is False


def test_progress_message_exposes_job_epoch_batch_and_eta() -> None:
    message = progress_message(
        candidate_id="a__tabm_p3__dl_standard__tr2019__va2020__s42",
        epoch=2,
        epochs=240,
        batch=10,
        batches=100,
        elapsed_seconds=120.0,
    )

    assert "TRAINING_PROGRESS" in message
    assert "epoch=3/240" in message
    assert "batch=10/100" in message
    assert "epoch_eta_seconds=1080" in message


def test_nonfinite_validation_brier_fails_before_checkpointing() -> None:
    with pytest.raises(RuntimeError, match="non-finite validation Brier"):
        require_finite_validation_brier(
            candidate_id="s3__tabm__id_frequency_and_oov__tr2023__va2024__s42",
            epoch=0,
            brier=float("nan"),
        )


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


@pytest.mark.parametrize("family", ["tabm", "mlp_resnet", "ft_transformer", "tabr"])
def test_fit_candidate_wires_append_only_reporter_for_every_family(
    tmp_path: Path, family: str
) -> None:
    request = _request()
    request = TrainRequest(
        candidate_id=f"{family}__raw_typed__p1__s42",
        family=family,
        seed=request.seed,
        epochs=request.epochs,
        model_config=request.model_config,
        training_config=request.training_config,
        train=request.train,
        valid=request.valid,
    )
    backend = _RecordingBackend()

    fit_candidate(request, _FakeAdapter(), tmp_path / family, backend=backend)

    reporter = backend.calls[0]["reporter"]
    assert isinstance(reporter, ProgressReporter)
    assert reporter.family == family
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / family / "progress.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events == ["CANDIDATE_COMPLETED"]


def test_fit_candidate_logs_failure_and_preserves_original_exception(
    tmp_path: Path,
) -> None:
    class FailingBackend:
        def run_attempt(self, **kwargs: object) -> BackendAttemptResult:
            raise KeyError("original failure")

    with pytest.raises(KeyError, match="original failure"):
        fit_candidate(_request(), _FakeAdapter(), tmp_path, backend=FailingBackend())

    event = json.loads((tmp_path / "progress.jsonl").read_text(encoding="utf-8"))
    assert event["event"] == "CANDIDATE_FAILED"
    assert event["error_type"] == "KeyError"
    assert "original failure" in event["error_message"]
    assert "Traceback" in event["traceback"]


def test_optional_adapter_receives_the_shared_reporter(tmp_path: Path) -> None:
    class Adapter:
        def set_progress_reporter(self, reporter: object) -> None:
            self.reporter = reporter

    adapter = Adapter()
    reporter = ProgressReporter(
        "tabr__raw_typed__p1__s42",
        "tabr",
        "2023->2024",
        tmp_path,
    )

    attach_progress_reporter(adapter, reporter)

    assert adapter.reporter is reporter


def test_training_window_reports_loss_throughput_eta_and_gpu_memory() -> None:
    class Reporter:
        def should_emit(self, *, completed_rows: int, stream: str) -> bool:
            assert completed_rows == 128
            assert stream == "training_epoch_2"
            return True

        def progress(self, event: str, **fields: object) -> None:
            self.call = (event, fields)

    class Cuda:
        @staticmethod
        def synchronize() -> None:
            pass

        @staticmethod
        def memory_allocated() -> int:
            return 10

        @staticmethod
        def memory_reserved() -> int:
            return 20

        @staticmethod
        def max_memory_allocated() -> int:
            return 30

    reporter = Reporter()

    report_training_window(
        reporter=reporter,
        torch=SimpleNamespace(cuda=Cuda()),
        completed_rows=128,
        total_rows=512,
        started_at=time.monotonic() - 2.0,
        epoch=2,
        epochs=20,
        batch=4,
        batches=16,
        loss=0.625,
    )

    event, fields = reporter.call
    assert event == "TRAINING_PROGRESS"
    assert fields["epoch"] == 2
    assert fields["batch"] == 4
    assert fields["loss"] == 0.625
    assert fields["gpu"] == {
        "allocated_bytes": 10,
        "reserved_bytes": 20,
        "peak_bytes": 30,
    }


def test_prediction_reports_completed_validation_chunks() -> None:
    torch = pytest.importorskip("torch")

    class Adapter:
        @staticmethod
        def probabilities(model, x_num, x_cat):
            return torch.full((len(x_num),), 0.5, device=x_num.device)

    class Model:
        @staticmethod
        def eval() -> None:
            pass

    class Reporter:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def should_emit(self, *, completed_rows: int, stream: str) -> bool:
            assert stream == "validation"
            return True

        def progress(self, event: str, **fields: object) -> None:
            self.calls.append((event, fields))

    batch = FeatureBatch(
        row_id=np.array(["a", "b", "c", "d", "e"]),
        season=np.full(5, 2024, dtype="int64"),
        game_type=np.array(["R"] * 5),
        x_num=np.zeros((5, 2), dtype="float32"),
        x_cat=np.zeros((5, 1), dtype="int64"),
        y=np.zeros(5, dtype="float32"),
    )
    reporter = Reporter()

    result = TorchTrainingBackend._predict(
        torch,
        batch,
        Model(),
        Adapter(),
        micro_batch_size=2,
        amp_enabled=False,
        device="cpu",
        reporter=reporter,
    )

    assert result.tolist() == [0.5] * 5
    assert [call[1]["completed_rows"] for call in reporter.calls] == [2, 4, 5]
    assert all(call[0] == "VALIDATION_PROGRESS" for call in reporter.calls)


def test_optional_adapter_lifecycle_binds_output_epoch_and_checkpoint_state(
    tmp_path: Path,
) -> None:
    class Adapter:
        def set_output_dir(self, output_dir: Path) -> None:
            self.output_dir = output_dir

        def prepare_initial_retrieval_cache(self, model, device) -> None:
            self.initial = (model, device)

        def on_epoch_start(self, model, epoch, device) -> None:
            self.started = (model, epoch, device)

        def on_epoch_end(self, model, epoch, device) -> None:
            self.ended = (model, epoch, device)

        def checkpoint_state(self):
            return {"search": "frozen"}

        def restore_checkpoint_state(self, payload, model, device):
            self.restored = (payload, model, device)
            return True

    adapter = Adapter()
    model = object()

    configure_adapter_output(adapter, tmp_path)
    prepare_initial_retrieval_cache(adapter, model, "cuda")
    on_adapter_epoch_start(adapter, model, 1, "cuda")
    on_adapter_epoch_end(adapter, model, 1, "cuda")
    state = adapter_checkpoint_state(adapter)
    restored = restore_adapter_checkpoint_state(
        adapter, state, model, "cuda"
    )

    assert adapter.output_dir == tmp_path
    assert adapter.initial == (model, "cuda")
    assert adapter.started == (model, 1, "cuda")
    assert adapter.ended == (model, 1, "cuda")
    assert state == {"search": "frozen"}
    assert restored is True
    assert adapter.restored == (state, model, "cuda")
