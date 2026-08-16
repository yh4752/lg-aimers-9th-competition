from __future__ import annotations

import json
import math
import sys
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from experiments.tabm_campaign import worker as worker_module
from experiments.tabm_campaign import row_feature_proxy as proxy_module
from experiments.tabm_campaign.artifacts import (
    ArtifactError,
    StageEvidence,
    write_stage_bundles,
)
from experiments.tabm_campaign.runner import CampaignJob, CampaignJobResult
from experiments.tabm_campaign.row_feature_contracts import (
    load_row_feature_proxy_contract,
)
from experiments.tabm_campaign.row_feature_proxy import (
    _code_file_paths,
    _code_sha256,
    _bundle_members,
    _decision,
    _valid_finite_tensor,
    _validate_checkpoint_payload,
    RowFeatureProxyError,
    build_proxy_jobs,
    run_row_feature_proxy,
)


CONTRACT = load_row_feature_proxy_contract()


def _write_training_checkpoint(
    path: Path,
    job: CampaignJob,
    *,
    epoch: int = 2,
    missing_key: str | None = None,
) -> None:
    import random

    import numpy as np
    import torch

    n_num = 2
    n_bins = 2
    first_width = n_num * 32 + 3
    generator = torch.Generator(device="cpu").manual_seed(7)
    shared = torch.rand(job.width * job.width, generator=generator)

    def compact(shape: tuple[int, ...]):
        size = math.prod(shape)
        return shared[:size].view(shape)

    model = {
        "model.num_module.linear0.weight": compact((n_num, 32)),
        "model.num_module.linear0.bias": compact((n_num, 32)),
        "model.num_module.impl.weight": compact((n_num, n_bins)),
        "model.num_module.impl.bias": compact((n_num, n_bins)),
        "model.num_module.linear.weight": compact((n_num, n_bins, 32)),
    }
    parameter_names = [
        "model.num_module.linear0.weight",
        "model.num_module.linear0.bias",
        "model.num_module.linear.weight",
    ]
    for block in range(job.blocks):
        block_input = first_width if block == 0 else job.width
        prefix = f"model.backbone.blocks.{block}.0"
        model.update(
            {
                f"{prefix}.weight": compact((job.width, block_input)),
                f"{prefix}.r": compact((job.k, block_input)),
                f"{prefix}.s": compact((job.k, job.width)),
                f"{prefix}.bias": compact((job.k, job.width)),
            }
        )
        parameter_names.extend(
            f"{prefix}.{suffix}" for suffix in ("weight", "r", "s", "bias")
        )
    model.update(
        {
            "model.output.weight": compact((job.k, job.width, 1)),
            "model.output.bias": compact((job.k, 1)),
        }
    )
    parameter_names.extend(("model.output.weight", "model.output.bias"))
    optimizer = {
        "state": {
            index: {
                "step": torch.tensor(3.0),
                "exp_avg": compact(tuple(model[name].shape)),
                "exp_avg_sq": compact(tuple(model[name].shape)),
            }
            for index, name in enumerate(parameter_names)
        },
        "param_groups": [
            {
                "lr": job.learning_rate,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0001,
                "amsgrad": False,
                "maximize": False,
                "foreach": None,
                "capturable": False,
                "differentiable": False,
                "fused": None,
                "decoupled_weight_decay": True,
                "params": list(range(len(parameter_names))),
            }
        ],
    }
    scheduler = {
        "factor": 0.1,
        "default_min_lr": 0,
        "min_lrs": [0],
        "patience": 10,
        "cooldown": 0,
        "cooldown_counter": 0,
        "mode": "min",
        "threshold": 0.0001,
        "threshold_mode": "rel",
        "eps": 1e-8,
        "last_epoch": 3,
        "_last_lr": [job.learning_rate],
        "mode_worse": float("inf"),
        "best": 0.04,
        "num_bad_epochs": 1,
    }
    payload = {
        "candidate_id": job.candidate_id,
        "epoch": epoch,
        "best_epoch": 1,
        "best_brier": 0.04,
        "validation_curve": [(0, 0.08), (1, 0.04), (2, 0.05)],
        "validation_time_curve": [(0, 1.0, 0.08), (1, 2.0, 0.04), (2, 3.0, 0.05)],
        "elapsed_seconds": 3.0,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": {
            "scale": 65536.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "_growth_tracker": 3,
        },
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": [
            torch.tensor(
                list(
                    (42).to_bytes(8, byteorder=sys.byteorder, signed=False)
                    + (4).to_bytes(8, byteorder=sys.byteorder, signed=True)
                ),
                dtype=torch.uint8,
            )
        ],
        "adapter_state": None,
    }
    if missing_key is not None:
        payload.pop(missing_key)
    torch.save(payload, path)


class _Runtime:
    def __init__(self, statuses: dict[str, str] | None = None) -> None:
        self.statuses = statuses or {}
        self.calls: list[tuple[str, tuple[object, ...], int, float]] = []
        self.state_snapshots: list[bytes] = []
        self.preexisting_checkpoints: list[str] = []

    def run_jobs(self, version, jobs, output_dir, *, gpu_count, job_deadline):
        self.calls.append((version, jobs, gpu_count, job_deadline))
        state_path = output_dir.parent / "stage_state.json"
        if state_path.is_file():
            self.state_snapshots.append(state_path.read_bytes())
        job = jobs[0]
        job_dir = output_dir / job.candidate_id
        job_dir.mkdir(parents=True, exist_ok=True)
        if (job_dir / "checkpoint.pt").is_file():
            self.preexisting_checkpoints.append(job.candidate_id)
        status = self.statuses.get(job.candidate_id, "completed")
        checkpoint = None
        predictions = None
        brier = None
        cache_digest = "c" * 64
        if status in {"completed", "inconclusive"}:
            _write_training_checkpoint(job_dir / "checkpoint.pt", job)
            import torch

            torch.save(
                {"model": {"weight": torch.tensor([1.0])}, "epoch": 1},
                job_dir / "best_checkpoint.pt",
            )
            checkpoint = job_dir / "best_checkpoint.pt"
            worker_module._atomic_json(
                job_dir / "checkpoint_meta.json",
                {
                    "candidate_id": job.candidate_id,
                    "epoch": 2,
                    "checkpoint": "checkpoint.pt",
                    "adapter_state": None,
                    "checkpoint_binding": {
                        "config_sha256": worker_module._job_sha(job),
                        "cache_sha256": cache_digest,
                        "training_source_sha256": worker_module._training_source_sha256(),
                    },
                },
            )
        if status == "completed":
            predictions = job_dir / "predictions.csv"
            predictions.write_text("row_id,target,probability\nr1,0,0.2\nr2,1,0.8\n", encoding="utf-8")
            brier = 0.04
        worker_module._atomic_json(job_dir / "job.json", worker_module._job_payload(job))
        (job_dir / "worker.log").write_text(f"{status}\n", encoding="utf-8")
        resource_evidence: dict[str, object] = {
            "gpu": 0,
            "devices": ({"index": 0, "name": "fixture"},),
            "training_device_indices": (0,),
        }
        if checkpoint is not None:
            resource_evidence.update(
                {
                    "cache_digest": cache_digest,
                    **worker_module._current_code_provenance(
                        worker_module._training_source_sha256()
                    ),
                }
            )
        if predictions is not None:
            resource_evidence.update(
                worker_module._artifact_hash_evidence(checkpoint, predictions)
            )
        result = CampaignJobResult(
            job.candidate_id,
            status,
            brier,
            1 if checkpoint else None,
            3 if checkpoint else 0,
            checkpoint,
            predictions,
            resource_evidence,
            None if status == "completed" else status,
        )
        worker_module._atomic_json(
            job_dir / "worker_result.json",
            worker_module._serialize_result(result, worker_module._job_sha(job)),
        )
        return (result,)


class _BudgetRuntime(_Runtime):
    def run_jobs(self, version, jobs, output_dir, *, gpu_count, job_deadline):
        completed = super().run_jobs(
            version,
            jobs,
            output_dir,
            gpu_count=gpu_count,
            job_deadline=job_deadline,
        )[0]
        result = CampaignJobResult(
                completed.candidate_id,
                "inconclusive",
                completed.brier,
                completed.best_epoch,
                completed.completed_epochs,
                completed.checkpoint,
                completed.predictions_path,
                completed.resource_evidence,
                "budget_reached",
        )
        job = jobs[0]
        worker_module._atomic_json(
            output_dir / job.candidate_id / "worker_result.json",
            worker_module._serialize_result(result, worker_module._job_sha(job)),
        )
        return (result,)


class _InvalidCompletedRuntime(_Runtime):
    def __init__(self, corruption: str) -> None:
        super().__init__()
        self.corruption = corruption

    def run_jobs(self, version, jobs, output_dir, *, gpu_count, job_deadline):
        result = super().run_jobs(
            version,
            jobs,
            output_dir,
            gpu_count=gpu_count,
            job_deadline=job_deadline,
        )[0]
        job = jobs[0]
        job_dir = output_dir / job.candidate_id
        invalid = result
        if self.corruption == "checkpoint_meta":
            (job_dir / "checkpoint_meta.json").write_text("{}", encoding="utf-8")
        elif self.corruption == "brier":
            invalid = CampaignJobResult(
                result.candidate_id,
                result.status,
                0.2,
                result.best_epoch,
                result.completed_epochs,
                result.checkpoint,
                result.predictions_path,
                result.resource_evidence,
                result.failure,
            )
        elif self.corruption.startswith("prediction_"):
            prediction_values = {
                "prediction_schema": "row_id,target\nr1,0\n",
                "prediction_nonfinite": "row_id,target,probability\nr1,0,nan\n",
                "prediction_duplicate": (
                    "row_id,target,probability\nr1,0,0.2\nr1,1,0.8\n"
                ),
                "prediction_target": "row_id,target,probability\nr1,2,0.2\n",
            }
            result.predictions_path.write_text(
                prediction_values[self.corruption], encoding="utf-8"
            )
            resource = dict(result.resource_evidence)
            resource["predictions_sha256"] = sha256(
                result.predictions_path.read_bytes()
            ).hexdigest()
            invalid = CampaignJobResult(
                result.candidate_id,
                result.status,
                result.brier,
                result.best_epoch,
                result.completed_epochs,
                result.checkpoint,
                result.predictions_path,
                resource,
                result.failure,
            )
        worker_module._atomic_json(
            job_dir / "worker_result.json",
            worker_module._serialize_result(invalid, worker_module._job_sha(job)),
        )
        if self.corruption == "worker_status":
            payload = json.loads((job_dir / "worker_result.json").read_bytes())
            payload["status"] = "failed"
            worker_module._atomic_json(job_dir / "worker_result.json", payload)
        return (invalid,)


class _DeadlineFallbackRuntime(_Runtime):
    def run_jobs(self, version, jobs, output_dir, *, gpu_count, job_deadline):
        completed = super().run_jobs(
            version,
            jobs,
            output_dir,
            gpu_count=gpu_count,
            job_deadline=job_deadline,
        )[0]
        job = jobs[0]
        fallback = CampaignJobResult(
            job.candidate_id,
            "inconclusive",
            None,
            None,
            completed.completed_epochs,
            completed.checkpoint,
            None,
            {},
            "worker_exceeded_deadline_grace",
        )
        worker_module._atomic_json(
            output_dir / job.candidate_id / "worker_result.json",
            worker_module._serialize_result(fallback, worker_module._job_sha(job)),
        )
        return (fallback,)


class _CorruptCheckpointRuntime(_DeadlineFallbackRuntime):
    def __init__(self, corruption: str) -> None:
        super().__init__()
        self.corruption = corruption

    def run_jobs(self, version, jobs, output_dir, *, gpu_count, job_deadline):
        result = super().run_jobs(
            version,
            jobs,
            output_dir,
            gpu_count=gpu_count,
            job_deadline=job_deadline,
        )[0]
        job = jobs[0]
        job_dir = output_dir / job.candidate_id
        if self.corruption == "damaged":
            (job_dir / "checkpoint.pt").write_bytes(b"not-a-checkpoint")
        elif self.corruption == "missing_optimizer":
            _write_training_checkpoint(
                job_dir / "checkpoint.pt",
                job,
                missing_key="optimizer",
            )
        elif self.corruption == "epoch_mismatch":
            meta = json.loads((job_dir / "checkpoint_meta.json").read_bytes())
            meta["epoch"] = 1
            worker_module._atomic_json(job_dir / "checkpoint_meta.json", meta)
        elif self.corruption == "malformed_curve":
            import torch

            checkpoint = torch.load(
                job_dir / "checkpoint.pt", map_location="cpu", weights_only=False
            )
            checkpoint["validation_curve"] = [(), (1, 0.04), (2, 0.05)]
            torch.save(checkpoint, job_dir / "checkpoint.pt")
        elif self.corruption in {
            "bogus_model_key",
            "wrong_model_shape",
            "empty_optimizer",
            "mismatched_optimizer",
            "unrelated_scheduler",
            "bad_scaler",
            "bad_python_rng",
            "bad_numpy_rng",
            "bad_torch_rng",
            "bad_cuda_rng",
            "bad_cuda_rng_offset",
        }:
            import torch

            checkpoint = torch.load(
                job_dir / "checkpoint.pt", map_location="cpu", weights_only=False
            )
            if self.corruption == "bogus_model_key":
                checkpoint["model"]["bogus.weight"] = torch.ones(1)
            elif self.corruption == "wrong_model_shape":
                checkpoint["model"][
                    "model.backbone.blocks.1.0.weight"
                ] = torch.ones(1)
            elif self.corruption == "empty_optimizer":
                checkpoint["optimizer"] = {"state": {}, "param_groups": []}
            elif self.corruption == "mismatched_optimizer":
                checkpoint["optimizer"]["param_groups"][0]["params"] = [0]
            else:
                if self.corruption == "unrelated_scheduler":
                    checkpoint["scheduler"] = {"totally": "unrelated"}
                elif self.corruption == "bad_scaler":
                    checkpoint["scaler"] = {}
                elif self.corruption == "bad_python_rng":
                    checkpoint["python_rng"] = (1, 2)
                elif self.corruption == "bad_numpy_rng":
                    checkpoint["numpy_rng"] = ("bad",)
                elif self.corruption == "bad_torch_rng":
                    checkpoint["torch_rng"] = torch.zeros(2, dtype=torch.float32)
                elif self.corruption == "bad_cuda_rng":
                    checkpoint["cuda_rng"] = [torch.zeros(12, dtype=torch.uint8)]
                else:
                    checkpoint["cuda_rng"] = [
                        torch.tensor(
                            list(
                                (42).to_bytes(
                                    8, byteorder=sys.byteorder, signed=False
                                )
                                + (2).to_bytes(
                                    8, byteorder=sys.byteorder, signed=True
                                )
                            ),
                            dtype=torch.uint8,
                        )
                    ]
            torch.save(checkpoint, job_dir / "checkpoint.pt")
        return (result,)


def _official_data(tmp_path: Path, monkeypatch) -> Path:
    data = tmp_path / "official"
    data.mkdir()
    train = data / "train.csv"
    train.write_text("tiny fixture\n", encoding="utf-8")
    monkeypatch.setattr(
        "experiments.tabm_campaign.row_feature_proxy._official_train_file_sha256",
        lambda path: CONTRACT.official_train_sha256,
    )
    return data


def _forge_consistent_resume(
    source: Path,
    target: Path,
    *,
    remove: str | None = None,
    extra: tuple[str, bytes] | None = None,
) -> Path:
    with ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members.pop("manifest.json"))
    if remove is not None:
        members.pop(remove)
        manifest["members"].pop(remove)
    if extra is not None:
        name, value = extra
        members[name] = value
        manifest["members"][name] = sha256(value).hexdigest()
    members["manifest.json"] = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    return target


def test_build_proxy_jobs_has_exact_paired_contract_order_and_fields() -> None:
    jobs = build_proxy_jobs(CONTRACT)

    assert [job.candidate_id for job in jobs] == [
        "rfp__baseline__s42",
        "rfp__baseline__s3407",
        *(
            f"rfp__{bundle}__s{seed}"
            for bundle in CONTRACT.feature_bundles
            for seed in CONTRACT.seeds
        ),
    ]
    assert len(jobs) == 14
    for job in jobs:
        assert (
            job.capacity,
            job.k,
            job.width,
            job.blocks,
            job.dropout,
            job.num_embedding,
            job.loss,
            job.scheduler,
            job.learning_rate,
        ) == (
            CONTRACT.baseline.capacity,
            CONTRACT.baseline.k,
            CONTRACT.baseline.width,
            CONTRACT.baseline.blocks,
            CONTRACT.baseline.dropout,
            CONTRACT.baseline.num_embedding,
            CONTRACT.baseline.loss,
            CONTRACT.baseline.scheduler,
            CONTRACT.baseline.learning_rate,
        )
        assert (job.train_end_year, job.valid_year, job.sample_mode) == (2023, 2024, "proxy")
        assert (job.max_epochs, job.min_epochs, job.patience) == (8, 3, 3)


@pytest.mark.parametrize(
    "corruption",
    [
        "checkpoint_meta",
        "brier",
        "worker_status",
        "prediction_schema",
        "prediction_nonfinite",
        "prediction_duplicate",
        "prediction_target",
    ],
)
def test_completed_result_rejects_semantically_corrupt_evidence(
    tmp_path: Path, monkeypatch, corruption: str
) -> None:
    data = _official_data(tmp_path, monkeypatch)

    with pytest.raises(RowFeatureProxyError, match="completed evidence"):
        run_row_feature_proxy(
            data_dir=data,
            output_dir=tmp_path / "out",
            runtime=_InvalidCompletedRuntime(corruption),
            wall_deadline=10_000.0,
            now=lambda: 1_000.0,
        )


def test_wall_deadline_cannot_exceed_sealed_budget(tmp_path: Path, monkeypatch) -> None:
    data = _official_data(tmp_path, monkeypatch)

    run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "exact",
        runtime=_Runtime({"rfp__baseline__s42": "failed"}),
        wall_deadline=11_800.0,
        now=lambda: 1_000.0,
    )
    with pytest.raises(RowFeatureProxyError, match="wall_seconds"):
        run_row_feature_proxy(
            data_dir=data,
            output_dir=tmp_path / "too-far",
            runtime=_Runtime(),
            wall_deadline=11_800.000001,
            now=lambda: 1_000.0,
        )


def test_code_binding_covers_only_the_sealed_runtime_file_set(tmp_path: Path) -> None:
    production = {path.as_posix() for path in _code_file_paths()}
    assert any(path.endswith("independent_dl/progress.py") for path in production)
    assert any(
        path.endswith("independent_dl/feature_sources/trackman.py")
        for path in production
    )

    source_root = Path(proxy_module.__file__).resolve().parents[1]
    root = tmp_path / "experiments"
    for source in _code_file_paths():
        destination = root / source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    initial = _code_sha256(root)
    added = root / "independent_dl" / "new_module.py"
    added.write_text("NEW = 1\n", encoding="utf-8")
    assert _code_sha256(root) == initial
    added.unlink()
    listed = root / "independent_dl" / "row_features.py"
    listed.write_bytes(listed.read_bytes() + b"\n# changed\n")
    assert _code_sha256(root) != initial


def test_run_is_sequential_guarded_and_publishes_exact_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    runtime = _Runtime()
    output = tmp_path / "out"

    run = run_row_feature_proxy(
        data_dir=data,
        output_dir=output,
        runtime=runtime,
        gpu_count=1,
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )

    assert [call[1][0].candidate_id for call in runtime.calls] == [
        job.candidate_id for job in build_proxy_jobs(CONTRACT)
    ]
    assert all(call[0] == "P" and call[2] == 1 and call[3] == 9_100.0 for call in runtime.calls)
    assert [
        sum(row["status"] == "completed" for row in json.loads(value)["results"])
        for value in runtime.state_snapshots
    ] == list(range(14))
    assert run.completed == tuple(job.candidate_id for job in build_proxy_jobs(CONTRACT))
    assert run.failed == () and run.inconclusive == ()
    assert run.decision.status == "complete"
    assert run.state_path == output / "stage_state.json"
    assert run.state_path.is_file()
    assert run.bundles.review.name == "tabm_row_feature_stage_P_review_bundle.zip"
    assert run.bundles.resume is not None
    assert run.bundles.resume.name == "tabm_row_feature_stage_P_resume_bundle.zip"

    with ZipFile(run.bundles.review) as archive:
        review_names = set(archive.namelist())
    assert review_names == {
        "manifest.json",
        "config/row_feature_proxy_v1.json",
        "metrics/job_results.json",
        "decisions/proxy_decision.json",
        "logs/stage.log",
        "state/stage_state.json",
        *(f"predictions/{job.candidate_id}.csv" for job in build_proxy_jobs(CONTRACT)),
    }
    assert not any(
        forbidden in name.lower()
        for name in review_names
        for forbidden in ("submission", "test_predictions", "evaluation")
    )


def test_inconclusive_stops_session_and_marks_remainder_not_started(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    jobs = build_proxy_jobs(CONTRACT)
    runtime = _Runtime({jobs[2].candidate_id: "inconclusive"})

    run = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "out",
        runtime=runtime,
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )

    assert [call[1][0].candidate_id for call in runtime.calls] == [
        job.candidate_id for job in jobs[:3]
    ]
    assert run.inconclusive == tuple(job.candidate_id for job in jobs[2:])
    assert run.decision.status == "incomplete"
    with ZipFile(run.bundles.review) as archive:
        assert "decisions/proxy_decision.json" not in archive.namelist()
        state = json.loads(archive.read("state/stage_state.json"))
    assert state["results"][2]["disposition"] == "inconclusive"
    assert all(row["disposition"] == "not_started" for row in state["results"][3:])


def test_budget_inconclusive_keeps_metric_but_bundles_only_training_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    jobs = build_proxy_jobs(CONTRACT)

    run = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "out",
        runtime=_BudgetRuntime(),
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )

    with ZipFile(run.bundles.resume) as archive:
        names = set(archive.namelist())
        state = json.loads(archive.read("state/stage_state.json"))
    assert state["results"][0]["brier"] == 0.04
    assert f"predictions/{jobs[0].candidate_id}.csv" not in names
    assert f"jobs/{jobs[0].candidate_id}/predictions.csv" not in names
    assert f"jobs/{jobs[0].candidate_id}/checkpoint.pt" in names

    resumed_runtime = _Runtime()
    run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "resumed",
        resume_bundle=run.bundles.resume,
        runtime=resumed_runtime,
        wall_deadline=12_000.0,
        now=lambda: 2_000.0,
    )
    assert resumed_runtime.preexisting_checkpoints[0] == jobs[0].candidate_id


def test_nested_optional_history_is_hash_bound_to_resume(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    nested = data / "nested"
    nested.mkdir()
    history = nested / "trackman_history.csv"
    history.write_bytes(b"pitcher_id,season\np1,2023\n")

    run = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "out",
        runtime=_BudgetRuntime(),
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )
    with ZipFile(run.bundles.resume) as archive:
        state = json.loads(archive.read("state/stage_state.json"))
    assert state["trackman_history_sha256"] == sha256(history.read_bytes()).hexdigest()

    history.write_bytes(b"pitcher_id,season\np2,2023\n")
    with pytest.raises(RowFeatureProxyError, match="trackman_history_sha256"):
        run_row_feature_proxy(
            data_dir=data,
            output_dir=tmp_path / "resume",
            resume_bundle=run.bundles.resume,
            runtime=_BudgetRuntime(),
            wall_deadline=12_000.0,
            now=lambda: 2_000.0,
        )


def test_post_training_bundle_streaming_stops_at_absolute_deadline(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    packaging = [False]
    ticks = [0]

    class Runtime(_BudgetRuntime):
        def run_jobs(self, *args, **kwargs):
            result = super().run_jobs(*args, **kwargs)
            packaging[0] = True
            return result

    def advancing_clock() -> float:
        if not packaging[0]:
            return 1_000.0
        ticks[0] += 1
        return 1_000.0 if ticks[0] < 5 else 10_000.0

    output = tmp_path / "out"
    with pytest.raises(TimeoutError, match="deadline"):
        run_row_feature_proxy(
            data_dir=data,
            output_dir=output,
            runtime=Runtime(),
            wall_deadline=10_000.0,
            now=advancing_clock,
        )
    assert not list(output.glob("tabm_row_feature_stage_P_*_bundle.zip"))
    assert not list(output.glob(".tabm_row_feature_stage_P_*"))


def test_deadline_grace_fallback_is_bound_bundled_and_resumable(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    jobs = build_proxy_jobs(CONTRACT)
    first = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "first",
        runtime=_DeadlineFallbackRuntime(),
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )

    assert first.inconclusive == tuple(job.candidate_id for job in jobs)
    with ZipFile(first.bundles.resume) as archive:
        names = set(archive.namelist())
    assert f"jobs/{jobs[0].candidate_id}/checkpoint.pt" in names
    assert f"jobs/{jobs[0].candidate_id}/checkpoint_meta.json" in names

    resumed_runtime = _Runtime()
    run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "second",
        resume_bundle=first.bundles.resume,
        runtime=resumed_runtime,
        wall_deadline=12_000.0,
        now=lambda: 2_000.0,
    )
    assert resumed_runtime.preexisting_checkpoints[0] == jobs[0].candidate_id


def test_checkpoint_validator_accepts_actual_stage_p_cpu_states(tmp_path: Path) -> None:
    import numpy as np
    import torch

    from experiments.independent_dl.models.common import ModelMetadata
    from experiments.independent_dl.models.tabm import TabMAdapter

    job = build_proxy_jobs(CONTRACT)[0]
    path = tmp_path / "checkpoint.pt"
    _write_training_checkpoint(path, job)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = ModelMetadata(
        n_num_features=2,
        categorical_cardinalities=(3,),
        train_x_num=None,
        piecewise_bin_edges=(
            np.array((-1.0, 0.0, 1.0), dtype="float32"),
            np.array((-2.0, -1.0, 0.0, 2.0), dtype="float32"),
        ),
    )
    adapter = TabMAdapter(loss_name=job.loss)
    model = adapter.build(
        {
            "architecture": "tabm",
            "k": job.k,
            "width": job.width,
            "blocks": job.blocks,
            "dropout": job.dropout,
            "num_embedding": job.num_embedding,
        },
        metadata,
        "cpu",
    )
    optimizer = adapter.optimizer(
        model,
        {"learning_rate": job.learning_rate, "weight_decay": 0.0001},
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    for brier in (0.08, 0.04, 0.05):
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step(brier)
    payload.update(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
    )
    torch.save(payload, path)
    result = CampaignJobResult(
        job.candidate_id,
        "inconclusive",
        None,
        None,
        3,
        path,
        None,
        {},
        "worker_exceeded_deadline_grace",
    )

    _validate_checkpoint_payload(
        job,
        result,
        path,
        meta_epoch=2,
        meta_adapter_state=None,
    )

    payload["cuda_rng"] = [
        torch.tensor(
            list((42).to_bytes(8, byteorder=sys.byteorder, signed=False)),
            dtype=torch.uint8,
        )
    ]
    torch.save(payload, path)
    _validate_checkpoint_payload(
        job,
        result,
        path,
        meta_epoch=2,
        meta_adapter_state=None,
    )


def test_checkpoint_scheduler_uses_plateau_threshold_semantics(tmp_path: Path) -> None:
    import torch

    job = build_proxy_jobs(CONTRACT)[0]
    path = tmp_path / "checkpoint.pt"
    _write_training_checkpoint(path, job)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.update(
        {
            "best_epoch": 1,
            "best_brier": 0.079999,
            "validation_curve": [(0, 0.08), (1, 0.079999), (2, 0.081)],
            "validation_time_curve": [
                (0, 1.0, 0.08),
                (1, 2.0, 0.079999),
                (2, 3.0, 0.081),
            ],
        }
    )
    payload["scheduler"]["best"] = 0.08
    payload["scheduler"]["num_bad_epochs"] = 2
    torch.save(payload, path)
    result = CampaignJobResult(
        job.candidate_id,
        "inconclusive",
        None,
        None,
        3,
        path,
        None,
        {},
        "worker_exceeded_deadline_grace",
    )

    _validate_checkpoint_payload(
        job,
        result,
        path,
        meta_epoch=2,
        meta_adapter_state=None,
    )


@pytest.mark.parametrize("kind", ["expand", "as_strided"])
def test_checkpoint_tensor_rejects_tiny_overlapping_storage(kind: str) -> None:
    import torch

    base = torch.zeros(1, dtype=torch.float32)
    if kind == "expand":
        tensor = base.expand(4096, 4096)
    else:
        tensor = torch.as_strided(base, (4096, 4096), (0, 0))

    assert not _valid_finite_tensor(
        tensor,
        (4096, 4096),
        torch,
        dtype=torch.float32,
    )


def test_checkpoint_size_is_rejected_before_torch_load(
    tmp_path: Path, monkeypatch
) -> None:
    import torch

    job = build_proxy_jobs(CONTRACT)[0]
    path = tmp_path / "checkpoint.pt"
    _write_training_checkpoint(path, job)
    result = CampaignJobResult(
        job.candidate_id,
        "inconclusive",
        None,
        None,
        3,
        path,
        None,
        {},
        "worker_exceeded_deadline_grace",
    )
    monkeypatch.setattr(proxy_module, "_MAX_CHECKPOINT_BYTES", path.stat().st_size - 1)

    def forbidden_load(*args, **kwargs):
        raise AssertionError("oversized checkpoint must not reach torch.load")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(RowFeatureProxyError, match="size limit"):
        _validate_checkpoint_payload(
            job,
            result,
            path,
            meta_epoch=2,
            meta_adapter_state=None,
        )


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("damaged", "cannot be loaded"),
        ("missing_optimizer", "payload keys"),
        ("epoch_mismatch", "epoch binding"),
        ("malformed_curve", "validation curves"),
        ("bogus_model_key", "model keys"),
        ("wrong_model_shape", "tensor shape"),
        ("empty_optimizer", "parameter groups"),
        ("mismatched_optimizer", "configuration"),
        ("unrelated_scheduler", "scheduler keys"),
        ("bad_scaler", "scaler"),
        ("bad_python_rng", "RNG"),
        ("bad_numpy_rng", "RNG"),
        ("bad_torch_rng", "RNG"),
        ("bad_cuda_rng", "RNG"),
        ("bad_cuda_rng_offset", "RNG"),
    ],
)
def test_inconclusive_checkpoint_payload_fails_closed(
    tmp_path: Path, monkeypatch, corruption: str, message: str
) -> None:
    data = _official_data(tmp_path, monkeypatch)

    with pytest.raises(RowFeatureProxyError, match=message):
        run_row_feature_proxy(
            data_dir=data,
            output_dir=tmp_path / corruption,
            runtime=_CorruptCheckpointRuntime(corruption),
            wall_deadline=10_000.0,
            now=lambda: 1_000.0,
        )


def test_same_directory_reuses_completed_and_failed_but_reruns_inconclusive(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    jobs = build_proxy_jobs(CONTRACT)
    first_runtime = _Runtime(
        {
            jobs[1].candidate_id: "failed",
        }
    )
    output = tmp_path / "out"
    first = run_row_feature_proxy(
        data_dir=data,
        output_dir=output,
        runtime=first_runtime,
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )
    assert first.decision.status == "blocked"

    retry_runtime = _Runtime()
    retry = run_row_feature_proxy(
        data_dir=data,
        output_dir=output,
        runtime=retry_runtime,
        wall_deadline=12_000.0,
        now=lambda: 2_000.0,
    )
    assert retry_runtime.calls == []
    assert retry.completed == (jobs[0].candidate_id,)
    assert retry.failed == (jobs[1].candidate_id,)


def test_resume_reuses_completed_and_chains_manifest(tmp_path: Path, monkeypatch) -> None:
    data = _official_data(tmp_path, monkeypatch)
    jobs = build_proxy_jobs(CONTRACT)
    first_runtime = _Runtime({jobs[2].candidate_id: "inconclusive"})
    first = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "first",
        runtime=first_runtime,
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )

    second_runtime = _Runtime()
    second = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "second",
        resume_bundle=first.bundles.resume,
        runtime=second_runtime,
        wall_deadline=12_000.0,
        now=lambda: 2_000.0,
    )

    assert [call[1][0].candidate_id for call in second_runtime.calls] == [
        job.candidate_id for job in jobs[2:]
    ]
    assert (tmp_path / "second" / "jobs" / jobs[2].candidate_id / "checkpoint.pt").is_file()
    with ZipFile(second.bundles.resume) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["prior_manifest_sha256"] == first.bundles.manifest_sha256


def test_failed_feature_is_terminal_but_independent_later_jobs_continue(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    jobs = build_proxy_jobs(CONTRACT)
    runtime = _Runtime({jobs[2].candidate_id: "failed"})

    run = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "out",
        runtime=runtime,
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )

    assert len(runtime.calls) == 14
    assert run.failed == (jobs[2].candidate_id,)
    assert run.decision.status == "complete"
    assert run.decision.failed_bundles == ("count_context",)


def test_same_directory_reruns_inconclusive_with_checkpoint_and_keeps_failed_terminal(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    jobs = build_proxy_jobs(CONTRACT)
    output = tmp_path / "out"
    first_runtime = _Runtime(
        {
            jobs[2].candidate_id: "failed",
            jobs[3].candidate_id: "inconclusive",
        }
    )
    run_row_feature_proxy(
        data_dir=data,
        output_dir=output,
        runtime=first_runtime,
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )

    retry_runtime = _Runtime()
    retry = run_row_feature_proxy(
        data_dir=data,
        output_dir=output,
        runtime=retry_runtime,
        wall_deadline=12_000.0,
        now=lambda: 2_000.0,
    )

    assert [call[1][0].candidate_id for call in retry_runtime.calls] == [
        job.candidate_id for job in jobs[3:]
    ]
    assert retry_runtime.preexisting_checkpoints == [jobs[3].candidate_id]
    assert retry.failed == (jobs[2].candidate_id,)


def test_job_guard_stops_before_starting_a_new_job_and_persists_progress(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    jobs = build_proxy_jobs(CONTRACT)
    times = iter((1_000.0, 1_000.0, 9_100.0))
    runtime = _Runtime()

    run = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "out",
        runtime=runtime,
        wall_deadline=10_000.0,
        now=lambda: next(times),
    )

    assert [call[1][0].candidate_id for call in runtime.calls] == [jobs[0].candidate_id]
    state = json.loads(run.state_path.read_bytes())
    assert state["results"][0]["disposition"] == "completed"
    assert all(row["disposition"] == "not_started" for row in state["results"][1:])


def test_resume_rejects_changed_code_binding(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    first = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "first",
        runtime=_Runtime({"rfp__count_context__s42": "inconclusive"}),
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )

    production_files = proxy_module._code_file_paths()
    monkeypatch.setattr(
        proxy_module,
        "_code_file_paths",
        lambda root=None: tuple(reversed(production_files)),
    )
    with pytest.raises(RowFeatureProxyError, match="code_sha256"):
        run_row_feature_proxy(
            data_dir=data,
            output_dir=tmp_path / "wrong-code",
            resume_bundle=first.bundles.resume,
            runtime=_Runtime(),
            wall_deadline=12_000.0,
            now=lambda: 2_000.0,
        )


@pytest.mark.parametrize(
    ("remove", "extra"),
    [
        ("logs/stage.log", None),
        (None, ("jobs/rfp__foreign__s42/job.json", b"{}")),
    ],
)
def test_resume_rejects_consistently_hashed_missing_or_unexpected_members(
    tmp_path: Path, monkeypatch, remove, extra
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    first = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "first",
        runtime=_Runtime({"rfp__baseline__s42": "failed"}),
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )
    forged = _forge_consistent_resume(
        first.bundles.resume,
        tmp_path / f"forged-{remove is None}.zip",
        remove=remove,
        extra=extra,
    )

    with pytest.raises(RowFeatureProxyError, match="member set"):
        run_row_feature_proxy(
            data_dir=data,
            output_dir=tmp_path / f"restored-{remove is None}",
            resume_bundle=forged,
            runtime=_Runtime(),
            wall_deadline=12_000.0,
            now=lambda: 2_000.0,
        )


def test_local_recovery_rejects_altered_completed_prediction(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    output = tmp_path / "out"
    run = run_row_feature_proxy(
        data_dir=data,
        output_dir=output,
        runtime=_Runtime(),
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )
    candidate = run.completed[0]
    (output / "jobs" / candidate / "predictions.csv").write_text(
        "row_id,target,probability\nr1,0,0.9\n", encoding="utf-8"
    )

    with pytest.raises(RowFeatureProxyError, match="artifact hash"):
        run_row_feature_proxy(
            data_dir=data,
            output_dir=output,
            runtime=_Runtime(),
            wall_deadline=12_000.0,
            now=lambda: 2_000.0,
        )


def test_resume_rejects_prediction_and_result_forged_away_from_checkpoint_brier(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    first = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "first",
        runtime=_Runtime({"rfp__baseline__s3407": "inconclusive"}),
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )
    forged = tmp_path / "forged-brier.zip"
    with ZipFile(first.bundles.resume) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    state = json.loads(members["state/stage_state.json"])
    row = state["results"][0]
    prediction = b"row_id,target,probability\nr1,0,0.5\nr2,1,0.5\n"
    prediction_sha = sha256(prediction).hexdigest()
    prediction_member = row["artifacts"]["predictions.csv"]["path"]
    members[prediction_member] = prediction
    members[f"predictions/{row['candidate_id']}.csv"] = prediction
    worker_member = row["artifacts"]["worker_result.json"]["path"]
    worker_result = json.loads(members[worker_member])
    worker_result["brier"] = 0.25
    worker_result["resource_evidence"]["predictions_sha256"] = prediction_sha
    worker_bytes = json.dumps(
        worker_result, sort_keys=True, separators=(",", ":")
    ).encode()
    members[worker_member] = worker_bytes
    row["brier"] = 0.25
    row["resource_evidence"]["predictions_sha256"] = prediction_sha
    row["artifacts"]["predictions.csv"]["sha256"] = prediction_sha
    row["artifacts"]["worker_result.json"]["sha256"] = sha256(
        worker_bytes
    ).hexdigest()
    members["state/stage_state.json"] = json.dumps(
        state, sort_keys=True, separators=(",", ":")
    ).encode()
    members["metrics/job_results.json"] = json.dumps(
        state["results"], sort_keys=True, separators=(",", ":")
    ).encode()
    manifest = json.loads(members["manifest.json"])
    for name in manifest["members"]:
        manifest["members"][name] = sha256(members[name]).hexdigest()
    members["manifest.json"] = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    with ZipFile(forged, "w", compression=ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)

    with pytest.raises(RowFeatureProxyError, match="checkpoint.*Brier"):
        run_row_feature_proxy(
            data_dir=data,
            output_dir=tmp_path / "restored",
            resume_bundle=forged,
            runtime=_Runtime(),
            wall_deadline=12_000.0,
            now=lambda: 2_000.0,
        )


def test_local_recovery_rejects_noncanonical_not_started_state(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    output = tmp_path / "out"
    times = iter((1_000.0, 9_100.0))
    first = run_row_feature_proxy(
        data_dir=data,
        output_dir=output,
        runtime=_Runtime(),
        wall_deadline=10_000.0,
        now=lambda: next(times),
    )
    state = json.loads(first.state_path.read_bytes())
    state["results"][0]["completed_epochs"] = "forged"
    first.state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(RowFeatureProxyError, match="completed_epochs"):
        run_row_feature_proxy(
            data_dir=data,
            output_dir=output,
            runtime=_Runtime(),
            wall_deadline=12_000.0,
            now=lambda: 2_000.0,
        )


def test_rejects_expired_deadline_and_ambiguous_or_symlink_train(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    with pytest.raises(RowFeatureProxyError, match="deadline"):
        run_row_feature_proxy(
            data_dir=data,
            output_dir=tmp_path / "expired",
            runtime=_Runtime(),
            wall_deadline=1_000.0,
            now=lambda: 1_000.0,
        )

    nested = data / "nested"
    nested.mkdir()
    (nested / "train.csv").write_text("second\n", encoding="utf-8")
    with pytest.raises(RowFeatureProxyError, match="exactly one"):
        run_row_feature_proxy(
            data_dir=data,
            output_dir=tmp_path / "ambiguous",
            runtime=_Runtime(),
            wall_deadline=10_000.0,
            now=lambda: 1_000.0,
        )

    (nested / "train.csv").unlink()
    (nested / "train.csv").symlink_to(data / "train.csv")
    run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "symlink-ignored",
        runtime=_Runtime({"rfp__baseline__s42": "failed"}),
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )


def test_rejects_data_dir_with_a_symlink_ancestor(tmp_path: Path, monkeypatch) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    data = _official_data(real_root, monkeypatch)
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(RowFeatureProxyError, match="symlink"):
        run_row_feature_proxy(
            data_dir=linked_root / data.name,
            output_dir=tmp_path / "out",
            runtime=_Runtime(),
            wall_deadline=10_000.0,
            now=lambda: 1_000.0,
        )


def test_equivalent_runs_produce_identical_bundle_bytes(tmp_path: Path, monkeypatch) -> None:
    data = _official_data(tmp_path, monkeypatch)
    first = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "first",
        runtime=_Runtime(),
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )
    second = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "second",
        runtime=_Runtime(),
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )

    assert sha256(first.bundles.review.read_bytes()).digest() == sha256(
        second.bundles.review.read_bytes()
    ).digest()
    assert sha256(first.bundles.resume.read_bytes()).digest() == sha256(
        second.bundles.resume.read_bytes()
    ).digest()


def test_checkpoint_bundle_publication_and_restore_are_streamed(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    output = tmp_path / "first"
    first = run_row_feature_proxy(
        data_dir=data,
        output_dir=output,
        runtime=_Runtime(),
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )
    original_path_read = Path.read_bytes

    def guarded_path_read(path: Path) -> bytes:
        if path.name in {"checkpoint.pt", "best_checkpoint.pt"}:
            raise AssertionError("checkpoint must not be materialized as bytes")
        return original_path_read(path)

    state = json.loads(first.state_path.read_bytes())
    rows = {row["candidate_id"]: row for row in state["results"]}
    jobs = build_proxy_jobs(CONTRACT)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", guarded_path_read)
        _bundle_members(
            output_dir=output,
            config_bytes=proxy_module._read_contract_bytes(
                proxy_module.DEFAULT_ROW_FEATURE_PROXY_CONTRACT
            ),
            state=state,
            rows=rows,
            decision=_decision(jobs, rows, CONTRACT),
        )

    original_zip_read = ZipFile.read

    def guarded_zip_read(archive, name, *args, **kwargs):
        if str(name).endswith(("checkpoint.pt", "best_checkpoint.pt")):
            raise AssertionError("checkpoint ZIP member must be streamed")
        return original_zip_read(archive, name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(ZipFile, "read", guarded_zip_read)
        run_row_feature_proxy(
            data_dir=data,
            output_dir=tmp_path / "restored",
            resume_bundle=first.bundles.resume,
            runtime=_Runtime(),
            wall_deadline=12_000.0,
            now=lambda: 2_000.0,
        )


def test_stage_p_publication_binds_file_to_state_sha_and_preserves_target(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    output = tmp_path / "out"
    first = run_row_feature_proxy(
        data_dir=data,
        output_dir=output,
        runtime=_Runtime({"rfp__baseline__s42": "inconclusive"}),
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )
    state = json.loads(first.state_path.read_bytes())
    rows = {row["candidate_id"]: row for row in state["results"]}
    jobs = build_proxy_jobs(CONTRACT)
    review, resume = _bundle_members(
        output_dir=output,
        config_bytes=proxy_module._read_contract_bytes(
            proxy_module.DEFAULT_ROW_FEATURE_PROXY_CONTRACT
        ),
        state=state,
        rows=rows,
        decision=_decision(jobs, rows, CONTRACT),
    )
    row = state["results"][0]
    checkpoint = output / row["artifacts"]["checkpoint.pt"]["path"]
    checkpoint.write_bytes(b"mutated after bundle member validation")
    review_before = first.bundles.review.read_bytes()
    resume_before = first.bundles.resume.read_bytes()

    with pytest.raises(ArtifactError, match="changed during publication"):
        write_stage_bundles(
            output,
            StageEvidence(
                "P",
                state["campaign_config_sha256"],
                state["prior_manifest_sha256"],
                review,
                resume,
            ),
            bundle_prefix="tabm_row_feature_stage",
        )

    assert first.bundles.review.read_bytes() == review_before
    assert first.bundles.resume.read_bytes() == resume_before


def test_resume_checkpoint_size_is_rejected_before_member_read(
    tmp_path: Path, monkeypatch
) -> None:
    data = _official_data(tmp_path, monkeypatch)
    first = run_row_feature_proxy(
        data_dir=data,
        output_dir=tmp_path / "first",
        runtime=_Runtime({"rfp__baseline__s42": "inconclusive"}),
        wall_deadline=10_000.0,
        now=lambda: 1_000.0,
    )
    monkeypatch.setattr(proxy_module, "_MAX_CHECKPOINT_BYTES", 100)
    original_read = ZipFile.read

    def guarded_read(archive, name, *args, **kwargs):
        if str(name).endswith(("checkpoint.pt", "best_checkpoint.pt")):
            raise AssertionError("oversized checkpoint must not be decompressed")
        return original_read(archive, name, *args, **kwargs)

    monkeypatch.setattr(ZipFile, "read", guarded_read)
    with pytest.raises(RowFeatureProxyError, match="size limit"):
        run_row_feature_proxy(
            data_dir=data,
            output_dir=tmp_path / "restored",
            resume_bundle=first.bundles.resume,
            runtime=_Runtime(),
            wall_deadline=12_000.0,
            now=lambda: 2_000.0,
        )
