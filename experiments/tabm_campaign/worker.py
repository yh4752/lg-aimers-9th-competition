from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Mapping

from experiments.independent_dl.preprocessing import (
    PreprocessingSpec,
    normalize_spec,
)
from experiments.independent_dl.row_features import (
    ROW_FEATURE_BUNDLES,
    row_segment_labels,
)

from .runner import CampaignJob, CampaignJobResult


_RUNTIME_ROOT = Path(__file__).resolve().parents[2]


def _worker_environment(gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(_RUNTIME_ROOT) + (
        os.pathsep + existing if existing else ""
    )
    return environment


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json(value))
    os.replace(temporary, path)


def _job_sha(job: CampaignJob) -> str:
    return sha256(_canonical_json(asdict(job))).hexdigest()


def _job_from_json(path: Path) -> CampaignJob:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return CampaignJob(**raw)


def _preprocessing_spec(job: CampaignJob) -> PreprocessingSpec:
    if job.feature_bundle is None:
        return PreprocessingSpec("dl_standard", ("hand_matchup",))
    if job.feature_bundle not in ROW_FEATURE_BUNDLES:
        raise ValueError(f"unknown row feature bundle: {job.feature_bundle}")
    return PreprocessingSpec(
        "dl_standard", ("hand_matchup", job.feature_bundle)
    )


def _prediction_evidence_frame(
    *,
    fit_rows,
    valid_rows,
    valid_batch,
    probability,
    category_maps: Mapping[str, Mapping[object, object]],
):
    import numpy as np
    import pandas as pd

    required_fit = {"pitcher_id", "batter_id"}
    required_valid = {
        "row_id",
        "control_success",
        "game_type",
        "pitcher_hand",
        "batter_hand",
        "pitcher_id",
        "batter_id",
        "li",
        "inning",
        "runner_on_2b",
        "runner_on_3b",
    }
    missing_fit = sorted(required_fit.difference(fit_rows.columns))
    missing_valid = sorted(required_valid.difference(valid_rows.columns))
    if missing_fit or missing_valid:
        raise RuntimeError(
            "prediction evidence is missing required columns: "
            f"fit={missing_fit}, valid={missing_valid}"
        )

    valid_row_ids = valid_rows["row_id"].astype(str).to_numpy(copy=False)
    cache_row_ids = np.asarray(valid_batch.row_id).astype(str)
    target = np.asarray(valid_batch.y, dtype="float64")
    probability = np.asarray(probability, dtype="float64")
    lengths = {
        len(valid_rows),
        len(valid_row_ids),
        len(cache_row_ids),
        len(target),
        len(probability),
        len(valid_batch.game_type),
    }
    if len(lengths) != 1:
        raise RuntimeError("prediction evidence arrays must have equal lengths")
    if len(set(valid_row_ids.tolist())) != len(valid_row_ids):
        raise RuntimeError("prediction evidence row_id values must be unique")
    if not np.array_equal(valid_row_ids, cache_row_ids):
        raise RuntimeError(
            "prediction evidence row_id order differs from cache.valid"
        )
    source_target = pd.to_numeric(
        valid_rows["control_success"], errors="coerce"
    ).to_numpy(dtype="float64")
    if not np.isfinite(source_target).all() or not np.array_equal(
        source_target, target
    ):
        raise RuntimeError("prediction evidence target differs from cache.valid.y")
    if not np.isfinite(probability).all():
        raise RuntimeError("prediction evidence probabilities must be finite")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise RuntimeError("prediction evidence probabilities must be in [0, 1]")

    payload: dict[str, object] = {
        "row_id": cache_row_ids,
        "target": target.astype("int64"),
        "probability": probability,
        "game_type": np.asarray(valid_batch.game_type).astype(str),
    }
    if "game_month" in valid_rows:
        payload["game_month"] = valid_rows["game_month"].to_numpy(copy=False)
    for entity in ("pitcher_id", "batter_id"):
        if entity in category_maps:
            known = {str(value) for value in category_maps[entity]}
            payload[f"{entity}_known"] = np.where(
                valid_rows[entity]
                .astype("string")
                .fillna("__MISSING__")
                .astype(str)
                .isin(known),
                "known",
                "oov",
            )

    segment_sources = valid_rows.loc[
        :,
        [
            "game_type",
            "pitcher_hand",
            "batter_hand",
            "pitcher_id",
            "batter_id",
            "li",
            "inning",
            "runner_on_2b",
            "runner_on_3b",
        ],
    ]
    segments = row_segment_labels(
        segment_sources,
        set(fit_rows["pitcher_id"].tolist()),
        set(fit_rows["batter_id"].tolist()),
    )
    for column in segments.columns:
        payload[str(column)] = segments[column].to_numpy(copy=False)
    return pd.DataFrame(payload)


def _find_one(root: Path, name: str, *, required: bool) -> Path | None:
    direct = root / name
    candidates = [direct] if direct.is_file() else sorted(root.rglob(name))
    candidates = [path for path in candidates if path.is_file()]
    if len(candidates) > 1 or (required and len(candidates) != 1):
        raise RuntimeError(f"official {name} candidate count must be {1 if required else '0 or 1'}; found={len(candidates)}")
    return candidates[0] if candidates else None


def _serialize_result(result: CampaignJobResult, job_sha256: str) -> dict[str, object]:
    return {
        "job_sha256": job_sha256,
        "candidate_id": result.candidate_id,
        "status": result.status,
        "brier": result.brier,
        "best_epoch": result.best_epoch,
        "completed_epochs": result.completed_epochs,
        "checkpoint": None if result.checkpoint is None else str(result.checkpoint),
        "predictions_path": None if result.predictions_path is None else str(result.predictions_path),
        "resource_evidence": dict(result.resource_evidence),
        "failure": result.failure,
    }


def _result_from_payload(payload: Mapping[str, object]) -> CampaignJobResult:
    return CampaignJobResult(
        candidate_id=str(payload["candidate_id"]),
        status=str(payload["status"]),
        brier=None if payload.get("brier") is None else float(payload["brier"]),
        best_epoch=None if payload.get("best_epoch") is None else int(payload["best_epoch"]),
        completed_epochs=int(payload.get("completed_epochs", 0)),
        checkpoint=None if payload.get("checkpoint") is None else Path(str(payload["checkpoint"])),
        predictions_path=None if payload.get("predictions_path") is None else Path(str(payload["predictions_path"])),
        resource_evidence=dict(payload.get("resource_evidence", {})),
        failure=None if payload.get("failure") is None else str(payload["failure"]),
    )


def run_worker(job: CampaignJob, data_dir: Path, output_dir: Path, cache_root: Path, deadline: float) -> CampaignJobResult:
    preprocessing_spec = _preprocessing_spec(job)
    normalized_spec = normalize_spec(preprocessing_spec)
    import numpy as np
    import pandas as pd

    from experiments.independent_dl.models.common import import_runtime_module, metadata_from_train
    from experiments.independent_dl.models.tabm import TabMAdapter
    from experiments.independent_dl.training import TrainRequest, fit_candidate
    from .cache import materialize_fixed_cache
    from .sampling import proxy_row_ids
    from .training import preflight

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = _find_one(data_dir, "train.csv", required=True)
    history_path = _find_one(data_dir, "trackman_history.csv", required=False)
    train_frame = pd.read_csv(train_path)
    history = pd.read_csv(history_path) if history_path is not None else pd.DataFrame()
    seasons = pd.to_numeric(train_frame["season"], errors="raise").astype("int64")
    fit_rows = train_frame.loc[seasons.le(job.train_end_year)].reset_index(drop=True)
    valid_rows = train_frame.loc[seasons.eq(job.valid_year)].reset_index(drop=True)
    if fit_rows.empty or valid_rows.empty:
        raise RuntimeError("temporal fold has no train or validation rows")
    sample_ids = (
        tuple(fit_rows["row_id"].astype(str))
        if job.sample_mode == "full"
        else proxy_row_ids(fit_rows, train_end_year=job.train_end_year, max_rows=400_000, seed=job.seed)
    )

    lock_path = cache_root / ".materialize.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        try:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        cache = materialize_fixed_cache(
            cache_root,
            train=fit_rows,
            valid=valid_rows,
            history=history,
            train_end_year=job.train_end_year,
            valid_year=job.valid_year,
            spec=preprocessing_spec,
            sample_ids=sample_ids,
        )
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except (ImportError, NameError):
            pass

    model_config = {
        "architecture": "tabm",
        "k": job.k,
        "width": job.width,
        "blocks": job.blocks,
        "dropout": job.dropout,
        "num_embedding": job.num_embedding,
    }
    adapter = TabMAdapter(loss_name=job.loss)

    def architecture_probe() -> dict[str, object]:
        torch = import_runtime_module("torch")
        model = adapter.build(model_config, cache.model_metadata, "cuda")
        count = min(32, len(cache.train.row_id))
        indices = np.arange(count, dtype="int64")
        x_num = torch.as_tensor(np.asarray(cache.train.x_num[indices]), dtype=torch.float32, device="cuda")
        x_cat = torch.as_tensor(np.asarray(cache.train.x_cat[indices]), dtype=torch.long, device="cuda")
        target = torch.as_tensor(np.asarray(cache.train.y[indices]), dtype=torch.float32, device="cuda")
        row_indices = torch.as_tensor(indices, dtype=torch.long, device="cuda")
        loss = adapter.loss(model, x_num, x_cat, target, row_indices=row_indices)
        loss.backward()
        evidence = {
            "loss": float(loss.detach().cpu()),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "gpu_allocated_bytes": int(torch.cuda.memory_allocated()),
            "gpu_reserved_bytes": int(torch.cuda.memory_reserved()),
            "device_name": str(torch.cuda.get_device_name(0)),
        }
        del model, x_num, x_cat, target, row_indices, loss
        torch.cuda.empty_cache()
        return evidence

    preflight_result = preflight(job.candidate_id, architecture_probe)
    preprocessing_evidence = {
        "cache_digest": cache.identity.digest(),
        "cache_reused": cache.reused,
        "feature_bundle": job.feature_bundle,
        "preprocessing_spec": asdict(normalized_spec),
    }
    if preflight_result.status != "completed":
        return CampaignJobResult(
            job.candidate_id,
            "failed",
            None,
            None,
            0,
            None,
            None,
            {
                **preprocessing_evidence,
                "preflight": asdict(preflight_result),
            },
            f"{preflight_result.failure_type}: {preflight_result.message}",
        )

    os.environ["PREPROCESSING_SESSION_DEADLINE_UNIX"] = str(deadline)
    request = TrainRequest(
        candidate_id=job.candidate_id,
        family="tabm",
        seed=job.seed,
        epochs=job.max_epochs,
        min_epochs=job.min_epochs,
        model_config=model_config,
        training_config={
            "optimizer": "adamw",
            "scheduler": job.scheduler,
            "learning_rate": job.learning_rate,
            "weight_decay": 0.0001,
            "effective_batch_size": 4096,
            "micro_batch_size": 512,
            "amp": True,
            "patience": job.patience,
        },
        train=cache.train,
        valid=cache.valid,
        checkpoint_binding={
            "config_sha256": _job_sha(job),
            "cache_sha256": cache.identity.digest(),
            "training_source_sha256": sha256(
                (Path(__file__).resolve().parents[1] / "independent_dl" / "training.py").read_bytes()
                + (Path(__file__).resolve().parents[1] / "independent_dl" / "models" / "tabm.py").read_bytes()
            ).hexdigest(),
        },
        model_metadata=cache.model_metadata,
    )
    started = time.monotonic()
    trained = fit_candidate(request, adapter, output_dir)
    probability = np.asarray(trained.predictions, dtype="float64")
    prediction_frame = _prediction_evidence_frame(
        fit_rows=fit_rows,
        valid_rows=valid_rows,
        valid_batch=cache.valid,
        probability=probability,
        category_maps=cache.state.category_maps,
    )
    target = prediction_frame["target"].to_numpy(dtype="float64")
    brier = float(np.mean(np.square(probability - target)))
    predictions_path = output_dir / "predictions.csv"
    prediction_frame.to_csv(predictions_path, index=False)
    evidence = dict(trained.hardware)
    evidence.update(
        {
            "wall_seconds": time.monotonic() - started,
            **preprocessing_evidence,
            "preflight": asdict(preflight_result),
        }
    )
    return CampaignJobResult(
        job.candidate_id,
        "inconclusive" if trained.budget_reached else "completed",
        brier,
        trained.best_epoch,
        trained.completed_epochs,
        trained.checkpoint,
        predictions_path,
        evidence,
        None,
    )


class SubprocessCampaignRuntime:
    """Schedule one independent Python training process on each visible GPU."""

    def __init__(self, data_dir: Path, *, python: str = sys.executable) -> None:
        self.data_dir = data_dir.resolve()
        self.python = python

    @staticmethod
    def _read_worker_result(
        path: Path, job: CampaignJob
    ) -> CampaignJobResult | None:
        result_path = path / "worker_result.json"
        if not result_path.is_file():
            return None
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("job_sha256") != _job_sha(job):
            return None
        result = _result_from_payload(payload)
        if result.status == "completed" and (
            result.checkpoint is None or not result.checkpoint.is_file()
        ):
            return None
        return result

    @classmethod
    def _read_result(
        cls, path: Path, job: CampaignJob
    ) -> CampaignJobResult | None:
        result = cls._read_worker_result(path, job)
        if result is None or result.status != "completed":
            return None
        return result

    def run_jobs(self, version, jobs, output_dir, *, gpu_count, job_deadline):
        output_dir.mkdir(parents=True, exist_ok=True)
        pending = list(jobs)
        results: list[CampaignJobResult] = []
        active: dict[int, tuple[subprocess.Popen[str], threading.Thread, CampaignJob, Path]] = {}
        cache_root = output_dir.parent / "feature_cache"

        def stream(process: subprocess.Popen[str], job: CampaignJob, log_path: Path, gpu: int) -> None:
            assert process.stdout is not None
            with log_path.open("a", encoding="utf-8") as log:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(f"WORKER[{gpu}:{job.candidate_id}] {line.rstrip()}", flush=True)

        while pending or active:
            if time.time() >= job_deadline + 120:
                for gpu, (process, thread, job, job_dir) in list(active.items()):
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    thread.join(timeout=5)
                    result = self._read_worker_result(job_dir, job)
                    if result is None:
                        best = job_dir / "best_checkpoint.pt"
                        meta_path = job_dir / "checkpoint_meta.json"
                        completed_epochs = 0
                        best_epoch = None
                        checkpoint = best if best.is_file() else None
                        if meta_path.is_file():
                            try:
                                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                                if isinstance(meta, dict):
                                    completed_epochs = int(meta.get("epoch", -1)) + 1
                                    if "best_epoch" in meta:
                                        best_epoch = int(meta["best_epoch"])
                                    checkpoint_name = str(meta.get("checkpoint", ""))
                                    resume_checkpoint = job_dir / Path(checkpoint_name).name
                                    if checkpoint is None and resume_checkpoint.is_file():
                                        checkpoint = resume_checkpoint
                            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                                pass
                        result = CampaignJobResult(
                            job.candidate_id,
                            "inconclusive",
                            None,
                            best_epoch,
                            completed_epochs,
                            checkpoint,
                            None,
                            {},
                            "worker_exceeded_deadline_grace",
                        )
                        _atomic_json(
                            job_dir / "worker_result.json",
                            _serialize_result(result, _job_sha(job)),
                        )
                    results.append(result)
                    active.pop(gpu)
                break
            for gpu in range(gpu_count):
                if gpu in active or not pending or time.time() >= job_deadline:
                    continue
                job = pending.pop(0)
                job_dir = output_dir / job.candidate_id
                reused = self._read_result(job_dir, job)
                if reused is not None:
                    print(f"JOB_REUSED job={job.candidate_id} status={reused.status}", flush=True)
                    results.append(reused)
                    continue
                job_dir.mkdir(parents=True, exist_ok=True)
                stale_result_path = job_dir / "worker_result.json"
                if stale_result_path.is_file():
                    stale_result_path.unlink()
                job_path = job_dir / "job.json"
                _atomic_json(job_path, asdict(job))
                env = _worker_environment(gpu)
                command = [
                    self.python,
                    "-m",
                    "experiments.tabm_campaign.worker",
                    "--job",
                    str(job_path),
                    "--data-dir",
                    str(self.data_dir),
                    "--output-dir",
                    str(job_dir),
                    "--cache-root",
                    str(cache_root),
                    "--deadline",
                    str(job_deadline),
                ]
                print(f"JOB_START job={job.candidate_id} gpu={gpu} deadline_unix={int(job_deadline)}", flush=True)
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
                thread = threading.Thread(target=stream, args=(process, job, job_dir / "worker.log", gpu), daemon=True)
                thread.start()
                active[gpu] = (process, thread, job, job_dir)
            finished = [gpu for gpu, (process, _, _, _) in active.items() if process.poll() is not None]
            for gpu in finished:
                process, thread, job, job_dir = active.pop(gpu)
                thread.join(timeout=5)
                result = self._read_worker_result(job_dir, job)
                if result is None:
                    result = CampaignJobResult(job.candidate_id, "failed", None, None, 0, None, None, {}, f"worker_exit_code={process.returncode}")
                    _atomic_json(
                        job_dir / "worker_result.json",
                        _serialize_result(result, _job_sha(job)),
                    )
                print(f"JOB_END job={job.candidate_id} gpu={gpu} status={result.status}", flush=True)
                results.append(result)
            if not finished:
                time.sleep(0.2)
            if time.time() >= job_deadline and not active:
                break
        results.extend(
            CampaignJobResult(job.candidate_id, "inconclusive", None, None, 0, None, None, {}, "not_started_before_stage_deadline")
            for job in pending
        )
        by_id = {result.candidate_id: result for result in results}
        return tuple(by_id[job.candidate_id] for job in jobs)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--deadline", required=True, type=float)
    args = parser.parse_args(argv)
    job_path = Path(args.job)
    job = _job_from_json(job_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = run_worker(job, Path(args.data_dir), output_dir, Path(args.cache_root), args.deadline)
    except Exception as exc:
        import traceback

        result = CampaignJobResult(job.candidate_id, "failed", None, None, 0, None, None, {}, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    _atomic_json(output_dir / "worker_result.json", _serialize_result(result, _job_sha(job)))
    return 0 if result.status in {"completed", "inconclusive"} else 1


if __name__ == "__main__":
    raise SystemExit(_main())
