"""Restartable state machine for preprocessing-model jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import statistics
import time
import traceback
from typing import Mapping, Protocol


class CampaignInterrupted(RuntimeError):
    """A deliberate interruption that must leave the current job resumable."""


@dataclass(frozen=True)
class PreprocessingJobResult:
    metrics_path: Path
    predictions_path: Path
    best_brier: float
    elapsed_seconds: float
    peak_ram_gb: float
    peak_gpu_gb: float


@dataclass(frozen=True)
class PreprocessingCampaignSummary:
    campaign_id: str
    completed: tuple[str, ...]
    failed: tuple[str, ...]
    pending: tuple[str, ...]
    output_root: Path


class PreprocessingRuntime(Protocol):
    def run_job(self, job: object, output_dir: Path) -> PreprocessingJobResult: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: object) -> object:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        return {
            str(key): _json_value(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    raise TypeError(f"job contains a non-JSON value: {type(value).__name__}")


def _job_payload(job: object) -> dict[str, object]:
    value = _json_value(job)
    if not isinstance(value, dict):
        raise TypeError("job must serialize to an object")
    return value


def _job_hash(job: object) -> str:
    encoded = json.dumps(
        _job_payload(job), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _relative(root: Path, path: Path) -> str:
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError("job artifact escapes the campaign output root")
    return str(resolved.relative_to(root))


def _artifact(root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = (root / value).resolve()
    if root != path and root not in path.parents:
        return None
    return path


def _valid_completion(root: Path, entry: Mapping[str, object]) -> bool:
    if entry.get("state") != "completed":
        return False
    for name in ("metrics", "predictions"):
        path = _artifact(root, entry.get(f"{name}_path"))
        expected = entry.get(f"{name}_sha256")
        if path is None or not path.is_file() or not isinstance(expected, str):
            return False
        if _hash(path) != expected:
            return False
    return True


def _jobs(campaign: object) -> tuple[object, ...]:
    jobs: list[object] = []
    for attribute in ("wave_a_jobs", "registered_jobs"):
        value = getattr(campaign, attribute, ())
        jobs.extend(tuple(value))
    unique: dict[str, object] = {}
    for job in jobs:
        job_id = str(getattr(job, "job_id"))
        if job_id in unique:
            raise ValueError(f"job ID repeats: {job_id}")
        unique[job_id] = job
    return tuple(unique.values())


def _new_entry(job: object) -> dict[str, object]:
    return {
        "job": _job_payload(job),
        "config_sha256": _job_hash(job),
        "state": "pending",
        "attempts": 0,
        "metrics_path": None,
        "predictions_path": None,
        "metrics_sha256": None,
        "predictions_sha256": None,
        "best_brier": None,
        "elapsed_seconds": None,
        "peak_ram_gb": None,
        "peak_gpu_gb": None,
        "failure_reason": None,
        "updated_at": _now(),
    }


def estimate_remaining_resources(
    rows: list[Mapping[str, object]],
    *,
    family: str,
    pending_same_family: int,
) -> dict[str, float | int | None]:
    """Estimate remaining wall time from completed same-family median jobs."""

    values = [
        float(row["elapsed_seconds"])
        for row in rows
        if row.get("family") == family
        and isinstance(row.get("elapsed_seconds"), (int, float))
        and float(row["elapsed_seconds"]) >= 0.0
    ]
    median = statistics.median(values) if values else None
    return {
        "family": family,
        "completed_same_family": len(values),
        "pending_same_family": int(pending_same_family),
        "median_seconds": median,
        "estimated_remaining_seconds": (
            None if median is None else float(median * pending_same_family)
        ),
    }


def run_preprocessing_campaign(
    campaign: object,
    output_root: str | Path,
    runtime: PreprocessingRuntime,
) -> PreprocessingCampaignSummary:
    """Run registered jobs sequentially and preserve independent failures."""

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "campaign_manifest.json"
    jobs = _jobs(campaign)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("campaign_id") != getattr(campaign, "campaign_id"):
            raise ValueError("campaign manifest identity differs")
    else:
        manifest = {
            "schema_version": 1,
            "campaign_id": str(getattr(campaign, "campaign_id")),
            "protocol": str(getattr(campaign, "protocol", "unknown")),
            "jobs": {},
            "updated_at": _now(),
        }
    entries = manifest.get("jobs")
    if not isinstance(entries, dict):
        raise ValueError("campaign job registry is invalid")
    for job in jobs:
        job_id = str(getattr(job, "job_id"))
        current = entries.get(job_id)
        if current is None:
            entries[job_id] = _new_entry(job)
        elif not isinstance(current, dict) or current.get("config_sha256") != _job_hash(job):
            raise ValueError(f"job config changed: {job_id}")
    manifest["updated_at"] = _now()
    _atomic_json(manifest_path, manifest)

    for job in jobs:
        job_id = str(getattr(job, "job_id"))
        entry = entries[job_id]
        if _valid_completion(root, entry):
            continue
        entry.update(
            {
                "state": "running",
                "attempts": int(entry.get("attempts", 0)) + 1,
                "failure_reason": None,
                "updated_at": _now(),
            }
        )
        _atomic_json(manifest_path, manifest)
        try:
            result = runtime.run_job(job, root / "jobs" / job_id)
        except (CampaignInterrupted, KeyboardInterrupt):
            entry.update({"state": "pending", "updated_at": _now()})
            _atomic_json(manifest_path, manifest)
            raise
        except Exception as error:
            entry.update(
                {
                    "state": "failed",
                    "failure_reason": (
                        f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
                    ),
                    "updated_at": _now(),
                }
            )
            _atomic_json(manifest_path, manifest)
            continue
        for path in (result.metrics_path, result.predictions_path):
            if not path.is_file():
                raise ValueError(f"runtime did not create artifact: {path}")
        entry.update(
            {
                "state": "completed",
                "metrics_path": _relative(root, result.metrics_path),
                "predictions_path": _relative(root, result.predictions_path),
                "metrics_sha256": _hash(result.metrics_path),
                "predictions_sha256": _hash(result.predictions_path),
                "best_brier": float(result.best_brier),
                "elapsed_seconds": float(result.elapsed_seconds),
                "peak_ram_gb": float(result.peak_ram_gb),
                "peak_gpu_gb": float(result.peak_gpu_gb),
                "updated_at": _now(),
            }
        )
        manifest["updated_at"] = _now()
        _atomic_json(manifest_path, manifest)

    completed = tuple(job_id for job_id, entry in entries.items() if _valid_completion(root, entry))
    failed = tuple(job_id for job_id, entry in entries.items() if entry.get("state") == "failed")
    pending = tuple(
        job_id
        for job_id, entry in entries.items()
        if job_id not in completed and entry.get("state") != "failed"
    )
    return PreprocessingCampaignSummary(
        str(getattr(campaign, "campaign_id")), completed, failed, pending, root
    )


class OfficialPreprocessingDLRuntime:
    """Official-data DL runtime called only by the user-owned Colab command."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        cache_root: str | Path,
        fit_function: object | None = None,
        adapter_factory: object | None = None,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.cache_root = Path(cache_root).resolve()
        self.fit_function = fit_function
        self.adapter_factory = adapter_factory
        self._train = None
        self._history = None

    def _load(self):
        if self._train is None:
            import pandas as pd

            train_path = self.data_dir / "train.csv"
            history_path = self.data_dir / "trackman_history.csv"
            if not train_path.is_file() or not history_path.is_file():
                raise ValueError("train.csv and trackman_history.csv are required")
            self._train = pd.read_csv(train_path)
            self._history = pd.read_csv(history_path)
        return self._train, self._history

    @staticmethod
    def _adapter(family: str):
        from .campaign import OfficialCampaignRuntime

        return OfficialCampaignRuntime._default_adapter(family)

    def run_job(self, job: object, output_dir: Path) -> PreprocessingJobResult:
        import numpy as np
        import pandas as pd

        from .features import materialize_preprocessed_fold_cache
        from .preprocessing import PreprocessingSpec
        from .training import TrainRequest, fit_candidate

        started = time.monotonic()
        train_frame, history = self._load()
        seasons = pd.to_numeric(train_frame["season"], errors="raise")
        train = train_frame.loc[seasons.le(job.train_end_year)].reset_index(drop=True)
        valid = train_frame.loc[seasons.eq(job.valid_year)].reset_index(drop=True)
        cache = materialize_preprocessed_fold_cache(
            self.cache_root,
            train,
            valid,
            history,
            job.feature_view,
            PreprocessingSpec(job.setting.profile, job.setting.components),
            job.train_end_year,
            job.valid_year,
        )
        request = TrainRequest(
            candidate_id=job.job_id,
            family=job.family,
            seed=job.seed,
            epochs=job.epochs,
            model_config=job.model,
            training_config=job.training,
            train=cache.train,
            valid=cache.valid,
        )
        adapter_factory = self.adapter_factory or self._adapter
        result = (self.fit_function or fit_candidate)(
            request, adapter_factory(job.family), output_dir
        )
        probability = np.asarray(result.predictions, dtype="float64")
        target = np.asarray(cache.valid.y, dtype="float64")
        if probability.shape != target.shape:
            raise ValueError("predictions are not aligned to validation")
        pitcher_index = cache.state.categorical_columns.index("pitcher_id")
        batter_index = cache.state.categorical_columns.index("batter_id")
        prediction_frame = pd.DataFrame(
            {
                "row_id": cache.valid.row_id.astype(str),
                "fold": f"valid_{job.valid_year}",
                "season": cache.valid.season.astype("int64"),
                "game_type": cache.valid.game_type.astype(str),
                "target": target.astype("int64"),
                "probability": probability,
                "anchor_id": job.anchor_id,
                "preprocessing_id": job.setting.setting_id,
                "seed": job.seed,
                "pitcher_oov": (cache.valid.x_cat[:, pitcher_index] == 0).astype(int),
                "batter_oov": (cache.valid.x_cat[:, batter_index] == 0).astype(int),
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / "predictions.csv"
        metrics_path = output_dir / "metrics.json"
        prediction_frame.to_csv(predictions_path.with_suffix(".csv.tmp"), index=False)
        os.replace(predictions_path.with_suffix(".csv.tmp"), predictions_path)
        brier = float(np.mean(np.square(probability - target)))
        _atomic_json(
            metrics_path,
            {
                "job_id": job.job_id,
                "anchor_id": job.anchor_id,
                "preprocessing_id": job.setting.setting_id,
                "fold": f"valid_{job.valid_year}",
                "seed": job.seed,
                "valid_rows": len(valid),
                "brier": brier,
            },
        )
        elapsed = time.monotonic() - started
        ram_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
        try:
            import torch

            gpu_gb = torch.cuda.max_memory_allocated() / (1024**3)
        except Exception:
            gpu_gb = 0.0
        return PreprocessingJobResult(
            metrics_path, predictions_path, brier, elapsed, ram_gb, gpu_gb
        )
