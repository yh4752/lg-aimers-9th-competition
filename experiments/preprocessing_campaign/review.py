"""Create a small, hash-validated review bundle without raw predictions."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping
import zipfile

import pandas as pd

from experiments.independent_dl.preprocessing_evaluation import build_metric_rows


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(root: Path, entry: Mapping[str, object], kind: str) -> Path:
    relative = entry.get(f"{kind}_path")
    expected = entry.get(f"{kind}_sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise ValueError(f"completed job has incomplete {kind} binding")
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file() or _hash(path) != expected:
        raise ValueError(f"completed job has a hash-invalid {kind} artifact")
    return path


def _segments(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["target"] = pd.to_numeric(frame["target"], errors="raise")
    frame["probability"] = pd.to_numeric(frame["probability"], errors="raise")
    frame["squared_error"] = (frame["probability"] - frame["target"]) ** 2
    bindings = ["anchor_id", "preprocessing_id", "seed", "fold"]
    pieces: list[pd.DataFrame] = []
    game_type = (
        frame.groupby(bindings + ["game_type"], observed=True, sort=True)
        .agg(rows=("row_id", "size"), brier=("squared_error", "mean"))
        .reset_index()
        .rename(columns={"game_type": "segment_value"})
    )
    game_type["segment_type"] = "game_type"
    pieces.append(game_type)
    masks = {
        "pitcher_oov": pd.to_numeric(frame["pitcher_oov"], errors="raise").eq(1),
        "batter_oov": pd.to_numeric(frame["batter_oov"], errors="raise").eq(1),
    }
    masks["both_oov"] = masks["pitcher_oov"] & masks["batter_oov"]
    for name, mask in masks.items():
        selected = frame.loc[mask]
        if selected.empty:
            continue
        grouped = (
            selected.groupby(bindings, observed=True, sort=True)
            .agg(rows=("row_id", "size"), brier=("squared_error", "mean"))
            .reset_index()
        )
        grouped["segment_type"] = name
        grouped["segment_value"] = "true"
        pieces.append(grouped)
    columns = bindings + ["segment_type", "segment_value", "rows", "brier"]
    return pd.concat(pieces, ignore_index=True).loc[:, columns]


def write_review_bundle(
    output_root: str | Path,
    config_path: str | Path,
    archive_path: str | Path,
) -> dict[str, object]:
    """Aggregate completed jobs and write one compact ZIP for human review."""

    root = Path(output_root).resolve()
    config = Path(config_path).resolve()
    archive = Path(archive_path).resolve()
    manifest_path = root / "campaign_manifest.json"
    if not manifest_path.is_file() or not config.is_file():
        raise ValueError("campaign manifest and config are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("jobs")
    if not isinstance(entries, dict):
        raise ValueError("campaign manifest job registry is invalid")
    predictions: list[pd.DataFrame] = []
    resources: list[dict[str, object]] = []
    failures: dict[str, object] = {}
    states: dict[str, int] = {}
    for job_id, value in entries.items():
        if not isinstance(value, dict):
            raise ValueError("campaign manifest job entry is invalid")
        state = str(value.get("state", "unknown"))
        states[state] = states.get(state, 0) + 1
        if state == "completed":
            _artifact(root, value, "metrics")
            predictions.append(pd.read_csv(_artifact(root, value, "predictions")))
            job = value.get("job", {})
            resources.append(
                {
                    "job_id": job_id,
                    "wave": job.get("wave"),
                    "family": job.get("family"),
                    "elapsed_seconds": value.get("elapsed_seconds"),
                    "peak_ram_gb": value.get("peak_ram_gb"),
                    "peak_gpu_gb": value.get("peak_gpu_gb"),
                }
            )
        elif state == "failed":
            failures[job_id] = value.get("failure_reason")
    summary: dict[str, object] = {
        "campaign_id": manifest.get("campaign_id"),
        "registered": len(entries),
        "completed": states.get("completed", 0),
        "failed": states.get("failed", 0),
        "pending": states.get("pending", 0) + states.get("running", 0),
        "hash_valid_completed": len(predictions),
        "manifest_sha256": _hash(manifest_path),
        "config_sha256": _hash(config),
    }
    with tempfile.TemporaryDirectory(prefix="preprocessing_review_") as temporary_value:
        temporary = Path(temporary_value)
        (temporary / "review_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "failed_jobs.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(manifest_path, temporary / manifest_path.name)
        shutil.copy2(config, temporary / "preprocessing_ablation_v1.json")
        if predictions:
            combined = pd.concat(predictions, ignore_index=True)
            build_metric_rows(combined).to_csv(
                temporary / "fold_metrics.csv", index=False
            )
            _segments(combined).to_csv(temporary / "segment_metrics.csv", index=False)
        else:
            pd.DataFrame().to_csv(temporary / "fold_metrics.csv", index=False)
            pd.DataFrame().to_csv(temporary / "segment_metrics.csv", index=False)
        pd.DataFrame(resources).to_csv(temporary / "resource_usage.csv", index=False)
        archive.parent.mkdir(parents=True, exist_ok=True)
        pending_archive = archive.with_suffix(archive.suffix + ".tmp")
        with zipfile.ZipFile(
            pending_archive, "w", compression=zipfile.ZIP_DEFLATED
        ) as bundle:
            for path in sorted(temporary.iterdir()):
                bundle.write(path, arcname=path.name)
        os.replace(pending_archive, archive)
    return summary
