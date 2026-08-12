"""Deterministic resume and review bundles for the budgeted campaign."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd


REQUIRED_REVIEW_NAMES = {
    "stage_summary.json",
    "decision_table.csv",
    "model_metrics.csv",
    "segment_metrics.csv",
    "learning_curves.csv",
    "resource_usage.csv",
    "validation_predictions.csv.gz",
    "artifact_manifest.json",
    "errors.json",
    "run.log",
}


@dataclass(frozen=True)
class StageBundles:
    resume: Path
    review: Path
    final_review: Path | None = None


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _gzip_csv_bytes(frame: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(_csv_bytes(frame))
    return output.getvalue()


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _write_zip(path: Path, members: Mapping[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with ZipFile(temporary, "w") as archive:
        for name in sorted(members):
            archive.writestr(_zip_info(name), members[name])
    temporary.replace(path)


def _read_json(path: Path, default: object) -> object:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _job_records(root: Path) -> list[tuple[str, dict[str, object], pd.DataFrame]]:
    records = []
    for metrics_path in sorted((root / "jobs").glob("*/metrics.json")):
        metrics = _read_json(metrics_path, {})
        predictions_path = metrics_path.parent / "predictions.csv"
        if not isinstance(metrics, dict) or not predictions_path.is_file():
            continue
        predictions = pd.read_csv(predictions_path)
        job_id = str(metrics.get("job_id", metrics_path.parent.name))
        records.append((job_id, metrics, predictions))
    return records


def _model_metrics(
    records: list[tuple[str, dict[str, object], pd.DataFrame]],
) -> pd.DataFrame:
    columns = [
        "job_id",
        "family",
        "setting_id",
        "train_end_year",
        "valid_year",
        "seed",
        "sample_mode",
        "train_rows",
        "valid_rows",
        "brier",
        "best_epoch",
        "completed_epochs",
        "validation_points",
        "elapsed_seconds",
        "sample_sha256",
    ]
    rows = [{column: metric.get(column) for column in columns} for _, metric, _ in records]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["job_id"], kind="stable", ignore_index=True
    )


def _learning_curves(
    records: list[tuple[str, dict[str, object], pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    for job_id, metric, _ in records:
        for raw in metric.get("validation_curve", []):
            if isinstance(raw, (list, tuple)) and len(raw) == 2:
                rows.append(
                    {
                        "job_id": job_id,
                        "epoch": int(raw[0]),
                        "validation_brier": float(raw[1]),
                    }
                )
    return pd.DataFrame(
        rows, columns=["job_id", "epoch", "validation_brier"]
    ).sort_values(["job_id", "epoch"], kind="stable", ignore_index=True)


def _resource_usage(
    records: list[tuple[str, dict[str, object], pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    for job_id, metric, _ in records:
        hardware = metric.get("hardware", {})
        rows.append(
            {
                "job_id": job_id,
                "elapsed_seconds": metric.get("elapsed_seconds"),
                "train_rows": metric.get("train_rows"),
                "valid_rows": metric.get("valid_rows"),
                "completed_epochs": metric.get("completed_epochs"),
                "hardware_json": json.dumps(
                    hardware, ensure_ascii=False, sort_keys=True
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "job_id",
            "elapsed_seconds",
            "train_rows",
            "valid_rows",
            "completed_epochs",
            "hardware_json",
        ],
    ).sort_values(["job_id"], kind="stable", ignore_index=True)


def _brier(frame: pd.DataFrame) -> float:
    target = pd.to_numeric(frame["target"], errors="raise").to_numpy(dtype="float64")
    probability = pd.to_numeric(
        frame["probability"], errors="raise"
    ).to_numpy(dtype="float64")
    return float(np.mean(np.square(probability - target)))


def _segment_metrics(
    records: list[tuple[str, dict[str, object], pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    for job_id, _, frame in records:
        groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", frame)]
        for column in ("pitcher_oov", "batter_oov", "game_type"):
            if column in frame:
                groups.extend(
                    (column, str(value), subset)
                    for value, subset in frame.groupby(column, dropna=False, sort=True)
                )
        for group_type, group_value, subset in groups:
            rows.append(
                {
                    "job_id": job_id,
                    "group_type": group_type,
                    "group_value": group_value,
                    "rows": len(subset),
                    "brier": _brier(subset),
                }
            )
    return pd.DataFrame(
        rows, columns=["job_id", "group_type", "group_value", "rows", "brier"]
    ).sort_values(
        ["job_id", "group_type", "group_value"], kind="stable", ignore_index=True
    )


def _wide_predictions(
    records: list[tuple[str, dict[str, object], pd.DataFrame]],
) -> pd.DataFrame:
    keys = ["row_id", "fold", "target"]
    wide: pd.DataFrame | None = None
    for job_id, _, frame in records:
        missing = set(keys + ["probability"]) - set(frame)
        if missing:
            raise ValueError(f"prediction file is missing columns: {sorted(missing)}")
        part = frame[keys + ["probability"]].copy()
        if part.duplicated(keys).any():
            raise ValueError(f"prediction rows are duplicated: {job_id}")
        part = part.rename(columns={"probability": f"probability__{job_id}"})
        wide = part if wide is None else wide.merge(part, on=keys, how="outer", validate="one_to_one")
    if wide is None:
        return pd.DataFrame(columns=keys)
    return wide.sort_values(keys, kind="stable", ignore_index=True)


def _decision_table(state: Mapping[str, object]) -> pd.DataFrame:
    rows = []
    selected = state.get("selected_model")
    if isinstance(selected, dict):
        rows.append(
            {
                "scope": "model",
                "candidate": selected.get("candidate_id", selected.get("family")),
                "status": "selected",
                "reason": selected.get("reason", "stage_1_selection"),
            }
        )
    for key, scope in (
        ("promoted_dl", "dl_preprocessing"),
        ("promoted_catboost", "catboost_preprocessing"),
    ):
        values = state.get(key, [])
        if isinstance(values, list):
            for value in values:
                if isinstance(value, dict):
                    rows.append(
                        {
                            "scope": scope,
                            "candidate": value.get("setting_id"),
                            "status": "promoted",
                            "reason": f"delta={value.get('delta')}",
                        }
                    )
    final = state.get("final_dl")
    if isinstance(final, dict):
        rows.append(
            {
                "scope": "final_preprocessing",
                "candidate": final.get("setting_id"),
                "status": state.get("preprocessing_status", "selected_for_full_check"),
                "reason": state.get("stage_five_reason", "stage_4_stability"),
            }
        )
    return pd.DataFrame(
        rows, columns=["scope", "candidate", "status", "reason"]
    )


def _errors(manifest: object) -> dict[str, object]:
    rows = []
    if isinstance(manifest, dict) and isinstance(manifest.get("jobs"), dict):
        for job_id, entry in manifest["jobs"].items():
            if isinstance(entry, dict) and entry.get("state") != "completed":
                rows.append({"job_id": job_id, **entry})
    return {"schema_version": 1, "errors": rows}


def _review_members(root: Path, stage_id: int) -> dict[str, bytes]:
    records = _job_records(root)
    state = _read_json(root / "stage_state.json", {})
    manifest = _read_json(root / "campaign_manifest.json", {})
    if not isinstance(state, dict):
        state = {}
    summary = {
        "schema_version": 1,
        "stage_id": stage_id,
        "completed_stage": state.get("completed_stage", 0),
        "job_count": len(records),
        "preprocessing_status": state.get("preprocessing_status"),
        "state": state,
    }
    members = {
        "stage_summary.json": _json_bytes(summary),
        "decision_table.csv": _csv_bytes(_decision_table(state)),
        "model_metrics.csv": _csv_bytes(_model_metrics(records)),
        "segment_metrics.csv": _csv_bytes(_segment_metrics(records)),
        "learning_curves.csv": _csv_bytes(_learning_curves(records)),
        "resource_usage.csv": _csv_bytes(_resource_usage(records)),
        "validation_predictions.csv.gz": _gzip_csv_bytes(
            _wide_predictions(records)
        ),
        "errors.json": _json_bytes(_errors(manifest)),
        "run.log": (
            (root / "run.log").read_bytes()
            if (root / "run.log").is_file()
            else b""
        ),
    }
    artifact_manifest = {
        "schema_version": 1,
        "files": [
            {
                "path": name,
                "size_bytes": len(payload),
                "sha256": sha256(payload).hexdigest(),
            }
            for name, payload in sorted(members.items())
        ],
    }
    members["artifact_manifest.json"] = _json_bytes(artifact_manifest)
    return members


def _write_resume_zip(
    path: Path,
    root: Path, *, stage_id: int, campaign_id: str
) -> None:
    manifest_path = root / "campaign_manifest.json"
    if not manifest_path.is_file():
        manifest_bytes = _json_bytes({"schema_version": 1, "jobs": {}})
    else:
        manifest_bytes = manifest_path.read_bytes()
    small_members = {
        "campaign_manifest.json": manifest_bytes,
        "stage_state.json": (
            (root / "stage_state.json").read_bytes()
            if (root / "stage_state.json").is_file()
            else _json_bytes({"completed_stage": 0})
        ),
    }
    resumable_paths: list[Path] = []
    jobs_root = root / "jobs"
    if jobs_root.is_dir():
        resumable_paths.extend(
            source
            for source in jobs_root.rglob("*")
            if source.is_file()
            and source.suffix not in {".pt", ".cbm"}
            and source.name not in {"checkpoint_meta.json"}
        )
    workers_root = root / "workers"
    if workers_root.is_dir():
        resumable_paths.extend(
            source for source in workers_root.rglob("*") if source.is_file()
        )
    file_records = [
        {
            "path": name,
            "size_bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
        }
        for name, payload in sorted(small_members.items())
    ]
    for source in sorted(resumable_paths):
        digest = sha256()
        with source.open("rb") as reader:
            while block := reader.read(1024 * 1024):
                digest.update(block)
        file_records.append(
            {
                "path": source.relative_to(root).as_posix(),
                "size_bytes": source.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    small_members["resume_metadata.json"] = _json_bytes(
        {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "completed_stage": stage_id,
            "manifest_sha256": sha256(manifest_bytes).hexdigest(),
            "files": file_records,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with ZipFile(temporary, "w") as archive:
        for name, payload in sorted(small_members.items()):
            archive.writestr(_zip_info(name), payload)
        for source in sorted(resumable_paths):
            name = source.relative_to(root).as_posix()
            with source.open("rb") as reader, archive.open(
                _zip_info(name), "w"
            ) as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
    temporary.replace(path)


def write_stage_bundles(
    *,
    campaign_root: str | Path,
    stage_id: int,
    campaign_id: str,
) -> StageBundles:
    """Write one restart bundle and one small cumulative review bundle."""

    if stage_id not in range(0, 6):
        raise ValueError("stage_id must be between 0 and 5")
    root = Path(campaign_root).resolve()
    resume = root.parent / f"preprocessing_stage_{stage_id:02d}_resume_bundle.zip"
    review = root.parent / f"preprocessing_stage_{stage_id:02d}_review_bundle.zip"
    _write_resume_zip(
        resume,
        root,
        stage_id=stage_id,
        campaign_id=campaign_id,
    )
    review_members = _review_members(root, stage_id)
    _write_zip(review, review_members)
    final_review = None
    if stage_id == 5:
        final_review = root.parent / "preprocessing_campaign_final_review_bundle.zip"
        _write_zip(final_review, review_members)
    return StageBundles(resume, review, final_review)
