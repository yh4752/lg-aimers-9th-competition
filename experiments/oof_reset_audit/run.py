from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import secrets
from typing import Sequence
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd

from .metrics import (
    align_pair, block_bootstrap_interval, calibration_deciles, paired_metrics,
    segment_metrics, single_model_metrics,
)
from .types import ArtifactRecord, PredictionSet, TrustClass


@dataclass(frozen=True)
class AuditResult:
    output_dir: Path
    result_zip: Path
    output_files: tuple[str, ...]


_REPORTS = (
    "artifact_inventory.json", "audit_summary.md", "calibration_deciles.csv",
    "correlation_matrix.csv", "model_comparison.csv", "next_experiment.json",
    "paired_comparison.csv", "segment_diagnostics.csv",
)


def _new_run_directory(root: Path) -> Path:
    if root.is_symlink():
        raise ValueError("output root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError("output root must be a directory")
    for _ in range(20):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = root / f"oof_reset_audit_{stamp}_{secrets.token_hex(4)}"
        try:
            candidate.mkdir(mode=0o700)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("cannot create a unique audit directory")


def _atomic_bytes(path: Path, value: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _anchors(predictions: tuple[PredictionSet, ...]) -> dict[str, PredictionSet]:
    return {
        item.fold: item for item in predictions
        if item.model_id == "tabm_stage_c_seed_3407" and item.trust is TrustClass.RULE_SAFE
    }


def _paired_class(item: PredictionSet, anchors: dict[str, PredictionSet]) -> str:
    if item.trust is TrustClass.QUARANTINED_DIAGNOSTIC:
        return "quarantined_diagnostic"
    anchor = anchors.get(item.fold)
    if anchor is None or anchor.model_id == item.model_id:
        return "descriptive_only"
    try:
        align_pair(anchor, item)
    except ValueError:
        return "descriptive_only"
    return "paired"


def _tables(predictions: tuple[PredictionSet, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anchors = _anchors(predictions)
    model_rows = []
    calibration = []
    paired_rows: list[dict[str, object]] = []
    segment_rows: list[pd.DataFrame] = []
    correlation_rows: list[dict[str, object]] = []
    for item in sorted(predictions, key=lambda row: (row.model_id, row.fold)):
        metrics = single_model_metrics(item)
        metrics["comparison_class"] = _paired_class(item, anchors)
        model_rows.append(metrics)
        calibration.append(calibration_deciles(item))
        anchor = anchors.get(item.fold)
        if anchor is None or anchor.model_id == item.model_id:
            continue
        try:
            base = paired_metrics(anchor, item)
            aligned = align_pair(anchor, item)
        except ValueError:
            continue
        correlation_rows.append({
            key: base[key] for key in (
                "anchor_model_id", "candidate_model_id", "fold", "rows",
                "prediction_correlation", "loss_correlation", "residual_correlation",
            )
        } | {"trust": item.trust.value})
        if item.trust is TrustClass.QUARANTINED_DIAGNOSTIC:
            continue
        target = aligned["target"].to_numpy("float64")
        anchor_p = aligned["anchor_probability"].to_numpy("float64")
        candidate_p = aligned["candidate_probability"].to_numpy("float64")
        anchor_loss = np.square(anchor_p - target)
        candidate_loss = np.square(candidate_p - target)
        blocks = aligned.get("game_month", pd.Series(["unknown"] * len(aligned))).astype(str)
        interval = block_bootstrap_interval(pd.DataFrame({
            "block": [f"{item.fold}:{value}" for value in blocks],
            "loss_delta": anchor_loss - candidate_loss,
        }))
        paired_rows.append(base | {
            "comparison": "candidate", "candidate_weight": 1.0,
            "interval_status": interval["status"],
            "interval_lower": interval.get("lower"), "interval_upper": interval.get("upper"),
        })
        segment_rows.append(segment_metrics(anchor, item))
        for weight in (0.25, 0.5, 0.75):
            blended = (1.0 - weight) * anchor_p + weight * candidate_p
            paired_rows.append({
                "anchor_model_id": anchor.model_id, "candidate_model_id": item.model_id,
                "fold": item.fold, "rows": len(aligned), "anchor_brier": float(anchor_loss.mean()),
                "candidate_brier": float(np.square(blended - target).mean()),
                "gain_vs_anchor": float((anchor_loss - np.square(blended - target)).mean()),
                "prediction_correlation": base["prediction_correlation"],
                "loss_correlation": base["loss_correlation"],
                "residual_correlation": base["residual_correlation"],
                "comparison": "fixed_blend", "candidate_weight": weight,
                "interval_status": "not_computed", "interval_lower": None, "interval_upper": None,
            })
    model_columns = [
        "model_id", "fold", "trust", "rows", "target_mean", "prediction_mean",
        "prediction_std", "prediction_min", "prediction_max", "brier", "local_bss",
        "roc_auc", "log_loss", "ece_10", "comparison_class",
    ]
    paired_columns = [
        "anchor_model_id", "candidate_model_id", "fold", "rows", "anchor_brier",
        "candidate_brier", "gain_vs_anchor", "prediction_correlation", "loss_correlation",
        "residual_correlation", "comparison", "candidate_weight", "interval_status",
        "interval_lower", "interval_upper",
    ]
    correlation_columns = [
        "anchor_model_id", "candidate_model_id", "fold", "rows", "prediction_correlation",
        "loss_correlation", "residual_correlation", "trust",
    ]
    segment_columns = [
        "anchor_model_id", "candidate_model_id", "fold", "segment", "level", "rows",
        "anchor_brier", "candidate_brier", "regression", "eligible",
    ]
    calibration_columns = ["model_id", "fold", "decile", "rows", "target_mean", "prediction_mean"]
    return (
        pd.DataFrame(model_rows, columns=model_columns),
        pd.DataFrame(paired_rows, columns=paired_columns),
        pd.concat(segment_rows, ignore_index=True) if segment_rows else pd.DataFrame(columns=segment_columns),
        pd.DataFrame(correlation_rows, columns=correlation_columns),
        pd.concat(calibration, ignore_index=True) if calibration else pd.DataFrame(columns=calibration_columns),
    )


def _diagnose(
    predictions: tuple[PredictionSet, ...], inventory: tuple[ArtifactRecord, ...],
    model: pd.DataFrame, paired: pd.DataFrame, segments: pd.DataFrame, correlations: pd.DataFrame,
) -> dict[str, object]:
    safe = sorted(set(model.loc[model["comparison_class"] == "paired", "model_id"]))
    quarantined = sorted({row.model_id for row in predictions if row.trust is TrustClass.QUARANTINED_DIAGNOSTIC})
    missing = sorted({row.role.value for row in inventory if row.status == "missing_evidence" and row.role.value != "unknown"})

    blend_rows = paired[paired["comparison"] == "fixed_blend"] if not paired.empty else paired
    blend_candidates: list[str] = []
    for candidate, group in blend_rows.groupby("candidate_model_id"):
        by_fold = group.groupby("fold")["gain_vs_anchor"].max()
        candidate_segments = segments[segments["candidate_model_id"] == candidate]
        maximum_regression = float(candidate_segments["regression"].max()) if not candidate_segments.empty else 0.0
        if len(by_fold) >= 2 and bool((by_fold > 0).all()) and maximum_regression <= 0.00075:
            blend_candidates.append(str(candidate))

    common_direction = False
    fold_order = ["2022->2023", "2023->2024"]
    changes = []
    calibration_changes = []
    for _, group in model[model["trust"] == TrustClass.RULE_SAFE.value].groupby("model_id"):
        indexed = group.set_index("fold")
        if all(fold in indexed.index for fold in fold_order):
            changes.append(float(indexed.loc[fold_order[1], "brier"] - indexed.loc[fold_order[0], "brier"]))
            old_gap = float(indexed.loc[fold_order[0], "prediction_mean"] - indexed.loc[fold_order[0], "target_mean"])
            latest_gap = float(indexed.loc[fold_order[1], "prediction_mean"] - indexed.loc[fold_order[1], "target_mean"])
            calibration_changes.append(latest_gap - old_gap)
    high_residual = bool(not correlations.empty and correlations.loc[correlations["trust"] == TrustClass.RULE_SAFE.value, "residual_correlation"].dropna().gt(0.9).any())
    common_direction = len(changes) >= 2 and (all(value > 0 for value in changes) or all(value < 0 for value in changes))
    common_calibration = len(calibration_changes) >= 2 and (
        all(value > 1e-12 for value in calibration_changes)
        or all(value < -1e-12 for value in calibration_changes)
    )
    recency = common_direction and common_calibration and high_residual
    diverse = bool(blend_candidates)
    stop = not recency and not diverse
    supported = [name for name, value in (
        ("RECENCY_WEIGHTING", recency), ("DIVERSE_BLEND", diverse), ("STOP_AND_REFRAME", stop)
    ) if value]
    return {
        "schema_version": 1,
        "automatic_acceptance": False,
        "manual_review_required": True,
        "eligible_model_ids": safe,
        "quarantined_model_ids": quarantined,
        "missing_evidence": missing,
        "supported_directions": supported,
        "evidence": {
            "recency_weighting": {
                "supported": recency,
                "common_fold_direction": common_direction,
                "common_calibration_shift": common_calibration,
                "high_residual_correlation": high_residual,
            },
            "diverse_blend": {"supported": diverse, "stable_candidates": blend_candidates},
            "stop_and_reframe": {"supported": stop, "reason": "no stable safe improvement" if stop else "other evidence exists"},
        },
    }


def _summary(inventory: tuple[ArtifactRecord, ...], model: pd.DataFrame, paired: pd.DataFrame, decision: dict[str, object]) -> str:
    verified = sum(row.status == "verified" for row in inventory)
    missing = sum(row.status == "missing_evidence" for row in inventory)
    best = None if paired.empty else float(paired["gain_vs_anchor"].max())
    worst = None if paired.empty else float(paired["gain_vs_anchor"].min())
    return "\n".join((
        "# OOF 리셋 감사 결과", "",
        f"- 검증된 입력: {verified}개", f"- 누락 증거: {missing}개",
        f"- 단독 모델·fold 행: {len(model)}개", f"- paired 비교 행: {len(paired)}개",
        f"- 최선/최악 paired gain: {best} / {worst}",
        f"- 지원 방향: {', '.join(decision['supported_directions']) or '없음'}", "",
        "격리 진단 모델은 상관 분석에만 사용했으며 후보나 블렌드에 포함하지 않았다.",
        "이 감사는 모델을 자동 승인하지 않는다. 다음 실험은 사람이 근거를 검토한 뒤 확정한다.", "",
    ))


def _result_zip(run_dir: Path) -> Path:
    target = run_dir / "oof_reset_audit_results.zip"
    temporary = run_dir / f".{target.name}.tmp"
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            for name in _REPORTS:
                value = (run_dir / name).read_bytes()
                info = ZipInfo(name, (2024, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, value)
        temporary.replace(target)
        with ZipFile(target) as archive:
            if set(archive.namelist()) != set(_REPORTS) or archive.testzip() is not None:
                raise RuntimeError("result ZIP verification failed")
            if sum(info.file_size for info in archive.infolist()) > 64 * 1024**2:
                raise RuntimeError("result ZIP is too large")
        return target
    finally:
        if temporary.exists():
            temporary.unlink()


def run_audit(
    *, predictions: Sequence[PredictionSet], inventory: Sequence[ArtifactRecord], output_root: Path,
) -> AuditResult:
    values = tuple(predictions)
    records = tuple(inventory)
    run_dir = _new_run_directory(Path(output_root))
    model, paired, segments, correlations, calibration = _tables(values)
    decision = _diagnose(values, records, model, paired, segments, correlations)
    inventory_value = [{
        "path": str(row.path), "sha256": row.sha256, "role": row.role.value,
        "artifact_kind": row.artifact_kind, "status": row.status, "detail": row.detail,
    } for row in records]
    contents = {
        "artifact_inventory.json": _json_bytes(inventory_value),
        "audit_summary.md": _summary(records, model, paired, decision).encode("utf-8"),
        "calibration_deciles.csv": _csv_bytes(calibration),
        "correlation_matrix.csv": _csv_bytes(correlations),
        "model_comparison.csv": _csv_bytes(model),
        "next_experiment.json": _json_bytes(decision),
        "paired_comparison.csv": _csv_bytes(paired),
        "segment_diagnostics.csv": _csv_bytes(segments),
    }
    for name in _REPORTS:
        _atomic_bytes(run_dir / name, contents[name])
    result_zip = _result_zip(run_dir)
    return AuditResult(run_dir, result_zip, tuple(sorted((*_REPORTS, result_zip.name))))
