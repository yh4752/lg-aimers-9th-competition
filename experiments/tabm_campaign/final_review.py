from __future__ import annotations

import json
import math
import os
import resource
import shutil
import statistics
import tempfile
import time
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from competition_rules.code_gate import inspect_inference_source
from competition_rules.contract import load_policy, policy_digest

from .artifacts import BundlePaths, StageEvidence, write_stage_bundles
from .contracts import Campaign
from .dependency_probe import run_clean_install_probe
from .runner import StageRunResult


class FinalReviewError(RuntimeError):
    """Raised when Version D cannot produce accepted review-only evidence."""


def fixed_epoch_count(best_epochs: list[int] | tuple[int, ...]) -> int:
    if not best_epochs or any(isinstance(value, bool) or int(value) < 0 for value in best_epochs):
        raise FinalReviewError("best epochs must be a non-empty collection of non-negative indices")
    value = round(statistics.median(int(epoch) + 1 for epoch in best_epochs))
    return min(40, max(2, int(value)))


def synthetic_scale_frame(sample: pd.DataFrame, *, row_count: int = 245_789) -> pd.DataFrame:
    if sample.empty or "row_id" not in sample or row_count <= 0:
        raise FinalReviewError("scale workload requires a non-empty public sample with row_id")
    positions = np.arange(row_count, dtype="int64") % len(sample)
    output = sample.iloc[positions].reset_index(drop=True).copy()
    width = max(6, len(str(row_count - 1)))
    output["row_id"] = [f"synthetic_{index:0{width}d}" for index in range(row_count)]
    return output


def _find_one(root: Path, name: str) -> Path:
    direct = root / name
    candidates = [direct] if direct.is_file() else sorted(root.rglob(name))
    candidates = [path for path in candidates if path.is_file()]
    if len(candidates) != 1:
        raise FinalReviewError(f"official {name} candidate count must be 1; found={len(candidates)}")
    return candidates[0]


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def run_final_review(
    campaign: Campaign,
    prior: dict[str, object],
    data_dir: Path,
    output_dir: Path,
    prior_manifest_sha256: str | None,
    campaign_config_sha256: str,
) -> StageRunResult:
    """Fit final members and emit review evidence only; package creation is absent."""

    if prior_manifest_sha256 is None:
        raise FinalReviewError("Version D requires a trusted Version C manifest")
    final_members = prior.get("final_members")
    champion = prior.get("champion")
    if not isinstance(final_members, list) or not final_members or not isinstance(champion, dict):
        raise FinalReviewError("Version D requires the Version C champion and final members")
    best_epochs = [int(member["best_epoch"]) for member in final_members if member.get("best_epoch") is not None]
    epochs = fixed_epoch_count(best_epochs)
    project_root = Path(__file__).resolve().parents[2]
    policy = load_policy(project_root / "competition_rules/policy.json", project_root=project_root)
    source_gate = inspect_inference_source([Path(__file__).with_name("inference_runtime.py")], project_root=project_root)
    review_path = project_root / "reports/rules/2026-08-13-policy-review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("policy_sha256") != policy_digest(policy) or review.get("verdict") != "unchanged":
        raise FinalReviewError("reviewed policy identity differs")

    # Heavy final fitting is deliberately delegated to the user-owned Kaggle
    # runtime. If its frozen fitter is unavailable, D cannot claim acceptance.
    from .final_training import fit_final_members

    output_dir.mkdir(parents=True, exist_ok=True)
    dependency = run_clean_install_probe(
        Path(__file__).with_name("requirements-kaggle.txt"),
        output_dir / "dependency_probe",
    )
    if dependency.status != "passed":
        raise FinalReviewError("clean dependency installation gate failed")
    artifact_root = output_dir / "frozen_inference"
    fit_report = fit_final_members(
        data_dir=data_dir,
        artifact_root=artifact_root,
        champion=champion,
        member_seeds=[int(member["seed"]) for member in final_members],
        epochs=epochs,
    )
    from .inference_runtime import FrozenTabMPredictor, audit_frozen_predictor

    sample_path = _find_one(data_dir, "test.csv")
    public_test = pd.read_csv(sample_path).head(5)
    predictor = FrozenTabMPredictor(artifact_root)
    independence = audit_frozen_predictor(predictor, public_test)
    source_gate_after = inspect_inference_source(
        [Path(__file__).with_name("inference_runtime.py")], project_root=project_root
    )
    if source_gate_after != source_gate:
        raise FinalReviewError("inference source changed during Version D")
    scale = synthetic_scale_frame(public_test)
    started = time.monotonic()
    probability = predictor.predict_batch(scale, batch_size=2048)
    inference_seconds = time.monotonic() - started
    if inference_seconds > 480 or probability.shape != (245_789,):
        raise FinalReviewError("synthetic production inference gate failed")
    try:
        import torch

        peak_gpu_bytes = int(torch.cuda.max_memory_allocated())
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
    except Exception:
        peak_gpu_bytes = peak_reserved_bytes = 0
    if peak_gpu_bytes > 20 * 1024**3:
        raise FinalReviewError("GPU allocation exceeds the 20 GiB safety gate")
    peak_rss_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_rss_bytes = peak_rss_raw if os.uname().sysname == "Darwin" else peak_rss_raw * 1024
    if peak_rss_bytes > 22_000_000_000:
        raise FinalReviewError("process peak RSS exceeds the 22 GB safety gate")
    artifact_bytes = sum(path.stat().st_size for path in artifact_root.rglob("*") if path.is_file())
    if artifact_bytes > 2_000_000_000:
        raise FinalReviewError("projected compressed artifacts exceed 2 GB")
    state = {
        "version": "D",
        "review_only": True,
        "acceptance_status": "review_ready",
        "fixed_epochs": epochs,
        "policy_version": policy["policy_version"],
        "policy_sha256": policy_digest(policy),
        "policy_review_sha256": _sha(review_path),
        "source_gate": source_gate,
        "fit_report": fit_report,
        "independence": asdict(independence),
        "scale": {
            "rows": len(scale),
            "elapsed_seconds": inference_seconds,
            "probability_sha256": sha256(np.asarray(probability, dtype="float32").tobytes()).hexdigest(),
            "peak_gpu_allocated_bytes": peak_gpu_bytes,
            "peak_gpu_reserved_bytes": peak_reserved_bytes,
            "peak_rss_bytes": peak_rss_bytes,
        },
        "dependency_probe": asdict(dependency),
        "artifact_bytes": artifact_bytes,
    }
    review_members = {
        "final_review.json": _canonical_json(state),
        "policy/policy.json": (project_root / "competition_rules/policy.json").read_bytes(),
        "policy/policy_review.json": review_path.read_bytes(),
    }
    for path in artifact_root.rglob("*"):
        if path.is_file():
            review_members[f"frozen_inference/{path.relative_to(artifact_root).as_posix()}"] = path.read_bytes()
    bundles = write_stage_bundles(
        output_dir,
        StageEvidence("D", campaign_config_sha256, prior_manifest_sha256, review_members, {}),
    )
    final_path = output_dir / "tabm_hand_matchup_final_review_bundle.zip"
    os.replace(bundles.review, final_path)
    bundles = BundlePaths(final_path, None, _sha(final_path), None, bundles.manifest_sha256)
    print(f"BUNDLE_SUCCESS version=D review={final_path} resume=None", flush=True)
    return StageRunResult("D", bundles, tuple(member["candidate_id"] for member in final_members), (), ())
