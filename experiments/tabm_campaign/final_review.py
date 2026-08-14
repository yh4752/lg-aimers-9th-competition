from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import time
from typing import Mapping

import numpy as np
import pandas as pd

from competition_rules.code_gate import inspect_inference_source
from competition_rules.contract import load_policy, policy_digest

from .artifacts import BundlePaths, StageEvidence, write_stage_bundles
from .contracts import Campaign
from .dependency_probe import run_clean_install_probe, run_python_probe
from .runner import StageRunResult
from .version_d import canonical_json, file_sha256, load_version_d_contract


class FinalReviewError(RuntimeError):
    """Raised when Version D cannot produce accepted review-only evidence."""


def final_epoch_count() -> int:
    return 3


def synthetic_scale_frame(
    sample: pd.DataFrame,
    *,
    row_count: int = 245_789,
) -> pd.DataFrame:
    if sample.empty or "row_id" not in sample or row_count <= 0:
        raise FinalReviewError(
            "scale workload requires a non-empty public sample with row_id"
        )
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
        raise FinalReviewError(
            f"official {name} candidate count must be 1; found={len(candidates)}"
        )
    return candidates[0]


def validate_frozen_manifest(artifact_root: Path) -> dict[str, object]:
    path = artifact_root / "inference_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise FinalReviewError(f"cannot read frozen inference manifest: {error}") from error
    if manifest.get("fit_scope") != "official_train_2019_2024_only":
        raise FinalReviewError("frozen inference fit scope differs")
    if manifest.get("epochs") != 3:
        raise FinalReviewError("frozen inference epoch count differs")
    if manifest.get("seeds") != [3407]:
        raise FinalReviewError("frozen inference seed differs")
    if manifest.get("scheduler") != "constant":
        raise FinalReviewError("frozen inference scheduler differs")
    members = manifest.get("members")
    if (
        not isinstance(members, list)
        or len(members) != 1
        or not isinstance(members[0], dict)
        or members[0].get("seed") != 3407
    ):
        raise FinalReviewError("frozen inference member differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise FinalReviewError("frozen inference file manifest differs")
    for name, expected in files.items():
        source = artifact_root / str(name)
        if source.is_symlink() or not source.is_file() or file_sha256(source) != expected:
            raise FinalReviewError(f"frozen inference artifact differs: {name}")
    return manifest


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def review_frozen_artifact(
    *,
    artifact_root: Path,
    review_data_dir: Path,
    output_dir: Path,
    prior_manifest_sha256: str,
    version_d_contract_sha256: str,
    fit_report: Mapping[str, object],
    runtime_versions: Mapping[str, str],
) -> StageRunResult:
    manifest = validate_frozen_manifest(artifact_root)
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or identity.get("contract_sha256") != version_d_contract_sha256:
        raise FinalReviewError("frozen inference contract identity differs")
    artifact_hashes_before = _artifact_hashes(artifact_root)
    project_root = Path(__file__).resolve().parents[2]
    policy = load_policy(
        project_root / "competition_rules/policy.json",
        project_root=project_root,
    )
    review_path = project_root / "reports/rules/2026-08-14-policy-review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("policy_sha256") != policy_digest(policy) or review.get("verdict") != "unchanged":
        raise FinalReviewError("reviewed policy identity differs")
    inference_source = Path(__file__).with_name("inference_runtime.py")
    source_gate = inspect_inference_source([inference_source], project_root=project_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    dependency = run_clean_install_probe(
        Path(__file__).with_name("requirements-kaggle.txt"),
        output_dir / "dependency_probe",
        timeout_seconds=480,
    )
    if dependency.status != "passed":
        raise FinalReviewError("clean dependency installation gate failed")

    public_test = pd.read_csv(_find_one(review_data_dir, "test.csv")).head(5)
    probe_frame_path = output_dir / "dependency_probe" / "five_rows.csv"
    public_test.to_csv(probe_frame_path, index=False)
    probe_python = (
        output_dir / "dependency_probe" / "dependency_probe_venv" / "bin" / "python"
    )
    probe_script = (
        "import pandas as pd; "
        "from experiments.tabm_campaign.inference_runtime import FrozenTabMPredictor; "
        f"frame=pd.read_csv({str(probe_frame_path)!r}); "
        f"predictor=FrozenTabMPredictor({str(artifact_root)!r}); "
        "probability=predictor.predict_batch(frame); "
        "assert probability.shape==(5,); "
        "print(','.join(f'{value:.8f}' for value in probability))"
    )
    frozen_sample_probe = run_python_probe(
        probe_python,
        probe_script,
        environment={"PYTHONPATH": str(project_root)},
    )
    if frozen_sample_probe["status"] != "passed":
        raise FinalReviewError("clean-environment five-row inference probe failed")

    from .inference_runtime import FrozenTabMPredictor, audit_frozen_predictor

    predictor = FrozenTabMPredictor(artifact_root)
    independence = audit_frozen_predictor(predictor, public_test)
    print("VERSION_D_INDEPENDENCE_PASSED", flush=True)

    scale = synthetic_scale_frame(public_test)
    started = time.monotonic()
    probability = predictor.predict_batch(scale, batch_size=2048)
    inference_seconds = time.monotonic() - started
    contract = load_version_d_contract()
    limits = contract["limits"]
    if (
        inference_seconds > float(limits["inference_seconds"])
        or probability.shape != (245_789,)
    ):
        raise FinalReviewError("synthetic production inference gate failed")
    print(
        f"VERSION_D_SCALE_GATE_PASSED rows=245789 seconds={inference_seconds:.3f}",
        flush=True,
    )

    try:
        import torch

        peak_gpu_bytes = int(torch.cuda.max_memory_allocated())
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
    except Exception:
        peak_gpu_bytes = peak_reserved_bytes = 0
    if peak_gpu_bytes > int(limits["gpu_bytes"]):
        raise FinalReviewError("GPU allocation exceeds the safety gate")
    peak_rss_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_rss_bytes = peak_rss_raw if os.uname().sysname == "Darwin" else peak_rss_raw * 1024
    if peak_rss_bytes > int(limits["rss_bytes"]):
        raise FinalReviewError("process peak RSS exceeds the safety gate")
    artifact_bytes = sum(
        path.stat().st_size for path in artifact_root.rglob("*") if path.is_file()
    )
    if artifact_bytes > int(limits["artifact_bytes"]):
        raise FinalReviewError("frozen artifacts exceed the size gate")
    source_gate_after = inspect_inference_source(
        [inference_source], project_root=project_root
    )
    artifact_hashes_after = _artifact_hashes(artifact_root)
    if source_gate_after != source_gate:
        raise FinalReviewError("inference source changed during Version D")
    if artifact_hashes_after != artifact_hashes_before:
        raise FinalReviewError("frozen artifacts changed during Version D")

    state = {
        "version": "D",
        "review_only": True,
        "acceptance_status": "review_ready",
        "fixed_epochs": final_epoch_count(),
        "policy_version": policy["policy_version"],
        "policy_sha256": policy_digest(policy),
        "policy_review_sha256": file_sha256(review_path),
        "source_gate": source_gate,
        "fit_report": dict(fit_report),
        "runtime_versions": dict(runtime_versions),
        "artifact_sha256": artifact_hashes_before,
        "independence": asdict(independence),
        "scale": {
            "rows": len(scale),
            "elapsed_seconds": inference_seconds,
            "probability_sha256": sha256(
                np.asarray(probability, dtype="float32").tobytes()
            ).hexdigest(),
            "peak_gpu_allocated_bytes": peak_gpu_bytes,
            "peak_gpu_reserved_bytes": peak_reserved_bytes,
            "peak_rss_bytes": peak_rss_bytes,
        },
        "dependency_probe": asdict(dependency),
        "frozen_sample_probe": frozen_sample_probe,
        "artifact_bytes": artifact_bytes,
    }
    review_members = {
        "final_review.json": canonical_json(state),
        "policy/policy.json": (project_root / "competition_rules/policy.json").read_bytes(),
        "policy/policy_review.json": review_path.read_bytes(),
    }
    for path in artifact_root.rglob("*"):
        if path.is_file():
            relative = path.relative_to(artifact_root).as_posix()
            review_members[f"frozen_inference/{relative}"] = path.read_bytes()
    bundles = write_stage_bundles(
        output_dir,
        StageEvidence(
            "D",
            version_d_contract_sha256,
            prior_manifest_sha256,
            review_members,
            {},
        ),
    )
    final_path = output_dir / "tabm_hand_matchup_final_review_bundle.zip"
    os.replace(bundles.review, final_path)
    final_bundles = BundlePaths(
        final_path,
        None,
        file_sha256(final_path),
        None,
        bundles.manifest_sha256,
    )
    print(f"VERSION_D_REVIEW_READY path={final_path}", flush=True)
    return StageRunResult("D", final_bundles, ("single_s3407",), (), ())


def run_final_review(
    campaign: Campaign,
    prior: dict[str, object],
    data_dir: Path,
    output_dir: Path,
    prior_manifest_sha256: str | None,
    campaign_config_sha256: str,
    training_deadline_unix: float,
) -> StageRunResult:
    raise FinalReviewError(
        "Version D requires the sealed Colab Version D entrypoint"
    )
