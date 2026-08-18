from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import time

import pandas as pd
import pytest

from experiments.hierarchical_tabm.calibration import fit_h2
from experiments.hierarchical_tabm.artifacts import (
    CampaignEvidence,
    verify_candidate_delivery,
    write_campaign_bundles,
)
from experiments.hierarchical_tabm.contracts import DEFAULT_CONTRACT, build_jobs, load_contract
from experiments.hierarchical_tabm.metrics import CandidateDecision
from experiments.hierarchical_tabm.inference import CandidateValidation
from experiments.hierarchical_tabm.runner import (
    choose_final_epochs,
    fit_final_calibration,
    run_campaign,
)
from experiments.hierarchical_tabm.training import TrainingJobResult
from experiments.hierarchical_tabm.training import training_result_payload


def _validation_report(
    candidate_id: str,
    role: str,
    *,
    passed: bool,
    members: dict[str, Path],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "delivery_role": role,
        "passed": passed,
        "independence": None if not passed else {
            "row_count": 2,
            "features_exact": True,
            "maximum_probability_delta": 0.0,
            "state_before": "1" * 64,
            "state_after": "1" * 64,
            "batch_sizes": [1, 257, 2048],
        },
        "resources": None if not passed else {
            "row_count": 245789,
            "python_version": "3.11.15",
            "elapsed_seconds": 1.0,
            "peak_gpu_bytes": 1,
            "peak_rss_bytes": 1,
            "artifact_bytes": 1,
            "passed": True,
        },
        "validated_members": {
            name: sha256(path.read_bytes()).hexdigest() for name, path in members.items()
        },
        "failure": None if passed else "fixture_rejection",
    }


def test_final_epoch_is_row_weighted_median_clamped_to_contract() -> None:
    assert choose_final_epochs([(100, 1), (300, 9)], minimum=2, maximum=8) == 8
    assert choose_final_epochs([(300, 3), (100, 7)], minimum=2, maximum=8) == 3
    assert choose_final_epochs([(100, 1), (100, 9)], minimum=2, maximum=8) == 2


def test_final_calibration_uses_only_2023_and_2024_oof_after_selection() -> None:
    first = pd.DataFrame(
        {"row_id": ["a", "b"], "probability": [0.2, 0.8], "target": [0, 1]}
    )
    second = pd.DataFrame(
        {"row_id": ["c", "d"], "probability": [0.3, 0.7], "target": [0, 1]}
    )
    selection = SimpleNamespace(
        kind="H2",
        selected_regularization=0.01,
        state=fit_h2(first.probability.to_numpy(), first.target.to_numpy(), regularization=0.01, clip=1e-6),
    )
    state = fit_final_calibration(
        selection=selection,
        oof_by_fold={"2022->2023": first, "2023->2024": second},
    )
    expected = sha256("a\nb\nc\nd".encode()).hexdigest()
    assert state.fit_row_ids_sha256 == expected


class FakeRuntime:
    def __init__(self, root: Path, *, h1_status="strong") -> None:
        self.root = root
        self.calls = []
        self.bindings = MappingProxyType(
            {
                key: sha256(key.encode()).hexdigest()
                for key in (
                    "contract_sha256", "code_sha256", "environment_sha256",
                    "input_manifest_sha256", "train_sha256", "history_sha256",
                    "stage_c_delivery_sha256", "stage_c_review_sha256",
                    "stage_c_state_sha256", "anchor_2022_2023_sha256",
                    "anchor_2023_2024_sha256",
                )
            }
        )
        self.h1_status = h1_status

    def select_k(self, fit_rows, valid_rows):
        self.calls.append("select_k_2021_2022")
        return SimpleNamespace(selected_k=128.0, fold="2021->2022", scores={32.0: 0.2}, row_count=len(valid_rows))

    def run_oof(self, job, **kwargs):
        self.calls.append(f"oof_{job.train_end_year}_{job.valid_year}")
        out = kwargs["output_dir"]
        out.mkdir(parents=True)
        predictions = out / "predictions.csv"
        pd.DataFrame(
            {
                "row_id": [f"{job.valid_year}-r"], "target": [0],
                "probability": [0.4], "season": [job.valid_year], "game_month": [4],
                "game_type": ["R"], "count_state": ["0_0"],
                "hand_matchup": ["R_L"], "base_out_state": ["000_0"],
                "pitcher_known": ["known"], "batter_known": ["known"],
            }
        ).to_csv(predictions, index=False)
        checkpoint = out / "checkpoint.pt"; checkpoint.write_bytes(b"checkpoint")
        feature = out / "feature_state.json"; feature.write_text("{}")
        return TrainingJobResult(job.job_id, "oof", "completed", 100, 10, 2, 0.16, 3, predictions, checkpoint, None, feature, None)

    def calibrate(self, kind, **kwargs):
        self.calls.append(f"calibrate_{kind}")
        path = kwargs["output_path"]
        path.write_text("{}")
        return SimpleNamespace(kind=kind, selected_regularization=0.01, state=None, path=path)

    def decide(self, candidate_id, **kwargs):
        self.calls.append(f"decide_{candidate_id}")
        status = self.h1_status if candidate_id == "H1" else "rejected"
        final = status in {"strong", "accepted"}
        role = "final_candidate" if final else ("public_diagnostic_only" if status == "frontier" else None)
        return CandidateDecision(candidate_id, status, final, role, {"2022->2023": .2, "2023->2024": .2}, {"2022->2023": 0., "2023->2024": 0.}, 0., 0., "fixture")

    def run_full_fit(self, job, **kwargs):
        self.calls.append("full_fit_2019_2024")
        out = kwargs["output_dir"]; out.mkdir(parents=True)
        checkpoint = out / "checkpoint.pt"; checkpoint.write_bytes(b"checkpoint")
        model = out / "final_checkpoint.pt"; model.write_bytes(b"model")
        feature = out / "feature_state.json"; feature.write_text("{}")
        return TrainingJobResult(job.job_id, "full_fit", "completed", 200, None, None, None, kwargs["final_epochs"], None, checkpoint, model, feature, None)

    def validate_candidates(
        self, *, delivery_roles, output_dir, final_checkpoint_path,
        feature_state_path, calibration_paths, **kwargs
    ):
        self.calls.append("validate_candidates")
        reports = {}
        for candidate_id in delivery_roles:
            path = output_dir / f"independence_{candidate_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            members = {
                "model/final_checkpoint.pt": final_checkpoint_path,
                "state/feature_state.json": feature_state_path,
            }
            if candidate_id != "H1":
                members[f"state/calibration_{candidate_id}.json"] = calibration_paths[candidate_id]
            path.write_text(json.dumps(_validation_report(
                candidate_id, delivery_roles[candidate_id], passed=True, members=members
            )))
            reports[candidate_id] = path
        return CandidateValidation(
            MappingProxyType(dict(delivery_roles)), MappingProxyType(reports)
        )


def _verified(tmp_path: Path):
    data = tmp_path / "data"; data.mkdir()
    pd.DataFrame({"row_id": [f"{year}-r" for year in range(2019, 2025)], "season": range(2019, 2025), "control_success": [0,1,0,1,0,1]}).to_csv(data / "train.csv", index=False)
    return SimpleNamespace(training=SimpleNamespace(data_dir=data), anchor_predictions={})


def test_campaign_runs_preregistered_order_and_writes_validated_delivery(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    run = run_campaign(
        _verified(tmp_path), tmp_path / "run", resume_bundle=None,
        absolute_deadline=time.time() + 60, runtime=runtime,
        on_verified_resume=lambda path: None,
    )
    assert runtime.calls == [
        "select_k_2021_2022", "oof_2022_2023", "oof_2023_2024",
        "calibrate_H2", "calibrate_H3", "decide_H1", "decide_H2", "decide_H3",
        "full_fit_2019_2024", "validate_candidates",
    ]
    assert run.state.selected_k == 128.0
    assert run.state.status == "completed_candidate_delivery"
    assert run.candidate_delivery is not None and run.candidate_delivery.is_file()
    assert run.bundles.review.is_file() and run.bundles.resume.is_file()


def test_candidate_validation_failure_is_isolated(tmp_path: Path) -> None:
    class PartiallyAcceptedRuntime(FakeRuntime):
        def decide(self, candidate_id, **kwargs):
            self.calls.append(f"decide_{candidate_id}")
            status = "strong" if candidate_id == "H1" else (
                "accepted" if candidate_id == "H2" else "rejected"
            )
            role = "final_candidate" if status in {"strong", "accepted"} else None
            return CandidateDecision(
                candidate_id, status, role is not None, role,
                {"2022->2023": .2, "2023->2024": .2},
                {"2022->2023": 0., "2023->2024": 0.}, 0., 0., "fixture",
            )

        def validate_candidates(
            self, *, delivery_roles, output_dir, final_checkpoint_path,
            feature_state_path, calibration_paths, **kwargs
        ):
            self.calls.append("validate_candidates")
            reports = {}
            for candidate_id in delivery_roles:
                path = output_dir / f"independence_{candidate_id}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                members = {
                    "model/final_checkpoint.pt": final_checkpoint_path,
                    "state/feature_state.json": feature_state_path,
                }
                if candidate_id != "H1":
                    members[f"state/calibration_{candidate_id}.json"] = calibration_paths[candidate_id]
                path.write_text(json.dumps(_validation_report(
                    candidate_id, delivery_roles[candidate_id],
                    passed=candidate_id == "H2", members=members,
                )))
                reports[candidate_id] = path
            return CandidateValidation(
                MappingProxyType({"H2": "final_candidate"}), MappingProxyType(reports)
            )

    runtime = PartiallyAcceptedRuntime(tmp_path)
    run = run_campaign(
        _verified(tmp_path), tmp_path / "run", resume_bundle=None,
        absolute_deadline=time.time() + 60, runtime=runtime,
        on_verified_resume=lambda path: None,
    )
    assert run.candidate_delivery is not None
    manifest = verify_candidate_delivery(
        run.candidate_delivery, expected_bindings=runtime.bindings
    )
    assert manifest["delivery_candidate_ids"] == ["H2"]
    assert run.state.delivery_candidate_ids == ("H2",)


def test_no_eligible_candidate_skips_full_fit(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, h1_status="rejected")
    run = run_campaign(
        _verified(tmp_path), tmp_path / "run", resume_bundle=None,
        absolute_deadline=time.time() + 60, runtime=runtime,
        on_verified_resume=lambda path: None,
    )
    assert "full_fit_2019_2024" not in runtime.calls
    assert run.state.status == "completed_no_candidate"
    assert run.candidate_delivery is None


def test_resume_reuses_verified_k_and_completed_first_fold(tmp_path: Path) -> None:
    verified = _verified(tmp_path)
    source_runtime = FakeRuntime(tmp_path)
    first_job = build_jobs(load_contract())[0]
    first_dir = tmp_path / "partial-source" / "jobs" / first_job.job_id
    first_result = source_runtime.run_oof(first_job, output_dir=first_dir)
    (first_dir / "worker_result.json").write_text(
        json.dumps(training_result_payload(first_result), sort_keys=True, separators=(",", ":"))
    )
    root = tmp_path / "partial-source"
    k_path = root / "k_selection.json"
    k_path.write_text(
        json.dumps(
            {"fold": "2021->2022", "selected_k": 128.0, "scores": {"32.0": 0.2}, "row_count": 1}
        )
    )
    state_path = root / "stage_state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "running",
                "bindings": dict(source_runtime.bindings),
                "selected_k": 128.0,
                "completed_job_ids": [first_job.job_id],
                "active_job_id": None,
                "decisions": {},
                "delivery_candidate_ids": [],
                "final_epochs": None,
                "final_fit_completed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    log = root / "campaign.log"; log.write_text("CAMPAIGN_START\n")
    resume = write_campaign_bundles(
        CampaignEvidence(
            source_runtime.bindings, DEFAULT_CONTRACT, log, state_path, k_path,
            MappingProxyType({first_job.job_id: first_dir}),
            MappingProxyType({}), MappingProxyType({}), None,
        ),
        tmp_path / "partial-bundles",
    ).resume

    runtime = FakeRuntime(tmp_path)
    callbacks: list[Path] = []
    run_campaign(
        verified, tmp_path / "resumed-run", resume_bundle=resume,
        absolute_deadline=time.time() + 60, runtime=runtime,
        on_verified_resume=callbacks.append,
    )
    assert callbacks == [resume]
    assert "select_k_2021_2022" not in runtime.calls
    assert "oof_2022_2023" not in runtime.calls
    assert "oof_2023_2024" in runtime.calls
