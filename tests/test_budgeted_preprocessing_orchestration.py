from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.preprocessing_campaign.budgeted_contracts import load_budgeted_campaign
from experiments.preprocessing_campaign.run_budgeted_campaign import (
    StageNeedsReview,
    build_stage_jobs,
    finalize_stage_five,
    inspect_resume_bundles,
    main,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/preprocessing_campaign/configs/budgeted_campaign_v1.json"


def _write_resume(
    path: Path,
    *,
    stage: int,
    manifest_hash: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.dumps({"stage": stage}, sort_keys=True).encode()
    observed = sha256(manifest).hexdigest()
    archive_path = path.with_suffix(".zip")
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("campaign_manifest.json", manifest)
        archive.writestr(
            "resume_metadata.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "campaign_id": "budgeted_preprocessing_campaign_v1",
                    "completed_stage": stage,
                    "manifest_sha256": manifest_hash or observed,
                    "files": [
                        {
                            "path": "campaign_manifest.json",
                            "size_bytes": len(manifest),
                            "sha256": observed,
                        }
                    ],
                }
            ),
        )
    return archive_path


def test_auto_mode_starts_stage_one_without_resume_bundle(tmp_path: Path) -> None:
    assert inspect_resume_bundles(tmp_path) is None


def test_auto_mode_selects_highest_hash_valid_resume_bundle(tmp_path: Path) -> None:
    _write_resume(tmp_path / "stage1_resume_bundle", stage=1)
    second = _write_resume(tmp_path / "stage2_resume_bundle", stage=2)

    result = inspect_resume_bundles(tmp_path)

    assert result is not None
    assert result.completed_stage == 2
    assert result.path == second


def test_auto_mode_selects_kaggle_extracted_resume_dataset(tmp_path: Path) -> None:
    archive_path = _write_resume(
        tmp_path / "upload" / "preprocessing_stage_00_fixed_resume_bundle",
        stage=0,
    )
    dataset = tmp_path / "input" / "preprocessing_stage_00_fixed_resume_bundle"
    dataset.mkdir(parents=True)
    with ZipFile(archive_path) as archive:
        archive.extractall(dataset)

    result = inspect_resume_bundles(tmp_path / "input")

    assert result is not None
    assert result.completed_stage == 0
    assert result.path == dataset


def test_auto_mode_selects_slugified_kaggle_resume_dataset(tmp_path: Path) -> None:
    archive_path = _write_resume(
        tmp_path / "upload" / "preprocessing_stage_00_fixed_resume_bundle",
        stage=0,
    )
    dataset = tmp_path / "input" / "preprocessing-stage-00-fixed-resume-bundle"
    dataset.mkdir(parents=True)
    with ZipFile(archive_path) as archive:
        archive.extractall(dataset)

    result = inspect_resume_bundles(tmp_path / "input")

    assert result is not None
    assert result.path == dataset


def test_invalid_extracted_resume_dataset_never_starts_fresh(tmp_path: Path) -> None:
    dataset = tmp_path / "input" / "preprocessing-stage-00-resume-bundle"
    dataset.mkdir(parents=True)
    (dataset / "resume_metadata.json").write_text("{}", encoding="utf-8")

    try:
        inspect_resume_bundles(tmp_path / "input")
    except StageNeedsReview as error:
        assert "invalid extracted resume" in str(error)
    else:
        raise AssertionError("an attached invalid resume must block a fresh campaign")


def test_conflicting_same_stage_bundles_need_review(tmp_path: Path) -> None:
    left = _write_resume(tmp_path / "a_resume_bundle", stage=2)
    right = tmp_path / "b_resume_bundle.zip"
    other = json.dumps({"stage": 2, "branch": "b"}, sort_keys=True).encode()
    with ZipFile(right, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("campaign_manifest.json", other)
        archive.writestr(
            "resume_metadata.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "campaign_id": "budgeted_preprocessing_campaign_v1",
                    "completed_stage": 2,
                    "manifest_sha256": sha256(other).hexdigest(),
                    "files": [
                        {
                            "path": "campaign_manifest.json",
                            "size_bytes": len(other),
                            "sha256": sha256(other).hexdigest(),
                        }
                    ],
                }
            ),
        )
    assert left != right

    try:
        inspect_resume_bundles(tmp_path)
    except StageNeedsReview as error:
        assert "conflicting" in str(error)
    else:
        raise AssertionError("conflicting bundles must not be selected arbitrarily")


def test_selected_stage_jobs_keep_model_and_training_configuration() -> None:
    campaign = load_budgeted_campaign(CONFIG)
    selected = campaign.stage_jobs(1)[1]
    state = {
        "selected_model": {
            "family": selected.family,
            "profile_id": selected.profile_id,
            "model": dict(selected.model),
            "training": dict(selected.training),
        }
    }

    jobs = build_stage_jobs(campaign, 2, state)

    assert len(jobs) == 4
    assert {job.family for job in jobs} == {"ft_transformer"}
    assert all(dict(job.model) == dict(selected.model) for job in jobs)
    assert all(dict(job.training) == dict(selected.training) for job in jobs)
    assert len({job.job_id for job in jobs}) == 4


def test_stage_three_includes_dl_and_catboost_single_ablations() -> None:
    campaign = load_budgeted_campaign(CONFIG)
    selected = campaign.stage_jobs(1)[0]
    state = {
        "selected_model": {
            "family": selected.family,
            "profile_id": selected.profile_id,
            "model": dict(selected.model),
            "training": dict(selected.training),
        }
    }

    jobs = build_stage_jobs(campaign, 3, state)

    assert sum(job.family == "tabm" for job in jobs) == 4
    assert sum(job.family == "catboost" for job in jobs) == 4
    assert {
        job.setting_id for job in jobs if job.family == "catboost"
    } == {
        "pitcher_smoothing_k100",
        "batter_smoothing_k250",
        "id_frequency_and_oov",
        "hand_matchup",
    }
    assert sum(job.max_seconds for job in jobs if job.family == "tabm") / 2 <= 4200
    assert sum(job.max_seconds for job in jobs if job.family == "catboost") / 2 <= 1200


def test_stage_three_id_frequency_tabm_disables_amp_only_for_that_job() -> None:
    campaign = load_budgeted_campaign(CONFIG)
    selected = campaign.stage_jobs(1)[0]
    state = {
        "selected_model": {
            "family": selected.family,
            "profile_id": selected.profile_id,
            "model": dict(selected.model),
            "training": dict(selected.training),
        }
    }

    jobs = build_stage_jobs(campaign, 3, state)
    dl_jobs = [job for job in jobs if job.family == "tabm"]
    treatment = next(
        job for job in dl_jobs if job.setting_id == "id_frequency_and_oov"
    )

    assert treatment.training["amp"] is False
    assert all(
        job.training["amp"] is True
        for job in dl_jobs
        if job.setting_id != "id_frequency_and_oov"
    )


def test_stage_four_worst_case_fits_before_archive_reserve() -> None:
    campaign = load_budgeted_campaign(CONFIG)
    selected = campaign.stage_jobs(1)[0]
    descriptor = lambda name, component: {
        "setting_id": name,
        "preprocessing_profile": "dl_standard",
        "components": [component],
        "delta": -0.001,
    }
    state = {
        "selected_model": {
            "family": selected.family,
            "profile_id": selected.profile_id,
            "model": dict(selected.model),
            "training": dict(selected.training),
        },
        "promoted_dl": [
            descriptor("first", "hand_matchup"),
            descriptor("second", "asof_count_log1p"),
        ],
        "promoted_catboost": [
            descriptor("first", "hand_matchup"),
            descriptor("second", "entity_frequency_and_oov"),
        ],
    }

    jobs = build_stage_jobs(campaign, 4, state)

    dl_jobs = [job for job in jobs if job.family != "catboost"]
    cat_jobs = [job for job in jobs if job.family == "catboost"]
    assert len(dl_jobs) == 4
    assert len(cat_jobs) == 7
    # Jobs are queued by family; two DL waves plus four CatBoost waves fit in 5,000 s.
    assert sum(job.max_seconds for job in dl_jobs) / 2 <= 3600
    assert ((len(cat_jobs) + 1) // 2) * max(
        job.max_seconds for job in cat_jobs
    ) <= 1400


def test_stage_five_never_requires_sixth_version_for_short_training() -> None:
    result = finalize_stage_five(
        baseline={"completed_epochs": 9, "validation_curve": [[8, 0.25]]},
        candidate={"completed_epochs": 10, "validation_curve": [[9, 0.24]]},
    )

    assert result.status == "inconclusive"
    assert result.campaign_terminal is True


def test_stage_five_compares_best_brier_only_over_common_epochs() -> None:
    result = finalize_stage_five(
        baseline={
            "completed_epochs": 12,
            "best_epoch": 9,
            "validation_curve": [[9, 0.25], [10, 0.20], [11, 0.19]],
        },
        candidate={
            "completed_epochs": 10,
            "best_epoch": 9,
            "validation_curve": [[8, 0.24], [9, 0.23]],
        },
    )

    assert result.status == "recommended"
    assert result.common_epochs == 10
    assert result.baseline_best_brier == 0.25
    assert result.candidate_best_brier == 0.23
    assert result.predictions_comparable is True


def test_stage_five_marks_segment_predictions_outside_common_range() -> None:
    result = finalize_stage_five(
        baseline={
            "completed_epochs": 12,
            "best_epoch": 11,
            "validation_curve": [[9, 0.25], [11, 0.20]],
        },
        candidate={
            "completed_epochs": 10,
            "best_epoch": 9,
            "validation_curve": [[9, 0.23]],
        },
    )

    assert result.status == "recommended"
    assert result.predictions_comparable is False


def test_status_dry_contract_reports_sealed_execution_shape(capsys) -> None:
    returncode = main(["status", "--config", str(CONFIG), "--dry-contract"])

    assert returncode == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "archive_reserve_seconds": 600,
        "campaign_id": "budgeted_preprocessing_campaign_v1",
        "gpu_workers": 2,
        "session_seconds": 6300,
        "stage_count": 5,
        "stop_new_jobs_seconds": 900,
    }
