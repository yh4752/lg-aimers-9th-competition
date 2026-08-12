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


def test_stage_five_never_requires_sixth_version_for_short_training() -> None:
    result = finalize_stage_five(
        baseline={"completed_epochs": 9, "validation_curve": [[8, 0.25]]},
        candidate={"completed_epochs": 10, "validation_curve": [[9, 0.24]]},
    )

    assert result.status == "inconclusive"
    assert result.campaign_terminal is True
