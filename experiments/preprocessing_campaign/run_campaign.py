"""Unified command line for the restartable preprocessing campaign."""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from types import MappingProxyType, SimpleNamespace
from typing import Mapping

import pandas as pd

from competition_rules.contract import assert_experiment_runnable

from experiments.catboost_preprocessing.campaign import (
    CatBoostPreprocessingRuntime,
    expand_catboost_jobs,
)
from experiments.independent_dl.preprocessing_campaign import (
    CampaignInterrupted,
    OfficialPreprocessingDLRuntime,
    run_preprocessing_campaign,
)
from experiments.independent_dl.preprocessing_evaluation import build_metric_rows
from experiments.independent_dl.preprocessing_contracts import (
    PreprocessingJob,
    PreprocessingSetting,
    load_preprocessing_campaign,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RULES_CONTRACT = Path(__file__).with_name("experiment_contract.json")
_DEFAULT_CONFIG = (
    _PROJECT_ROOT / "experiments/independent_dl/configs/preprocessing_ablation_v1.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    run = actions.add_parser("run", help="create or resume one registered wave")
    run.add_argument("--wave", choices=("a", "b", "c", "d", "e"), required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--data-dir", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    run.add_argument(
        "--max-jobs",
        type=int,
        help="attempt at most this many unfinished jobs in the current session",
    )
    run.add_argument(
        "--max-session-seconds",
        type=int,
        help="checkpoint and stop cleanly after this wall-time budget",
    )
    promote = actions.add_parser("promote", help="register the next wave from valid evidence")
    promote.add_argument("--from-wave", choices=("a", "b", "c", "d"), required=True)
    promote.add_argument("--config", required=True)
    promote.add_argument("--output-dir", required=True)
    status = actions.add_parser("status", help="show sealed counts or current progress")
    status.add_argument("--config")
    status.add_argument("--output-dir")
    status.add_argument("--dry-contract", action="store_true")
    summarize = actions.add_parser("summarize", help="write a terminal-state summary")
    summarize.add_argument("--output-dir", required=True)
    return parser


def _read_json(path: Path) -> object:
    if not path.is_file():
        raise RuntimeError(f"required file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_entry(root: Path, entry: Mapping[str, object]) -> bool:
    if entry.get("state") != "completed":
        return False
    for kind in ("metrics", "predictions"):
        relative = entry.get(f"{kind}_path")
        expected = entry.get(f"{kind}_sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            return False
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or _hash(path) != expected:
            return False
    return True


def _registered_path(root: Path) -> Path:
    return root / "registered_jobs.json"


def _serialize_job(job: PreprocessingJob) -> dict[str, object]:
    return {
        "job_id": job.job_id,
        "wave": job.wave,
        "anchor_id": job.anchor_id,
        "family": job.family,
        "profile_id": job.profile_id,
        "epochs": job.epochs,
        "model": dict(job.model),
        "training": dict(job.training),
        "feature_view": job.feature_view,
        "setting": {
            "setting_id": job.setting.setting_id,
            "profile": job.setting.profile,
            "components": list(job.setting.components),
        },
        "train_end_year": job.train_end_year,
        "valid_year": job.valid_year,
        "seed": job.seed,
    }


def _load_registered(root: Path) -> tuple[PreprocessingJob, ...]:
    path = _registered_path(root)
    if not path.is_file():
        return ()
    payload = _read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), list):
        raise RuntimeError("registered job file is invalid")
    jobs = []
    for raw in payload["jobs"]:
        setting_raw = raw["setting"]
        jobs.append(
            PreprocessingJob(
                job_id=str(raw["job_id"]),
                wave=str(raw["wave"]),
                anchor_id=str(raw["anchor_id"]),
                family=str(raw["family"]),
                profile_id=str(raw["profile_id"]),
                epochs=int(raw["epochs"]),
                model=MappingProxyType(dict(raw["model"])),
                training=MappingProxyType(dict(raw["training"])),
                feature_view=str(raw["feature_view"]),
                setting=PreprocessingSetting(
                    str(setting_raw["setting_id"]),
                    str(setting_raw["profile"]),
                    tuple(setting_raw["components"]),
                ),
                train_end_year=int(raw["train_end_year"]),
                valid_year=int(raw["valid_year"]),
                seed=int(raw["seed"]),
            )
        )
    return tuple(jobs)


def _wave_campaign(campaign, jobs) -> SimpleNamespace:
    return SimpleNamespace(
        campaign_id=campaign.campaign_id,
        protocol=campaign.protocol,
        registered_jobs=tuple(jobs),
    )


def _promote_a(campaign, root: Path) -> dict[str, object]:
    manifest = _read_json(root / "campaign_manifest.json")
    entries = manifest.get("jobs") if isinstance(manifest, dict) else None
    if not isinstance(entries, dict):
        raise RuntimeError("campaign manifest is invalid")
    expected = {job.job_id: job for job in campaign.wave_a_jobs}
    invalid = [
        job_id
        for job_id in expected
        if job_id not in entries or not _valid_entry(root, entries[job_id])
    ]
    if invalid:
        raise RuntimeError(
            f"Wave A promotion requires 760 hash-valid jobs; missing_or_invalid={len(invalid)}"
        )
    predictions = pd.concat(
        [
            pd.read_csv(root / entries[job_id]["predictions_path"])
            for job_id in expected
        ],
        ignore_index=True,
    )
    predictions["squared_error"] = (
        pd.to_numeric(predictions["probability"], errors="raise")
        - pd.to_numeric(predictions["target"], errors="raise")
    ) ** 2
    metrics = build_metric_rows(predictions)
    fold_metrics = root / "fold_metrics.csv"
    metrics.to_csv(fold_metrics.with_suffix(".csv.tmp"), index=False)
    os.replace(fold_metrics.with_suffix(".csv.tmp"), fold_metrics)
    selected: dict[str, set[str]] = {}
    decisions: list[dict[str, object]] = []
    for anchor_id in campaign.anchors:
        anchor = metrics.loc[metrics["anchor_id"].eq(anchor_id)]
        baseline = anchor.loc[anchor["preprocessing_id"].eq("dl_standard")]
        rankings = []
        for setting in campaign.dl_settings:
            if setting.setting_id == "dl_standard":
                continue
            candidate = anchor.loc[
                anchor["preprocessing_id"].eq(setting.setting_id)
            ].merge(
                baseline,
                on=["anchor_id", "seed", "fold"],
                suffixes=("_candidate", "_baseline"),
                validate="one_to_one",
            )
            weights = candidate["valid_rows_candidate"]
            delta = candidate["brier_candidate"] - candidate["brier_baseline"]
            weighted = float((delta * weights).sum() / weights.sum())
            improved = int((delta < 0).sum())
            oov_improved = any(
                int(
                    (
                        candidate[f"{column}_candidate"]
                        - candidate[f"{column}_baseline"]
                        < 0
                    ).sum()
                )
                >= 3
                and float(
                    (
                        (candidate[f"{column}_candidate"] - candidate[f"{column}_baseline"])
                        * weights
                    ).sum()
                    / weights.sum()
                )
                < 0
                for column in ("pitcher_oov_brier", "batter_oov_brier")
            )
            row_candidate = predictions.loc[
                predictions["anchor_id"].eq(anchor_id)
                & predictions["preprocessing_id"].eq(setting.setting_id)
            ]
            row_baseline = predictions.loc[
                predictions["anchor_id"].eq(anchor_id)
                & predictions["preprocessing_id"].eq("dl_standard")
            ]
            segment_pairs = row_candidate.merge(
                row_baseline,
                on=["row_id", "fold", "season", "game_type", "anchor_id", "seed"],
                suffixes=("_candidate", "_baseline"),
                validate="one_to_one",
            )
            segment_pairs["delta"] = (
                segment_pairs["squared_error_candidate"]
                - segment_pairs["squared_error_baseline"]
            )
            game_type_improved = any(
                float(group["delta"].mean()) < 0
                and int(
                    (
                        group.groupby("fold", observed=True)["delta"].mean() < 0
                    ).sum()
                )
                >= 3
                for _, group in segment_pairs.groupby("game_type", observed=True)
            )
            rankings.append(
                (
                    weighted,
                    -improved,
                    setting.setting_id,
                    oov_improved or game_type_improved,
                )
            )
        rankings.sort()
        chosen = {
            setting_id
            for weighted, negative_improved, setting_id, oov_improved in rankings
            if (weighted < 0 and -negative_improved >= 3) or oov_improved
        }
        chosen.update(item[2] for item in rankings[:2])
        chosen.add("dl_standard")
        selected[anchor_id] = chosen
        decisions.append(
            {"anchor_id": anchor_id, "selected": sorted(chosen), "ranking": rankings}
        )
    new_jobs: list[PreprocessingJob] = []
    templates = {
        (job.anchor_id, job.setting.setting_id, job.train_end_year, job.valid_year): job
        for job in campaign.wave_a_jobs
    }
    for anchor_id, setting_ids in selected.items():
        for setting_id in sorted(setting_ids):
            for train_end_year, valid_year in campaign.folds:
                template = templates[(anchor_id, setting_id, train_end_year, valid_year)]
                for seed in campaign.seeds[1:]:
                    new_jobs.append(
                        replace(
                            template,
                            job_id=(
                                f"b__{anchor_id}__{setting_id}__tr{train_end_year}__"
                                f"va{valid_year}__s{seed}"
                            ),
                            wave="b",
                            seed=seed,
                        )
                    )
    existing = _load_registered(root)
    combined = {job.job_id: job for job in (*existing, *new_jobs)}
    _atomic_json(
        _registered_path(root),
        {"schema_version": 1, "jobs": [_serialize_job(job) for job in combined.values()]},
    )
    payload = {
        "from_wave": "a",
        "input_manifest_sha256": _hash(root / "campaign_manifest.json"),
        "fold_metrics_sha256": _hash(fold_metrics),
        "registered_wave_b_jobs": len(new_jobs),
        "decisions": decisions,
    }
    _atomic_json(root / "promotion_decisions.json", payload)
    return payload


def _status_contract(config: str) -> dict[str, int]:
    campaign = load_preprocessing_campaign(config)
    return {
        "wave_a_jobs": len(campaign.wave_a_jobs),
        "catboost_wave_e_jobs": len(expand_catboost_jobs(campaign)),
        "dl_settings": len(campaign.dl_settings),
        "catboost_settings": len(campaign.catboost_settings),
        "folds": len(campaign.folds),
        "seeds": len(campaign.seeds),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "status" and args.dry_contract:
        if not args.config:
            raise RuntimeError("--config is required with --dry-contract")
        print(json.dumps(_status_contract(args.config), indent=2, sort_keys=True))
        return 0
    if args.action == "status":
        if not args.output_dir:
            raise RuntimeError("--output-dir is required")
        root = Path(args.output_dir).resolve()
        manifest = _read_json(root / "campaign_manifest.json")
        counts: dict[str, dict[str, int]] = {}
        for entry in manifest.get("jobs", {}).values():
            wave = str(entry.get("job", {}).get("wave", "unknown"))
            state = str(entry.get("state", "unknown"))
            counts.setdefault(wave, {}).setdefault(state, 0)
            counts[wave][state] += 1
        print(json.dumps({"waves": counts, "output_root": str(root)}, indent=2))
        return 0
    if args.action == "summarize":
        assert_experiment_runnable(
            project_root=_PROJECT_ROOT,
            contract_path=_RULES_CONTRACT,
            config_path=_DEFAULT_CONFIG,
        )
        root = Path(args.output_dir).resolve()
        manifest = _read_json(root / "campaign_manifest.json")
        entries = manifest.get("jobs", {})
        running = [job_id for job_id, entry in entries.items() if entry.get("state") in {"pending", "running"}]
        if running:
            raise RuntimeError(f"registered jobs are not terminal: {len(running)}")
        payload = {
            "campaign_id": manifest.get("campaign_id"),
            "completed": sum(entry.get("state") == "completed" for entry in entries.values()),
            "failed": sum(entry.get("state") == "failed" for entry in entries.values()),
            "manifest_sha256": _hash(root / "campaign_manifest.json"),
        }
        _atomic_json(root / "campaign_summary.json", payload)
        print(json.dumps(payload, indent=2))
        return 0
    assert_experiment_runnable(
        project_root=_PROJECT_ROOT,
        contract_path=_RULES_CONTRACT,
        config_path=args.config,
    )
    campaign = load_preprocessing_campaign(args.config)
    root = Path(args.output_dir).resolve()
    if args.action == "promote":
        if args.from_wave != "a":
            raise RuntimeError(
                "later-wave promotion requires the preceding user-run evidence; return the current result ZIP for review"
            )
        print(json.dumps(_promote_a(campaign, root), indent=2))
        return 0
    if args.wave == "a":
        jobs = campaign.wave_a_jobs
        runtime = OfficialPreprocessingDLRuntime(
            args.data_dir, cache_root=root / "feature_cache"
        )
    elif args.wave == "e":
        jobs = expand_catboost_jobs(campaign)
        runtime = CatBoostPreprocessingRuntime(args.data_dir, task_type=args.task_type)
    else:
        jobs = tuple(job for job in _load_registered(root) if job.wave == args.wave)
        if not jobs:
            raise RuntimeError(f"Wave {args.wave.upper()} has no promoted jobs")
        runtime = OfficialPreprocessingDLRuntime(
            args.data_dir, cache_root=root / "feature_cache"
        )
    if args.max_session_seconds is not None:
        if args.max_session_seconds < 1:
            raise RuntimeError("--max-session-seconds must be positive")
        os.environ["PREPROCESSING_SESSION_DEADLINE_UNIX"] = str(
            time.time() + args.max_session_seconds
        )
    try:
        summary = run_preprocessing_campaign(
            _wave_campaign(campaign, jobs), root, runtime, max_jobs=args.max_jobs
        )
    except CampaignInterrupted as error:
        print(
            json.dumps(
                {
                    "campaign_id": campaign.campaign_id,
                    "requested_wave": args.wave,
                    "state": "checkpointed_time_budget",
                    "message": str(error),
                    "output_root": str(root),
                },
                indent=2,
            )
        )
        return 0
    print(
        json.dumps(
            {
                "campaign_id": summary.campaign_id,
                "completed": len(summary.completed),
                "failed": len(summary.failed),
                "pending": len(summary.pending),
                "requested_wave": args.wave,
                "output_root": str(summary.output_root),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
