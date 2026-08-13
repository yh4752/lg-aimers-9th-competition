"""Command-line entry point for the user-owned independent DL campaign."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from competition_rules.contract import assert_experiment_runnable

from .campaign import OfficialCampaignRuntime, run_campaign
from .contracts import load_campaign


_FAMILIES = ("tabm", "mlp_resnet", "ft_transformer", "tabr", "tabicl_v2")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RULES_CONTRACT = Path(__file__).with_name("experiment_contract.json")
_DEFAULT_CONFIG = Path(__file__).with_name("configs") / "campaign_v1.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    run = actions.add_parser("run", help="create or resume official-data candidates")
    run.add_argument("--config", required=True)
    run.add_argument("--data-dir", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--family", choices=_FAMILIES)
    run.add_argument("--max-candidates", type=int)
    status = actions.add_parser("status", help="print the current campaign manifest")
    status.add_argument("--output-dir", required=True)
    summarize = actions.add_parser("summarize", help="write or refresh campaign summary")
    summarize.add_argument("--output-dir", required=True)
    summarize.add_argument("--aligned-ml-oof")
    handoff = actions.add_parser("handoff", help="export one completed candidate")
    handoff.add_argument("--output-dir", required=True)
    handoff.add_argument("--candidate-id", required=True)
    handoff.add_argument("--result", required=True)
    handoff.add_argument("--runtime-sha256", required=True)
    handoff.add_argument("--requirements", required=True)
    handoff.add_argument("--environment", required=True)
    return parser


def _read_json(path: Path) -> object:
    if not path.is_file():
        raise RuntimeError(f"필수 캠페인 파일이 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _with_family_status(manifest: object) -> dict[str, object]:
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("candidates"), dict
    ):
        raise RuntimeError("campaign manifest candidates are invalid")
    response = dict(manifest)
    candidates = manifest["candidates"]
    family_status: dict[str, object] = {}
    for family in _FAMILIES:
        matching = [
            (candidate_id, entry)
            for candidate_id, entry in candidates.items()
            if isinstance(entry, dict)
            and isinstance(entry.get("candidate"), dict)
            and entry["candidate"].get("family") == family
        ]
        completed = [
            candidate_id
            for candidate_id, entry in matching
            if entry.get("state") == "completed"
        ]
        failed = [
            candidate_id
            for candidate_id, entry in matching
            if entry.get("state") == "failed"
        ]
        pending = [
            candidate_id
            for candidate_id, entry in matching
            if entry.get("state") in {"pending", "running"}
        ]
        family_status[family] = {
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "next_candidate": pending[0] if pending else None,
            "remaining_count": len(pending),
        }
    response["family_status"] = family_status
    return response


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    if args.action == "run":
        gate = assert_experiment_runnable(
            project_root=_PROJECT_ROOT,
            contract_path=_RULES_CONTRACT,
            config_path=args.config,
        )
        campaign = load_campaign(args.config)
        covered = set(gate["covered_candidate_ids"])
        campaign = replace(
            campaign,
            candidates=tuple(
                candidate
                for candidate in campaign.candidates
                if candidate.candidate_id in covered
            ),
        )
        runtime = OfficialCampaignRuntime(
            args.data_dir, cache_root=output_dir / "feature_cache"
        )
        summary = run_campaign(
            campaign,
            output_dir,
            runtime,
            family=args.family,
            max_candidates=args.max_candidates,
        )
        print(
            json.dumps(
                {
                    "campaign_id": summary.campaign_id,
                    "completed": len(summary.completed),
                    "failed": len(summary.failed),
                    "pending": len(summary.pending),
                    "output_root": str(summary.output_root),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.action == "status":
        manifest = _read_json(output_dir / "campaign_manifest.json")
        print(
            json.dumps(
                _with_family_status(manifest),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.action == "handoff":
        assert_experiment_runnable(
            project_root=_PROJECT_ROOT,
            contract_path=_RULES_CONTRACT,
            config_path=_DEFAULT_CONFIG,
            candidate_ids=[args.candidate_id],
        )
        from .handoff import write_candidate_handoff

        result = write_candidate_handoff(
            output_dir,
            args.candidate_id,
            args.result,
            args.runtime_sha256,
            args.requirements,
            args.environment,
        )
        print(
            json.dumps(
                {
                    "status": "handoff_ready",
                    "candidate_id": args.candidate_id,
                    "path": str(result.path),
                    "size_bytes": result.size_bytes,
                    "sha256": result.sha256,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    summary = _read_json(output_dir / "campaign_summary.json")
    response = {
        "status": "summarized",
        "campaign_summary": summary,
        "aligned_ml_oof": (
            None
            if args.aligned_ml_oof is None
            else str(Path(args.aligned_ml_oof).resolve())
        ),
        "blend_diagnostics": (
            "unavailable_without_aligned_ml_oof"
            if args.aligned_ml_oof is None
            else "pending_evaluation"
        ),
    }
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
