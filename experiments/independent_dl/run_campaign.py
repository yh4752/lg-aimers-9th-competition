"""Command-line entry point for the user-owned independent DL campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign import OfficialCampaignRuntime, run_campaign
from .contracts import load_campaign


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    run = actions.add_parser("run", help="create or resume official-data candidates")
    run.add_argument("--config", required=True)
    run.add_argument("--data-dir", required=True)
    run.add_argument("--output-dir", required=True)
    status = actions.add_parser("status", help="print the current campaign manifest")
    status.add_argument("--output-dir", required=True)
    summarize = actions.add_parser("summarize", help="write or refresh campaign summary")
    summarize.add_argument("--output-dir", required=True)
    summarize.add_argument("--aligned-ml-oof")
    return parser


def _read_json(path: Path) -> object:
    if not path.is_file():
        raise RuntimeError(f"필수 캠페인 파일이 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    if args.action == "run":
        campaign = load_campaign(args.config)
        runtime = OfficialCampaignRuntime(
            args.data_dir, cache_root=output_dir / "feature_cache"
        )
        summary = run_campaign(campaign, output_dir, runtime)
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
        print(
            json.dumps(
                _read_json(output_dir / "campaign_manifest.json"),
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
