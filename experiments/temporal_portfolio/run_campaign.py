"""CLI entry point for one temporal portfolio physical stage."""
from __future__ import annotations

import argparse
from pathlib import Path

from .inputs import verify_official_data
from .planner import PHYSICAL_STAGES
from .runner import run_stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one restartable temporal portfolio stage")
    parser.add_argument("--stage", required=True, choices=PHYSICAL_STAGES)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--deadline-unix", required=True, type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verified = verify_official_data(args.data_root)
    if args.stage != "T1":
        raise RuntimeError("non-T1 CLI resume loading is provided by the platform handoff layer")
    result = run_stage(
        stage=args.stage,
        verified=verified,
        output_root=args.output_root,
        deadline=args.deadline_unix,
    )
    print(f"TEMPORAL_STAGE_RESULT stage={args.stage} status={result.status} handoff={result.handoff.path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
