from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.catboost_50_50_realign.contracts import load_contract
from experiments.catboost_50_50_realign.inputs import (
    RealignSourcePaths,
    prepare_input_archive,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare one verified CatBoost 50:50 realignment handoff."
    )
    parser.add_argument("--training-input", type=Path, required=True)
    parser.add_argument("--stage-c-delivery", type=Path, required=True)
    parser.add_argument("--deployment-resume", type=Path, required=True)
    parser.add_argument("--deployment-review", type=Path, required=True)
    parser.add_argument("--oof-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    prepared = prepare_input_archive(
        RealignSourcePaths(
            training_input=args.training_input,
            stage_c_delivery=args.stage_c_delivery,
            deployment_resume=args.deployment_resume,
            deployment_review=args.deployment_review,
            oof_audit=args.oof_audit,
        ),
        args.output,
        load_contract(),
    )
    print(
        "REALIGN_INPUT_READY "
        f"path={prepared.path.resolve()} sha256={prepared.sha256} "
        f"size_bytes={prepared.path.stat().st_size}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
