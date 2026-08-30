from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.direct_expert.inputs import (  # noqa: E402
    file_sha256,
    prepare_direct_expert_input,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the bound direct-expert Kaggle input.")
    parser.add_argument("--s4-handoff", type=Path, required=True)
    parser.add_argument("--e2-submission", type=Path, required=True)
    parser.add_argument("--e2-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = prepare_direct_expert_input(
        args.s4_handoff,
        args.e2_submission,
        args.e2_receipt,
        args.output,
    )
    print(
        f"DIRECT_EXPERT_INPUT_READY path={output.resolve()} "
        f"sha256={file_sha256(output)} size_bytes={output.stat().st_size}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
