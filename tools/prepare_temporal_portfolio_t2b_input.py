from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.temporal_portfolio.t2b_input import (
    prepare_t2b_input,
    verify_t2b_input,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1-review", type=Path, required=True)
    parser.add_argument("--t2a-handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = prepare_t2b_input(args.t1_review, args.t2a_handoff, args.output)
    verified = verify_t2b_input(output)
    print(
        f"T2B_INPUT_READY path={output} sha256={verified.archive_sha256} "
        f"promoted={','.join(verified.promoted)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

