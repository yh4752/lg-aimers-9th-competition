#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.temporal_portfolio.t2c_input import (  # noqa: E402
    prepare_t2c_input,
    verify_t2c_input,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the sealed temporal T2-C confirmation input."
    )
    parser.add_argument("--t2b-input", required=True, type=Path)
    parser.add_argument("--t2b-handoff", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = prepare_t2c_input(args.t2b_input, args.t2b_handoff, args.output)
    verified = verify_t2c_input(output)
    print(
        f"T2C_INPUT_READY path={output} sha256={verified.archive_sha256} "
        f"candidate={verified.candidate_id}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
