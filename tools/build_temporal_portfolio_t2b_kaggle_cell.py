from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.temporal_portfolio.t2b_platform import build_t2b_kaggle_cell


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiments"
        / "temporal_portfolio"
        / "T2B_KAGGLE_CELL.py",
    )
    args = parser.parse_args()
    output = build_t2b_kaggle_cell(args.output)
    print(f"T2B_KAGGLE_CELL_READY path={output} size_bytes={output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
