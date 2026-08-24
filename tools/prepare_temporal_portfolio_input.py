from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.temporal_portfolio.inputs import prepare_input_archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a verified temporal portfolio input-only archive."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    prepared = prepare_input_archive(args.data_dir, args.output, replace=args.replace)
    print(
        "TEMPORAL_INPUT_READY "
        f"path={prepared.path.absolute()} sha256={prepared.sha256} "
        f"size_bytes={prepared.size_bytes}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
