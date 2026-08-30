from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tree_privileged.inputs import file_sha256, prepare_input


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the sealed privileged-tree Kaggle input.")
    parser.add_argument("--e2-handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = prepare_input(args.e2_handoff, args.output).resolve()
    print(f"TREE_PRIV_INPUT_READY path={output} sha256={file_sha256(output)} size_bytes={output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

