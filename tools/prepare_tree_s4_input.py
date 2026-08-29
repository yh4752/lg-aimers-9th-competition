from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tree_expert.s4_inputs import file_sha256, prepare_s4_input


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2-handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = prepare_s4_input(args.e2_handoff, args.output)
    print(f"TREE_S4_INPUT_READY path={output.resolve()} sha256={file_sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
