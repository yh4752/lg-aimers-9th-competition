from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tree_expert.t3_inputs import file_sha256, prepare_t3_input


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--e2-handoff", required=True)
    result.add_argument("--output", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    result = prepare_t3_input(
        e2_handoff=Path(args.e2_handoff), output=Path(args.output),
    )
    print(f"TREE_T3_INPUT_READY path={result} sha256={file_sha256(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
