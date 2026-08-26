from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tree_expert.contracts import load_e1_contract
from experiments.tree_expert.inputs import file_sha256, prepare_e1_input


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-c-delivery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = prepare_e1_input(
        args.stage_c_delivery,
        args.output,
        load_e1_contract(),
    )
    print(
        f"TREE_E1_INPUT_READY path={output.resolve()} "
        f"sha256={file_sha256(output)} size_bytes={output.stat().st_size}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
