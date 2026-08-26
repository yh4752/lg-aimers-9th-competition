from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.tree_expert.e2_inputs import file_sha256, prepare_e2_input


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify prior evidence and create the compact Tree Expert E2 input."
    )
    parser.add_argument("--e1-handoff", type=Path, required=True)
    parser.add_argument("--stage-c-delivery", type=Path, required=True)
    parser.add_argument("--tabm-submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = prepare_e2_input(
        args.e1_handoff,
        args.stage_c_delivery,
        args.tabm_submission,
        args.output,
    ).resolve()
    print(
        f"TREE_E2_INPUT_READY path={output} sha256={file_sha256(output)} "
        f"size_bytes={output.stat().st_size}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
