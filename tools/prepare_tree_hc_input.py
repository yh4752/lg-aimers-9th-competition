from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.tree_expert.hc_inputs import prepare_hc_input


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the accepted E2 evidence for the hierarchical Kaggle campaign."
    )
    parser.add_argument("--e2-handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = prepare_hc_input(e2_handoff=args.e2_handoff, output=args.output)
    payload = output.read_bytes()
    print(
        f"TREE_HC_INPUT_READY path={output.resolve()} "
        f"sha256={sha256(payload).hexdigest()} size_bytes={len(payload)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
