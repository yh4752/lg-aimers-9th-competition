from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tree_expert.s4_kaggle import runtime_identity_sha256
from experiments.tree_expert.s4_recovery import compact_recovery_handoff


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify and compact the interrupted S4 handoff for Kaggle recovery."
    )
    parser.add_argument("--source-handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = compact_recovery_handoff(
        args.source_handoff,
        args.output,
        destination_code_sha256=runtime_identity_sha256(ROOT),
    )
    print(
        f"TREE_S4_RECOVERY_SOURCE_VERIFIED sha256={result.source_sha256}",
        flush=True,
    )
    print(
        f"TREE_S4_RECOVERY_INPUT_READY path={result.path.resolve()} "
        f"sha256={result.sha256} retained_bytes={result.retained_bytes} "
        f"dropped_bytes={result.dropped_bytes}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
