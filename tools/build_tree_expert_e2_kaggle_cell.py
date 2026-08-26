from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.tree_expert.e2_kaggle import build_e2_kaggle_cell


def main() -> int:
    output = REPOSITORY_ROOT / "experiments/tree_expert/KAGGLE_E2_CELL.py"
    build_e2_kaggle_cell(output, REPOSITORY_ROOT)
    payload = output.read_bytes()
    print(
        f"TREE_E2_CELL_READY sha256={sha256(payload).hexdigest()} "
        f"size_bytes={len(payload)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
