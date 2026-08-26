from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tree_expert.kaggle import build_e1_kaggle_cell


def main() -> int:
    output = build_e1_kaggle_cell(
        ROOT / "experiments" / "tree_expert" / "KAGGLE_E1_CELL.py",
        ROOT,
    )
    print(f"TREE_E1_CELL_READY path={output} size_bytes={output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
