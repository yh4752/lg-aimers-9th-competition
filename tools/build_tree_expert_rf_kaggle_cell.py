from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tree_expert.rf_kaggle import build_rf_kaggle_cell


def main() -> int:
    output = ROOT / "experiments/tree_expert/KAGGLE_RF_CELL.py"
    result = build_rf_kaggle_cell(output, ROOT)
    print(f"TREE_RF_CELL_READY path={result} size_bytes={result.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
