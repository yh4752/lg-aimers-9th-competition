from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from experiments.tree_privileged.kaggle import build_kaggle_cell


def main() -> int:
    output = ROOT / "experiments/tree_privileged/KAGGLE_CELL.py"
    build_kaggle_cell(output, root=ROOT)
    print(f"TREE_PRIV_KAGGLE_CELL_READY path={output} sha256={sha256(output.read_bytes()).hexdigest()} size_bytes={output.stat().st_size}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
