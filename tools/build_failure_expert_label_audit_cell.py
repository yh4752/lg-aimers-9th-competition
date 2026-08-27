from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tree_expert.failure_audit_kaggle import build_failure_audit_cell


def main() -> int:
    output = ROOT / "experiments/tree_expert/KAGGLE_FAILURE_AUDIT_CELL.py"
    result = build_failure_audit_cell(output, ROOT)
    print(f"FAIL_AUDIT_CELL_READY path={result} size_bytes={result.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
