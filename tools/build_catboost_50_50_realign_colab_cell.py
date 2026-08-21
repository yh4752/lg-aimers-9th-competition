from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.catboost_50_50_realign.runtime_inventory import render_colab_cell


def main() -> int:
    destination = (
        ROOT
        / "experiments/catboost_50_50_realign/COLAB_CATBOOST_50_50_REALIGN_CELL.py"
    )
    payload = render_colab_cell(ROOT)
    destination.write_bytes(payload)
    print(
        f"REALIGN_CELL_READY path={destination} "
        f"sha256={sha256(payload).hexdigest()} size_bytes={len(payload)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
