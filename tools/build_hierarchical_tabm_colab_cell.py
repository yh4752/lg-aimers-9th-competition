from __future__ import annotations

import argparse
from hashlib import sha256
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hierarchical_tabm.runtime_inventory import render_cell


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    destination = ROOT / "experiments/hierarchical_tabm/COLAB_HIERARCHICAL_TABM_CELL.py"
    payload = render_cell(ROOT)
    if args.check:
        observed = destination.read_bytes() if destination.is_file() else b""
        if observed != payload:
            print(
                f"HIER_CELL_MISMATCH expected={sha256(payload).hexdigest()} "
                f"observed={sha256(observed).hexdigest()}"
            )
            return 1
        return 0
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    print(
        f"HIER_CELL_READY path={destination} sha256={sha256(payload).hexdigest()} "
        f"size_bytes={len(payload)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
