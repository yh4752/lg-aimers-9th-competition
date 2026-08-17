from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.catboost_tabm_blend.contracts import load_contract
from experiments.catboost_tabm_blend.inputs import file_sha256, prepare_input_archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the direct-upload training ZIP for the CatBoost TabM blend."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    output = prepare_input_archive(
        args.data_dir,
        args.output,
        load_contract(),
        replace=args.replace,
    )
    print(
        f"CATBOOST_BLEND_INPUT_READY path={output.resolve()} "
        f"sha256={file_sha256(output)} size_bytes={output.stat().st_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
