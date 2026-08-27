from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.tree_expert.rf_inputs import file_sha256, prepare_rf_input


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--e2-handoff", required=True)
    result.add_argument("--output", required=True)
    return result


def main() -> int:
    stage = "arguments"
    try:
        args = parser().parse_args()
        stage = "prepare"
        output = prepare_rf_input(
            e2_handoff=Path(args.e2_handoff),
            output=Path(args.output),
        )
        print(f"TREE_RF_INPUT_SUCCESS path={output} sha256={file_sha256(output)}")
        return 0
    except Exception as error:
        print(
            f"TREE_RF_INPUT_ERROR stage={stage} type={type(error).__name__} "
            f"message={str(error).replace(' ', '_')}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
