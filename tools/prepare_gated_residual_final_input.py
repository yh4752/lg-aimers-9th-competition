from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gated_residual_final.inputs import file_sha256, prepare_final_input  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the bound final residual Kaggle input.")
    parser.add_argument("--e2-input", type=Path, required=True)
    parser.add_argument("--stage-a", type=Path, required=True)
    parser.add_argument("--stage-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-e2-sha256")
    parser.add_argument("--expected-stage-a-sha256")
    parser.add_argument("--expected-stage-b-sha256")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = (args.expected_e2_sha256, args.expected_stage_a_sha256, args.expected_stage_b_sha256)
    if any(overrides) and not all(overrides):
        raise ValueError("all expected SHA-256 overrides are required together")
    expected = None
    if all(overrides):
        expected = {"e2_input": overrides[0], "stage_a": overrides[1], "stage_b": overrides[2]}
    output = prepare_final_input(
        e2_input=args.e2_input,
        stage_a=args.stage_a,
        stage_b=args.stage_b,
        output=args.output,
        expected_hashes=expected,
    )
    print(
        f"GATED_RESIDUAL_INPUT_READY path={output.resolve()} sha256={file_sha256(output)} size_bytes={output.stat().st_size}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
