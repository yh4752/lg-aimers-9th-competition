from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tabm_campaign.row_feature_colab import file_sha256, prepare_input_archive
from experiments.tabm_campaign.row_feature_contracts import (
    load_row_feature_proxy_contract,
    row_feature_contract_sha256,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the exact direct-upload data ZIP for Stage P Colab."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)

    contract = load_row_feature_proxy_contract()
    output = prepare_input_archive(
        args.data_dir,
        args.output,
        expected_train_sha256=contract.official_train_sha256,
        campaign_config_sha256=row_feature_contract_sha256(),
        replace=args.replace,
    )
    print(
        f"ROW_FEATURE_INPUT_READY path={output.resolve()} "
        f"sha256={file_sha256(output)} size_bytes={output.stat().st_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
