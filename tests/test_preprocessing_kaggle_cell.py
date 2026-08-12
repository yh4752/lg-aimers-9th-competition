from __future__ import annotations

import base64
from hashlib import sha256
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "experiments/preprocessing_campaign/KAGGLE_CELL.py"


def test_kaggle_cell_is_complete_self_contained_and_copyable() -> None:
    text = CELL.read_text(encoding="utf-8")

    compile(text, str(CELL), "exec")
    for required in (
        "/kaggle/input",
        "/kaggle/working",
        "EMBEDDED_RUNTIME_B64",
        "EMBEDDED_RUNTIME_SHA256",
        "MAX_JOBS_PER_SESSION",
        "MAX_SESSION_SECONDS",
        "--max-jobs",
        "--max-session-seconds",
        "preprocessing_campaign_v1",
        "preprocessing_campaign_review_bundle.zip",
        "PREPROCESSING_KAGGLE_SUCCESS",
        "PREPROCESSING_KAGGLE_ERROR",
        "torch.cuda.is_available()",
    ):
        assert required in text
    for forbidden in (
        "google.colab",
        "/content/drive",
        "GITHUB_TOKEN",
        "git clone",
        "submission",
    ):
        assert forbidden not in text


def test_embedded_runtime_matches_declared_hash() -> None:
    text = CELL.read_text(encoding="utf-8")
    encoded = re.search(r'EMBEDDED_RUNTIME_B64 = "([A-Za-z0-9+/=]+)"', text)
    expected = re.search(r'EMBEDDED_RUNTIME_SHA256 = "([0-9a-f]{64})"', text)

    assert encoded is not None and expected is not None
    assert sha256(base64.b64decode(encoded.group(1))).hexdigest() == expected.group(1)
