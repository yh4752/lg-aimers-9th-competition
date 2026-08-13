from __future__ import annotations

import base64
import gzip
from hashlib import sha256
import io
from pathlib import Path
import re
import tarfile


ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "experiments/preprocessing_campaign/KAGGLE_BUDGETED_CELL.py"
MAX_KAGGLE_SOURCE_BYTES = 900_000


def _embedded_archive() -> bytes:
    text = CELL.read_text(encoding="utf-8")
    match = re.search(r'EMBEDDED_RUNTIME_B64 = "([A-Za-z0-9+/=]+)"', text)
    assert match is not None
    return base64.b64decode(match.group(1), validate=True)


def test_budgeted_cell_is_one_self_contained_kaggle_cell_without_network_source() -> None:
    text = CELL.read_text(encoding="utf-8")

    assert CELL.stat().st_size < MAX_KAGGLE_SOURCE_BYTES
    compile(text, str(CELL), "exec")
    for required in (
        "/kaggle/input",
        "/kaggle/working",
        "EMBEDDED_RUNTIME_B64",
        "EMBEDDED_RUNTIME_SHA256",
        "MAX_SESSION_SECONDS = 6300",
        "run_budgeted_campaign",
        "preprocessing_campaign_final_review_bundle.zip",
        "STAGE_ERROR",
        "T4_X2_READY",
    ):
        assert required in text
    for forbidden in (
        "github.com",
        "git clone",
        "google.colab",
        "/content/drive",
        "submission",
    ):
        assert forbidden not in text


def test_embedded_archive_matches_hash_and_contains_budgeted_cli() -> None:
    text = CELL.read_text(encoding="utf-8")
    payload = _embedded_archive()
    expected = re.search(r'EMBEDDED_RUNTIME_SHA256 = "([0-9a-f]{64})"', text)

    assert expected is not None
    assert sha256(payload).hexdigest() == expected.group(1)
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(payload)), mode="r:") as archive:
        names = set(archive.getnames())
    assert "experiments/preprocessing_campaign/run_budgeted_campaign.py" in names
    assert "experiments/preprocessing_campaign/requirements-kaggle-budgeted.txt" in names
