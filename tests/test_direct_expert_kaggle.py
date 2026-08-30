from pathlib import Path
from zipfile import ZipFile

import pytest

from experiments.direct_expert.kaggle import (
    DirectExpertKaggleError,
    discover_inputs,
    verify_t4x2,
)


def _root(tmp_path: Path, *, duplicates: int = 1) -> Path:
    official = tmp_path / "official"
    official.mkdir()
    (official / "train.csv").write_text("row_id\na\n", encoding="utf-8")
    (official / "trackman_history.csv").write_text("season\n2024\n", encoding="utf-8")
    for index in range(duplicates):
        campaign = tmp_path / f"campaign-{index}"
        campaign.mkdir()
        (campaign / "manifest.json").write_text(
            '{"artifact_kind":"direct_expert_input_v1"}', encoding="utf-8"
        )
    return tmp_path


def test_discovery_accepts_expanded_input_and_rejects_two_logical_inputs(tmp_path: Path) -> None:
    found = discover_inputs(_root(tmp_path), stage="A")
    assert found.official_data.is_dir()
    assert found.campaign_input.exists()
    other = tmp_path / "other"
    other.mkdir()
    (other / "manifest.json").write_text(
        '{"artifact_kind":"direct_expert_input_v1","identity":"other"}', encoding="utf-8"
    )
    with pytest.raises(DirectExpertKaggleError, match="campaign input count must be one"):
        discover_inputs(tmp_path, stage="A")


def test_discovery_deduplicates_zip_and_expanded_copy(tmp_path: Path) -> None:
    root = _root(tmp_path)
    manifest = (root / "campaign-0/manifest.json").read_bytes()
    with ZipFile(root / "same.zip", "w") as archive:
        archive.writestr("manifest.json", manifest)
    found = discover_inputs(root, stage="A")
    assert found.campaign_input.exists()


class FakeCuda:
    def __init__(self, names: list[str]): self.names = names
    def device_count(self) -> int: return len(self.names)
    def get_device_name(self, index: int) -> str: return self.names[index]


class FakeTorch:
    def __init__(self, names: list[str]): self.cuda = FakeCuda(names)


def test_gpu_contract_requires_exactly_two_t4_devices() -> None:
    assert verify_t4x2(FakeTorch(["Tesla T4", "Tesla T4"])) == ("Tesla T4", "Tesla T4")
    with pytest.raises(DirectExpertKaggleError, match="two Tesla T4"):
        verify_t4x2(FakeTorch(["Tesla T4"]))
