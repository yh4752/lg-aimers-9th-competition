from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/INDEPENDENT_DL_CAMPAIGN.ipynb"

FAMILIES = {
    "TabM 후보 1개": "tabm",
    "MLP/ResNet 후보 1개": "mlp_resnet",
    "FT-Transformer 후보 1개": "ft_transformer",
    "TabR 후보 1개": "tabr",
    "TabICLv2 후보 1개": "tabicl_v2",
}


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def test_independent_dl_notebook_has_split_execution_flow() -> None:
    notebook = _notebook()
    cells = notebook["cells"]
    markdown = [_source(cell) for cell in cells if cell["cell_type"] == "markdown"]

    expected = [
        "공통 준비",
        "현재 상태 확인",
        *FAMILIES,
        "결과 요약",
    ]
    positions = [
        next(index for index, text in enumerate(markdown) if heading in text)
        for heading in expected
    ]
    assert positions == sorted(positions)
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["colab"]["name"] == "INDEPENDENT_DL_CAMPAIGN.ipynb"
    assert notebook["metadata"]["accelerator"] == "GPU"
    for cell in cells:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_setup_and_read_only_cells_do_not_start_training() -> None:
    cells = _notebook()["cells"]
    setup_index = next(
        index
        for index, cell in enumerate(cells)
        if cell["cell_type"] == "markdown" and "공통 준비" in _source(cell)
    )
    setup = _source(cells[setup_index + 1])
    for required in (
        'drive.mount("/content/drive")',
        'userdata.get("GITHUB_TOKEN")',
        "pip",
        "torch.cuda.device_count()",
        "CAMPAIGN_OUTPUT_DIR",
        "def run_one_family",
    ):
        assert required in setup
    assert setup.count("run_one_family(") == 1

    for heading in ("현재 상태 확인", "결과 요약"):
        index = next(
            index
            for index, cell in enumerate(cells)
            if cell["cell_type"] == "markdown" and heading in _source(cell)
        )
        code = _source(cells[index + 1])
        assert '"run",' not in code


def test_each_family_cell_runs_exactly_one_candidate() -> None:
    cells = _notebook()["cells"]
    for heading, family in FAMILIES.items():
        index = next(
            index
            for index, cell in enumerate(cells)
            if cell["cell_type"] == "markdown" and heading in _source(cell)
        )
        explanation = _source(cells[index])
        code = _source(cells[index + 1])
        assert f'run_one_family("{family}")' in code
        for phrase in (
            "후보 하나",
            "예상 시간",
            "checkpoint",
            "재실행",
            "정상 완료",
            "traceback",
            "제출",
        ):
            assert phrase in explanation
    tabicl = next(text for text in map(_source, cells) if "TabICLv2 후보 1개" in text)
    assert "research_only" in tabicl


def test_notebook_keeps_broad_and_deep_campaign_visible() -> None:
    text = "\n".join(_source(cell) for cell in _notebook()["cells"])
    for phrase in ("P3", "P4", "boundary expansion", "confirmation"):
        assert phrase in text
    assert '"--max-candidates"' in text
    assert '"1"' in text
    assert "submission.zip" not in text
    assert "submit.zip" not in text
