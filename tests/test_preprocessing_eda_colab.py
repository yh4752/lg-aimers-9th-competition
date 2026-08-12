from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "PREPROCESSING_EDA_COLAB.md"
NOTEBOOK = ROOT / "notebooks" / "PREPROCESSING_EDA_COLAB.ipynb"

EXPECTED_CELL_IDS = [f"{index:02d}" for index in range(1, 9)]
EXPECTED_OUTPUTS = {
    "eda_summary.json",
    "run_manifest.json",
    "integrity_checks.csv",
    "feature_profile.csv",
    "season_drift.csv",
    "temporal_univariate_probes.csv",
    "missingness_probes.csv",
    "id_coverage.csv",
    "adversarial_validation.csv",
    "adversarial_importance.csv",
    "correlation_clusters.csv",
    "asof_reliability.csv",
    "asof_smoothing_grid.csv",
    "numeric_transform_diagnostics.csv",
    "interaction_probes.csv",
}


def read_document() -> str:
    return DOC.read_text(encoding="utf-8")


def python_cells() -> list[str]:
    return re.findall(
        r"```python\n(.*?)\n```",
        read_document(),
        flags=re.DOTALL,
    )


def load_pure_helpers(cells: list[str]) -> dict[str, object]:
    wanted = {"make_expanding_folds", "stable_sample_ids", "json_safe"}
    nodes: list[ast.stmt] = []
    for cell in cells:
        tree = ast.parse(cell)
        nodes.extend(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in wanted
        )
    found = {node.name for node in nodes if isinstance(node, ast.FunctionDef)}
    if found != wanted:
        raise AssertionError(f"missing pure helpers: {sorted(wanted - found)}")
    module = ast.Module(body=nodes, type_ignores=[])
    namespace: dict[str, object] = {
        "hashlib": hashlib,
        "math": math,
    }
    exec(compile(module, str(DOC), "exec"), namespace)
    return namespace


class PreprocessingEdaColabContractTest(unittest.TestCase):
    def test_notebook_matches_markdown_code_cells(self) -> None:
        self.assertTrue(NOTEBOOK.is_file())
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["metadata"]["kernelspec"]["name"], "python3")
        self.assertIn("colab", notebook["metadata"])
        code_cells = [
            "".join(cell["source"]).rstrip("\n")
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        self.assertEqual(code_cells, python_cells())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["execution_count"], None)
                self.assertEqual(cell["outputs"], [])

    def test_document_has_eight_ordered_python_cells(self) -> None:
        self.assertTrue(DOC.is_file())
        cells = python_cells()
        self.assertEqual(len(cells), 8)
        cell_ids: list[str] = []
        for cell in cells:
            match = re.search(r"^# CELL_ID: (\S+)$", cell, re.MULTILINE)
            self.assertIsNotNone(match)
            cell_ids.append(match.group(1))
        self.assertEqual(cell_ids, EXPECTED_CELL_IDS)

    def test_every_python_cell_compiles(self) -> None:
        for index, cell in enumerate(python_cells(), start=1):
            compile(cell, f"{DOC.name}:cell-{index:02d}", "exec")

    def test_document_keeps_user_run_and_test_schema_boundaries(self) -> None:
        text = read_document()
        self.assertIn('TEST_USAGE = "schema_only"', text)
        self.assertIn('fit_audit["test_rows_used"] = 0', text)
        self.assertIn("사용자가 실행", text)
        self.assertIn("EDA_SUCCESS", text)
        self.assertIn("EDA_ERROR", text)
        self.assertNotIn("pip install", text)
        self.assertNotIn("/content/drive/MyDrive", text)
        self.assertNotIn("sample_submission", text)

    def test_document_declares_every_structured_output(self) -> None:
        text = read_document()
        for filename in EXPECTED_OUTPUTS:
            self.assertIn(filename, text)

    def test_expanding_folds_never_train_on_future_seasons(self) -> None:
        helpers = load_pure_helpers(python_cells())
        make_folds = helpers["make_expanding_folds"]
        folds = make_folds([2024, 2021, 2023, 2022, 2021])
        self.assertEqual(
            folds,
            [
                {
                    "name": "through_2021_to_2022",
                    "train_seasons": (2021,),
                    "valid_season": 2022,
                },
                {
                    "name": "through_2022_to_2023",
                    "train_seasons": (2021, 2022),
                    "valid_season": 2023,
                },
                {
                    "name": "through_2023_to_2024",
                    "train_seasons": (2021, 2022, 2023),
                    "valid_season": 2024,
                },
            ],
        )
        for fold in folds:
            self.assertLess(max(fold["train_seasons"]), fold["valid_season"])

    def test_stable_sample_is_order_independent(self) -> None:
        helpers = load_pure_helpers(python_cells())
        sample = helpers["stable_sample_ids"]
        first = sample(["r9", "r1", "r7", "r2", "r4"], 3, 20260812)
        second = sample(["r4", "r2", "r7", "r1", "r9"], 3, 20260812)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(set(first)), 3)

    def test_json_safe_replaces_only_nonfinite_numbers(self) -> None:
        helpers = load_pure_helpers(python_cells())
        json_safe = helpers["json_safe"]
        payload = {
            "ok": 0.25,
            "bad": [float("nan"), float("inf"), -float("inf")],
            "nested": {"value": 3},
        }
        self.assertEqual(
            json_safe(payload),
            {
                "ok": 0.25,
                "bad": [None, None, None],
                "nested": {"value": 3},
            },
        )


if __name__ == "__main__":
    unittest.main()
