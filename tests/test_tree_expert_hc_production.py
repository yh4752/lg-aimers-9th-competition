from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pandas as pd

from experiments.tree_expert.hc_contracts import load_hc_contract
from experiments.tree_expert.hc_production import HCProductionDependencies, HCProductionRuntime
from experiments.tree_expert.hc_state import HCBindings, initial_state
from experiments.tree_expert.hc_training import C1JobResult, OOFFeatureMaterialization
from experiments.tree_expert.inputs import PREDICTION_COLUMNS, VerifiedOfficialData


def _prediction(year: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"r{year}_0", f"r{year}_1"],
            "target": [0, 1],
            "probability": [0.45, 0.55],
            "game_type": ["R", "F"],
            "game_month": [4, 5],
            "pitcher_id_known": ["known", "known"],
            "batter_id_known": ["known", "known"],
        }
    ).loc[:, PREDICTION_COLUMNS]


def test_h1_selects_on_structure_folds_and_runs_one_confirmation(tmp_path: Path):
    train = tmp_path / "train.csv"
    history = tmp_path / "trackman_history.csv"
    pd.DataFrame({"season": [2020]}).to_csv(train, index=False)
    pd.DataFrame({"x": [1]}).to_csv(history, index=False)
    data = VerifiedOfficialData(tmp_path, train, history, "1" * 64, "2" * 64)
    folds = {}
    for year in (2022, 2023, 2024):
        path = tmp_path / f"e2_{year}.csv"
        _prediction(year).to_csv(path, index=False)
        folds[(year - 1, year)] = path
    evidence = SimpleNamespace(
        manifest_sha256="3" * 64,
        fold_predictions=MappingProxyType(folds),
        e2_delivery=tmp_path / "delivery.zip",
    )
    evidence.e2_delivery.write_bytes(b"delivery")

    def source_baseline(**kwargs):
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        path = output / "predictions.csv"
        _prediction(2021).to_csv(path, index=False)
        (output / "checkpoint.pt").write_bytes(b"checkpoint")
        (output / "baseline_result.json").write_text("{}")
        return SimpleNamespace(
            status="completed", ready_for_residual=True, predictions_path=path
        )

    def source_residual(**kwargs):
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        path = output / "predictions.csv"
        kwargs["baseline"].to_csv(path, index=False)
        (output / "model.cbm").write_bytes(b"model")
        (output / "worker_result.json").write_text("{}")
        return SimpleNamespace(status="completed", predictions_path=path)

    def materialize(official_rows, **kwargs):
        del official_rows, kwargs
        rows = []
        for year in (2021, 2022, 2023, 2024):
            for index in range(2):
                rows.append(
                    {
                        "row_id": f"r{year}_{index}",
                        "oof_year": year,
                        "target": index,
                        "p0": 0.45 if index == 0 else 0.55,
                        "game_type": "R" if index == 0 else "F",
                        "pitcher_id": f"p{index}",
                        "batter_id": f"b{index}",
                        "feature": float(index),
                    }
                )
        return OOFFeatureMaterialization(pd.DataFrame(rows), ("feature", "p0"), ())

    def c1_job(**kwargs):
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        valid = kwargs["valid_rows"].copy()
        profile = kwargs["profile_name"]
        improvement = {"hc_strong": 0.01, "hc_balanced": 0.04, "hc_light": 0.02}[profile]
        valid["p1"] = valid["p0"] + (valid["target"] * 2 - 1) * improvement
        predictions = output / "predictions.csv"
        valid.to_csv(predictions, index=False)
        model = output / "model.cbm"
        model.write_bytes(b"model")
        (output / "metrics.json").write_text('{"best_iteration":7}')
        return C1JobResult(
            kwargs["job_id"], "completed", kwargs["seed"], profile, 7, 0.2, model, predictions
        )

    runtime = HCProductionRuntime(
        data=data,
        evidence=evidence,
        output=tmp_path / "campaign",
        contract=load_hc_contract(),
        dependencies=HCProductionDependencies(
            source_baseline=source_baseline,
            source_residual=source_residual,
            materialize=materialize,
            c1_job=c1_job,
        ),
    )
    state = initial_state(HCBindings(*tuple(str(index) * 64 for index in range(1, 7))))
    outcome = runtime.run_stage("H1", state, 10**12, 2)
    assert outcome.state.stage == "H2"
    assert outcome.state.decisions["profile"]["selected"] == "hc_balanced"
    confirmation = [
        job for job in outcome.state.completed_jobs if "tr2023__va2024" in job and "c1_" in job
    ]
    assert confirmation == ["hc__c1_hc_balanced__tr2023__va2024__s3407"]
    skipped = dict(outcome.state.skipped_jobs)
    assert "hc__c1_hc_strong__tr2023__va2024__s3407" in skipped
    assert "hc__c1_hc_light__tr2023__va2024__s3407" in skipped
