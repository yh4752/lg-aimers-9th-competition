# 전처리 연구용 EDA Colab 실행 코드

아래 Python 셀 8개를 위에서부터 순서대로 Colab에 복사해 실행한다. 사용자가 실행할
코드이며, Codex는 공식 전체 데이터를 실행하지 않았다.

- 입력: Drive 안의 `official_open.zip`
- 예상 시간: 일반 Colab CPU 기준 약 45~180분
- 예상 메모리: 약 8~16GB
- 산출물: ZIP 옆 `preprocessing_eda_runs/<RUN_ID>/`
- 재실행 안전성: 완성된 같은 `RUN_ID`는 덮어쓰지 않는다. 다시 실행하려면 셀 01의
  `RUN_ID`를 바꾼다.
- 성공 시 반환할 값: 마지막 `EDA_SUCCESS ...` 한 줄과 생성된 결과 디렉터리
- 실패 시 반환할 값: `EDA_ERROR ...` 한 줄과 전체 traceback

## 셀 01 — 환경, 경로와 고정 설정

```python
# CELL_ID: 01
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import uuid
import warnings
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 1))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

warnings.filterwarnings("ignore", category=FutureWarning)

STAGE = "setup"
SEED = 20260812
RUN_ID = "preprocessing_eda_v1"  # 같은 이름의 완성 결과가 있으면 새 이름으로 바꾼다.
ARCHIVE_PATH = ""  # 자동 검색이 실패할 때만 Drive 안 ZIP의 절대 경로를 입력한다.
TEST_USAGE = "schema_only"

SMOOTH_K = 50.0
NUMERIC_BIN_COUNT = 20
RARE_COUNT_MAX = 10
DRIFT_SAMPLE_PER_SEASON = 200_000
ADVERSARIAL_SAMPLE_PER_SEASON = 200_000
IMPORTANCE_SAMPLE_MAX = 50_000
IMPORTANCE_REPEATS = 5
CORRELATION_THRESHOLD = 0.95
TRANSFORM_SAMPLE_MAX = 200_000
SMOOTHING_K_GRID = (0, 5, 10, 25, 50, 100, 125, 250, 500, 1000, 2500, 5000, 10000)

CSV_OUTPUTS = (
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
)
JSON_OUTPUTS = ("eda_summary.json", "run_manifest.json")

try:
    from google.colab import drive
except ImportError as exc:
    print(f"EDA_ERROR stage={STAGE} type={type(exc).__name__} message=Google_Colab_required")
    raise

try:
    drive.mount("/content/drive")
    drive_root = Path("/content/drive")
    if ARCHIVE_PATH.strip():
        archive_path = Path(ARCHIVE_PATH).expanduser()
        candidates = [archive_path]
    else:
        candidates = sorted(drive_root.rglob("official_open.zip"))
    candidates = [path for path in candidates if path.is_file()]
    if len(candidates) != 1:
        raise RuntimeError(
            "official_open.zip candidate count must be 1; "
            f"found={len(candidates)} candidates={[str(path) for path in candidates[:10]]}. "
            "Set ARCHIVE_PATH in cell 01."
        )
    archive_path = candidates[0]
    output_root = archive_path.parent / "preprocessing_eda_runs"
    final_output_dir = output_root / RUN_ID
    if final_output_dir.exists():
        raise FileExistsError(
            f"completed RUN_ID already exists: {final_output_dir}; change RUN_ID"
        )
    runtime_dir = Path(tempfile.mkdtemp(prefix=f"{RUN_ID}_", dir="/content"))
    runtime_output_dir = runtime_dir / "result"
    runtime_output_dir.mkdir(parents=True, exist_ok=False)
    plots_dir = runtime_output_dir / "plots"
    plots_dir.mkdir()
    started_at = time.time()
    np.random.seed(SEED)
    print(f"SETUP_OK archive={archive_path} runtime={runtime_output_dir}")
except Exception as exc:
    print(f"EDA_ERROR stage={STAGE} type={type(exc).__name__} message={str(exc).replace(' ', '_')}")
    raise
```

## 셀 02 — 공통 헬퍼

```python
# CELL_ID: 02
STAGE = "helpers"

def make_expanding_folds(seasons):
    ordered = sorted({int(value) for value in seasons})
    if len(ordered) < 2:
        raise ValueError("at least two seasons are required")
    return [
        {
            "name": f"through_{valid - 1}_to_{valid}",
            "train_seasons": tuple(year for year in ordered if year < valid),
            "valid_season": valid,
        }
        for valid in ordered[1:]
    ]


def stable_sample_ids(row_ids, limit, seed):
    unique = {}
    for value in row_ids:
        text = str(value)
        unique.setdefault(text, value)
    scored = []
    for text, original in unique.items():
        score = hashlib.sha256(f"{seed}|{text}".encode("utf-8")).hexdigest()
        scored.append((score, text, original))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in scored[: max(0, min(int(limit), len(scored)))]]


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return json_safe(value.item())
    if value.__class__.__module__ == "pathlib":
        return str(value)
    return value


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(handle, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    while chunk := handle.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_values(values):
    digest = hashlib.sha256()
    for value in sorted(map(str, values)):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sample_frame(frame, limit, salt):
    if len(frame) <= limit:
        return frame.sort_values("row_id", kind="stable").copy()
    chosen = stable_sample_ids(frame["row_id"].tolist(), limit, SEED + int(salt))
    chosen_text = set(map(str, chosen))
    return frame.loc[frame["row_id"].astype(str).isin(chosen_text)].sort_values(
        "row_id", kind="stable"
    ).copy()


def brier_score(target, prediction):
    y = np.asarray(target, dtype="float64")
    p = np.clip(np.asarray(prediction, dtype="float64"), 1e-6, 1 - 1e-6)
    return float(np.mean(np.square(y - p)))


def finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def weighted_mean(rows, value_key, weight_key="valid_rows"):
    valid = [row for row in rows if row.get(value_key) is not None and row.get(weight_key, 0) > 0]
    if not valid:
        return None
    weights = np.asarray([row[weight_key] for row in valid], dtype="float64")
    values = np.asarray([row[value_key] for row in valid], dtype="float64")
    return float(np.average(values, weights=weights))


def categorical_text(series):
    return series.astype("string").fillna("__MISSING__").astype(str)


def numeric_values(series):
    return pd.to_numeric(series, errors="coerce").astype("float64").replace([np.inf, -np.inf], np.nan)


def numeric_keys(train_series, valid_series, bins=NUMERIC_BIN_COUNT):
    train_num = numeric_values(train_series)
    valid_num = numeric_values(valid_series)
    available = train_num.dropna()
    if available.nunique() < 2:
        return categorical_text(train_num), categorical_text(valid_num), None
    quantiles = np.linspace(0, 1, min(bins, available.nunique()) + 1)
    edges = np.unique(available.quantile(quantiles).to_numpy(dtype="float64"))
    if len(edges) < 2:
        return categorical_text(train_num), categorical_text(valid_num), None
    edges[0], edges[-1] = -np.inf, np.inf
    train_key = pd.cut(train_num, bins=edges, include_lowest=True, duplicates="drop")
    valid_key = pd.cut(valid_num, bins=edges, include_lowest=True, duplicates="drop")
    return categorical_text(train_key), categorical_text(valid_key), edges.tolist()


def fit_rate_predictions(train_key, train_target, valid_key, smoothing=SMOOTH_K):
    prior = float(np.mean(np.asarray(train_target, dtype="float64")))
    stats = pd.DataFrame(
        {"key": categorical_text(train_key), "target": np.asarray(train_target, dtype="float64")}
    ).groupby("key", sort=True)["target"].agg(["sum", "count"])
    rates = (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)
    valid_text = categorical_text(valid_key)
    prediction = valid_text.map(rates).fillna(prior).to_numpy(dtype="float64")
    coverage = float(valid_text.isin(stats.index).mean())
    return np.clip(prediction, 1e-6, 1 - 1e-6), prior, coverage


def run_temporal_probe(frame, feature_name, values, feature_type, folds):
    rows = []
    series = pd.Series(values, index=frame.index, name=feature_name)
    for fold in folds:
        train_mask = frame["season"].isin(fold["train_seasons"])
        valid_mask = frame["season"].eq(fold["valid_season"])
        y_train = frame.loc[train_mask, "control_success"].to_numpy(dtype="float64")
        y_valid = frame.loc[valid_mask, "control_success"].to_numpy(dtype="float64")
        if feature_type == "numeric":
            train_key, valid_key, edges = numeric_keys(series.loc[train_mask], series.loc[valid_mask])
        else:
            train_key = categorical_text(series.loc[train_mask])
            valid_key = categorical_text(series.loc[valid_mask])
            edges = None
        prediction, prior, coverage = fit_rate_predictions(train_key, y_train, valid_key)
        prior_prediction = np.full(len(y_valid), prior, dtype="float64")
        prior_brier = brier_score(y_valid, prior_prediction)
        probe_brier = brier_score(y_valid, prediction)
        rows.append(
            {
                "scope": "fold",
                "fold": fold["name"],
                "feature": feature_name,
                "feature_type": feature_type,
                "valid_rows": len(y_valid),
                "prior_brier": prior_brier,
                "probe_brier": probe_brier,
                "delta_brier": probe_brier - prior_brier,
                "coverage": coverage,
                "unseen_rate": 1.0 - coverage,
                "bin_edges": json.dumps(json_safe(edges), ensure_ascii=False),
                "improved_fold_count": None,
                "worsened_fold_count": None,
                "direction_consistent": None,
            }
        )
    deltas = [row["delta_brier"] for row in rows]
    rows.append(
        {
            "scope": "aggregate",
            "fold": "all_forward_folds",
            "feature": feature_name,
            "feature_type": feature_type,
            "valid_rows": sum(row["valid_rows"] for row in rows),
            "prior_brier": weighted_mean(rows, "prior_brier"),
            "probe_brier": weighted_mean(rows, "probe_brier"),
            "delta_brier": weighted_mean(rows, "delta_brier"),
            "coverage": weighted_mean(rows, "coverage"),
            "unseen_rate": weighted_mean(rows, "unseen_rate"),
            "bin_edges": "not_applicable",
            "improved_fold_count": int(sum(delta < 0 for delta in deltas)),
            "worsened_fold_count": int(sum(delta > 0 for delta in deltas)),
            "direction_consistent": bool(all(delta <= 0 for delta in deltas) or all(delta >= 0 for delta in deltas)),
        }
    )
    return rows


def ensure_nonempty(frame, columns, reason):
    if not frame.empty:
        return frame.loc[:, columns]
    row = {column: "not_applicable" for column in columns}
    row[columns[0]] = reason
    return pd.DataFrame([row], columns=columns)


try:
    assert make_expanding_folds([2021, 2022, 2023])[-1]["valid_season"] == 2023
    assert stable_sample_ids([3, 1, 2], 2, SEED) == stable_sample_ids([2, 3, 1], 2, SEED)
    assert json_safe({"x": float("nan")}) == {"x": None}
    print("HELPERS_OK")
except Exception as exc:
    print(f"EDA_ERROR stage={STAGE} type={type(exc).__name__} message={str(exc).replace(' ', '_')}")
    raise
```

## 셀 03 — 데이터 로딩과 무결성 검사

```python
# CELL_ID: 03
STAGE = "integrity"

INPUT_COLUMNS = (
    "row_id", "season", "game_month", "game_dayofweek", "inning", "top_bottom",
    "game_type", "balls_before", "strikes_before", "outs_before", "run_top_before",
    "run_bot_before", "run_total_before", "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state",
    "home_win_expectancy", "away_win_expectancy", "li", "pitcher_id", "batter_id",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
)
TARGET_COLUMN = "control_success"
CATEGORICAL_COLUMNS = {
    "game_month", "game_dayofweek", "top_bottom", "game_type", "balls_before",
    "strikes_before", "outs_before", "runner_on_1b", "runner_on_2b", "runner_on_3b",
    "num_runners_on", "base_state", "pitcher_id", "batter_id", "pitcher_hand",
    "batter_hand", "pitcher_team_id", "batter_team_id",
}
ID_COLUMNS = ("pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id")
RATE_COLUMNS = tuple(column for column in INPUT_COLUMNS if column.startswith("asof_") and column.endswith("_rate"))
COUNT_COLUMNS = ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n")

integrity_rows = []
fit_audit = {"test_rows_used": 0, "test_usage": TEST_USAGE}
sample_audit = {}

def add_check(name, status, observed, expected, detail=""):
    integrity_rows.append(
        {"check": name, "status": status, "observed": str(observed), "expected": str(expected), "detail": detail}
    )


def check_relation(name, valid_mask, expected, detail=""):
    failures = int((~pd.Series(valid_mask).fillna(False)).sum())
    add_check(name, "pass" if failures == 0 else "warn", failures, expected, detail)


try:
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        train_members = [name for name in members if Path(name).name == "train.csv"]
        test_members = [name for name in members if Path(name).name == "test.csv"]
        if len(train_members) != 1 or len(test_members) != 1:
            raise RuntimeError(
                f"ZIP must contain one train.csv and one test.csv; train={train_members}, test={test_members}"
            )
        train_member, test_member = train_members[0], test_members[0]
        with archive.open(train_member) as handle:
            train_member_sha256 = sha256_stream(handle)
        with archive.open(test_member) as handle:
            test_member_sha256 = sha256_stream(handle)
        with archive.open(train_member) as handle:
            train = pd.read_csv(handle, low_memory=False)
        with archive.open(test_member) as handle:
            test_schema = pd.read_csv(handle, low_memory=False)

    archive_sha256 = sha256_file(archive_path)
    add_check("train_columns", "pass" if tuple(train.columns) == (*INPUT_COLUMNS, TARGET_COLUMN) else "fail", tuple(train.columns), (*INPUT_COLUMNS, TARGET_COLUMN))
    add_check("test_columns", "pass" if tuple(test_schema.columns) == INPUT_COLUMNS else "fail", tuple(test_schema.columns), INPUT_COLUMNS)
    add_check("test_target_absent", "pass" if TARGET_COLUMN not in test_schema else "fail", TARGET_COLUMN in test_schema, False)
    add_check("row_id_non_null", "pass" if train["row_id"].notna().all() else "fail", int(train["row_id"].isna().sum()), 0)
    add_check("row_id_unique", "pass" if train["row_id"].is_unique else "fail", int(train["row_id"].duplicated().sum()), 0)
    target_numeric = numeric_values(train[TARGET_COLUMN])
    target_valid = target_numeric.notna().all() and target_numeric.isin([0, 1]).all()
    add_check("binary_target", "pass" if target_valid else "fail", sorted(target_numeric.dropna().unique().tolist()), [0, 1])

    range_rules = {
        "balls_before": (0, 3), "strikes_before": (0, 2), "outs_before": (0, 2),
        "game_month": (1, 12), "game_dayofweek": (0, 6),
        "runner_on_1b": (0, 1), "runner_on_2b": (0, 1), "runner_on_3b": (0, 1),
        "num_runners_on": (0, 3), "home_win_expectancy": (0, 100),
        "away_win_expectancy": (0, 100),
    }
    for column, (lower, upper) in range_rules.items():
        values = numeric_values(train[column])
        invalid = int((values.notna() & ~values.between(lower, upper)).sum())
        add_check(f"range_{column}", "pass" if invalid == 0 else "warn", invalid, 0, f"[{lower}, {upper}]")
    for column in COUNT_COLUMNS:
        values = numeric_values(train[column])
        invalid = int((values.notna() & values.lt(0)).sum())
        add_check(f"nonnegative_{column}", "pass" if invalid == 0 else "warn", invalid, 0)
    for column in RATE_COLUMNS:
        values = numeric_values(train[column])
        invalid = int((values.notna() & ~values.between(0, 1)).sum())
        add_check(f"rate_range_{column}", "pass" if invalid == 0 else "warn", invalid, 0, "[0, 1]")
    for column in [name for name in INPUT_COLUMNS if name not in CATEGORICAL_COLUMNS and name != "row_id"]:
        raw_numeric = pd.to_numeric(train[column], errors="coerce").to_numpy(dtype="float64")
        infinite = int(np.isinf(raw_numeric).sum())
        add_check(f"finite_{column}", "pass" if infinite == 0 else "warn", infinite, 0, "infinite values are treated as missing in diagnostics")
    for column, allowed in {
        "top_bottom": {"T", "B"},
        "base_state": {"___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"},
    }.items():
        observed = set(categorical_text(train[column]).unique()) - {"__MISSING__"}
        invalid = sorted(observed - allowed)
        add_check(f"allowed_values_{column}", "pass" if not invalid else "warn", invalid, sorted(allowed))

    runners = train[["runner_on_1b", "runner_on_2b", "runner_on_3b"]].apply(pd.to_numeric, errors="coerce")
    runner_sum = runners.sum(axis=1, min_count=3)
    check_relation("runner_count_consistency", numeric_values(train["num_runners_on"]).eq(runner_sum), 0)
    base_map = {"___": "000", "1__": "100", "_2_": "010", "__3": "001", "12_": "110", "1_3": "101", "_23": "011", "123": "111"}
    observed_base = runners.fillna(-9).astype(int).astype(str).agg("".join, axis=1)
    expected_base = categorical_text(train["base_state"]).map(base_map)
    check_relation("base_state_consistency", expected_base.eq(observed_base), 0)
    check_relation("run_total_consistency", numeric_values(train["run_total_before"]).eq(numeric_values(train["run_top_before"]) + numeric_values(train["run_bot_before"])), 0)
    home_diff = numeric_values(train["run_bot_before"]) - numeric_values(train["run_top_before"])
    check_relation("home_score_diff_consistency", numeric_values(train["score_diff_home"]).eq(home_diff), 0)
    pitcher_expected = np.where(train["top_bottom"].eq("T"), home_diff, -home_diff)
    check_relation("pitcher_team_score_diff_consistency", np.isclose(numeric_values(train["score_diff_pitcher_team"]), pitcher_expected, equal_nan=False), 0)
    expectancy_sum = numeric_values(train["home_win_expectancy"]) + numeric_values(train["away_win_expectancy"])
    check_relation("win_expectancy_sum", np.isclose(expectancy_sum, 100.0, atol=0.15, equal_nan=False), 0, "absolute tolerance 0.15")
    duplicate_n = train["asof_pitcher_n"].equals(train["asof_pitcher_pitchmix_n"])
    add_check("pitcher_n_pitchmix_n_exact_duplicate", "warn" if duplicate_n else "pass", duplicate_n, False, "warn means confirmed removable duplicate candidate")
    column_hashes = {}
    for column in [name for name in INPUT_COLUMNS if name not in {"row_id", "season"}]:
        hashed = pd.util.hash_pandas_object(train[column], index=False).to_numpy(dtype="uint64")
        column_hashes.setdefault(hashlib.sha256(hashed.tobytes()).hexdigest(), []).append(column)
    duplicate_pairs = []
    for candidates_with_same_hash in column_hashes.values():
        for left_index, left_column in enumerate(candidates_with_same_hash):
            for right_column in candidates_with_same_hash[left_index + 1:]:
                if train[left_column].equals(train[right_column]):
                    duplicate_pairs.append(f"{left_column}={right_column}")
    add_check(
        "all_exact_duplicate_feature_pairs", "warn" if duplicate_pairs else "pass",
        "|".join(sorted(duplicate_pairs)), "none", "warn marks removable duplicate candidates",
    )

    for count_column, prefixes in {
        "asof_pitcher_n": ("asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate"),
        "asof_batter_n": ("asof_batter_success_rate", "asof_batter_middle_rate"),
        "asof_pitcher_pitchmix_n": ("asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"),
    }.items():
        zero = numeric_values(train[count_column]).eq(0)
        for rate_column in prefixes:
            add_check(
                f"zero_count_missing_{rate_column}", "pass", int(train.loc[zero, rate_column].isna().sum()),
                f"diagnostic_only_of_{int(zero.sum())}", "both missing and non-missing are retained",
            )

    for column in INPUT_COLUMNS:
        train_numeric = pd.api.types.is_numeric_dtype(train[column])
        test_numeric = pd.api.types.is_numeric_dtype(test_schema[column])
        add_check(f"dtype_compatibility_{column}", "pass" if train_numeric == test_numeric else "warn", f"train_numeric={train_numeric},test_numeric={test_numeric}", "same broad dtype family")

    integrity_checks = pd.DataFrame(integrity_rows).sort_values("check", kind="stable").reset_index(drop=True)
    if integrity_checks["status"].eq("fail").any():
        failed = integrity_checks.loc[integrity_checks["status"].eq("fail"), "check"].tolist()
        raise RuntimeError(f"fatal integrity checks failed: {failed}")

    folds = make_expanding_folds(train["season"].dropna().tolist())
    NUMERIC_FEATURES = [column for column in INPUT_COLUMNS if column not in CATEGORICAL_COLUMNS and column not in {"row_id", "season"}]
    ANALYSIS_FEATURES = [column for column in INPUT_COLUMNS if column not in {"row_id", "season"}]
    fit_audit["test_rows_used"] = 0
    tables = {"integrity_checks.csv": integrity_checks}
    print(f"INTEGRITY_OK train_rows={len(train)} test_schema_rows={len(test_schema)} folds={len(folds)}")
except Exception as exc:
    print(f"EDA_ERROR stage={STAGE} type={type(exc).__name__} message={str(exc).replace(' ', '_')}")
    raise
```

## 셀 04 — 프로파일과 인접 시즌 분포 이동

```python
# CELL_ID: 04
STAGE = "profile_and_drift"

try:
    profile_rows = []
    scopes = [("all", "all", train)] + [
        ("season", str(int(season)), train.loc[train["season"].eq(season)])
        for season in sorted(train["season"].dropna().unique())
    ]
    for scope, season_label, frame in scopes:
        for column in ANALYSIS_FEATURES:
            series = frame[column]
            is_numeric = column in NUMERIC_FEATURES
            row = {
                "scope": scope, "season": season_label, "feature": column,
                "feature_type": "numeric" if is_numeric else "categorical", "rows": len(frame),
                "missing_rate": float(series.isna().mean()), "nunique": int(series.nunique(dropna=True)),
                "constant": bool(series.nunique(dropna=False) <= 1),
                "top_frequency_rate": float(series.astype("string").fillna("__MISSING__").value_counts(normalize=True, dropna=False).iloc[0]),
                "mean": None, "std": None, "min": None, "q01": None, "q25": None,
                "median": None, "q75": None, "q99": None, "max": None, "skew": None,
                "zero_rate": None, "tail_rate": None, "singleton_rate": None, "rare_rate": None,
            }
            if is_numeric:
                values = numeric_values(series)
                finite = values[np.isfinite(values)]
                if len(finite):
                    quant = finite.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
                    iqr = float(quant.loc[0.75] - quant.loc[0.25])
                    row.update({
                        "mean": finite_or_none(finite.mean()), "std": finite_or_none(finite.std()),
                        "min": finite_or_none(finite.min()), "q01": finite_or_none(quant.loc[0.01]),
                        "q25": finite_or_none(quant.loc[0.25]), "median": finite_or_none(quant.loc[0.5]),
                        "q75": finite_or_none(quant.loc[0.75]), "q99": finite_or_none(quant.loc[0.99]),
                        "max": finite_or_none(finite.max()), "skew": finite_or_none(finite.skew()),
                        "zero_rate": float(finite.eq(0).mean()),
                        "tail_rate": float(((finite < quant.loc[0.25] - 3 * iqr) | (finite > quant.loc[0.75] + 3 * iqr)).mean()) if iqr > 0 else 0.0,
                    })
            else:
                counts = categorical_text(series).value_counts()
                row["singleton_rate"] = float(counts.eq(1).mean())
                row["rare_rate"] = float(counts.le(RARE_COUNT_MAX).sum() / max(1, len(counts)))
            profile_rows.append(row)
    feature_profile = pd.DataFrame(profile_rows).sort_values(["scope", "season", "feature"], kind="stable")

    drift_rows = []
    seasons = sorted(int(value) for value in train["season"].dropna().unique())
    for left, right in zip(seasons[:-1], seasons[1:]):
        left_full = train.loc[train["season"].eq(left)]
        right_full = train.loc[train["season"].eq(right)]
        left_sample = sample_frame(left_full, DRIFT_SAMPLE_PER_SEASON, left * 10 + right)
        right_sample = sample_frame(right_full, DRIFT_SAMPLE_PER_SEASON, right * 10 + left)
        sample_audit[f"drift_{left}_{right}"] = {
            "left_rows": len(left_sample), "right_rows": len(right_sample),
            "left_id_sha256": sha256_values(left_sample["row_id"]),
            "right_id_sha256": sha256_values(right_sample["row_id"]),
        }
        for column in ANALYSIS_FEATURES:
            common = {
                "season_left": left, "season_right": right, "feature": column,
                "feature_type": "numeric" if column in NUMERIC_FEATURES else "categorical",
                "left_rows": len(left_sample), "right_rows": len(right_sample),
                "missing_rate_diff": float(right_full[column].isna().mean() - left_full[column].isna().mean()),
                "ks_statistic": None, "wasserstein": None, "wasserstein_iqr": None,
                "total_variation": None, "js_divergence": None, "unseen_rate": None,
                "outside_train_range_rate": None,
            }
            if column in NUMERIC_FEATURES:
                a = numeric_values(left_sample[column]).dropna().to_numpy()
                b = numeric_values(right_sample[column]).dropna().to_numpy()
                if len(a) and len(b):
                    ks = ks_2samp(a, b).statistic
                    wd = wasserstein_distance(a, b)
                    iqr = np.quantile(a, 0.75) - np.quantile(a, 0.25)
                    common.update({
                        "ks_statistic": float(ks), "wasserstein": float(wd),
                        "wasserstein_iqr": float(wd / iqr) if iqr > 0 else None,
                        "outside_train_range_rate": float(((b < np.min(a)) | (b > np.max(a))).mean()),
                    })
            else:
                a = categorical_text(left_sample[column])
                b = categorical_text(right_sample[column])
                labels = sorted(set(a.unique()).union(b.unique()))
                pa = a.value_counts(normalize=True).reindex(labels, fill_value=0).to_numpy(dtype="float64")
                pb = b.value_counts(normalize=True).reindex(labels, fill_value=0).to_numpy(dtype="float64")
                common.update({
                    "total_variation": float(0.5 * np.abs(pa - pb).sum()),
                    "js_divergence": float(jensenshannon(pa, pb, base=2) ** 2),
                    "unseen_rate": float((~b.isin(set(a.unique()))).mean()),
                })
            drift_rows.append(common)
    season_drift = pd.DataFrame(drift_rows).sort_values(["season_left", "season_right", "feature"], kind="stable")
    tables["feature_profile.csv"] = feature_profile
    tables["season_drift.csv"] = season_drift
    print(f"PROFILE_DRIFT_OK profile_rows={len(feature_profile)} drift_rows={len(season_drift)}")
except Exception as exc:
    print(f"EDA_ERROR stage={STAGE} type={type(exc).__name__} message={str(exc).replace(' ', '_')}")
    raise
```

## 셀 05 — 시간 전이 피처, 결측과 ID 진단

```python
# CELL_ID: 05
STAGE = "temporal_probes"

try:
    temporal_rows = []
    for column in ANALYSIS_FEATURES:
        temporal_rows.extend(
            run_temporal_probe(
                train, column, train[column],
                "numeric" if column in NUMERIC_FEATURES else "categorical", folds,
            )
        )
    temporal_univariate = pd.DataFrame(temporal_rows).sort_values(["feature", "scope", "fold"], kind="stable")

    missing_rows = []
    for column in ANALYSIS_FEATURES:
        if not train[column].isna().any():
            continue
        indicator = train[column].isna().astype("int8")
        probe_rows = run_temporal_probe(train, f"{column}__is_missing", indicator, "categorical", folds)
        for row in probe_rows:
            row["source_feature"] = column
            row["overall_missing_rate"] = float(indicator.mean())
            missing_rows.append(row)
    missing_columns = [
        "source_feature", "feature", "scope", "fold", "feature_type", "valid_rows",
        "overall_missing_rate", "prior_brier", "probe_brier", "delta_brier", "coverage",
        "unseen_rate", "bin_edges",
    ]
    missingness_probes = ensure_nonempty(pd.DataFrame(missing_rows), missing_columns, "no_missing_features")
    if not missingness_probes.empty and "source_feature" in missingness_probes:
        missingness_probes = missingness_probes.sort_values(["source_feature", "scope", "fold"], kind="stable")

    id_rows = []
    coverage_columns = sorted(CATEGORICAL_COLUMNS)
    for column in coverage_columns:
        for fold in folds:
            train_mask = train["season"].isin(fold["train_seasons"])
            valid_mask = train["season"].eq(fold["valid_season"])
            x_train = categorical_text(train.loc[train_mask, column])
            x_valid = categorical_text(train.loc[valid_mask, column])
            y_train = train.loc[train_mask, TARGET_COLUMN].to_numpy(dtype="float64")
            y_valid = train.loc[valid_mask, TARGET_COLUMN].to_numpy(dtype="float64")
            counts = x_train.value_counts()
            prediction, prior, coverage = fit_rate_predictions(x_train, y_train, x_valid)
            known = x_valid.isin(counts.index).to_numpy()
            prior_pred = np.full(len(y_valid), prior)
            id_rows.append({
                "fold": fold["name"], "feature": column, "segment": "overall",
                "field_kind": "id" if column in ID_COLUMNS else "low_cardinality",
                "train_vocabulary": int(len(counts)),
                "singleton_share": float(counts.eq(1).mean()),
                "rare_share": float(counts.le(RARE_COUNT_MAX).mean()),
                "valid_rows": len(y_valid), "unseen_rate": 1.0 - coverage,
                "known_rows": int(known.sum()), "cold_start_rows": int((~known).sum()),
                "target_rate": float(np.mean(y_valid)),
                "probe_brier": brier_score(y_valid, prediction),
                "prior_brier": brier_score(y_valid, prior_pred),
                "known_probe_brier": brier_score(y_valid[known], prediction[known]) if known.any() else None,
                "known_prior_brier": brier_score(y_valid[known], prior_pred[known]) if known.any() else None,
                "cold_start_brier": brier_score(y_valid[~known], prior_pred[~known]) if (~known).any() else None,
                "train_frequency_min": None, "train_frequency_max": None,
            })
            if column in ID_COLUMNS and counts.nunique() >= 2:
                quantile_count = min(10, int(counts.nunique()))
                edges = np.unique(np.quantile(counts.to_numpy(dtype="float64"), np.linspace(0, 1, quantile_count + 1)))
                if len(edges) >= 2:
                    edges[0], edges[-1] = -np.inf, np.inf
                    valid_frequency = x_valid.map(counts)
                    frequency_bucket = pd.cut(valid_frequency, bins=edges, include_lowest=True, duplicates="drop")
                    for bucket in frequency_bucket.dropna().cat.categories:
                        bucket_mask = frequency_bucket.eq(bucket).to_numpy()
                        if not bucket_mask.any():
                            continue
                        id_rows.append({
                            "fold": fold["name"], "feature": column,
                            "segment": f"train_frequency_{bucket}",
                            "field_kind": "id" if column in ID_COLUMNS else "low_cardinality",
                            "train_vocabulary": int(len(counts)),
                            "singleton_share": float(counts.eq(1).mean()),
                            "rare_share": float(counts.le(RARE_COUNT_MAX).mean()),
                            "valid_rows": int(bucket_mask.sum()), "unseen_rate": 0.0,
                            "known_rows": int(bucket_mask.sum()), "cold_start_rows": 0,
                            "target_rate": float(np.mean(y_valid[bucket_mask])),
                            "probe_brier": brier_score(y_valid[bucket_mask], prediction[bucket_mask]),
                            "prior_brier": brier_score(y_valid[bucket_mask], prior_pred[bucket_mask]),
                            "known_probe_brier": brier_score(y_valid[bucket_mask], prediction[bucket_mask]),
                            "known_prior_brier": brier_score(y_valid[bucket_mask], prior_pred[bucket_mask]),
                            "cold_start_brier": None,
                            "train_frequency_min": float(valid_frequency[bucket_mask].min()),
                            "train_frequency_max": float(valid_frequency[bucket_mask].max()),
                        })
    id_coverage = pd.DataFrame(id_rows).sort_values(["feature", "fold", "segment"], kind="stable")
    tables["temporal_univariate_probes.csv"] = temporal_univariate
    tables["missingness_probes.csv"] = missingness_probes
    tables["id_coverage.csv"] = id_coverage
    print(f"TEMPORAL_PROBES_OK feature_rows={len(temporal_univariate)} missing_rows={len(missingness_probes)} id_rows={len(id_coverage)}")
except Exception as exc:
    print(f"EDA_ERROR stage={STAGE} type={type(exc).__name__} message={str(exc).replace(' ', '_')}")
    raise
```

## 셀 06 — 인접 시즌 판별과 상관 군집 중요도

```python
# CELL_ID: 06
STAGE = "adversarial_validation"

def fit_adversarial_encoder(train_frame, valid_frame, feature_columns):
    train_encoded = pd.DataFrame(index=train_frame.index)
    valid_encoded = pd.DataFrame(index=valid_frame.index)
    for column in feature_columns:
        if column in NUMERIC_FEATURES:
            train_values = numeric_values(train_frame[column])
            median = float(train_values.median()) if train_values.notna().any() else 0.0
            train_encoded[column] = train_values.fillna(median)
            valid_encoded[column] = numeric_values(valid_frame[column]).fillna(median)
        else:
            train_text = categorical_text(train_frame[column])
            valid_text = categorical_text(valid_frame[column])
            frequency = train_text.value_counts(normalize=True)
            train_encoded[column] = train_text.map(frequency).fillna(0.0)
            valid_encoded[column] = valid_text.map(frequency).fillna(0.0)
    return train_encoded.astype("float32"), valid_encoded.astype("float32")


def correlation_components(frame, columns, threshold):
    if not columns:
        return []
    parent = {column: column for column in columns}
    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)
    correlation = frame.loc[:, columns].corr(method="spearman").abs()
    for index, left in enumerate(columns):
        for right in columns[index + 1:]:
            value = correlation.loc[left, right]
            if pd.notna(value) and value >= threshold:
                union(left, right)
    groups = {}
    for column in columns:
        groups.setdefault(find(column), []).append(column)
    return [sorted(group) for group in sorted(groups.values(), key=lambda values: values[0])]


try:
    correlation_sample = sample_frame(train, DRIFT_SAMPLE_PER_SEASON, 9090)
    sample_audit["correlation"] = {
        "rows": len(correlation_sample),
        "id_sha256": sha256_values(correlation_sample["row_id"]),
    }
    corr_input = correlation_sample[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    corr_groups = correlation_components(corr_input, NUMERIC_FEATURES, CORRELATION_THRESHOLD)
    correlation_rows = []
    for group_index, group in enumerate(corr_groups, start=1):
        for column in group:
            correlation_rows.append({
                "cluster_id": f"cluster_{group_index:03d}", "feature": column,
                "cluster_size": len(group), "threshold": CORRELATION_THRESHOLD,
                "members": "|".join(group),
            })
    correlation_clusters = pd.DataFrame(correlation_rows).sort_values(["cluster_id", "feature"], kind="stable")

    adversarial_rows, importance_rows = [], []
    seasons = sorted(int(value) for value in train["season"].dropna().unique())
    adversarial_features = ANALYSIS_FEATURES
    for left, right in zip(seasons[:-1], seasons[1:]):
        pair_limit = min(
            ADVERSARIAL_SAMPLE_PER_SEASON,
            int(train["season"].eq(left).sum()),
            int(train["season"].eq(right).sum()),
        )
        left_sample = sample_frame(train.loc[train["season"].eq(left)], pair_limit, left * 100 + right)
        right_sample = sample_frame(train.loc[train["season"].eq(right)], pair_limit, right * 100 + left)
        combined = pd.concat([left_sample, right_sample], ignore_index=True)
        labels = np.concatenate([np.zeros(len(left_sample), dtype="int8"), np.ones(len(right_sample), dtype="int8")])
        indices = np.arange(len(combined))
        fit_index, holdout_index = train_test_split(indices, test_size=0.2, stratify=labels, random_state=SEED)
        fit_frame, holdout_frame = combined.iloc[fit_index], combined.iloc[holdout_index]
        x_fit, x_holdout = fit_adversarial_encoder(fit_frame, holdout_frame, adversarial_features)
        y_fit, y_holdout = labels[fit_index], labels[holdout_index]
        model = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=31, learning_rate=0.08, random_state=SEED)
        model.fit(x_fit, y_fit)
        probability = model.predict_proba(x_holdout)[:, 1]
        baseline_auc = float(roc_auc_score(y_holdout, probability))
        adversarial_rows.append({
            "season_left": left, "season_right": right, "fit_rows": len(fit_index),
            "holdout_rows": len(holdout_index), "roc_auc": baseline_auc,
            "log_loss": float(log_loss(y_holdout, probability)),
            "left_sample_id_sha256": sha256_values(left_sample["row_id"]),
            "right_sample_id_sha256": sha256_values(right_sample["row_id"]),
        })
        importance_holdout = sample_frame(
            holdout_frame.assign(__label=y_holdout), IMPORTANCE_SAMPLE_MAX, left + right
        )
        selected_index = importance_holdout.index
        x_importance = x_holdout.loc[selected_index]
        y_importance = importance_holdout["__label"].to_numpy(dtype="int8")
        baseline = float(roc_auc_score(y_importance, model.predict_proba(x_importance)[:, 1]))
        rng = np.random.default_rng(SEED + left + right)
        for column in adversarial_features:
            drops = []
            for _ in range(IMPORTANCE_REPEATS):
                permuted = x_importance.copy()
                order = rng.permutation(len(permuted))
                permuted[column] = permuted[column].to_numpy()[order]
                drops.append(baseline - roc_auc_score(y_importance, model.predict_proba(permuted)[:, 1]))
            importance_rows.append({
                "season_left": left, "season_right": right, "importance_scope": "season_classifier",
                "importance_type": "individual", "group": column, "members": column,
                "holdout_rows": len(x_importance), "auc_drop_mean": float(np.mean(drops)),
                "auc_drop_std": float(np.std(drops)),
            })
        for group_index, group in enumerate((group for group in corr_groups if len(group) > 1), start=1):
            drops = []
            for _ in range(IMPORTANCE_REPEATS):
                permuted = x_importance.copy()
                order = rng.permutation(len(permuted))
                permuted.loc[:, group] = permuted.loc[:, group].to_numpy()[order]
                drops.append(baseline - roc_auc_score(y_importance, model.predict_proba(permuted)[:, 1]))
            importance_rows.append({
                "season_left": left, "season_right": right, "importance_scope": "season_classifier",
                "importance_type": "correlation_group", "group": f"group_{group_index:03d}",
                "members": "|".join(group), "holdout_rows": len(x_importance),
                "auc_drop_mean": float(np.mean(drops)), "auc_drop_std": float(np.std(drops)),
            })
        sample_audit[f"adversarial_{left}_{right}"] = {
            "left_rows": len(left_sample), "right_rows": len(right_sample),
            "importance_rows": len(x_importance), "importance_id_sha256": sha256_values(importance_holdout["row_id"]),
        }
    adversarial_validation = pd.DataFrame(adversarial_rows).sort_values(["season_left", "season_right"], kind="stable")
    adversarial_importance = pd.DataFrame(importance_rows).sort_values(["season_left", "season_right", "importance_type", "group"], kind="stable")
    tables["adversarial_validation.csv"] = adversarial_validation
    tables["adversarial_importance.csv"] = adversarial_importance
    tables["correlation_clusters.csv"] = correlation_clusters
    print(f"ADVERSARIAL_OK pairs={len(adversarial_validation)} importance_rows={len(adversarial_importance)}")
except Exception as exc:
    print(f"EDA_ERROR stage={STAGE} type={type(exc).__name__} message={str(exc).replace(' ', '_')}")
    raise
```

## 셀 07 — `asof_*`, DL 수치 변환과 의미 기반 상호작용

```python
# CELL_ID: 07
STAGE = "preprocessing_diagnostics"

EXPERIENCE_BINS = (-np.inf, 0, 10, 50, 100, 500, 1000, 5000, np.inf)
SUCCESS_RATE_PAIRS = {
    "asof_pitcher_success_rate": "asof_pitcher_n",
    "asof_batter_success_rate": "asof_batter_n",
    "asof_pitcher_prev1_game_success_rate": "asof_pitcher_n",
    "asof_pitcher_prev3_game_success_rate": "asof_pitcher_n",
    "asof_pitcher_prev5_game_success_rate": "asof_pitcher_n",
}
SMOOTHING_PAIRS = {
    "asof_pitcher_success_rate": "asof_pitcher_n",
    "asof_batter_success_rate": "asof_batter_n",
}

def distribution_summary(values):
    array = np.asarray(values, dtype="float64").reshape(-1)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"finite_rate": 0.0, "skew": None, "mad": None, "q01": None, "q99": None, "min": None, "max": None}
    median = float(np.median(finite))
    series = pd.Series(finite)
    return {
        "finite_rate": float(len(finite) / len(array)), "skew": finite_or_none(series.skew()),
        "mad": float(np.median(np.abs(finite - median))), "q01": float(np.quantile(finite, 0.01)),
        "q99": float(np.quantile(finite, 0.99)), "min": float(np.min(finite)), "max": float(np.max(finite)),
    }


try:
    reliability_rows = []
    for rate_column, count_column in SUCCESS_RATE_PAIRS.items():
        experience = pd.cut(numeric_values(train[count_column]), bins=EXPERIENCE_BINS, include_lowest=True).astype("string").fillna("__MISSING__")
        for season_label, mask in [("all", pd.Series(True, index=train.index))] + [
            (str(int(season)), train["season"].eq(season)) for season in sorted(train["season"].dropna().unique())
        ]:
            subset = pd.DataFrame({
                "bucket": experience.loc[mask], "rate": numeric_values(train.loc[mask, rate_column]),
                "target": numeric_values(train.loc[mask, TARGET_COLUMN]),
            })
            for bucket, group in subset.groupby("bucket", observed=True, sort=True):
                valid = group.dropna(subset=["rate", "target"])
                reliability_rows.append({
                    "season": season_label, "rate_feature": rate_column, "experience_column": count_column,
                    "experience_bucket": str(bucket), "rows": len(group), "valid_rate_rows": len(valid),
                    "mean_rate": float(valid["rate"].mean()) if len(valid) else None,
                    "observed_target_rate": float(valid["target"].mean()) if len(valid) else None,
                    "brier": brier_score(valid["target"], valid["rate"]) if len(valid) else None,
                    "calibration_gap": float(valid["rate"].mean() - valid["target"].mean()) if len(valid) else None,
                })
    asof_reliability = pd.DataFrame(reliability_rows).sort_values(["rate_feature", "season", "experience_bucket"], kind="stable")

    smoothing_rows = []
    for rate_column, count_column in SMOOTHING_PAIRS.items():
        for fold in folds:
            train_mask = train["season"].isin(fold["train_seasons"])
            valid_mask = train["season"].eq(fold["valid_season"])
            prior = float(train.loc[train_mask, TARGET_COLUMN].mean())
            y_valid = numeric_values(train.loc[valid_mask, TARGET_COLUMN]).to_numpy()
            rate = numeric_values(train.loc[valid_mask, rate_column]).to_numpy()
            count = numeric_values(train.loc[valid_mask, count_column]).fillna(0).clip(lower=0).to_numpy()
            for k in SMOOTHING_K_GRID:
                numerator = np.nan_to_num(rate * count, nan=0.0) + k * prior
                denominator = count + k
                prediction = np.divide(numerator, denominator, out=np.full_like(count, prior), where=denominator > 0)
                prediction = np.where(np.isfinite(rate) & (count > 0), prediction, prior)
                smoothing_rows.append({
                    "scope": "fold", "rate_feature": rate_column, "count_feature": count_column, "fold": fold["name"],
                    "k": k, "valid_rows": len(y_valid), "prior": prior,
                    "brier": brier_score(y_valid, prediction),
                    "prior_brier": brier_score(y_valid, np.full(len(y_valid), prior)),
                })
    smoothing_fold = pd.DataFrame(smoothing_rows)
    smoothing_fold["delta_brier"] = smoothing_fold["brier"] - smoothing_fold["prior_brier"]
    smoothing_aggregate_rows = []
    for (rate_feature, count_feature, k), group in smoothing_fold.groupby(["rate_feature", "count_feature", "k"], sort=True):
        weights = group["valid_rows"].to_numpy(dtype="float64")
        smoothing_aggregate_rows.append({
            "scope": "aggregate", "rate_feature": rate_feature, "count_feature": count_feature,
            "fold": "all_forward_folds", "k": k, "valid_rows": int(weights.sum()),
            "prior": float(np.average(group["prior"], weights=weights)),
            "brier": float(np.average(group["brier"], weights=weights)),
            "prior_brier": float(np.average(group["prior_brier"], weights=weights)),
            "delta_brier": float(np.average(group["delta_brier"], weights=weights)),
        })
    asof_smoothing = pd.concat([smoothing_fold, pd.DataFrame(smoothing_aggregate_rows)], ignore_index=True)
    asof_smoothing = asof_smoothing.sort_values(["rate_feature", "scope", "fold", "k"], kind="stable")

    transform_rows = []
    for fold_index, fold in enumerate(folds):
        fit_full = train.loc[train["season"].isin(fold["train_seasons"])]
        valid_full = train.loc[train["season"].eq(fold["valid_season"])]
        fit_sample = sample_frame(fit_full, TRANSFORM_SAMPLE_MAX, 7000 + fold_index)
        valid_sample = sample_frame(valid_full, TRANSFORM_SAMPLE_MAX, 8000 + fold_index)
        sample_audit[f"numeric_transform_{fold['name']}"] = {
            "fit_rows": len(fit_sample), "valid_rows": len(valid_sample),
            "fit_id_sha256": sha256_values(fit_sample["row_id"]),
            "valid_id_sha256": sha256_values(valid_sample["row_id"]),
        }
        for column in NUMERIC_FEATURES:
            fit_values = numeric_values(fit_sample[column]).to_numpy().reshape(-1, 1)
            valid_values = numeric_values(valid_sample[column]).to_numpy().reshape(-1, 1)
            finite_fit = fit_values[np.isfinite(fit_values)]
            median = float(np.median(finite_fit)) if len(finite_fit) else 0.0
            fit_filled = np.nan_to_num(fit_values, nan=median, posinf=median, neginf=median)
            valid_filled = np.nan_to_num(valid_values, nan=median, posinf=median, neginf=median)
            unique_count = int(np.unique(fit_filled).size)
            transformers = {
                "raw_median": None,
                "standard": StandardScaler(),
                "robust": RobustScaler(),
            }
            if unique_count >= 2:
                transformers["quantile_normal"] = QuantileTransformer(
                    n_quantiles=min(1000, unique_count, len(fit_filled)), output_distribution="normal",
                    subsample=min(TRANSFORM_SAMPLE_MAX, len(fit_filled)), random_state=SEED,
                )
                transformers["yeo_johnson"] = PowerTransformer(method="yeo-johnson", standardize=True)
            for transform_name, transformer in transformers.items():
                try:
                    if transformer is None:
                        transformed_fit, transformed_valid = fit_filled, valid_filled
                    else:
                        transformed_fit = transformer.fit_transform(fit_filled)
                        transformed_valid = transformer.transform(valid_filled)
                    fit_summary = distribution_summary(transformed_fit)
                    valid_summary = distribution_summary(transformed_valid)
                    fit_finite = transformed_fit[np.isfinite(transformed_fit)]
                    valid_flat = transformed_valid.reshape(-1)
                    outside = float(((valid_flat < np.min(fit_finite)) | (valid_flat > np.max(fit_finite))).mean()) if len(fit_finite) else None
                    transform_rows.append({
                        "fold": fold["name"], "feature": column, "transform": transform_name,
                        "fit_rows": len(fit_filled), "valid_rows": len(valid_filled), "unique_fit_values": unique_count,
                        **{f"fit_{key}": value for key, value in fit_summary.items()},
                        **{f"valid_{key}": value for key, value in valid_summary.items()},
                        "valid_outside_fit_range_rate": outside, "status": "ok", "error": "",
                    })
                except Exception as transform_exc:
                    transform_rows.append({
                        "fold": fold["name"], "feature": column, "transform": transform_name,
                        "fit_rows": len(fit_filled), "valid_rows": len(valid_filled), "unique_fit_values": unique_count,
                        "fit_finite_rate": None, "fit_skew": None, "fit_mad": None, "fit_q01": None,
                        "fit_q99": None, "fit_min": None, "fit_max": None, "valid_finite_rate": None,
                        "valid_skew": None, "valid_mad": None, "valid_q01": None, "valid_q99": None,
                        "valid_min": None, "valid_max": None, "valid_outside_fit_range_rate": None,
                        "status": "not_applicable", "error": f"{type(transform_exc).__name__}: {transform_exc}",
                    })
    numeric_transform_diagnostics = pd.DataFrame(transform_rows).sort_values(["feature", "fold", "transform"], kind="stable")

    count_state = categorical_text(train["balls_before"]) + "-" + categorical_text(train["strikes_before"])
    score_diff = numeric_values(train["score_diff_pitcher_team"])
    interactions = {
        "count_state": (count_state, "categorical"),
        "hand_matchup": (categorical_text(train["pitcher_hand"]) + "-" + categorical_text(train["batter_hand"]), "categorical"),
        "base_out_state": (categorical_text(train["base_state"]) + "-" + categorical_text(train["outs_before"]), "categorical"),
        "game_type_count_state": (categorical_text(train["game_type"]) + "-" + count_state, "categorical"),
        "pitcher_score_state": (pd.Series(np.select([score_diff.gt(0), score_diff.lt(0)], ["leading", "trailing"], default="tied"), index=train.index), "categorical"),
        "abs_pitcher_score_diff": (score_diff.abs(), "numeric"),
        "pitcher_team_win_expectancy": (pd.Series(np.where(train["top_bottom"].eq("T"), numeric_values(train["home_win_expectancy"]), numeric_values(train["away_win_expectancy"])), index=train.index), "numeric"),
    }
    interaction_rows = []
    for name, (values, kind) in interactions.items():
        interaction_rows.extend(run_temporal_probe(train, name, values, kind, folds))
    interaction_probes = pd.DataFrame(interaction_rows).sort_values(["feature", "scope", "fold"], kind="stable")

    tables["asof_reliability.csv"] = asof_reliability
    tables["asof_smoothing_grid.csv"] = asof_smoothing
    tables["numeric_transform_diagnostics.csv"] = numeric_transform_diagnostics
    tables["interaction_probes.csv"] = interaction_probes
    print(f"PREPROCESSING_DIAGNOSTICS_OK reliability={len(asof_reliability)} smoothing={len(asof_smoothing)} transforms={len(numeric_transform_diagnostics)} interactions={len(interaction_probes)}")
except Exception as exc:
    print(f"EDA_ERROR stage={STAGE} type={type(exc).__name__} message={str(exc).replace(' ', '_')}")
    raise
```

## 셀 08 — 그래프, 요약, 검증과 결과 게시

```python
# CELL_ID: 08
STAGE = "publish"

def save_figure(filename):
    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()


try:
    season_summary = train.groupby("season", sort=True)[TARGET_COLUMN].agg(["size", "mean"]).reset_index()
    fig, left_axis = plt.subplots(figsize=(9, 5))
    left_axis.bar(season_summary["season"].astype(str), season_summary["size"], color="#4C78A8")
    left_axis.set_ylabel("rows")
    right_axis = left_axis.twinx()
    right_axis.plot(season_summary["season"].astype(str), season_summary["mean"], color="#E45756", marker="o")
    right_axis.set_ylabel("target rate")
    plt.title("Season rows and target rate")
    save_figure("01_season_rows_target_rate.png")

    missing_plot = tables["feature_profile.csv"].query("scope == 'all'").nlargest(20, "missing_rate")
    plt.figure(figsize=(10, 6)); plt.barh(missing_plot["feature"], missing_plot["missing_rate"], color="#72B7B2"); plt.gca().invert_yaxis(); plt.title("Top missing rates")
    save_figure("02_missing_rates.png")

    drift_plot = tables["season_drift.csv"].copy()
    drift_plot["effect"] = drift_plot[["ks_statistic", "total_variation"]].max(axis=1, skipna=True)
    drift_plot = drift_plot.nlargest(20, "effect")
    plt.figure(figsize=(10, 6)); plt.barh(drift_plot["feature"] + " " + drift_plot["season_left"].astype(str) + "→" + drift_plot["season_right"].astype(str), drift_plot["effect"], color="#F58518"); plt.gca().invert_yaxis(); plt.title("Largest adjacent-season effects")
    save_figure("03_season_drift.png")

    probe_plot = tables["temporal_univariate_probes.csv"].query("scope == 'aggregate'").nsmallest(20, "delta_brier")
    plt.figure(figsize=(10, 6)); plt.barh(probe_plot["feature"], probe_plot["delta_brier"], color="#54A24B"); plt.gca().invert_yaxis(); plt.axvline(0, color="black", linewidth=1); plt.title("Temporal univariate Brier delta")
    save_figure("04_temporal_probes.png")

    adversarial = tables["adversarial_validation.csv"]
    labels = adversarial["season_left"].astype(str) + "→" + adversarial["season_right"].astype(str)
    plt.figure(figsize=(8, 5)); plt.bar(labels, adversarial["roc_auc"], color="#B279A2"); plt.axhline(0.5, color="black", linestyle="--"); plt.ylim(0.45, 1.0); plt.title("Adjacent-season classifier AUC")
    save_figure("05_adversarial_auc.png")

    id_plot = tables["id_coverage.csv"].loc[
        tables["id_coverage.csv"]["field_kind"].eq("id")
        & tables["id_coverage.csv"]["segment"].eq("overall")
    ]
    id_plot = id_plot.assign(label=id_plot["feature"] + " " + id_plot["fold"])
    plt.figure(figsize=(10, 6)); plt.barh(id_plot["label"], id_plot["unseen_rate"], color="#FF9DA6"); plt.gca().invert_yaxis(); plt.title("ID next-season unseen rate")
    save_figure("06_id_unseen_rate.png")

    reliability = tables["asof_reliability.csv"].loc[tables["asof_reliability.csv"]["season"].eq("all")].dropna(subset=["mean_rate", "observed_target_rate"])
    plt.figure(figsize=(7, 7))
    for feature, group in reliability.groupby("rate_feature", sort=True):
        plt.scatter(group["mean_rate"], group["observed_target_rate"], label=feature, alpha=0.75)
    plt.plot([0, 1], [0, 1], color="black", linestyle="--"); plt.xlabel("mean rate"); plt.ylabel("observed target rate"); plt.legend(fontsize=7); plt.title("Success-rate reliability")
    save_figure("07_asof_reliability.png")

    smoothing = tables["asof_smoothing_grid.csv"].loc[
        tables["asof_smoothing_grid.csv"]["scope"].eq("aggregate")
    ]
    fig, (left_axis, right_axis) = plt.subplots(1, 2, figsize=(15, 5))
    for feature, group in smoothing.groupby("rate_feature", sort=True):
        left_axis.plot(group["k"], group["brier"], marker="o", label=feature)
    left_axis.set_xscale("symlog", linthresh=1); left_axis.set_xlabel("K"); left_axis.set_ylabel("weighted Brier"); left_axis.legend(fontsize=8); left_axis.set_title("Smoothing grid")
    transform_plot = tables["numeric_transform_diagnostics.csv"].loc[
        tables["numeric_transform_diagnostics.csv"]["status"].eq("ok")
    ].groupby("transform", sort=True)["valid_skew"].apply(lambda values: float(np.nanmedian(np.abs(values)))).sort_values()
    right_axis.barh(transform_plot.index, transform_plot.values, color="#9D755D")
    right_axis.set_xlabel("median absolute validation skew"); right_axis.set_title("Numeric transform distribution")
    save_figure("08_smoothing_grid.png")

    expected_table_names = set(CSV_OUTPUTS)
    if set(tables) != expected_table_names:
        raise RuntimeError(f"table contract mismatch missing={sorted(expected_table_names - set(tables))} extra={sorted(set(tables) - expected_table_names)}")
    for filename in CSV_OUTPUTS:
        frame = tables[filename]
        if frame.empty:
            raise RuntimeError(f"empty required table: {filename}")
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.size and np.isinf(numeric.to_numpy(dtype="float64")).any():
            raise RuntimeError(f"infinite numeric value in table: {filename}")
        output_frame = frame.replace([np.inf, -np.inf], np.nan).astype(object).where(frame.notna(), "not_applicable")
        output_frame.to_csv(runtime_output_dir / filename, index=False)

    aggregate_probes = tables["temporal_univariate_probes.csv"].query("scope == 'aggregate'").sort_values("delta_brier")
    top_drift = drift_plot[["feature", "season_left", "season_right", "effect"]].head(10).to_dict("records")
    top_probe = aggregate_probes[["feature", "delta_brier", "coverage"]].head(15).to_dict("records")
    summary = {
        "status": "complete", "run_id": RUN_ID, "test_usage": TEST_USAGE,
        "train_rows": len(train), "test_schema_rows": len(test_schema),
        "fatal_integrity_failures": 0,
        "integrity_warnings": int(tables["integrity_checks.csv"]["status"].eq("warn").sum()),
        "top_temporal_probes": top_probe, "top_drift": top_drift,
        "adversarial_auc": tables["adversarial_validation.csv"].to_dict("records"),
        "followup_candidates": [
            {"candidate": "model_specific_missing_policy", "evidence_file": "missingness_probes.csv", "model_families": ["CatBoost", "XGBoost", "TabM", "DeepFM"], "ablation": "native vs selective indicator", "automatic_adoption": False},
            {"candidate": "success_rate_smoothing", "evidence_file": "asof_smoothing_grid.csv", "model_families": ["CatBoost", "XGBoost", "TabM", "DeepFM"], "ablation": "raw vs fixed-K smoothing", "automatic_adoption": False},
            {"candidate": "numeric_representation", "evidence_file": "numeric_transform_diagnostics.csv", "model_families": ["TabM", "DeepFM"], "ablation": "raw/standard/robust/quantile/power", "automatic_adoption": False},
            {"candidate": "id_cold_start_policy", "evidence_file": "id_coverage.csv", "model_families": ["CatBoost", "TabM", "DeepFM"], "ablation": "native/frequency bucket/OOV field", "automatic_adoption": False},
            {"candidate": "semantic_interactions", "evidence_file": "interaction_probes.csv", "model_families": ["CatBoost", "XGBoost", "TabM", "DeepFM"], "ablation": "one interaction family at a time", "automatic_adoption": False},
        ],
    }
    manifest = {
        "run_id": RUN_ID, "seed": SEED, "archive_path": str(archive_path),
        "archive_sha256": archive_sha256, "train_member": train_member,
        "train_member_sha256": train_member_sha256, "test_member": test_member,
        "test_member_sha256": test_member_sha256, "train_rows": len(train),
        "test_schema_rows": len(test_schema),
        "train_columns": list(train.columns), "test_columns": list(test_schema.columns),
        "train_dtypes": {column: str(dtype) for column, dtype in train.dtypes.items()},
        "test_dtypes": {column: str(dtype) for column, dtype in test_schema.dtypes.items()},
        "fit_audit": fit_audit,
        "folds": folds, "sample_audit": sample_audit,
        "settings": {
            "smooth_k": SMOOTH_K, "numeric_bin_count": NUMERIC_BIN_COUNT,
            "rare_count_max": RARE_COUNT_MAX, "drift_sample_per_season": DRIFT_SAMPLE_PER_SEASON,
            "adversarial_sample_per_season": ADVERSARIAL_SAMPLE_PER_SEASON,
            "importance_sample_max": IMPORTANCE_SAMPLE_MAX, "importance_repeats": IMPORTANCE_REPEATS,
            "correlation_threshold": CORRELATION_THRESHOLD, "transform_sample_max": TRANSFORM_SAMPLE_MAX,
            "smoothing_k_grid": SMOOTHING_K_GRID,
        },
        "versions": {"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "elapsed_seconds": time.time() - started_at,
    }
    for filename, payload in (("eda_summary.json", summary), ("run_manifest.json", manifest)):
        with (runtime_output_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)

    required = [runtime_output_dir / name for name in (*CSV_OUTPUTS, *JSON_OUTPUTS)]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"missing or empty outputs: {missing}")
    plot_files = sorted(plots_dir.glob("*.png"))
    if len(plot_files) != 8 or any(path.stat().st_size == 0 for path in plot_files):
        raise RuntimeError(f"plot contract failed: {[path.name for path in plot_files]}")
    if fit_audit["test_rows_used"] != 0:
        raise RuntimeError("test rows entered a fit operation")
    for fold in folds:
        if max(fold["train_seasons"]) >= fold["valid_season"]:
            raise RuntimeError(f"temporal fold violation: {fold}")
    assert make_expanding_folds([2021, 2022, 2023])[-1]["train_seasons"] == (2021, 2022)
    assert stable_sample_ids(["c", "a", "b"], 2, SEED) == stable_sample_ids(["b", "c", "a"], 2, SEED)
    assert json_safe([float("inf"), 1.0]) == [None, 1.0]

    output_root.mkdir(parents=True, exist_ok=True)
    drive_staging = output_root / f".{RUN_ID}.incomplete_{uuid.uuid4().hex}"
    if drive_staging.exists() or final_output_dir.exists():
        raise FileExistsError("publication target already exists")
    shutil.copytree(runtime_output_dir, drive_staging)
    for relative in (*CSV_OUTPUTS, *JSON_OUTPUTS):
        if not (drive_staging / relative).is_file():
            raise RuntimeError(f"Drive copy verification failed: {relative}")
    if len(list((drive_staging / "plots").glob("*.png"))) != 8:
        raise RuntimeError("Drive plot copy verification failed")
    drive_staging.rename(final_output_dir)
    summary_path = final_output_dir / "eda_summary.json"
    print(f"EDA_SUCCESS run_id={RUN_ID} output_dir={final_output_dir} summary={summary_path}")
except Exception as exc:
    print(f"EDA_ERROR stage={STAGE} type={type(exc).__name__} message={str(exc).replace(' ', '_')}")
    raise
```
