# CatBoost 50:50 Deployment Realignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 세 개의 시간 폴드에서 같은 CatBoost tree 수와 `0.50 × TabM + 0.50 × CatBoost`가 고정 gate를 통과할 때만 전체 CatBoost 학습 delivery를 만드는 Colab T4 캠페인을 구현한다.

**Architecture:** 기존 70:30 배포 패키지는 수정하지 않고 `experiments/catboost_50_50_realign`에 계약, 입력 검증, metric, 학습, resume artifact, runner와 Colab supervisor를 분리한다. 로컬 준비 도구가 다섯 source artifact를 하나의 검증된 handoff ZIP으로 만들고, Colab은 신규 2022 fold만 학습한 뒤 기존 두 fold evidence와 합쳐 고정 prefix를 평가한다. 통과 전에는 full fit과 delivery를 차단하고, 이 프로젝트에는 추론·submission writer를 만들지 않는다.

**Tech Stack:** Python 3.11, pandas, NumPy, PyTorch/TabM 0.0.3, CatBoost 1.2.10, dataclasses, `Decimal`, ZIP/JSON SHA-256 manifests, pytest, Google Colab T4.

---

## 실행 전제와 파일 구조

현재 main worktree에는 사용자의 미커밋 TabM 파일이 있다. 실행을 시작할 때
`superpowers:using-git-worktrees`로 현재 HEAD에서 별도
`codex/catboost-50-50-realignment` worktree를 만든다. 기존 dirty 파일을 stash,
checkout, reset 또는 수정하지 않는다.

새 파일의 책임은 다음처럼 고정한다.

```text
experiments/catboost_50_50_realign/
├── __init__.py
├── contract.json
├── contracts.py
├── inputs.py
├── metrics.py
├── tabm_fold.py
├── training.py
├── state.py
├── artifacts.py
├── runner.py
├── colab.py
├── runtime_inventory.py
├── requirements-colab.txt
└── COLAB_CATBOOST_50_50_REALIGN_CELL.py

tools/
├── prepare_catboost_50_50_realign_input.py
└── build_catboost_50_50_realign_colab_cell.py

tests/
├── test_catboost_50_50_realign_contracts.py
├── test_catboost_50_50_realign_inputs.py
├── test_catboost_50_50_realign_metrics.py
├── test_catboost_50_50_realign_tabm_fold.py
├── test_catboost_50_50_realign_training.py
├── test_catboost_50_50_realign_artifacts.py
├── test_catboost_50_50_realign_runner.py
├── test_catboost_50_50_realign_colab.py
└── test_catboost_50_50_realign_colab_cell.py
```

공통 테스트 Python은 다음 절대 경로를 사용한다.

```bash
PY=/Users/yonghyun/Documents/lg-aimers-9th-competition/artifacts/tabm_submission_python311/bin/python
```

## Task 1: 불변 계약과 세 job 정의

**Files:**
- Create: `experiments/catboost_50_50_realign/__init__.py`
- Create: `experiments/catboost_50_50_realign/contract.json`
- Create: `experiments/catboost_50_50_realign/contracts.py`
- Create: `tests/test_catboost_50_50_realign_contracts.py`

- [ ] **Step 1: 계약이 없어서 실패하는 테스트 작성**

```python
def test_contract_seals_fixed_grid_weight_and_three_folds():
    contract = load_contract()
    assert contract.campaign_id == "catboost_50_50_realign_v2"
    assert contract.tabm_weight == Decimal("0.50")
    assert contract.tree_prefixes == (4, 8, 12, 16, 20, 24, 28, 32)
    assert [(f.train_end_year, f.valid_year) for f in contract.folds] == [
        (2021, 2022), (2022, 2023), (2023, 2024)
    ]
    assert contract.minimum_weighted_gain == Decimal("0.00003")
    assert contract.maximum_segment_regression == Decimal("0.00075")
    assert contract.selection_tolerance == Decimal("1e-12")


def test_jobs_only_train_new_fold_and_full_fit():
    jobs = build_jobs(load_contract())
    assert [(j.job_id, j.kind) for j in jobs] == [
        ("tabm_f1_2022", "tabm_alignment"),
        ("catboost_f1_2022", "catboost_alignment"),
        ("catboost_full_2024", "full_fit"),
    ]
```

source SHA, weight, prefix 순서, fold, bootstrap seed/repeats, segment gate,
TabM seed/config, CatBoost parameter, 3시간 budget를 각각 바꾼 contract를 모두
거부하는 parametrized test도 작성한다.

- [ ] **Step 2: RED 확인**

```bash
$PY -m pytest tests/test_catboost_50_50_realign_contracts.py -q
```

Expected: collection 단계에서
`ModuleNotFoundError: experiments.catboost_50_50_realign`.

- [ ] **Step 3: canonical contract와 strict parser 작성**

`contract.json`의 핵심 값은 다음과 같다.

```json
{
  "schema_version": 1,
  "campaign_id": "catboost_50_50_realign_v2",
  "review_only": true,
  "source_sha256": {
    "training_input": "43557583fbd78efc0d3d83ab7ad17f701a1bd63f2cd423234d1dbee30006fe00",
    "stage_c_delivery": "f054e8e819d1053299fef4749a32e923db9136e20fc9c12cda79bbc7ef8c484a",
    "deployment_resume": "1eef761337e34054973a36fc4b098e278094f5fc7472b2b12240fc6f87be5f9f",
    "deployment_review": "fc65693aa9439bdbd808fcd29c421c1cfcfcb23abebd1a2c6512109b034641e8",
    "oof_audit": "0740e5904edd1bad7c4a11c9ca5953e3650e04e8b5b9c3cdc340612ae865409a"
  },
  "folds": [[2021, 2022], [2022, 2023], [2023, 2024]],
  "tabm_weight": "0.50",
  "tree_prefixes": [4, 8, 12, 16, 20, 24, 28, 32],
  "gates": {
    "minimum_weighted_gain": "0.00003",
    "latest_bootstrap_lower_minimum": "0",
    "maximum_segment_regression": "0.00075"
  },
  "selection_tolerance": "1e-12",
  "bootstrap": {"repeats": 2000, "seed": 3407},
  "budget": {"session_seconds": 10800, "new_job_guard_seconds": 900}
}
```

같은 파일에 TabM 설정
`p2/k32/width512/blocks4/dropout0.1/piecewise_linear/BCE/plateau/
lr0.0006/seed3407/max_epochs40/min_epochs3/patience10`과 기존 CatBoost
400-tree parameter를 정확히 넣는다.

`contracts.py`는 duplicate JSON key와 NaN/Infinity를 거부하고 문자열 threshold를
`Decimal`로 파싱한다. 공개 job builder는 다음 형태로 고정한다.

```python
@dataclass(frozen=True)
class RealignJob:
    job_id: str
    kind: str
    train_end_year: int
    valid_year: int | None
    seed: int


def build_jobs(contract: RealignContract) -> Sequence[RealignJob]:
    return (
        RealignJob("tabm_f1_2022", "tabm_alignment", 2021, 2022, 3407),
        RealignJob("catboost_f1_2022", "catboost_alignment", 2021, 2022, 42),
        RealignJob("catboost_full_2024", "full_fit", 2024, None, 42),
    )
```

- [ ] **Step 4: GREEN과 기존 계약 회귀 확인**

```bash
$PY -m pytest tests/test_catboost_50_50_realign_contracts.py \
  tests/test_catboost_deployment_contracts.py -q
```

Expected: PASS. 기존 `catboost_deployment_v1` bytes와 hash는 변경하지 않는다.

- [ ] **Step 5: 계약 커밋**

```bash
git add experiments/catboost_50_50_realign/__init__.py \
  experiments/catboost_50_50_realign/contract.json \
  experiments/catboost_50_50_realign/contracts.py \
  tests/test_catboost_50_50_realign_contracts.py
git commit -m "feat: seal CatBoost 50-50 realignment contract"
```

## Task 2: 다섯 artifact를 단일 입력 ZIP으로 준비

**Files:**
- Create: `experiments/catboost_50_50_realign/inputs.py`
- Create: `tools/prepare_catboost_50_50_realign_input.py`
- Create: `tests/test_catboost_50_50_realign_inputs.py`

- [ ] **Step 1: source 연결과 안전 ZIP 경계 RED 테스트 작성**

```python
sources = RealignSourcePaths(
    training_input=fixtures.training_input,
    stage_c_delivery=fixtures.stage_c_delivery,
    deployment_resume=fixtures.deployment_resume,
    deployment_review=fixtures.deployment_review,
    oof_audit=fixtures.oof_audit,
)
prepared = prepare_input_archive(
    sources, tmp_path / "realign_input.zip", contract
)
verified = verify_and_extract_input(
    prepared.path, tmp_path / "verified", contract
)
assert verified.fold_keys == ("2021->2022", "2022->2023", "2023->2024")
assert verified.audit_weight == Decimal("0.50")
```

다음 변조를 각각 명시적으로 실패시킨다.

```text
source outer SHA mismatch
duplicate member / ../ traversal / absolute path / symlink
member count, total bytes, per-member bytes, compression-ratio 초과
deployment resume prediction SHA != deployment review prediction SHA
audit inventory source SHA mismatch
audit fixed weight != 0.50
audit DIVERSE_BLEND evidence의 gain/bootstrap/segment 불일치
train 또는 Trackman history identity mismatch
test.csv, sample_submission.csv, submission.csv member 포함
```

- [ ] **Step 2: RED 확인**

```bash
$PY -m pytest tests/test_catboost_50_50_realign_inputs.py -q
```

Expected: `RealignSourcePaths` import 실패.

- [ ] **Step 3: streaming verifier와 handoff writer 구현**

공개 API를 다음처럼 고정한다.

```python
@dataclass(frozen=True)
class RealignSourcePaths:
    training_input: Path
    stage_c_delivery: Path
    deployment_resume: Path
    deployment_review: Path
    oof_audit: Path


@dataclass(frozen=True)
class PreparedRealignInput:
    path: Path
    sha256: str


@dataclass(frozen=True)
class VerifiedRealignInput:
    root: Path
    data_dir: Path
    tabm_predictions: Mapping[str, Path]
    catboost_models: Mapping[str, Path]
    catboost_states: Mapping[str, Path]
    catboost_predictions: Mapping[str, Path]
    audit_decision: Path
    manifest_sha256: str
    fold_keys: Sequence[str]
    audit_weight: Decimal


def prepare_input_archive(
    sources: RealignSourcePaths,
    output: Path,
    contract: RealignContract,
) -> PreparedRealignInput:
    verified = verify_source_artifacts(sources, contract)
    members = collect_handoff_members(verified, contract)
    write_deterministic_handoff(output, members, contract)
    return PreparedRealignInput(path=output, sha256=file_sha256(output))


def verify_and_extract_input(
    archive: Path,
    destination: Path,
    contract: RealignContract,
    check_deadline: Callable[[], None] | None = None,
) -> VerifiedRealignInput:
    manifest = verify_handoff_archive(archive, contract, check_deadline)
    extract_handoff_members(archive, destination, manifest, check_deadline)
    return bind_verified_input(destination, manifest, contract)
```

같은 Task에서 위 함수가 호출하는 private helper는 다음 반환 계약으로 구현한다.

| helper | 입력 | 반환 |
|---|---|---|
| `verify_source_artifacts` | `RealignSourcePaths`, `RealignContract` | `VerifiedSources` |
| `collect_handoff_members` | `VerifiedSources`, `RealignContract` | `Mapping[str, TrustedSource]` |
| `write_deterministic_handoff` | output path, trusted member map, contract | `None` |
| `verify_handoff_archive` | archive, contract, deadline callback | `HandoffManifest` |
| `extract_handoff_members` | archive, destination, manifest, deadline callback | `None` |

`TrustedSource`는 `(path, expected_sha256, expected_size)`를 담는 frozen dataclass다.
`VerifiedSources`는 다섯 verifier의 typed 결과를 담고, `HandoffManifest`는 exact
member map과 source binding만 담는다. helper 본문은 아래 streaming 규칙을 그대로
구현하며 이외의 입력 discovery 경로를 두지 않는다.

ZIP은 중앙 directory와 실제 stream을 두 번 검증하고, 각 member를 1 MiB chunk로
temporary regular file에 쓴 뒤 digest가 manifest와 일치할 때만 `os.replace`한다.
member bytes 전체를 dict에 누적하지 않는다. `O_NOFOLLOW`와 `fstat`로 source file
치환을 차단한다. 출력 manifest는 canonical JSON이며 source artifact SHA, extracted
member SHA, contract SHA와 `submission_package: false`를 기록한다.

CLI는 repository root를 `sys.path`에 넣어 이전
`ModuleNotFoundError: experiments`가 재발하지 않게 한다.

```python
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

성공 marker는 다음이다.

```text
REALIGN_INPUT_READY path=<absolute path> sha256=<64hex> size_bytes=<int>
```

- [ ] **Step 4: GREEN, 결정성, source 회귀 확인**

```bash
$PY -m pytest tests/test_catboost_50_50_realign_inputs.py \
  tests/test_catboost_deployment_artifacts.py \
  tests/test_oof_reset_audit_artifacts.py -q
```

Expected: PASS. 같은 fixture를 두 경로에 만들었을 때 ZIP bytes와 SHA가 동일하다.

- [ ] **Step 5: 입력 계층 커밋**

```bash
git add experiments/catboost_50_50_realign/inputs.py \
  tools/prepare_catboost_50_50_realign_input.py \
  tests/test_catboost_50_50_realign_inputs.py
git commit -m "feat: prepare verified 50-50 realignment input"
```

## Task 3: 세 fold metric과 결정 함수

**Files:**
- Create: `experiments/catboost_50_50_realign/metrics.py`
- Create: `tests/test_catboost_50_50_realign_metrics.py`

- [ ] **Step 1: 정렬, gate, 선택 순서 RED 테스트 작성**

prediction frame의 고정 schema는 다음이다.

```python
TABM_COLUMNS = (
    "row_id", "target", "probability", "game_type", "game_month",
    "pitcher_id_known", "batter_id_known",
)
CATBOOST_COLUMNS = (
    "row_id", "target", "p_4", "p_8", "p_12", "p_16",
    "p_20", "p_24", "p_28", "p_32", "game_type", "game_month",
    "pitcher_id_known", "batter_id_known",
)
```

세 fold 모두 양의 gain, weighted gain, 최신 bootstrap lower, 전체 eligible segment
regression을 각각 경계 바로 아래·같음·바로 위에서 검사한다.

```python
decision = evaluate_prefixes(tabm_by_fold, catboost_by_fold, contract)
assert decision.status == "promoted"
assert decision.selected_tree_count == 16
assert decision.selection_order == (
    "maximum_minimum_fold_gain",
    "maximum_weighted_gain",
    "minimum_tree_count",
)
```

row reversal은 같은 결과, row 삭제·중복·target/segment mismatch는 실패해야 한다.
NaN, infinity, 확률 범위 밖, fold 누락/추가도 거부한다.

- [ ] **Step 2: RED 확인**

```bash
$PY -m pytest tests/test_catboost_50_50_realign_metrics.py -q
```

Expected: `evaluate_prefixes` import 실패.

- [ ] **Step 3: metric과 immutable decision 구현**

```python
@dataclass(frozen=True)
class PrefixEvidence:
    tree_count: int
    fold_gain: Mapping[str, float]
    weighted_gain: float
    latest_bootstrap_lower: float | None
    maximum_segment_regression: float
    passed: bool


@dataclass(frozen=True)
class RealignDecision:
    status: str
    selected_tree_count: int | None
    candidates: Sequence[PrefixEvidence]
    reason: str


def _blend(tabm: np.ndarray, catboost: np.ndarray) -> np.ndarray:
    return 0.5 * tabm + 0.5 * np.clip(catboost, 0.0, 1.0)


def _passes(item: PrefixEvidence, contract: RealignContract) -> bool:
    return (
        all(Decimal(str(value)) > Decimal("0") for value in item.fold_gain.values())
        and Decimal(str(item.weighted_gain)) >= contract.minimum_weighted_gain
        and item.latest_bootstrap_lower is not None
        and Decimal(str(item.latest_bootstrap_lower)) >= Decimal("0")
        and Decimal(str(item.maximum_segment_regression))
            <= contract.maximum_segment_regression
    )
```

bootstrap은 기존 audit와 같은 2,000회/seed3407/block 정의를 사용하고 parity
fixture에서 `experiments.oof_reset_audit.metrics.block_bootstrap_interval`과 결과를
비교한다. segment도 같은 alias, minimum row와 eligible 정의를 사용한다.
bootstrap status가 `completed`가 아니거나 eligible segment가 하나도 없으면 해당
prefix는 통과하지 못한다. 선택은 `1e-12` 이내 값을 동률로 묶어 minimum fold gain,
weighted gain, 작은 tree 수 순으로 수행한다.

- [ ] **Step 4: GREEN과 audit parity 확인**

```bash
$PY -m pytest tests/test_catboost_50_50_realign_metrics.py \
  tests/test_oof_reset_audit_metrics.py tests/test_oof_reset_audit_run.py -q
```

Expected: PASS. threshold equality와 serialize→deserialize 후 결정이 같다.

- [ ] **Step 5: metric 커밋**

```bash
git add experiments/catboost_50_50_realign/metrics.py \
  tests/test_catboost_50_50_realign_metrics.py
git commit -m "feat: evaluate three-fold 50-50 realignment"
```

## Task 4: 신규 2022 TabM fold adapter

**Files:**
- Create: `experiments/catboost_50_50_realign/tabm_fold.py`
- Create: `tests/test_catboost_50_50_realign_tabm_fold.py`

- [ ] **Step 1: 봉인된 F1 job과 worker 결과 RED 테스트 작성**

```python
job = build_tabm_f1_job(load_contract())
assert asdict(job) == {
    "candidate_id": "realign_tabm__tr2021__va2022__s3407",
    "capacity": "p2", "k": 32, "width": 512, "blocks": 4,
    "dropout": 0.1, "num_embedding": "piecewise_linear",
    "loss": "bce", "scheduler": "plateau", "learning_rate": 0.0006,
    "seed": 3407, "train_end_year": 2021, "valid_year": 2022,
    "sample_mode": "full", "max_epochs": 40, "min_epochs": 3,
    "patience": 10, "feature_bundle": None,
}
```

fake worker 완료 결과는 exact schema, row alignment, candidate/config/checkpoint/
prediction SHA를 통과해야 한다. failed, inconclusive, missing artifact, 잘못된 fold와
binding은 거부한다.

- [ ] **Step 2: RED 확인**

```bash
$PY -m pytest tests/test_catboost_50_50_realign_tabm_fold.py -q
```

Expected: `build_tabm_f1_job` import 실패.

- [ ] **Step 3: 기존 worker를 좁게 감싸는 adapter 구현**

새 TabM trainer를 복제하지 않고 `CampaignJob`과 기존
`experiments.tabm_campaign.worker.run_worker`를 사용한다.

```python
@dataclass(frozen=True)
class TabMFoldResult:
    status: str
    checkpoint_path: Path | None
    predictions_path: Path | None
    completed_epochs: int
    brier: float | None


def run_tabm_f1(
    *,
    data_dir: Path,
    output_dir: Path,
    cache_root: Path,
    absolute_deadline: float,
    worker: Callable = run_worker,
) -> TabMFoldResult:
    job = build_tabm_f1_job(load_contract())
    raw = worker(job, data_dir, output_dir, cache_root, absolute_deadline)
    return validate_tabm_f1_result(raw, job, output_dir)
```

wrapper는 job canonical SHA, worker_result, checkpoint/meta, prediction SHA, training
source SHA와 `2021->2022` row/target/segment를 교차 확인한다. checkpoint payload는
기존 Stage P semantic validator를 재사용하고 완화 경로를 만들지 않는다.

- [ ] **Step 4: GREEN과 기존 worker 회귀 확인**

```bash
$PY -m pytest tests/test_catboost_50_50_realign_tabm_fold.py \
  tests/test_tabm_campaign_worker.py tests/test_tabm_campaign_training.py -q
```

Expected: PASS. 실제 GPU/공식 데이터는 실행하지 않는다.

- [ ] **Step 5: TabM adapter 커밋**

```bash
git add experiments/catboost_50_50_realign/tabm_fold.py \
  tests/test_catboost_50_50_realign_tabm_fold.py
git commit -m "feat: add sealed 2022 TabM fold job"
```

## Task 5: CatBoost F1 prefix와 조건부 전체 학습

**Files:**
- Create: `experiments/catboost_50_50_realign/training.py`
- Create: `tests/test_catboost_50_50_realign_training.py`

- [ ] **Step 1: fake CatBoost로 prefix/full-fit RED 테스트 작성**

```python
result = run_catboost_f1(
    data_dir=data_dir,
    output_dir=tmp_path / "f1",
    absolute_deadline=clock.now + 600,
    model_factory=factory,
)
assert factory.model.predict_prefixes == [4, 8, 12, 16, 20, 24, 28, 32]
assert tuple(pd.read_csv(result.predictions_path)) == CATBOOST_COLUMNS

with pytest.raises(RealignTrainingError, match="promoted decision"):
    run_full_fit(
        data_dir=data_dir,
        output_dir=tmp_path / "full",
        decision=blocked_decision,
        absolute_deadline=clock.now + 600,
        model_factory=factory,
    )
```

F1은 2019~2021만 train, 2022만 valid로 사용하고 full fit은 공식 train 모든 행을
한 번 사용한다. `use_best_model=False`, full fit eval_set 없음, 선택 tree 수와
iterations 일치를 검사한다.

- [ ] **Step 2: RED 확인**

```bash
$PY -m pytest tests/test_catboost_50_50_realign_training.py -q
```

Expected: `run_catboost_f1` import 실패.

- [ ] **Step 3: strict CatBoost training 구현**

```python
@dataclass(frozen=True)
class CatBoostJobResult:
    job_id: str
    status: str
    model_path: Path | None
    preprocessing_path: Path | None
    predictions_path: Path | None
    snapshot_path: Path | None


state, x_train = fit_catboost_features(train, components=("hand_matchup",))
x_valid = transform_catboost_features(valid, state)

if decision.status != "promoted" or (
    decision.selected_tree_count not in contract.tree_prefixes
):
    raise RealignTrainingError("promoted decision is required for full fit")
```

alignment model은 400 trees로 한 번 fit하고 고정 prefix만 예측한다. full fit은 선택
prefix로 모든 공식 train을 학습하며 validation/early stopping을 쓰지 않는다.
snapshot은 300초 간격이며 job/contract/input/code/decision SHA가 같을 때만
재사용한다. model, preprocessing, prediction과 metrics는 temporary file에서 SHA를
계산한 뒤 원자 게시한다. segment label은 공식 train-fitted state로 만든다.

- [ ] **Step 4: GREEN과 CatBoost 전처리 회귀 확인**

```bash
$PY -m pytest tests/test_catboost_50_50_realign_training.py \
  tests/test_catboost_deployment_training.py \
  tests/test_catboost_preprocessing.py -q
```

Expected: PASS. CatBoost import가 없는 로컬 환경에서도 fake factory만 사용한다.

- [ ] **Step 5: 학습 계층 커밋**

```bash
git add experiments/catboost_50_50_realign/training.py \
  tests/test_catboost_50_50_realign_training.py
git commit -m "feat: train fixed-prefix CatBoost realignment"
```

## Task 6: 상태·resume·review·delivery artifact

**Files:**
- Create: `experiments/catboost_50_50_realign/state.py`
- Create: `experiments/catboost_50_50_realign/artifacts.py`
- Create: `tests/test_catboost_50_50_realign_artifacts.py`

- [ ] **Step 1: 상태 전이와 artifact 변조 RED 테스트 작성**

허용 상태는 다음뿐이다.

```python
ALLOWED_STATUS = (
    "fresh", "f1_active", "f1_complete", "deployment_blocked",
    "full_fit_active", "completed",
)
```

`completed`에는 promoted decision과 full model이 필요하고
`deployment_blocked`에는 full model이 없어야 한다. completed→active 전이,
다른 binding resume, checkpoint/meta mismatch, prediction/decision/model 변조와
extra/missing member를 각각 실패시킨다.

```python
verified = verify_delivery_bundle(delivery, expected_bindings=bindings)
assert verified.status == "completed"
assert verified.submission_package is False
assert verified.selected_tree_count == 16
```

- [ ] **Step 2: RED 확인**

```bash
$PY -m pytest tests/test_catboost_50_50_realign_artifacts.py -q
```

Expected: `verify_delivery_bundle` import 실패.

- [ ] **Step 3: strict state와 streaming ZIP 구현**

```python
@dataclass(frozen=True)
class Bindings:
    contract_sha256: str
    code_sha256: str
    input_manifest_sha256: str
    train_sha256: str
    history_sha256: str


@dataclass(frozen=True)
class TrustedFile:
    path: Path
    sha256: str


@dataclass(frozen=True)
class CampaignFiles:
    status: str
    resume: Mapping[str, TrustedFile]
    review: Mapping[str, TrustedFile]
    delivery: Mapping[str, TrustedFile]

    def resume_members(self) -> Mapping[str, TrustedFile]:
        return self.resume

    def review_members(self) -> Mapping[str, TrustedFile]:
        return self.review

    def delivery_members(self) -> Mapping[str, TrustedFile]:
        return self.delivery

    def require_completed_delivery(self) -> None:
        if self.status != "completed" or not self.delivery:
            raise RealignArtifactError("completed delivery evidence is required")


@dataclass(frozen=True)
class RestoredRun:
    root: Path
    status: str
    state_path: Path


def write_resume_bundle(source: CampaignFiles, output: Path, bindings: Bindings) -> Path:
    return write_verified_bundle("realign_resume_v1", source.resume_members(), output, bindings)


def write_review_bundle(source: CampaignFiles, output: Path, bindings: Bindings) -> Path:
    return write_verified_bundle("realign_review_v1", source.review_members(), output, bindings)


def write_delivery_bundle(source: CampaignFiles, output: Path, bindings: Bindings) -> Path:
    source.require_completed_delivery()
    return write_verified_bundle("realign_delivery_v1", source.delivery_members(), output, bindings)


def restore_resume(
    path: Path, destination: Path, expected_bindings: Bindings
) -> RestoredRun:
    verified = verify_resume_bundle(path, expected_bindings)
    extract_verified_resume(path, destination, verified)
    return load_restored_run(destination, verified)
```

ZIP은 fixed timestamp, sorted member, canonical manifest를 사용한다. source path와
trusted expected SHA를 함께 전달해 manifest와 stream 사이 TOCTOU를 차단한다.
checkpoint는 크기 제한 후 isolated process에서 기존 semantic validator를 실행한다.
deadline callback을 hashing, ZIP write/read, extraction과 child poll에 전달한다.
`write_verified_bundle`, `verify_resume_bundle`, `extract_verified_resume`와
`load_restored_run`은 이 Task에서만 쓰는 private helper로 두고, 모두 `Bindings`와
member digest를 인자로 받아 self-described SHA를 신뢰하지 않는다.

delivery allowlist는 설계 문서의 decision 4개, frozen CatBoost 3개, log, policy,
manifest뿐이다. resume에는 checkpoint/snapshot과 state를 허용하지만 test,
submission과 cached inference prediction은 허용하지 않는다.

- [ ] **Step 4: GREEN과 기존 artifact 회귀 확인**

```bash
$PY -m pytest tests/test_catboost_50_50_realign_artifacts.py \
  tests/test_catboost_deployment_artifacts.py \
  tests/test_tabm_campaign_artifacts.py -q
```

Expected: PASS. 기존 A~D와 deployment fixture SHA는 그대로다.

- [ ] **Step 5: artifact 계층 커밋**

```bash
git add experiments/catboost_50_50_realign/state.py \
  experiments/catboost_50_50_realign/artifacts.py \
  tests/test_catboost_50_50_realign_artifacts.py
git commit -m "feat: add resumable 50-50 realignment artifacts"
```

## Task 7: 순차 runner와 통과 전 full-fit 차단

**Files:**
- Create: `experiments/catboost_50_50_realign/runner.py`
- Create: `tests/test_catboost_50_50_realign_runner.py`

- [ ] **Step 1: 상태기계 RED 테스트 작성**

fake runtime으로 다음 네 경로를 검사한다.

```text
fresh → TabM F1 → CatBoost F1 → promoted → full fit → completed
fresh → F1 완료 → no prefix passed → deployment_blocked
fresh → TabM checkpoint에서 budget_inconclusive → resumable incomplete
verified resume → completed job reuse → 남은 job만 실행
```

```python
assert runtime.calls == [
    "tabm_f1_2022",
    "catboost_f1_2022",
    "evaluate_prefixes",
    "catboost_full_2024",
]
```

blocked 경로에서는 full fit 호출이 0이고 delivery가 `None`이어야 한다. F2/F3는
학습 호출 없이 verified input model/state에서 prediction을 계산한다.

- [ ] **Step 2: RED 확인**

```bash
$PY -m pytest tests/test_catboost_50_50_realign_runner.py -q
```

Expected: `run_campaign` import 실패.

- [ ] **Step 3: orchestration 구현**

```python
@dataclass(frozen=True)
class RealignRun:
    status: str
    selected_tree_count: int | None
    review_bundle: Path
    resume_bundle: Path
    delivery_bundle: Path | None


def run_campaign(
    *,
    verified: VerifiedRealignInput,
    output_dir: Path,
    resume_bundle: Path | None,
    absolute_deadline: float,
    on_phase_resume: Callable[[Path, str], None] | None = None,
    tabm_runtime: Callable = run_tabm_f1,
    catboost_runtime: Callable = run_catboost_f1,
    full_runtime: Callable = run_full_fit,
) -> RealignRun:
    context = restore_or_initialize(verified, output_dir, resume_bundle)
    run_missing_f1_jobs(context, absolute_deadline, tabm_runtime, catboost_runtime)
    decision = evaluate_context(context)
    if decision.status != "promoted":
        return publish_blocked_run(context, decision, on_phase_resume)
    run_missing_full_fit(context, decision, absolute_deadline, full_runtime)
    return publish_completed_run(context, decision, on_phase_resume)
```

단계는 다음 순서를 지킨다.

```python
if not state.tabm_f1_complete:
    run_and_publish_tabm_f1()
if not state.catboost_f1_complete:
    run_and_publish_catboost_f1()
publish_phase_resume("f1_complete")
decision = evaluate_all_three_folds()
if decision.status != "promoted":
    return publish_blocked_review_and_resume()
if not state.full_fit_complete:
    run_full_fit_with_selected_prefix(decision)
return publish_completed_review_resume_and_delivery()
```

F2/F3 400-tree model은 restored preprocessing state로 exact validation rows를
transform해 8개 prefix를 계산한다. 기존 review의 공통 prefix 4/32와 row별 수치가
맞아야 decision으로 진행한다. resume decision은 재계산한 canonical bytes와 같을
때만 재사용한다.

`restore_or_initialize`, `run_missing_f1_jobs`, `evaluate_context`,
`publish_blocked_run`, `run_missing_full_fit`, `publish_completed_run`은 runner의
private 단계 함수다. 각각 `CampaignContext` 하나를 받아 state를 원자 게시하며,
각 단계가 끝날 때 Task 6 verifier를 통과한 bundle만 callback으로 넘긴다.

- [ ] **Step 4: GREEN과 no-submission audit 확인**

```bash
$PY -m pytest tests/test_catboost_50_50_realign_runner.py \
  tests/test_catboost_deployment_runner.py \
  tests/test_oof_reset_audit_run.py -q
rg -n "sample_submission|submission\\.csv|test\\.csv|package_submission" \
  experiments/catboost_50_50_realign
```

Expected: pytest PASS. `rg`는 입력 금지 member 문자열 외 실행 경로 hit가 없다.

- [ ] **Step 5: runner 커밋**

```bash
git add experiments/catboost_50_50_realign/runner.py \
  tests/test_catboost_50_50_realign_runner.py
git commit -m "feat: orchestrate guarded CatBoost realignment"
```

## Task 8: Colab supervisor, 단일 셀과 제한된 다운로드

**Files:**
- Create: `experiments/catboost_50_50_realign/colab.py`
- Create: `experiments/catboost_50_50_realign/runtime_inventory.py`
- Create: `experiments/catboost_50_50_realign/requirements-colab.txt`
- Create: `experiments/catboost_50_50_realign/COLAB_CATBOOST_50_50_REALIGN_CELL.py`
- Create: `tools/build_catboost_50_50_realign_colab_cell.py`
- Create: `tests/test_catboost_50_50_realign_colab.py`
- Create: `tests/test_catboost_50_50_realign_colab_cell.py`

- [ ] **Step 1: supervisor와 생성 셀 RED 테스트 작성**

업로드는 handoff 하나, 또는 handoff+resume 두 개만 허용한다.

```python
verified, resume = classify_and_verify_uploads(
    [input_zip], run_root=tmp_path / "run", contract=contract
)
assert resume is None
```

정상 download phase는 정확히 다음이다.

```python
assert [event.phase for event in downloads] == [
    "f1_complete", "completed_review", "completed_resume", "completed_delivery"
]
```

epoch가 여러 번 진행되어도 F1 완료 전 browser download가 없어야 한다.
오류/deadline에서는 latest verified resume 한 개만 요청한다. callback 실패는 이전
resume을 보존해야 한다. `deployment_blocked`의 terminal phase는 review와 resume만
요청하고 delivery는 요청하지 않는다.

생성 셀은 다음 정적 계약을 지킨다.

```python
cell = render_colab_cell(REPO_ROOT)
assert len(cell) < 1_000_000
assert b"files.upload()" in cell
assert b"drive.mount" not in cell
assert b"github" not in cell.lower()
assert b"http://" not in cell and b"https://" not in cell
assert checked_in_cell.read_bytes() == cell
```

- [ ] **Step 2: RED 확인**

```bash
$PY -m pytest tests/test_catboost_50_50_realign_colab.py \
  tests/test_catboost_50_50_realign_colab_cell.py -q
```

Expected: upload classifier와 renderer import 실패.

- [ ] **Step 3: supervisor와 sealed runtime 구현**

셀은 upload 전에 deadline을 잡는다.

```python
SESSION_DEADLINE = time.time() + 10800
uploaded = files.upload()
```

runtime root는 실행마다
`/content/catboost_50_50_realign/runs/<unique>`를 사용하고 검증된 upload cache만
재사용한다. CatBoost 1.2.10, TabM 0.0.3, rtdl-num-embeddings 0.0.12, torch/CUDA,
Tesla T4를 확인한다. CPU-only이면 GPU 작업 전에 실패한다.

runtime inventory는 explicit allowlist이며 contract와 requirements도 code identity에
포함한다. 격리된 temporary directory에서 F1 job build와 fake worker delayed import가
성공해야 한다.

정상 marker는 다음이다.

```text
REALIGN_CAMPAIGN_SUCCESS status=completed selected_tree_count=<n>
REALIGN_CAMPAIGN_SUCCESS status=deployment_blocked selected_tree_count=none
```

예외는 `REALIGN_ERROR stage=<stage> type=<type> message=<message>`를 출력하고
latest verified resume이 있으면 한 번 다운로드 요청한 뒤 다시 raise한다.

- [ ] **Step 4: renderer와 Colab GREEN 확인**

```bash
$PY tools/build_catboost_50_50_realign_colab_cell.py
$PY -m pytest tests/test_catboost_50_50_realign_colab.py \
  tests/test_catboost_50_50_realign_colab_cell.py \
  tests/test_catboost_deployment_colab.py \
  tests/test_catboost_deployment_colab_cell.py \
  tests/test_tabm_campaign_colab_cell.py -q
```

Expected: PASS. builder 두 번 실행 후 checked-in cell SHA와 size가 동일하다.

- [ ] **Step 5: Colab handoff 커밋**

```bash
git add experiments/catboost_50_50_realign/colab.py \
  experiments/catboost_50_50_realign/runtime_inventory.py \
  experiments/catboost_50_50_realign/requirements-colab.txt \
  experiments/catboost_50_50_realign/COLAB_CATBOOST_50_50_REALIGN_CELL.py \
  tools/build_catboost_50_50_realign_colab_cell.py \
  tests/test_catboost_50_50_realign_colab.py \
  tests/test_catboost_50_50_realign_colab_cell.py
git commit -m "feat: add Colab T4 50-50 realignment handoff"
```

## Task 9: 실행 안내, 전체 검증과 최종 리뷰

**Files:**
- Create: `docs/CATBOOST_50_50_REALIGN_RUNBOOK.md`
- Modify: `README.md`

- [ ] **Step 1: 실행 runbook 작성**

runbook에는 다섯 로컬 원본 경로를 받는 준비 명령, 생성 입력 ZIP, Colab T4 선택,
예상 1~2시간, 재실행 안전성, F1 완료·최종 성공·오류/시간제한의 세 다운로드 시점과
반환 파일을 적는다. 최종 성공 시에는 review, resume, delivery 세 파일을 연속으로
요청하므로 정상 실행의 browser download 요청은 총 네 번이다.

```text
catboost_50_50_realign_review.zip
catboost_50_50_realign_resume.zip
catboost_50_50_realign_delivery.zip   # promoted + full fit 성공 때만 존재
```

사용자가 전달할 것은 최종 review와 resume, 존재할 경우 delivery, 그리고
`REALIGN_CAMPAIGN_SUCCESS`부터 마지막 줄까지의 로그다. `deployment_blocked`이면
delivery가 없는 것이 정상이라고 설명한다.

- [ ] **Step 2: README에 runbook 링크 추가**

```markdown
- [CatBoost 50:50 배포 정렬 재검증 실행 안내](docs/CATBOOST_50_50_REALIGN_RUNBOOK.md)
```

- [ ] **Step 3: focused suite 실행**

```bash
$PY -m pytest \
  tests/test_catboost_50_50_realign_contracts.py \
  tests/test_catboost_50_50_realign_inputs.py \
  tests/test_catboost_50_50_realign_metrics.py \
  tests/test_catboost_50_50_realign_tabm_fold.py \
  tests/test_catboost_50_50_realign_training.py \
  tests/test_catboost_50_50_realign_artifacts.py \
  tests/test_catboost_50_50_realign_runner.py \
  tests/test_catboost_50_50_realign_colab.py \
  tests/test_catboost_50_50_realign_colab_cell.py -q
```

Expected: PASS. full-data/GPU 실행 없음.

- [ ] **Step 4: 관련 회귀와 정적 검사 실행**

```bash
$PY -m pytest tests/test_catboost_deployment_*.py \
  tests/test_catboost_tabm_blend_*.py \
  tests/test_oof_reset_audit_*.py \
  tests/test_tabm_campaign_worker.py \
  tests/test_tabm_campaign_training.py \
  tests/test_tabm_campaign_colab_cell.py -q
$PY -m compileall -q experiments/catboost_50_50_realign tools
git diff --check
rg -n 'TO''DO|TB''D|PLACE''HOLDER' experiments/catboost_50_50_realign \
  tests/test_catboost_50_50_realign_*.py \
  docs/CATBOOST_50_50_REALIGN_RUNBOOK.md
```

Expected: pytest와 compileall PASS, diff-check clean, placeholder scan no hit.

- [ ] **Step 5: 전체 suite와 artifact audit 실행**

```bash
$PY -m pytest -q
git ls-files | rg '\\.(zip|pt|cbm|cbsnapshot)$'
git status --short
```

Expected: 전체 suite PASS. 새 binary/archive가 tracked되지 않는다.

- [ ] **Step 6: 독립 코드 리뷰와 finding 수정**

`superpowers:requesting-code-review`로 다음 항목을 검토한다.

```text
Critical/Important correctness and competition-rule findings only
full fit cannot run before promoted decision
all five source identities are transitively bound
F2/F3 reused models reproduce sealed p_4/p_32 evidence
three folds share one weight/tree count
resume rejects stale or forged checkpoint/model/decision
downloads occur only at documented phases
no test inference, submission writer, Drive/GitHub/network source path
```

finding이 있으면 각각 RED test로 재현하고 최소 수정 후 Steps 3~5를 다시 실행한다.
리뷰 승인 전에는 완료라고 보고하지 않는다.

- [ ] **Step 7: 문서와 최종 검증 커밋**

```bash
git add README.md docs/CATBOOST_50_50_REALIGN_RUNBOOK.md
git commit -m "docs: explain CatBoost 50-50 realignment run"
git status --short
```

Expected: worktree clean. push, Colab 실행, full-data 학습과 제출물 생성은 하지 않는다.

## 구현 완료 조건

- 고정 다섯 source SHA와 단일 handoff가 재귀 검증된다.
- 신규 F1 TabM/CatBoost와 기존 F2/F3 evidence가 같은 8-prefix 계약으로 정렬된다.
- 세 fold gate와 deterministic selection이 exact boundary test를 통과한다.
- `deployment_blocked`는 full fit/delivery를 만들 수 없다.
- promoted일 때만 선택 prefix로 CatBoost 전체 학습 delivery를 만든다.
- Colab은 T4 한 셀, 단일 handoff 업로드, 제한된 다운로드와 verified resume을 지원한다.
- 기존 TabM, CatBoost 70:30, OOF audit와 제출 후보 코드는 변경되지 않는다.
- 제출 ZIP 생성 경로는 존재하지 않는다.
