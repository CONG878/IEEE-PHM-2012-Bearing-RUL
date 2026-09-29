# 04_Dataset Construction

---

## 1. 목적

04 Dataset Builder는 02의 Engineered Dataset과 03에서 결정된 Candidate
Feature / Target 정보를 바탕으로 05가 사용할 Dataset을 구성하고,
Learning 데이터의 Prefix Evaluation Scenario를 정의한다.

현재 구조에서는 03이 최종 모델 입력 Feature를 확정하는 것이 아니라
`candidate_features`를 결정하고, 05가 그 Candidate 집합 안에서
`SELECTED_FEATURE_COLUMNS`를 통해 최종 모델 Feature를 확정한다. 따라서
`model_ready_*`라는 기존 산출물 명칭은 유지하되, 의미상으로는
**Candidate Feature Dataset**이다.

## 2. 책임 범위

04의 책임은 다음 세 가지다.

1.  Candidate Dataset Construction
2.  Model-ready Dataset Validation
3.  Learning-only Prefix Evaluation Scenario Definition

04는 Feature Transformation, RUL Target Generation, EDA, Candidate/Final
Feature Selection, 모델 학습, LOBO Validation, Test 추론, Ensemble,
Scaling, Sequence/Windowing을 수행하지 않는다.

## 3. 현재 Feature 결정 구조

``` text
02 Engineered Dataset
        ↓
03 EDA & Candidate Selection
        ↓
candidate_features + selected_target
        ↓
04 Dataset Builder
        ↓
model_ready_learning/test_dataset
        ↓
05 Final Feature Selection
        ↓
SELECTED_FEATURE_COLUMNS
        ↓
LOBO Training
```

03의 `candidate_features`는 EDA Evidence와 사용자 판단을 통해 결정된
모델링 후보 집합이다. 05는 그 안에서 최종 모델 Feature를 별도로
확정한다.

따라서:

``` text
CANDIDATE_FEATURES ≠ SELECTED_FEATURE_COLUMNS
```

이다.

## 4. 입력 인터페이스

04의 입력은 다음과 같다.

``` text
outputs/02_feature_transformation/
├── learning_engineered_features.csv
└── test_engineered_features.csv

outputs/03_eda_selection/
└── selection_config.json
```

Selection Manifest의 핵심 구조는:

``` json
{
    "candidate_features": [...],
    "selected_target": "RUL_norm"
}
```

이다.

`candidate_features`는 04에서 보존할 Feature 집합이며,
`selected_target`은 Learning Dataset에 포함할 Target이다.

## 5. Dataset 구성

### Learning

``` text
META_COLUMNS
+
candidate_features
+
selected_target
```

### Test

``` text
META_COLUMNS
+
candidate_features
```

Test에는 Target을 포함하지 않는다.

04는 전체 Engineered Dataset을 자동으로 Feature로 해석하지 않고 명시적
Whitelist를 사용한다.

## 6. Metadata 계약

공통 Metadata는 다음과 같다.

``` text
bearing
condition
rpm
load
file_idx
hour
minute
second
elapsed_sec
```

다음은 Metadata가 아니다.

``` text
life_ratio
RUL_sec
RUL_norm
failure_time
```

RUL 관련 변수는 Target 또는 Target 생성용 보조 정보다.

## 7. Dataset Validation

Validation은 Column Filtering 이후 수행한다.

검증 대상은 Candidate Feature와 Selected Target이다.

-   필수 Metadata 존재
-   Candidate Feature 존재
-   Selected Target 존재
-   Candidate Feature 중복 없음
-   Target/Feature 중복 없음
-   Numeric dtype
-   Bearing별 `elapsed_sec` 단조 증가
-   Inf 없음

### NaN 정책

현재 구조적으로 존재할 수 있는 Temperature Feature의 결측을 허용한다.

``` python
ALLOWED_NAN_PREFIXES = ("temp_",)
```

즉 `temp_`로 시작하는 Candidate Feature의 NaN은 허용한다.

다만:

-   Selected Target의 NaN은 허용하지 않는다.
-   `temp_`에 해당하지 않는 Feature의 NaN은 허용하지 않는다.
-   결측을 자동 제거하거나 Imputation하지 않는다.

허용된 Temperature 결측은 경고와 함께 현황을 출력한다.

## 8. Prefix Evaluation Scenario

Prefix Definition은 Learning 전용 산출물이다.

``` text
Model-ready Learning Dataset
             ↓
       Prefix Builder
             ↓
      prefix_definition
```

Prefix 계산에는 다음만 필요하다.

``` text
bearing
elapsed_sec
prefix ratio
```

다음 RUL 정보에는 의존하지 않는다.

``` text
failure_time
life_ratio
RUL_sec
RUL_norm
remaining_sec
```

현재 기본 설정은:

``` python
DEFAULT_PREFIX_START = 0.65
DEFAULT_PREFIX_END = 0.92
DEFAULT_PREFIX_STEP = 0.05
```

Bearing의 관측 행 수를 `n`, Prefix ratio를 `r`이라 하면:

``` text
end_idx = floor(r × (n - 1))
```

으로 종료 Index를 결정한다.

Prefix Definition Schema:

``` text
bearing
prefix_id
prefix_no
prefix_ratio
end_idx
end_time
```

## 9. Prefix Validation

다음을 검증한다.

1.  필수 컬럼 존재
2.  모든 Bearing 존재
3.  Bearing별 Prefix 개수 동일
4.  `prefix_ratio` 증가
5.  `end_idx` 증가
6.  `end_idx`가 Dataset 범위를 초과하지 않음
7.  Prefix Definition과 Learning Dataset의 Bearing 집합 일치

Prefix Definition은 05 Prefix Validation의 입력이다.

## 10. Utility 구조

``` text
dataset_utils.py
│
├── Model-ready Dataset
│   ├── build_model_ready_dataset()
│   └── validate_model_ready_dataset()
│
└── Prefix Evaluation Scenario
    ├── generate_prefix_ratios()
    ├── calculate_prefix_indices()
    ├── build_prefix_definition_for_bearing()
    ├── create_prefix_definition()
    └── validate_prefix_definition()
```

## 11. Notebook 역할

04 Notebook은 Orchestrator다.

``` text
Configuration
    ↓
Selection Manifest Load
    ↓
Engineered Learning/Test Load
    ↓
Candidate Dataset Construction
    ↓
Dataset Validation
    ↓
Learning Prefix Definition
    ↓
Prefix Validation
    ↓
Output Save
    ↓
Summary
```

계산 및 도메인 로직은 `dataset_utils.py`에 둔다.

## 12. Output 계약

``` text
outputs/04_dataset_builder/
├── model_ready_learning_dataset.csv
├── model_ready_test_dataset.csv
├── prefix_definition.csv
└── dataset_config.json
```

### model_ready_learning_dataset.csv

``` text
META_COLUMNS
+
candidate_features
+
selected_target
```

05에서 최종 Feature Selection을 수행한다.

### model_ready_test_dataset.csv

``` text
META_COLUMNS
+
candidate_features
```

### prefix_definition.csv

Learning Prefix Evaluation Scenario다.

### dataset_config.json

현재 Dataset 생성 조건을 기록한다.

``` json
{
    "candidate_features": [...],
    "selected_target": "RUL_norm",
    "nan_policy": {
        "allowed_feature_prefixes": ["temp_"]
    },
    "prefix_settings": {
        "start": 0.65,
        "end": 0.92,
        "step": 0.05
    }
}
```

## 13. 01\~06 데이터 흐름

``` text
01 Generate Features
        ↓
Base Feature Dataset
        ↓
02 Feature Transformation
   + RUL Target Generation
        ↓
Engineered Learning/Test
        ↓
03 EDA & Candidate Selection
        ↓
Candidate Features + Selected Target
        ↓
04 Dataset Builder
        ├── Model-ready Learning
        ├── Model-ready Test
        └── Prefix Definition
                ↓
05 Modeling & LOBO Validation
        ├── Final Feature Selection
        ├── LOBO Models
        ├── Prefix Validation
        └── Model Bundle
                ↓
06 Test Inference & Ensemble
```

Test는 03의 EDA와 Selection Decision을 다시 수행하지 않는다.

## 14. 책임 경계

  단계   Feature 관련 책임
  ------ -----------------------------------------
  03     모델링 후보 Feature 결정
  04     Candidate Feature를 Dataset에 보존·검증
  05     Candidate 중 최종 모델 Feature 결정

전체 단계는 다음과 같이 요약된다.

  -----------------------------------------------------------------------
  단계                    핵심 책임               주요 Output
  ----------------------- ----------------------- -----------------------
  01                      Sensor → Base Feature   Base Feature Dataset

  02                      Feature                 Engineered
                          Transformation + RUL    Learning/Test
                          Target Generation       

  03                      EDA + Candidate         Evidence + Selection
                          Feature/Target Decision Manifest

  04                      Candidate Dataset       Model-ready Dataset +
                          Construction +          Prefix Definition
                          Validation + Prefix     
                          Definition              

  05                      Final Feature           Model Bundle +
                          Selection + Modeling +  Validation Results
                          LOBO Validation         

  06                      Test Inference +        Final Test Prediction
                          Ensemble                
  -----------------------------------------------------------------------

## 15. 핵심 불변 조건

1.  04는 Feature Transformation을 수행하지 않는다.
2.  04는 RUL Target을 생성하지 않는다.
3.  04는 EDA를 수행하지 않는다.
4.  04는 최종 Feature Selection을 수행하지 않는다.
5.  04는 `candidate_features`를 명시적 Whitelist로 사용한다.
6.  Learning에는 `selected_target`을 포함한다.
7.  Test에는 Target을 포함하지 않는다.
8.  선택되지 않은 Feature Candidate는 Output에 포함하지 않는다.
9.  구조적으로 허용된 `temp_` Feature의 NaN은 허용한다.
10. Target의 NaN은 허용하지 않는다.
11. Inf는 허용하지 않는다.
12. Bearing별 시간 순서를 검증한다.
13. Prefix Definition은 Learning 전용이다.
14. Prefix Definition은 RUL 계산에 의존하지 않는다.
15. Scaling과 Sequence/Windowing은 수행하지 않는다.
16. Notebook은 Orchestrator 역할을 수행한다.
17. 05의 최종 Feature Selection과 04의 Candidate Dataset Construction을
    혼동하지 않는다.

## 16. 설계의 핵심

04의 본질은 다음과 같다.

> **03에서 결정된 모델링 후보 Feature와 Target을 실제 Dataset 계약으로
> 확정하고, Learning 평가를 위한 Prefix 조건을 정의하여 05가 모델링을
> 수행할 수 있는 입력을 제공하는 단계다.**

따라서:

``` text
01  무엇을 Feature로 만들 것인가
02  Feature를 어떻게 변환하고 Target을 어떻게 만들 것인가
03  어떤 Feature들을 모델링 후보로 삼을 것인가
04  그 후보와 Target을 어떤 Dataset 계약으로 전달할 것인가
05  후보 중 어떤 Feature 조합으로 실제 모델을 학습할 것인가
06  확정된 모델을 Test에 어떻게 적용하고 통합할 것인가
```

로 책임을 구분한다.