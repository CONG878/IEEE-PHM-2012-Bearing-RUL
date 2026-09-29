# 03_Exploratory Data Analysis and Feature Selection

---

## 1. 단계 개요

03단계는 **02단계에서 생성된 Engineered Learning Dataset을 분석하여 Feature와 Target을 선택하는 단계**이다.

02단계가 Base Feature Dataset으로부터 Feature Transformation을 수행하고 Learning 데이터에 RUL Target Candidate를 생성하는 단계라면, 03단계는 그 결과를 입력으로 받아 Feature의 품질과 특성, Target과의 관계 및 Feature 간 중복성을 분석하고, 분석 결과를 근거로 실제 모델링에 사용할 Feature와 Target을 확정한다.

전체 흐름은 다음과 같다.

```text
02 Engineered Learning Dataset
            ↓
     Generic Validation
            ↓
     Column Classification
            ↓
        EDA Analysis
            ↓
      EDA Evidence 저장
            ↓
      사용자 Selection
            ↓
 Selected Features / Target
            ↓
       04 Dataset Builder
```

03단계는 **EDA와 Feature Selection을 담당하며 Feature Transformation이나 RUL Target Generation을 수행하지 않는다.** 또한 Ranking이나 Correlation 결과를 이용하여 Feature를 자동으로 선택하지 않는다. 현재 Notebook도 Feature Transformation, Target Generation, Automatic Feature Selection, Automatic Target Selection, Scaling, Windowing 및 Dataset Builder를 수행하지 않는 것으로 명시되어 있다. 

---

## 2. 입력과 분석 대상

03단계의 입력은 다음 경로의 Engineered Learning Dataset이다.

```text
outputs/02_feature_transformation/
└─ learning_engineered_features.csv
```

이 데이터셋은 02단계에서 Feature Transformation과 Learning용 RUL Target Generation을 완료한 결과이다. 02단계에서는 Raw Feature로부터 Rectified Feature와 RDI를 생성하고, Learning 데이터에 `life_ratio`, `RUL_sec`, `RUL_norm` 등의 Target Candidate를 생성한다.

03단계에서는 입력 컬럼을 다음과 같이 구분한다.

```text
Metadata
Target Candidates
Feature Candidates
Auxiliary Columns
```

`failure_time`은 RUL 계산을 위해 존재하는 보조 계산 컬럼이며 Metadata나 Target Candidate 또는 Feature Candidate에는 포함하지 않는다. 

Feature Candidate에는 02단계에서 생성된 `Rectified_*`와 `RDI_*`를 포함하여 이후 분석 가능한 Engineered Feature가 포함된다. 따라서 03단계에서 Feature를 다시 생성하거나 변환하지 않는다.

---

## 3. 데이터 무결성 및 품질 확인

EDA에 앞서 입력 데이터의 기본적인 무결성을 확인한다.

주요 확인 대상은 다음과 같다.

* 중복 행
* Infinite 값
* Missing Value
* Bearing별 데이터 구성
* Feature Candidate의 자료형 및 분석 가능 여부

특히 Missing Value는 화면에 결측이 실제로 존재하는 Feature만 표시하도록 구성하였다. 반면 `data_quality.csv`에는 전체 컬럼에 대한 데이터 품질 정보를 보존한다.

이는 **분석 결과의 저장 범위와 Notebook 화면의 표시 범위를 분리한 것**이다. 화면에 모든 정보를 반복하여 출력하는 대신, 전체 Evidence는 파일로 보존하고 화면에서는 분석에 필요한 결과를 중심으로 확인한다.

---

## 4. Feature EDA

Feature EDA에서는 Feature의 기본적인 분포와 운전조건에 따른 변화, Bearing별 열화 추세를 확인한다.

### 4.1 Feature Distribution

Feature의 값 분포를 확인하여 각 Feature의 기본적인 데이터 특성을 파악한다.

현재 시각화에서는 모든 Feature Candidate를 무차별적으로 표시하지 않고 **비누적 Feature**를 대상으로 한다.

```text
Raw Feature
Rectified Feature
```

이 구분은 Feature Selection을 의미하는 것이 아니라 **EDA 시각화의 목적에 따른 표시 대상의 분리**이다.

---

### 4.2 Operating Condition

운전조건에 따른 Feature의 변화를 확인한다.

이 역시 Raw 및 Rectified와 같은 비누적 Feature를 중심으로 표시하여 운전조건에 따른 원자료 특성의 차이를 확인한다.

---

### 4.3 Bearing Degradation Trend

Bearing별 Feature의 열화 진행 추세를 확인한다.

여기서는 Bearing마다 전체 수명이 다르다는 점을 고려하여 가로축을 `life_ratio`로 통일하였다.

```text
life_ratio = 0
    초기
      ↓
life_ratio = 1
    고장
```

따라서 `elapsed_sec`와 `life_ratio`를 모두 출력하지 않고, 03단계의 비교 목적에는 `life_ratio`를 사용한다.

현재 `plot_bearing_trend()`는 `x_col`을 통해 축을 명시적으로 지정할 수 있으며 기본 축도 `life_ratio`로 구성되어 있다. 

이 시각화에서는 비누적 Feature를 사용한다. 누적 손상 정보를 표현하는 RDI는 별도의 Feature–Target 관계 분석에서 다룬다.

---

## 5. Feature Quality Metrics

Feature Candidate 각각에 대해 다음 지표를 계산한다.

* **Monotonicity**
* **Trendability**
* **Prognosability**
* **Condition Stability (CV)**
* **Composite Ranking Score**

`rank_features()`는 이 지표들을 계산하고 정규화하여 Composite Score를 산출한다. 현재 기본 가중치는 다음과 같다.

```text
Monotonicity  : 0.35
Trendability  : 0.35
Prognosability: 0.20
CV            : 0.10
```



각 Metric은 Feature의 열화 특성과 운전조건 안정성 등을 서로 다른 관점에서 평가하기 위한 Evidence이다.

### Ranking의 의미

Composite Ranking Score는 **Feature Selection의 자동 결정값이 아니다.**

즉,

```text
Feature Metrics
      ↓
Ranking Evidence
      ↓
사용자의 판단
      ↓
Selected Features
```

의 구조를 유지한다.

현재 Notebook에서도 Ranking을 Evidence로 규정하고 있으며 Score가 높은 Feature를 자동으로 Selection하지 않는다.

이는 특정 단일 지표가 Feature의 최종적인 유효성을 완전히 결정한다고 가정하지 않기 위한 것이다.

---

## 6. Feature Redundancy Analysis

Feature 간 상관 및 중복성을 분석한다.

목적은 높은 유사성을 보이는 Feature를 식별하여 Selection 과정에서 함께 고려할 수 있도록 **중복성 Evidence를 제공하는 것**이다.

중요한 점은 높은 상관관계가 발견되었다고 해서 어느 한 Feature를 자동으로 제거하지 않는다는 것이다.

```text
Feature Correlation
        ↓
Redundancy Evidence
        ↓
사용자 판단
```

의 구조를 유지한다.

따라서 Redundancy Analysis는 Selection Algorithm이 아니라 Selection을 위한 분석 자료이다.

---

## 7. Feature–Target Correlation

03단계에서는 기존 별도 RUL 검증 과정에서 수행하던 Feature–RUL 관계 분석을 EDA Evidence의 일부로 통합하였다.

Global 수준에서 각 Feature Candidate와 Target Candidate의 조합에 대해 다음 상관계수를 계산한다.

* Pearson
* Spearman
* Kendall

`calculate_rul_correlation()`은 Feature와 Target의 모든 조합에 대한 결과를 반환하며, 상관계수의 부호도 그대로 보존한다. 

또한 Bearing별로 동일한 관계를 계산하여 특정 Bearing에만 나타나는 관계인지, 여러 Bearing에서 일관되게 나타나는 관계인지 확인할 수 있도록 하였다.

이 결과는:

```text
rul_correlation.csv
rul_correlation_by_bearing.csv
```

으로 저장한다.

---

## 8. Feature–Target Visualization

Feature–Target 관계의 시각화는 Correlation Evidence와 별도의 보조 분석으로 구성하였다.

기존처럼 하나의 그래프에서:

```text
Feature ↔ 시간
Feature ↔ Target
```

을 동시에 표현하지 않고 책임을 분리하였다.

```text
plot_bearing_trend()
    → Feature의 열화 추세

plot_feature_vs_target()
    → Feature ↔ Target 관계
```

현재 `plot_feature_vs_target()`은 하나의 명시적인 Feature–Target 조합에 대한 산점도를 생성하고 Bearing별 데이터를 구분하여 표시한다. 

따라서 모든 Feature에 대해 상세 그래프를 자동 생성하지 않고, `RUL_PLOT_FEATURES`에 지정한 Feature만 시각화한다.

이 구조를 통해 **Evidence 계산은 전체 Feature Candidate에 대해 수행하면서도 Notebook의 시각적 출력은 필요한 분석으로 제한**할 수 있다.

---

## 9. Evidence의 생성과 보존

03단계의 중요한 설계 원칙은 **Evidence 생성과 Selection Decision의 분리**이다.

현재 생성되는 주요 Evidence는 다음과 같다.

```text
outputs/03_eda_selection/
└─ eda_evidence/
   ├─ feature_metrics.csv
   ├─ feature_redundancy.csv
   ├─ rul_correlation.csv
   ├─ rul_correlation_by_bearing.csv
   ├─ data_quality.csv
   ├─ bearing_profile.csv
   └─ figures/
```

각 파일은 03단계의 분석 결과를 이후에도 다시 확인할 수 있도록 보존한다. 현재 산출물 경로 정책 역시 EDA Evidence와 Selection Configuration을 03단계 산출물로 구분하고 있다. 

특히 Notebook 화면에서는 분석 목적에 따라 표시 Feature를 제한하더라도 **Evidence 파일에는 전체 분석 결과를 저장**한다.

따라서:

```text
화면 출력
    ≠
Evidence 전체
```

라는 원칙을 적용하였다.

---

## 10. Feature 및 Target Selection

EDA가 완료된 후 실제 모델링에 사용할 Feature와 Target을 명시적으로 결정한다.

Selection은 다음의 Whitelist 방식으로 구현하였다.

```text
All Feature Candidates
        ↓
    EDA Evidence
        ↓
   User Decision
        ↓
 CANDIDATE_FEATURES
```

Target도 동일하게:

```text
Target Candidates
        ↓
   User Decision
        ↓
 SELECTED_TARGET
```

으로 결정한다.

따라서 Feature Ranking, Redundancy, RUL Correlation 등의 결과는 Selection을 위한 판단 근거이며 **자동 Selection 규칙으로 사용되지 않는다.** 이 원칙은 현재 Notebook과 설계 명세 모두에서 명시되어 있다.

### 현재 선택 결과

현재 구현에서 확정된 Feature는 다음과 같다.

```text
RDI_rms
Rectified_rms

RDI_cf
Rectified_cf

RDI_p2p
Rectified_p2p

RDI_kurt
Rectified_kurt

temp_mean
temp_max

spectral_spread
spectral_crest
principal_energy_ratio
```

Selected Target은:

```text
RUL_norm
```

이다. 

이 선택 결과는:

```text
outputs/03_eda_selection/selection_config.json
```

에 저장한다.

---

## 11. 03단계와 Test 데이터의 관계

03단계의 EDA와 Selection은 **Learning 데이터에 대해서만 수행한다.**

Test 데이터에는 다음을 수행하지 않는다.

```text
Test
 ├─ EDA                  X
 └─ Selection Decision   X
```

대신 Learning에서 확정된 결과를 이후 단계에서 Test에 적용한다.

즉 Test가 03단계의 분석 과정을 다시 수행하는 것이 아니라:

```text
Learning
   ↓
03 EDA
   ↓
Selection
   ↓
Selected Features / Target
   │
   ├────────────→ Learning
   │
   └────────────→ Test
```

의 구조이다.

이 원칙은 04 Dataset Builder가 03에서 결정된 Feature/Target Selection을 적용하여 모델 입력 Dataset을 구성하도록 하는 단계 경계와 연결된다. 04의 `build_model_ready_dataset()` 역시 03에서 선택된 Feature와 Target을 입력으로 받도록 구현되어 있다. 

---

## 12. 구현 결과와 단계적 의의

03단계의 최종 구현은 단순한 EDA Notebook을 넘어 **모델링 이전의 Feature/Target 의사결정을 독립된 단계로 분리한 것**에 의미가 있다.

구조적으로는 다음과 같다.

```text
01 Generate Features
        ↓
Base Feature Dataset
        ↓
02 Feature Transformation
        ↓
Engineered Learning Dataset
        ↓
03 EDA & Feature Selection
        │
        ├─ Data Quality
        ├─ Feature EDA
        ├─ Feature Quality
        ├─ Redundancy
        ├─ RUL Correlation
        └─ User Selection
        ↓
Selected Feature / Target
        ↓
04 Dataset Builder
```

이 단계에서 Feature를 새로 만들거나 모델을 학습하지 않는다. 대신 **02단계에서 생성된 후보를 분석하고, 분석 결과를 Evidence로 보존하며, 그 Evidence를 바탕으로 실제 모델링 입력을 결정하는 역할**에 집중한다.

결과적으로 03단계의 출력은 단순히 여러 EDA 그래프가 아니라 다음 두 종류의 산출물로 구분된다.

```text
EDA Evidence
    └─ "무엇을 관찰했는가?"

Selection Configuration
    └─ "그 관찰을 바탕으로 무엇을 사용하기로 했는가?"
```

이 분리는 이후 04 Dataset Builder가 **분석 과정 자체가 아니라 확정된 Selection 결과를 입력 계약으로 사용**할 수 있도록 한다.

---

## 13. 최종 단계 경계

03단계의 책임을 최종적으로 정리하면 다음과 같다.

| 구분                      | 03단계                        |
| ----------------------- | --------------------------- |
| 입력                      | Engineered Learning Dataset |
| Feature Transformation  | 수행하지 않음                     |
| RUL Target Generation   | 수행하지 않음                     |
| Data Quality EDA        | 수행                          |
| Feature EDA             | 수행                          |
| Feature Quality Metrics | 수행                          |
| Feature Redundancy      | 수행                          |
| RUL Correlation         | 수행                          |
| Feature Ranking         | Evidence로 수행                |
| 자동 Feature Selection    | 수행하지 않음                     |
| 자동 Target Selection     | 수행하지 않음                     |
| Feature Selection       | 사용자 명시적 Whitelist           |
| Target Selection        | 사용자 명시적 결정                  |
| EDA Evidence 저장         | 수행                          |
| Test EDA                | 수행하지 않음                     |
| 다음 단계                   | 04 Dataset Builder          |

이것으로 03단계는 **"Engineered Dataset → Evidence → 명시적 Selection"이라는 하나의 독립된 파이프라인 단계를 완성**한다. 설계상으로도 03은 정확히 EDA Analysis와 Feature Selection을 담당하고, Test에서는 분석과 Selection Decision을 우회하도록 정의되어 있다.