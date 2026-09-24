# 02_Feature_Transformation_and_Target_Generation

---

## 1. 단계 개요

02단계는 01단계에서 생성된 **Base Feature Dataset**을 입력으로 받아, 이후 EDA와 Feature Selection에 사용할 **Engineered Dataset**을 생성하는 단계이다.

이 단계의 핵심 책임은 두 가지이다.

1. 원본 Feature에 도메인 기반 Feature Transformation을 적용하여 Derived Feature를 생성한다.
2. Learning Dataset에 대해 Bearing별 수명을 기반으로 RUL 관련 Target Candidate를 생성한다.

따라서 02단계의 출력은 단순히 모델 입력을 만드는 데이터셋이 아니라, **03단계에서 Feature와 Target을 평가하고 선택할 수 있도록 후보군을 완성한 Dataset**이다.

전체 흐름은 다음과 같다.

```text
01 Generate Features
        │
        ▼
Base Feature Dataset
        │
        ├───────────────┐
        ▼               ▼
Feature Transformation  RUL Target Generation
        │               │
        └───────┬───────┘
                ▼
      Engineered Dataset
                │
                ▼
        03 EDA & Selection
```

02단계에서 생성된 `Rectified_*`, `RDI_*`는 최종 모델에 반드시 사용되는 Feature가 아니라 **Feature Candidate**이다. 마찬가지로 `life_ratio`, `RUL_sec`, `RUL_norm`도 최종 Target이 아니라 **Target Candidate**이다.

---

## 2. 입력과 출력

### 2.1 입력

02단계는 01단계의 Base Feature Dataset을 Learning과 Test 각각 입력으로 사용한다.

```text
Learning Base Feature Dataset
Test Base Feature Dataset
```

Notebook은 입력 데이터에 대해 먼저 공통적인 Schema와 시간 순서를 확인한 뒤 Transformation을 수행한다.

---

### 2.2 출력

Learning과 Test의 출력 역할은 다르다.

### Learning

```text
Metadata
+ Raw Feature Candidates
+ Derived Feature Candidates
+ Target Candidates
```

### Test

```text
Metadata
+ Raw Feature Candidates
+ Derived Feature Candidates
```

Test에는 실제 고장 시점이 존재하지 않으므로 Learning과 같은 방식으로 RUL Target을 생성하지 않는다.

현재 Notebook도 이 원칙을 그대로 구현하여 `generate_rul_targets()`는 Learning에만 호출하고, Test에는 호출하지 않는다. 또한 최종 검증에서도 Learning에는 Target Candidate가 존재하고 Test에는 존재하지 않는지를 확인한다.

---

# 3. 데이터 의미 분류

02단계 이후의 모든 컬럼은 세 가지 의미적 범주로 구분된다.

## 3.1 Metadata

현재 Metadata는 다음과 같다.

```text
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

이는 Bearing 식별, 운전 조건 및 시간 정보를 나타내며 Feature나 Target의 선택 대상이 아니다. `META_COLUMNS`는 `bearing_utils.py`가 관리한다.

---

## 3.2 Target Candidates

02단계에서 Learning에 생성되는 Target Candidate는 다음 세 가지다.

```text
life_ratio
RUL_sec
RUL_norm
```

이들은 모두 RUL과 수명 위치를 표현하지만, **후속 단계에서 어떤 것을 실제 모델 Target으로 사용할지는 03단계의 Selection에서 결정한다.**

따라서:

```text
Target Candidate ≠ Selected Target
```

이라는 구분을 유지한다.

---

## 3.3 Feature Candidates

Metadata와 Target Candidate를 제외한 나머지 컬럼은 Feature Candidate이다.

여기에는 01단계에서 생성된 Raw Feature뿐 아니라 02단계에서 새롭게 생성되는 다음 Derived Feature도 포함된다.

```text
Rectified_*
RDI_*
```

특히 `Rectified_*`는 RDI를 계산하기 위한 중간 계산 과정에서 만들어지지만, 최종 출력에서 버리지 않는다. 이 Feature 자체가 모델에 유용할 가능성이 있으므로 03단계의 EDA 및 Feature Selection 대상이 된다.

---

# 4. Feature Transformation

## 4.1 Transformation 대상

현재 Transformation 대상의 기본 목록은 `feature_utils.RDI_FEATURES`에 정의되어 있다.

```python
RDI_FEATURES = ["rms", "p2p", "cf", "kurt"]
```

이 목록은 **Feature Selection 결과가 아니다.**

즉 다음과 같이 구분된다.

```text
RDI_FEATURES
    ↓
Transformation을 적용할 원본 Feature

Selected Features
    ↓
03단계에서 실제 모델에 사용할 Feature
```

`build_damage_features()`는 `features` 인자를 별도로 받을 수 있으며, `RDI_FEATURES`는 그 인자의 기본값이다. 이를 통해 Transformation 대상과 Feature Selection을 분리한다.

---

## 4.2 Transformation 구조

하나의 원본 Feature에 대한 Transformation은 다음 순서로 수행된다.

```text
Raw Feature
     │
     ▼
Baseline 설정
     │
     ▼
Threshold 계산
     │
     ▼
Shock
     │
     ▼
Rectified Feature
     │
     ├───────────────┐
     ▼               │
Cumulative           │
Rectified            │
     │               │
     ▼               │
RDI ◄───────────────┘
```

이 계산은 `feature_utils.py`의 다음 Primitive를 통해 구성된다.

```text
calculate_shock()
calculate_cumulative_shock()
calculate_rectified_feature()
calculate_relative_damage()
```

그리고 하나의 Feature에 대한 전체 계산은:

```text
build_damage_feature()
```

가 담당하며, 여러 Feature에 대해서는:

```text
build_damage_features()
```

가 Bearing별로 반복 적용한다.

---

# 5. Shock 및 Rectified Feature

각 Feature에 대해 먼저 초기 구간을 Baseline으로 설정한다.

현재 기본 설정은:

```text
baseline_ratio = 0.05
threshold_k = 0.0
```

이다.

Baseline으로부터 평균과 표준편차를 계산하고:

```text
threshold
    = baseline_mean
      + threshold_k × baseline_std
```

형태로 Threshold를 결정한다.

이후 원본 Feature에서 Threshold를 초과하는 부분만 Shock으로 취급한다.

```text
Shock = max(Feature - threshold, 0)
```

Shock은 다시 Baseline 표준편차를 이용해 정규화하여 Rectified Feature를 만든다.

```text
Rectified
    = Shock / baseline_std
```

수치적으로 Baseline 표준편차가 0이 되는 경우를 고려하여 작은 양의 값으로 하한을 두어 0으로 나누는 문제를 방지한다. 현재 구현은 `eps = 1e-12`를 사용한다.

---

# 6. Relative Damage Index

Rectified Feature는 시간에 따른 누적 손상량을 계산하는 데 사용된다.

```text
Cumulative Rectified
    = cumulative sum(Rectified)
```

그리고 Baseline 구간의 누적 손상을 기준으로 정규화하여 Relative Damage Index를 계산한다.

```text
RDI
    = Cumulative Rectified
      / Baseline Damage
```

따라서 하나의 원본 Feature `x`에 대해 최종적으로 다음 두 Feature Candidate가 생성된다.

```text
Rectified_x
RDI_x
```

Cumulative Rectified는 RDI 계산에 필요한 내부 중간값이므로 최종 Dataset에는 별도 Feature로 추가하지 않는다. 반면 Rectified 자체는 최종 Feature Candidate로 보존한다.

---

# 7. Bearing별 독립 Transformation

Transformation은 전체 Dataset을 하나의 연속적인 시계열로 처리하지 않고 **각 Bearing별로 독립적으로 수행한다.**

`build_damage_features()`는 다음과 같이 동작한다.

```text
Engineered Dataset
        │
        ├── Bearing A
        │      ├─ Feature 1 → Rectified / RDI
        │      ├─ Feature 2 → Rectified / RDI
        │      └─ ...
        │
        ├── Bearing B
        │      ├─ Feature 1 → Rectified / RDI
        │      ├─ Feature 2 → Rectified / RDI
        │      └─ ...
        │
        └── ...
```

따라서 각 Bearing의 초기 상태를 해당 Bearing의 Baseline으로 사용한다.

Learning과 Test에도 동일한 Transformation 함수를 독립적으로 적용하며, Notebook에서 `dataset_type`에 따라 다른 Transformation 경로를 선택하지 않는다.

---

# 8. RUL Target Generation

## 8.1 책임 분리

RUL Target Generation은 Feature Transformation과 별개의 책임이다.

이를 위해 Bearing 도메인에 종속된 계산은 `bearing_utils.py`에 둔다.

주요 구성은 다음과 같다.

```text
calculate_failure_time()
calculate_rul_sec()
calculate_normalized_targets()
generate_rul_targets()
```

`generate_rul_targets()`는 이 Primitive들을 조합하는 상위 함수다.

---

## 8.2 Failure Time

각 Bearing의 완전한 수명은 해당 Bearing에서 관측된 최대 `elapsed_sec`로 정의한다.

```text
failure_time
    = max(elapsed_sec | bearing)
```

이 값은 RUL 계산에 필요한 **보조 계산값**이다.

따라서:

```text
failure_time
```

은 Metadata도 아니고 Target Candidate도 아니며 Feature Candidate도 아니다.

03단계에서도 이 원칙에 따라 별도의 Candidate로 취급하지 않는다.

---

## 8.3 RUL_sec

Remaining Useful Life는 다음과 같이 계산한다.

```text
RUL_sec
    = failure_time - elapsed_sec
```

즉 현재 관측 시점에서 해당 Bearing의 관측 종료 시점까지 남은 시간을 초 단위로 표현한다. 현재 구현은 이 계산을 `calculate_rul_sec()`가 담당한다.

---

## 8.4 정규화된 Target

두 가지 정규화 Target을 함께 생성한다.

```text
life_ratio
    = elapsed_sec / failure_time

RUL_norm
    = RUL_sec / failure_time
```

이 두 계산은 동일한 Bearing별 `failure_time`을 기준으로 수행한다.

수치적으로 `failure_time`이 0인 비정상적인 경우에는 `machine epsilon`을 사용하여 0으로 나누는 문제를 방지한다.

---

# 9. `failure_time`과 Target Candidate의 관계

02단계에서 생성되는 컬럼을 역할별로 구분하면 다음과 같다.

| 컬럼             | 역할                       | 최종 Candidate 여부 |
| -------------- | ------------------------ | --------------- |
| `failure_time` | RUL 계산용 보조값              | 아니오             |
| `life_ratio`   | 수명 위치 Target Candidate   | 예               |
| `RUL_sec`      | 실제 RUL Target Candidate  | 예               |
| `RUL_norm`     | 정규화 RUL Target Candidate | 예               |

이 구분을 통해 `failure_time`이 03단계의 Feature 또는 Target Selection에 잘못 포함되는 것을 방지한다.

---

# 10. Learning과 Test의 차이

02단계에서 Learning과 Test는 가능한 한 동일한 Transformation 경로를 사용한다.

```text
Learning
    │
    ├─ Generic Validation
    ├─ Feature Transformation
    └─ RUL Target Generation

Test
    │
    ├─ Generic Validation
    └─ Feature Transformation
```

핵심 차이는 **Target Generation의 존재 여부**다.

Learning에서는 각 Bearing의 전체 수명을 알 수 있으므로 `failure_time`을 계산할 수 있고 RUL Target Candidate를 생성할 수 있다.

반면 Test에서는 실제 고장 시점이 알려져 있지 않으므로 동일한 Target Generation을 수행하지 않는다.

이러한 구분은 Notebook의 구현에서도 명시적으로 유지된다.

---

# 11. Notebook의 역할

02 Notebook 자체에는 도메인 계산식을 중복 구현하지 않는다.

Notebook은 다음 Utility를 조립한다.

```text
project_utils
    ├─ split_columns()
    ├─ sort_by_time()
    ├─ assert_required_columns()
    └─ assert_monotonic_order()

bearing_utils
    ├─ META_COLUMNS
    ├─ TARGET_CANDIDATES
    └─ generate_rul_targets()

feature_utils
    ├─ RDI_FEATURES
    └─ build_damage_features()
```

즉 Notebook의 책임은:

```text
입력
 ↓
Validation
 ↓
Utility 호출
 ↓
Candidate 확인
 ↓
출력 저장
```

으로 제한된다.

이 구조는 계산 및 도메인 로직과 실행 순서를 분리한다.

---

# 12. Validation

02 Notebook에서는 Transformation 및 Target Generation 이전에 기본적인 Dataset 검증을 수행한다.

주요 검증은 다음과 같다.

### Schema

필수 Metadata가 존재하는지 확인한다.

```text
META_COLUMNS
```

### 시간 순서

Bearing별로 `elapsed_sec`가 시간 순서에 맞게 정렬되었는지 확인한다.

이를 위해:

```text
sort_by_time()
assert_monotonic_order()
```

를 사용한다.

### 기본 데이터 무결성

중복 행과 Infinite 값의 존재 여부도 확인한다.

이는 Feature Selection이나 EDA가 아니라 **02단계 입력 데이터의 기본적인 실행 가능성 확인**을 위한 것이다. 현재 Notebook은 이 검증을 Transformation 전에 수행한다.

---

# 13. Candidate 분류

Transformation과 Target Generation이 완료된 뒤 Notebook에서는 결과 컬럼을 의미적으로 분류한다.

Learning에서는:

```text
Metadata
Target Candidates
Feature Candidates
```

를 확인하고, Test에서는:

```text
Metadata
Feature Candidates
```

를 확인한다.

이 단계는 Feature Selection이나 Target Selection을 수행하는 것이 아니다.

특히 Feature Candidate의 자동 Ranking이나 자동 제거는 02단계의 책임이 아니다. 03단계가 모든 Candidate에 대한 EDA Evidence를 생성하고, 그 Evidence를 바탕으로 사용자가 최종 선택한다.

---

# 14. 03단계와의 인터페이스

02단계의 최종 출력은 03단계의 입력이 된다.

```text
02 Engineered Learning Dataset
        │
        ▼
03 EDA & Feature Selection
```

03단계에서는:

* Raw Feature
* `Rectified_*`
* `RDI_*`

를 모두 Feature Candidate로 평가한다.

그리고:

```text
life_ratio
RUL_sec
RUL_norm
```

을 Target Candidate로 평가한다.

이후 사용자가 선택한 Feature와 Target만 04 Dataset Builder로 전달된다.

따라서 02단계에서는 **무엇이 최종 모델에 사용될 것인지 결정하지 않는다.**

---

# 15. 현재 구현 구조

현재 02단계의 핵심 모듈 구조는 다음과 같다.

```text
project_utils.py
    └─ 범용 DataFrame / Validation

bearing_utils.py
    ├─ META_COLUMNS
    ├─ TARGET_CANDIDATES
    ├─ calculate_failure_time()
    ├─ calculate_rul_sec()
    ├─ calculate_normalized_targets()
    └─ generate_rul_targets()

feature_utils.py
    ├─ RDI_FEATURES
    ├─ calculate_shock()
    ├─ calculate_cumulative_shock()
    ├─ calculate_rectified_feature()
    ├─ calculate_relative_damage()
    ├─ build_damage_feature()
    └─ build_damage_features()

02_Feature_Transformation.ipynb
    └─ 위 Utility를 순서에 따라 조립
```

각 모듈은 자신의 책임을 벗어난 작업을 수행하지 않는다.

---

# 16. 단계 완료 상태

현재 02단계의 구현은 다음 계약을 충족하는 형태로 정리되어 있다.

### 입력

```text
Base Feature Dataset
```

### Learning 처리

```text
Generic Validation
        ↓
Feature Transformation
        ↓
RUL Target Generation
        ↓
Candidate Classification
        ↓
Engineered Learning Dataset
```

### Test 처리

```text
Generic Validation
        ↓
Feature Transformation
        ↓
Candidate Classification
        ↓
Engineered Test Dataset
```

### 생성되는 Derived Feature

```text
Rectified_<feature>
RDI_<feature>
```

### 생성되는 Learning Target Candidate

```text
life_ratio
RUL_sec
RUL_norm
```

### 생성되지만 Selection 대상이 아닌 보조값

```text
failure_time
```

### 02단계에서 수행하지 않는 작업

```text
EDA
Feature Ranking
Feature Selection
Target Selection
Scaling
Sequence / Windowing
Dataset Construction
Model Training
```

이로써 02단계는 **Base Feature Dataset을 이후 분석과 선택이 가능한 Engineered Dataset으로 변환하는 전처리 단계**로 역할이 명확하게 정리되었다.