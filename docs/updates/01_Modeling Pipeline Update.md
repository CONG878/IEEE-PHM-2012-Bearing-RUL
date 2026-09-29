# 01_Modeling Update

## 1. 문서 목적

본 문서는 최초 `05_Modeling.ipynb`에서 현재 구현에 이르기까지 진행된 Stage 05의 변경 사항을 정리한다.

변경 사항은 Feature Selection 책임의 구체화, 모델링 조건의 변경, 검증 및 분석 기능의 확장을 포함한다. 또한 이러한 변경이 Stage 06의 산출물 및 인터페이스에 미치는 영향을 함께 확인한다.

---

## 2. 변경 요약

| 구분                           | 최초 버전                                                    | 현재 버전                                                                                | 변경 성격                           |
| ---------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------- |
| Feature 해석                   | Dataset의 Metadata와 Target을 제외한 나머지를 자동으로 Feature로 사용     | 03에서 전달된 `candidate_features`를 기준으로 05에서 최종 Feature를 명시적으로 선택                        | Feature Selection 책임 및 입력 해석 변경 |
| Feature 목록                   | `split_columns()`로 자동 추출                                 | `selection_config.json`의 `candidate_features` 검증 후 `SELECTED_FEATURE_COLUMNS`로 최종 확정 | 명시적 Whitelist                   |
| Target                       | `RUL_norm`                                               | `RUL_norm`                                                                           | 유지                              |
| TrainingRange                | `0.20 ~ 0.95`                                            | `0.30 ~ 1.00`                                                                        | 실험 조건 변경                        |
| RandomForest                 | `n_estimators=250`, `max_depth=12`, `min_samples_leaf=2` | `n_estimators=100`, `max_depth=10`, `min_samples_leaf=7`                             | 실험 조건 변경                        |
| 하이퍼파라미터 탐색                   | 초기 설정                                                    | 수동 실험을 통해 조건 갱신                                                                      | 파이프라인 외 실험 과정                   |
| LOBO Training                | LOBO 모델 생성                                               | 동일                                                                                   | 유지                              |
| Prefix Validation            | Prefix별 단일 샘플 예측 및 평가                                    | 동일                                                                                   | 유지                              |
| Full Validation Prediction   | 없음                                                       | Held-out Bearing의 전체 수명 예측 추가                                                        | 검증·분석 기능 추가                     |
| RUL Trajectory Visualization | 없음                                                       | Bearing별 전체 RUL 궤적 시각화 추가                                                            | 분석·시각화 기능 추가                    |
| Prefix Marker                | 없음                                                       | Prefix 위치를 전체 수명 그래프에 표시                                                             | 시각화 기능 보강                       |
| 그래프 배치                       | 해당 없음                                                    | 2열 subplot                                                                           | 시각화 편의성 개선                      |
| Feature Importance           | 없음                                                       | LOBO 모델별 Feature Importance 및 요약 출력                                                  | 해석·분석 기능 추가                     |
| 주요 산출물                       | Model Bundle, Prefix Validation 결과 및 평가 결과               | 동일한 공식 산출물 + Notebook 내 추가 분석 결과                                                     | 공식 인터페이스 유지                     |

---

## 3. Feature Selection 구조 변경

최초 버전에서는 Model-ready Dataset의 컬럼을 `META_COLUMNS`와 `RUL_norm`을 제외한 뒤 자동으로 Feature로 분리하였다.

```text
model_ready_learning_dataset
        ↓
META_COLUMNS + RUL_norm 제외
        ↓
전체 Feature Columns
        ↓
LOBO Training
```

따라서 05에서는 별도의 Feature Selection을 수행하지 않고 Dataset에 존재하는 모든 Feature가 모델 입력으로 사용되었다.

현재는 03에서 작성된 Selection Manifest의 `candidate_features`를 기준으로 입력 후보를 확인한 뒤, 05에서 최종 Feature를 명시적으로 선택한다.

```text
03 EDA & Selection
        ↓
candidate_features
        ↓
04 Model-ready Dataset
        ↓
05 Final Feature Selection
        ↓
Selected Feature Columns
        ↓
LOBO Training
```

최종 선택은 Notebook의 `SELECTED_FEATURE_COLUMNS`에 명시된 Whitelist를 사용하며, 현재 선택된 Feature는 다음과 같다.

```text
RDI_rms
RDI_cf
Rectified_cf
Rectified_kurt
```

이에 따라 05의 모델 입력 Feature는 Dataset의 전체 Feature 집합에 의해 암묵적으로 결정되지 않고, 03에서 전달된 `candidate_features`에 포함되는지를 확인한 뒤 최종 선택 목록으로 확정된다.

Target은 기존과 동일하게 `RUL_norm`을 유지한다.

이 변경으로 03단계에서 결정된 `candidate_features`와 05단계의 실제 모델 입력 사이의 연결을 명시적으로 관리할 수 있게 되었다.

---

## 4. 모델링 조건 변경

### 4.1 TrainingRange

TrainingRange는 최초 `0.20 ~ 0.95`에서 현재 `0.30 ~ 1.00`으로 변경되었다.

각 LOBO fold에서는 Held-out Bearing을 제외한 학습 Bearing에 동일한 TrainingRange 정책을 적용한다.

이는 모델 학습에 사용하는 구간을 조정한 실험 조건의 변경이며, TrainingRange의 개념이나 적용 방식 자체가 변경된 것은 아니다.

### 4.2 RandomForest 설정

최초 설정은 다음과 같다.

```text
algorithm         = RandomForest
n_estimators      = 250
max_depth         = 12
min_samples_leaf  = 2
random_state      = 42
n_jobs             = -1
```

현재 설정은 다음과 같다.

```text
algorithm         = RandomForest
n_estimators      = 100
max_depth         = 10
min_samples_leaf  = 7
random_state      = 42
n_jobs             = -1
```

현재 설정은 별도의 자동화된 Hyperparameter Search 인터페이스를 추가한 결과가 아니다. 여러 조건을 수동으로 실험하여 모델링 조건을 갱신한 것이다.

따라서 최종 파이프라인은 최초와 마찬가지로 하나의 확정된 `ModelingConfig`를 사용하여 LOBO 모델을 생성한다.

---

## 5. 검증 및 분석 기능 확장

### 5.1 LOBO 및 공식 Validation 구조

LOBO의 기본 구조는 변경하지 않았다.

각 Bearing을 한 번씩 Held-out으로 제외하여 모델을 생성하고, 해당 Bearing은 자신의 Held-out 모델을 사용하여 검증한다.

```text
Bearing A held-out → Model A
Bearing B held-out → Model B
...
```

Prefix Definition은 공식 Validation에서 평가할 validation row를 결정한다. 선택된 row의 예측은 해당 Bearing을 Held-out한 LOBO 모델에서 생성한다.

예측 Target은 계속 `RUL_norm`을 사용하고, 공식 평가는 이를 `elapsed_sec`와 함께 `RUL_sec`로 복원한 뒤 수행한다.

따라서 기존 Prefix Validation은 Stage 05의 공식 평가 구조로 그대로 유지된다. Stage 05에서는 LOBO 모델 간 앙상블을 수행하지 않는다.

### 5.2 Full Validation Prediction 추가

최초 버전에서는 공식 Prefix Validation만 수행했으나, 현재는 각 Held-out Bearing의 전체 수명에 대해 해당 LOBO 모델의 예측을 별도로 생성한다.

```text
LOBO Model
    ↓
Held-out Bearing 전체 수명
    ↓
Full Validation Prediction
    ↓
RUL_sec Reconstruction
    ↓
Trajectory Visualization
```

Full Validation Prediction은 전체 열화 궤적을 확인하기 위한 분석 목적으로 사용한다.

Full Validation Prediction은 Prefix Definition이나 TrainingRange에 의해 예측 대상 구간을 제한하지 않고, Held-out Bearing의 전체 수명에 대해 수행한다. 또한 공식 Validation Metric의 계산 대상이 아니며, 공식 평가에는 기존 Prefix Validation 결과만 사용한다.

이 기능은 전체 수명에 대한 모델의 예측 궤적을 확인할 수 있도록 분석 범위를 확장하지만, 모델 학습 방식이나 Stage 06의 추론 구조를 변경하지 않는다.

### 5.3 RUL Trajectory Visualization 및 Prefix Marker

Full Validation Prediction을 이용하여 각 Bearing의 실제 RUL과 예측 RUL을 전체 수명에 걸쳐 시각화한다.

그래프의 축은 다음과 같이 구성한다.

```text
X-axis: life_ratio
Y-axis: RUL_sec
```

`life_ratio`는 `actual_RUL_norm`으로부터 계산하며, RUL은 물리적 해석이 가능한 초 단위로 표시한다.

또한 Prefix Definition의 `end_idx`를 해당 Bearing의 실제 validation trajectory에서 찾아 동일한 `life_ratio` 좌표로 변환한 뒤 Prefix marker를 표시한다.

이를 통해 공식 평가에 사용되는 Prefix 경계를 전체 수명 궤적 위에서 함께 확인할 수 있다.

여러 Bearing의 궤적은 현재 2열 subplot으로 배치하며, Bearing 수가 홀수인 경우 마지막 빈 subplot은 숨긴다. 그래프 함수는 외부 `Axes`를 받을 수 있도록 구성하여 1열 또는 2열 등의 배치 변경을 Notebook의 레이아웃 계층에서 수행할 수 있도록 했다.

### 5.4 Feature Importance 추가

각 LOBO 모델에서 모델 종속적인 Feature Importance를 추출하고, Bearing별 Importance와 Mean/Std/Min/Max 요약을 출력하도록 기능을 추가했다.

Feature Importance는 각 모델이 입력 Feature를 얼마나 중요하게 활용하는지를 확인하기 위한 분석 자료이며, 이를 이용하여 Feature Selection을 자동으로 수행하지 않는다.

```text
Feature Importance
        ↓
해석 및 추가 Evidence
        X
자동 Feature Selection
```

따라서 최종 Feature Selection은 여전히 명시적인 Whitelist를 통해 결정된다.

---

## 6. 산출물 및 Stage 06 영향

### 6.1 Stage 05 공식 산출물

공식 산출물의 성격은 변경되지 않았다.

`model_bundle.pkl`은 Stage 06에서 모델 정의의 기준으로 사용하며, 다음 정보를 포함한다.

* `ModelingConfig`
* `TrainingRange`
* `feature_columns`
* `target_column`
* Ordered LOBO models
* Fold metadata

공식 Validation 결과는 다음 산출물로 구성된다.

* `lobo_validation_predictions.csv`
* `lobo_bearing_metrics.csv`
* `validation_summary.json`

공식 Validation은 Prefix Prediction 결과를 기반으로 한다.

반면 Full Validation Prediction과 RUL Trajectory Visualization은 Notebook 실행 과정에서 분석 및 시각화를 위해 생성되는 결과로, Stage 06의 공식 입력 산출물로 취급하지 않는다.

### 6.2 Stage 06 영향

이번 Stage 05 변경으로 Stage 06의 설계나 인터페이스를 변경할 필요는 없다.

Stage 06은 05단계에서 전달된 `Model Bundle`을 모델 정의의 기준으로 사용한다. 따라서 다음과 같은 모델 관련 정보가 Stage 05에서 변경되더라도 Stage 06에서 이를 별도로 재정의하지 않는다.

```text
Feature Columns
Target Column
TrainingRange
Algorithm
Hyperparameters
LOBO Models
```

이번 변경으로 Feature 구성과 `ModelingConfig`의 내용은 달라졌지만, 해당 정보가 최종 Model Bundle에 포함되어 전달되는 방식과 Stage 06에서 이를 소비하는 방식은 동일하다.

따라서 Stage 05 내부의 변경은 다음과 같은 경계를 유지한다.

```text
05 내부
├── Feature Selection 방식 변경
├── Modeling 조건 변경
├── Full Validation 및 Visualization 추가
└── Feature Importance 등 분석 기능 추가
        ↓
    Model Bundle
        ↓
06 기존 인터페이스 유지
```

즉, Stage 05에서 모델링 조건과 분석 기능은 확장되었지만, `Model Bundle → Test Inference & Ensemble`으로 이어지는 Stage 06의 인터페이스 계약에는 변화가 없다.

---

## 7. 최종 판단

이번 변경은 Stage 05의 **모델 입력 결정 방식을 명시화하고, 모델링 조건을 조정하며, 검증 및 분석 기능을 확장한 과정**으로 정리할 수 있다.

Feature Selection은 03에서 전달된 `candidate_features`를 기반으로 05에서 최종 모델 입력을 명시적으로 확정하는 구조로 구체화되었으며, TrainingRange와 RandomForest 설정은 수동 실험을 거쳐 현재 조건으로 갱신되었다. 또한 기존 Prefix Validation을 유지하면서 Full Validation Prediction과 RUL trajectory visualization, Prefix marker, Feature Importance를 추가하여 모델의 전체 수명 궤적과 입력 Feature의 중요도를 별도로 분석할 수 있도록 했다.

또한 공식 Validation과 추가 분석 결과의 역할을 구분하여, 검증 체계와 Stage 06 입력 산출물의 경계를 유지하였다.

결과적으로 이번 변경은 Stage 05 내부의 모델링 및 분석 범위를 확장했지만, 최종 `Model Bundle`의 역할과 이를 Stage 06에서 소비하는 방식은 유지되었다. 따라서 **Stage 06의 설계서 및 인터페이스 명세를 별도로 개정할 필요는 없으며**, 본 문서는 최초 Stage 05 구성에서 현재 구현에 이르기까지의 변경과 그에 따른 구조적 경계를 기록하는 업데이트 문서로 사용할 수 있다.