# 05_Modeling_Update

## 1. 문서 목적

본 문서는 최초 `05_Modeling.ipynb`에서 현재
`05_Modeling-Copy3.ipynb`까지 진행된 Stage 05의 변경 사항을 정리한다.

변경 사항은 기능 추가, 책임 조정, 모델링 조건의 개선 및 시각화 보강을
포함한다. 특히 이번 변경 과정에서 Stage 06의 인터페이스 및 설계 책임에
영향을 주는 변경이 발생했는지도 함께 확인한다.

------------------------------------------------------------------------

## 2. 변경 요약

  -----------------------------------------------------------------------------------------
  구분              최초 버전              현재 버전                      변경 성격
  ----------------- ---------------------- ------------------------------ -----------------
  Feature 해석      Dataset의 Metadata와   03에서 전달된                  Feature Selection
                    Target을 제외한        `candidate_features`를         책임 및 입력 해석
                    나머지를 자동으로      기준으로 05에서 최종 Feature를 변경
                    Feature로 사용         명시적으로 선택                

  Feature 목록      `split_columns()`로    `selection_config.json`의      명시적 Whitelist
                    자동 추출              `candidate_features` 검증 후   
                                           `SELECTED_FEATURE_COLUMNS`로   
                                           최종 확정                      

  Target            `RUL_norm`             `RUL_norm`                     유지

  TrainingRange     `0.20 ~ 0.95`          `0.30 ~ 1.00`                  실험 조건 변경

  RandomForest      `n_estimators=250`,    `n_estimators=100`,            실험 조건 변경
                    `max_depth=12`,        `max_depth=10`,                
                    `min_samples_leaf=2`   `min_samples_leaf=7`           

  하이퍼파라미터    초기 설정              수동 실험을 통해 더 적절한     파이프라인 외
  탐색                                     조건으로 갱신                  실험 과정

  LOBO Training     LOBO 모델 생성         동일                           유지

  Prefix Validation Prefix별 단일 샘플     동일                           유지
                    예측 및 평가                                          

  Full Validation   없음                   Held-out Bearing의 전체 수명   검증 시각화 기능
  Prediction                               예측 추가                      추가

  RUL Trajectory    없음                   Bearing별 전체 RUL 궤적 시각화 분석/시각화 기능
  Visualization                            추가                           추가

  Prefix Marker     없음                   Prefix 위치를 전체 수명        시각화 기능 보강
                                           그래프에 표시                  

  그래프 배치       해당 없음              2열 subplot                    시각화 편의성
                                                                          개선

  Feature           없음                   LOBO 모델별 Feature Importance 해석/분석 기능
  Importance                               및 요약 출력                   추가

  주요 산출물       Model Bundle, Prefix   동일한 공식 산출물 + Notebook  공식 인터페이스는
                    Validation 결과 및     내 추가 분석 결과              유지
                    평가 결과                                             
  -----------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 3. Feature Selection 구조 변경

### 3.1 최초 구조

최초 버전에서는 다음과 같이 Dataset의 컬럼을 자동 분리했다.

``` text
model_ready_learning_dataset
        ↓
META_COLUMNS + RUL_norm 제외
        ↓
전체 Feature Columns
        ↓
LOBO Training
```

즉 05에서 별도의 Feature Selection을 수행하지 않고, Model-ready
Dataset에 존재하는 모든 Feature를 모델 입력으로 사용했다.

### 3.2 현재 구조

현재는 03에서 작성된 Selection Manifest를 읽어 `candidate_features`를
확인한 뒤, 05에서 최종 Feature를 명시적으로 선택한다.

``` text
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

현재 최종 선택은 Notebook의 `SELECTED_FEATURE_COLUMNS`에 명시된
Whitelist를 사용한다.

현재 선택된 Feature는 다음과 같다.

``` text
RDI_rms
RDI_cf
Rectified_cf
Rectified_kurt
```

이 변경으로 인해 05의 모델 입력 Feature는 더 이상 Dataset의 전체 Feature
집합에 의해 암묵적으로 결정되지 않는다. `candidate_features`에
포함되는지를 검증한 뒤 최종 선택 목록을 확정한다.

Target은 기존과 동일하게 `RUL_norm`을 유지한다.

------------------------------------------------------------------------

## 4. 모델링 조건 변경

### 4.1 TrainingRange

최초:

``` text
0.20 ~ 0.95
```

현재:

``` text
0.30 ~ 1.00
```

각 LOBO fold에서 Held-out Bearing을 제외한 학습 Bearing에 동일한
TrainingRange 정책을 적용한다.

이는 모델 학습의 실험 조건 변경이며, TrainingRange의 개념이나 적용 방식
자체가 변경된 것은 아니다.

### 4.2 RandomForest 설정

최초 설정:

``` text
algorithm      = RandomForest
n_estimators   = 250
max_depth      = 12
min_samples_leaf = 2
random_state   = 42
n_jobs         = -1
```

현재 설정:

``` text
algorithm      = RandomForest
n_estimators   = 100
max_depth      = 10
min_samples_leaf = 7
random_state   = 42
n_jobs         = -1
```

현재 설정은 별도의 자동화된 Hyperparameter Search 인터페이스를 추가한
결과가 아니다. 수동으로 여러 실험 조건을 확인하여 더 적절한 조건으로
갱신한 것이다.

따라서 최종 파이프라인은 최초와 마찬가지로 하나의 확정된
`ModelingConfig`를 사용하여 LOBO 모델을 생성한다.

------------------------------------------------------------------------

## 5. LOBO 및 공식 Validation 구조

핵심적인 LOBO 구조는 유지된다.

각 Bearing을 한 번씩 Held-out으로 제외하여 모델을 생성하고, 해당
Bearing은 자신의 Held-out 모델로만 검증한다.

``` text
Bearing A held-out → Model A
Bearing B held-out → Model B
...
```

Prefix Definition은 공식 Validation의 평가 대상을 결정한다.

각 Prefix Definition은 하나의 validation row를 선택하고, 그 예측값은
해당 Bearing을 Held-out한 LOBO 모델에서 생성한다.

예측 Target은 계속 `RUL_norm`이며, 공식 평가는 이를 `elapsed_sec`와 함께
`RUL_sec`로 복원한 뒤 수행한다.

Stage 05에서는 LOBO 모델 간 앙상블을 수행하지 않는다.

------------------------------------------------------------------------

## 6. Full Validation Prediction 추가

최초 버전에는 공식 Prefix Validation만 존재했다.

현재는 별도로 각 Held-out Bearing의 전체 수명에 대해 해당 LOBO 모델의
예측을 생성한다.

``` text
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

이 결과는 전체 열화 궤적을 확인하기 위한 시각화 목적이다.

따라서:

-   Prefix Definition은 Full Validation Prediction을 제한하지 않는다.
-   TrainingRange도 Full Validation Prediction을 제한하지 않는다.
-   Full Validation Prediction은 공식 Validation Metric 계산 대상이
    아니다.
-   공식 평가에는 기존 Prefix Validation 결과만 사용한다.

이 추가 기능은 모델 학습이나 06단계 추론을 변경하지 않는다.

------------------------------------------------------------------------

## 7. RUL Trajectory Visualization 및 Prefix Marker

Full Validation Prediction을 이용하여 각 Bearing의 실제 RUL과 예측 RUL을
전체 수명에 걸쳐 시각화한다.

그래프의 축은 다음으로 확정했다.

``` text
X-axis: life_ratio
Y-axis: RUL_sec
```

`life_ratio`는 `actual_RUL_norm`으로부터 계산하며, RUL은 물리적 해석이
가능한 초 단위로 표시한다.

또한 Prefix Definition의 `end_idx`를 해당 Bearing의 실제 validation
trajectory에서 찾아 동일한 `life_ratio` 좌표로 변환한 뒤 그래프에 Prefix
marker를 표시한다.

따라서 Prefix marker는 공식 평가에 사용되는 Prefix 경계를 전체 수명 궤적
위에서 시각적으로 확인하기 위한 것이다.

------------------------------------------------------------------------

## 8. Visualization Layout 개선

최초 Full Validation Visualization에는 그래프가 존재하지 않았으나,
현재는 여러 Bearing의 궤적을 2열 subplot으로 출력한다.

``` text
┌──────────────┬──────────────┐
│ Bearing 1    │ Bearing 2    │
├──────────────┼──────────────┤
│ Bearing 3    │ Bearing 4    │
├──────────────┼──────────────┤
│ Bearing 5    │              │
└──────────────┴──────────────┘
```

Bearing 수가 홀수인 경우 마지막 빈 subplot은 숨긴다.

그래프 함수는 외부 `Axes`를 받을 수 있도록 구성하여, 향후 화면 너비에
따라 1열 또는 2열로 배치하는 변경을 Notebook의 배치 계층에서 수행할 수
있도록 했다.

------------------------------------------------------------------------

## 9. Feature Importance 추가

현재는 각 LOBO 모델에서 모델 종속적인 Feature Importance를 추출하고,
Bearing별 Importance와 Mean/Std/Min/Max 요약을 출력한다.

이 결과는 Feature Selection을 자동으로 수행하지 않는다.

즉:

``` text
Feature Importance
        ↓
해석 및 추가 Evidence
        X
자동 Feature Selection
```

의 관계를 유지한다.

최종 Feature Selection은 여전히 Notebook에서 명시적인 Whitelist로
결정한다.

------------------------------------------------------------------------

## 10. Stage 05 공식 산출물

공식 산출물의 성격은 변경되지 않았다.

### Model Bundle

`model_bundle.pkl`

06단계에서 모델 정의의 기준으로 사용한다.

Bundle에는 다음 정보가 포함된다.

-   `ModelingConfig`
-   `TrainingRange`
-   `feature_columns`
-   `target_column`
-   Ordered LOBO models
-   Fold metadata

### Validation Results

-   `lobo_validation_predictions.csv`
-   `lobo_bearing_metrics.csv`
-   `validation_summary.json`

공식 Validation은 Prefix Prediction 결과를 기반으로 한다.

### 시각화용 결과

Full Validation Prediction과 RUL trajectory는 Notebook 실행 중 분석 및
시각화를 위해 생성되며 공식 06 입력 산출물로 취급하지 않는다.

------------------------------------------------------------------------

## 11. Stage 06 영향 검토

이번 Stage 05 개정으로 Stage 06의 설계나 인터페이스를 변경할 필요는
없다.

Stage 06은 05단계의 `Model Bundle`을 모델 정의의 유일한 기준으로
사용한다.

따라서 다음 정보가 05에서 변경되어도 06은 이를 다시 정의하지 않는다.

``` text
Feature Columns
Target Column
TrainingRange
Algorithm
Hyperparameters
LOBO Models
```

06은 전달받은 Model Bundle을 사용하여 Test Dataset에 대한 Per-LOBO
Prediction과 Intra-Config Ensemble을 수행한다.

이번 개정에서 변경된 Feature 구성과 ModelingConfig는 최종 Model Bundle에
반영되지만, **Model Bundle의 역할과 06단계의 소비 방식은 동일하다.**

따라서 이번 업데이트는:

``` text
05 내부
├── Feature Selection 방식 변경
├── Modeling 조건 변경
├── Full Validation 및 Visualization 추가
└── 분석 결과 보강
        ↓
    Model Bundle
        ↓
06 기존 인터페이스 유지
```

로 정리할 수 있다.

------------------------------------------------------------------------

## 12. 최종 판단

이번 변경의 본질은 **Stage 05의 모델링 실험과 검증·분석 기능을 발전시킨
것**이다.

특히 Feature Selection 책임이 03의 결정 결과를 받아 05에서 최종 모델
입력을 확정하는 구조로 구체화되었고, 모델링 조건은 수동 실험을 통해
갱신되었다. 또한 Full Validation Prediction, RUL trajectory
visualization, Prefix marker 및 Feature Importance가 추가되었다.

그러나 이러한 변화는 06단계가 요구하는
`Model Bundle → Test Inference & Ensemble` 계약을 변경하지 않는다.

따라서 **06단계 설계서 및 인터페이스 명세의 개정은 필요하지 않으며**, 본
문서는 최초 05단계 설계 이후 현재 구현까지의 변경 이력을 기록하는
업데이트 문서로 사용할 수 있다.
