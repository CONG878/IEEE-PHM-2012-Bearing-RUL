# 02_RUL Calibration Design and Application

## 1. 목적

RUL 출력 보정(Calibration)의 설계, parameter fitting 및 평가 구조, 실험 조건 변경, 적용 결과와 산출물, 그리고 최종 pipeline 편입 상태를 정리한다.

Calibration은 모델 자체를 변경하지 않고 학습된 모델의 `RUL_norm` 출력에 폐형식 pointwise 함수를 적용하는 후처리이다. 따라서 기존 Model Bundle의 모델 정의는 변경하지 않는다.

## 2. Calibration 설계

### 2.1 보정 원칙

Calibration의 목적은 예측 trajectory의 진동성 noise를 제거하는 것이 아니라, 모델의 체계적인 출력 수준 및 비선형 편향을 최소한의 함수로 보정하는 것이다.

* temporal smoothing을 수행하지 않는다.
* temporal monotonicity를 강제하지 않는다.
* 원래 prediction의 noise/oscillation 특성을 보존한다.
* `RUL_norm = 0` 경계를 보존한다.
* pointwise monotone mapping만 사용한다.
* calibration parameter는 LOBO 모델마다 독립적으로 결정한다.

### 2.2 보정 함수

#### Power

`f(y) = y^gamma`, `gamma > 0`

`f(0)=0`, `f(1)=1`이며 양수 gamma에서 단조 증가한다.

#### Rational

`f(y) = a*y / (1 + (a-1)*y)`, `a > 0`

`f(0)=0`, `f(1)=1`이며 양수 a에서 단조 증가한다.

Power와 Rational 모두 시간축 자체를 보정하지 않고 각 prediction 값을 재매핑한다.

### 2.3 Calibration 설정

지원 방법은 다음과 같다.

```text
None
"power"
"rational"
"both"
```

초기 구현에서는 다음을 기본값으로 설정하였다.

```text
CALIBRATION_METHOD = "both"
CALIBRATION_OBJECTIVE = "PHM_Score"
```

`None`이면 Calibration을 수행하지 않는다.

지원 objective는 `RMSE`, `MAE`, `SMAPE`, `R2`, `PHM_Score`이며, 초기 기본 objective는 PHM Score이다.

최종 실험에서는 Rational Calibration과 RMSE를 primary objective로 사용하였다.

### 2.4 Parameter fitting

각 LOBO 모델에 대해 Calibration parameter를 독립적으로 최적화한다.

```text
LOBO Model 1 → parameter 1
LOBO Model 2 → parameter 2
...
```

parameter에는 함수의 정의에 필요한 양수 조건만 적용한다.

```text
gamma > 0
a > 0
```

초기 탐색 범위는 `10^-8 ~ 10^8`의 넓은 로그 범위로 설정하였다. 극단적인 parameter가 선택되는 경우 단순히 탐색 범위를 확대하기보다 Calibration 함수의 표현력이나 원본 prediction의 특성을 우선 검토한다.

## 3. Calibration Fitting 및 평가

### 3.1 Fitting 범위

Calibration fitting은 `prefix_definition`에 정의된 prefix와 동일한 life-ratio 지점을 사용한다.

별도의 Calibration prefix 설정은 두지 않는다. 따라서 prefix 시나리오를 변경하면 Calibration fitting 지점도 함께 변경된다.

현재 구현에서는 각 평가 지점의 prediction을 사용하며 별도의 sample weighting은 적용하지 않는다.

### 3.2 평가 기준

Prediction target은 계속 `RUL_norm`으로 유지하지만, Calibration parameter의 최적화와 결과 평가는 `RUL_sec` 공간에서 수행한다.

```text
RUL_norm Prediction
        ↓
Calibration
        ↓
RUL_sec Reconstruction
        ↓
Metric Evaluation
```

평가 결과로 RMSE, MAE, SMAPE, R² 및 PHM Score를 출력한다.

### 3.3 Validation 데이터에 대한 Fitting

각 LOBO 모델이 held-out한 Bearing의 Prefix 정답을 이용하여 해당 모델의 Calibration parameter를 직접 fitting하였다.

따라서 각 모델의 Calibration parameter는 해당 validation bearing의 평가 지점에서 독립적으로 결정된다.

이 방식은 Calibration parameter가 별도의 학습 데이터에서 사전에 결정된 구조가 아니므로, 해당 결과를 leakage-safe한 일반화 성능으로 해석하지 않는다. Calibration fitting 방식 자체의 독립성 및 일반화에 관한 방법론적 제약은 연구의 한계에서 별도로 다룬다.

## 4. 실험 조건 변경

### 4.1 Feature set 변경

Calibration 실험 과정에서 사용한 Feature set을 다음과 같이 변경하였다.

기존:

```text
["RDI_rms", "RDI_cf", "Rectified_cf", "Rectified_kurt"]
```

변경:

```text
["RDI_cf", "RDI_kurt", "Rectified_rms", "Rectified_p2p", "temp_mean"]
```

변경된 Feature set을 최종 Calibration 실험에 사용하였다.

### 4.2 RandomForest 설정 변경

Calibration 실험 과정에서 RandomForest 설정을 다음과 같이 변경하였다.

```text
max_depth: 12
min_samples_leaf: 7
```

변경된 모델 설정을 최종 Calibration 실험에 사용하였다.

## 5. 적용 결과 및 관찰

### 5.1 Validation 결과

최종 Calibration 실험은 변경된 Feature set과 RandomForest 설정에서 Rational Calibration 및 RMSE primary objective를 사용하여 수행하였다.

42개 Prefix 평가점 전체에 대한 결과는 다음과 같다.

| Method   |       RMSE |        MAE |     SMAPE |        R² |  PHM Score |
| -------- | ---------: | ---------: | --------: | --------: | ---------: |
| Raw      |    2349.53 |    1809.74 |     66.57 |    -0.257 |     0.1437 |
| Rational | **868.84** | **683.20** | **34.29** | **0.828** | **0.3206** |

Bearing별 RMSE 역시 모든 validation Bearing에서 Rational 적용 후 감소하였다.

| Bearing    |     Raw |    Rational |     변화 |
| ---------- | ------: | ----------: | -----: |
| Bearing1_1 | 4381.34 | **1147.33** | -73.8% |
| Bearing1_2 |  843.73 |  **672.76** | -20.3% |
| Bearing2_1 | 2706.98 |  **554.48** | -79.5% |
| Bearing2_2 | 1896.98 |  **805.37** | -57.5% |
| Bearing3_1 |  773.58 |  **479.80** | -38.0% |
| Bearing3_2 | 1299.62 | **1254.60** |  -3.5% |

세부적인 성능 변화와 그 의미에 대한 해석은 결과 및 해석 문서에서 다룬다.

### 5.2 Calibration Parameter 관찰

Rational parameter는 다음과 같이 결정되었다.

```text
model_0 (Bearing1_1): a = 3.8670994
model_1 (Bearing1_2): a = 0.72492888
model_2 (Bearing2_1): a = 0.26372612
model_3 (Bearing2_2): a = 0.34402919
model_4 (Bearing3_1): a = 0.54049298
model_5 (Bearing3_2): a = 0.88665213
```

Calibration parameter의 방향은 모델마다 다르게 나타났다. 즉, 모든 모델에 동일한 방향의 pointwise 보정이 적용된 것은 아니다.

각 parameter의 의미를 모델 자체의 공통적인 출력 편향으로 해석할지, 특정 validation bearing에 대한 보정 결과로 해석할지는 이 fitting 구조만으로 확정하지 않는다.

### 5.3 Trajectory 및 Full Validation

Rational Calibration은 Raw prediction의 temporal structure를 smoothing하지 않는다.

실험에서도 Raw prediction의 변동 특성이 유지되면서 prediction level이 변경되었다. 따라서 이번 Calibration은 trajectory smoothing이 아니라 **prediction value의 monotone nonlinear remapping**으로 해석한다.

이는 진동성 noise를 별도로 제거하지 않고 prediction의 수준을 보정한다는 설계 원칙에 부합한다.

Prefix fitting으로 결정한 parameter는 해당 held-out Bearing의 전체 validation trajectory에도 적용하였다.

Full Validation은 trajectory 시각화 및 분석을 위한 용도로 사용하며, Calibration fitting이나 새로운 공식 평가 데이터로 취급하지 않는다.

Temporal smoothing과 monotonic projection은 수행하지 않는다.

## 6. 산출물 및 Pipeline 연계

### 6.1 Calibration 산출물

#### `calibration_results.json`

다음 정보를 저장한다.

* Calibration method
* primary objective
* objective direction
* LOBO 모델별 Calibration parameter

  * Power: `gamma`
  * Rational: `a`
* Raw 및 calibrated global metric 결과

따라서 모델별 Calibration parameter를 별도 파일에서 관리하고 동일한 Calibration을 재현할 수 있다.

#### `calibrated_validation_predictions.csv`

Prefix validation 결과에 대해 Raw 및 Calibration 결과를 함께 저장한다.

```text
Raw predicted_RUL_norm
Raw predicted_RUL_sec

Power:
    predicted_RUL_norm_power
    predicted_RUL_sec_power

Rational:
    predicted_RUL_norm_rational
    predicted_RUL_sec_rational
```

Raw prediction은 덮어쓰지 않는다.

### 6.2 Model Bundle과의 관계

**Calibration parameter는 Model Bundle에 추가하지 않는다.**

Calibration은 모델의 학습 결과나 구조를 변경하지 않으며, 함수가 폐형식으로 정의되어 있고 parameter가 `calibration_results.json`에 별도로 저장된다.

따라서 Model Bundle은 기존 모델 정의의 기준으로 유지한다.

### 6.3 Pipeline 적용 구조

Calibration은 Model Bundle과 별도의 산출물로 관리하며, Test inference에서 다음과 같이 적용한다.

```text
Model Bundle
    ↓
Per-LOBO Test RUL_norm Prediction
    ↓
Model-specific Calibration
    ↓
Calibrated RUL_norm
    ↓
Existing Ensemble
    ↓
Test RUL Prediction
```

Calibration parameter는 각 LOBO 모델에 대응하여 적용한다.

## 7. 최종 적용 및 상태

### 7.1 Test inference 적용

Stage 06 Test inference에서는 각 LOBO 모델의 Test `RUL_norm` prediction에 해당 모델의 Calibration parameter를 적용하였다.

Calibration은 기존 ensemble 처리에 앞서 적용되며, Raw prediction을 별도로 유지하여 Calibration 전후 결과를 비교할 수 있도록 구성하였다.

Test set에서의 구체적인 성능 변화와 그 해석은 결과 및 해석 문서에서 다룬다.

### 7.2 Rational Calibration의 최종 선택

Calibration 실험을 거쳐 최종 적용 방식으로 Rational Calibration을 선택하였다.

Rational은 다음 조건을 유지하면서 prediction 값을 비선형적으로 재매핑한다.

```text
f(0) = 0
f(1) = 1
a > 0
```

따라서 `RUL_norm`의 경계를 보존하면서 모델별 parameter를 이용한 pointwise Calibration을 수행할 수 있다.

### 7.3 Pipeline 편입

Rational Calibration과 모델별 parameter 관리 구조를 최종 pipeline에 편입하였다.

최종적인 처리 관계는 다음과 같다.

```text
Stage 05
    │
    ├─ LOBO Model Training
    ├─ Prefix Validation
    ├─ RUL Calibration
    │    └─ calibration_results.json
    │
    └─ Model Bundle
             │
             ▼
        Test Inference
             │
             ├─ RUL_norm Prediction
             ├─ Model-specific Calibration
             └─ Ensemble
                    │
                    ▼
              Test RUL Prediction
```

이에 따라 Calibration은 모델 학습 과정과 분리된 독립적인 후처리 단계로 유지하면서, Test inference 및 ensemble 과정에 일관되게 적용할 수 있는 상태로 확정하였다.