# 05_Calibration_Update

## 1. 목적

Stage 05에서 수행한 RUL 출력 보정(Calibration) 실험의 설계, 검증 결과, 산출물 및 Stage 06 전달 사항을 정리한다.

Calibration은 모델 자체를 변경하지 않고 학습된 모델의 `RUL_norm` 출력에 폐형식 pointwise 함수를 적용하는 후처리이다. 따라서 기존 Model Bundle의 모델 정의는 변경하지 않는다.

## 2. 보정 원칙

목적은 예측 trajectory의 진동성 noise를 제거하는 것이 아니라, 모델의 체계적인 출력 수준 및 비선형 편향을 최소한의 함수로 보정하는 것이다.

- temporal smoothing을 수행하지 않는다.
- temporal monotonicity를 강제하지 않는다.
- 원래 prediction의 noise/oscillation 특성을 보존한다.
- `RUL_norm = 0` 경계를 보존한다.
- pointwise monotone mapping만 사용한다.
- calibration parameter는 LOBO 모델마다 독립적으로 결정한다.

## 3. 보정 함수

### Power

`f(y) = y^gamma`, `gamma > 0`

`f(0)=0`, `f(1)=1`이며 양수 gamma에서 단조 증가한다.

### Rational

`f(y) = a*y / (1 + (a-1)*y)`, `a > 0`

`f(0)=0`, `f(1)=1`이며 양수 a에서 단조 증가한다.

Power와 Rational 모두 시간축 자체를 보정하지 않고 각 prediction 값을 재매핑한다.

## 4. Calibration 설정

지원 방법:

```text
None
"power"
"rational"
"both"
```

기본값:

```text
CALIBRATION_METHOD = "both"
CALIBRATION_OBJECTIVE = "PHM_Score"
```

`None`이면 calibration을 수행하지 않는다.

지원 objective는 `RMSE`, `MAE`, `SMAPE`, `R2`, `PHM_Score`이며 기본 objective는 PHM Score이다.

## 5. Parameter fitting

각 LOBO 모델에 대해 독립적으로 parameter를 최적화한다.

```text
LOBO Model 1 → parameter 1
LOBO Model 2 → parameter 2
...
```

수학적으로 필요한 양수 조건만 적용한다.

```text
gamma > 0
a > 0
```

초기 탐색 범위는 매우 넓은 로그 범위 `10^-8 ~ 10^8`으로 설정하였다. 극단적인 parameter가 선택되면 단순히 탐색 범위를 확대하기보다 calibration 함수의 표현력이나 원본 prediction의 문제 가능성을 먼저 검토한다.

## 6. Fitting 범위

Calibration fitting은 `prefix_definition`에 정의된 prefix와 **동일한 life-ratio 지점**을 사용한다.

별도의 calibration prefix 설정은 두지 않는다. 따라서 prefix 시나리오를 변경하면 calibration fitting 지점도 함께 변경된다.

현재 구현에서는 이 평가 지점의 prediction을 사용하며 별도의 sample weighting은 적용하지 않는다.

## 7. 평가 기준

Prediction target은 계속 `RUL_norm`이지만 calibration parameter의 최적화와 결과 평가는 `RUL_sec` 공간에서 수행한다.

```text
RUL_norm Prediction
        ↓
Calibration
        ↓
RUL_sec Reconstruction
        ↓
Metric Evaluation
```

RMSE, MAE, SMAPE, R² 및 PHM Score를 모두 출력한다.

## 8. Validation에서의 낙관 편향

이번 실험에서는 각 LOBO 모델이 held-out한 Bearing의 Prefix 정답을 이용하여 해당 모델의 최적 calibration parameter를 직접 fitting하였다.

따라서 현재 결과는 최종적인 leakage-safe calibration 성능이 아니다.

현재 단계에서는 다음 두 가능성을 구분하지 않는다.

1. 모델 자체의 공통적인 출력 편향을 보정한 경우
2. 특정 Bearing의 특성에 맞춰 parameter가 조정된 경우

즉 validation 성능의 개선에는 낙관 편향이 포함될 가능성이 있다. 이번 단계의 목적은 calibration의 일반화 성능을 확정하는 것이 아니라 단순한 출력 보정으로 현재 모델의 체계적인 오차를 얼마나 줄일 수 있는지 확인하는 것이다.

## 9. 실험 결과

추가 실험에서 사용한 특징 조합:

```text
["RDI_cf", "RDI_kurt", "Rectified_rms", "Rectified_p2p", "temp_mean"]
```

모델 parameter:

```text
max_depth = 12
min_samples_leaf = 7
```

Rational calibration과 RMSE primary objective를 사용하였다.

### Global comparison

| Method | RMSE | MAE | SMAPE | R² | PHM Score |
|---|---:|---:|---:|---:|---:|
| Raw | 2349.53 | 1809.74 | 66.57 | -0.257 | 0.841 |
| Rational | **868.84** | **683.20** | **34.29** | **0.828** | **0.936** |

42개 Prefix 평가점 전체에서 Rational 적용 후 RMSE는 약 63%, MAE는 약 62%, SMAPE는 약 49% 감소하였다. R²는 -0.257에서 0.828로 상승하였다.

### Bearing-level RMSE

| Bearing | Raw | Rational | 변화 |
|---|---:|---:|---:|
| Bearing1_1 | 4381.34 | **1147.33** | -73.8% |
| Bearing1_2 | 843.73 | **672.76** | -20.3% |
| Bearing2_1 | 2706.98 | **554.48** | -79.5% |
| Bearing2_2 | 1896.98 | **805.37** | -57.5% |
| Bearing3_1 | 773.58 | **479.80** | -38.0% |
| Bearing3_2 | 1299.62 | **1254.60** | -3.5% |

모든 validation Bearing에서 RMSE가 감소하였다.

## 10. Calibration parameter 관찰

Rational parameter:

```text
model_0 (Bearing1_1): a = 3.8670994
model_1 (Bearing1_2): a = 0.72492888
model_2 (Bearing2_1): a = 0.26372612
model_3 (Bearing2_2): a = 0.34402919
model_4 (Bearing3_1): a = 0.54049298
model_5 (Bearing3_2): a = 0.88665213
```

보정 방향은 모델마다 달랐다. 따라서 모든 모델에 공통적인 상향 또는 하향 bias가 있는 것은 아니다.

동시에 일부 parameter가 크게 벗어나므로, 현재 개선이 모델 자체의 체계적인 출력 편향인지 Validation Bearing 특화 보정인지는 아직 확정할 수 없다.

## 11. Trajectory 관찰

Rational calibration은 Raw prediction의 temporal structure를 smoothing하지 않는다.

실험 결과에서도 Raw의 변동 특성이 상당 부분 유지되면서 prediction level이 변경되었다. 따라서 이번 calibration은 trajectory smoothing이 아니라 **prediction value의 monotone nonlinear remapping**으로 해석한다.

이는 진동성 noise를 별도로 제거하지 않고 추세 수준만 보정한다는 당초 목적에 부합한다.

## 12. Full Validation

Prefix fitting으로 결정한 parameter는 해당 held-out Bearing의 전체 validation trajectory에도 적용한다.

Full Validation은 trajectory 시각화 및 분석용으로 사용하며, calibration fitting이나 새로운 공식 평가 데이터로 취급하지 않는다.

Temporal smoothing과 monotonic projection은 수행하지 않는다.

## 13. 저장 산출물

### `calibration_results.json`

다음을 저장한다.

- calibration method
- primary objective
- objective direction
- LOBO 모델별 calibration parameter
  - Power: `gamma`
  - Rational: `a`
- Raw 및 calibrated global metric 결과

따라서 Stage 06에서 calibration을 재현하기 위한 모델별 parameter가 이 파일에 포함된다.

### `calibrated_validation_predictions.csv`

Prefix validation 결과에 대해 Raw 및 calibration 결과를 함께 저장한다.

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

## 14. Model Bundle

**Calibration parameter는 Model Bundle에 추가하지 않는다.**

Calibration은 모델의 학습 결과나 구조를 변경하지 않으며, 함수가 폐형식으로 정의되어 있고 parameter가 `calibration_results.json`에 별도로 저장된다.

따라서 Model Bundle은 기존 모델 정의의 기준으로 유지한다.

## 15. Stage 06 전달 사항

Stage 06에서 필요한 추가 입력은 다음으로 충분하다.

```text
Model Bundle
    +
calibration_results.json
```

예상 흐름:

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

Calibration parameter는 모델별로 대응하여 적용한다.

## 16. Stage 06에서 확인할 사항

현재 validation에서는 각 held-out Bearing에 맞춰 parameter를 직접 fitting했으므로, 관찰된 개선을 일반화 성능으로 해석하지 않는다.

Stage 06에서 실제 Test inference에 동일한 model-specific parameter를 적용하여 다음을 확인한다.

- Validation의 개선이 Test에서도 유지되는가
- 특정 모델의 calibration 방향이 Test에서도 유효한가
- Calibration 적용 후 Test 성능이 악화되는가
- Power와 Rational 중 어느 계열이 더 안정적인가
- 현재 개선이 모델 자체의 체계적인 출력 편향 보정인지 Validation Bearing에 대한 과적합성 보정인지 판단할 근거가 나타나는가

따라서 Calibration의 정식 pipeline 편입 여부는 Stage 06 결과 이후 결정한다.

## 17. Stage 05 최종 상태

```text
Stage 05 Modeling & LOBO Validation
│
├─ LOBO Training
├─ Prefix Validation
├─ Full Validation / Trajectory Visualization
├─ Feature Importance
│
└─ Experimental RUL Calibration
     ├─ Power
     ├─ Rational
     ├─ Model-specific parameter fitting
     ├─ Raw vs Calibrated evaluation
     └─ Calibration artifacts
          ├─ calibration_results.json
          └─ calibrated_validation_predictions.csv

                         ↓

                 Stage 06 Test Inference
                         │
                         ├─ Model Bundle
                         └─ calibration_results.json
```

**현재 Stage 05에서 Calibration과 관련하여 추가로 수행할 작업은 없다.**

다음 판단은 Stage 06에서 실제 Test inference에 보정을 적용한 뒤 수행한다.