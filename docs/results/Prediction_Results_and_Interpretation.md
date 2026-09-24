# Prediction_Results_and_Interpretation

## 1. LOBO Validation에서의 Rational Calibration 효과

LOBO validation에서 Rational calibration 적용 전후를 비교한 결과, 전반적인 예측 성능이 크게 개선되었다.

| Method   | RMSE (s) | MAE (s) | SMAPE (%) |     R² | PHM Score |
| -------- | -------: | ------: | --------: | -----: | --------: |
| Raw      |  2349.53 | 1809.74 |     66.57 | -0.257 |    0.1437 |
| Rational |   868.84 |  683.20 |     34.29 |  0.828 |    0.3206 |

RMSE는 약 63%, MAE는 약 62%, SMAPE는 약 49% 감소했으며, R² 역시 음수에서 0.828로 크게 개선되었다. PHM Score도 0.1437에서 0.3206으로 상승하였다.

베어링별 결과를 보면 개선의 방향은 대체로 일관되지만 모든 베어링에서 동일한 정도로 나타나지는 않는다. 예를 들어 Bearing1_1은 RMSE가 4381초에서 1147초로 크게 감소했고, Bearing2_1 역시 2707초에서 554초로 감소했다. 반면 Bearing3_2에서는 RMSE가 774초에서 1255초로 오히려 증가했다.

이는 Rational calibration을 **예측 trajectory 자체를 새롭게 학습하는 방법이라기보다, 기존 모델의 RUL scale에 존재하는 체계적인 편향을 보정하는 방법**으로 해석하는 것이 적절함을 보여준다. Rational 함수는 단조 증가 변환이므로 기존 예측값의 상대적인 순서와 trajectory의 형태를 기본적으로 보존하면서 전체적인 RUL 규모를 조정한다.

따라서 calibration은 공통적인 scale bias에는 효과적이지만, 특정 베어링에만 나타나는 비정상적인 degradation trajectory나 모델의 잘못된 판단 자체를 해결해 주지는 않는다.

---

## 2. Test Set에서의 Calibration 효과

11개 공식 test bearing에 대해 harmonic ensemble을 적용한 결과에서도 Rational calibration의 효과가 확인되었다.

| Method              | RMSE (s) | MAE (s) | SMAPE (%) |     R² | PHM Score |
| ------------------- | -------: | ------: | --------: | -----: | --------: |
| Harmonic Raw        |  2931.41 | 2081.17 |     79.08 | -0.241 |    0.0773 |
| Harmonic + Rational |  2561.62 | 1500.06 |     61.68 |  0.052 |    0.3095 |

Calibration 이후 RMSE는 약 12.6%, MAE는 약 27.9%, SMAPE는 약 22.0% 감소했다. R²도 음수에서 양수로 전환되었다.

특히 **MAE와 SMAPE의 개선 폭이 RMSE보다 크다는 점**이 중요하다. 이는 많은 베어링에서 예측값이 실제 RUL에 가까워졌지만, 여전히 일부 베어링에서는 큰 오차가 남아 있음을 시사한다. 실제로 Bearing1_5, Bearing1_6, Bearing2_3, Bearing3_3 등에서는 calibration 이후에도 상당한 오차가 남아 있다.

따라서 calibration의 효과를 "모든 베어링의 예측을 고르게 개선했다"고 표현하기보다는,

> **전체적인 RUL scale을 실제 값에 더 가깝게 조정하여 다수의 예측에서 오차를 줄였지만, 일부 베어링의 큰 trajectory error까지 해결하지는 못했다.**

라고 해석하는 것이 적절하다.

---

## 3. PHM Score에서 나타나는 베어링별 성능 기여의 차이

Test set의 일반적인 회귀 지표와 PHM Score 사이에는 상당한 차이가 나타난다.

최종 Harmonic + Rational 결과는 RMSE 2561.6초, MAE 1500.1초, SMAPE 61.68%, R² 0.052로, 일반적인 회귀 성능만 놓고 보면 여전히 큰 오차가 존재한다. 그러나 PHM Score는 **0.3095**까지 상승하였다.

이는 PHM 2012 평가 함수의 특성으로 설명할 수 있다. 이 평가지표는 단순한 평균 오차가 아니라 **과대추정과 과소추정을 비대칭적으로 평가하고, 오차에 대해 지수적인 penalty를 적용한다.**

그 결과 일부 베어링에서 극단적인 오차가 발생하더라도 해당 베어링의 Score 기여가 매우 작아질 수 있는 반면, 비교적 정확하게 예측한 베어링은 전체 Score에 상당한 기여를 할 수 있다.

따라서 이번 결과는 직관적으로는 **"버릴 것은 버리고 맞출 것은 맞춘다"**는 양상을 보인다. 그러나 이것은 모델이 특정 베어링을 의도적으로 포기했다는 의미가 아니다. **PHM Score의 평가 구조가 베어링별 오차의 기여도를 비대칭적으로 만들기 때문에 나타나는 결과**이다.

즉, 이번 결과에서 일반 회귀지표와 PHM Score가 서로 다른 모습을 보이는 것은 오류라기보다, **서로 다른 목적의 평가 지표가 같은 예측 결과를 서로 다르게 평가한 것**으로 볼 수 있다.

---

## 4. Harmonic Ensemble의 역할

Arithmetic ensemble과 Harmonic ensemble을 비교하면 다음과 같다.

| Ensemble   | Calibration |        RMSE |         MAE |     SMAPE |        R² |  PHM Score |
| ---------- | ----------- | ----------: | ----------: | --------: | --------: | ---------: |
| Arithmetic | Raw         |     4238.60 |     3120.84 |     84.84 |    -1.595 |     0.0695 |
| Arithmetic | Rational    |     3327.59 |     2278.83 |     73.89 |    -0.600 |     0.1875 |
| Harmonic   | Raw         |     2931.41 |     2081.17 |     79.08 |    -0.241 |     0.0773 |
| Harmonic   | Rational    | **2561.62** | **1500.06** | **61.68** | **0.052** | **0.3095** |

Harmonic ensemble은 단순히 여러 모델의 RUL을 평균내는 대신, **RUL의 역수를 평균한 뒤 다시 역수를 취하는 방식**으로 해석할 수 있다. `1/RUL`은 **마모율**에 연관된 값으로 해석하였다.

이는 여러 모델이 예측한 remaining time을 직접 평균하는 것보다 **짧은 RUL 예측에 더 큰 영향력을 부여한다.** 따라서 이 방법의 주된 설계 의도는 PHM Score를 직접 최적화하는 것이 아니라,

> **여러 모델의 예측에서 종단/EoL에 가까운 판단을 보다 강하게 반영한다.**

PHM 2012의 평가 함수가 late prediction을 더 강하게 penalize한다는 점에서 이러한 특성이 결과적으로 PHM Score와도 잘 맞을 수 있지만, 이는 **설계 목적 자체라기보다 파생적인 효과**로 봄이 적절하다.

다만 Harmonic ensemble은 항상 유리한 것은 아니다. 예를 들어 이미 잘못 낮게 예측된 값이 있다면 harmonic aggregation이 그 낮은 값을 더욱 강하게 반영할 수 있다. 실제로 B2_3과 B3_3에서는 raw harmonic prediction이 각각 약 412초→168초, 176초→93초로 더 낮아졌다.

따라서 harmonic ensemble은 **짧은 RUL을 강조하는 대신 잘못된 low-RUL prediction 역시 강화할 수 있는 trade-off를 가진 방법**이다.

---

## 5. Rational Calibration과 Harmonic Ensemble의 상호작용

이번 실험에서 가장 주목할 부분은 두 방법을 독립적으로 적용했을 때보다 **함께 적용했을 때의 결과**이다.

Raw 상태에서는 harmonic과 arithmetic의 PHM Score 차이가 크지 않았다.

* Arithmetic Raw: **0.0695**
* Harmonic Raw: **0.0773**

반면 Rational calibration 이후에는 차이가 크게 벌어졌다.

* Arithmetic + Rational: **0.1875**
* Harmonic + Rational: **0.3095**

즉, 이번 실험에서는 단순히 "Harmonic이 Arithmetic보다 항상 우수하다"고 해석하기보다는, **두 방법이 서로 다른 문제를 다루면서 상호 보완적으로 작용했다**고 보는 것이 타당하다.

* **Rational calibration** → 개별 모델의 systematic RUL scale bias를 보정
* **Harmonic ensemble** → 여러 모델의 예측 중 짧은 RUL에 더 큰 영향력 부여

따라서 먼저 개별 모델의 scale을 calibration으로 조정한 뒤, 그 결과를 harmonic 방식으로 결합하면서 **보정된 개별 모델의 terminal-oriented 정보를 보다 강하게 반영하는 구조**가 만들어졌다.

이번 실험에서 최종적으로 가장 높은 성능을 보인 조합은 **Harmonic + Rational calibration**이었다.

---

## 6. 최종 Test 결과와 한계

최종 구성의 test-set 결과는 다음과 같다.

* RMSE: **2561.62 s**
* MAE: **1500.06 s**
* SMAPE: **61.68%**
* R²: **0.0521**
* PHM Score: **0.3095**

이 결과는 한 가지 지표만으로 평가하기 어려운 특성을 보여준다.

일반적인 회귀 관점에서는 여전히 상당한 오차가 존재하며, 특히 일부 bearing에서는 실제 RUL과 예측 RUL 사이의 차이가 매우 크다. 반면 공식 PHM Score에서는 높은 값을 얻었으며, 이는 일부 bearing에서의 상대적으로 정확한 예측이 비대칭적인 scoring function을 통해 크게 반영된 결과로 볼 수 있다.

따라서 이번 모델의 성과를 **"전체적인 RUL 예측 오차를 완전히 해결했다"**고 설명하기보다는,

> **RUL scale calibration을 통해 전반적인 예측 편향을 상당 부분 완화하고, harmonic ensemble을 통해 terminal 영역의 예측을 강조함으로써 공식 PHM 평가에서 높은 성능 기여를 확보했다. 그러나 일부 bearing-specific trajectory error는 여전히 남아 있으며, 이러한 오류는 현재의 calibration과 ensemble만으로 해결되지 않는다.**

라고 정리함이 가장 정확하다.

---

## 7. IEEE PHM 2012 결과와의 비교

공식 challenge의 평가 함수와 동일한 방식으로 계산한 PHM Score가 **0.3095**라는 점은 의미 있는 외부 기준을 제공한다.

당시 공개된 결과에서는 A.L.D.의 Sergey Porotsky 등이 제시한 결과가 **0.25**의 overall score를 기록했다고 보고되어 있으며, GE Global Research의 Tianyi Wang 등은 **0.0981**을 보고했다. 다만 이 수치들은 당시 실제 competition 참가 방법과 당시의 정보 및 실험 절차를 기반으로 한 역사적 결과이며, 현재의 LOBO validation이나 본 프로젝트의 modeling pipeline과 직접적인 순위 비교를 할 수 있는 자료는 아니다.

따라서 현재의 0.3095는 **"2012년 competition에서 우승한 것과 대등한 성능"이라는 의미가 아니라, 동일한 공식 test set과 scoring rule을 사용했을 때 과거 공개 결과와 비교할 수 있는 하나의 수치적 reference**로 사용하는 것이 적절하다.

---

### 전체적인 해석

이번 실험에서 가장 중요한 결론은 **하나의 기법이 모든 문제를 해결했다는 것이 아니라, 서로 다른 층위의 문제를 다루는 세 요소가 결합되었다는 점**이다.

> **Calibration은 scale bias를 보정하고, Harmonic ensemble은 short-RUL/terminal prediction의 영향력을 높이며, PHM Score는 이러한 예측을 비대칭적으로 평가한다.**

그 결과 일반적인 회귀지표와 공식 PHM Score 사이에는 차이가 존재하지만, 이는 두 평가 관점이 서로 다른 특성을 반영하기 때문이다. 동시에 일부 베어링에서 큰 오류가 지속된다는 사실은 현재 방법론의 한계도 분명히 보여준다.

이렇게 정리하면 **"높은 PHM Score를 얻었다"는 성과와 "아직 해결하지 못한 bearing-specific prediction error가 있다"는 한계를 동시에 유지**할 수 있다.