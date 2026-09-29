# 05_06_Modeling and Test Inference

---

## 1. 개요 및 설계 원칙

* **책임 분리:** 05단계는 단일 모델링 설정에 따른 **학습 및 일반화 검증**을 수행하며, 06단계는 확정된 모델을 바탕으로 **Test 추론 및 예측 통합**을 전담한다.
* **역할 혼합 금지:** 05단계에서는 앙상블을 수행하지 않으며, 06단계에서는 모델링 조건을 재정의하거나 모델을 재학습하지 않는다.
* **Config 분리:** 모델 알고리즘과 학습 조건은 `ModelingConfig`로 05단계에서 결정하고, 예측 통합 방식은 `EnsembleConfig`로 06단계에서 독립적으로 결정한다.
* **평가 단위:** 모델의 예측 Target은 `RUL_norm`일 수 있으나, 성능 평가는 초 단위 `RUL_sec`를 기준으로 수행한다.
* **공통 모듈 활용:** 05의 검증 평가와 06의 Test 평가는 동일한 공통 평가 Utility를 공유한다.

### 기본 정책

* `ModelingConfig.algorithm`의 기본값은 `RandomForest`다.
* `EnsembleConfig.method`의 기본값은 `arithmetic`이다.
* `EnsembleConfig.weights`의 기본값은 `None`이며, 이는 단순 평균을 의미한다.

---

## 2. 전체 데이터 흐름도

```text
[04 Dataset Builder]
        ├─ model_ready_learning_dataset
        ├─ prefix_definition
        └─ model_ready_test_dataset
                  │
                  ▼
[05 Modeling & LOBO Validation] ── (Input: ModelingConfig)
        ├─ LOBO Training & Prefix Validation
        ├─ RUL_norm Prediction → RUL_sec Reconstruction
        └─ Output: Validation Metrics, Model Bundle
                  │
                  ▼
[06 Test Inference & Ensemble] ─── (Input: EnsembleConfig)
        ├─ Per-LOBO Prediction & Intra-Config Ensemble
        ├─ RUL_norm Prediction → RUL_sec Reconstruction
        └─ Output: Final Test Prediction, Optional Test Metrics
```

---

## 3. 단계별 명세

### 3.1. [05 단계] Modeling & LOBO Validation

지정된 모델링 조건을 사용하여 LOBO(Leave-One-Bearing-Out) 모델 집합을 생성하고, Prefix 시나리오에 대한 일반화 성능을 검증하는 단계다.

#### 입력 (Input)

* `model_ready_learning_dataset`: 선택된 Feature와 Target이 포함된 학습 데이터.
* `prefix_definition`: 학습 데이터의 평가 절단 시점을 정의한 시나리오.
* `ModelingConfig`: 모델링 알고리즘 및 모델 하이퍼파라미터 정의.

지원 알고리즘은 다음과 같다.

```text
RandomForest
LightGBM
XGBoost
CatBoost
```

기본값은 `RandomForest`다.

#### 처리 (Processing)

* **LOBO Training:** 전체 K개의 Bearing에 대해 하나씩 제외하며 K개의 모델을 생성한다.
* **Prefix Validation:** 각 모델은 자신이 학습에서 제외했던 Bearing의 Prefix 데이터에 대해서만 예측한다.
* **RUL Reconstruction:** 모델이 `RUL_norm`을 예측하는 경우, 해당 샘플의 `elapsed_sec`와 RUL Target 정의에 따른 역변환을 통해 초 단위 `RUL_sec` 예측값을 복원한다.
* **Validation Evaluation:** 복원된 `RUL_sec` 예측값과 실제 `RUL_sec`를 사용하여 성능을 평가한다.

> **제약:** 동일 Validation 샘플에 대한 다중 모델 예측이나 앙상블은 본 단계에서 수행하지 않는다.

#### 출력 (Output)

* **`Model Bundle`**

  * 확정된 `ModelingConfig`
  * 순서가 보장된 K개의 LOBO 모델 객체
  * Fold 매핑 메타데이터

  06단계는 `Model Bundle`에 포함된 모델링 정보를 모델 정의의 유일한 기준으로 사용하며, 알고리즘·Feature·Target·하이퍼파라미터를 별도로 재정의하지 않는다.

* **`Validation Results`**

  * 각 Held-out Bearing의 Prefix Prediction을 수집한 LOBO 검증 결과
  * 검증 성능 지표: RMSE, MAE, SMAPE, R², IEEE PHM Score

---

### 3.2. [06 단계] Test Inference & Ensemble

05단계에서 확정된 LOBO 모델 집합을 Test 데이터에 적용하고, 독립적인 앙상블 정책에 따라 최종 예측값을 산출하는 단계다.

#### 입력 (Input)

* `Model Bundle`: 05단계의 산출물이며, 06단계는 이를 모델 정의의 유일한 기준으로 사용한다.
* `model_ready_test_dataset`: Target이 없는 추론용 Test 데이터.
* `EnsembleConfig`: 예측 통합 연산 정책.
* `external_test_ground_truth`: 선택적 입력. 최종 성능 평가를 위한 외부 실제 RUL 정답지.

#### 처리 (Processing)

* **Per-LOBO Prediction:** Test 입력에 대해 K개 모델이 각각 예측하여 Prediction Matrix ((N \times K))를 생성한다.
* **Intra-Config Ensemble:** `EnsembleConfig`에 따라 동일 `ModelingConfig`에서 생성된 K개 예측값을 하나로 통합한다.
* **RUL Reconstruction:** 최종 `RUL_norm` 예측값을 각 샘플의 `elapsed_sec`와 RUL Target 정의에 따라 초 단위 `RUL_sec`로 복원한다.
* **Optional Test Evaluation:** 외부 Ground Truth가 제공된 경우에만 최종 `RUL_sec` 예측값을 평가한다.

#### EnsembleConfig

`method`와 `weights`는 서로 독립적인 설정이다.

* `method`

  * `arithmetic`: 산술평균
  * `harmonic`: 조화평균
* `weights`

  * `None`: 단순 평균
  * 명시적 가중치 배열: 가중 평균

따라서 지원하는 조합은 다음과 같다.

| Method       | Weights | 통합 방식   |
| ------------ | ------- | ------- |
| `arithmetic` | `None`  | 단순 산술평균 |
| `arithmetic` | 지정      | 가중 산술평균 |
| `harmonic`   | `None`  | 단순 조화평균 |
| `harmonic`   | 지정      | 가중 조화평균 |

#### 출력 (Output)

* **`Final Prediction`**: Test Dataset에 대한 최종 통합 `RUL_sec` 예측값.
* **`Test Metrics`**: 외부 Ground Truth가 제공된 경우에 한해 계산된 RMSE, MAE, SMAPE, R², IEEE PHM Score.

---

## 4. 유틸리티(Utility) 모듈 설계

Notebook은 흐름(Orchestration)만 제어하며, 핵심 로직은 다음 Utility에 분리한다.

### 1. Modeling Utility (`model_utils`)

* Algorithm Factory 구현 및 모델 생성
* LOBO 분할 및 모델 학습
* Prefix Validation Prediction
* `Model Bundle` 패키징

### 2. Ensemble Utility (`ensemble_utils`)

* (N \times K) Prediction Matrix 검증 및 처리
* `method`와 `weights` 조합에 따른 예측 통합

  * 단순 산술평균
  * 가중 산술평균
  * 단순 조화평균
  * 가중 조화평균
* 가중치 배열과 Ordered LOBO Model의 대응 검증

### 3. RUL Reconstruction Utility

* `RUL_norm` 예측값과 `elapsed_sec`를 사용한 `RUL_sec` 복원
* 평가 이전에 모델 출력 단위를 실제 RUL 평가 단위로 변환

### 4. Evaluation Utility (`eval_utils`)

* 동일한 초 단위 `RUL_sec` Actual과 Prediction을 입력받아 성능 평가
* RMSE
* MAE
* SMAPE
* R²
* IEEE PHM 2012 Accuracy Score
* 05단계 Validation과 06단계 Test Evaluation에서 공통 사용

---

## 5. 핵심 불변 조건 (Invariants)

구현 과정에서 타협할 수 없는 절대 제약사항은 다음과 같다.

1. **앙상블 범위 제한:** 앙상블은 동일한 `ModelingConfig`로 생성된 LOBO 모델 간(Intra-Config)에만 수행하며, Cross-Config 혼합 앙상블은 지원하지 않는다.

2. **모델 구성의 불변성:** 06단계에서는 모델 재학습이나 알고리즘, Feature, Target, 하이퍼파라미터 변경을 수행하지 않고 `Model Bundle`을 유일한 모델 정의 기준으로 사용한다.

3. **가중치의 명시적 매핑:** 가중치 배열을 사용할 경우 `Model Bundle`의 Ordered LOBO Model 및 Fold Metadata와 1:1로 대응해야 하며, 모델 수와 동일한 길이를 가져야 한다. 세부 수치 유효성 검증은 Utility 계약에서 정의한다.

4. **Test 데이터 무결성:** Test Dataset 파일 자체에 Target 컬럼을 추가하지 않으며, 평가는 외부 Ground Truth를 별도로 로드하여 수행한다.

5. **평가 단위의 불변성:** 모델의 학습·예측 Target이 `RUL_norm`인 경우에도 RMSE, MAE, SMAPE, R² 및 IEEE PHM Score는 직접 `RUL_norm`을 평가하지 않는다. 예측값은 `elapsed_sec`와 Target 정의에 따라 `RUL_sec`로 복원한 뒤 실제 `RUL_sec`와 동일 단위에서 평가한다.

6. **05 검증의 비앙상블성:** 05단계의 각 Validation 샘플은 해당 Bearing을 Held-out한 LOBO 모델의 예측으로만 평가하며, Validation 단계에서 LOBO 모델 간 예측 통합을 수행하지 않는다.