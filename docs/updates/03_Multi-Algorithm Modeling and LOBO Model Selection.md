# 03_Multi-Algorithm Modeling and LOBO Model Selection

## 1. 개요

본 문서는 05단계 Modeling과 06단계 Test Inference & Ensemble에 도입한 **복수 알고리즘 모델링 및 LOBO Model 선택 구조**를 정리한다.

기존에는 하나의 알고리즘으로 모든 LOBO Model을 학습하고 이를 06단계에서 일괄적으로 사용하는 구조였으나, 변경 후에는 여러 알고리즘으로 독립적인 LOBO Model Bundle을 생성하고 각 LOBO Model에 사용할 알고리즘을 개별적으로 선택할 수 있도록 구성하였다.

전체 흐름은 다음과 같다.

```text
복수 알고리즘별 Model Bundle 생성
        ↓
LOBO Model별 Algorithm Selection
        ↓
Mixed Prediction Matrix
        ↓
Model-specific Calibration
        ↓
Ensemble
        ↓
Final Test RUL
```

현재 실제 실험에는 RandomForest와 CatBoost를 사용하였다. LightGBM과 XGBoost는 공통 Model Factory에서 지원되지만, 이번 최종 실험 구성에는 포함하지 않았다.

---

## 2. 구조 변경

### 2.1 기존 구조

기존 05단계는 하나의 `ModelingConfig`를 기준으로 LOBO Model을 생성하고, 하나의 Model Bundle을 06단계에 전달하는 구조였다.

```text
ModelingConfig
      ↓
LOBO Training
      ↓
Model Bundle
      ↓
Stage 06 Test Inference
      ↓
Per-LOBO Test Prediction
      ↓
Ensemble
```

이 구조에서는 모든 LOBO Model이 동일한 알고리즘과 하이퍼파라미터를 사용한다.

그러나 LOBO에서는 각 Model이 서로 다른 Bearing을 Held-out 대상으로 하며, 학습에 사용되는 Bearing 조합도 달라진다. 따라서 모든 LOBO Model에 동일한 알고리즘을 적용하는 방식과 별개로, Model별 알고리즘을 선택할 수 있는 구조를 구성할 수 있다.

### 2.2 변경된 구조

변경 후에는 여러 알고리즘으로 각각 전체 LOBO Model을 생성한 뒤, 동일한 LOBO Model 구조를 기준으로 사용할 알고리즘을 선택한다.

```text
Algorithm A ─→ LOBO Models ─┐
                            ├→ LOBO Model별 Algorithm 선택
Algorithm B ─→ LOBO Models ─┘
                                      ↓
                           Mixed Prediction Matrix
                                      ↓
                         Model-specific Calibration
                                      ↓
                                   Ensemble
```

따라서 Algorithm의 선택 단위는 전체 Stage 05가 아니라 **LOBO Model**이 된다.

---

## 3. 복수 알고리즘 모델링

### 3.1 알고리즘 구성

05단계에서는 알고리즘별로 독립적인 `ModelingConfig`를 정의한다.

현재 실제 실험에는 다음 두 알고리즘을 사용하였다.

```python
MODEL_CONFIGS = {
    "RandomForest": ModelingConfig(
        algorithm="RandomForest",
        params={
            "n_estimators": 100,
            "max_depth": 12,
            "min_samples_leaf": 7,
            "random_state": 42,
            "n_jobs": -1,
        },
    ),
    "CatBoost": ModelingConfig(
        algorithm="CatBoost",
        params={
            "iterations": 100,
            "depth": 12,
            "min_data_in_leaf": 7,
            "learning_rate": 0.1,
            "random_seed": 42,
            "verbose": False,
        },
    ),
}
```

선택된 알고리즘 목록은 `MODEL_CONFIGS`에서 파생한다.

```python
SELECTED_ALGORITHMS = list(MODEL_CONFIGS)
```

이를 통해 알고리즘별 설정을 공통 인터페이스로 관리하면서도 각 알고리즘에 대해 독립적인 Model Bundle을 생성할 수 있다.

### 3.2 알고리즘별 Model Bundle

각 알고리즘은 독립적으로 전체 LOBO Training을 수행한다.

```text
RandomForest ModelingConfig
        ↓
RandomForest LOBO Training
        ↓
RandomForest Model Bundle

CatBoost ModelingConfig
        ↓
CatBoost LOBO Training
        ↓
CatBoost Model Bundle
```

각 Model Bundle은 기존과 동일하게 다음 정보를 포함한다.

* `ModelingConfig`
* `TrainingRange`
* `feature_columns`
* `target_column`
* Ordered LOBO models
* Fold metadata

알고리즘을 여러 개 사용하더라도 개별 Model Bundle의 계약은 변경하지 않는다. 기존의 단일 Bundle 구조가 알고리즘별 병렬 Bundle 구조로 확장된 것이다.

현재 주요 산출물은 알고리즘별 디렉터리에 분리된다.

```text
outputs/05_modeling/
├── RandomForest/
│   ├── model_bundle.pkl
│   ├── lobo_validation_predictions.csv
│   ├── lobo_bearing_metrics.csv
│   ├── validation_summary.json
│   └── ...
│
└── CatBoost/
    ├── model_bundle.pkl
    ├── lobo_validation_predictions.csv
    ├── lobo_bearing_metrics.csv
    ├── validation_summary.json
    └── ...
```

### 3.3 알고리즘별 LOBO Model 구성

각 알고리즘은 동일한 LOBO 구조를 독립적으로 수행하며, LOBO Model의 순서와 held-out bearing의 대응을 동일하게 유지한다.

| Model   | Held-out Bearing |
| ------- | ---------------- |
| Model 0 | Bearing1_1       |
| Model 1 | Bearing1_2       |
| Model 2 | Bearing2_1       |
| Model 3 | Bearing2_2       |
| Model 4 | Bearing3_1       |
| Model 5 | Bearing3_2       |

따라서 RandomForest와 CatBoost Bundle에서 동일한 Model index는 동일한 held-out bearing을 의미한다.

이 대응 관계는 이후 Model Selection에서 알고리즘을 선택하기 위한 기준이 된다.

---

## 4. LOBO Model별 알고리즘 선택

### 4.1 알고리즘 선택 설정

LOBO Model별 알고리즘 선택은 `ModelSelectionConfig`로 관리한다.

```python
@dataclass
class ModelSelectionConfig:
    default_algorithm: str = "RandomForest"
    overrides: dict[str, str] = field(default_factory=dict)

    def algorithm_for(self, held_out_bearing: str) -> str:
        return self.overrides.get(
            held_out_bearing,
            self.default_algorithm,
        )
```

`default_algorithm`은 모든 LOBO Model에 기본적으로 적용할 알고리즘을 지정한다.

`overrides`에는 특정 held-out bearing에 대해 기본 알고리즘과 다른 알고리즘을 적용할 경우 해당 bearing과 알고리즘의 대응을 기록한다.

예를 들어 다음과 같이 설정하면,

```python
ModelSelectionConfig(
    default_algorithm="RandomForest",
    overrides={
        "Bearing1_1": "CatBoost",
    },
)
```

다음과 같은 선택이 이루어진다.

```text
Model 0 → CatBoost
Model 1 → RandomForest
Model 2 → RandomForest
Model 3 → RandomForest
Model 4 → RandomForest
Model 5 → RandomForest
```

### 4.2 알고리즘 선택 단위

알고리즘 선택의 단위는 LOBO Model이며, 선택 기준은 해당 Model의 **held-out bearing**이다.

따라서 Test Inference 단계에서 Test Bearing마다 알고리즘을 새롭게 선택하는 것이 아니라, 05단계에서 생성된 LOBO Model과 held-out bearing의 대응을 이용하여 알고리즘을 결정한다.

이 구조에서는 동일한 Test endpoint에 대해 서로 다른 알고리즘으로 학습된 LOBO Model들이 각각 예측을 수행할 수 있으며, 최종적으로 선택된 Model들의 예측을 하나의 Matrix로 구성한다.

### 4.3 Mixed Prediction Matrix 구성

선택된 각 LOBO Model은 모든 Test endpoint를 예측한다.

실제 Prediction Matrix는 다음과 같이 **행을 Test endpoint, 열을 LOBO Model**로 구성한다.

```text
                 Model 0   Model 1   Model 2   Model 3   Model 4   Model 5
Test Endpoint 0     ...       ...       ...       ...       ...       ...
Test Endpoint 1     ...       ...       ...       ...       ...       ...
Test Endpoint 2     ...       ...       ...       ...       ...       ...
...
```

각 열은 하나의 LOBO Model을 나타내며, 해당 Model이 RandomForest인지 CatBoost인지는 Algorithm Selection 결과와 대응한다.

따라서 Prediction Matrix는 알고리즘 종류와 독립적인 형태를 유지하면서, 서로 다른 알고리즘으로 생성된 Model의 예측값을 함께 포함할 수 있다.

이후 이 Matrix를 기존 Ensemble 함수에 전달한다.

---

## 5. Test Inference 연계

### 5.1 Model별 Calibration 적용

Prediction Matrix의 각 열에는 해당 LOBO Model이 선택한 알고리즘에 대응하는 Calibration parameter를 적용한다.

```text
Prediction Matrix
       │
       ├─ Model 0 → CatBoost Calibration
       ├─ Model 1 → RandomForest Calibration
       ├─ Model 2 → RandomForest Calibration
       ├─ Model 3 → RandomForest Calibration
       ├─ Model 4 → RandomForest Calibration
       └─ Model 5 → RandomForest Calibration
       │
       ↓
Calibrated Prediction Matrix
```

따라서 알고리즘을 혼합하더라도 각 Model의 Calibration parameter는 해당 알고리즘의 결과와 연결된다. 서로 다른 알고리즘의 Calibration parameter를 교차하여 사용하지 않는다.

Raw Prediction은 덮어쓰지 않고 Calibrated Prediction과 별도로 유지하여 내부 비교와 평가에 사용할 수 있도록 한다.

### 5.2 Ensemble

현재 지원되는 Ensemble 방식은 기존과 동일하게 다음 두 가지이다.

* Weighted Arithmetic Mean
* Weighted Harmonic Mean

이번 변경에서는 Ensemble 계산 방법 자체를 변경하지 않고, **Ensemble에 입력되는 LOBO Model 집합의 구성을 확장**하였다.

기존에는 동일 알고리즘으로 생성된 LOBO Model들이 Ensemble의 입력이었다면, 변경 후에는 Algorithm Selection 결과에 따라 서로 다른 알고리즘으로 생성된 Model들이 하나의 Prediction Matrix를 구성하고 Ensemble에 입력된다.

```text
동일 알고리즘 LOBO Models
        ↓
Ensemble
```

에서

```text
Algorithm-selected LOBO Models
        ↓
Mixed Prediction Matrix
        ↓
Ensemble
```

으로 입력 구성이 확장되었다.

### 5.3 RUL 복원 및 Output

Ensemble 결과는 기존과 동일하게 `RUL_norm`을 기준으로 생성한 뒤 Test endpoint의 `elapsed_sec`와 Bearing의 failure-time 정보를 이용하여 `RUL_sec`으로 복원한다.

이후 기존 Test Evaluation 및 submission artifact 생성 절차를 유지한다.

따라서 복수 알고리즘 모델링과 LOBO Model 선택은 최종 RUL 복원 방식이나 submission 출력 형식을 변경하지 않는다.

---

## 6. 알고리즘 선택 실험

### 6.1 실험 구성

복수 알고리즘으로 LOBO Model을 생성한 이후, 실제 최종 예측에서 사용할 알고리즘 구성을 결정하기 위해 LOBO Model별 Single Override 실험을 수행하였다.

기준선은 모든 LOBO Model을 RandomForest로 구성한 `All RF`이다.

이후 한 번에 하나의 Model만 CatBoost로 변경하였다.

```text
All RF

Model 0 CB
Model 1 CB
Model 2 CB
Model 3 CB
Model 4 CB
Model 5 CB
```

둘 이상의 Model을 동시에 변경하는 조합은 이번 실험의 주된 비교 대상에서 제외하였다.

### 6.2 실험 결과

전체 실험 결과는 다음과 같다.

| Configuration |    RMSE |     MAE | SMAPE |      R² | PHM Score |
| ------------- | ------: | ------: | ----: | ------: | --------: |
| All RF        | 2561.62 | 1500.06 | 61.68 |  0.0521 |    0.3095 |
| Model 0 CB    | 2447.90 | 1286.02 | 57.25 |  0.1344 |    0.3596 |
| Model 1 CB    | 2685.21 | 1628.45 | 63.47 | -0.0416 |    0.2657 |
| Model 2 CB    | 2624.69 | 1580.32 | 62.31 |  0.0048 |    0.2149 |
| Model 3 CB    | 2566.75 | 1558.39 | 60.40 |  0.0483 |    0.1893 |
| Model 4 CB    | 2628.68 | 1544.71 | 61.17 |  0.0018 |    0.2459 |
| Model 5 CB    | 2631.47 | 1498.40 | 61.21 | -0.0003 |    0.3340 |

Model 0을 CatBoost로 변경한 경우에는 모든 비교 지표가 All RF 구성과 비교하여 개선되었다.

Model 5를 CatBoost로 변경한 경우에는 MAE와 PHM Score는 개선되었지만 RMSE와 R²는 악화되는 등 지표별 변화가 다르게 나타났다.

다른 Single Override 구성에서도 지표 변화가 일관되지 않았으며, 각 LOBO Model에 따라 CatBoost 적용 결과가 달랐다.

### 6.3 최종 선택

Single Override 실험을 바탕으로 최종 Test inference에서는 다음 구성을 사용한다.

```python
MODEL_SELECTION = ModelSelectionConfig(
    default_algorithm="RandomForest",
    overrides={
        "Bearing1_1": "CatBoost",
    },
)
```

따라서 최종 선택은 다음과 같다.

```text
Model 0 → CatBoost
Model 1 → RandomForest
Model 2 → RandomForest
Model 3 → RandomForest
Model 4 → RandomForest
Model 5 → RandomForest
```

05단계에서는 RandomForest와 CatBoost의 전체 LOBO Model Bundle을 모두 생성하고, 06단계에서 Algorithm Selection 결과에 따라 필요한 Model을 선택한다.

따라서 최종 Ensemble에는 CatBoost Bundle 전체가 사용되는 것이 아니라 CatBoost로 선택된 Model 0과 RandomForest로 선택된 Models 1–5가 사용된다.

---

## 7. 최종 구현 상태

최종 구현에서는 다음과 같은 Pipeline을 구성한다.

```text
                    Stage 05
                       │
        ┌──────────────┴──────────────┐
        ↓                             ↓
 RandomForest                    CatBoost
 Model Bundle                    Model Bundle
        │                             │
        └──────────────┬──────────────┘
                       ↓
                    Stage 06
                       ↓
              Model Selection
                       ↓
        LOBO Model별 Algorithm Mapping
                       ↓
             Test Prediction
                       ↓
          Mixed Prediction Matrix
                       ↓
          Model-specific Calibration
                       ↓
                  Ensemble
                       ↓
              RUL_sec Reconstruction
                       ↓
              Evaluation / Output
```

구현 과정에서는 다음의 구조적 연결을 확인하였다.

* 알고리즘별 Model Bundle 생성
* 알고리즘별 LOBO Model 순서 보존
* LOBO Held-out Bearing과 Model index의 대응 유지
* 선택된 Algorithm과 Model index의 대응 검증
* 선택된 LOBO Model의 Test endpoint 예측
* Mixed Prediction Matrix 구성
* Algorithm별 Calibration artifact 연결
* Calibration 이후 Prediction domain 및 유효성 검증
* Raw / Calibrated Ensemble 결과의 병렬 보존
* 기존 RUL reconstruction 및 output 절차와의 연결

이에 따라 현재 Pipeline은 **복수 알고리즘으로 LOBO Model을 생성하고, LOBO Model별로 알고리즘을 선택한 뒤, 선택된 Model을 Mixed Prediction Matrix, Model-specific Calibration, Ensemble로 연결하는 구조**로 구현되어 있다.