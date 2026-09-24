# Execution_Output_Path_Policy

## 1. 목적

본 문서는 IEEE-PHM-2012 Bearing RUL Prediction 프로젝트의 각 단계에서 생성되는 **프로그램 실행 산출물의 저장 위치와 관리 원칙**을 정의한다.

본 방침에서 말하는 '산출물'은 노트북 또는 프로그램의 실행을 통해 생성되는 데이터 파일, 설정 파일, 모델 파일, 평가 결과 등의 **실행 결과물**을 의미한다.

연구 결과 보고서, 설명 문서, 설계 문서 등의 문서류는 본 방침의 대상에서 제외한다. 문서의 저장 및 정리는 별도의 문서 관리 기준에서 다룬다.

---

## 2. 기본 원칙

### 2.1 단계별 내부 산출물은 `outputs/`에서 관리한다

파이프라인의 각 단계에서 생성되는 실행 산출물은 기본적으로 프로젝트 루트의 `outputs/` 아래에 단계별 디렉터리를 두어 관리한다.

```text
outputs/
├─ 02_feature_transformation/
├─ 03_eda_selection/
├─ 04_dataset_builder/
├─ 05_modeling/
└─ 06_test_inference/
```

이를 통해 프로젝트 루트에 실행 결과 파일이 무분별하게 축적되는 것을 방지하고, **파이프라인의 단계별 책임과 산출물을 디렉터리 구조에 직접 반영**한다.

### 2.2 프로젝트의 대표 데이터셋은 루트에 둘 수 있다

모든 실행 산출물을 일률적으로 `outputs/` 아래에 넣지는 않는다.

01단계에서 생성되는 Base Feature Dataset은 원시 센서 데이터를 이후 분석에 사용할 수 있는 정형 데이터셋으로 변환한 것으로, 단순한 일회성 중간 파일이 아니라 **프로젝트 전체에서 독립적인 의미를 가지는 기초 데이터 자산**이다.

따라서 다음 파일은 프로젝트 루트에 유지한다.

```text
learning_set_features.csv
test_set_features.csv
```

이는 01단계 산출물을 특별히 예외 처리하는 것이 아니라, **프로젝트의 대표적인 기초 데이터셋을 루트에서 직접 확인하고 재사용할 수 있도록 한다는 원칙**에 따른 것이다.

### 2.3 최종 제출 산출물은 루트에 둔다

06단계의 최종 제출용 산출물인 `submission.csv`는 프로젝트 루트에 단독으로 둔다.

```text
submission.csv
```

06단계에서는 제출용 산출물과 연구 내부에서 사용하는 검증·분석용 산출물을 구분한다.

```text
06 Test Inference & Ensemble
├─ 제출 산출물
│  └─ submission.csv
│
└─ 내부 산출물
   ├─ lobo_test_predictions.csv
   ├─ test_rul_comparison.csv
   ├─ test_metrics.json
   └─ ensemble_config.json
```

`submission.csv`는 파이프라인 내부에서 다음 단계가 소비하는 중간 데이터가 아니라 **외부 제출을 목적으로 생성되는 최종 실행 산출물**이므로 루트에서 바로 확인할 수 있도록 한다.

반면 내부 산출물은 `outputs/06_test_inference/`에서 관리한다.

---

## 3. 확정된 산출물 구조

전체 실행 산출물의 구조는 다음과 같이 확정한다.

```text
project/
│
├─ learning_set_features.csv
├─ test_set_features.csv
├─ submission.csv
│
└─ outputs/
   │
   ├─ 02_feature_transformation/
   │  ├─ learning_engineered_features.csv
   │  └─ test_engineered_features.csv
   │
   ├─ 03_eda_selection/
   │  ├─ eda_evidence/
   │  │  ├─ feature_metrics.csv
   │  │  ├─ feature_redundancy.csv
   │  │  ├─ rul_correlation.csv
   │  │  ├─ rul_correlation_by_bearing.csv
   │  │  ├─ data_quality.csv
   │  │  ├─ bearing_profile.csv
   │  │  └─ figures/
   │  │
   │  └─ selection_config.json
   │
   ├─ 04_dataset_builder/
   │  ├─ model_ready_learning_dataset.csv
   │  ├─ model_ready_test_dataset.csv
   │  ├─ prefix_definition.csv
   │  └─ dataset_config.json
   │
   ├─ 05_modeling/
   │  ├─ model_bundle.pkl
   │  ├─ lobo_validation_predictions.csv
   │  ├─ lobo_bearing_metrics.csv
   │  ├─ validation_summary.json
   │  ├─ calibration_results.json
   │  └─ calibrated_validation_predictions.csv
   │
   └─ 06_test_inference/
      ├─ lobo_test_predictions.csv
      ├─ test_rul_comparison.csv
      ├─ test_metrics.json
      └─ ensemble_config.json
```

실제 파일 목록은 각 단계의 구현 및 산출물 생성 방식에 따라 위 원칙에 맞게 유지한다.

---

## 4. 단계별 산출물 관리 원칙

### 4.1 01 Generate Features

01단계의 대표 산출물인 Base Feature Dataset은 루트에 둔다.

```text
learning_set_features.csv
test_set_features.csv
```

이 데이터셋은 원시 센서 데이터를 프로젝트의 후속 처리에 적합한 정형 데이터셋으로 변환한 결과이며, 이후 단계의 입력이 되는 프로젝트의 기초 데이터 자산이다.

### 4.2 02 Feature Transformation

Feature Transformation 및 RUL Target Generation의 결과는 02단계의 내부 산출물로 관리한다.

```text
outputs/02_feature_transformation/
├─ learning_engineered_features.csv
└─ test_engineered_features.csv
```

01단계의 Base Feature Dataset과 달리 이 파일들은 이후 단계의 처리를 위한 **파이프라인 내부의 가공 데이터셋**으로 취급한다.

### 4.3 03 EDA & Feature Selection

EDA에서 생성되는 분석 근거와 Feature/Target 선택 결과를 03단계 디렉터리에서 관리한다.

```text
outputs/03_eda_selection/
├─ eda_evidence/
│  ├─ ...
│  └─ figures/
└─ selection_config.json
```

EDA evidence는 분석 과정에서 생성된 근거 자료이고, `selection_config.json`은 그 분석을 바탕으로 확정된 선택 결과를 저장한다. 둘은 성격이 다르지만 모두 03단계의 실행 산출물이므로 동일한 단계 디렉터리에서 관리한다.

### 4.4 04 Dataset Builder

모델 학습에 직접 사용되는 데이터셋과 데이터 구성 정보를 04단계 디렉터리에서 관리한다.

```text
outputs/04_dataset_builder/
├─ model_ready_learning_dataset.csv
├─ model_ready_test_dataset.csv
├─ prefix_definition.csv
└─ dataset_config.json
```

### 4.5 05 Modeling & LOBO Validation

모델 학습, LOBO Validation 및 Calibration에서 생성되는 결과는 기존의 05단계 산출물 구조를 유지한다.

```text
outputs/05_modeling/
```

특히 `model_bundle.pkl`은 06단계가 사용하는 공식적인 단계 간 인터페이스이므로 05단계 산출물로 관리한다.

### 4.6 06 Test Inference & Ensemble

06단계의 내부 실행 결과는 다음 디렉터리에서 관리한다.

```text
outputs/06_test_inference/
```

여기에는 테스트 예측, 평가 결과, 앙상블 설정 등 연구 내부에서 필요한 산출물을 저장한다.

최종 제출용 `submission.csv`만 프로젝트 루트에 둔다.

---

## 5. 루트 산출물의 의미

따라서 프로젝트 루트에 존재하는 실행 산출물은 다음과 같이 해석한다.

```text
learning_set_features.csv
test_set_features.csv
        │
        └─ 프로젝트의 기초 데이터 자산

submission.csv
        │
        └─ 프로젝트의 최종 외부 제출 산출물
```

즉, 루트는 **파이프라인 내부 산출물을 모아놓는 공간이 아니라 프로젝트의 시작과 끝을 대표하는 실행 산출물을 보여주는 공간**으로 사용한다.

반대로 `outputs/`는 프로젝트의 내부 처리 과정을 단계별로 추적하기 위한 공간이다.

```text
루트
├─ Base Feature Dataset       ← 파이프라인의 시작점
├─ submission.csv             ← 파이프라인의 최종 외부 결과
│
└─ outputs/
   ├─ Stage 02                ← 내부 가공
   ├─ Stage 03                ← 내부 분석·선택
   ├─ Stage 04                ← 내부 데이터 구성
   ├─ Stage 05                ← 내부 모델링·검증
   └─ Stage 06                ← 내부 테스트·앙상블
```

이 구조는 **가시성**과 **단계별 관리**를 동시에 확보하는 것을 목적으로 한다.