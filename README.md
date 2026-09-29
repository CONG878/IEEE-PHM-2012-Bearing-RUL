# IEEE-PHM-2012-Bearing-RUL

This project develops an end-to-end bearing Remaining Useful Life (RUL) prediction pipeline using the PRONOSTIA dataset from the IEEE PHM 2012 Data Challenge. RUL is modeled as a normalized regression target and reconstructed in seconds at inference time, while feature selection is supported by exploratory evidence and model generalization is evaluated through Leave-One-Bearing-Out (LOBO) validation. Post-hoc calibration and LOBO-model ensemble methods are applied during test inference.

## 프로젝트 개요

이 프로젝트는 IEEE PHM 2012 Data Challenge의 PRONOSTIA Bearing Dataset을 대상으로 Bearing Remaining Useful Life (RUL) 예측을 수행한다.

전체 pipeline은 원시 센서 데이터로부터 특징을 생성하고 변환하는 단계에서 시작하여, EDA를 통한 근거 기반 특징 선택, model-ready dataset 구성, LOBO (Leave One Bearing Out) validation, test inference 및 ensemble prediction으로 이어진다.

실행 가능한 Notebook과 각 단계의 기능 및 설계를 설명하는 Pipeline 문서는 서로 다른 명명 체계를 사용한다. Notebook은 수행 작업을, Pipeline 문서는 단계의 기능과 역할을 나타내도록 명명한다.

## 주요 방법론

### 1. RUL Target 및 Reconstruction

학습에서는 정규화된 잔여 수명인 `RUL_norm`을 regression target으로 사용한다.

Test inference에서는 관측된 누적 가동 시간 `elapsed_sec`과 예측된 `RUL_norm`을 결합하여 실제 시간 단위의 잔여 수명 `RUL_sec`으로 복원한다.

$$
RUL_{sec} = 
elapsed_{sec} \times
\frac{RUL_{norm}}
    {1 - RUL_{norm}}
$$

이를 통해 학습 단계에서는 bearing별 전체 수명 차이를 정규화된 target으로 다루고, 추론 단계에서는 관측 시점의 가동 시간을 반영한 실제 잔여 수명으로 변환한다.

### 2. Evidence-based Feature Selection

Feature selection은 단일 자동 ranking 결과에 의존하지 않는다.

EDA 단계에서 data quality, distribution, operating-condition stability, degradation trend, monotonicity, trendability, prognosability, redundancy 및 feature-target relationship 등을 검토하고, 이러한 분석 결과를 근거로 최종 feature set을 명시적으로 결정한다.

현재 선택된 feature set은 다음과 같다.

```text
RDI_cf
RDI_kurt
Rectified_rms
Rectified_p2p
temp_mean
```

### 3. LOBO Modeling and Model Selection

모델의 미지 bearing에 대한 일반화 성능은 **LOBO (Leave One Bearing Out)** validation을 중심으로 평가한다.

각 bearing을 차례로 hold-out하고 나머지 bearing으로 모델을 학습하여, 해당 bearing에 대한 예측 성능을 평가한다.

복수의 tree-based algorithm을 평가하며, LOBO 결과를 바탕으로 held-out bearing별 model selection을 수행할 수 있다.

### 4. EOL Boundary-Preserving Calibration

Test inference에서는 필요에 따라 post-hoc calibration을 적용한다.

Calibration은 예측값의 end-of-life (EOL) 영역에서 발생할 수 있는 경계 문제를 완화하고, $0 \le RUL_{norm} \le 1$ 범위와 경계 조건을 고려한 변환을 적용하기 위한 것이다.

Calibration은 Model Bundle 자체를 변경하지 않고 inference 단계의 후처리로 적용된다.

### 5. Ensemble Test Inference

Stage 06에서는 LOBO model bundle에서 생성된 개별 예측을 결합하여 최종 test prediction을 구성한다.

Ensemble은 arithmetic mean 또는 harmonic mean을 사용할 수 있으며, calibration을 적용한 예측과 적용하지 않은 예측을 내부 평가 과정에서 구분하여 관리할 수 있다.

## Dataset

본 프로젝트는 FEMTO-ST Institute가 IEEE PHM 2012 Data Challenge를 위해 제공한 **PRONOSTIA Bearing Dataset**을 사용한다.

현재 원래 FEMTO-ST 배포 경로는 직접 이용할 수 없으므로, 다음의 공개 mirror를 데이터 다운로드 경로로 사용한다.

* [IEEE PHM 2012 Data Challenge Dataset — Lucky-Loek](https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset)

원시 데이터셋은 용량을 고려하여 repository에 포함하지 않는다.

Challenge의 공식 설명 문서는 repository에 포함되어 있다.

* [IEEE PHM 2012 Challenge Details](docs/competition/IEEEPHM2012-Challenge-Details.pdf)

## Pipeline

각 단계에서는 실행 Notebook과 해당 단계의 기능 및 설계를 설명하는 Pipeline 문서를 별도로 제공한다.

| Stage  | Notebook                          | 주요 역할                                                  | Documentation                                                                                                                    |
| ------ | --------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| **01** | `01_Generate_Features.ipynb`      | Raw sensor data로부터 Base Feature Dataset 생성             | [Feature Generation](docs/pipeline/01_Feature%20Generation.md)                                                                   |
| **02** | `02_Feature_Transformation.ipynb` | Feature Transformation 및 Target Generation             | [Feature Transformation and Target Generation](docs/pipeline/02_Feature%20Transformation%20and%20Target%20Generation.md)         |
| **03** | `03_EDA_&_Selection.ipynb`        | EDA 및 evidence-based Feature Selection                 | [Exploratory Data Analysis and Feature Selection](docs/pipeline/03_Exploratory%20Data%20Analysis%20and%20Feature%20Selection.md) |
| **04** | `04_Dataset_Builder.ipynb`        | Model-ready Dataset 구성 및 Prefix Evaluation Scenario 정의 | [Dataset Construction](docs/pipeline/04_Dataset%20Construction.md)                                                               |
| **05** | `05_Modeling.ipynb`               | Model Training 및 LOBO Validation                       | [Modeling and Test Inference](docs/pipeline/05_06_Modeling%20and%20Test%20Inference.md)                                          |
| **06** | `06_Test_Inference.ipynb`         | Test Inference, Calibration 및 Ensemble Prediction      | [Modeling and Test Inference](docs/pipeline/05_06_Modeling%20and%20Test%20Inference.md)                                          |

### 실행 순서

데이터셋을 준비한 후 다음 순서로 Notebook을 실행한다.

```text
01_Generate_Features.ipynb
        ↓
02_Feature_Transformation.ipynb
        ↓
03_EDA_&_Selection.ipynb
        ↓
04_Dataset_Builder.ipynb
        ↓
05_Modeling.ipynb
        ↓
06_Test_Inference.ipynb
```

## Evaluation and Results

Validation 및 test evaluation에서는 `RUL_sec`으로 복원한 실제 시간 단위 RUL을 기준으로 평가 지표를 계산한다.

주요 평가지표는 다음과 같다.

* RMSE
* MAE
* SMAPE
* R²
* IEEE PHM 2012 Score

구체적인 예측 결과와 해석은 다음 문서에서 다룬다.

* [`예측 결과 및 해석`](docs/results/예측%20결과%20및%20해석.md)

연구의 적용 범위와 한계는 다음 문서에서 별도로 다룬다.

* [`연구의 한계`](docs/results/연구의%20한계.md)

최종 submission 파일은 repository root에 위치한다.

* [`submission.csv`](submission.csv)

## Updates

Pipeline 구조가 확립된 이후의 주요 변경 사항과 방법론적 업데이트는 별도의 Update 문서로 기록한다.

* [`01_Modeling Pipeline Update`](docs/updates/01_Modeling%20Pipeline%20Update.md)
* [`02_RUL Calibration Design and Application`](docs/updates/02_RUL%20Calibration%20Design%20and%20Application.md)
* [`03_Multi-Algorithm Modeling and LOBO Model Selection`](docs/updates/03_Multi-Algorithm%20Modeling%20and%20LOBO%20Model%20Selection.md)

Update 문서는 기존 Pipeline의 역할과 구조를 반복해서 설명하기보다, 이후 추가되거나 변경된 방법과 설계의 핵심을 기록하는 데 목적이 있다.

## Repository Structure

```text
IEEE-PHM-2012-Bearing-RUL/
├── 01_Generate_Features.ipynb
├── 02_Feature_Transformation.ipynb
├── 03_EDA_&_Selection.ipynb
├── 04_Dataset_Builder.ipynb
├── 05_Modeling.ipynb
├── 06_Test_Inference.ipynb
├── submission.csv
├── utils/
│   ├── bearing_utils.py
│   ├── dataset_utils.py
│   ├── eda_utils.py
│   ├── ensemble_utils.py
│   ├── eval_utils.py
│   ├── feature_utils.py
│   ├── model_utils.py
│   └── project_utils.py
└── docs/
    ├── competition/
    ├── pipeline/
    ├── updates/
    ├── results/
    └── project/
```

`utils/`에는 여러 Notebook에서 공통으로 사용하는 domain, feature, dataset, evaluation, ensemble 및 project-level 기능을 모듈별로 구성하였다.

## Documentation

### Competition

* [IEEE PHM 2012 Challenge Details](docs/competition/IEEEPHM2012-Challenge-Details.pdf)

### Pipeline

* [`01_Feature Generation`](docs/pipeline/01_Feature%20Generation.md)
* [`02_Feature Transformation and Target Generation`](docs/pipeline/02_Feature%20Transformation%20and%20Target%20Generation.md)
* [`03_Exploratory Data Analysis and Feature Selection`](docs/pipeline/03_Exploratory%20Data%20Analysis%20and%20Feature%20Selection.md)
* [`04_Dataset Construction`](docs/pipeline/04_Dataset%20Construction.md)
* [`05_06_Modeling and Test Inference`](docs/pipeline/05_06_Modeling%20and%20Test%20Inference.md)

### Updates

* [`01_Modeling Pipeline Update`](docs/updates/01_Modeling%20Pipeline%20Update.md)
* [`02_RUL Calibration Design and Application`](docs/updates/02_RUL%20Calibration%20Design%20and%20Application.md)
* [`03_Multi-Algorithm Modeling and LOBO Model Selection`](docs/updates/03_Multi-Algorithm%20Modeling%20and%20LOBO%20Model%20Selection.md)

### Results

* [`예측 결과 및 해석`](docs/results/예측%20결과%20및%20해석.md)
* [`연구의 한계`](docs/results/연구의%20한계.md)

### Project

* [`실행 산출물 경로 관리 방침`](docs/project/실행%20산출물%20경로%20관리%20방침.md)
* [`코드 구조화 원칙`](docs/project/코드%20구조화%20원칙.md)

## Citation

본 프로젝트에서 사용하는 PRONOSTIA dataset의 연구 및 파생 작업에서는 다음 원 논문을 함께 인용하는 것을 권장한다.

> P. Nectoux, R. Gouriveau, K. Medjaher, E. Ramasso, B. Morello, N. Zerhouni, and C. Varnier, “PRONOSTIA: An Experimental Platform for Bearings Accelerated Life Test,” *IEEE International Conference on Prognostics and Health Management*, Denver, CO, USA, 2012.
