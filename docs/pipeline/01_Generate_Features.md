# 01_Generate_Features

---

## 1. 단계 개요

01단계 **Generate_Features**는 IEEE PHM 2012 Bearing Dataset의 원시 센서 파일을 읽어 개별 측정 시점 단위의 **Base Feature Dataset**으로 변환하는 단계이다.

전체 데이터 처리 흐름에서 01단계의 위치는 다음과 같다.

```text
Sensor Dataset
      ↓
01 Generate_Features
      ↓
Base Feature Dataset
      ↓
02 Feature Transformation / RUL Target Generation
      ↓
Engineered Dataset
      ↓
03 EDA & Feature Selection
      ↓
04 Dataset Builder
      ↓
Model-ready Dataset
```

01단계의 핵심 책임은 **센서 데이터의 파일 단위 표현을 모델링에 사용할 수 있는 행 단위 Feature 데이터로 변환하는 것**이다.

따라서 이 단계에서는 센서 신호로부터 직접 계산할 수 있는 기본 Feature와 측정 시점 및 운전조건에 관한 Metadata를 생성한다.

반면 다음 작업은 01단계의 책임에 포함하지 않는다.

* RDI 및 기타 Damage Transformation
* RUL Target Generation
* EDA
* Feature Ranking
* Feature Selection
* Target Selection
* Model-ready Dataset 구성
* 모델 학습 및 검증

01단계와 이후 단계의 책임 경계는 현재 파이프라인의 데이터 계약을 기준으로 한다. 실제 02단계는 01단계의 결과를 **Base Feature Dataset**으로 입력받아 Feature Transformation과 RUL Target Generation을 수행한다.

---

# 2. 입력 데이터

## 2.1 센서 데이터 구성

01단계는 Bearing별 디렉터리에 저장된 센서 파일을 직접 읽는다.

주요 입력은 다음 두 종류이다.

* `acc_*.csv`: 진동 데이터
* `temp_*.csv`: 온도 데이터

IEEE PHM 2012 데이터셋에서 진동은 서로 직교하는 두 개의 가속도 센서로 측정되며, 가속도 데이터의 샘플링 주파수는 25.6 kHz이다. 온도는 별도의 센서로 측정되며 샘플링 주파수는 0.1 Hz이다.

01단계 구현에서는 센서 파일의 구분자를 첫 번째 행에서 판별하여 `;` 또는 `,`를 사용하도록 처리한다. 파일에는 별도의 Header를 사용하지 않는다.

---

## 2.2 운전조건 정보

Bearing 이름으로부터 운전조건을 식별하고, 현재 구현에 정의된 `CONDITION_MAP`을 이용하여 `condition`, `rpm`, `load`를 부여한다.

현재 운전조건은 다음과 같다.

| Condition |  RPM |   Load |
| --------- | ---: | -----: |
| C1        | 1800 | 4000 N |
| C2        | 1650 | 4200 N |
| C3        | 1500 | 5000 N |

이는 데이터셋의 세 가지 운전조건을 01단계의 Metadata로 명시적으로 보존하기 위한 것이다.

---

# 3. 데이터 처리 단위와 시간축

01단계에서 최종 Dataset의 한 행은 **하나의 가속도 측정 파일에 대응하는 하나의 관측 시점**을 나타낸다.

각 `acc_*.csv` 파일에서 다음 정보를 추출한다.

* `file_idx`
* `hour`
* `minute`
* `second`

`file_idx`는 파일명에서 숫자를 추출하여 구성한다.

경과 시간은 다음과 같이 계산한다.

```text
elapsed_sec = (file_idx - 1) × 10
```

따라서 각 가속도 파일은 10초 간격의 관측 시점으로 표현되며, 이 값을 이후 단계에서 Bearing별 시간축의 기준으로 사용한다.

`hour`, `minute`, `second`는 원본 센서 파일의 첫 번째 측정 행에서 추출한다.

---

# 4. 진동 신호 처리

## 4.1 원본 진동 신호

각 가속도 파일에는 두 개의 직교 방향 진동 신호가 존재한다.

현재 구현에서는 다음과 같이 사용한다.

```text
H signal → column 5
V signal → column 6
```

즉, 데이터프레임의 zero-based index 기준으로 각각 `4`, `5`번째 열을 사용한다.

두 신호는 서로 다른 방향의 진동 정보를 가지므로, 01단계에서는 이들을 단일 방향의 단순 선택으로 축약하지 않고 두 가지 방식으로 표현한다.

1. Vector Magnitude
2. Principal Axis Projection

---

## 4.2 Vector Magnitude

두 방향의 진동 신호를 다음과 같이 결합한다.

$$
a_{\mathrm{vector}}(t)
=
\sqrt{a_H(t)^2+a_V(t)^2}
$$

이를 통해 센서 방향에 독립적인 진동 크기 표현을 구성한다.

현재 구현에서는 Vector Magnitude를 이용하여 다음 Feature를 계산한다.

* `rms`
* `cf`

즉, 진동 크기의 전반적인 수준과 peak 대비 RMS 관계를 나타내는 Feature는 Vector Magnitude를 기준으로 계산한다.

---

## 4.3 Principal Axis Projection

두 방향의 진동 신호에 존재하는 주된 변동 방향을 추출하기 위해 Principal Axis Projection을 수행한다.

처리 과정은 다음과 같다.

```text
H / V signal
      ↓
중심화
      ↓
2×2 covariance matrix
      ↓
eigendecomposition
      ↓
최대 eigenvalue의 eigenvector 선택
      ↓
Principal Signal Projection
```

구체적으로 H/V 신호를 2차원 벡터로 구성하고 평균을 제거한 뒤 covariance matrix를 계산한다. 이후 eigenvalue가 가장 큰 eigenvector를 주축으로 선택하여 신호를 해당 축으로 projection한다.

이 과정에서 주축의 방향과 두 축에 대한 설명력을 나타내는 다음 값도 계산한다.

* `principal_axis_cos`
* `principal_axis_sin`
* `principal_energy_ratio`

`principal_energy_ratio`는 선택된 주축의 eigenvalue를 전체 eigenvalue의 합으로 나눈 값이다.

Principal Signal은 이후 시간영역 및 주파수영역 Feature 계산의 입력으로 사용한다.

---

# 5. Base Feature 추출

01단계의 Feature는 크게 다음 네 범주로 구성된다.

```text
Base Features
├── Time-domain Features
├── Frequency-domain Features
├── Vibration-derived Features
└── Temperature Features
```

이들은 모두 원시 센서 데이터로부터 직접 계산되는 Feature이며, 02단계에서 수행하는 Damage Transformation과 구별된다.

---

## 5.1 시간영역 Feature

Principal Signal을 대상으로 다음 시간영역 Feature를 계산한다.

| Feature | 의미           |
| ------- | ------------ |
| `p2p`   | 최대값과 최소값의 차이 |
| `skew`  | 왜도           |
| `kurt`  | 첨도           |

현재 구현의 `calculate_time_features()`는 다음과 같이 계산한다.

$$
p2p=\max(x)-\min(x)
$$

그리고 `scipy.stats`의 `skew` 및 `kurtosis`를 사용한다. Kurtosis는 Fisher 보정 없이 계산한다.

---

## 5.2 주파수영역 Feature

Principal Signal에 대해 FFT를 수행하여 주파수영역 Feature를 계산한다.

현재 구현에서 FFT sampling frequency는 25,600 Hz로 설정되어 있다.

생성되는 Feature는 다음과 같다.

| Feature           | 설명                    |
| ----------------- | --------------------- |
| `dom_freq`        | 지배 주파수                |
| `centroid`        | 중심 주파수                |
| `spectral_spread` | 주파수 분산                |
| `entropy`         | Spectral Entropy      |
| `spectral_crest`  | Spectral Crest Factor |
| `rcr`             | RMS-to-Centroid Ratio |
| `spectral_gini`   | Spectral Gini Index   |

FFT 결과로 amplitude spectrum과 power spectrum을 구성한 뒤 각 Feature를 계산한다.
특히 주파수영역 Feature는 특정 주파수 하나를 선택하는 목적보다는 진동 신호의 주파수 분포와 집중도, 복잡성 및 비균질성을 서로 다른 관점에서 표현하기 위해 구성되어 있다.

---

## 5.3 진동 파생 Feature

Vector Magnitude와 Principal Axis Projection 과정에서 직접 계산되는 값도 Base Feature로 보존한다.

### Vector Magnitude 기반

* `rms`
* `cf`

RMS는 신호의 전반적인 진동 수준을 나타내며, Crest Factor는 최대 절대 진폭과 RMS의 비율로 계산한다. RMS가 0인 경우 현재 구현에서는 Crest Factor를 1로 처리한다.

### Principal Axis 기반

* `principal_axis_cos`
* `principal_axis_sin`
* `principal_energy_ratio`

이 값들은 두 가속도 축의 상대적인 주축 방향과 해당 주축이 설명하는 에너지 비율을 보존한다.

---

# 6. 온도 Feature

온도 데이터는 가속도 데이터의 관측 시점과 연결하여 Feature로 통합한다.

## 6.1 Temperature Lookup

각 Bearing 디렉터리에서 `temp_*.csv` 파일을 읽어 온도 정보를 별도의 Lookup Table로 구성한다.

온도 파일의 첫 번째 행에서 `hour`, `minute`을 읽고, 온도 데이터 열을 이용하여 다음 Feature를 계산한다.

* `temp_mean`
* `temp_max`
* `temp_slope`

`temp_slope`는 해당 온도 파일의 마지막 값과 첫 번째 값의 차이로 계산한다.

---

## 6.2 가속도 관측과의 결합

가속도 파일의 `(hour, minute)`을 Key로 사용하여 해당 시점의 Temperature Feature를 Lookup한다.

따라서 한 관측 행은 다음과 같은 구조를 갖는다.

```text
ACC observation
    │
    ├── vibration features
    │
    ├── timestamp
    │
    └── (hour, minute)
             │
             ↓
      Temperature Lookup
             │
             └── temp_mean
                 temp_max
                 temp_slope
```

해당 시점의 온도 파일을 찾을 수 없는 경우 현재 구현에서는 세 온도 Feature를 `NaN`으로 기록한다. 온도 파싱 자체에 실패한 파일도 별도로 출력하여 확인할 수 있도록 되어 있다.

따라서 01단계는 온도 결측을 임의의 값으로 대체하지 않는다. 결측 자체를 Base Feature Dataset에 보존하고, 이후 단계에서 허용되는 결측 정책에 따라 처리한다.

현재 04단계의 Dataset validation에서도 온도 Feature의 NaN은 허용 가능한 Feature 결측으로 별도 취급한다.

---

# 7. Metadata 구성

01단계에서 생성되는 각 관측 행은 Feature뿐 아니라 이후 단계에서 데이터의 시간적·물리적 의미를 유지하기 위한 Metadata를 포함한다.

현재 파이프라인의 `META_COLUMNS`는 다음과 같다.

| Metadata      | 역할               |
| ------------- | ---------------- |
| `bearing`     | Bearing 식별자      |
| `condition`   | 운전조건 식별자         |
| `rpm`         | 회전 속도            |
| `load`        | 하중               |
| `file_idx`    | 원본 가속도 파일 순번     |
| `hour`        | 관측 시각 — 시        |
| `minute`      | 관측 시각 — 분        |
| `second`      | 관측 시각 — 초        |
| `elapsed_sec` | 시작 시점으로부터의 경과 시간 |

이 Metadata는 Feature와 별도로 보존되며, 이후 단계에서 시간 정렬, Bearing별 분석, 운전조건별 분석, RUL Target Generation 및 Prefix 구성 등에 사용된다.

특히 `elapsed_sec`는 단순 표시용 시간이 아니라 이후 파이프라인에서 Bearing별 시간축을 정의하는 핵심 변수이다.

---

# 8. Learning과 Test의 처리

01단계의 Generate_Features는 Learning과 Test 모두에 적용된다.

개념적으로 두 데이터셋은 동일한 변환 구조를 따른다.

```text
Sensor Dataset
      ↓
Sensor File Parsing
      ↓
Timestamp / Metadata
      ↓
Vibration Representation
      ↓
Base Feature Extraction
      ↓
Temperature Integration
      ↓
Base Feature Dataset
```

Learning과 Test의 차이는 01단계의 Feature 계산 방식에 있지 않다.

특히 01단계에서는 Learning 데이터의 결과를 이용하여 Test의 Feature를 사후적으로 생성하거나, Test의 데이터에 맞추어 별도의 Feature Selection을 수행하지 않는다.

Learning과 Test 모두 센서 데이터 자체로부터 동일한 종류의 Base Feature를 생성한다.

---

# 9. Base Feature Dataset의 의미

01단계의 최종 산출물은 **Base Feature Dataset**이다.

대표적인 출력은 다음과 같다.

```text
learning_set_features.csv
test_set_features.csv
```

이 Dataset은 각 행에 하나의 관측 시점을 나타내며 다음의 세 가지 성격을 함께 가진다.

```text
Base Feature Dataset
├── Metadata
├── Base Features
└── 관측 시점별 결측 정보
```

여기에는 아직 02단계에서 생성되는 RDI나 RUL Target이 존재하지 않는다.

현재 02단계는 `learning_set_features.csv`와 `test_set_features.csv`를 Base Feature Dataset으로 읽어들이고, 이후 Feature Transformation과 Target Generation을 수행한다.

---

# 10. Feature와 Metadata의 경계

01단계의 출력에서 모든 수치형 열이 동일한 의미를 가지는 것은 아니다.

현재 파이프라인은 다음과 같은 의미적 분류를 사용한다.

```text
Dataset
├── Metadata
├── Target Candidates
└── Feature Candidates
```

01단계 종료 시점에는 Target Candidates가 아직 생성되지 않는다.

따라서 01단계에서 Metadata를 제외한 나머지 생성값은 이후 단계에서 Feature Candidate로 취급할 수 있는 Base Feature 영역을 구성한다.

02단계에서는 `META_COLUMNS`를 기준으로 Metadata를 분리하고, Base Feature를 대상으로 Transformation 대상을 확인한다.

이 구조를 통해 01단계는 **어떤 Feature가 최종 모델에 사용될지를 결정하는 단계가 아니라, 이후 분석과 선택의 대상이 될 Base Feature 후보를 충분히 생성하는 단계**로 정의된다.

---

# 11. 02단계와의 책임 경계

01단계와 02단계의 가장 중요한 경계는 **Raw/Base Feature와 Derived Feature의 구분**이다.

01단계:

```text
Sensor Signal
    ↓
Base Feature
```

02단계:

```text
Base Feature
    ↓
Shock
    ↓
Rectified
    ↓
Cumulative
    ↓
RDI
```

현재 02단계에서 기본 Transformation 대상은 `rms`, `p2p`, `cf`, `kurt`이며, 이들은 01단계에서 생성된 Raw/Base Feature이다.

02단계의 Rectified Feature와 RDI는 01단계의 산출물이 아니다.

또한 02단계에서는 Learning에 대해서만 다음 Target Candidate를 생성한다.

* `life_ratio`
* `RUL_sec`
* `RUL_norm`

Test에는 Target Generation을 수행하지 않는다.

따라서 01단계의 책임은 센서 데이터로부터 **후속 Transformation과 분석에 필요한 출발점**을 제공하는 데 있다.

---

# 12. 03단계 및 04단계와의 관계

01단계는 Feature의 최종 사용 여부를 결정하지 않는다.

02단계를 거친 후 03단계에서 다음과 같은 분석이 수행된다.

* 데이터 품질 및 분포
* 운전조건별 안정성
* Bearing별 열화 추세
* Monotonicity
* Trendability
* Prognosability
* Feature 간 상관 및 중복성
* Feature와 Target 간 상관
* Feature Ranking

그리고 그 결과를 바탕으로 Feature Selection이 이루어진다.

따라서 01단계에서 생성되는 Feature의 존재 자체는 최종 모델 사용을 의미하지 않는다.

03단계에서 후보 Feature를 평가하고 선택한 뒤, 04단계가 선택된 Feature와 Target을 이용하여 Model-ready Dataset을 구성한다. 04단계는 선택되지 않은 Feature가 이후 검증에 영향을 주지 않도록 Feature filtering 이후 validation을 수행한다.

---

# 13. 현재 구현의 처리 원칙

현재 01단계 구현은 다음 원칙을 따른다.

### 13.1 파일 단위 관측

각 `acc_*.csv`를 하나의 관측 시점으로 취급한다.

### 13.2 원본 시간정보 보존

파일 내부 timestamp와 파일 순번 기반 `elapsed_sec`를 모두 보존한다.

### 13.3 2축 진동정보의 복수 표현

H/V 진동을 단일 방향으로 임의 선택하지 않고,

* Vector Magnitude
* Principal Axis Projection

두 가지 관점으로 표현한다.

### 13.4 시간·주파수 영역의 병행 추출

Principal Signal에서 시간영역 및 주파수영역 Feature를 모두 생성한다.

### 13.5 운전조건 정보의 명시적 보존

Bearing별 운전조건을 `condition`, `rpm`, `load`로 기록한다.

### 13.6 온도정보의 시점 결합

온도 센서의 낮은 샘플링 빈도를 고려하여 `(hour, minute)` 기준으로 가속도 관측과 결합한다.

### 13.7 결측의 보존

온도 데이터가 없는 경우 임의의 대체값을 넣지 않고 `NaN`을 유지한다.

### 13.8 후속 분석과의 분리

Feature Transformation, Target Generation, EDA, Selection 및 Dataset Building은 01단계에서 수행하지 않는다.

---

# 14. 산출물 계약

01단계의 출력 계약은 다음과 같이 정리할 수 있다.

| 항목                | 내용                                                   |
| ----------------- | ---------------------------------------------------- |
| 입력                | Bearing별 원시 `acc_*.csv`, `temp_*.csv`                |
| 관측 단위             | 가속도 파일 1개 = 관측 행 1개                                  |
| 시간축               | `file_idx`, timestamp, `elapsed_sec`                 |
| Metadata          | Bearing, 운전조건, 시간정보                                  |
| 진동 표현             | Vector Magnitude, Principal Axis Projection          |
| Base Feature      | 시간영역, 주파수영역, 진동 파생 Feature                           |
| 온도 Feature        | `temp_mean`, `temp_max`, `temp_slope`                |
| 결측 정책             | 온도 미매칭 시 NaN 보존                                      |
| Target            | 생성하지 않음                                              |
| RDI               | 생성하지 않음                                              |
| Feature Selection | 수행하지 않음                                              |
| 주요 출력             | `learning_set_features.csv`, `test_set_features.csv` |

이 산출물은 02단계의 직접적인 입력이 되며, 02단계는 이를 검증하고 Transformation 및 Target Generation을 수행한다.

---

# 15. 단계 전체에서의 역할

현재 확정된 파이프라인에서 01단계의 역할은 다음 한 문장으로 요약할 수 있다.

> **01단계 Generate_Features는 원시 센서 파일에서 관측 시점별 Metadata와 Base Feature를 추출하여, 이후 Feature Transformation·RUL Target Generation·EDA·Feature Selection의 공통 입력이 되는 Base Feature Dataset을 생성한다.**

이를 데이터 흐름으로 표현하면 다음과 같다.

```text
┌──────────────────────────┐
│      Sensor Dataset      │
│                          │
│  acc_*.csv               │
│  temp_*.csv              │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   01 Generate_Features   │
│                          │
│  • File / Time Parsing   │
│  • Vibration Processing  │
│  • Vector Magnitude      │
│  • Principal Axis        │
│  • Time Features         │
│  • Spectral Features     │
│  • Temperature Features  │
│  • Metadata Construction │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Base Feature Dataset   │
│                          │
│  Metadata                │
│  Base Features           │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 02 Feature Transformation│
│    / RUL Target Generation│
└──────────────────────────┘
```

따라서 01단계는 최종적인 Feature를 선택하거나 모델 입력을 만드는 단계가 아니라, **센서 데이터의 물리적·시간적 정보를 손실하지 않으면서 후속 데이터 처리 단계가 사용할 수 있는 Base Feature 표현으로 변환하는 최초의 데이터 구성 단계**이다.