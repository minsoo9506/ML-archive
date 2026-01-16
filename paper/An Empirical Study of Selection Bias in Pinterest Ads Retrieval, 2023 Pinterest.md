# An Empirical Study of Selection Bias in Pinterest Ads Retrieval, 2023 Pinterest

## 논문 정보:
*   **저자:** Yuan Wang, Peifeng Yin, Zhiqiang Tao, Hari Venkatesan, Jin Lai, Yi Fang, PJ Xiao
*   **학회:** KDD '23 (2023년 8월 6-10일)

## 핵심 내용:
이 논문은 핀터레스트(Pinterest)의 광고 랭킹 시스템 중 광고 검색(Ads Retrieval) 단계에서 발생하는 '데이터 선택 편향(data selection bias)' 문제를 다룹니다.

## 문제점 (선택 편향):
- 모델 학습에 사용되는 데이터(레이블이 있는 아이템)의 분포와, 실제 모델이 예측해야 하는 데이터(인퍼런스 단계에서 만나는 후보군)의 분포가 크게 다른 상황을 의미합니다. 이로 인해 모델 성능이 저하될 수 있습니다. 여기서 훈련 데이터는 retrieval model 의 훈련데이터를 의미합니다. 모델은 two-tower model 로 이해하면 됩니다.
- retrieval model -> ranking model -> auction model 순서로 추천하는데 상위랭크 아이템만 노출되고 이들로 훈련데이터가 만들어져서 retrieval model 을 훈련합니다. 이로 인해 selection bias 가 발생합니다.

## 선택 편향 확인 방법
- retrieval 이후 ranking candidate 아이템, ranking 이후 auction candidate 아이템, auction 이후 실제 노출 아이템 3가지 그룹들의 분포를 비교합니다.
  - retrieval 에서 주요 feature 들의 분포
  - ranking model prediction 값의 분포

## 해결 노력:
연구팀은 이 문제를 해결하기 위해 다음과 같은 최신 기법들을 평가했습니다.
*   전이 학습 (Transfer Learning)
*   적대적 학습 (Adversarial Learning)
*   비지도 도메인 적응 (Unsupervised Domain Adaptation)

## 제안 방법 (MUDA):
기존의 비지도 도메인 적응 방법을 개선한 **MUDA (Modified version of Unsupervised Domain Adaptation)** 라는 새로운 방법을 제안했습니다.

## 결과:
온라인 A/B 테스트 결과, MUDA는 다른 방법들이나 기존 운영 모델에 비해 핀터레스트 광고 랭킹 시스템의 성능을 가장 크게 향상시키는 것으로 나타났습니다.

## MUDA 방법론 상세

MUDA(Modified Unsupervised Domain Adaptation)는 기존의 **비지도 도메인 적응(UDA, Unsupervised Domain Adaptation)** 방법을 수정한 것입니다. 두 방법의 가장 큰 차이점은 **'가짜 레이블(pseudo label)'을 만드는 방식과 '손실 함수(loss function)'**에 있습니다.

### 1. 기존의 UDA (Unsupervised Domain Adaptation) 방식
- **가짜 레이블 생성**: 광고 **랭킹 모델의 예측값(연속적인 값)**을 그대로 가짜 레이블로 사용합니다.
- **손실 함수**: 예측값과 실제값의 차이를 측정하는 **LogMAE(Log Mean Absolute Error)**를 사용합니다.

### 2. MUDA (Modified Unsupervised Domain Adaptation) 방식
-  **가짜 레이블 생성**: 랭킹 모델의 예측값을 **'클릭' 또는 '클릭 안함'과 같은 이진 클래스(binary class)로 변환**하여 가짜 레이블로 사용합니다.
   -  이때, 확실한 값들만 사용하기 위해 2개의 threshold 를 두고 각 threshold 이하, 이상인 경우만 데이터로 사용
   -  threshold 는 예측값을 bucketing 하여 empirical ctr 의 변화를 토대로 결정
- **손실 함수**: 두 클래스를 분류하는 문제에 적합한 **BCE(Binary Cross-Entropy)**를 사용합니다.

### 핵심 요약
- 기존 UDA가 랭킹 점수를 직접 예측하는 '회귀(Regression)' 문제처럼 접근했다면, MUDA는 랭킹 점수를 바탕으로 '클릭 여부'를 예측하는 '분류(Classification)' 문제로 바꾸어 접근한 것입니다. 핀터레스트의 온라인 A/B 테스트에서 이 방식이 더 좋은 성능을 보였습니다.
- retrieval model -> ranking model -> auction model 순서로 추천하고 auction model 의 상위 예측이 유저에게 노출
  - 이 때, auction model 의 결과로 retrieval model 의 훈련데이터를 만든다. 최종적으로 노출된 경우가 true label, 그렇지 않지만 ranking prediction 이 pseudo label
  - retieval model 은 매일 훈련하여 최신성 유지

## 선택 편향 완화 원리 (Pseudo-Labeling)

가짜 레이블(Pseudo-Label)을 사용하는 것이 선택 편향을 완화하는 원리는 다음과 같습니다.

### 1. 선택 편향 문제
*   **편향된 학습 데이터 (Source Domain)**: 리트리벌 모델은 '과거에 사용자에게 노출되었던 광고' 데이터로 학습합니다. 이는 전체 광고 중 일부이며, 성능이 좋을 것으로 예상되는 광고 위주로 구성된 '편향된' 데이터입니다.
*   **편향 없는 실제 데이터 (Target Domain)**: 모델이 실제로 추천을 수행하는 환경에서는 '모든 광고'를 후보로 고려해야 합니다. 이 두 데이터 간의 분포 차이가 '선택 편향'입니다.

### 2. 가짜 레이블의 역할: 분포 맞추기
핵심 아이디어는 **"실제 추천 환경과 동일한 분포의 데이터로 모델을 학습시키는 것"**입니다.

*   **문제**: '모든 광고' 데이터에는 정답(레이블)이 없습니다.
*   **해결책 (가짜 레이블)**: 더 정교한 **'선생님 모델'(광고 랭킹 모델)**을 사용하여 '모든 광고' 데이터에 대한 예측값(클릭 확률 등)을 생성합니다. 이것이 '가짜 레이블'입니다.

### 3. 편향이 완화되는 이유
*   리트리벌 모델은 이제 편향된 과거 데이터가 아닌, **실제 환경과 동일한 분포를 가진 '모든 광고' 데이터셋**과 **'가짜 레이블'**을 가지고 학습합니다.
*   이를 통해 모델은 과거에 노출될 기회가 없었던 좋은 광고를 발견하는 법을 배우고, 특정 광고에 과적합되는 것을 피할 수 있습니다.

#### 비유
*   **편향된 학습**: '족보(기출문제)'만 보고 시험 준비.
*   **가짜 레이블 학습**: '선생님(랭킹 모델)이 모든 범위에서 내준 모의고사'를 풀어보는 것.

결론적으로, 가짜 레이블링은 모델의 학습 데이터 분포를 실제 서빙 환경의 데이터 분포와 일치시켜주므로 선택 편향 완화에 효과적입니다.

## Adversarial Regularization (적대적 학습)
논문에서 시도한 또 다른 방법인 '적대적 학습'은 두 개의 모델을 경쟁적으로 학습시켜 선택 편향을 완화하는 방식입니다. 논문에 상세한 설명은 없지만, 일반적인 방법론은 다음과 같습니다.

### 핵심 아이디어
1.  **메인 모델 (Generator)**: 리트리벌 모델. 좋은 광고를 찾는 것이 목표.
2.  **도메인 판별자 (Domain Discriminator)**: 주어진 광고가 편향된 학습 데이터(Source Domain)에서 왔는지, 편향 없는 전체 후보군(Target Domain)에서 왔는지 구별하는 모델.

### 학습 방식
*   **판별자**는 두 도메인을 더 잘 구별하도록 학습합니다.
*   **메인 모델**은 원래의 추천 성능을 높이는 동시에, **판별자를 속이도록** 학습합니다. 즉, 판별자가 데이터의 출처를 구별할 수 없는 '도메인에 무관한(domain-invariant)' 특징을 만들도록 강제됩니다.

### 편향 완화 원리
메인 모델이 판별자를 성공적으로 속인다는 것은, 모델이 편향된 학습 데이터에만 존재하는 특징에 의존하는 대신, 두 도메인 모두에 공통적으로 중요한 특징을 학습했다는 의미입니다. 이를 통해 편향에 강건한(robust) 모델이 만들어져 실제 추천 환경에서 일반화 성능이 향상됩니다.

## online 실험
- binary classification 은 imp 수는 유지 되었으나 ctr 하락
- in-batch negative, knowledge distillation 은 ctr은 상승했으나 imp 하락
- MUDA 는 imp 유지, ctr 상승

## Knowledge Distillation과 MUDA의 차이점

"An Empirical Study of Selection Bias in Pinterest Ads Retrieval, 2023 Pinterest" 논문에서 설명하는 MUDA와 일반적인 Knowledge Distillation의 가장 큰 차이점은 **리트리벌 모델(Student)을 학습시키기 위해 사용하는 '가짜 레이블(pseudo label)'의 형태와 그에 따른 '손실 함수(loss function)'** 입니다.

해당 논문은 광고 검색(Ads Retrieval) 단계의 선택 편향을 완화하기 위해, 더 정교한 모델인 **랭킹 모델(Teacher)**의 예측을 활용하여 리트리벌 모델(Student)을 학습시키는 방법을 사용합니다.

### Knowledge Distillation (일반적인 접근)

*   **레이블 형태**: 랭킹 모델이 예측한 **연속적인 점수(continuous score)** 자체를 가짜 레이블로 사용합니다. (예: 클릭 확률 0.78, 0.32 등)
*   **학습 방식**: 리트리벌 모델이 랭킹 모델의 연속적인 점수를 그대로 모방하도록 학습시킵니다. 이는 **회귀(Regression)** 문제와 유사하며, 보통 LogMAE(Log Mean Absolute Error) 같은 손실 함수를 사용합니다.
*   **목표**: "Student 모델이 Teacher 모델처럼 똑같이 점수를 예측해봐!"

### MUDA (논문에서 제안한 방법)

*   **레이블 형태**: 랭킹 모델이 예측한 연속적인 점수를 특정 임계값(threshold)을 기준으로 **'클릭' 또는 '클릭 안함' 같은 이진(Binary) 레이블로 변환**하여 사용합니다. (예: 점수가 0.8 이상이면 '클릭(1)', 0.1 이하이면 '클릭 안함(0)')
*   **학습 방식**: 리트리벌 모델이 변환된 이진 레이블을 맞추도록 학습시킵니다. 이는 **분류(Classification)** 문제이며, BCE(Binary Cross-Entropy) 손실 함수를 사용합니다.
*   **목표**: "Student 모델아, Teacher 모델이 '클릭될 것 같다'고 판단한 아이템을 너도 '클릭될 것'으로 분류해봐!"

### 핵심 차이 요약

| 구분 | Knowledge Distillation | MUDA (논문 제안) |
| :--- | :--- | :--- |
| **접근 방식** | 회귀 (Regression) | **분류 (Classification)** |
| **가짜 레이블** | 랭킹 모델의 **연속적인 점수** | 점수를 변환한 **이진 레이블** |
| **손실 함수** | LogMAE 등 (차이 측정) | **BCE** (분류 오차 측정) |

결론적으로 MUDA는 랭킹 모델의 예측을 단순히 따라하는 것을 넘어, **'클릭 여부'라는 더 명확한 신호로 변환하여 학습**함으로써 리트리벌 모델의 성능을 더 효과적으로 향상시켰고, 핀터레스트의 온라인 테스트에서 가장 좋은 결과를 얻었습니다.