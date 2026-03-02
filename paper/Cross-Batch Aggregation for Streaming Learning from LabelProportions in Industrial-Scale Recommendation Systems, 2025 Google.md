# Cross-Batch Aggregation for Streaming Learning from Label Proportions in Industrial-Scale Recommendation Systems

- RecSys 2025, Google DeepMind / Google LLC
- Jonathan Valverde, Tiansheng Yao, Xiang Li, Yuan Gao, Yin Zhang, Andrew Evdokimov, Adam Kraft, Samuel Ieong, Jerry Zhang, Ed H. Chi, Derek Zhiyuan Cheng, Ruoxi Wang

## 핵심 요약

GDPR, Apple ATT 등 프라이버시 규제로 인해 event-level 레이블이 aggregated item-level 레이블로 대체되면서 추천 모델의 편향 및 miscalibration 문제가 발생. 이를 해결하기 위해 **Learning from Label Proportions (LLP)** 프레임워크를 스트리밍 환경에 적용한 **Cross-Batch Aggregate (XBA) Loss**를 제안. Google Ads 시스템에 실제 배포하여 **온라인 bias 48.8% 감소**, AuC +0.47% 향상 달성.

## 문제 정의

- 프라이버시 규제로 개별 클릭-전환 매칭 불가 → 전환이 aggregated 형태로만 제공됨
- 기존 event-level 학습 패러다임에 noisy aggregate를 그대로 끼워 맞추면 모델 bias + miscalibration 발생
- **LLP(Learning from Label Proportions)**: 집계된 bag-level 신호로 instance-level 예측 모델을 학습하는 프레임워크
  - 핵심 가정: 하나의 bag에 속한 모든 샘플이 동일한 배치 내에 있어야 함
  - 산업용 스트리밍 시스템에서는 이 가정이 깨짐

### 스트리밍 환경에서 LLP 적용의 3가지 장벽

1. **배치/bag 불일치**: 순차적 스트리밍 학습으로 aggregate가 여러 배치에 분산
2. **Bag 크기 초과**: aggregate가 단일 배치 크기를 초과하는 경우 빈번
3. **레이블 노이즈**: 전환 신호의 랜덤 지연으로 인해 bag의 시간 경계 파악 불가

# Introduction

- 산업용 추천 시스템은 스트리밍 프레임워크로 학습 → 샘플이 시간순으로 소비되어 non-stationarity 포착
- 기존 LLP 기법들은 bag 내 모든 샘플이 단일 배치에 있어야 하므로 직접 적용 불가
- Aggregate loss를 스트리밍 설정에 맞게 적응시키는 방향으로 문제 접근

# Methodology

## Cross-Batch Aggregate (XBA) Loss

### 핵심 아이디어

각 배치에서 bag별 **누적 통계(running estimates)**를 유지하고, 이를 활용해 aggregated loss의 gradient를 pointwise 수준에서 근사.

**전체 aggregated loss:**
$$L_{total}(\theta) = \sum_{j=1}^{m} |B_j| \mathcal{L}(\mu_{pred,j}, \mu_{label,j})$$

**분해된 pointwise loss (동일한 gradient 보장):**
$$L_{dec.total}(\theta) = \sum_{i=1}^{n} \text{sg}\left(\frac{\partial \mathcal{L}}{\partial \mu_{pred, bag(i)}}\right) h(x_i; \theta)$$

- `sg`: stop gradient 연산자 (해당 항을 상수로 취급)
- bag별 평균 prediction/label 추정치 $\tilde{\mu}_{pred}$, $\tilde{\mu}_{label}$를 **decay가 있는 moving average**로 유지

### XBA L₂ Loss (회귀 문제)

$$L_{XBA,2}(x_i, \theta) = \text{sg}(\tilde{\mu}_{pred,bag(i)} - \tilde{\mu}_{label,bag(i)}) \cdot h(x_i; \theta)$$

- bag의 평균 오차(mean error)를 가중치로 사용
- 오버예측 시 → 예측값을 낮추는 방향으로 업데이트
- 언더예측 시 → 예측값을 높이는 방향으로 업데이트
- 완벽히 calibrated 상태 → 가중치 0, 업데이트 없음

### XBA Cross-Entropy Loss (이진 분류)

$$L_{XBA,CE}(x_i, \theta) = \text{sg}(\tilde{\mu}_{pred,bag(i)} - \tilde{\mu}_{label,bag(i)}) \cdot z(x_i; \theta)$$

- $z(x_i; \theta)$: logit (sigmoid vanishing gradient 문제 회피)
- 수치 안정성을 위해 sigmoid 미분을 loss weight에 포함시키는 추가 근사 적용

### 주요 특성

- **Calibration 직접 최적화**: loss 최솟값이 bag-level calibration과 일치
- **노이즈 강건성**: 레이블이 aggregate 형태로만 loss에 반영됨
- **Bag 크기 독립성**: 스트리밍에서 전체 bag 크기 추정 불필요

## ELT/ALT 혼합 아키텍처

일부 샘플은 event-level 레이블, 일부는 aggregate 레이블을 가지는 혼합 환경을 위한 아키텍처:

```
[Model Features]
      ↓
[Embedding]
      ↓
[ELT Tower] ──────────────────────────────────────────
  (Pointwise Loss)    ↓ stop gradient        ↓ stop gradient
               [ALT Tower 1]           [ALT Tower 2]
               (XBA Loss)              (XBA Loss)
                    ↓                        ↓
               [ALT 1 Logit]          [ALT 2 Logit]
                    └────────[Ensemble]────────┘
                         [ALT Ensemble Prediction]
```

- **ELT Tower**: event-level 레이블 샘플만 학습, aggregate 신호에 완전히 독립
- **ALT Tower 1, 2**: ELT 임베딩 재사용 + ELT logit과 합산 (stop gradient로 ELT 보호)
- **앙상블**: 두 ALT 타워를 별도 학습 후 앙상블하여 overfitting 방지
- **서빙**: 쿼리 유형에 따라 ALT 또는 ELT 출력 선택

# Experiments

Google Ads 전환 예측 시스템 (Apple ATT 등 프라이버시 제약으로 신호 손실 발생한 시스템)에 적용.

## 결과

| 모델 | Offline 평균 Pred/Label 비율 | Offline AuC | Online 가중치 Bias Mean | Online 가중치 Bias IQR |
|---|---|---|---|---|
| pCVR Model | 1.05 → **1.00** | **+0.47%** | **−48.8%** | **−26.2%** |

- **Offline bias 거의 제거**: pred/label 비율이 1.05에서 1.00으로 수렴
- **Calibration 개선**: online weighted bias mean 48.8% 감소, IQR 26.2% 감소
- **예측 정확도 향상**: AuC +0.47% (aggregated 레이블로 calibration 이상의 학습 성공)
- **광고주 경험 개선**: 과다/과소 지출 모두 감소, 광고주 가치 향상

# Conclusion

- 프라이버시 규제로 인한 신호 손실 문제에 LLP를 스트리밍 환경에 적용하는 XBA 제안
- 배치/bag 불일치, bag 크기 초과, 레이블 노이즈 문제를 moving average 기반 누적 통계로 해결
- L₂, Cross-Entropy 두 가지 loss 형태 제공 → 다양한 문제 설정에 유연하게 적용 가능
- Google Ads에 실제 배포하여 calibration, 광고주 가치, 핵심 비즈니스 지표 모두 개선
- LLP 연구와 산업용 대규모 추천 시스템 간의 간극을 메우는 실용적 솔루션
