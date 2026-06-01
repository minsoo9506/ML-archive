# Item-centric Exploration for Cold Start Problem

- **Authors**: Dong Wang, Junyi Jiao, Arnab Bhadury, Yaping Zhang, Mingyan Gao, Onkar Dalal (Google LLC)
- **Venue**: RecSys 2025, September 22–26, Prague, Czech Republic
- **링크**: https://doi.org/10.1145/3705328.3748113

---

## 1. 문제 정의

### Item Cold-Start Problem
- 신규 아이템은 interaction 데이터가 없어 추천 시스템에 노출되기 어려움
- 결과적으로 popularity bias 심화, content diversity 저하

### 기존 접근법의 한계
- auxiliary data (item attributes, side information), meta-learning, transfer learning 등이 주류
- 그런데 이 논문은 다른 근본적인 문제를 지적: **user-centric 패러다임 자체의 한계**

### User-centric vs Item-centric
| | User-centric | Item-centric |
|---|---|---|
| 목표 | 특정 유저에게 최적 아이템 찾기 | 특정 아이템에 최적 유저 찾기 |
| Cold-start 문제 | 신규 아이템이 잘못된 audience에 노출 → 평가 실패 → 영원히 묻힘 | 신규 아이템의 진짜 audience를 먼저 찾아줌 |

- Figure 1: 2x2 user-item score 테이블에서 user-centric은 각 유저에게 score 높은 아이템 선택 (파란색), item-centric은 각 아이템에 score 높은 유저 선택 (보라색) → 관점의 전환

---

## 2. 방법론 (Methodology)

### 2.1 전체 파이프라인

```
User Request
    → Candidate Retrieval (신규 아이템 풀에서)
    → Ranking (multi-task ranking model)
    → [Item-centric Filter] ← 이 논문의 핵심 추가 컴포넌트
    → 유저에게 노출
```

- 기존 exploration 시스템의 ranking stage 이후에 **item-centric filtering component** 삽입
- ranking 결과를 활용하기 때문에 ranking stage 뒤에 위치

### 2.2 필터링 조건

아이템을 유저에게 노출하지 않는 조건:

$$p(S_+|u, i) < \mu_i - 2\sigma_i$$

- $p(S_+|u, i)$: ranking model이 예측한 해당 유저-아이템 쌍의 만족 확률
- $\mu_i$: 아이템의 satisfaction rate posterior mean
- $\sigma_i$: 아이템의 satisfaction rate posterior standard deviation

**직관**: 예측된 유저 만족도가 아이템의 평균 만족도보다 유의미하게 낮으면 → 이 유저는 이 아이템의 적절한 audience가 아님 → 필터링

### 2.3 아이템의 Intrinsic Satisfaction Rate 모델링

**Beta distribution** 사용 (Bernoulli 분포의 conjugate prior)
- satisfied / not satisfied 이진 outcome에 자연스럽게 맞음
- conjugate 성질로 posterior 계산이 매우 효율적 (파라미터 2개만 저장)

**Prior**: $B(\alpha_0, \beta_0)$

$N_+$: satisfied 횟수, $N$: 전체 impression 수 일 때,

**Posterior mean**:
$$\mu_i = \frac{\alpha_0 + N_+}{\alpha_0 + \beta_0 + N}$$

**Posterior variance**:
$$\sigma_i^2 = \frac{(\alpha_0 + N_+)(\beta_0 + N - N_+)}{(\alpha_0 + \beta_0 + N)^2(\alpha_0 + \beta_0 + N + 1)}$$

- impression이 쌓일수록 $\sigma_i$ 감소 → 추정 신뢰도 증가 (Figure 3)
- 초반 수백 impression 동안 빠르게 수렴 → **low-latency aggregation** 필요

### 2.4 유저 만족 확률 예측

- $p(S_+|u, i)$는 large-scale multi-task ranking model의 prediction head 출력값 사용 [ref 12]

---

## 3. 실험 결과

### Calibration 분석 (Figure 4)
- 모델의 예측값과 실제 ground truth 간 alignment 확인
- 매우 낮은 satisfaction rate 구간에서 약간의 miscalibration 존재하나, 전체 성능에 영향 없음

### Live Experiment (Table 1)

| 지표 | 변화 |
|---|---|
| Satisfaction Metric 1 | **+50%** |
| Satisfaction Metric 2 | **+40%** |
| Exploration Impressions | **-20%** |
| Recommendable Corpus | **+10%** |

**해석**:
- Exploration impression 20% 감소: 더 selective하게 audience 선정 → 효율 향상
- 유저 만족도 50%, 40% 향상: 탐색된 콘텐츠 품질 개선
- Recommendable corpus 10% 증가: 더 많은 신규 아이템이 실제 추천 가능한 상태로 진입

### 실험 설계
- 기존 user-diverted 실험으로는 item corpus 변화 측정 불가 (disjoint item set 필요)
- → **user-corpus co-diverted exploration experiment framework** 개발
- user satisfaction과 corpus expansion을 동시에 측정 가능

---

## 4. 핵심 기여 및 의의

1. **패러다임 전환**: user-centric → item-centric, 새로운 관점 제시
2. **경량 솔루션**: 기존 시스템에 filtering component만 추가, 전체 시스템 교체 불필요
3. **Bayesian 모델**: Beta distribution으로 아이템 satisfaction rate를 효율적으로 추정
4. **실증적 검증**: Google YouTube short-form 대규모 실서비스에서 significant 개선 확인

---

## 5. 비판적 고찰: 두 값의 비교가 정당한가?

필터링 조건 $p(S_+|u, i) < \mu_i - 2\sigma_i$ 에서 좌변과 우변이 동일선상에서 비교 가능한지에 대한 논점.

### 정당화 근거
- 두 값 모두 **"satisfied/not satisfied 이진 label에 대한 확률"** 을 추정 → 이론적으로 같은 공간
  - $p(S_+|u, i)$: 특정 유저-아이템 쌍의 만족 확률 (모델 예측)
  - $\mu_i$: 해당 아이템을 본 유저들의 실제 만족률 평균 (경험적 추정)
- ranking model이 cross-entropy loss로 학습된 calibrated 모델이라면 출력값이 실제 확률로 해석 가능
- 논문의 Figure 4 calibration plot이 이를 사후 검증하려는 의도

### 의심스러운 지점

**1. Calibration 가정의 취약성**
- Ranking model은 보통 순위 학습 (listwise/pairwise loss) 목적 → 출력값이 진짜 확률이라는 보장 없음
- 논문 스스로도 "very low satisfaction rate 구간에서 miscalibration 존재" 인정

**2. Selection bias**
- $\mu_i$는 **이미 노출된 유저들**의 반응으로부터 추정됨 → 대표성 있는 샘플인지 불확실
- Cold-start 초기엔 소수의 noisy한 impression으로 추정되어 분산이 큼

**3. 개념적 비대칭**
- $\mu_i$: "평균적인 유저"에 대한 아이템 품질
- $p(S_+|u, i)$: **특정 유저**에 대한 예측
- 아이템이 niche 콘텐츠라면 $\mu_i$가 원래 낮은데, 그 아이템의 진짜 핵심 팬인 유저도 필터링될 위험 존재

### 결론
- 비교 자체가 틀린 건 아니지만, "ranking model이 잘 calibrated 되어 있다"는 암묵적 가정에 강하게 의존
- Figure 4로 사후 검증하는 방식 → 이론적 정당화보다는 **실용적 근사**에 가까운 설계
- 더 엄밀하게 하려면 Platt scaling 등으로 $p(S_+|u, i)$ 보정 후 비교하거나, 두 값을 통합된 모델 체계 내에서 추정하는 방향이 필요

---

## 6. 한계 및 향후 연구

- 논문 자체는 현재 구현이 item-centric 비전의 "첫 번째 실현"임을 명시
- 극히 낮은 satisfaction rate 구간에서 miscalibration 존재
- 더 완전한 item-centric 시스템으로의 전환은 향후 과제

---

## 핵심 요약 (한 줄)

> 신규 아이템에 적합한 유저를 찾아주는 item-centric 관점으로 전환하고, Beta distribution 기반 Bayesian 필터를 ranking 이후 단계에 삽입하여 cold-start 탐색 효율과 유저 만족도를 동시에 크게 개선함.
