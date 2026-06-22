# Not All Impressions Are Created Equal: Psychology-Informed Retention Optimization for Short-Form Video Recommendation

- RecSys 2025, Meta (Facebook Reels) / Stanford
- Yuyan Wang, Jing Zhong, Yuxin Cui, Zhaohui Guo, Chuanqi Wei, Yanchen Wang, Zellux Wang

## 핵심 요약

숏폼 비디오 추천에서 **장기 retention(재방문)을 모델링**할 때 기존 방식은 세션 내 모든 impression(영상)에 동일한 retention 라벨을 부여한다. 하지만 숏폼은 영상이 짧고(10~30초) 수동적으로 빠르게 소비되기 때문에, 개별 영상과 재방문 행동을 연결하는 신호가 매우 noisy하다. 이 논문은 심리학의 **peak-end rule**(사람은 경험을 평균이 아니라 가장 강렬한 순간 "peak"과 마지막 순간 "end"으로 평가한다)을 추천에 적용한다. 세션의 peak/end 영상만을 retention 학습 신호로 사용해 모델을 학습하고, 이를 ranking 함수에 결합. Facebook Reels에서 2.5개월 장기 A/B 테스트 결과 **DAU, 세션 수 등 핵심 비즈니스 지표가 유의미하게 향상**.

## 문제 정의

- 단기 engagement(클릭, 좋아요, dwell time)만 최적화하면 clickbait, filter bubble, echo chamber 등 장기 경험을 해치는 부작용 발생 → retention 신호 도입 필요
- 기존 retention 모델의 한계 (숏폼 맥락에서):
  1. **신호 attribution의 어려움**: Reels 영상은 평균 10~30초로 매우 짧아, 단일 영상이 사용자 재방문을 유의미하게 유발한다고 보기 어려움. YouTube/Netflix 같은 long-form 대비 소비가 fragmented하고 signal-to-noise ratio가 낮음
  2. **동일 가중치 가정의 비현실성**: 사용자는 피드를 수동적으로 스크롤하며 모든 영상에 동일하게 engage하지 않음 → 모든 impression을 동일 가중하면 attention 변동을 무시해 bias 유입

## Methodology

### Peak-End Rule 적용

- **Peak-end rule** (Kahneman, 1993): 사람은 경험을 평균이 아니라 가장 강렬한 순간(peak)과 마지막 순간(end)으로 기억/평가
- 숏폼 세션에 적용: 사용자는 세션 전체 중 **소수의 기억에 남는 순간**을 기준으로 재방문 여부를 결정한다고 가정

### Peak / End 정의

- **Session**: immersive 피드 진입부터 일정 시간 비활성 또는 명시적 종료까지의 연속된 영상 impression/상호작용 시퀀스
- Peak는 직접 측정이 어려우므로(생리적 arousal 측정 불가) **명시적·능동적 사용자 행동**으로 추론. 수동적 소비 환경에서 의도적 행동은 높은 attention/arousal 신호
  - **Positive peaks**: follow, comment, share 중 하나 이상 받은 영상
  - **Negative peaks**: dislike, hide, exit 중 하나 이상 받은 영상
  - **Ends**: 세션 종료 직전 마지막으로 소비한 영상

### Reward Attribution (핵심 기여)

#### 문제: "재방문이라는 결과를 어떤 영상 탓으로 돌릴 것인가?"

- Retention 모델 학습에는 **(영상, 라벨)** 쌍이 필요한데, 라벨("다음 날 돌아왔는가?")은 **세션 단위**로 매겨짐
- 한 세션에서 영상 30개를 보고 다음 날 돌아왔다면(return=1), **30개 중 무엇이 재방문을 유발했는가?** → 세션 레벨 결과를 개별 영상(item-level)에 배분하는 것이 attribution 문제

#### 기존 방식: 균등 attribution (Equal attribution)

- 세션이 return=1이면 그 세션의 **모든 영상에 라벨 1**을 부여

```
세션 (return=1):
[V1, V2(share), V3(hide), ..., V30]  →  전부 라벨 = 1
```

- **문제점**:
  - share를 유발한 영상(V2)과 hide를 유발한 영상(V3)이 **둘 다 라벨 1** → "이런 영상 보여줬더니 돌아왔다"가 정반대 특성 영상에 동시에 붙음 = **상충하는 supervised 신호(conflicting signals)** → 학습 혼란, 성능 저하
  - 숏폼은 대부분 무반응으로 스쳐가는 영상 → 이런 영상까지 라벨링하면 **noise만 증가** (SNR 하락)

#### 제안 방식: Peak-End 기반 attribution

- 직관: 사람은 **기억에 남는 긍정적 순간(positive peak)** 때문에 돌아오고, **부정적 순간(negative peak)·불만족스러운 마무리(end)** 때문에 안 돌아온다
- 라벨에 따라 **학습에 쓸 영상을 선별**:
  - **재방문(return=1)** → **positive peaks만** 사용, 라벨 1
  - **미재방문(return=0)** → **negative peaks + end 영상만** 사용, 라벨 0
  - 그 외 (무반응) 영상은 모두 **학습에서 제외(excluded)**

```
return=1: [V1, V2(share), V3, ..., V30(end)]  →  V2만 사용(라벨 1)
return=0: [V1, V2, V3(hide), ..., V30(end)]   →  V3·V30만 사용(라벨 0)
```

#### 핵심 효과

| | 균등 attribution | Peak-End attribution |
|---|---|---|
| 상충 신호 | share·hide 영상이 같은 라벨 | valence(긍정/부정)와 라벨 정합 |
| Noise | 무반응 영상까지 전부 학습 | salient한 순간만 학습 |
| Signal quality / 일반화 | 낮음 / 어려움 | 높음 / 세션 간 일반화 개선 |

- 핵심은 **라벨 부호(1/0)와 영상 valence(긍정/부정)를 일치**시켜 모순을 제거한 것
- ※ (각주 2) **end 영상은 항상 세션 종료와 동시에 발생**하므로 정의상 non-return(0) 행동에만 연결됨 → end는 "긍정적 마무리"로는 쓰이지 않고 항상 부정 신호 쪽에만 사용 (peak-end rule의 end를 보수적으로 해석한 설계, → future work의 "end 이후 recency bias" 탐구로 이어짐)

### Retention Modeling

- **Retention 라벨**:
  - return=1: 다음 날 플랫폼에 복귀 + 복귀 세션에서 최소 X분 이상 소비 (의미 있는 engagement 보장)
  - return=0: 그 외
- 모델 구조 (Figure 3): `User features / Video features / User×Video cross features` → MLP → **MMOE task shared network** → sigmoid → `p(return)`
  - Dense feature: 나이, watch time, user-creator 간 view/like 수(여러 trailing window)
  - Sparse feature: user/content ID → embedding
  - Loss: Binary cross-entropy
  - Inference 시 `p(return)` 확률 출력

### Final Ranking Function

- 기존 production ranking은 multi-task model로 like/comment/share 등 **단기 신호**를 예측해 가중합:

  $$R^{prod}_i = \sum_{s \in S} w_s \cdot p_i(s)$$

- 여기에 cross-session retention 신호를 곱셈 형태로 결합:

  $$R^{proposed}_i = \underbrace{R^{prod}_i}_{\text{단기 신호}} \cdot \big(1 + w_r \cdot \tilde{p}_i(return)\big)$$

  - $\tilde{p}_i(return) = p_i(return) - \frac{1}{N}\sum_{i=1}^{N} p_i(return)$ : **세션 내 후보 N개의 평균을 빼서 정규화** → user-level 효과 제거, 세션 내 상대적 차이만 ranking에 반영
  - $w_r$: retention 신호의 상대적 중요도를 조절하는 하이퍼파라미터, A/B 테스트로 튜닝

#### "cross-session retention 신호"란?

신호를 성격에 따라 두 종류로 구분하는 것이 핵심:

| | 의미 | 예시 |
|---|---|---|
| **in-session 단기 신호** | 지금 이 세션 **안에서** 이 영상에 바로 보일 반응 | like, comment, share, dwell time |
| **cross-session 신호** | 세션 경계를 **가로지르는** 행동 ("내일 다시 올까?") | `p(return)` |

- 기존 ranking($R^{prod}$)은 전부 **in-session 단기 신호**의 가중합 — "지금 이 영상 보여주면 좋아요 누를까?" 같은 즉각 반응만 봄
- `p(return)`은 **"이 영상이 사용자를 다시 오게 만드는 데 기여하는가?"** 라는 세션을 넘나드는 장기 신호 → "단기 신호를 넘어(go beyond) cross-session retention을 ranking에 넣겠다"는 의미

#### 왜 "곱셈" 형태인가 (덧셈이 아니라)

- **덧셈이었다면** `R = R_prod + w_r·p(return)` → retention 점수가 절대값으로 더해져, 단기 relevance가 낮은(관련 없는) 영상도 retention만 높으면 위로 끌려 올라감. 두 신호가 독립적으로 경쟁
- **곱셈은** `(1 + w_r·p̃)`가 **1을 기준으로 한 보정 배수(multiplier)** 로 작동:
  - retention이 세션 평균보다 **높으면** ($\tilde{p}>0$) → 계수 > 1 → 점수 **부스팅**
  - 세션 평균보다 **낮으면** ($\tilde{p}<0$) → 계수 < 1 → 점수 **감점**
  - 평균과 같으면 → 계수 = 1 → 그대로
- 즉 곱셈은 retention이 **기존 단기 ranking을 base로 깔고 그 위에서 순위를 미세 조정(modulate)** 하는 역할만 함. 단기 점수가 0인 영상을 retention만으로 살려내지 않음 → 단기 relevance를 보존하면서 장기 가치로 재정렬

#### 정규화 $\tilde{p}$가 중요한 이유

- 곱셈 계수가 1을 기준으로 위아래로 움직이려면 retention 값이 **0 기준 +/- 분포**여야 함 → 그래서 세션 내 후보 N개의 평균을 빼줌
- (각주 5) 사용자마다 `p(return)`의 절대 수준이 다름 (헤비유저는 전반적으로 높음). 평균을 빼면 **user-level 효과(절대 수준)가 상쇄**되고, "이 사용자가 원래 잘 돌아오냐"가 아니라 **"이 후보들 중 상대적으로 어떤 게 재방문에 더 기여하냐"** 만 ranking에 반영됨

## Online A/B Test 결과

- Facebook Reels에서 2.5개월 장기 A/B 테스트 (treatment: peak-end retention 모델 vs. control: production)
- 통계적으로 유의미한 개선:
  - **Facebook Reels 세션 +0.42%**
  - **DAU +0.03%**
  - **전체 Facebook 세션 +0.05%**
- 실험 기간 내내 일관된 상승 추세 → 사용자가 반복 engagement와 retention을 유발하는 콘텐츠를 지속적으로 발견·소비하도록 도움 (장기 경험 개선 시사)

## Conclusion & Future Work

- peak-end rule을 활용해 세션의 가장 salient한 순간에 집중함으로써, **개별 영상에 장기 retention을 attribute하는 문제**를 효과적으로 해결
- 별도 시스템이 아니라 기존 multi-stage 추천 시스템의 ranking에 retention 신호를 통합하는 형태
- Future work:
  - candidate generation, reranking 등 추천 시스템의 다른 단계로 확장
  - peak-end rule의 변형 실험: end 이후의 recency bias, negativity bias
  - 심리학의 인지·정서 이론을 장기 최적화 전략 설계에 통합

## 인사이트

- "Not all impressions are created equal" — 모든 노출/상호작용을 동등하게 다루지 말고, **신호 품질이 높은 소수의 순간(peak/end)에 집중**하면 noisy한 장기 신호 학습이 개선된다는 발상이 핵심
- 긍정/부정 행동을 valence로 구분해 return=1/0 라벨과 정합적으로 매핑함으로써 **상충 신호 제거**한 점이 단순하지만 효과적
- 매우 거대한 플랫폼(Reels)에서는 +0.42% 같은 작은 lift도 비즈니스적으로 큰 의미를 가짐
- ranking에 retention을 **곱셈(1 + w·p̃)** 형태로 결합하고 세션 평균을 뺀 정규화로 user-level bias를 제거한 설계는 실무 적용 시 참고할 만함
