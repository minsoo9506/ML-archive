# Leveraging Explicit Negative Feedback in Large-Scale Recommendation Systems: A Case Study (TikTok, 2025)

- RecSys '25 (TikTok)
- 저자: Madhura Raju, Manisha Sharma, Hongyu Xiong, Bingfeng Deng, Meng Na

## 한줄 요약
대부분의 추천 시스템은 좋아요/시청시간 같은 **positive engagement**에 치중되어 있는데, 사용자가 **싫어하는 것**(explicit negative feedback)도 동등하게 중요하다. TikTok은 **light-weight, context-aware in-feed survey**로 명시적 부정 피드백을 수집하고, 이를 denoise/debias하여 랭킹 모델에 통합해 피드 품질과 장기 retention을 개선했다.

## 1. Motivation
- 기존 신호(시청시간, skip, dislike)는 **약하고 간접적인 cue** — 사용자가 *무엇을* 하는지는 알려주지만 *왜* 하는지는 못 알려줌
- Negative feedback은 rarer하지만 user intent를 깊이 이해하는 데 필수
- "왜 싫어했는지"를 명시적으로 물어보면 personalization, safety, trust 모두 개선 가능

## 2. Methodology — Survey 설계
비디오 시청 직후, 그 비디오에 대한 **context-aware survey**를 일부 사용자에게 노출.

세 가지 survey 형태:
1. **Boolean**: 예) "이 콘텐츠가 TikTok에 적합한가?"
2. **Categorical**: 예) "이 콘텐츠에 대해 어떻게 느끼는가?" (4~5개 옵션)
3. **Two-stage**: top-level binary → secondary categorical (예: 부적절하다고 답하면 → "왜?" 라는 카테고리 질문)

### 설계 원칙
- 높은 응답률 + 낮은 fatigue
- 참여 optionality
- 명확한 objective (UX 모니터링 / 모델 최적화)
- 문구·디자인 iterate, distribution throttle, drop-off / incompletion 모니터링, A/B로 개선

## 3. Case Study — For-You-Page Two-stage Survey
예시: "Do you think this video is appropriate for TikTok?" → Yes/No/18+만 적합 → No면 이유 카테고리 (Disgusting, Violent, Spam, Hateful, Sexually suggestive, Uninteresting, Other)

### 두 가지 핵심 챌린지
1. **Response bias** → **Unbiased survey modeling**: survey-submit 모델로 이 유저/맥락에서 survey 에 응답을 제출할 확률을 이용하여 **inverse propensity weighting (IPW)** 로 각 응답을 동등하게 가중 (Ref: USM, Yu et al. 2024)
2. **Noisy negative feedback** → 고품질 데이터로 학습한 **filtering**으로 weak response 제거

### 모델 아키텍처
- **Multi-head model**: 각 issue가 별도의 auxiliary head (예: spam head, hateful head ...)
- 각 head는 "사용자가 그 issue를 선택할 확률"을 예측
- **Late-stage ranking**(Recall → PreRank → Rank → LTR 중 후반부)에 붙일 때 가장 임팩트 큼
- Threshold는 A/B로 튜닝하여 survey metric 감소와 guardrail metric 사이의 trade-off 최적화

### 라벨 구성 (label) — 주의할 점
- head 들의 라벨 = 2단계 survey 의 **"부정 이유 카테고리"** (Disgusting / Violent / Spam / Hateful / Sexually suggestive / Uninteresting / Other)
- 단, 각 head 내부는 **binary 분류**: 해당 이유를 골랐으면 positive(1), **나머지 응답 전부**가 negative(0)
  - 예) spam head → "It's because this video is spam" 응답 = 1, 그 외 응답(다른 부정 이유 + 1단계에서 "Yes/적절"이라 답한 긍정·중립 응답) = 0
  - 즉 "전부 부정 라벨"이 아니라 **"이 특정 부정 이유인가(1) vs 나머지(0)"** 구조이고, 0쪽에는 긍정 응답도 섞임
- 카테고리 성격이 균일하지 않음: 대부분(Disgusting/Violent/Spam/Hateful/Sexually suggestive)은 **safety·violative** 이슈지만, **Uninteresting** 은 안전이 아닌 **relevance/취향** 이슈 → intervention 도 달라짐(violative=Filter, 취향=Deboost/Dispersion)
- 학습 시: 이 **label(정답)** + 앞의 **IPW(샘플 가중치 $w_i=1/p_i$)** 를 함께 사용 → $\mathcal{L}=\sum_i w_i \cdot \text{loss}(\hat{y}_i, y_i)$. label="무엇을 맞출까", IPW="이 샘플을 얼마나 중요하게 볼까"로 역할 분리

### 이슈 종류별 intervention
- **Filter**: 극단적 violative content 제거
- **Deboost**: 노출 감소
- **Dispersion**: echo chamber 탈출 유도

### End-to-End Pipeline
User Survey → Yes/No + Category → Denoise/Debias/Filtering → Model Training (user, response labels) → A/B로 threshold 선정 → Apply feed interventions (Deboost / Filter / Dispersion) → 온라인 metric 모니터링 & iterate

## 4. 실험 결과 (AB 상대 변화)
| Metric | 변화 |
|---|---|
| Average Time Spent per User | **+0.34%** |
| Inappropriate Survey Rate | **−5.60%** |
| Confirmed Policy Violations | **−3.38%** |

추가 인사이트:
- 단기적으로는 safety intervention이 engagement에 부정적일 수 있지만, **장기적으로는 거의 항상 positive** (avg time spent ↑, DAU 상승 트렌드)
- Retention 개선의 **가장 큰 기여는 low/medium engagement 코호트**에서 발생 (이미 high engaging한 유저보다)

## 5. Takeaways
- 명시적 negative signal을 **first-class citizen**으로 다루는 것이 핵심
- Survey라는 가볍지만 의도적인 메커니즘만으로도 의미있는 trust/quality 개선 가능
- 단순히 데이터 수집이 아니라 **denoise + debias(IPW) + multi-head 학습 + late-stage ranking 통합** 의 풀 파이프라인이 중요
- Negative feedback은 특히 **저참여 유저의 retention**을 끌어올리는 레버

## 관련 참고
- USM: Unbiased Survey Modeling (Yu et al. 2024, arXiv:2412.10674) — IPW 기반 survey debiasing
- Garcia-Puyol et al. 2023 (WWW) — Detecting and Limiting Negative User Experiences
