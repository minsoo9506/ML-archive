# Improve the Personalization of Large-Scale Ranking Systems by Integrating User Survey Feedback (Meta, 2025)

- RecSys '25 (Meta, 빌리언 단위 사용자 보유 비디오 플랫폼)
- 저자: Mengxi Lv, Drew Hogg, Thomas Grubb, Shashank Bassi, Min Li, Cayman Simpson, Senthil Rajagopalan
- 링크: https://doi.org/10.1145/3705328.3748119

## 1. 문제의식
- 개인화의 핵심은 "이 앱이 나를 잘 안다"라는 느낌을 주는 **interest relevance**.
- 기존 방식: 사용자의 과거 engagement(시청, 좋아요, 공유 등)를 미리 정의된 **interest cluster**에 매핑해서 관심사를 추정.
- 한계
  - 휴리스틱 클러스터 기반 → 노이즈, 확장성, 해석성 문제.
  - **랭킹 피드백 루프**에 갇혀 사용자의 실제 관심사와 어긋남.
  - 사용자가 비디오를 끝까지 봐도 "정말 흥미가 있어서" 본 건 아닐 수 있음. engagement signal ≠ true interest.

## 2. 핵심 아이디어: UTIS (User True Interest Survey) 모델
- 시청 세션 중 일부 사용자에게 **단일 문항 in-session survey**를 노출:
  - 질문: *"how well the content matches your interests"*, 1~5점 평가.
- 이 explicit feedback을 학습 데이터로 사용하는 **별도의 가벼운 분류 모델**을 학습.
  - 라벨 이진화: 상위 2점 = positive, 하위 2점 = negative.
  - Loss: 표준 binary cross-entropy.
  - 입력 feature: user feature(인구통계, 토픽 선호, 방문 이력) + video feature(품질, engagement, semantic representation) + interaction feature.
  - **해석 가능성(interpretability)**을 의도적으로 고려한 설계.
- 출력: 임의의 (user, video) 쌍에 대해 "사용자가 이 콘텐츠를 정말 관심 있어할 확률" 점수를 imputation.

## 3. 메인 랭킹 시스템에 통합하는 두 가지 방식
메인 랭커는 multi-task / multi-label 거대 모델. UTIS는 **calibration / 보조 layer**로 붙임.

### Use case I — Retrieval 단계 (early stage)
- 기존 interest retrieval은 user interest cluster 에서 interest 뽑아서 profile 만들고 → interest 에 해당하는 비디오 retrieve.
- UTIS로 (user, video) affinity를 예측 → video→cluster 룩업으로 **사용자의 true interest profile을 재구성**.
- 재구성된 프로필로 interest cluster를 re-rank해서, 사용자의 진짜 관심에 부합하는 후보군을 더 많이 후속 단계로 전달.

### Use case II — Late-stage Ranking
- Late stage는 like / comment / share / watch time / hide 등 다양한 engagement 확률을 예측하는 multi-task 모델, 결과는 value model(효용함수)로 합산.
- UTIS를 late stage 모델과 **병렬로** 두고, 그 점수를 최종 value formula F의 추가 input feature로 투입.
- 비즈니스 가치(예: 매출)와 사용자의 진짜 관심을 함께 균형 맞추도록 fine-tune.

## 4. 실험 결과

### Offline
- Baseline: 과거 engagement 기반의 휴리스틱 cluster 매칭.
- UTIS 모델 vs baseline (새로 들어오는 survey 응답 기준):
  - Accuracy 71.5% vs 59.5%
  - Precision 63.2% vs 48.3%
  - Recall   66.1% vs 45.4%
- 신규 survey로 지속 모니터링 → 운영 환경에서도 일반화 성능 유지.

### Online A/B (10M+ 사용자, 2개월+, 3회 이상 반복)
| Quality Metric | Relative Δ % |
|---|---|
| High survey ratings | **+5.4 ± 0.8%** |
| Low survey ratings | **−6.84 ± 0.8%** |
| Total user engagement | **+5.2 ± 0.8%** |
| Integrity violation | **−0.34 ± 0.1%** |

- **Tier 0 retention**(가장 중요한 리텐션 지표) 통계적으로 유의하게 상승.
- 사용자 만족도(survey self-report) 동시 상승.
- 저품질 / 무관한 콘텐츠 demote → integrity / quality 동반 개선.

## 5. 선행 연구와의 차별점
- **DFN (2021)**: explicit/implicit feedback 상호작용 메커니즘.
- **USM (2024)**: 부정 경험을 줄이기 위한 unbiased survey modeling.
- **SAQRec (2024)**: 설문지 피드백으로 만족도에 정렬.
- 위 연구들은 explicit feedback이 questionnaire만큼 sparse하지 않거나, **전반적 만족도**에 초점. 본 논문은
  - 매우 sparse한 **interest 자체에 특화된 설문**을 직접 모델링.
  - 응답을 imputation하여 ranking stack 전 단계에 enrichment로 활용.

## 6. 결론 & Future Work
- Sparse한 사용자 직접 피드백(설문)을 ranking stack의 **calibration layer**로 통합하면, 피드백 루프 편향을 줄이고 진짜 관심사 기반 개인화를 강화할 수 있음.
- 향후
  - UTIS를 ranking stack의 더 많은 단계에 end-to-end 통합.
  - "관심" 외에 **exploration**, **entertainment** 등 다른 차원의 사용자 경험에도 동일한 survey-modeling 패러다임 확장.

## 7. 핵심 takeaway
- Engagement는 진짜 관심의 **proxy일 뿐**, 둘은 자주 어긋난다 → 피드백 루프 문제.
- **소량의 explicit interest survey + imputation 모델**로 sparse signal을 dense하게 확장하면, 거대 ranking system에서도 의미 있는 retention/quality 개선이 가능.
- 무거운 메인 랭커를 다시 학습할 필요 없이, **가벼운 보조 모델 + value formula 입력** 형태로 붙이는 실용적인 패턴.
