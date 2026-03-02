# Cold Starting a New Content Type: A Case Study with Netflix Live

- RecSys 2025, Netflix
- Yunan Hu, Mark Thornburg, Mario Garcia Armas, Vito Ostuni, Anne Cocos, Kriti Kohli, Christoph Kofler, Rob Saltiel

## 핵심 요약

Netflix가 기존 VOD 추천 시스템에 **라이브 이벤트**라는 완전히 새로운 콘텐츠 유형을 cold-start하는 방법을 다룬 논문. 4가지 알고리즘 개선(Feature/Label Engineering, Model Architecture Enhancement, Reward Innovation, Exploration Enrichment)을 통해 **라이브 engagement +20% 향상**을 달성하면서 기존 비즈니스 지표에는 부정적 영향 없음.

## 문제 정의

- 라이브 이벤트는 기존 VOD와 근본적으로 다름: **실시간 시청 시에만 최대 가치**, 시간/장소 종속적, fandom/경쟁/소셜 요소 포함
- 방송 시작 전에 추천해야 하지만, 너무 일찍 추천하면 잊혀짐 → **타이밍이 핵심**
- 이벤트 메타데이터(장소, 시간, 선수 로스터 등)의 급격한 분포 변화에 적응 필요
- 비정기적 대규모 단독 이벤트 위주 → 한 이벤트의 실험 결과가 다음 이벤트에 일반화되기 어려움

# Introduction

- Netflix는 VOD(TV/영화), 게임, 라이브 이벤트를 제공하는 종합 엔터테인먼트 서비스
- "Jake Paul vs. Mike Tyson" 라이브 이벤트가 6,500만 동시 스트림 달성
- 라이브 콘텐츠는 실시간 경험 시 최대 가치를 제공하며, 문화적 zeitgeist를 포착
- 기존 cold-start 문제가 **완전히 새로운 콘텐츠 유형** 도입 시 더욱 심화됨

# Related Work

## Content-based Approaches

- **Content embedding**: 아이템 메타데이터로부터 neural embedding을 추출, 유사 아이템 사용자에게 신규 아이템 추천
- **Two-tower model**: 사용자-아이템을 동일 feature space에 임베딩, nearest-neighbor search로 cold-start 아이템의 관련 사용자 탐색
- **LLM 기반**: 아이템 메타데이터 + verbalized 사용자 상호작용을 프롬프트로 제공하여 관련성 예측

## Exploration-based Approaches

- **Contextual multi-armed bandit**: 아이템 메타데이터, 사용자 피처, exploration 파라미터 기반 정책 학습
- **Reinforcement learning**: explore-and-exploit 프레임워크로 사용자 trajectory의 기대 보상 최대화
- **Meta learning**: 기존 카탈로그 상호작용 데이터로 사전학습 + 신규 cold-start 아이템에 대한 빠른 일반화 최적화

# Methodology

4가지 접근법으로 구성:

## Feature and Label Engineering

- 라이브 이벤트의 고유 차원(fandom, 경쟁, 소셜 상호작용, 실시간성)을 반영하는 **새로운 피처** 설계
- 사용자의 다양한 라이브 참여 방식을 반영한 **새로운 라벨** 생성:
  - 실시간 시청, 다시보기, 좋아요/싫어요, 친구 공유, 리마인더 요청, 프리뷰 시청, 워치리스트 추가
- **Prelaunch label**: 방송 시작 전 프리뷰 경험에 대한 사용자 피드백을 인코딩 (핵심 기여)
  - 라이브 이벤트는 방송 전까지 재생 불가 → 기존 cold-start 방식의 효과 제한적
- Ablation study에서 prelaunch label과 sampling 개선이 타겟팅/적시성 목표 달성의 핵심임을 확인

## Model Architecture Enhancement

- Netflix Homepage 추천 시스템: **title-level + row-level ranking model**로 구성
- 라이브 이벤트 포함을 위한 최적화:
  - Label sampling, feature selection, layer composition, training optimizer의 하이퍼파라미터 최적화
- Netflix **foundation model**에 cold-start 강화 적용:
  - **Semantic embedding infusion**: 출연진, 선수 로스터, 언어 등 사전학습 지식 활용 강화
  - 사전학습 시 **data sampling 방법** 조정

## Reward Innovation

- Reward model `f(u, a) → R`: 사용자 u의 라이브 이벤트 a 상호작용을 **장기 만족도 기반 스칼라 보상값**으로 매핑
- 단기 + 중기 피드백 시그널 조합 활용: 타이밍, 행동 패턴, 이벤트 즐거움 정도 고려
- 핵심 보상 컴포넌트: **member anticipation(기대감)** 추정 → 방송 전 관심 있는 사용자에게 추천 유도
- 추정된 보상으로 **model alignment** 수행 → 라이브 포함 전체 콘텐츠의 장기 만족도 최적화

## Exploration Enrichment

- 기존 **contextual multi-armed bandit** 알고리즘에 라이브 전용 휴리스틱 추가
- Explore/exploit 트레이드오프 시 고려 요소: 콘텐츠 피처, 사용자 친밀도, **타이밍, 배치, 긴급도**
- UI 개선:
  - 홈페이지에 **"Live on Netflix" 행** 추가
  - 라이브 타이틀 전용 **콘텐츠 뱃지** 도입

# Experiments

- 2024년 다양한 라이브 이벤트 동안 **8개의 온라인 실험** 수행
- 190개국 이상, 수천 종의 디바이스 지원
- False positive 방지를 위해 **반복 테스트 + experiment holdback** 활용
- 대규모 이벤트에서 핵심 알고리즘 검증, 소규모 이벤트에서 하이퍼파라미터 튜닝

## Offline Evaluation

- Sequential recommendation benchmarking task 활용
- 기준 시점 t 이전 최대 2년 데이터로 추천 생성 → 미래 [t, t+w] 구간 engagement과 비교
- 평가 지표: **Recall@k, MRR, reward-aligned MRR**
- 결과 (H2 2024):
  - Model Architecture + Label Engineering (light): Overall Recall +1.02%, **Live Recall +193.62%**
  - Model Architecture + Label Engineering (medium): Overall Recall -1.02%, **Live Recall +163.82%**

## Online Experimentation

- 주요 평가 기준: 라이브 engagement 증가 + 기존 스트리밍 지표 미훼손

| Experiment | Methods | Live Engagement Lift | Significance | Overall Satisfaction |
|---|---|---|---|---|
| 1 | Feature Engineering, Reward Innovation | >5.0% | significant | no significant change |
| 2 | Reward Innovation, Exploration Enrichment | >10.0% | significant | no significant change |
| 3 | Feature Engineering, Reward Innovation | >10.0% | significant | no significant change |
| 4 | Label Engineering, Model Architecture Enhancement | >10.0% | significant | no significant change |
| 5 | Label Engineering, Model Architecture Enhancement | >0.0% | not significant | no significant change |

- **종합 결과: 라이브 engagement +20% 증가, 전체 회원 만족도에 부정적 영향 없음**

# Conclusion

- 새로운 콘텐츠 유형의 cold-start는 산업용 추천 시스템의 도전적 과제
- 4가지 접근(Feature/Label Engineering, Model Architecture, Reward Innovation, Exploration Enrichment)으로 적시성 있고 타겟팅된 추천 달성
- 라이브 포함 전체 콘텐츠에 대한 장기 만족도 최적화 가능
- **별도의 추천 시스템 없이 기존 시스템을 확장**하여 새로운 콘텐츠 유형을 통합 가능
- 다른 산업의 cold-start 이니셔티브에도 적용 가능한 접근법
