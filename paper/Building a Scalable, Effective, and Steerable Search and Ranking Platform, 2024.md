- Zalando: 이커미스 회사
- **해당 논문은 추천시스템 전체적인 로직을 공유**

## Key Contributions

이 연구의 주요 기여는 다음과 같습니다:

**(1) 확장 가능한 랭킹 플랫폼 구축**
- 모든 랭킹 레이어에서 실시간 추론이 가능한 포괄적이고 유연한 시스템
- 최신 모델과 표준 디자인 패턴 기반으로 다양한 검색/랭킹 사례에 적용 가능

**(2) 효율적인 모델 아키텍처 개선**
- 기존 최신 랭킹 모델을 품질 저하 없이 더 효율적으로 개선
- 시퀀스 기반 모델이 모든 랭킹 단계에서 전통적 시스템을 대체하고 성능을 크게 향상시킬 수 있음을 입증

**(3) 광범위한 실험 및 성능 검증**
- 오프라인 평가 지표 10-40% 개선
- 온라인 A/B 테스트에서 사용자 참여도 15% 증가, 순수익 +2.2% 개선
- 높은 부하 상황에서도 효과적으로 확장 가능함을 입증

## system design
- 검색·추천·랭킹을 하나의 공통 플랫폼으로 통합하면서도 대규모 트래픽, 실시간성, 그리고 비즈니스 목표에 따른 조정(steerability)을 동시에 만족시키는 것

## candidate generation layer
- top500
- two-tower
- multi-class
- sampled softmax loss with log-uniform sampling, with negative classes that correspond to 0.42% of the total number of classes
- 추론시에는 user, item tower 는 독립적으로 작동
  - item vector 는 vector store 에 저장, vector 가 바뀌거나 새로운 item이 들어오면 업데이트됨
  - user vector 는 실시간으로 생성, feature store 를 실시간으로 가져오고 cache 사용

## ranking layer
- pointwise multi-task prediction
  - click, add-to-wishlist, add-tocart, purchase 들을 positive 나머지는 negative
- 구조는 논문 그림 참고
  - 주요 키워드: embedding layer (user context, user action sequence, cand items), encoder layer, position de-biasing, ranking head
- 유저 데이터
  - 행동 데이터
    - 유저 액션 시쿼스: 아이템 임베딩, 시간 임베딩, 아이템에 대한 유저액션 임베딩
  - 콘텍스트 데이터
    - 나라, 디바이스, 아이템 카테고리, 검색어, 페이지 등 임베딩 avg 해서 하나의 임베딩 생성
- position debiasing
  - train 에는 사용하고 serving 시에는 사용 못하니 모델 구조에서 구분
  - train 시에 아이템들의 position 을 알려주는 구조

## policy layer
- exploration with new items
  - epsilon-greedy exploration 으로 epsilong 의 확률로 새로운 아이템을 뽑고 나머지의 확률로 ranked list 에서 뽑음
- business heuristics
  - 최근 2개월 이내 구매한 아이템 down-sorting
  - diversity 추가: 같은 브랜드 연속 노출되지 않도록, 다른 브랜드가 연속적으로 나오도록

## model productionization
- item embedding
  - 새 모델 훈련, 새 아이템 추가, 속성 변동 -> 새롭게 임베딩 생성 -> kafka 기반으로 stream 하고 ElasticSearch 에 인덱싱
- 가장 어려운 점은 two-tower model 이 새롭게 훈련되어 나오면 전체 임베딩을 업데이트해야한다. 그래서 이전 버전도 같이 유지되면서 blue-green 배포를 한다.
- 블루-그린 배포란? 블루-그린 배포는 두 개의 동일한 운영 환경(환경 하나는 'Blue', 다른 하나는 'Green')을 유지하는 전략입니다. 
  - Blue(블루): 현재 실제 사용자에게 서비스를 제공하고 있는 기존 버젼의 환경입니다. 
  - Green(그린): 새로운 모델과 새로운 임베딩 데이터가 적용된 차기 버젼의 환경입니다.
- 배포
  1. Green 환경에 새 모델을 올림
  2. 새 모델로 모든 item embedding 재생성 + 새 벡터 인덱스(ANN index) 구축
  3. Green이 제대로 동작하는지 검증
     1. latency, QPS, 결과 품질, 에러율, 샘플 트래픽 테스트 등
  4. 문제가 없으면 트래픽을 Blue → Green으로 “스위치”
     1. 라우터/로드밸런서에서 한 번에 혹은 점진적으로 전환
  5. 문제가 생기면 즉시 Green → Blue로 롤백 (스위치 되돌리기)
- cpu 로 serving, 각 layer 마다 p99 latency 10ms

## experiments
- Retrieval에서 Recall을 보는 이유는 "어떤 아이템이 좋은지 정확히 맞히는 것보다, 좋은 아이템을 하나도 놓치지 않고 랭커(Ranker)에게 넘겨주는 것" 이 이 단계의 핵심 목표
- 지표
  - Recall@k, NDCG@k
  - Diversity: “같은 브랜드가 연속으로 얼마나 길게 나오나(max run)”를 프록시로 사용(길수록 다양성 나쁨).
  - Novelty: “새 아이템에 대한 recall”을 프록시로 사용(콜드스타트/신상품 노출 능력).
- cg 에서 trainable item embedding이 오프라인 성능을 크게 올리지만 item cold-start를 악화시킬 수 있어(그래서 policy layer에서 exploration을 넣음).
- 이질적인 입력(컨텍스트/히스토리/아이템 표현)을 early에 함께 섞는 게 성능의 핵심

## conclusion