- https://dan.naver.com/24/sessions

# 추천, 검색
### 홈피드: 네이버의 진입점에서 추천피드를 외치다! 추천피드 도입 고군분투기
- https://tv.naver.com/v/67443984
- 콘텐츠풀 + 사용자 콘텍스트 -> 리트리버 -> 랭커 -> rank1 최적화
- 리트리버
  - 구독정보, 키워드(사용자 클릭/검색에서 키워드), 인기도(선호 카테고리 인기 문서, 동일 성연령 인기 문서), 소비이력(EASE, two-tower 모델 등), 검색이력 (검색 쿼리 연관 문서)
- 사용자의 행동에 즉각 반응하여 리트리버 구축
  - AfterSearch model
    - 사용자가 하루 이내 검색한 이력과 연관된 문서 추천
    - 검색 로그 -> 키워드 추출 -> 연관 문서 선정
    - 키워드 추출
      - 키워드 중에서 탐색형/시의성 키워드 선정, 그 외에는 연관 문서 품질이 별로라 제외
      - 사용자 별 키워드 선호도 부여, 꾸준히 좋아하면 boost
      - 키워드 노출 패널티 적용, 노출 피로도 낮추기
    - 연관 문서
      - TF-IDF, TSDAE
      - 위 로직으로는 아쉬워서 유저피드백을 사용 (다른 사용자들도 좋아했던 것들)
      - LLM embedding 으로 고도화할 예정
  - Click2Click model
    - 하루 이내 클릭한 문서 중 최신 이력과 연관된 문서를 추천
    - 사용자 클릭 문서 중 최신 seed 문서 선정 -> 임베딩 모델로부터 홈피드 문서의 임베딩 벡터 -> ANN 으로 seed 와 가까운 N개 문서 선정
- LLM 활용사례
  - LLM 을 활용하여 사용자의 콘텍스트를 표현하는 태그 추출
    - 검색 로그 수집 -> LLM 으로 태그 추출 -> AfterSearch 에서 활용
  - 주제 분류기 도입 및 사용자 주제 선호도 고도화
    - 홈피드에서는 다양한 곳에서 데이터 가져와서 주제 분류 필요
    - LLM 으로 760개 주제 분류 사용
- 홈피드 추천 랭킹 로직 구성하기
  - 다양한 특징을 한 피드 안에서 랭킹 -> DCN Ranker 로 해결 (DCN V2 모델 사용한듯)
  - 유저가 클릭할 확률뿐이 아니라 만족할 확률 또한 예측하여 랭킹 -> 자체 모델 MDE ranker 사용
    - 클릭할 확률 + 체류 시간(유저만족)
    - MMoE(Multi-gate Mixture of Expert) 아키텍처 + DCN V2
  - 정확한 추천뿐 아니라 다양하게 추천 -> calibration
    - calibrated recommendation 18
    - ranker로 추천된 카테고리 분포와 실제 유저의 선호 카테고리 분포 차이에 대해 calibration
- Future work
  - 시계열 TransAct 모델 도입 (Pinterest 23년 paper)
- 사용자 맞춤형 첫번째 컨텐츠 노출하기
  - 네이버 앱에 들어오면 모든 사용자에게 홈피드 노출됨
  - 리트리버 4가지 사용
    - AfterSearch: 컨텍스트가 적은 사용자 -> 홈피드 인식 낮음 -> 현재 관심사 검색 결과로 추천
    - Click2Click: 컨텐스트가 충분한 사용자 -> 선호도 파악 가능 -> 최근 클릭한 컨텐츠와 유사한 것 추천
    - Bandit
      - epsilon-greedy algorithm 사용
      - arm: 컨텍스트가 충분하면 카테고리, 부족하면 서비스 (eg 블로그)
      - 최근 클릭한 컨텐츠의 카테고리(서비스)에 가중치를 줘서 선택될 확률 높음 (exploitation)
      - 일정 확률로 카테고리(서비스) 선택 변화 (exploration)
    - MDE ranker
  - 사용자의 행동/컨텍스트를 모델링해서 모델을 만듬: retriever bandit
    - 행동 패턴: 최근 N시간 이내 검색/클릭 횟수, 검색/클릭 이후 경과 시간, 클릭한 컨텐츠
    - 컨텍스트: 사용자의 클릭 빈도, 사용자의 세대성별 정보, 사용자의 컨텍스트(구독, 선호도)
    - LinUCB 사용 (위 유저 feature 사용하고 클릭 여부를 보상으로 사용)
    - 컨텐츠를 직접 선택하는 것이 아니라 rank1 리트리버를 선택하는 방식으로 활용
- future work
  - 사용자 모델링 고도화: 피드 사용 목적에 따른 추천 구분 (트렌드, 정보, 관심사)
  - 랭킹 모델 개선
  - 사용자 맞춤형 rank1 고도화

### 네이버 검색이 이렇게 좋아졌어? LLM의 Re-Ranking Ability 검색에 이식하기
- https://tv.naver.com/v/67444172
- 네이버 검색에서 발표팀 담당 크게 2가지
  - 롱테일/정보성: 출처 단위로 묶은 컬렉션, 사용성 기반 ML 랭커 (ex. 19개월 잠만자요?)
  - 숏헤드/탐색형: 의도 단위로 컬렉션을 분리, 개인화 고려한 뉴럴 랭커 (ex. 캠핑장비)
- 검색에서 LLM 사용의 한계
  - query -> retriever -> LLM ranker
  - list-wise: 문서1 > 문서3 ... 같은 형태
  - point-wise: 질의-문서1: 관련있음, 질의-문서2: 관련없음... 같은 형태
  - 실시간 ranking 하기에는 느리고 비싸고 관리가 어렵다.
- long-tail?
  - 사람들이 자주 검색하지는 않는 질의들
  - 이들을 해결하기 위해 여러 시도들 했음
    - 유저 피드백 데이터: long tail 이라 데이터 부족
    - 작은 llm: 성능 부족
- long tail 질의 문서 랭킹 품질 개선
  - LLM 으로는 연관성 정답데이터 labeling
    - 네이버 하이퍼클로바 모델 사용
    - query -> pre-ranker 로 topk, bottomk 뽑음 -> 이들을 LLM 으로 ranking 하라고 함 -> 낮은 애들을 negative 로 사용
  - distillation 해서 ranker 개발
    - RRA-BERT, RRA-GPT (2024 EMNLP 논문도 냈음), RRA -> re ranking agent 인듯
    - RRA-BERT 가 가장 좋아서 이를 사용
    - 실시간 서빙 관련해서는 네이버 D2 블로그 참고
- 연관성만으로는 해결하기 어려운 ranking 문제 존재, 즉 최신성도 고려해야하는 검색결과가 존재함
  - 이 과정에서 LLM 사용한 내용 공유
- 문서의 최신성을 score 화
  - age = search date - doc date
  - recency = (1 - age/max_age)^2
- 가중치는 통해서 연관성, 최신성 사이 줄타기함
  - 단, 너무 연관성이 떨어지는 경우 추가 패널티 존재

### 서치피드: SERP를 넘어 SURF로! 검색의 새로운 물결
- https://tv.naver.com/v/67444300
- SURF: Search User recommendation Feed 
  - 통함검색 하단에 개인화된 콘텐츠를 제공하는 피드
  - 목표: 피드에서 콘텐츠 발견과 탐색의 즐거움 제공
  - 검색의도 + 관심사 기반 확장된 피드, 연속된 새로운 정보 발견과 탐색 기능
- 기능1: 신선하고 다양한 파도
  - 사용자 취향과 실시간 관심사에 기반하여 다양한 츌처의 콘텐츠 제공
- 기능2: 그라데이션
  - 상단에서는 검색어와 유사하고 하단으로 내려갈수록 확장된 콘텐츠 추천
  - q=감자요리 -> 감자전 -> 감자뇨끼 -> 홈파티음식
- 기능3: 실시간 피드백 파도타기
  - 사용자 피드백으로 실시간 관심사를 반영해, 최적화된 정보 제공
- SURF 기대효과와 사용자 활동 지표
  - 주요 측정 지표: 체류 시간, 재검색 클릭, 문서 클릭
- SURF 시스템 구조
  - 다양한 리트리버로 후보 문서들 + 사용자 피드백 -> 랭커
- SURF 의 4대 파도: 연관, 반응, 확장, 개인화
- Decoder 기반 sLM 임베딩 모델 개발 -> 이걸로 4대 파도 만들자
  - 사내 LLM 모델을 백본, 사내에서 구축한 학습셋으로 SFT 진행
- 반응
  - Doc2Doc: 클릭문서와 연관된 문서
  - sLM 임베딩 사용해서 유사임베딩
- 연관
  - 짧은 질의 맥락 이해하려면? 질의를 설명하는 텍스트 추가해서 sLM 임베딩 사용 (현재 맥락텍스트는 개인화는 되어 있지 않음)
    - 맥락 텍스트: "손흥민" 이라는 질의에 "손흥민 토트넘 경기 출전" 같이 텍스트를 추가
  - 근데 최신 문서도 필요 -> 질의 최신 문서 k개 가져와서 sLM 임베딩 구해서 위해서 뽑힌 애들과 cosine similarity 구해서 cut-off, rerank 진행
- 확장
  - 손흥민 검색하면 이강인 관련 컨텐츠 나오는 것
  - knowledge graph + 맥락텍스트
  - sLM 임베딩 이용
- 사용자를 이해하는 랭커
  - 구체적인 알고리즘은 다른 발표 참고하면 될듯
- 개인화
  - 현재 개발중
  - 사용자 활동 로그 -> sLM -> 사용자 프로필 -> 개인화
- future work
  - 나의 검색을 도와주는 나만의 Agent
  - 필요한 문서 요약, 추가 질문 제안, 제품 탐색부터 리뷰/결정/구매 서포트

### 검색과 피드의 만남: LLM으로 완성하는 초개인화 서비스
- https://tv.naver.com/v/67444402
- 홈피드 개인화 추천 (홈피드: 네이버의 진입점에서 추천피드를 외치다! 추천피드 도입 고군분투기 발표와 연관)
- 기존 유저 컨텍스트의 한계
  - 홈피드 초기 높은 라이트 유저 비율 (일주일 2회 이하 클릭, 85%)
  - 근데 라이트 유저 대부분이 검색, 메인판 을 사용한 유저 -> cross domain 유저 컨텍스트 확장 해보자!
- AiRScout: LLM 기반 유저 컨텍스트 확장 모델
  - 그림을 보면 좀 더 이해 쉬움
  - 3가지 LLM 모듈, 5가지 유저 컨텍스트 생성
  - 사용자 관심 주제 추출
    - 다양한 컨텐츠에서 IAB 분류 체계(약 600개)로 주제 분류 -> 검색, 홈피드, 메인판 소비 이력으로 유저별 주제 선호도 계산 (배치)
  - 사용자 검색 의도 세분화
    - 다양한 컨텐츠에서 entity 추출 -> 검색 소비 이력으로 의도 세분화 -> 원질의에서 요약 생성 질의 생성 (실시간)
      - 에스파 -> 에스파 일본 앨범, 뉴진스 -> 뉴진스 일본 데뷔곡
  - 사용처
    - 홈피드 - 검색 이력 기반 추천 AfterSearch 모델에서 사용중
    - 서피 피드 - 숏텐츠 & 재검 질의 추천
- 사용자 검색 의도 세분화
  - 질의, 문서 를 동시 고려한 태그 생성 기술의 필요성
    - 질의만 보니 뾰족함을 놓치고 문서만 보면 맥락을 놓친다 -> 질의 q + 문서 d 를 받아서 LLM 으로 요약 생성 질의 q' 를 만들자
  - 검색 로그를 활용한 supervised fine-tuning
    - 학습데이터 수집 과정
      - 한 문서에 유입된 모든 질의 수집 -> 포함 관계인 질의쌍을 나열 (에스파 - 에스파 공연, 일본 앨범 에스파) -> 토큰 수가 가장 많은 것을 q' (에스파 일본 앨범)
    - 수천건 q, d, q' 수집 -> 작은 llm 모델로 SFT 진행
    - 근데 자주 등장하는 특정 검색 패턴에 bias 발생 (ooo 인스타, ooo 등장인물)
    - 큰 llm 모델을 이용해서 knowledge distillation
      - human labeled good dataset (수백건) -> 큰 Llm SFT -> 이를 통해 Data Augmentation (수천건) -> 작은 llm SFT
      - 오프라인 평가: Rouge, Bert Score, HCX-Eval
  - 서빙 최적화: vLLM 적용
- 사용자 관심 주제 추출
  - IAB 분류체계로 640개로 분류, LLM 모델 사용, Beam Search 를 활용한 Multi-class 분류, 여러 서비스 활용 가능한 단일 모델
  - 사용한 모델
    - HCX-Large 로 모델 평가
      - 주제분류 결과의 연관성 측면 평가는 LLM 으로 가능해보임 (논문들도 있음)
      - 논문들을 참고해서 생성 결과물 평가를 위한 프롬프트 진행, 각 주제별로 000점인지 채우는 form filling
    - HCX-Small 로 SFT 진행
      - 문서를 주고 주제 예측
      - 추론은 Beam Search
      - 학습데이터
        - 사람이 3천개 했는데 분류들이 많다보니 데이터가 부족
        - SFT 한 HCX-small 모델을 통해 3가지 weak label 을 생성하고 HCX-large 모델로 80점 이상 주제를 데이터셋으로 추가 수집
        - 다양한 주제 데이터를 더 보강하기 위해 주제를 정해주고 제목, 블로그 글을 작성하도록 해서 데이터 추가 수집
  - 주제분류 결과를 서비스에 적용하려면 일관된 응답 필요 -> Guided Text Generation 을 활용
    - Efficient Guided Generation for LLMs 2023 논문, candidate beam search 같은 것
    - 사전에 정의된 후보군 중 가능한 vocab 을 제외하고 마스킹 처리
  - 이들을 활용해서 배치로 유저 홈피드 주제 선호도 score 만들기 -> ranker 에서 user feature 로 사용
  - 세분화 주제를 사용할수록 결과가 좋았음
  - 홈피드, 검색, 메인 등을 합쳐서 통함선호도 생성 (cross-domain) -> 단, 홈피드 사용성에 따라 cross-domain 비율을 다르게 설정
    - 홈피드를 많이 사용하면 다른 도멘인 적게 활용

### 클립 크리에이터와 네이버 유저를 연결하기: 숏폼 컨텐츠 개인화 추천
- https://tv.naver.com/v/67444550
- 클릭할만한 클립 노출하고 실시간으로 흥미를 가질만한 클립 연속적으로 추천
- 시스템 구성 (two-stage system)
  - 배치&실시간으로 feature store 가 만들어지고
  - 백엔드에서 api call 이 오면 이를 이용해서 실시간으로 추천 결과 전달
    - context fetch -> candidate retrieval, feature fetch -> ranker
- 실시간 클립 시청이력 기반 개인화 추천
  - user context in feature store
    - 클립 시청 이력 데이터로 유저 데이터 만듬 (채널, 키워드, 카테고리 등), 배치로 진행
    - 실시간성
      - play quality score (pos feedback): 유의미한 시청을 판단하기 위한 점수, 재생수 대비 재생시간, 영상 시간 등 간단한 score 사용
      - skip rate (neg feedback): 재생수 대비 재생시간 고려
  - retrieval: CF based, 통계, LinUCB, LinTS
    - long term log 기반, short-term log 기반 병렬적으로 사용
  - ranker: DCN
  - 숏폼 하단으로 내려갈수록 실시간 short-term 기반 모델의 중요도가 높아짐
  - 기존에는 api 호출에 10개씩 전달했는데 이를 4개로 줄여서 더 정확하게 추천하다보니 지표 상승
- 실시간 시청이력 반영 주기 단축
  - 시청이력 stream 연동 지연 단축
    - 클라 시청 이력을 status 로 관리하여 api 파라미터로 전송하여 실시간 시청이력을 추천에 활용
- explicit negative feedback
  - 해당 컨텐츠/채널 추천하지 않기 클릭
  - negative feedback co-occurrence log 기반 CF 모델 만들어서 (EASE) 로 rank down
  - 클립/채널의 해시태그/본문내용으로 named entity 를 추출해서 흡사하게 유사 키워드를 갖는 컨텐츠의 노출 비중 조절
- 콜드 스타드 추천
  - user side 도 고려해야하지만 creator side 도 고려해야함
  - 신규 크리에이터는 user feedback 이 부족해서 contextual bandit 사용
    - cold-start item 추천에 효과적인 feature selection 이 중요
      - 대부분의 user context feature 를 활용, item feature 는 contents-based feature 위주
    - 아이템 feature weight disjoint LinUCB 사용
  - future works
    - vid-LLM 등의 모델을 도입해서 contents-feature 보완
    - kpi 다각화, 보조지표, 가드레일 지표로 활용
- 검색 이력 기반 개인화 추천
  - 검색어-클립 매칭을 위한 메타데이터 구축
    - entity 추출 -> scoring -> 형태소 분석 및 indexing
  - 매칭 알고리즘
    - 사용자 실시간 검색 이력 조회 -> 형태소 분석 -> index 조회 (retriever) -> ranking
    - retrieval: 에서는 matched term ratio 가 70% 이상 매칭되는 결과만 가져옴
    - ranking: 클립의 최신성, 질의-클립 유사도, 재생 품질 점수를 종합적으로 고려하여 ranking
  - 하지만 키워크 매칭으로만 할 경우 연관성이 떨어지는 경우 존재 -> query-category score
    - query-category score: 검색어-클릭 문서 cate, sub-cate 로 점수 계산
    - 12시간 이내의 실시간 검색 이력
  - 하지만 아쉬운 성능,,,why?
    - 클립 추천에 부적합한 질의 존재 (정보성)
    - 너무 검색어에 국한된 결과, 비슷한 내용의 반복되는 클립들
    - 사용자의 컨텍스트를 모르는 문제 (같은 질의여도 사람마다 다른 상황)
  - 이를 해결하기 위해 LLM 으로 최근 검색어를 통해 관심사 키워드를 추천
    - one shot prompt, 너무 예시 많으면 오히려 overfit
    - 한시간 동안의 질의를 묶어서 처리
    - 추가로 성능 보완을 위해 사내 named entity tagger 활용
    - 탐색형 질의 선별 (클립 추천하기 부적합한 질의 필터링), intent, topic + rule 기반
  - 대용량 llm inference
    - 매시간 400만명, 1400만개 질의 처리
    - spark cluster 사용
    - 파티션 수로 concurrence 제어해서 모든 excutor 모두 사용

### LLM 기반 추천/광고 파운데이션 모델
- https://tv.naver.com/v/67445059
- 4년 동안 진행한 연구
- 유저 로그는 모두 텍스트로 생각할 수 있음 -> 이걸로 추천/광고 파운데이션 모델 -> 범용 유저/아이템 임베딩
  - 왜 임베딩 기반?
    - 특정 서비스의 콜드 유저도 잘 다룰 수 있음
    - 컴퓨팅 자원 절약
    - 유저 & 아이템 간의 스코어 생성 가능
    - 데이터를 많이 쓰니 성능이 좋음
  - 활용
    - 시드 유저 확장, 임베딩 그냥 써도 되고 분류모델 만들어서 써도 되고 (시드 유저는 Positive)
    - 유저의 알려진 레이블과 유저 임베딩으로 이진 분류기 학습 -> 다른 유저에 대해 예측
    - text2user
- 모델 변천사
  - ShopperBERT -> CLUE -> LMRec -> SSLBomb -> OBP
- 자연어 foundation model 의 특징
  - language modeling 은 무수히 많은 multi-task learning 이다. 다음 단어 맞추기만 하려고 했는데 그냥 다됨.
  - pretrain loss 가 computation 이 늘어남에 따라 줄어들어야 한다.
    - pretrain loss 가 낮을수록 few-shot 만으로 finetuning 과 유사한 성능을 낼 수 있다.
  - 중요한간 pretrain loss 지 computation 은 아니다. scale up 자체가 목표인게 아니다.
- 유저 모델링에서 SSL 칵테일 요법
  - causal language modeling + contrastive learning + next sequence contrastive learning
  - 모델구조와 다양한 논문도 설명
- OBP
  - SSLBomb 은 foundation model 이 될 수 있는가? 조금 아쉬움
  - 모델 최대 토큰 길이 16k, 총 토큰 수 260B
  - 모델 설명
- 서비스 적용 사례 및 성과

### 내 손 안의 알딱핑! (੭˃ᴗ˂)੭ 네게 맞는 웹툰을 알아서 딱! 추천해줄게
- https://tv.naver.com/v/67445183: 강화학습을 이용한 추천 모델 + ML 플랫폼 발표
- 알아서 딱 정렬 런칭: 홈에서 요일별로 웹툰이 노출되는 것 추천 (인기도 등 룰 기반도 존재)
- 도입 배경
  - 웹툰이 많아짐에 따라 잘 추천해서 더 웹툰소비할 수 있도록
- 잘 배치해보자! 잘 읽고 있는 작품은 적절하게, 그 외 취향, 적당한 비율로
- 가설 분석
  - Markov Decicion Process 를 이용한 장기효과 분석
    - 열람 작품 수를 많이하는게 목표
    - state: 지난 주 열람 작품 수, action: 신규 열람 비율 (주간), reward: 열람 작품 수
    - 이를 통해 분석해보니 기존 읽는 작품 수, 신규 작품 수의 비율을 찾을 수 있었음 (균형을 잘 찾아야함)
- 모델
  - off-policy actor-critic recommender
    - 읽던/신규 웹툰의 극명한 coversion likelihood 차이로 인해 전통적인 MLE 방식은 어려움이 있음
      - MLE 로 하면 읽던 웹툰이 다 상위랭크
    - actor-critic 방법론 사용, 참고 논문 youtube 에서 off-policy actor-critic recommeder system 2022
      - 모델 설명 진행하심
- 서비스화
  - 연구/서비스 를 나눠서 생각하는 과정 설명
  - ML 플랫폼에 대한 설명

### 사용자 경험을 극대화하는 AI 기반 장소 추천 시스템 : LLM과 유저 데이터의 융합
- https://tv.naver.com/v/67445325
- 서비스
  - 스마트어라운드, 테마 추천, 함께 찾아본 장소, 취향 추천, 코스 추천
- 장소 추천 시스템의 구성
  - 일반적인 추천과 비슷하지만 '지역' 이라는 조건이 있음
- 장소 추천 domain 의 특징
  - 장소 검색의 long-tail problem
  - 항상 같은 지역에서 장소를 찾지 않는다
  - 유저의 취향은 자주 바뀐다 (장소 찾는 시점에 빠르게 추천 필요)
  - item 카테고리가 비교적 한정적이다
- 추천시스템에서 LLM 의 사용 case 5가지
  - 추천 사유 설명, domain 정보 추출
  - llm embedding 추출 후 벡터 기반 search 해서 추천
  - llm embedding + RS 연동
  - llm tokens + RS
  - llm as RS
- 추천 사용 설명 및 domain 정보 추출
  - poi 속성을 이용해 업체에 대한 설명 생성
  - 리뷰에서 장소 domain 에 특화된 키워드 추출
- embedding 추출 후 similarity search
  - 업체의 다양한 text 를 llm 을 통해 embedding 으로 표현
- 장소 추천에서 llm 역할
  - poi 에 대한 깊은 이해, cold-start 완화, 확정성 및 유연성
  - but user-poi 의 interaction 에 다시 집중할 필요성 있음
    - 개인화 부족, 유저 경험 무시 등의 단점
- 유저, 장소 행동
  - implicit(클릭, 길찾기, 저장, 공유 등), explicit data (리뷰, 블로그 -> 텍스트, 이미지)
- LLM 과 유저 데이터 융합
  - poi & user embedding -> feature extraction layer -> gnn layer -> poi & user embdding 재추출 -> infoNCE loss
    - 먼저 LLM 으로 poi embedding 추출, 이들로 유저 액션이 있는 poi embedding 을 평균내서 user embedding 생성
    - lightgcn 구조 사용한듯
    - infoNCE 를 통한 contrastive learning
  - 기존에 쓰던 POISAGE 보다 좋음 (pinsage 기반 임베딩)
- LLM, 유저 데이터와의 융합
  - 키워드 추출, 여행 코스 설명 요약 (prompt 사용), 코스 생성 서비스에서 적용 (코스 생성을 하는 건 아니고 뒤에 설명 생성)
- 서빙 아키텍처 설명
- vector search
  - 지역 필터 4가지가 조합된 filter query 사용 필요 (region, spot, geohash, distance)
  - opensearch, milvus, faiss -> opensearch 사용

### LLM for Search: 꽁꽁 얼어붙은 검색 서비스 위로 LLM이 걸어다닙니다
- https://tv.naver.com/v/67452448
- LLM 모델 만드는 research 발표, inference 성능 향상을 위한 과정
- 서비스 사용
  - 검색 품질 평가: 사람을 대신해야 진짜 AI지? : LLM 기반 임베딩부터 검색 품질 자동 평가 모델까지
  - 문서확장: doc2query, doc2title
    - 예를 들어, 공공 문서는 제목이 구체적이지 않아서 검색에 어려움
  - 질의정규화: 자연어 쿼리를 검색형 쿼리로 변환
  - 0건 검색어 교정
    - 오타, 외래어 표기 등 새로운 검색어 추천

### 사람을 대신해야 진짜 AI지? : LLM 기반 임베딩부터 검색 품질 자동 평가 모델까지
- 검색 품질 평가의 중요성
  - 기존에는 human annotator
  - 근데 사람마다 불일치율 존재, 비용 대비 정확도의 한계 비용이 점차 감소, 시간 대비 작업량는 peak 를 찍고 감소
- LLM 기반 embedding model
  - MTEB 리더보드 확인하면 LLM 모델들이 임베딩에서 잘함
- 상위 모델 공통적인 부분들이 존재
  - decode only pretrained 모델 사용
  - 양질의 학습 데이터 확보
    - contrastive learning 시에서 negative pair 가 너무 쉬우면 안됨
    - hard negative 만들어서 학습 진행
      - 2-step hard negative mining
        - 문서 pool 확보 -> retrieval top100 -> reranking -> rank score 낮은 경우를 hard negative
      - llm 으로 synthetic hard negative 생성 (논문 improving text embedding with llm, 2024)
  - Training objective
    - 연구마다 조금씩 다름
    - 주로 bi-directional attention 적용
    - last hidden state mean pooling 이 성능이 더 좋았음
- embedding 차원 축소
  - generative representation instruction tuning, 24 에서 generation 에 비해 embedding task 에서는 성능 손실이 적다.
  - 차원 축소하는 Linear layer 추가
- quey-document relevance labeling (그냥 쓰기는 어려웠고 좀 더 좁은 범위로 fine-tune)
  - 사람 라벨링, llm synthetic generation
    - llm prompt: 입력질의/문서유형 세분화 (정답, 리뷰, 사이트형), few shot example
  - reranker fine-tuning: llm 임베딩 모델에 linear layer 붙여서 사용
  - fine-tune 데이터가 많을수록, 복잡하고 자세한 prompt 1개의 special token 간 성능차이 없음
- 서비스 적용 사례
  - QUPID: 검색시 최상단에 보이는 지식스니펫 저품질 문서 클렌징
  - 통합검색 저품질 모니터링
- 실험중
  - user click history 를 활용한 임베딩 모델 개발 중, click 이 없는 문서도 QUPID score 활용해서 보완함
  - 질의 재작성 모델 평가, 검색어 자동완성 평가
  - multi-modality, feature 확장

### 벡터 검색의 정점에 오르다: 최적의 뉴럴 검색 엔진으로 업그레이드 하기
- https://tv.naver.com/v/67452264

### SQM으로 네이버 검색 품질 췍↗!
- https://tv.naver.com/v/67324891

# AI

### eFoundation: 상품을 딱 잘 표현하는 임베딩을 만들었지 뭐야 ꒰⍢꒱ 완전 럭키비키잔앙 ☘︎
- https://tv.naver.com/v/67444878
- 이커머스를 위한 파운데이션 모델
  - 유저, 상품, 검색, 미디어 등 다양한 도메인 존재
  - 모든 도메인의 관계를 파악하기는 어려움 -> 각 도메인 마다 foundation model 필요
  - 우리는 product embedding model 을 만들었다
- eCLIP
  - 상품명, 썸네일 이미지 쌍을 이용한 학습
  - 동일한 상품을 negative 로 처리하지 않도록 주의
  - infoNCE loss 사용, 상품 수: 330M, vocab: bert-multillingual(105k)
  - 효능
    - content 가 유사한 데이터는 가까운 embedding 을 가진다. (foundation 역할)
    - unlabeled 쇼핑 데이터를 모델이 최소 1번씩 학습한다. (pretrained 역할)
  - 다른 구성 더보기 에서 적용됨
    - 특정 상품 하단에 해당 가게의 유사 상품들 노출
- image-text contrastive learning, languange model 의 한계점
  - contrastive learning 의 단점
    - 사람과의 인식과 약간 간극이 있음, 사람은 유사성을 상대적으로 생각함, 같은 neg 여도 다 그 정도가 다르다, but 모델은 pos or neg
      - 관련 논문: vision language pre-traning by constrastive learning with cross-modal similarity regulation, ALBEF
  - languange model
    - 상품 텍스트는 일반적인 문장보다 맥락이 명확하지 않음, 명사 비율이 높음
    - 큰 모델, 더 큰 데이터가 필요
    - 10억개 이상의 상품의 임베딩, 디코딩 추론 비용이 너무 높음
- efoundation
  - efoundation 의 본질
    - 본연의 목적 representation learning
    - 상품음 여러가지 방법으로 표현 가능 (상품썸네일, 이름, 속성, 검색)
    - 잘 학습되었다면 생성도 가능해야함
    - literal content similarity -> semantic similarity 로 전환
  - 이번 발표는 model 위주, 디테일하게 발표해주심
  - BLIP 에서 영감을 많이 받았다고 함

### 2시간짜리 영상을 언제 다 보고 있지? Sinossi: 비디오 구간 분석 서비스
- https://tv.naver.com/v/67444966

### 인공지능의 마법으로 실시간 라이브 인코딩에 날개를 달다
- https://tv.naver.com/v/67446801

### HyperCLOVA X Vision: 눈을 떠라! 클로바X!
- https://tv.naver.com/v/67447111

### CLOVA-X 이미지 편집: AI가 선사하는 픽셀 마법의 세계
- https://tv.naver.com/v/67447240

### HyperCLOVA X Skill Universe: LLM 기능 확장의 새로운 지평
- https://tv.naver.com/v/67447342

### HyperCLOVA-X는 네이버 콘텐츠를 이렇게 바꿉니다. -> 지금 업무랑 관련이 있어서 들어봄
- https://tv.naver.com/v/67447431
- 플레이스 클림: AI 가 만들어주는 클립
- 홈피드 콘텐츠 꾸미기
  - 제목 개선 모델로 제목 추천, 썸네일 크롭
- LLM 콘텐츠 서비스 컴포넌트
  - LLM distillation 에 집중함, infer 비용이 크니까
  - structured data
- NEVU 라는 기능: unstructured data -> structured data 로 변환
  - 다양한 출처의 비정형 리뷰 데이터를 분석 -> 키워드 형태로 구조화
  - 키워드 추출 (LLM) -> encoding & clustering -> summarizing (LLM) -> ranking (너무 일반적인 키워드 내리기)
    - 키워드 추출시 그대로 추출, 변환없이
  - 이를 활용해서 다양한 서비스에서 활용중
- LLM distillation
  - quality, latency, scalability 고려
  - knowledge distillation
    - black-box: teacher 가 생성한 결과 따라가기
    - white-box
      - divergence minimization: 생성 확률 분포를 비슷하게
      - similarity alignment: hidden state 가 유사해지도록
  - 간단한 black-box 로 먼저 시도 -> 성공적이었음
    - 적은 수의 골드 데이터셋으로 teacher LoRA tuning -> pseudo-labeling 으로 데이터 생성 -> student full fine-Tune
    - 근데 pseudo-labeling 에 노이즈가 있어서 성능 아쉽 -> LLM 평가로 필터링 과정을 추가함 -> 성능 상승
    - teacher 로 왜 해당 label 인지에 대해서 이유도 생성하게 하고 student 가 이를 multi-task tuning 으로 같이 학습하면 성능이 더 좋다

### HyperCLOVA X Audio: 자연스러운 음성 대화를 위한 기술
- https://tv.naver.com/v/67447528

### HyperCLOVA X, MLOps로 Hyperscale AI 개발의 새로운 장을 열다
- https://tv.naver.com/v/67326721

### TaaS (Tune-as-a-service), HyperCLOVA X와 비즈니스를 잇다. : 나만의 Custom AI 만들기
- https://tv.naver.com/v/67334968

### LLM, Multi-modal Model로 Place vertical service 개발하기
- https://tv.naver.com/v/67452576
- 플레이스 ai 팀 발표
- 1.3B~7B 모델을 task 마다 만들어서 진행
- 플레이스 상세, 호텔여행 검색
  - 방문자 리뷰 요약
  - 마이크로 리뷰: 한줄소개
  - 호텔 검색 고도화: 다국어 음차 변환, 번역 변환, 검색 키워드 추출 & 스니펫 추출
  - 여행 검색 고도화: 블로그에서 poi, 테마 키워드 추출 해서 테마 방문 추천
- 모델학습은 knowledge distillation 에서 black box + CoT + rationale loss
  - teacher 가 정답, 그 근거 2가지 만들고 student 가 이를 multitask 학습
  - 정답을 먼저 그리고 근거 생성하도록 함
- 모델 최적화
  - 원래는 FFT 했으나 LoRA 로 진행
- MoE 도 진행했으나 서비스에 사용은 x
- POI platform
  - poi 관리, 유지
  - poi matching 활용한 서비스
    - 영수증 매칭, 결제내역 매칭, 블로그에서 플레이스 추출
  - how to matching? 모델링
    - 2021, supervised contrastive learning of sentence representation
    - 같은 poi 도 들어오는 데이터들이 조금씩 다른데 같은 poi 끼리 positive 가 되도록 학습
    - embedding 만으로 정확히 top1 을 찾기는 어려움 -> LLM 활용
      - 발표 자료 봐야 구조 이해가능
- 이미지 검색 솔루션
  - place 검색 결과에 따른 적절한 이미지 노출
- 관련 이슈
  - 사진탭
    - 사진탭의 필터링 조건이 세분화 되지 않음
    - 업체와 무관한 이미지 노출
  - 상단이미지
    - 좋은 퀄리티의 이미지
    - 계절변화, 이벥트 등에 따라 제공되는 이미지 변경
  - fine grained model 운영의 문제
    - 모델 추가, 운영 비용 등의 문제
- 데이터셋 생성
  - image clustering: 클래스 내에서 다양한 스타일의 이미지들을 학습셋으로 구성하기 위해 kmean clustering 사용
  - 퀄리티 학습
    - 고퀄리티 이미지에는 각 카테고리를 대표하는 special token 추가 (단풍 -> <|여행|><|실외|>단풍)
- continual learning 도입
    - layer-wise discriminative learning rate
    - DAS
- place vlm TODO
  - 이미에서 텍스트 생성, 이미지 묘사
  - 키워드 추출
  - 메뉴판 OCR 을 통한 질의응답

### 어? GPU 그거 어떻게 쓰는건가요? : AI 서빙 헤딩팟 실전 노하우 (feat. AI 이어북)
- https://tv.naver.com/v/67327010
- snow vision 팀 발표

### 속도와 효율의 레이스! : LLM 서빙 최적화의 모든것.
- https://tv.naver.com/v/67337608

# 타겟팅
### 글로벌 웹툰의 ML 기반 CRM 실전 적용기 : Uplift Model과 Survival Model을 활용한 타겟팅 고도화 (네이버 웹툰)
- https://tv.naver.com/v/67320890
- CRM 타겟팅: 비즈니스 목표를 달성하기 위해 적절한 대상자에게 CRM 액션 진행
- 기존에는 rule-based 로 진행
  - 일관성, 효용성 아쉽
  - 사후 대응성 성격이 강함
  - 비용 비효율
- 모델 기반, 사전 케어, 비용 효율성을 고려한 타겟팅
- 문제 개선을 위한 두 모델 사용
  - 열람 이탈 케어 survival model
  - 결제 전환 케어 uplift model
- 이탈 케어 활동으로 리텐션 늘려보자
  - 이탈?
    - 1. 명시적 이탈
    - 2. 암묵적 이탈: 웹툰은 여기
  - 따라서 암묵적 이탈을 명시화하는 작업 필요
    - 이탈 단위, 이탈 대상, 이탈 기준 정하기
- 분석을 통해 기준 찾는 과정 설명하심 -> 재열람 하는 유저의 80% 이상은 N일 이내에 재열람
- survival model
  - 언제까지 열람을 이어 나갈까? = 특정 시점에 이탈할 확률?
  - 모델 관련 설명
- 이탈 케어 최종 성과
  - 이탈 사전 대응 관점의 도달률, 이탈 케어 관점의 열람 리텐션 AB TEST
- 전환 케어
  - 타겟 규모를 적절하게 조절
- detargeting model
  - 타겟팅이 필요없는 sure things 유저를 제외하자
  - 고려할 점
    - 연재작 or 완결작
    - 어떤 수단으로 결제?
    - 어떤 밀도? 몰아보기, 무료 최신 회차 따라가기
  - 유저별 열람/결제/행동 시계열 + 프로필 input 을 처리하기 위한 모델 구조 사용, binary label (7일내 구매 여부)
  - 문제점
    - 확률값이 애매한 유저들이 존재
  - 해결
    - label smoothing 적용 Binary 라벨 대신 결제액으로 라벨 스무딩해서 연속적인 출력값 유도 (0~1)
    - 결제 세그먼트 별로 focal loss 적용
- targeting model
  - persuadables 를 잘 포함하자
  - uplift model 사용
  - 학습데이터 확보 전략
    - AB TEST 실험 로그 -> 가장 좋은 학습 데이터
    - 이벤트 참여 로그 -> 전환 여부만 알 수 있고 왜 타겟팅 되었는지는 모르는 데이터 -> 선택 편향 발생 -> 보정해서 사용 (IPW score)
- future work
  - 현재
    - 이탈, 전환 사전 예측
    - 비용을 고려한 타겟 유저 선정
  - 미래
    - 메시지/채널 최적화
    - 타이밍 최적화/피로도 관리
    - 비용 최적화/효과성 분석

# 기타
### 당신의 Python 모델이 이븐하게 추론하지 못하는 이유 [CPU 추론/모델서빙 Python 딥다이브]
- https://tv.naver.com/v/67452152

# platform
### AI 플랫폼에 딱 맞는 Storage: AiSuite에 JuiceFS 적용기
- https://tv.naver.com/v/67469089
