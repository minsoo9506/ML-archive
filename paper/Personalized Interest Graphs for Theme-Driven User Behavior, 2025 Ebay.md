# Personalized Interest Graphs for Theme-Driven User Behavior

- RecSys 2025, eBay
- Oded Zinman, Nazmul Chowdhury, Leandro Fiaschetti, Yuri M. Brovman, Guy Feigenblat, Yotam Eshel

## 핵심 요약

eBay 사용자는 특정 상품을 찾는 **directed intent**뿐 아니라, 하나의 테마(예: Star Wars, 시카고 불스, 피트니스)를 중심으로 **여러 카테고리에 걸친 관심사(theme-oriented interest)**를 가진다. 기존 추천 시스템은 next-click 같은 **단기 engagement**에 최적화되어 카테고리 경계를 잘 넘지 못하고 cross-category 아이템을 잘 surfacing하지 못한다.

이를 해결하기 위해 **LLM chain 기반의 end-to-end 추천 프레임워크**를 제안. 사용자를 **interest graph(관심사 그래프)**로 모델링하여 broad theme → specific shopping mission까지 다층적으로 탐색하고, **relevance(연관성)와 serendipity(우연한 발견)의 균형**을 맞춘다. 수백만 사용자·수십억 아이템 규모로 배포되었으며, eBay 홈페이지 A/B 테스트에서 **새로운 카테고리 클릭(inspiration metric) +17.02%**, 구매 +0.25%, 구매자 수 +0.14% 상승.

## 문제 정의

- **Product-oriented interest** (단일 카테고리, 예: "sneakerhead") → 탐지 쉬움
- **Theme-oriented interest** (여러 느슨하게 연결된 카테고리에 걸침, 예: 피트니스 → 장비·보충제·웨어러블·책) → 탐지 어려움
- 기존 시스템은 단기 engagement 신호에 최적화되어 카테고리 경계 안에 머무름 → serendipity 희생
- **과제**: 사용자의 핵심 관심사와 연결된 아이템을 **낯선 카테고리에서도** relevance/UX 손상 없이 surfacing

### 기존 연구와의 차별점

- Christakopoulou et al. (YouTube User Interest Journeys): 영상을 user journey로 클러스터링·LLM 라벨링했으나 **다운스트림 추천에 활용 안 함**
- Google DeepMind (LLMs for User Interest Exploration): LLM fine-tuning으로 신규 클러스터 engagement 예측했으나 **static·global interest로 개인화 부족**
- FolkScope 등 intention knowledge graph 계열: **category-rigidity** 문제로 cross-category 용도에 부적합

## 아키텍처 (Near-real-time LLM Chain)

LLM latency 완화를 위해 **near-real-time 파이프라인**으로 실행. 신규 user activity 발생 시 Kafka 큐로 트리거.

```
User Activity → User-Interest Graph Generator (LLM)
            → Interest Ranking (ML)
            → Interest-to-Recall (LLM) → query expansions
            → eBERT (embedding) → Key-Value Store
            → [Online] 검색엔진 + KNN recall → DL 랭킹
```

1. **User-Interest Graph Generator (2.1)** — 사용자 활동(특히 검색 쿼리)으로 interest graph 생성
2. **Interest Ranking (2.2)** — 그래프를 linear path(interest)로 분해 후 재방문 가능성으로 랭킹
3. **Interest-to-Recall (2.3)** — 상위 interest를 query expansion으로 변환
4. **eBERT** (eBay fine-tuned BERT)로 임베딩 생성 → KV store 저장
5. **Online**: 홈페이지 진입 시 query expansion + 임베딩 조회 → 검색엔진 + KNN으로 수백 개 후보 recall → DL 모델로 수십 개로 랭킹

### 개인화 vs Global

이 논문의 셀링포인트는 선행연구(Google DeepMind 등)의 *static·global interest*와 달리 **user-specific personalization**이라는 점. 다만 개인화된 뼈대에 global 지식이 결합되는 구조다.

| 구성 | 개인화 / Global | 설명 |
|---|---|---|
| **Graph 구조·노드** (2.1) | **개인화** | 그 유저 본인의 검색 쿼리로 생성·갱신 (*"constructed from the user's site activity"*) |
| Interest Types 35개 태그 | global 스키마 | 모든 유저 공통 태그 집합에서 선택 |
| LLM world knowledge | global | "스타워즈면 펀코팝도 산다" 같은 일반 상식은 LLM이 보유 |
| **Interest Ranking** (2.2) | **개인화 + CF** | 내 재방문 패턴으로 랭킹하되 collaborative filtering으로 grounding |
| **Query expansion** (2.3) | **개인화된 입력 + global 변환** | 입력 interest는 내 그래프에서 나온 개인화된 path. 그걸 실제 상품 검색어로 푸는 단계는 LLM의 **global world knowledge** 사용 → 즉 "무엇에 관심 있나"는 개인화, "그 관심사면 어떤 상품이 있나"는 global |
| **Item retrieval** (2.3 online) | global 인벤토리 | 수십억 상품 중 검색/KNN |

→ 한 줄: **"개인화된 그래프를, global 지식과 집단 행동(CF)으로 살을 붙여 만든다."**

## 2.1 User-Interest Graph

먼저 결과물 예시부터 보면 — 한 사용자의 검색 쿼리들이 아래와 같은 관심사 계층(DAG)으로 정리된다. (상위 = broad 관심사, 하위 sink = 실제 검색 쿼리)

```
SOURCE
├── Manga [Genre]
│   └── Collectible Cards [Collectibles]
│       └── Pokémon [TV Show] → Pikachu → "Pokémon Card"(쿼리)
└── Sci-fi Movies [Genre]
    └── Memorabilia [Collectibles]
        ├── Watches → "Terminator watch"(쿼리)
        └── Posters → Star Wars → "Star Wars Poster"(쿼리)
```

- 그래프는 **DAG(방향성 비순환 그래프)** 로 모델링
  - **상위 노드** = broad·cross-category interest
  - **하위(sink) 노드** = 가장 구체적인 interest = 사용자가 실제 제출한 검색 쿼리
  - **엣지** = interest 간 의미적 관계 (예: `Sci-fi Movies → Memorabilia`는 장르 관련 수집품 affinity로 refine)
  - 노드의 의미는 **조상(ancestor) 노드 맥락**에서 해석됨
- 각 노드의 두 가지 속성
  - **(a) Interest Name**: LLM이 생성한 open-ended 설명
  - **(b) Interest Types**: 35개 카테고리 중에서 선택된 태그 (free-form 이름에 구조 부여)

### DAG 생성 과정 (검색 쿼리 → 그래프)

핵심은 별도의 그래프 알고리즘이 아니라 **LLM의 생성 능력 자체로** 계층 구조를 짜낸다는 점이다.

1. **입력: 사용자의 검색 쿼리들**
   - 사이트 활동 중 특히 **검색 쿼리**(intent의 압축된 표현)를 입력으로 받음
   - 예: `pokemon card`, `terminator watch`, `star wars poster`, `star wars watch` 같은 raw 쿼리 묶음

2. **LLM의 step-by-step 추론** (구체적 쿼리 → broad한 관심사로 일반화)
   - **(a) Thoughts 작성**: 사용자 선호에 대한 자유로운 'thoughts'를 먼저 적음
   - **(b) Shopping mission 식별**: 개별 shopping mission과 그 밑에 깔린 passion을 찾음
   - **(c) Buyer passion 추출**: 미션들을 관통하는 핵심 buyer passion을 뽑음
   - 이 추론이 **모두 끝난 뒤에야** 그래프를 생성 (cross-category 일반화를 유도하는 reasoning 단계)

3. **출력: S-expression 형태의 DAG**
   - 계층 구조를 텍스트로 표현하기 좋은 **S-expression**(LISP 스타일 괄호 중첩) 포맷으로 출력 → LLM 구조적 출력에 적합
   - 결과물은 위 예시처럼 상위(broad) → 하위(sink=실제 검색 쿼리)로 이어지는 **DAG**

> 정리: **"검색 쿼리들을 LLM에 넣고, 점점 추상화된 관심사 계층을 추론하게 한 뒤, DAG를 S-expression 텍스트로 출력시킨다."** 비용 문제 때문에 teacher가 만든 그래프를 student가 distill하여 흉내내도록 했다(아래 참조).

### Teacher–Student 패러다임

그래프 생성은 token-intensive하고 지속적 업데이트가 필요 → 프로프라이어터리 LLM은 비용 과다. 따라서:

- **Teacher**: Gemini Flash 1.5 + in-context prompt
  - step-by-step 추론: (a) 사용자 선호에 대한 'thoughts' 작성 → (b) 별개의 shopping mission과 그 underlying passion 식별 → (c) 핵심 buyer passion 추출
  - 추론 완료 후 **S-expression**(컴팩트한 구조적 텍스트 포맷)으로 그래프 생성
- **Student**: Mistral 7B Chat + **LoRA adapter**로 teacher와 동일 포맷 S-expression 생성하도록 fine-tune
  - prompt 단순화 + few-shot 제거 → **입력 토큰 94% 감소**
  - 추론 과정 생략하고 S-expression 직접 생성 → **출력 토큰 79% 감소**
  - **33개 A100 GPU**에 배포, **하루 1,000만+ 그래프** 생성
- 그래프 zoom-in 분석을 위한 **interactive tool**도 구축

#### Distillation 방식 (어떻게 옮겼나)

soft distillation(로짓 맞추기)이 아니라, **teacher의 출력 텍스트(S-expression)를 정답으로 student를 supervised fine-tune**하는 **sequence-level distillation**.

1. **데이터 생성**: teacher(Gemini)가 `검색 쿼리 → 추론(a→b→c) → S-expression` 형태의 (입력, 정답) 쌍을 대량 생성
2. **학습**: student(Mistral 7B) + LoRA로 동일 S-expression을 생성하도록 모방학습 (전체 weight가 아닌 저랭크 adapter만 학습)
3. **압축 포인트** — 입출력을 둘 다 깎음

| 구분 | Teacher (Gemini Flash 1.5) | Student (Mistral 7B + LoRA) | 절감 |
|---|---|---|---|
| **입력** | 긴 prompt + few-shot 예시 | 예시를 weight에 내재화 → 예시 제거 | **94%↓** |
| **출력** | 추론(a→b→c) + S-expression | 추론 생략, **S-expression만** 생성 | **79%↓** |
| **추론 능력** | 매 추론마다 토큰으로 출력 | weight에 distill (출력엔 안 나타남) | — |

4. **검증**: LLM-as-a-judge로 teacher vs student 비교 → 성능 저하 미미 (4절 Table 1)

> teacher의 비싼 '사고 과정'이 student의 weight 속으로 흡수되어, student는 매번 추론을 토큰으로 출력할 필요가 없어진 게 핵심.

#### scale: 그래프 생성량은 "활동량"에서, 인덱싱 부담은 "아이템 수"에서

그래프 생성이 부담스러운 건 **상품(아이템)이 많아서가 아니라 사용자 활동에서 비롯**된다. 논문 표현 기준:

- 개념적으로 *"users are modeled as graphs"* — 한 유저의 관심사를 하나의 그래프로 표현
- 단, *"graphs must be continuously updated as users engage"* + *"pipeline is triggered upon new user activity"* → **유저가 engage할수록 그래프가 계속 재생성/갱신**됨
- 따라서 *"10 million graphs daily"*는 distinct 유저 수가 아니라, **활동에 의해 트리거된 생성/갱신 이벤트 수**(≈ 활동한 유저 × 갱신 빈도)

| 단계 | 규모를 결정하는 것 | 대응 |
|---|---|---|
| **Graph 생성** (2.1) | **사용자 활동량** (engage할 때마다 갱신, 일 1,000만+ 생성) | teacher→student distillation으로 비용 절감 |
| **Retrieval** (2.3 online) | **아이템 수** (수십억 상품) | 상품을 미리 eBERT 임베딩 → KNN 인덱스 적재 |

→ "많이 *생성되는* 건 그래프(=유저 활동)", "많아서 *인덱싱이 빡센* 건 아이템"으로 갈라서 보면 정확하다.

## 2.2 Interest Ranking

### 그래프 분해 (DAG → linear path = interest)

랭킹 단위를 만들기 위해, 가지가 갈라지는 DAG를 **source 노드에서 출발하는 직선 경로(linear path)들로 펼친다.** 이 경로 하나하나를 **interest**라고 부르며, 이것이 랭킹의 단위가 된다.

- **노드 하나 = interest가 아니라, "루트부터 그 노드까지의 경로 전체 = 하나의 interest"**
  - 노드의 의미는 ancestor 맥락에서 해석되기 때문 (`Star Wars`만으론 모호 → `Sci-fi Movies→Memorabilia→Posters→Star Wars` = "스타워즈 포스터 수집"으로 명확)
- 같은 가지에서도 **어디까지 내려가느냐에 따라 granularity가 다른 여러 interest**가 추출됨

```
Sci-fi Movies → Memorabilia → Posters → Star Wars 가지에서:

Sci-fi Movies
Sci-fi Movies → Memorabilia
Sci-fi Movies → Memorabilia → Posters            ← coarse-grained
Sci-fi Movies → Memorabilia → Posters → Star Wars ← fine-grained
```

→ 이렇게 세밀한 경로와 coarse한 경로가 **모두 랭킹 후보**가 된다 (Figure 2 예시).

### 랭킹

- **collaborative filtering**으로 실제 사용자 행동에 anchoring하고, **cross-category interest에 우선순위** 부여

목표는 "사용자가 **어떤 interest에 다시 engage할지**"를 점수화하는 것. LLM이 생성한 자유로운 interest를 **실제 재방문 데이터로 grounding**하는 단계다.

**문제 설정**: pointwise 이진 분류 (interest 단위로 "재방문할까?" 0/1 예측) → 예측 확률로 interest들을 **랭킹**

| 구분 | 내용 |
|---|---|
| **단위(unit)** | interest 하나 = linear path 하나 (위에서 분해한 경로) |
| **Input (X)** | `user context` + `해당 interest(path)` — 즉 "이 사용자에게 이 관심사를 보여줄까?"의 (사용자, 관심사) 쌍 |
| **Output (y)** | 그 interest를 **마지막 세션에서 다시 engage했는지** 여부 (1/0) |
| **Model** | **XGBoost** (gradient boosted tree) — 임베딩 기반 딥러닝이 아니라 경량 tree 모델 |

**라벨 생성 (학습 데이터 만드는 법)**

1. 과거 사이트 활동으로 user-interest 그래프 생성 → linear path로 분해
2. 각 **sink node(=실제 활동/쿼리)**에 거기까지 도달한 **path를 태그**로 부착
3. 사용자 활동을 시간순으로 자르되 **마지막 세션을 떼어내** prediction target으로 사용
4. 마지막 세션 직전까지의 활동으로 만든 interest들 중, **마지막 세션에서 다시 등장한 interest = 양성(1)**, 나머지 = 음성(0)
   - → "그 전에 보이던 관심사가 다음 세션에서도 이어지는가"를 self-supervised하게 라벨링

**Feature 예시**

- 해당 interest를 루트로 하는 **subgraph의 sink node 개수** (그 관심사가 사용자 활동에서 얼마나 "두꺼운지" = 빈도/폭의 신호)
- 그 외 user context 기반 feature들

**Cross-category 촉진**

- 카테고리 경계를 넘는 interest를 더 잘 올리기 위해 **sample weighting** 적용 (예측 대상 interest가 새 카테고리로 넘어가는 케이스에 가중치)

**성능**: NDCG·MRR 기준, **activity recency/frequency 베이스라인 대비 11% 향상**

## 2.3 Interest-to-Recall

- 상위 interest 각각을 LLM에 개별 입력 → **k=10개 query expansion** 출력
  - cross-category 상품 포함하도록 설계, e-commerce 인벤토리에 정렬
  - 예: `Sci-fi Movies→Memorabilia→Star Wars` → "Star Wars Funko Pops", "Star Wars Lego Sets", "Star Wars The Black Series Action Figure" 등
- 프롬프트가 짧고 단순하며, 시간이 지나면 신규 path가 드물어져 호출 수렴 → teacher-student 없이 **Gemini Flash 1.5 직접 사용(in-context)**

### query expansion → 임베딩 → retrieval 흐름

각 query expansion **문자열 하나하나**가 eBERT를 통과해 임베딩 벡터가 되고, **그 벡터가 곧 KNN 검색의 query 벡터(seed)**가 된다. 별도의 seed 개념이 아니라 **검색어 임베딩 = seed**.

```
interest: Sci-fi Movies→Memorabilia→Star Wars
   │ (Interest-to-Recall LLM, k=10)
   ├─ "Star Wars Funko Pops"        ─eBERT→ [벡터] ─┐
   ├─ "Star Wars Lego Sets"         ─eBERT→ [벡터] ─┤ 각 벡터가
   ├─ "Star Wars Black Series ..."  ─eBERT→ [벡터] ─┤ KNN query(seed)
   └─ ... (총 10개)                  ─eBERT→ [벡터] ─┘
```

- **핵심**: eBERT가 **query 쪽과 item(상품) 쪽을 같은 임베딩 공간**에 매핑 → eBay 인벤토리 상품들도 미리 eBERT로 임베딩되어 KNN 인덱스에 적재됨
  - seed 벡터 → eBay KNN 서비스로 ANN(approximate nearest neighbor) 검색 → 가까운 **상품 임베딩**들을 후보로 recall
- 같은 query expansion을 **두 갈래로 동시에** 활용 (둘 다 KV store에 저장 후 online에서 조회)

| 형태 | 용도 |
|---|---|
| **raw 문자열** | eBay 검색엔진(lexical search)으로 후보 fetch |
| **eBERT 임베딩** | KNN 서비스로 ANN 검색 (semantic, seed 역할) |

→ 키워드 매칭으로 놓치는 상품을 semantic 임베딩이 보완. recall된 수백 개 후보는 online DL 모델로 수십 개로 최종 랭킹.

## 3. Online A/B Test

- 홈페이지 모듈 **"Inspired by Your Interests"**에서 2주 A/B 테스트
  - Control: eBay 기존 cross-category 알고리즘 (structured data 기반)
  - Treatment: 제안 방법
- 결과
  - **Inspiration metric(미탐색 카테고리 아이템 클릭) +17.02%**
  - 구매 +0.25%, 구매자 수 +0.14%
- novelty와 relevance는 보통 trade-off지만, 홈페이지 추천 **동질성(homogeneity)을 줄여 둘 다 향상** 가능성

## 4. Offline Evaluation

- 핵심 모듈인 user-interest graph 품질·안전성을 **LLM-as-a-judge (GPT-4)**로 평가
  - 그래프가 크므로(노드 수십 개) **subgraph로 분해**하여 judge에 제시
  - 평가 항목: 그래프 품질, hallucination 탐지(무관 엣지·잘못 분류된 interest type), unsafe 추천 flagging, **task alignment**(generic 행동이 아닌 deep passion 반영 여부)
- 도메인 전문가 응답과 **강한 상관관계** → 자동 judging 신뢰성 확인
- Table 1: teacher vs student 오류 비교 → student의 성능 저하는 **미미(minor)**

| Model | Edge | Node | Task Alignment | Safety |
|---|---|---|---|---|
| Teacher | 6.61 | 33.70 | 24.21 | 0.02 |
| Student | 7.33 | 34.01 | 23.23 | 0.02 |

## Conclusion

- 사용자의 theme-oriented interest를 **그래프로 모델링**한 확장 가능한 LLM 기반 추천 시스템
- **multi-level user modeling + collaborative filtering + LLM query generation**을 결합해 relevance와 serendipity의 균형 달성
- cross-category 탐색을 지원하며, A/B 테스트에서 engagement·신규 콘텐츠 발견·상업적 가치 모두 향상

## 핵심 takeaway

- **단기 engagement 최적화의 한계**(카테고리 고착)를 LLM의 world knowledge + 그래프 구조로 돌파
- **DAG 다층 구조**가 broad theme ↔ specific mission 간 traversal을 가능케 함 (relevance/serendipity 다이얼)
- 비용 문제는 **teacher-student + LoRA + S-expression**으로 해결 (입력 94%·출력 79% 토큰 절감, 일 1,000만 그래프)
- LLM의 자유로운 생성을 **collaborative filtering 랭킹**으로 grounding하여 환각·비현실 interest 억제
