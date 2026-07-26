# You Say Search, I Say Recs: A Scalable Agentic Approach to Query Understanding and Exploratory Search at Spotify

- **저자**: Enrico Palumbo, Marcus Isaksson, Alexandre Tamborrino et al. (Spotify) — 저자 16명
- **연도**: 2025 (RecSys '25, Prague)
- **링크**: https://dl.acm.org/doi/10.1145/3705328.3748127
- **분량**: 5쪽 industry short paper

---

## 0. 한 줄 요약

**"검색이랑 추천은 사실 같은 동전의 양면인데(Belkin & Croft 1992), 시스템은 따로 놀고 있다."**

Spotify는 *exploratory query*("new releases for me", "italian 80s nostalgia")를 처리하기 위해 **LLM 라우터가 쿼리 의도를 파악해 검색/추천 툴로 라우팅**하는 에이전트 시스템을 배포했다. 핵심은 **PFR(Parallel Fusion Router)** — 풀 에이전트 오케스트레이션의 유연성과 단순 라우터의 효율 사이 절충안. Post-training으로 작은 LLM에 증류해 **latency -60%, cost -99%**로 프로덕션(~450ms p75) 투입.

---

## 1. 문제 정의

### 1.1 Narrow intent vs Exploratory intent

| | Narrow | Exploratory |
|---|---|---|
| 예시 | "Bohemian Rhapsody", "Queen" | "italian 80s nostalgia", "new releases **for me**", "composers **like** Mozart" |
| 성격 | 특정 엔티티 타겟 (navigational) | vibe / 장르 / 시간 맥락 등 broad preference |
| 잘 푸는 시스템 | **Search** (lexical/semantic matching) | **Recommendation** (user-item, item-item 시그널) |

Exploratory intent는 선행 연구에서 'non-focused intent' 또는 'intrinsically diverse query'로도 불린다. **후보 풀이 넓고 유저 선호에 크게 의존**하므로 사실상 추천 태스크에 가깝다. 심지어 어떤 쿼리는 **추천 의도를 명시적으로 호출**한다 ("new podcasts *for me*").

### 1.2 기존 검색이 exploratory에 약한 이유

```
검색 시스템: lexical/semantic matching으로 아이템 retrieve하도록 학습
            → 개인화는 re-ranking 단계에서야 들어감
            → "for me", "like X" 같은 의도를 retrieval에서 못 살림

추천 시스템: 유저 선호/아이템 유사도를 추론하도록 학습
            → exploratory엔 적합하지만, 쿼리를 입력으로 받는 구조가 아님
```

→ 둘을 **쿼리 의도에 따라 연결해주는 레이어**가 필요하다.

### 1.3 에이전트를 쓸 때의 현실적 제약

LLM + tool calling(Toolformer, Gpt4tools 등)은 자연스러운 해법이지만:

```
① 범용 오케스트레이터 (ReAct 계열)
   유연하지만 LLM ↔ 툴 사이 sequential call 다중 발생 → latency 폭발

② 단순 LLM 라우터
   쿼리를 가장 적합한 툴 하나로 보냄 → 빠르지만 표현력 부족
   (의도가 애매한 쿼리를 하나로 강제 축소)
```

Spotify처럼 **수백만 유저 × 다수 마켓**에서 개인화된 결과를 내야 하는 환경에선 ①이 불가능하다.

---

## 2. 핵심 아이디어: Parallel Fusion Router (PFR)

**①과 ② 사이의 중간 아키텍처.** 라우터가 툴을 **여러 개 병렬로** 호출하고, 각 결과를 **SERP의 서로 다른 섹션**에 매핑한다.

```
                    ┌──────────────┐
   User Features ───┼──────────────┼────────────────┐
                    │              │                │  (downstream로만 전달)
                    │  Query       │                ▼
   User ──query───► │  Understanding│         ┌───────────┐
                    │  ┌─────────┐ │ params   │  Route R_i │
                    │  │LLM Router├─┼─────────►│ ┌────────┐│      ┌──────────┐
                    │  └────┬────┘ │          │ │Section1││      │Structured│
                    │       │      │          │ │ Tool 1 ││ ───► │   SERP   │
                    │  ┌────▼────┐ │          │ ├────────┤│      └──────────┘
                    │  │  Cache  │ │          │ │Section2││
                    │  └─────────┘ │          │ │ Tool 2 ││
                    └──────────────┘          │ └────────┘│
                                              └───────────┘
```

### 2.0 용어 정리: Route ≠ Tool (헷갈리기 쉬움)

```
Route R_i  =  (Section 1, Tool 1) + (Section 2, Tool 2) + ...
              └─ 어떤 툴을 부르고, 결과를 SERP 어느 섹션에 넣을지 ─┘
                              │
                         Tool = 실제 모델/시스템
```

- **Route**: 미리 정의된 **툴 호출 묶음 + SERP 배치 계획**. 그 자체는 모델이 아니라 **설정(configuration)**에 가깝다. Pre-fusion이므로 이 조합이 사전 고정됨(§2.2). Route 하나가 툴 하나만 가질 수도(단일 호출), 여러 개를 가질 수도(병렬 멀티콜) 있다.
- **Tool**: 실제로 결과를 만드는 모델/서비스. **이종(heterogeneous)** — 전통 ML 모델일 수도, LLM 기반 sub-agent일 수도 있다(§2.4).

### 2.1 동작 방식

- **단일 호출**: 툴 하나가 쿼리를 처리 → 결과 그대로 노출
- **다중 호출**: 여러 툴을 **병렬로** 발행 → 각 결과를 SERP의 **다른 섹션**에 배치

**예시 — "latest albums by Lady Gaga"** (의도가 애매함)

| 가설 | 대응 툴 호출 | SERP 섹션 |
|---|---|---|
| 최신 앨범을 찾고 싶다 | 최신 릴리즈 검색 | Latest albums |
| 이미 릴리즈됐는지/날짜 확인 | 릴리즈 정보 조회 | Upcoming releases |
| "album"을 네비게이션 단서로 쓴 것 | 앨범 엔티티 검색 | Latest singles 등 |

→ 세 개를 **동시에** 트리거해 각각 별도 섹션으로 정리.

> **핵심 가치**: 가장 유력한 의도에 대한 정확도를 올리는 동시에, **덜 명시적인 대안 의도까지 하나의 응답 안에 coherent하게 담는다.** 단일 라우터가 못 하는 부분.

### 2.2 Pre-fusion vs Post-fusion

| | 방식 | 장점 | 단점 |
|---|---|---|---|
| **Pre-fusion** ✅ | 멀티콜 route를 **번들로 미리 정의**. LLM은 알려진 route의 **파라미터만 생성** | output token 최소화 → 빠르고 쌈 | route 조합이 고정 |
| Post-fusion | LLM이 런타임에 툴을 **동적으로 선택·조합** | 유연 | latency 비용 |

→ **Spotify는 pre-fusion 채택.** 시스템 반응성을 유지하면서 확장성 요구를 만족시키기 때문.

### 2.3 캐싱: 유저 피처를 라우터에 안 넣는 이유

```
LLM 라우터 입력 = query만 (user features 제외)
                        │
                        ▼
              동일 쿼리 → 캐시 히트
              → cache hit rate 극대화, 연산 비용 대폭 절감

user features = downstream 툴로만 전달
                → 거기서 개인화/문맥화 수행
```

라우팅 결정은 "이 쿼리가 무슨 의도인가"에만 의존하고 개인화는 아래 단계에서 처리한다는 **책임 분리**. 이게 캐시 가능성을 만들어낸다.

### 2.4 툴 구성 (하이브리드)

| 툴 종류 | 예시 | LLM 호출? |
|---|---|---|
| 전통 ML 검색/추천 리트리버·랭커 | 검색용 retriever, user-item / item-item 유사도 모델 | ❌ |
| **Sub-agent** | **AI DJ** — 쿼리 기반으로 음악 세션을 on-the-fly 생성 | ✅ |

LLM sub-agent는 **꼭 필요할 때만** 호출된다. 논문이 든 조건:

1. 외부 지식이 필요할 때
2. 복잡한 추론이 필요할 때
3. relevance / content diversity 같은 **SERP refinement**가 필요할 때

> **왜 아껴 쓰나**: 450ms p75 예산 때문. 모든 route가 LLM sub-agent를 물고 있으면 라우터를 아무리 작게 증류해도 소용없다 — **비용이 라우터가 아니라 아래쪽에서 터지니까.** Table 2에서 broad music search만 AI DJ로 처리된다고 명시된 걸 보면, 정말 *생성*이 필요한 use case에만 배치한 듯. 반대로 similar artists(+115%)는 전통적인 item-item 유사도 모델로 충분한 영역.

### 2.5 라우터가 파라미터까지 생성한다

라우터는 route 선택뿐 아니라 **툴 호출의 최적 파라미터화**까지 담당한다.

#### "파라미터"가 뭔가

**downstream 툴(API)에 넘길 인자(arguments).** 함수 호출의 인자와 정확히 같은 의미다. Downstream 툴들은 이미 존재하는 검색/추천 시스템이고, 각자 정해진 시그니처를 갖는다:

```
personalized_recs(genre=?, entity_types=?, max_weeks_from_release=?, ...)
similar_artists(seed_artist=?, ...)
```

라우터가 하는 일 = ① 어떤 route를 부를지 + ② **그 툴의 인자를 뭘로 채울지**

**예시 — "new indie rock releases"**

```json
{
  "route": "personalized_recs",       // ① 어떤 툴
  "genre": "indie rock",              // ┐
  "entity_types": ["track", "album"], // ├ ② 인자들
  "max_weeks_from_release": 4         // ┘
}
```

#### 인자를 채우는 것이 곧 query understanding

| 쿼리의 조각 | 채워진 인자 | 내부에서 일어난 일 |
|---|---|---|
| "indie rock" | `genre: "indie rock"` | **facet extraction** — 자유 텍스트에서 장르 패싯 추출 |
| "new" | `max_weeks_from_release: 4` | **정규화/rewriting** — 모호한 "new"를 구체적 수치로 변환 |
| "releases" | `entity_types: ["track","album"]` | **expansion** — 릴리즈가 트랙일 수도 앨범일 수도 |

핵심은 **"new" → `4주`** 부분. 검색 인덱스는 "new"라는 단어를 이해 못 한다. 누군가는 "new = 최근 4주"라고 결정해줘야 하고, **그 판단을 LLM이 한다.** 맥락에 따라 달라질 수도 있다 — 팟캐스트의 "new"와 클래식 음반의 "new"는 다른 기간일 테니.

#### 기존 방식과의 대비

전통적으로는 **인자마다 전용 모델**이 필요했다:

```
쿼리 ─┬─► 장르 분류기           → genre
      ├─► freshness 의도 분류기  → max_weeks_from_release
      ├─► 엔티티 타입 분류기     → entity_types
      └─► ... (인자 추가될 때마다 모델 하나씩)
```

각각 학습 데이터 만들고 학습·배포·모니터링해야 한다. LLM 라우터는 이걸 **단일 구조화 출력 하나로 통합**한다. 새 인자는 스키마에 필드 하나 추가하면 끝.

> **Pre-fusion과의 연결**: route 조합이 이미 고정돼 있으므로 LLM이 생성하는 출력은 사실상 **JSON 몇 필드**뿐이다. 출력 토큰이 적으니 빠르고 싸다. 논문 표현으로 *"LLM only needs to generate the parameters for a known route, minimizing the number of output tokens."*

각 route는 **SERP의 특정 시각적 구성요소**와 연결된다. broad한 추천 의도라면 콘텐츠 탐색을 지원하는 레이아웃(더 풍부한 시각 요소, 문맥화)이 필요하기 때문.

---

## 3. Post-training: 작은 LLM으로 증류

캐싱 외에 확장성의 또 다른 축 — **작고 효율적인 LLM으로 높은 라우팅 정확도 달성하기**.

```
                    ┌──────────┐
   Training Data ──►│ Teacher  │──sampling──► Route_1 → SERP ─┐
   (real+synthetic) │LLM Router│              Route_2 → SERP ─┼─► LLM-as-a-judge
                    └──────────┘              Route_n → SERP ─┘   (Reward Model)
         │                                                          │
         │                                              PASS 받은 것만 채택
         │                                                          │
         └────────────────► LLM Router (small) ◄───target───────────┘
```

**절차 (Rejection sampling Fine-Tuning, RFT)**

1. 학습 데이터셋 구성 — **real + synthetic** 예제, 평가셋과 분리(unbiased assessment)
2. 강력한 LLM을 **teacher**로 두고, task instruction + 수작업 few-shot이 담긴 프롬프트로 정답 라우팅 생성
3. **high temperature**로 샘플링 → 다양한 후보 route 확보
4. **LLM-as-a-judge**를 reward model로 써서 저품질 샘플 필터링
5. **PASS 받을 때까지 샘플링 계속** → 통과한 응답을 fine-tuning 데이터 포인트로 사용

> **결국 하는 일**: teacher로 라벨 만들고 judge로 거른 뒤 **small 모델에 지도학습(SFT)**. 학습되는 건 **student 라우터 하나뿐**이고 teacher와 judge는 프롬프팅만 하는 고정 부품이다.
>
> - 라벨 = 라우팅 결정 JSON (§2.5). 입력=쿼리, 출력=JSON인 평범한 seq2seq SFT.
> - "RFT", "reward model"이라는 이름 때문에 RLHF/PPO를 떠올리기 쉽지만 **강화학습이 아니다.** judge를 통해 gradient가 흐르지 않고, reward는 **데이터 필터로만** 쓰인다.
> - **student가 teacher를 +3% 넘어서는 이유**: 학습하는 게 `teacher 평균 품질`이 아니라 **`teacher best-of-N`**(judge 통과분)이기 때문.

### Table 1. Teacher LLM 대비 offline 상대 개선

| | LLM router (prompting) | **LLM router (post-training)** |
|---|---|---|
| Quality | -5% | **+3%** |
| Latency | -53% | **-60%** |
| Cost | -93% | **-99%** |

- 작은 모델 + 프롬프팅만 해도 이미 latency/cost는 크게 줄지만 **품질이 5% 손해**
- Post-training하면 **품질이 teacher를 오히려 소폭 상회**하면서 cost -99%
- 같은 모델의 prompt-based 대비로도 더 빠르고 싼데, **input context가 줄어들기 때문** (few-shot 프롬프트가 필요 없어짐)

---

## 4. 평가

### 4.1 LLM-as-a-judge (offline 주력)

**실시간 세계 지식에 접근 가능한** 강력한 LLM을 judge로 사용.

| judge 입력 | judge 평가 축 |
|---|---|
| query | relevance |
| user profile | diversity |
| SERP에 반환된 아이템들 + 메타데이터<br>(title, artist, genre, release date) | freshness |

- 평가 프롬프트 = 명확한 instruction set + 수작업 few-shot
- 출력: **absolute score**(PASS / NO PASS) 또는 **pairwise preference**(X가 Y보다 낫다)
- 테스트셋: real + synthetic query-user 쌍 blend → 다양한 use case와 검색 전략 커버

### 4.2 Online

- 유저 인터랙션 시그널 — **click, stream**
- 운영 guardrail로 **latency와 cost를 상시 모니터링**

---

## 5. 결과

### 5.1 배포

- 주요 영어권 국가, **수백만 active user** 대상 배포
- 프로덕션 지연 기준 충족 — **~450ms p75**
- → 확장성과 비용 실현 가능성 모두 확인

### 5.2 Table 2. 일반 검색 대비 offline 상대 개선

| Use-case | 예시 쿼리 | Improvement |
|---|---|---|
| Finding similar artists | "find artists similar to X" | **+115%** |
| New music releases search | "new albums by lady gaga" | **+91%** |
| Broad music searches | "morning motivation with my favorite upbeat tracks" | **+25%** |
| Broad podcast search | "english lessons for spanish speakers podcast" | **+15%** |

- **similar artists(+115%)가 압도적** — 전형적인 item-item 추천 태스크인데 기존 검색이 가장 못 하던 영역
- Broad music search는 **AI DJ가 sub-agent로 동작**해 쿼리 기반으로 음악 세션을 on-the-fly 생성
- 여러 플랫폼에 롤아웃 완료, 메인 앱으로 확대 중

---

## 6. 핵심 기여 & 인사이트

1. **PFR 아키텍처** — 풀 오케스트레이션(유연/느림) ↔ 단일 라우터(빠름/경직) 사이의 실용적 절충. 병렬 멀티콜 + SERP 섹션 매핑으로 **애매한 의도를 축소하지 않고 여러 가설을 동시에 제시**
2. **Pre-fusion 선택** — LLM이 route를 조합하지 않고 **파라미터만 생성** → output token 최소화가 곧 latency/cost 절감
3. **query만 라우터에 넣는 캐싱 설계** — 라우팅(쿼리 의존) ↔ 개인화(유저 의존) 책임 분리가 캐시 히트율을 만들어냄. 개인화는 downstream 툴에서
4. **RFT 기반 post-training** — LLM-as-a-judge를 reward model로 쓴 rejection sampling으로 teacher 성능을 유지하며 cost -99%
5. **LLM이 다중 classifier/re-ranker를 대체** — rewriting, expansion, facet extraction을 툴 호출 파라미터화 하나로 통합

### 배울 점 / 생각할 거리

- **"에이전트 = 느리고 비싸다"는 통념을 실제 프로덕션 제약(450ms p75) 안에서 깬 사례.** 그 방법이 대단한 게 아니라 ⓐ route를 미리 번들로 고정하고 ⓑ 유저 피처를 라우터에서 빼서 캐싱하고 ⓒ 작은 모델로 증류한 것 — **전부 "LLM이 생성할 토큰을 줄이는" 방향**이라는 게 일관적이다.
- **검색/추천 통합 논의(UniCoRn 등)와 결이 다르다.** 모델을 하나로 합치는 대신, **의도 분류 레이어를 LLM으로 두고 기존 시스템들을 그대로 툴로 쓴다.** 레거시를 갈아엎지 않아도 되는 접근이라 실무 이식성이 높음. 실제 시스템의 모습을 그리면:

```
LLM 라우터 (작게 증류된 모델, 캐시됨)
     │  ← 여기만 새로 만든 것
     ▼
route 선택 + 파라미터 채우기
     │
     ▼
기존에 이미 있던 Spotify 검색/추천 모델들  +  일부 LLM sub-agent
     ← 여기는 대부분 원래 있던 것
```

  Route/Tool 계층이 **어댑터** 역할을 해서, 기존 모델을 그대로 두고 그 위에 의도 분류 레이어만 얹은 구조.
- **`+115%` 같은 수치는 relative improvement over regular search이고 baseline이 낮은 영역**이라는 점은 감안해야 한다. 기존 검색이 애초에 못 하던 걸 시켰으니 큰 게 당연.
- 아쉬운 점: 5쪽 short paper라 **route 정의 개수, teacher/student 모델 스펙, judge 신뢰도 검증(human agreement), online A/B 수치**가 전부 빠져 있다. LLM-as-a-judge를 offline 주력 지표로 쓰면서 judge 자체의 validation을 안 보여준 건 약점.
- 재현 관점: **route 카탈로그를 어떻게 설계할 것인가**가 사실상 이 시스템의 전부다. 논문은 이 부분을 거의 안 다룬다.

> ⚠️ **주의**: 논문은 전체 파라미터 스키마를 공개하지 않는다. `"new indie rock releases"` 예시 **하나**만 나온다. 따라서 §2.5의 "쿼리 조각 → 인자" 대응표는 그 예시 하나에서 역추론한 것이고, 실제로 인자가 몇 종류이고 route가 몇 개인지는 알 수 없다.
