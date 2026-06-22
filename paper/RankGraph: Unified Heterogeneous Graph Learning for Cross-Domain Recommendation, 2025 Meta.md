# RankGraph: Unified Heterogeneous Graph Learning for Cross-Domain Recommendation

- **저자**: Renzhi Wu, Junjie Yang, Li Chen, Hong Li, Li Yu, Hong Yan (Meta MRS / Facebook Monetization)
- **연도**: 2025 (RecSys '25)
- **링크**: https://arxiv.org/abs/2509.02942

---

## 1. 문제 정의

추천 파운데이션 모델(FM)이 점점 백본으로 자리 잡고 있지만, **여러 제품 도메인에 걸친 세밀한 user-item 관계를 통합(cross-domain)** 하는 것이 여전히 난제.

- 전통적 모델: 각 유저의 독립적인 데이터 포인트에 집중 → 엔티티 간 관계 구조 손실
- 그래프 학습: 데이터를 상호 연결된 노드/엣지로 다룸 → 의존성, 상호작용, 맥락 정보를 자연스럽게 표현
- 하지만 cross-surface(광고/포스트/유저 등)는 **homogeneous 그래프로 표현 불가** → heterogeneous 그래프 필요

**RankGraph**: 추천 FM의 핵심 컴포넌트로 쓸 수 있는, GPU 가속 + 실시간 heterogeneous 그래프 학습 프레임워크.

```
[기존]                          [RankGraph]
유저별 독립 데이터              users · posts · ads · ... 를
포인트 단위 학습          →     하나의 이종(heterogeneous) 그래프로 통합
관계 구조 소실                  → 그래프 표현을 FM에 토큰으로 주입
```

---

## 2. 시스템 아키텍처

전체 파이프라인은 4단계로 구성 (Figure 1):

```
① Realtime Heterogeneous Graph   ② Advanced Graph Model (GPU)
   - cross-surface 소스로 그래프 구축    - Graph Feature Encoder
   - 그래프 큐레이션 & 프루닝            - RGCN-style 메시지 패싱
   - 노드/엣지 피처 enrich              - Contrastive Learning
            │                                  │
            └──────────────┬───────────────────┘
                           ▼
③ Efficient GPU Clustering        ④ Foundation Model 통합
   - Clustering / Indexing / KNN     - graph 임베딩을 토큰으로 주입
   - Similar Items 검색              - Temporal / Item Graph /
                                       Item Semantic Token 결합
```

### 2.1 Heterogeneous Graph 구축

- 다양한 **노드 타입**(Ads, User, Post 등)과 **엣지 타입**을 cross-product 상호작용에서 도출
- 엣지는 여러 상호작용 타입(클릭, 좋아요, 공유 등)의 **가중 조합**으로 engagement 신호를 인코딩
- **Semantic edge**: 인접 행렬에 직접 드러나는 engagement 엣지 외에, 멀티홉 이웃을 통한 간접 상호작용을 모델링 → 더 풍부한 맥락/행동 의미 포착

---

## 3. 모델 아키텍처

### 3.1 Graph Feature Encoder (타입 인식 피처 인코더)

노드 타입마다 피처 공간이 다름(예: raw id 피처, 다른 모델의 semantic embedding). 이를 **통합 임베딩 공간**으로 투영:

$$h_t = M_t\left( \text{concat}_{j=1}^{n_t} f_{t,j}(x_{t,j}) \right)$$

- $x_{t,j}$: 노드 타입 $t$의 $j$번째 피처 행렬
- $f_{t,j}$: MLP (피처별 투영)
- $\text{concat}$: 각 피처 출력 이어붙임
- $M_t$: **feature mixer** — 모든 피처 타입과 그 상호작용(두 피처 타입 간 곱)을 결합

### 3.2 Information Aggregation (RGCN 스타일)

Relational GCN에서 영감받은 메시지 패싱. relation `r`에 대한 노드 업데이트:

$$h_i^{l+1} = M_t\left( \text{concat}_r \left( f_r\left( c_{i,r} \sum_{j \in \mathcal{N}_i^r} W_r h_j^l \right) \right) \right)$$

- $\mathcal{N}_i^r$: relation $r$ 하에서 노드 $i$의 이웃 (예: user-ad의 click relation, ad-ad의 co-engagement relation 등 다수 relation 존재)
- $c_{i,r}$: 정규화 계수
- $W_r$: relation별 가중치 행렬
- $M_t$: 각 relation의 집계 임베딩을 결합하는 feature mixer

→ self-loop로 원본 피처를 보존하면서 이웃의 맥락 정보를 집계.

### 3.3 Contrastive Learning

엣지가 있는 노드 쌍(positive)과 없는 쌍(negative)을 구분하도록 학습 → 의미적으로 유사한 노드가 유사한 임베딩을 갖도록 유도.

**(1) Negative Sampling — 3가지 방식**

| 방식 | 설명 |
|------|------|
| **In-batch** | 같은 배치 내 다른 엣지의 노드를 negative로 샘플링 |
| **Out-of-batch** | 배치를 가로질러 샘플링 → negative 분포를 전역 분포에 가깝게. GPU에 노드 타입별 **candidate pool**을 유지하고 배치마다 증분 업데이트 |
| **Semantic** | 모델 컴포넌트(인코더/집계기)를 **multi-head**로 설계 → 같은 쌍이라도 다른 head가 만든 임베딩을 추가 negative로 사용 (robustness 강화) |

**(2) Contrastive Loss — Triplet + InfoNCE 결합**

```
Triplet loss  : 개별 positive/negative 쌍을 local 수준에서 분리
InfoNCE loss  : positive/negative 클러스터를 global 수준에서 분리
        │
        └─→ local + global 관계를 동시에 포착
```

---

## 4. 실시간 학습/서빙 & FM 통합

### 4.1 Real-Time Training and Serving

- 학습/추론 모두 **완전 GPU 가속** → 고처리량 시나리오에서 노드 임베딩을 on-the-fly 생성
- **GPU 최적화 클러스터링**으로 대규모 그래프에서 유사 아이템/유저를 빠르게 식별 → 최근 상호작용 기반 유사 post/ad/user 검색
- **retrieval generator**로 실제 런칭
- 기본 heterogeneous 그래프에서 **item-item, user-user 등 서브그래프 추출** 가능 → 해당 서브그래프만으로 충분한 use case에 유연하게 대응

#### 어떻게 "실시간"으로 임베딩을 생성하는가

> ⚠️ 4쪽짜리 논문이라 이 부분은 "GPU 가속으로 on-the-fly 생성"이라고만 명시됨. 아래 메커니즘은 논문 근거 + RGCN/GNN 서빙 일반론에 기반한 추론.

**논문이 명시한 근거**
- 학습/추론 모두 **완전 GPU 가속**, 학습된 모델이 임베딩을 **on-the-fly 생성**
- Figure 1에 **"Realtime Heterogeneous Graph"** — 그래프 자체가 실시간 갱신됨
- negative sampling: GPU에 노드 타입별 **candidate pool 유지 + 배치마다 증분 업데이트**

**핵심: RGCN은 inductive 함수**

임베딩을 미리 구워서 테이블에 저장(transductive)하는 게 아니라, **그래프 구조 + 피처를 입력받아 그 자리에서 계산하는 함수**다. 가중치($W_r, f_t, M_t$)는 고정이고 구조/피처만 바뀌면 즉시 새 임베딩이 나온다.

```
새 상호작용 발생 (유저가 광고 클릭)
        │
        ▼
그래프에 엣지 추가 / 노드 피처 갱신        ← "Realtime Heterogeneous Graph"
        │
        ▼
영향받은 노드의 이웃만 메시지 패싱 재계산   ← 전체가 아니라 국소(k-hop)만, GPU 병렬
        │
        ▼
해당 노드 임베딩 갱신 → KNN 인덱스 업데이트
```

**왜 GPU가 핵심인가**
- 메시지 패싱($\sum W_r h_j$)은 **희소 행렬 × 밀집 행렬 곱**의 반복 → GPU가 압도적
- 수십억 노드에서 매번 전체 재계산은 불가 → **바뀐 노드의 k-hop 이웃만** 부분 재계산
- candidate pool 등 자주 쓰는 데이터를 GPU 메모리에 상주 → I/O 최소화

→ 한 줄 요약: 임베딩을 미리 저장하는 게 아니라, **RGCN을 inductive 함수처럼 써서 그래프가 바뀔 때 영향받은 노드만 GPU에서 즉석 재계산** → 방금 일어난 상호작용도 곧바로 반영.

#### 임베딩을 retrieval에 쓰는 방식 (i2i 검색)

핵심은 결국 **그래프로 만든 노드 임베딩을 KNN/클러스터링으로 인덱싱**해서, 유저의 최근 행동과 가까운 아이템을 뽑는 item-to-item(i2i) 검색이다.

```
① 인덱스 구축
   RankGraph 아이템 임베딩 → GPU 클러스터링/인덱싱
   각 아이템마다 top-K 최근접 이웃(KNN) 테이블
   예) 광고 A → [A와 임베딩이 가까운 광고 20개]

② Trigger 설정 (유저별)
   지난 1주간 상호작용한 아이템 = trigger
   상호작용 타입별 가중치 부여 (예: 구매 > 클릭 > 노출)

③ 후보 생성
   각 trigger의 KNN 이웃을 후보로 끌어옴
   trigger A (가중 0.9) → 이웃 a1, a2, ...
   trigger B (가중 0.5) → 이웃 b1, b2, ...

④ 후보 정렬
   score = (trigger 가중치) × (trigger ↔ 후보 임베딩 유사도)
   → 점수순 정렬 후 최종 추천 후보 리스트
```

즉 **"유저가 최근 좋아한 것 ≈ 임베딩이 가까운 것들"** 을 추천하는 구조 (협업필터링의 "비슷한 상품" 로직을 그래프 임베딩으로 구현).

RankGraph의 차별점:
- **실시간성**: GPU 가속으로 임베딩을 on-the-fly 갱신 → 방금 본 아이템까지 trigger에 반영
- **cross-surface**: heterogeneous 그래프라 광고·포스트·유저가 한 임베딩 공간 → 포스트에서의 행동으로 광고를 retrieval하는 **도메인 교차 검색** 가능
- (인덱싱 알고리즘 디테일은 4쪽 논문이라 "GPU 최적화 클러스터링"으로만 언급)

### 4.2 Foundation Model 통합

- 그래프 임베딩을 시퀀스 기반 FM의 **입력 토큰**으로 주입
- 다른 토큰 타입(예: timestamp token)과 결합 → user-to-item 추천 모델의 표현력 강화
- 구조화된 그래프 지식을 FM 파이프라인에 직접 주입 → 여러 surface에서 개인화/랭킹 성능 향상

---

## 5. 실험 결과

베이스라인: **Filament2** (Meta에서 흔히 쓰는 그래프 학습 시스템)

### 5.1 Next-day Graph Edge Recall

전날 생성한 임베딩으로 다음날 생성될 그래프 엣지를 얼마나 잘 맞추는가 (1000개 엣지 샘플).

| Method | Recall@5 | Recall@10 | Recall@50 | Recall@100 |
|--------|----------|-----------|-----------|------------|
| Filament2 | 0.051 | 0.079 | 0.268 | 0.379 |
| **RankGraph** | **0.143** | **0.239** | **0.485** | **0.614** |

→ 전 구간에서 약 1.6~2.8배 향상.

### 5.2 Engagement Recall

오프라인 recall과 온라인 A/B 결과 간 괴리를 줄이기 위해 **미래 engagement 예측력**을 직접 측정하는 지표 제안.

```
① 시각 t에 최신 아이템 임베딩으로 각 아이템의 top-20 이웃 계산
② 각 유저: 지난 1주 상호작용 아이템을 trigger(상호작용 타입별 가중)로,
   그 이웃을 미래 상호작용 예측 후보로 사용 (trigger 가중 × 임베딩 유사도로 정렬)
③ t+1 ~ t+4시간의 실제 상호작용으로 recall 계산
```

수십억 유저 규모 surface, 하루 평균:

| Method | Recall@100 | Recall@200 | Recall@500 |
|--------|-----------|-----------|-----------|
| Filament2 | 0.071 | 0.125 | 0.221 |
| **RankGraph** | **0.106** | **0.157** | **0.234** |

### 5.3 온라인 A/B 테스트

| 지표 | 개선 |
|------|------|
| Click | **+0.92%** |
| Conversion | **+2.82%** |

---

## 6. 결론 및 의의

- **RankGraph**: 추천 FM의 핵심 컴포넌트로 동작하는 확장성/효율성 높은 그래프 프레임워크
- multi-type 노드 + semantic edge로 cross-surface user-item 관계의 다중 관계성을 포착
- GPU 가속 GNN = **타입 인식 피처 인코더 + relational 메시지 패싱 + contrastive learning**
- 실시간 임베딩 생성 + GPU 클러스터링으로 retrieval generator 역할, 그래프 토큰을 FM에 주입
- 오프라인 recall과 온라인 A/B(클릭 +0.92%, 전환 +2.82%) 모두에서 효과 입증
