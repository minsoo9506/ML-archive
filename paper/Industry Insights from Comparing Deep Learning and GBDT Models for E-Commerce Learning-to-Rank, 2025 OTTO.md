# Industry Insights from Comparing Deep Learning and GBDT Models for E-Commerce Learning-to-Rank (2025, OTTO)

> Yunus Lutz, Timo Wilm, Philipp Duwe — OTTO (GmbH & Co. KGaA)  
> RecSys 2025 / arXiv:2507.20353

---

## 핵심 요약

OTTO의 production LambdaMART(LightGBM) 모델과 세 가지 DNN 아키텍처를 대규모 proprietary 데이터셋 + 8주 온라인 A/B 테스트로 비교.  
결론: **간단한 Two-Tower DNN이 GBDT baseline을 이김** — 클릭 +1.86%, 매출 +0.56%, 판매 수량은 동등.

---

## 일반 Ranker vs LTR 차이

| | 일반 Ranker (pointwise) | LTR (listwise) |
|---|---|---|
| **데이터 단위** | `(item, label)` 1쌍 = 1 sample | `(context, [items], [labels])` = 1 sample |
| **학습 신호** | "이 아이템이 클릭될 확률은?" | "이 query에서 어떤 아이템이 더 위에 있어야 하나?" |
| **Loss** | BCE 등 — 아이템별 독립 | Softmax CE / RankNet — list 내부에서 정규화 |
| **Negative 의미** | 절대적 (안 클릭됨) | 상대적 (positive보다 덜 좋음) |
| **Gradient** | 자기 label만 보고 결정 | 같은 list 내 다른 아이템과의 상대 비교 |

### Two-Tower의 경우 forward는 아이템별 독립

```
context c   → context tower → h_c   (1번)
item p_i    → product tower → h_p_i (각자 독립적으로)
s_i = h_c · h_p_i
```

→ **LTR스러움은 loss 단계에서 발생** — Softmax가 list 전체에 걸쳐 정규화하므로
gradient가 "이 아이템을 다른 아이템보다 위로 올려라" 형태로 흐름.

→ Cross-Encoder/Transformer는 forward 단계부터 list를 통째로 입력받아 아이템 간
interaction까지 학습 (대신 item embedding 사전 계산 불가).

---

## 문제 정의: LTR at OTTO

- 검색 후 candidate retrieval 결과 `n`개를 re-ranking하는 **contextualized LTR**
- 입력: candidate 상품 목록 `p = [p₁, ..., pₙ]` + 컨텍스트 `c` (유저 행동, 검색 의도, 디바이스 등)
- 레이블: click `yᶜ ∈ {0,1}ⁿ`, order `yᵒ ∈ {0,1}ⁿ` (implicit feedback, multi-positive)
- 평가 cutoff = 15 (OTTO 검색 결과 페이지 중앙값 스크롤 깊이)

---

## Feature Embedding

| 피처 종류 | 처리 방식 |
|---|---|
| 수치형 (`fⁿᵘᵐ`) | 우편향 → power-law 정규화, 나머지 → z-score 정규화 |
| 범주형 (`fᶜᵃᵗ`) | Embedding layer → dense vector |
| 텍스트형 (`fᵗᵉˣᵗ`) | Bag-of-words → 단어 embedding 합산 |

최종 상품 embedding: `xᵖ = concat([numerical, categorical embeddings, text embeddings])`

---

## 모델 아키텍처 3종

공통 backbone: `B(k)(z) = LayerNorm(ReLU(Dropout(W(k)z + b(k))))` + skip connection  
→ k=3 layers, hidden size h=1024 사용

### 1. Two-Tower (TT)
- 상품 features → backbone → `hᵖ`
- 컨텍스트 features → linear layer → `hᶜ`
- score = `hᶜᵀ · hᵖ` (dot product)
- **장점**: 상품 embedding 사전 계산 가능 → inference 속도 빠름

### 2. Cross-Encoder (CR)
- `[xᵖ, xᶜ]` concat → backbone → scoring layer
- 더 복잡한 feature interaction 캡처 가능

### 3. Transformer (TR)
- Cross-Encoder + Multi-Head Self-Attention (위치 인코딩 없음)
- Listwise contextual embedding 생성
- Latent Cross로 MHSA output과 backbone output 결합

---

## Loss Function 2종

### Softmax Cross-Entropy (CE)
- 레이블을 확률 분포로 정규화: `ỹᵢ = yᵢ / Pₙ`

### RankNet (RN)
- Pairwise loss를 positive 수로 나눠 multi-positive 문제 완화: `L_RN = (1/Pₙ) * L̃_RN`

두 loss 모두 click/order를 `α=0.5`로 가중합하여 결합:  
`L = α * Lᶜ + (1-α) * Lᵒ`

---

## 실험 결과

### 오프라인 (vs LGBM LambdaMART baseline)

| 모델 | Loss | NDCG_c | NDCG_o | AIV |
|---|---|---|---|---|
| Two-Tower | CE | **+4.32%** | 0.00% | +2.28% |
| Two-Tower | RN | +3.70% | -1.48% | +2.28% |
| Cross-Encoder | CE | **+4.32%** | -0.30% | +2.28% |
| Cross-Encoder | RN | +4.63% | -0.59% | +2.37% |
| Transformer | CE | +2.16% | -1.48% | +2.89% |
| Transformer | RN | +3.09% | -1.78% | +3.16% |

- 모든 DNN이 `NDCG_c`(클릭), `AIV`(매출 proxy)에서 LGBM 상회
- `NDCG_o`(주문)에서는 대부분 소폭 하락 — α=0.5 설정 및 click-skewed 데이터 특성 때문
- **Two-Tower + CE**가 NDCG_o를 0%로 유지하면서 NDCG_c·AIV 모두 개선 → 온라인 검증 후보 선정

### 온라인 A/B 테스트 (8주, Two-Tower CE vs LGBM)

| 지표 | 결과 |
|---|---|
| 총 클릭 수 | **+1.86%** (p < 0.0001) |
| 매출 | **+0.56%** (p < 0.01) |
| 판매 수량 | 동등 (유의미한 차이 없음) |

- 학습·서빙 비용 증가는 성능 이득 대비 무시할 수준

---

## 핵심 인사이트

1. **단순한 DNN도 충분히 강력**: Two-Tower처럼 구조가 단순한 모델이 복잡한 Transformer보다 실용적으로 더 나은 결과를 냄
2. **Transformer의 함정**: listwise attention이 오히려 order NDCG를 악화시킴 — 복잡성이 항상 이득이 아님
3. **오프라인 → 온라인 전이**: 오프라인 결과가 온라인 결과와 일관되게 나타남 (offline evaluation 신뢰성 확인)
4. **Multi-positive 처리**: click/order 모두 여러 아이템이 positive일 수 있어 loss 정규화 필요
5. **산업 현장에서 DNN 도입**: 적절히 튜닝된 DNN은 LambdaMART 대체재로 실용적

---

## 실험 세팅 요약

| 항목 | 값 |
|---|---|
| 학습 데이터 | 43M samples (OTTO 검색 로그) |
| 테스트 데이터 | 700k samples |
| Backbone layers | k=3, h=1024 |
| Embedding dim | d_cat=128, d_text=512 |
| Batch size | 1000 |
| LGBM baseline | LambdaRank, lr=0.1, max_depth=12, num_leaves=25, 400 trees |

---

## 관련 연구 포인트

- **LTR 벤치마크 데이터셋 한계**: Web30K, Yahoo! LTR은 human-label 기반 → 실제 e-commerce implicit feedback과 괴리
- **Baidu Unbiased LTR 데이터셋**: 현실적이나 중요 상품 feature 부재
- 선행 연구들은 온라인 A/B 테스트 없이 오프라인만 평가 → 본 논문의 차별점
