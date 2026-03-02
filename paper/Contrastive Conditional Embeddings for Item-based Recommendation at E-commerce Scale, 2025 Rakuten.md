# Contrastive Conditional Embeddings for Item-based Recommendation at E-commerce Scale, 2025

- **저자:** Akira Fukumoto, Aghiles Salah, Sarthak Shrivastava, Alexandru Tatar, Yannick Schwartz, Vincent Michel, Lee Xiong (Rakuten Group / Meta)
- **발표:** RecSys 2025, Prague
- **링크:** https://dl.acm.org/doi/10.1145/3705328.3748095

---

## 1. 문제 정의

E-commerce에서 Item-based recommendation은 사용자가 본/구매한 아이템과 유사한 아이템을 추천하는 핵심 기능이다. 대부분의 방법은 item-item co-occurrence 데이터로부터 임베딩을 학습하는 방식을 사용하지만, **대규모 배포**에는 세 가지 근본적인 어려움이 있다:

1. **모델 크기**: 아이템 수에 비례해 선형적으로 증가 → 메모리/스토리지 문제
2. **학습 데이터 규모**: Co-occurrence 기반 데이터가 방대하여 연산/인프라 부담이 큼
3. **데이터 희소성**: 실제 관측된 co-occurrence는 전체 아이템 쌍의 0.001% 미만 → 모델 품질 저하

---

## 2. 핵심 아이디어: CCE (Contrastive Conditional Embeddings)

Co-occurrence 신호 + 텍스트 정보를 결합한 **Conditional Factor Model**을 제안.

> **아이템 유사도 정의:** 같은 유저가 자주 구매하거나 탐색한 아이템 쌍을 "관련 있는" 아이템으로 정의한다.

### 모델 구조

각 아이템 $i$의 최종 임베딩 $z_i$를 다음과 같이 분해:

$$z_i = \frac{x_i + o_i}{\|x_i + o_i\|}$$

- $x_i$: fine-tuned text embedding (고정, SBERT 기반)
- $o_i$: offset factor (학습 대상) — 텍스트로 설명되지 않는 co-occurrence 패턴을 포착

### 조건부 인수분해 모델

$$c_{ij} | x, o \approx \tau \cdot z_i^\top z_j$$

- $c_{ij}$: 아이템 $i$와 $j$의 co-occurrence 횟수
- $\tau$: 스케일링 상수
- 학습 시 **offset factor만 업데이트**, text embedding은 고정

### 손실 함수: Multi-class N-pair Contrastive Loss

$$-c_{ij} \cdot \log \frac{\exp(\tau \cdot z_i^\top z_j)}{\sum_{j' \in B \cup S} \exp(\tau \cdot z_i^\top z_{j'})}$$

- $B$: 배치 내 아이템 (in-batch negatives)
- $S$: 균등 샘플링한 uniform negatives
- **Mixed Negative Sampling (MNS)**: in-batch + uniform negatives 혼합 → 성능 향상

### 텍스트 임베딩 (SBERT Fine-tuning)

- Sentence-BERT 아키텍처 사용
- 아이템 타이틀 쌍을 입력으로, co-occurrence를 타겟으로 동일한 N-pair loss로 fine-tune
- 추론 시 `[CLS]` 토큰의 마지막 hidden state를 $x$로 사용
- 모든 벡터는 **L2-normalize** → norm 편향 제거, 정규화 필요성 감소

**[Q] "co-occurrence를 타겟으로 SBERT를 학습한다"는 게 무슨 의미?**

일반적인 SBERT는 문장 의미 유사도(STS, NLI)로 fine-tune하지만, 여기서는 **유저 행동 기반 co-occurrence**를 supervised signal로 사용한다.

학습 데이터 구조:
- Input: 아이템 타이틀 쌍 `(title_i, title_j)`
- Target: 두 아이템의 co-occurrence 빈도 `c_ij`
- Positive pair: `c_ij > 0`인 쌍 (같이 구매/탐색된 아이템)
- Negative: in-batch + uniform negatives (MNS)

모델 구조 (Siamese BERT):
```
title_i ──► [ BERT ] ──► [CLS] embedding ──► x_i (L2 norm)
                                                      ↘
                                               cosine sim → N-pair loss (c_ij 가중)
                                                      ↗
title_j ──► [ BERT ] ──► [CLS] embedding ──► x_j (L2 norm)
            (shared weights)
```

결과적으로 **"의미적 유사도"가 아닌 "유저 행동 패턴이 반영된 텍스트 임베딩"** `x_i`를 얻는다. 이 임베딩은 CCE에서 고정된 채로 offset `o_i`의 베이스로 사용된다.

> Ablation에서 `Text only < Offset only`인 이유: 텍스트 임베딩만으로는 co-occurrence 패턴을 완전히 포착하지 못하며, offset이 그 갭을 메운다.

---

## 3. 구현 & 배포

**규모:** 4천만 아이템, 20억 co-occurrence 쌍, 100억 파라미터 (절반인 offset factor가 매일 업데이트)

### Production Pipeline (Figure 1)

```
┌─────────────────────────────────────────────────────────────────────┐
│ ONLINE (Real-time)                                                   │
│                                                                      │
│  ┌─────┐   Seed Item    request: find related items                  │
│  │ APP │ ─────────────────────────────────────────► ┌─────────────┐ │
│  │     │ ◄─────────────────────────────────────────  │  Candidate  │ │
│  └─────┘      Recommendations                       │  Retrieval  │ │
│                    ▲                                 └──────┬──────┘ │
│             ┌──────┴──────┐   Other candidate               │get top │
│             │  Re-ranking │◄── sources                      │items   │
│             └─────────────┘                          ┌──────▼──────┐ │
│                                                      │  ANN Index  │ │
│                                                      └──────▲──────┘ │
│                                               quantize & push │      │
├──────────────────────────────────────────────────────────────┼───────┤
│ OFFLINE (Daily batch)                                         │      │
│                                                               │      │
│  ┌──────────────┐                                             │      │
│  │  User events │──(lid,rid,coc)[train]──────────────►┐      │      │
│  │     dump     │──(lid,rid,coc)[test] ──────────────►│      │      │
│  └──────┬───────┘                                  ┌──┴──────┴───┐  │
│         │ (iid, text)        Text embeddings        │  Train CCE  │  │
│         └──►┌─────────┐─────────────────────────►  │             │  │
│             │  SBERT  │                             └──────┬──────┘  │
│             │Finetuned│                                    │         │
│             └─────────┘                             ┌──────▼──────┐  │
│                                                     │  Eval CCE   │  │
│                                                     └──────┬──────┘  │
│                                               NDCG@10 > th?          │
│                                          ┌──────────┴──────────┐     │
│                                        True                  False   │
│                                          │                Manual check│
│                                    run inference                      │
│                                          ▼                            │
│                                   ┌─────────────┐  quantize & push   │
│                                   │CCE Embeddings│──────────────────►│
│                                   └─────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
```

**Offline 흐름 요약:**
1. User events dump → Spark → `(lid, rid, coc)` 쌍 생성
2. `(iid, text)` → SBERT fine-tune → Text embeddings (고정)
3. Train CCE (offset factors 학습) → Eval CCE (NDCG@10 검증)
4. 통과 시: 추론 → CCE embeddings → int8 양자화 → ANN Index push
5. 실패 시: Manual check

**Online 흐름 요약:**
- APP에서 Seed item → Candidate Retrieval이 ANN Index 조회 → Re-ranking → 추천 결과 반환

### 학습 데이터 파이프라인

- Spark로 실시간 유저 이벤트 로그 처리
- 클릭: 1시간 윈도우 / 구매: 24시간 윈도우로 co-occurrence 쌍 생성
- 노이즈 제거: co-occurrence ≥ 3인 쌍만 사용, 최대값 20으로 캡
- sparse triplet `(left_item_id, right_item_id, co-occurrence)` 형태로 저장
- 학습 시 데이터를 **전체 메모리(pandas DataFrame)에 로드** → 디스크 I/O 병목 제거

### GPU 효율화: Torch Sparse Embeddings

| Embedding Type   | Training Time (1 epoch) | Embedding Dim |
|-----------------|------------------------|---------------|
| Dense Embedding  | 6시간                   | 50            |
| Sparse Embedding | 2시간                   | 100           |

- **Sparse Adam Optimizer** 사용: 배치에서 사용된 임베딩만 업데이트
- Dense 대비 **3배 속도 향상, 2배 메모리 절약** (단일 V100 GPU 기준)

### 추론 및 서빙

1. 학습 후 holdout set에서 **NDCG@10**이 임계값을 넘으면 배포
2. 임베딩을 **int8로 양자화** → 메모리/레이턴시 절감
3. **ANN(Approximate Nearest Neighbor) 인덱스**에 업로드
4. 서비스 시 seed item ID로 인덱스를 조회하여 실시간 추천 (O(1) 추론)
5. 전체 파이프라인 **CI/CD로 자동화**

---

## 4. 실험 결과

### 데이터셋

| 데이터셋         | 아이템 수     | 인터랙션 수   | 윈도우  |
|----------------|------------|------------|--------|
| Ichiba Purchase | 2,500만+   | 10억+      | 24시간 |
| Ichiba Browsing | 4,000만    | 20억       | 1시간  |

- 학습/테스트 비율: 90:10 / 10 epochs / lr=1e-2 / batch size=20,000 / embedding dim=128

### Ablation 1: Loss 함수 비교 (N-pair vs MSE)

| 데이터셋 | Loss  | Recall@10 | NDCG@10 |
|--------|-------|-----------|---------|
| Ichiba Browsing | MSE   | 0.2718    | 0.1292  |
| Ichiba Browsing | N-Pair | **0.3718** | **0.1669** |
| Ichiba Purchase | MSE   | 0.4322    | 0.1837  |
| Ichiba Purchase | N-Pair | **0.5481** | **0.2577** |

→ N-pair loss가 MSE 대비 압도적으로 우수

### Ablation 2: 임베딩 구성 요소 비교

| 구성            | Recall@10 (Browse) | NDCG@10 (Browse) | Recall@10 (Purchase) | NDCG@10 (Purchase) |
|---------------|----------|---------|----------|---------|
| Text only      | 0.2089   | 0.0949  | 0.2525   | 0.1206  |
| Offset only    | 0.3421   | 0.1470  | 0.5144   | 0.2395  |
| Text + Offset  | **0.3718** | **0.1669** | **0.5481** | **0.2577** |

→ Text와 Offset 결합이 최고 성능. Offset만도 Text only를 크게 상회 → 텍스트는 보완적 역할

### Ablation 3: Offset 초기화 방식 & Mixed Negative Sampling

- Offset을 텍스트 임베딩으로 초기화하는 것보다 **랜덤 초기화**가 더 유연하여 성능 우수
- In-batch negatives + uniform negatives 혼합 (2x 배치 크기)이 최고 성능

### A/B 테스트 결과 (2주간, Rakuten Ichiba)

| 위젯            | 구매율 증가    | p-value |
|---------------|------------|---------|
| Top-homepage  | **+16.38%** | < 0.01  |
| Checkout page | **+4.01%**  | < 0.01  |

- 2단계 시스템: CCE + 다른 기존 방법들이 후보 생성 → Re-ranking
- 두 위젯 모두 통계적으로 유의미한 구매율 향상

---

## 5. 핵심 기여 및 의의

1. **CCE 모델**: 텍스트(고정)와 co-occurrence 기반 offset(학습)을 분리하여 희소성 문제 해결
2. **N-pair Contrastive Loss + MNS**: MSE 대비 우월한 ranking 품질
3. **Sparse Embedding + Sparse Adam**: 대규모 배포를 위한 3x 속도, 2x 메모리 효율화
4. **최초의 상세 배포 사례**: 100억 파라미터, 20억 co-occurrence 쌍 규모의 성공적 실서비스 적용 문서화
