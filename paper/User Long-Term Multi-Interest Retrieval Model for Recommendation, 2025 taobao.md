# User Long-Term Multi-Interest Retrieval Model for Recommendation (ULIM)

- **저자**: Yue Meng, Cheng Guo, Xiaohui Hu et al. (Taobao & Tmall Group of Alibaba)
- **연도**: 2025 (RecSys '25)
- **링크**: https://arxiv.org/abs/2507.10097

---

## 0. 한 줄 요약

랭킹 단계에서는 이미 수천 개의 장기 행동 시퀀스를 쓰는데, **검색(retrieval) 단계는 여전히 수십~수백 개**에 머물러 있다.
ULIM은 **카테고리 단위로 시퀀스를 쪼개고(멀티 인터레스트)**, **"먼저 카테고리를 예측 → 그 카테고리 안에서만 아이템 검색"** 하는 2단계(cascaded) 구조로 **수천 개 장기 행동을 검색 단계에 도입**한 모델이다. Taobaomiaosha(타오바오 미아오샤, 타임세일 미니앱)에서 클릭 +5.54%, 주문 +11.01%, GMV +4.03% 달성.

---

## 1. 문제 정의

### 1.1 배경: 랭킹은 되는데 검색은 왜 안 되나

| 단계 | 시퀀스 길이 | 대표 모델 |
|------|------------|----------|
| **Ranking** | 수천 개 (long) | DIN, SIM, ETA |
| **Retrieval** | 수십 개 (short) | YouTube DNN, MIND |

검색 단계가 장기 시퀀스를 못 쓰면 → 랭킹과 **일관성(consistency)** 이 깨진다. 검색이 후보를 못 넣어주면 랭킹이 아무리 좋아도 소용없음.

### 1.2 검색 단계에서 장기 시퀀스가 어려운 2가지 이유

```
① Latency (지연 제약)
   랭킹: 후보 수천 개만 스코어링 → 여유 있음
   검색: 수천만 개 후보를 스캔 → 시퀀스까지 길어지면 연산량 폭발

② 아키텍처 한계 (target-aware 부재)
   검색은 보통 "next-item prediction" (user emb · item emb 내적)
   → target-aware cross-interaction 이 없음
   → SIM 처럼 "카테고리로 먼저 좁히는" 계층적 단순화를 못 씀
```

### 1.3 추가 이슈: Single-Interest의 gradient conflict

하나의 user 임베딩으로 다양한 관심사(옷/전자기기/식품…)를 모두 표현하려 하면,
서로 다른 방향의 아이템들이 임베딩을 잡아당겨 **gradient가 충돌** → 표현 품질 저하.
→ **멀티 인터레스트(multi-interest)** 모델링이 필요 (MIND가 캡슐 네트워크로 시도했던 문제).

---

## 2. 핵심 아이디어

ULIM은 두 개의 축으로 구성된다. **학습(training)** 과 **서빙(serving)** 을 각각 담당.

```
┌──────────────────────────────────────────────────────────────┐
│  ① Category-Aware Hierarchical Dual-Interest Learning (학습)  │
│     - 긴 시퀀스를 "카테고리별 서브시퀀스"로 분할              │
│     - 장기(long) + 단기(short) 관심을 함께 최적화              │
├──────────────────────────────────────────────────────────────┤
│  ② Pointer-Enhanced Cascaded Category-to-Item Retrieval(서빙) │
│     - PGIN이 "다음 관심 카테고리 Top-K" 예측                   │
│     - 그 K개 카테고리 안에서만 병렬 아이템 검색                │
└──────────────────────────────────────────────────────────────┘
```

핵심 통찰: **"수천만 아이템 전체에서 찾지 말고, 관심 카테고리로 먼저 좁힌 뒤 그 안에서 찾자."**
→ 복잡도를 O(L) → O(L/N) 로 줄이고(N=카테고리 수), 병렬화로 지연도 잡음.

---

## 3. 방법론 ① : Category-Aware Hierarchical Dual-Interest Learning (학습)

### 3.1 Granularity-Aligned Category Clustering (카테고리 분할)

원본 긴 시퀀스를 카테고리별 서브시퀀스로 나눈다.

```
원본 장기 행동 시퀀스 (수천 개, 시간순 뒤섞임)
[👕 👟 📱 🍎 👗 💻 🍊 👖 ...]
        │  카테고리로 그룹핑
        ▼
┌─────────┬─────────┬─────────┐
│ 의류    │ 전자기기 │ 식품    │  ← 각각이 하나의 "interest cluster"
│ 👕👗👖  │ 📱💻     │ 🍎🍊    │
└─────────┴─────────┴─────────┘
```

- 복잡도 O(L) → **O(L/N)** 로 감소
- **중요**: 카테고리 granularity를 **랭킹 단계와 동일하게 정렬** → 검색↔랭킹 피처 공간 일관성 유지

### 3.2 Training Objective Redefinition (학습 목표 재정의)

이게 이 논문의 핵심 트릭 중 하나다.

- 기존: 전체 후보 풀에서 클릭 확률 예측
- **ULIM: "각 interest cluster 안에서" 클릭 확률 예측**
  - Positive 샘플은 반드시 해당 서브시퀀스의 카테고리와 일치
  - Negative 샘플도 **같은 카테고리 subspace에서만** 추출

```
카테고리별로 batch를 미리 그룹핑 (category-homogeneous batch)
+ batch 안에서 negative 공유
        │
        ▼
"카테고리 누출(category leakage)" 차단
→ 모델이 "카테고리만 보고 대충 맞추는" trivial classification / 학습 붕괴 방지
→ offline 지표 부풀림(inflated metric) 방지
```

> 만약 negative를 전체 풀에서 뽑으면, 모델은 "이 아이템이 의류인지 식품인지"만 구분해도 positive를 맞출 수 있어서(너무 쉬움) 실제 개인화를 학습하지 못한다. 그래서 **같은 카테고리 안에서만** 경쟁시킨다.

### 3.3 Model Structure (Fig. 1)

두 개의 인코더가 병렬로 동작 + 아이템 타워.

```
                    Loss_long           Loss_short
                       ▲                   ▲
                       │                   │
              ┌────────┴──────┐    ┌───────┴──────┐
              │  DNN layer    │    │  DNN layer   │
              └────────▲──────┘    └───────▲──────┘
     user long-term multi-interest    user short-term
        embeddings (K개)               interest emb
                       ▲                   ▲
              ┌────────┴──────┐    ┌───────┴──────┐
              │Target-Attention│   │pooling +     │
              │(query=단기pooling)│ │self-attention│
              └────────▲──────┘    └───────▲──────┘
       category-aware 서브시퀀스들    최근 행동(≤100)
                       ▲                   ▲
              ┌────────┴───────────────────┴──────┐
              │           embedding layer          │
              └────────────────────────────────────┘
              long 행동   user profile   short 행동   target item
```

- **Short-Term Encoder**: 최근 행동(최대 100개) → MHSA(멀티헤드 셀프어텐션) → average pooling
- **Long-Term Encoder**: 카테고리 서브시퀀스 추출 → **단기 pooling 결과를 query로 하는 Target-Attention** → 장기 관심 임베딩 생성
  - 학습 시: positive 샘플 하나가 **단 하나의 카테고리 서브시퀀스**를 활성화
  - 서빙 시: 여러 서브시퀀스를 **병렬 처리**하여 K개의 장기 멀티 인터레스트 임베딩 생성

### 3.4 Loss Function

장기/단기 임베딩을 모두 discriminative하게 만들기 위한 복합 손실:

```
L = α · L_long + β · L_short
```

L_long, L_short 각각은 **sampled softmax** 형태:

```
              exp( v_u^(l) · e_i )
L_long = -Σ log ─────────────────────────
             Σ_{j∈I_neg} exp( v_u^(l) · e_j )
```

(short도 동일 구조, v_u^(s) 사용)
- v_u^(l), v_u^(s): 유저의 장기/단기 관심 임베딩
- e_i: 클릭한 아이템 임베딩, I_neg: negative 집합

---

## 4. 방법론 ② : Pointer-Enhanced Cascaded Category-to-Item Retrieval (서빙)

offline 학습과 online 서빙을 정렬하기 위한 **2단계 계층 검색**.

```
1단계: PGIN → 관심 카테고리 분포 예측 → Top-K 카테고리 선택
2단계: K+1개 임베딩으로 병렬 ANN 검색
        (K개 장기 카테고리별 + 1개 단기 전체)
```

### 4.1 PGIN (Pointer-Generator Interest Network) — Fig. 2

"다음 카테고리 예측"을 **multiclass classification**으로 풀되, 두 네트워크를 게이팅으로 결합.

```
    generator dist          final dist            pointer dist
        ▲          ×(1-P_poi)    ▲    ×P_poi          ▲
        │              └─────────┼─────────┘          │
   ┌────┴─────┐              (gating)            ┌────┴─────┐
   │Generator │                                  │ Pointer  │
   │   Net    │                                  │   Net    │
   │          │                                  │          │
   │MHSA+풀링 │                                  │Target-   │
   │+User-attn│                                  │Attention │
   └────▲─────┘                                  └────▲─────┘
  user profile / long cate seq / short cate seq   pointer seq
                                          (long+short 카테고리 병합·중복제거·시간순)
```

- **Pointer-Net**: 장기+단기 카테고리 이력을 **중복 제거 + 시간순 정렬**한 시퀀스 입력 → Target-Attention으로 multiscale 카테고리 피처 추출 → backward projection으로 전체 카테고리 공간에 매핑
  - (유저가 **실제로 상호작용했던** 카테고리를 "가리키는(point)" 역할 → copy mechanism)
- **Generator-Net**: raw 장기/단기 행동 시퀀스를 MHSA+pooling으로 인코딩 → 유저 피처와 Target-Attention으로 융합 → stacked MLP로 카테고리 확률 출력
  - (이력에 없던 **새로운 카테고리도 생성(generate)** 가능)
- **Gating으로 결합**:

```
ŷ = P_poi · y_poi + (1 - P_poi) · y_gen
```

  - true 카테고리 라벨에 대한 cross-entropy로 학습
  - Pointer(과거 반복 관심) ↔ Generator(탐색/신규 관심)의 균형을 학습된 게이트 P_poi가 조절

> Pointer-Generator는 원래 텍스트 요약에서 "원문 단어 복사 vs 새 단어 생성"을 섞던 구조. 여기선 "익숙한 카테고리 복사 vs 새 카테고리 생성"으로 차용한 것.

#### 왜 게이팅인가? (그냥 하나로 예측하면 안 되나)

두 분포는 **support와 inductive bias가 다르다**. 하나로는 둘 다 못 잡는다.

| | Generator-Net 단독 | Pointer-Net 단독 |
|---|---|---|
| 출력 공간 | 전체 카테고리 | **유저 이력에 있는 카테고리만** |
| 강점 | 신규/탐색 관심(cold interest) | 반복 관심(repeat), sharp한 분포 |
| 약점 | 인기 카테고리 편향, 분포가 뭉툭함 | 이력에 없는 카테고리 **절대 못 뽑음** |

- **Generator만**: 수천 개 카테고리 중 유저 이력의 5~10개를 콕 집으려면, "이게 내 이력에 있는가"를 임베딩만 보고 *암묵적으로* 학습해야 함 → MLP한테 어려운 문제. 추천에서 반복 소비 비중이 큰데, 이건 분류보다 **선택(selection) 문제**에 가깝다.
- **Pointer만**: 출력 support가 이력으로 제한되는 게 구조적 장점이자 한계. 탐색/신규 관심이 전부 죽어서 retrieval 단계엔 치명적.
- **게이팅**: `P_poi`가 유저/문맥마다 **학습되는 값**이라는 게 핵심. 이력 길고 패턴 뚜렷한 헤비 유저 → P_poi 높게, 짧거나 관심이 튀는 유저 → generator 쪽으로. 룰/하이퍼파라미터로 고정하면 이 적응이 안 된다. 미분 가능하므로 **단일 CE loss 하나로 end-to-end 학습**.

> **피처를 concat해서 한 네트워크로 예측하면 안 되나?** → 안 된다. Pointer 분포는 피처가 아니라 **출력 공간의 제약**이다. Concat하는 순간 최종 softmax가 다시 전체 카테고리에 대한 자유로운 분포가 되어 "이력 안에서만 고른다"는 구조적 보장이 사라진다. 게이팅은 **확률 분포 레벨의 mixture**라서 각 브랜치의 제약이 그대로 살아있다. (See et al. 2017의 원조 pointer-generator에서 OOV 단어는 복사로만, 새 표현은 생성으로만 가능했던 것과 같은 논리)

### 4.2 Category-Constrained Retrieval (서빙 시점)

PGIN이 뽑은 Top-K 카테고리로 **K+1개 임베딩에 대해 병렬 ANN 검색**:

```
i) 각 예측 카테고리마다 → 장기 시퀀스에서 해당 카테고리 서브시퀀스 추출
   → K개 장기 관심 임베딩 생성
ii) 단기 시퀀스에서 → 단기 관심 임베딩 1개 생성
```

**핵심 제약 (offline-online 일관성)**:

| 임베딩 | 검색 범위 |
|--------|-----------|
| 장기 멀티 인터레스트 (K개) | **해당 카테고리 후보 안에서만** 검색 |
| 단기 관심 (1개) | 전체 후보 풀 검색 |

→ 이 제약이 3.2의 재정의된 학습 목표(카테고리 내 예측)와 정확히 맞물려, **학습↔서빙 distribution shift를 방지**한다.

> 단기 임베딩만 전체 풀을 보는 이유: 장기 쪽은 PGIN이 이미 K개 카테고리로 좁혀놨으니 탐색 커버리지가 부족하다. 단기 1개가 그 밖의 영역까지 담당하는 안전장치.

### 4.3 추론 시점 Input 정리

**서빙 시엔 target item이 전혀 들어가지 않는다.** Retrieval 단계라 수천만 아이템에 모델을 돌릴 수 없고, **유저 타워만으로** 임베딩을 만들어 ANN에 던져야 한다.

**1단계 — PGIN (카테고리 예측)**

| 입력 | 내용 |
|---|---|
| user profile | 유저 정적 피처 |
| long category seq | 장기 행동의 **카테고리 ID 시퀀스** |
| short category seq | 최근 행동의 카테고리 ID 시퀀스 |
| pointer seq | 위 둘을 병합 → 중복 제거 → 시간순 정렬 |

→ 출력: 전체 카테고리 공간 확률 분포 → Top-K 카테고리

**2단계 — 유저 타워 (임베딩 생성)**

| 입력 | 출력 |
|---|---|
| Top-K 카테고리로 필터링한 **장기** 서브시퀀스 K개 | Target-Attention → 장기 임베딩 K개 |
| 최근 행동 시퀀스(≤100) 전체 (카테고리 필터링 **없음**) | MHSA+pooling → 단기 임베딩 1개 |

```
PGIN → Top-K 카테고리 {c1...cK}
   ├─ 장기 시퀀스에서 c1 아이템만 추출 → Target-Attn → emb_1 → ANN(c1 인덱스)
   ├─ 장기 시퀀스에서 c2 아이템만 추출 → Target-Attn → emb_2 → ANN(c2 인덱스)
   ├─ ...                                                     (K개 병렬)
   └─ 최근 행동 100개 전체     → MHSA+pooling → emb_short → ANN(전체 풀)
                                                     └→ 결과 merge
```

**헷갈리기 쉬운 지점**

1. K개 임베딩을 만드는 건 **장기 시퀀스**를 카테고리로 필터링한 것이다. "최근 행동을 카테고리별로 나눈 것"이 아니다. 시간 제약이 아니라 **의미(카테고리) 제약**.
2. Long-Term Encoder의 Target-Attention **query는 target item이 아니라 단기 pooling 결과**다 (§3.3). 그래서 유저 타워가 target item 없이 자기완결적으로 돌아가고 서빙이 가능해진다. Fig. 1의 `target item`은 loss 계산용 **아이템 타워** 입력.
3. 같은 장기 이력이라도 단기 맥락에 따라 다른 임베딩이 나온다 ("지금 이 유저의 단기 맥락에서 이 카테고리 이력 중 뭐가 중요한가").
4. 학습↔서빙 gap: 학습 시엔 positive 하나가 서브시퀀스 **하나만** 활성화, 서빙 시엔 K개를 **병렬** forward. 이 gap을 §3.2(카테고리 내 예측)와 §4.2(카테고리 제한 검색)가 메운다.

---

## 5. 전체 흐름 정리

```
[학습]  긴 시퀀스 → 카테고리별 분할 → 장기(Target-Attn) + 단기(MHSA) 인코딩
         → 카테고리 내부에서만 sampled softmax (누출 차단)

[서빙]  ┌─ PGIN으로 Top-K 관심 카테고리 예측 ───────────────┐
         │                                                     │
         ├─ 장기: K개 카테고리 서브시퀀스 → K개 임베딩         │
         │        → 각 카테고리 후보 안에서만 ANN              │  병렬
         └─ 단기: 1개 임베딩 → 전체 풀에서 ANN ────────────────┘
                                │
                                ▼
                  검색 결과를 랭킹 단계로 전달
                  (독립 retrieval 채널로 추가)
```

---

## 6. 실험 결과

### 6.1 Offline (Taobao, 일일 수천만 positive 샘플, 2년치 행동 사용)

| Method | HR@500 | HR@1000 | HR@2000 |
|--------|--------|---------|---------|
| YouTube DNN Variant | 4.95% | 9.53% | 14.93% |
| MIND Variant | 5.03% | 9.66% | 15.15% |
| **ULIM** | **6.02%** | **10.76%** | **16.55%** |

→ 장기 행동 시퀀스를 활용하는 것이 확실히 효과적. (MIND variant는 이미 온라인 배포된 채널)

### 6.2 Online A/B (Taobaomiaosha, 3주)

| 지표 | 개선 |
|------|------|
| 클릭 (clicks) | **+5.54%** |
| 주문 (orders) | **+11.01%** |
| GMV | **+4.03%** |
| 시스템 RT | +약 15ms (수용 가능) |

독립 검색 채널로 통합 → 랭킹에 incremental 후보 기여.

### 6.3 Ablation Study

| Method | HR@500 | HR@1000 | HR@2000 |
|--------|--------|---------|---------|
| ULIM-half-sequence (장기 시퀀스 절반으로 축소) | 4.75% | 8.03% | 13.36% |
| ULIM-self-attention (Target-Attn → Self-Attn 교체) | 5.39% | 9.70% | 15.22% |
| **ULIM (full)** | **6.02%** | **10.76%** | **16.55%** |

- **장기 시퀀스 길이가 중요** (절반으로 줄이면 가장 큰 성능 하락)
- **Target-Attention(cross-sequence interaction)이 Self-Attention보다 우수** → target-aware의 중요성 입증

### 6.4 파라미터 민감도: K (선택 카테고리 수 = interest cluster 수)

- K가 클수록 HR@2000 상승하지만 **marginal(한계 효용 체감)**
- K가 클수록 online RT 증가
- → K는 시나리오 특성에 따라 실험적으로 결정해야 함 (trade-off)

---

## 7. 핵심 기여 & 인사이트

1. **검색 단계에 수천 개 장기 행동 도입** — 기존 검색이 수십 개에 머물던 한계를 돌파, 랭킹과의 일관성 확보
2. **Category-Aware 계층화** — O(L)→O(L/N) 복잡도 감소 + 멀티 인터레스트로 gradient conflict 해소 + 랭킹과 granularity 정렬
3. **학습 목표 재정의(카테고리 내 예측 + 카테고리 동질 배치)** — category leakage 차단으로 offline 지표 부풀림/학습 붕괴 방지
4. **Cascaded Category→Item + PGIN** — "카테고리 먼저 예측 → 그 안에서 병렬 검색"으로 지연 제약 해결, offline-online 일관성 유지
5. **Pointer-Generator 차용** — 반복 관심(pointer) vs 신규 관심(generator)을 게이팅으로 균형

### 배울 점 / 생각할 거리
- **"먼저 좁히고 나중에 찾는다"** 는 계층적 검색 철학이 핵심. 랭킹의 SIM/ETA식 category search를 검색 단계로 가져온 셈.
- 학습-서빙 정렬(training-serving consistency)을 손실 설계 + negative 샘플링 + 서빙 제약까지 일관되게 맞춘 점이 실무적으로 인상적.
- 논문 자체는 4쪽 short paper라 수식/디테일은 압축적. 재현하려면 카테고리 granularity 정의, negative 샘플링 구현이 관건.
