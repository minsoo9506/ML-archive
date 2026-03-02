# Decoupled Entity Representation Learning for Pinterest Ads Ranking
**Pinterest, 2025** | arXiv:2509.04337

## 한 줄 요약
Pinterest 광고 랭킹을 위해 Upstream-Downstream을 분리한 DERM(Decoupled Entity Representation Model) 프레임워크를 제안, CTR/CVR 예측 모두에서 유의미한 성능 향상을 달성.

---

## 문제 정의
- 유저, 아이템, 쿼리 모두 임베딩화해서 광고 추천 랭킹 파이프라인에서 사용됨
- Pinterest는 Home Feed, RePin, Search 등 다양한 서피스와 클릭·전환 등 여러 최적화 타입을 가짐
- 각 서피스·태스크마다 별도 데이터셋과 모델이 존재 → **표현이 파편화**되고 정보 공유가 어려움
- 인프라 제약으로 모델이 일부 피처만 사용 → 유저 행동의 전체적 뷰 확보 불가
- 도메인 간 효율적인 지식 전이(Knowledge Transfer)가 큰 과제

---

## 핵심 아이디어: Upstream-Downstream 분리 패러다임

```
[다양한 데이터 소스]
    ↓
[Upstream DERM 모델들] ← CTR 데이터, CVR 데이터 각각 학습
    ↓ (매일 오프라인 생성 → Key-Value 피처 스토어)
[Entity Embeddings] ← Moving Average로 안정화
    ↓
[Downstream 태스크들] ← CTR 예측, CVR 예측, 광고 검색
```

**핵심**: Upstream 임베딩을 실시간이 아닌 **오프라인으로 사전 계산**하여 Downstream과 비동기 서빙 → 확장성 확보

---

## DERM 모델 아키텍처

### 아키텍처 다이어그램
- 아래 아키텍처로 user, item embedding 을 일배치로 업데이트
```
┌─────────────────────────────────────────────────────────────────────┐
│                        UPSTREAM MODEL (DERM)                        │
│                                                                     │
│                                              ┌──────────────────┐   │
│                                              │   Supervised     │   │
│                                              │      Loss        │   │
│                                              │  (CTR / CVR)     │   │
│                                              └────────▲─────────┘   │
│                                                       │             │
│                                              ┌────────┴────────┐    │
│                                              │      MLP        │    │
│                                              └────────▲────────┘    │
│                                                       │             │
│                  ┌────────────────────────────────────┴──────────┐  │
│                  │       Overall Interaction Tower (DHEN)         │ │
│                  │    [Context / Interaction Features도 입력]      │ │
│                  └──────────▲──────────────────────────▲──────────┘ │
│                             │                          │             │
│         ┌───────────────────┴──┐   ┌───────────────────┴──────────┐  │
│         │    User Embedding    │   │      Pin Embedding           │  │
│         │      (ℓ₂ norm)       │   │       (ℓ₂ norm)              │  │
│         └───────▲──────┬───────┘   └────────┬──────────▲──────────┘  │
│                 │      │                    │           │            │
│                 │   ┌──┴────────────────────┴──┐        │            │
│                 │   │     Self-supervised Loss  │        │           │
│                 │   │        (Contrastive)      │        │           │
│                 │   └───────────────────────────┘        │           │
│         ┌───────┴──────────┐         ┌───────────────────┴──────┐    │
│         │   User Tower     │         │       Pin Tower           │   │
│         │     (DHEN)       │         │         (DHEN)            │   │
│         │ ┌──────────────┐ │         │   ┌──────────────┐        │   │
│         │ │[MaskNet|MLP] │ │         │   │[MaskNet|MLP] │        │   │
│         │ │[Transformer  │ │         │   │[Transformer  │        │   │
│         │ │    |MLP]     │ │         │   │    |MLP]     │        │   │
│         │ │     ...      │ │         │   │     ...      │        │   │
│         │ └──────────────┘ │         │   └──────────────┘        │   │
│         └───────▲──────────┘         └───────────────▲───────────┘   │
│                 │                                     │              │
│   ┌─────────────┴─────────────────────────────────────┴───────────┐  │
│   │                  Feature Preprocessing Layer                  │  │
│   │   [embedding lookup]  [dense norm]  [timestamp transform] ... │  │
│   └──────────▲─────────────────▲────────────────────▲─────────────┘  │
│              │                 │                    │               │ 
│   ┌──────────┴──┐    ┌───────────────┐    ┌─────────┴─────────┐     │
│   │ User Feature│    │  Pin Features │    │   Other Features   │     │
│   └─────────────┘    └───────────────┘    └────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

### Upstream → Downstream 전체 파이프라인

```
 ┌───────────┐  ┌───────────┐  ┌───────────┐
 │ CTR 예측  │  │ CVR 예측  │  │ ANN 검색  │
 │(MoE+DCNv2)│  │(DHEN 기반)│  │(임베딩 NN)│
 └─────▲─────┘  └─────▲─────┘  └─────▲─────┘
       └───────────────┼───────────────┘
                       │
          ┌────────────┴────────────┐
          │   Key-Value Feature     │
          │        Store            │
          └────────────▲────────────┘
                       │
           Moving Average 안정화
           E(t) = 0.8·E(t-1) + 0.2·E_daily(t)
                       ▲
           [매일 오프라인 임베딩 생성]
                       ▲
             ┌─────────┴──────────┐
    ┌────────┴────────┐   ┌───────┴─────────┐
    │   DERM (CTR)    │   │   DERM (CVR)    │   ...
    │  Upstream Model │   │  Upstream Model │
    └────────▲────────┘   └────────▲────────┘
             └──────────┬──────────┘
                        ▲
  ┌─────────────────────┴───────────────────────┐
  │  Ad Engagement Data  │  Ad Conversion Data  │  ...
  └─────────────────────────────────────────────┘
```

### 3-Tower 구조
| Tower | 입력 | 역할 |
|-------|------|------|
| **User Tower** | 유저 활동 시퀀스, 인구통계, 관심사, 카운팅 피처, 사전학습 표현 | 유저 임베딩 생성 |
| **Pin Tower** | 콘텐츠 메타데이터, 집계 카운팅 피처, 사전학습 표현 | Pin(광고) 임베딩 생성 |
| **Overall Interaction Tower** | 두 타워 출력 + 컨텍스트/인터랙션 피처 | 최종 교차 처리 |

- 각 Tower의 백본: **DHEN** (Deep Hierarchical Ensemble Network)
  - 각 레이어: MaskNet, Transformer, MLP 중 두 모듈 조합

### Self-Supervised Learning (Contrastive Loss)
- Positive pair: 클릭/전환 레이블이 있는 (유저, 타겟 Pin) 쌍
- In-batch 랜덤 네거티브 샘플링
- **Sampled Softmax Loss** with 샘플링 바이어스 보정항 Q

$$L_{u_i, p_i} = -\log \frac{e^{u_i^\top p_i / \tau - \log Q(p_i)}}{e^{u_i^\top p_i / \tau - \log Q(p_i)} + \sum_{p_j \in N_i^-} e^{u_i^\top p_j / \tau - \log Q(p_j)}}$$

- Final Loss = Supervised Loss (CTR/CVR) + Self-supervised Contrastive Loss

### Interaction Tower가 분리된 이유
- 컨텍스트/인터랙션 피처가 Entity 임베딩을 오염시키는 것을 방지
- Entity 임베딩의 **안정성 및 범용성** 보장

---

## 임베딩 안정화: Moving Average

매일 새로 생성된 임베딩과 기존 임베딩을 가중 평균:

$$E_{agg}(t) = w \cdot E_{agg}(t-1) + (1-w) \cdot E_{daily}(t)$$

- 최적 가중치: **w = 0.8** (과거 임베딩에 더 높은 가중치)
- 비교 실험 결과:

| ACC | MA w=0.2 | MA w=0.5 | **MA w=0.8** | AP |
|-----|----------|----------|--------------|-----|
| 0.02% | 0.03% | 0.06% | **0.09%** | 0.03% |

---

## Downstream 태스크

### CTR 모델
- **MoE (Mixture of Experts)** 기반 멀티태스크 학습
- DCNv2로 피처 인터랙션
- Transformer Encoder로 유저 시퀀스 처리
- DERM 임베딩을 Concatenation Layer에 추가 입력

### CVR 모델
- DHEN 기반 아키텍처 (Upstream과 유사)
- 동일한 방식으로 DERM 임베딩 활용

### 차원 축소 Projection Layer
- DERM 임베딩 차원 축소로 인프라 비용 절감
- 512차원 Projection: ROC-AUC 0.18% lift (vs. 미사용 시 0.19%)
  - 성능 손실 최소화하면서 학습 처리량 5.68% 향상
  - 연간 약 **21.2만 달러** 인프라 비용 절감

---

## 실험 결과

### Offline: CTR 다운스트림에서 임베딩 조합 효과

| CTR User | CTR Item | CVR User | CVR Item | ROC-AUC Lift | PR-AUC Lift |
|----------|----------|----------|----------|--------------|-------------|
| ✓ | - | - | - | 0.095% | 0.2% |
| ✓ | ✓ | - | - | 0.160% | 0.4% |
| **✓** | **✓** | **✓** | **✓** | **0.190%** | **0.46%** |

→ CTR + CVR 양쪽 임베딩을 모두 사용할 때 최고 성능 (크로스 도메인 전이 효과)

### Online A/B Test: CTR 예측 (Related Pins 서피스, 10% 트래픽)

| 지표 | 전체 플랫폼 | RP 서피스만 | 쇼핑 광고 | 비쇼핑 광고 |
|------|------------|------------|----------|------------|
| CTR lift | **+1.38%** | +2.83% | +1.31% | +1.14% |
| gCTR lift | **+1.96%** | +3.74% | +1.83% | +1.83% |

### Online A/B Test: CVR 예측 (9% 트래픽, 웹 전환 최적화)
- **CPA -1.61%** 감소 (낮을수록 좋음)
- **cCVR +2.75%** 향상 (클릭 전환율)
- **vCVR +2.8%** 향상 (뷰 전환율)

---

## 서빙 파이프라인

1. **Raw Feature 중복 제거**: 하루 중 마지막 임베딩만 유지 (일 내 변동 미미)
2. **Feature Aggregation**: Moving Average로 일별 임베딩 병합
3. **Coverage 관리**: 최근 3개월 활성 유저/Pin 임베딩만 보관
4. **Online Serving**:
   - User 임베딩 → Key-Value 피처 스토어
   - Pin 임베딩 → Pin 인덱싱 (랭킹 & 검색용)

---

## 기존 대비 혁신점
1. **Multi-tower + Overtower 분리**: 컨텍스트 오염 없이 순수 Entity 임베딩 학습
2. **Supervised + Self-supervised 결합**: Contrastive Loss로 임베딩 품질 향상
3. **Moving Average 안정화**: 급격한 임베딩 변화 방지, 커버리지 확보
4. **크로스 도메인 전이**: CTR 도메인 임베딩 → CVR 태스크에서도 효과적

---

## Future Work
- 단일 Upstream 모델로 통합된 임베딩 공간 학습
- Non-ID 피처만 사용하는 컨텐츠 기반 임베딩 개발 (cross-domain transfer learning 강화)
- 추가 데이터 소스 통합 및 학습 효율성 개선
