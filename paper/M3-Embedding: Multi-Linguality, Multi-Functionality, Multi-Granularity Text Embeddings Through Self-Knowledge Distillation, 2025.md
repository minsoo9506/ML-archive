# M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation, 2025

- BAAI (Beijing Academy of Artificial Intelligence)
- ACL 2024 Findings
- [ArXiv](https://arxiv.org/abs/2402.03216) | [HuggingFace](https://huggingface.co/BAAI/bge-m3)

## 개요

하나의 임베딩 모델로 세 가지 "M"을 동시에 달성하는 것이 목표이다.

| M | 의미 | 내용 |
|---|------|------|
| **Multi-Linguality** | 다국어 | 100개 이상 언어 지원, 크로스링구얼 검색 가능 |
| **Multi-Functionality** | 다기능 | Dense, Sparse, Multi-vector 3가지 검색을 하나의 모델로 |
| **Multi-Granularity** | 다양한 입력 길이 | 짧은 문장 ~ 최대 8192 토큰의 긴 문서까지 처리 |

## 아키텍처 (XLM-RoBERTa 기반)

하나의 인코더에서 토큰 임베딩을 추출한 뒤, 세 가지 검색 방식을 동시에 수행한다.

1. **Dense Retrieval** — `[CLS]` 토큰의 임베딩을 사용하여 쿼리-문서 간 내적(inner product)으로 유사도 계산
2. **Sparse (Lexical) Retrieval** — 각 토큰에 학습 가능한 projection을 적용하여 term importance weight를 추정하고, 쿼리와 문서 간 공통 토큰의 가중치 곱의 합으로 유사도 계산 (학습 가능한 BM25)
3. **Multi-Vector Retrieval** — ColBERT 스타일의 late interaction. 모든 토큰 임베딩을 활용하여 쿼리의 각 토큰과 문서 토큰 간 최대 유사도의 평균으로 계산

## 핵심 기법: Self-Knowledge Distillation

**문제**: Dense와 Sparse는 본질적으로 다른 방식이라, 단순히 같이 학습하면 서로 간섭하여 특히 Sparse 성능이 크게 하락한다.

**해결**: 앙상블 학습의 원리를 활용한다.

```
통합 점수 = w₁·s_dense + w₂·s_sparse + w₃·s_multi-vector
```

1. 세 가지 검색 함수의 relevance score를 가중 결합하여 통합 점수를 만든다
2. 이 통합 점수를 teacher signal(soft label)로 사용한다
3. 각 개별 검색 함수를 이 teacher에 맞춰 knowledge distillation으로 학습한다

즉, 모델이 자기 자신의 앙상블 결과를 교사로 삼아 각 기능을 강화하는 구조이다. 이를 통해 dense/sparse/multi-vector가 서로를 보완하며 함께 개선된다.

Ablation에서 Self-KD를 제거하면 Sparse 성능이 53.9 → 36.7로 급락하여, 이 기법의 중요성이 확인되었다.

## 학습 과정

M3-Embedding은 3단계 점진적 학습을 거친다. 각 단계가 이전 단계 위에 쌓이는 커리큘럼 구조이다.

```
XLM-RoBERTa
    │
    ▼ Stage 0: RetroMAE (8192 토큰 확장)
    │
    ▼ Stage 1: 비지도 대조학습 (12억 쌍, Dense만)
    │
    ▼ Stage 2: 미세조정 워밍업 (6K steps, 개별 기능)
    │
    ▼ Stage 2: Self-Knowledge Distillation (Dense + Sparse + Multi-vector 통합 학습)
    │
    ▼ M3-Embedding (bge-m3)
```

### Stage 0: RetroMAE 사전학습 (인코더 초기화)

XLM-RoBERTa를 바로 쓰지 않고, RetroMAE(Masked Auto-Encoding) 방식으로 먼저 적응시킨다.

- **목적**: 최대 시퀀스 길이를 8192 토큰으로 확장하고, 검색에 적합한 표현력을 확보
- **데이터**: Pile, Wudao, mC4에서 105개 언어, 1.84억 샘플
- **설정**: LR 7×10⁻⁵, batch 32 × gradient accumulation 16, 20K steps
- **하드웨어**: A100 40GB × 32대

### Stage 1: 비지도 대조 사전학습 (Dense Retrieval만)

대규모 다국어 코퍼스에서 자연적으로 존재하는 텍스트 쌍을 활용하여 Dense Retrieval만 학습한다.

**데이터 규모**: 194개 언어, 12억 텍스트 쌍, 2655개 크로스링구얼 대응

**텍스트 쌍 구성 방식** (별도 라벨링 없이 자연 구조 활용):

| 소스 | 쌍 구성 방식 |
|------|-------------|
| Wikipedia | 제목 ↔ 본문 |
| S2ORC (학술 논문) | 제목 ↔ 초록 |
| xP3 | instruction ↔ output |
| mC4, CC-News | 제목 ↔ 본문 |
| NLLB, CCMatrix | 언어A 문장 ↔ 언어B 번역문 |

**배치 크기 최적화 — 길이별 그룹핑**:

길이가 비슷한 데이터끼리 묶어서 패딩 낭비를 줄이고 배치 크기를 극대화한다.

| 시퀀스 길이 | 배치 크기 |
|------------|----------|
| 0–500 토큰 | 67,200 |
| 500–1000 | 54,720 |
| 1000–2000 | 37,248 |
| 2000–8192 | 9,984 |

**학습 설정**: Query 최대 512 토큰, Passage 최대 8192 토큰, LR 5×10⁻⁵, 25K steps, A800 80GB × 96대

### Stage 2: 지도 미세조정 + Self-Knowledge Distillation

이 단계에서 비로소 3가지 검색 기능을 동시에 학습한다.

#### 학습 데이터

| 언어 | 샘플 수 | 출처 |
|------|---------|------|
| 영어 | 110만 | MS MARCO, HotpotQA, NQ, SQuAD, PubMedQA 등 |
| 중국어 | 38.6만 | DuReader, T2-Ranking, CMedQAv2 등 |
| 다국어 | 8.9만 | MIRACL, Mr.TyDi |
| 합성 장문서 | 4.1만 | GPT-3.5로 13개 언어 장문서에서 QA 쌍 생성 |

각 쿼리당 7개의 hard negative를 ANCE 방식으로 마이닝한다.

#### 손실 함수 구성

**Step 1 — 각 검색 방식의 기본 Contrastive Loss**:

$$\mathcal{L}_{s} = -\log\frac{\exp(s(q, p^*) / \tau)}{\sum_{p \in \{p^*, P'\}} \exp(s(q, p) / \tau)}$$

- $s$: dense/sparse/multi-vector 중 하나의 유사도 함수
- $p^*$: positive
- $P'$: negative 집합

**Step 2 — 통합 점수 계산 (앙상블 교사)**:

$$s_{inter} = w_1 \cdot s_{dense} + w_2 \cdot s_{sparse} + w_3 \cdot s_{multi\text{-}vector}$$

- 가중치: $w_1 = 1,\ w_2 = 0.3,\ w_3 = 1$
- Sparse에 낮은 가중치를 주는 이유는 sparse score의 스케일이 다른 두 방식과 다르기 때문이다.

**Step 3 — Self-Knowledge Distillation Loss**:

통합 점수를 softmax로 확률 분포로 변환한 뒤, 각 개별 함수의 출력을 이 분포에 정렬시킨다:

$$\mathcal{L}'_* = -p(s_{inter}) \cdot \log p(s_*)$$

**Step 4 — 최종 손실 결합**:

$$\mathcal{L}_{final} = \frac{\mathcal{L} + \mathcal{L}'}{2}$$

- $\mathcal{L} = \lambda_1 \mathcal{L}_{dense} + \lambda_2 \mathcal{L}_{sparse} + \lambda_3 \mathcal{L}_{multi}$
- 가중치: $\lambda_1=1, \lambda_2=0.1, \lambda_3=1$
- Sparse의 loss 가중치가 0.1로 낮은 이유는, sparse가 contrastive loss만으로는 학습이 어렵고 distillation에 더 의존하기 때문이다.

#### 웜업 단계

미세조정 초반 약 6000 steps 동안은 Self-KD 없이 각 검색 기능을 개별적으로 워밍업한다. 각 기능이 어느 정도 안정된 후에야 통합 점수가 의미 있는 교사 신호가 될 수 있기 때문이다.

### 보너스: MCLS (장문서 추론 기법)

장문서 학습 데이터가 부족할 때 사용하는 추가 학습 없는 추론 기법이다.

- 256 토큰마다 `[CLS]` 토큰을 삽입
- 각 `[CLS]`의 hidden state를 평균하여 최종 임베딩 생성
- 장문서의 여러 구간을 고르게 반영 → nDCG@10이 41.2 → 45.0으로 향상

## 주요 실험 결과

| 벤치마크 | 성능 | 비교 |
|----------|------|------|
| MIRACL (다국어 검색) | 71.5 nDCG@10 (하이브리드) | mE5-large 66.6 대비 우위 |
| MKQA (크로스링구얼) | 75.5 Recall@100 | 저자원 언어(아랍어, 크메르어 등)에서 특히 강점 |
| MLDR (장문서 검색) | 65.0 nDCG@10 | 장문서에서는 Sparse가 개별적으로 가장 강력 |

세 가지 검색 방식을 결합(하이브리드)하면 개별 방식보다 항상 성능이 향상되는 것이 핵심 발견이다.
