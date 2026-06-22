# Scaling Retrieval for Web-Scale Recommenders: Lessons from Inverted Indexes to Embedding Search

- RecSys 2025, LinkedIn
- Yuchin Juan, Jianqiang Shen, Shaobo Zhang, Qianqi Shen, Caleb Johnson, Luke Simon, Liangjie Hong, Wenjing Zhang

## 사전 지식: Inverted Index란?

- **단어 → 그 단어를 포함한 문서 목록(posting list)** 형태로 미리 구축해두는, 키워드 검색용 자료구조 (도서관 책의 "찾아보기"와 동일한 원리)
- "문서 → 단어"(forward index)의 방향을 뒤집은 것이라 "inverted(역색인)"
  - 예: `python → [문서1, 문서2]`, `backend → [문서1, 문서3]`
  - "python AND backend" 검색 시 두 목록의 교집합만 계산 → 전체 문서를 뒤지지 않고 빠르게 후보 추출
- Apache Lucene, Elasticsearch 등 전통적 검색 엔진의 기반 (LinkedIn의 **Galene**도 이 방식)
- **핵심 한계**: term(단어)이 **정확히 일치**해야 매칭됨 → 의미는 같지만 표현이 다른 경우("GenAI" vs "LLM") 매칭 실패. 이 의미 매칭 문제를 풀기 위해 임베딩 기반 검색(EBR)으로 넘어가는 것이 이 논문의 흐름.

## 핵심 요약

LinkedIn의 retrieval 레이어가 **CPU 기반 inverted index → GPU 기반 embedding 검색**으로 진화한 여정을 정리한 논문. Job matching(주당 6,500만+ 구직자)을 사례로, term matching의 한계 → learning-to-retrieve → EBR → GPU 기반 통합 검색 시스템으로 발전. 최종적으로 **서빙 인프라 비용 75% 절감**, 실험/반복 속도 30% 향상, A/B 테스트에서 **job application +4.6%, 예산 소진율 +5.8%** 달성.

## 배경: 2단계 아키텍처

- 웹 스케일 추천의 공통 패턴: **retrieval(후보 빠르게 좁히기) → ranking(무거운 모델로 best 선정)**
- 이 논문은 **retrieval 레이어의 진화**에 초점
- 산업 트렌드: 수동 엔지니어링 파이프라인 → ML 기반 시스템
- Rich Sutton의 "Bitter Lesson" 철학을 인용: 연산량이 커져도 계속 스케일하는 general-purpose 방법의 힘

# 1단계: CPU 기반 Inverted Index 시스템

## Inverted Index (Galene)

- **Galene**: Apache Lucene + Hadoop 인덱싱 파이프라인 기반의 CPU inverted index 프레임워크
- 다단계 아키텍처: **Federator, Broker, Searcher** 컴포넌트로 분산 쿼리 실행
- **Base-Middle-Live (BML)** 인덱싱: 대규모 데이터셋 쿼리 성능과 저지연 실시간 업데이트의 균형
- 장점: 빠르고 설명 가능한 exact term matching
- 한계: 프로필이 풍부해지고 개인화 요구가 커지면서 수작업 clause, query expansion, query rewriting에 의존 → 후보셋이 너무 넓거나 좁아지는 문제가 brittle

## Term Matching의 한계 (Table 1)

| 범주 | 예시 | 문제 |
|---|---|---|
| Seniority | Director | 회사마다 요구 경력이 다름 |
| Role | Admin | System / DB / Office Admin 중 무엇? |
| Specificity | Cloud Computing | AWS, Azure, GCP 중 무엇? 전부? |
| Taxonomy | GenAI | taxonomy 상 대응되는 skill ID가 무엇? |

## Inverted Index 위의 Retrieval 모델

- **Shutterspeed**: 학습된 relevance score로 user attribute를 랭킹, top-K 부분집합으로 disjunctive retrieval. TF-IDF에서 영감 + behavior 기반 trend score → 지연/engagement 개선
- **Graph 기반 방법**: confirmed hire 같은 engagement 시그널로 member-item attribute 간 link 학습 → 설명 가능하고 적응적, 학습된 link를 inverted index 쿼리로 변환
- 공통 한계: inverted index가 **추출된 attribute**(job title, company, skill, location 등 taxonomy 기반)에 의존 → 구조화된 attribute 모델링/추출 자체가 본질적으로 복잡

## EBR로의 초기 시도 (Galene + ANN)

- Galene를 확장해 **ANN(approximate nearest neighbor) 검색** 지원 → 초기 성과는 있었으나 한계 노출
- 생산성 장벽: 모델 반복마다 cluster centroid, quantization codebook 학습 등 복잡한 워크플로 필요 → relevance 저하 위험
- 신규 임베딩 적용이 **주간 오프라인 인덱스 빌드**에 의존 → 반복 지연이 한 달까지 늘어남
- inverted index 구조상 **고차원 임베딩 사용 비용이 prohibitive**

# 2단계: GPU 기반 Retrieval 시스템 (새로운 패러다임)

CPU/inverted index 기반은 모델링 유연성이 부족하고 multi-objective 최적화 구현이 어려움 → GPU 가속 EBR로 전환.

## 설계 원칙

- **임의의 복잡도에 적응 가능한 meta-component** 도입 (특정 규칙/근사를 하드코딩하지 않음)
- 고정된 휴리스틱/사전계산 중간값 → **동적, 데이터 기반 scoring engine**으로 전환
- AI 엔지니어가 GPU의 막대한 연산력을 활용해 검색 전략을 직접 구성/반복

## 핵심 구조 (3개 컴포넌트)

**핵심 아이디어**: inverted index의 "posting list 교집합 탐색"을 버리고, 검색을 통째로 **거대한 행렬 연산**으로 재정의. GPU가 가장 잘하는 일이 행렬곱이기 때문에, 모든 아이템(구인공고)을 두 종류의 행렬로 표현해 GPU 메모리에 올려두고 연산한다.

- **Sparse matrix (희소 행렬) — 키워드(term) 매칭용**
  - **행 = 아이템, 열 = attribute(속성/단어)**. 공고가 가진 속성 칸만 1, 나머지는 0 → 대부분 0이라 sparse
  - inverted index가 하던 키워드 매칭을 **0/1 행렬**로 표현한 것

    ```
              python  seoul  backend  java
    공고A        1       1       1      0
    공고B        1       0       0      1
    공고C        0       1       1      0
    ```
- **Dense matrix (밀집 행렬) — 의미(semantic) 매칭용**
  - **각 행 = 아이템의 embedding 벡터**(의미를 실수 N개로 압축, 논문은 3584차원). 모든 칸이 실수값으로 꽉 차서 dense
  - 단어가 달라도("GenAI" vs "LLM") 벡터가 가까우면 매칭 → term 매칭의 한계를 보완

    ```
    공고A → [0.12, -0.88, 0.04, ... ]   (3584개 숫자)
    공고B → [0.91,  0.33, -0.50, ... ]
    ```
- **Messenger — 컴포넌트 간 데이터 전달 통로**
  - 중간 연산 결과를 컴포넌트끼리 주고받음. **zero-copy**(데이터를 복사하지 않고 GPU 메모리 상의 위치만 전달) → 큰 행렬 복사 오버헤드 제거

**이 구조가 주는 장점**

- 모든 sparse/dense 행렬을 **GPU 메모리에 상주**시켜 직접 연산 → 진정한 **hybrid retrieval (TBR + EBR)을 단일 GPU 파이프라인에서** 지원
  - **Term query**: 쿼리를 Conjunctive Normal Form(`(A or B) and C` 같은 논리식)으로 변환 → 대규모 병렬 sparse matrix 연산 (early stopping 같은 휴리스틱 불필요, GPU가 전부 계산)
  - **Embedding query**: dense matrix 곱으로 모든 아이템과 유사도를 다 계산하는 **exhaustive KNN**
- **근사 검색을 안 써도 됨**: 보통 임베딩 검색은 아이템이 많아 IVFPQ·HNSW 같은 **근사(approximate)** 방법을 쓰지만 정확도 저하 + centroid/codebook 튜닝 지옥이 따름(앞선 EBR의 고통). GPU 행렬곱이 워낙 빨라 **400만 아이템을 전부 exhaustive하게 계산**해도 SLA 내(A100 1장) → 정확하면서 빠르고 튜닝 불필요
- GPU 접근 패턴 최적화 메모리 레이아웃: sparse는 column-major, dense는 row/column-major 혼합

## 그래서 검색 한 번은 어떻게 도는가

> 핵심: 검색 = **쿼리를 벡터로 만들어 → 거대한 아이템 행렬과 곱해(모든 아이템 점수 동시 계산) → top-K 뽑기**. inverted index의 "posting list 탐색"이 GPU에선 "행렬곱 + top-K"로 바뀐다.

### A. Term query (sparse matrix 키워드 검색)

"python AND backend" 검색 예시:

1. **쿼리를 벡터로**: 해당 단어 열만 1 → `q = [python=1, seoul=0, backend=1, java=0]`
2. **행렬 × 벡터 곱**: 각 공고가 쿼리 단어를 몇 개 가졌는지 점수가 한 번에 나옴

   ```
   공고A · q = 1·1 + 1·0 + 1·1 + 0·0 = 2   ← python, backend 둘 다
   공고B · q = 1·1 + 0·0 + 0·1 + 1·0 = 1   ← python만
   공고C · q = 0·1 + 1·0 + 1·1 + 0·0 = 1   ← backend만
   ```
3. **조건 필터**: AND는 점수 = 쿼리 단어 수(2)인 것만 → 공고A / OR은 점수 ≥ 1 전부
   - 공고를 하나씩 뒤지지 않고 **행렬곱 한 번으로 전체 공고 점수 동시 계산**. CNF(`(A or B) and C`)도 이 0/1 곱셈 + 조건 비교의 조합

### B. Embedding query (dense matrix 의미 검색)

1. **쿼리도 하나의 임베딩 벡터로** 변환 (3584차원)
2. **모든 공고 벡터와 유사도(내적/코사인) 계산** = 아이템 행렬 × 쿼리 벡터

   ```
   score_A = 공고A · q = 0.81
   score_B = 공고B · q = 0.12
   score_C = 공고C · q = 0.79
   ```
3. 점수 정렬 → **top-K 반환**. "GenAI"로 검색해도 "LLM" 공고 벡터가 가까우면 높은 점수
   - 400만 공고면 쿼리 1개를 **400만 개 전부와 내적**(exhaustive). 보통은 비싸서 근사를 쓰지만 GPU라 전부 계산 가능

### C. top-K 선택 & hybrid

- A·B 모두 결국 **"전체 아이템에 대한 점수 벡터"** 를 만든 뒤 **상위 K개만 추출** → 논문이 Bucket Sort 기반으로 최적화한 부분이 바로 이 단계
- **Hybrid/multi-objective**: term 점수 + embedding 점수 + custom scoring(cosine 너머)을 **단일 GPU pass에서 결합** → engagement·revenue 동시 최적화 가능

## 시스템 최적화 (Figure 1, QPS 누적 개선)

- Operation Opt (Bucket Sort 기반 top-K, zero-copy message passing): **+27%**
- Batching (GPU 활용 극대화, 큐잉 최소화): **+102%**
- Quantization (OP+ORP 유사: random permutation → k bin → sign-flip aggregation): **+279%**
- BF16 (bfloat16으로 메모리/연산 효율): **+27%**
- CUDA Kernel Opt + Quantization 제거 (cuBLAS/GEMM 튜닝, 근사 오차 제거): **+41%**
- Partition by Country (지리적 locality로 연산/메모리 비용 절감): **+71%**
- 최종: 임시로 도입한 quantization 레이어는 시스템 효율이 충분해지자 **완전히 제거** → 아키텍처 단순화 + 근사 오차/튜닝 오버헤드 제거

# 시스템 임팩트와 교훈

## 결과

- LinkedIn job matching(주당 6,500만+ 구직자)에서 **서빙 인프라 비용 75% 절감** (relevance 저하 없음)
- A100 한 장으로 **3584차원 임베딩 400만 아이템**을 SLA 내 처리
- 실험/반복 속도 **30% 향상**
- 첫 프로덕션 A/B 테스트(레거시 EBR 대비): **job application +4.6%, 예산 소진율 +5.8%**
- cosine similarity를 넘어선 커스텀 scoring + 학습 표현 + 사람이 만든 규칙을 결합, **단일 retrieval pass에서 multi-objective(engagement, revenue) 직접 최적화** 가능

## 교훈

1. **Hardware-aware 설계가 핵심**: CPU에서 통하던 가정이 GPU에선 안 통함 → 데이터 접근 패턴, 배칭, 메모리 레이아웃 재고 필요
2. **retrieval 단계의 모델링 유연성이 직접적으로 비즈니스 임팩트**로 연결 (복잡한 multi-objective 최적화를 서빙 파이프라인 앞단에서 수행)
3. inverted index는 인상적으로 스케일하지만 결국 **flexibility wall**에 부딪힘
4. 패러다임 전환은 인프라뿐 아니라 **tooling, monitoring, 운영 관행까지 full-stack 투자** 필요

# 결론 / Future Work

- retrieval 시스템은 모델/제품 복잡도와 함께 진화해야 함
- 향후: 더 표현력 있는 hybrid retrieval, ranking 레이어와의 긴밀한 통합, **LLM 기법을 통한 retrieval 지능 강화**
