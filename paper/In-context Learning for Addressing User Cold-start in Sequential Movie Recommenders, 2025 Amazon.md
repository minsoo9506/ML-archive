# In-context Learning for Addressing User Cold-start in Sequential Movie Recommenders (Amazon, 2025)

- RecSys '25 (Amazon, 비디오 스트리밍 서비스)
- 저자: Xurong Liang, Vu Nguyen, Vuong Le, Paul Albert, Julien Monteil
- 링크: https://doi.org/10.1145/3705328.3748109

## 1. 문제의식
- Sequential recommender의 고질적 문제: **user cold-start**.
  - **Absolute cold user**: 시청 이력 0개.
  - **Nearly cold user**: 시청 이력 1~5개.
  - Amazon 스트리밍에서는 이 두 그룹이 사용자 베이스의 상당 비중을 차지.
- 기존 sequential 모델은 충분한 interaction history가 있어야 잘 동작 → cold user에서 sub-optimal.
- 보유 데이터 중 **user metadata(인구통계 등)** 는 거의 활용되지 않음.
  - 이유: 데이터 가용성 낮고 low-level이라 직접적으로 선호 추론이 어려움.
  - 그러나 그룹 단위의 선호 트렌드는 분명히 존재함 → cold-start에서 가치 있음.

## 2. 핵심 아이디어
**LLM으로 user metadata를 해석해 "가상의 시청 아이템(imaginary items)"을 생성**한 뒤, 이를 sequential recommender의 입력 시퀀스에 끼워 넣어 cold-start를 보완하자.

- LLM의 사전학습된 world knowledge가 "demographic ↔ 선호" 상관을 이미 내포한다는 가정.
- 생성된 imaginary item은 표준화된 key-value 속성(title, genre 등)으로 표현 → content-based backbone(RecFormer 등)에 그대로 투입 가능.

## 3. 4-step 프레임워크

### Step 1. Prompt building
프롬프트 구성:
- **Instructions**: user metadata / item metadata JSON 포맷, 길이 제한, 카테고리 옵션(genre 등) 안내.
- **Context**: c개의 (user metadata, 첫 interaction item) 예시 — in-context examples.
- **Query**: 타깃 user의 metadata.

In-context example 선택 3가지 전략:
- **a. Random**: 학습셋에서 랜덤 샘플링. 다양성 확보.
- **b. Metadata matching**: 타깃 유저와 한 개 이상의 속성 필드가 겹치는 유저 풀에서 샘플링.
- **c. Top-c Nearest Neighbors**: 아이템 feature를 자연어로 만들어 텍스트 임베딩(BERT/RoBERTa/Longformer 류)으로 거리 계산, 가장 가까운 c명을 선택.

### Step 2. LLM generation
- 프롬프트를 LLM(실험: **Llama-3.3-70B-Instruct**)에 넣어 k개 imaginary item 생성.
  - 각 item = textual / categorical attribute + timestamp.
- 실제 카탈로그 아이템과 일치할 필요 없음 (content-based backbone이라 OK).
- 필요 시 ID 기반 모델용으로 임베딩 매칭으로 실제 item과 매핑도 가능.

### Step 3. Imaginary–historical input fusion
LLM이 만든 X̃ = {x̃₁,…,x̃_k}와 historical X = {x₁,…,x_n}를 결합하는 두 가지 방식:

- **a. Early Fusion**: 생성 timestamp로 정렬 후 historical sequence 앞에 그대로 concat → X̂ = [X̃, X].
  - Absolute cold면 historical이 비어 있으므로 input은 imaginary만.
  - 직관: metadata에서 추론한 정적 user intent가 실제 상호작용보다 선행.
- **b. Late Fusion**: k개의 imaginary 변형 각각을 historical과 따로 묶음 → x̂ᵢ = [x̃ᵢ, X], i=1..k.
  - 추천기 encoder를 k번 돌려 hidden h_i를 얻고 **average pooling**으로 합쳐 디코더에 입력.
  - imaginary 생성의 stochastic 분포를 활용.

### Step 4. Sequential recommendation
- Backbone은 일반적인 encoder–decoder 구조의 sequential recommender.
- 실험 backbone: **RecFormer**.
- Early fusion은 단일 encoder pass, late fusion은 k번 pass + 평균.

## 4. 실험 결과

### 데이터셋
- **ML-1M** (공개 MovieLens 1M): 일부 유저의 history를 잘라 cold-start 시뮬레이션.
- **Amazon Proprietary (AP)**: 실제로 cold/nearly cold가 많은 내부 데이터.

### 평가
- Leave-one-out, ground truth + 100 negatives 안에서 NDCG@20, Recall@20.
- 하이퍼파라미터: c=10 (in-context example 수), k=5 (imaginary item 수).
- Baseline: **no generator** — absolute cold에는 임의 영화 하나를 끼워 넣고, 나머지는 그대로.

### 핵심 결과 (Table 1, RecFormer 기준)
- LLM imaginary augmentation이 NDCG/Recall을 **일관되게 향상**.
- 효과는 **cold(0) / nearly cold(ML-1M 1~10, AP 1~4)** 에서 특히 큼.
- Historical interaction이 늘어날수록 효과는 점차 줄어 baseline에 수렴.

### Insight
- **In-context 샘플링 전략**:
  - 1개 이상 interaction이 있는 유저 → **random** 샘플링이 강함 (탐색적이라 유리).
  - Absolute cold(0 interaction) → **c-NN**이 더 안전 (보수적인 예시가 더 효과적).
  - 직관: cold일수록 탐색은 위험, 어느 정도 history가 있으면 탐색이 도움.
- **Fusion 전략**:
  - **ML-1M**: early fusion 우세 (콘텐츠 다양성이 높음).
  - **AP**: late fusion 우세 (콘텐츠 다양성 낮아 표현 공간이 compact → interpolation이 더 의미 있음).

## 5. Practical considerations
- LLM 추론 비용이 큼: 유저당 평균 **72s (ML-1M)**, **175s (AP)**.
- 따라서 **오프라인으로 미리 생성** → user metadata를 키로 imaginary item을 캐싱, 서빙 시점에는 lookup으로 latency 충족.
- LLM이 미리 정의한 categorical 값(genre 등) 밖으로 벗어나는 경우가 종종 있어 **재프롬프트로 valid 결과 강제**.
- Fine-tuning 없이 zero-shot으로 한 한계가 있어 향후 도메인 fine-tuning을 계획.

## 6. 핵심 takeaway
- **User metadata + LLM의 world knowledge**를 결합해 "있을 법한 시청 이력"을 생성하면 cold-start 해소에 유효.
- ID 기반이 아닌 **content-based sequential recommender**라면 가짜 아이템이 실제 카탈로그와 일치할 필요가 없어 통합이 쉬움.
- Cold-start는 보수적(c-NN) 예시 + early fusion, 어느 정도 history 있는 경우엔 random + 데이터 특성에 맞는 fusion이 유리 — **유저 상태별 전략 분기**가 중요.
- LLM 호출 비용은 **메타데이터 단위 캐싱**으로 운영 가능.
