# Enhancing Online Ranking Systems via Multi-Surface Co-Training for Content Understanding (Google/YouTube, RecSys '25)

- 논문: https://dl.acm.org/doi/10.1145/3705328.3748101
- 저자: Gwendolyn Zhao, Yilin Zheng et al. (Google / YouTube, Google DeepMind)
- 키워드: Recommender Systems, Content-based Recommendation, Learning to Rank

## 배경 / 문제의식
- 기존 비디오 추천 시스템은 user-video interaction memorization(CTR 예측 등)에 크게 의존 → 신규/인기 낮은(tail) 비디오에는 정보가 부족해 예측 정확도가 떨어짐.
- Content 이해 강화를 위해 multi-modality, content-based ID, LLM 기반 content adaptation 등 여러 연구가 있었음.
- 실무 ranker에서 content embedding을 쓸 때 네 가지 challenge 존재:
  1. **Task Alignment**: pre-trained embedding은 satisfied engagement와 align되어 있지 않음
  2. **Efficient Multi-Surface Learning**: 각 surface마다 따로 학습하면 비효율
  3. **Scalable Video Representation**: 무거운 content 이해 구조는 online ranker에 못 태움
  4. **Resource / Maintenance Efficiency**: 여러 surface에서 공유해야 유지보수 편함

## "Surface"가 뭐야?
- 논문에서 **surface = 비디오가 추천/소비되는 개별 피드(UI 지점)**.
- YouTube 기준 예시:
  - Homepage feed (홈 피드)
  - Watch Next (재생 중 옆/아래 추천)
  - Shorts feed
  - Search results 등
- 각 surface마다 보통 **별도 ranker, 별도 로그 데이터, 살짝 다른 task**(CTR / watch time 등 가중치·정의 차이)를 씀.
- **Multi-surface co-training**은 이걸 따로 학습하지 않고:
  1. 여러 surface의 **로그 데이터**를 합치고
  2. 여러 surface의 **task(label)** 를 같이 objective로 걸어서
  3. **하나의 content tower**를 공동 학습.
- 저자 bio 힌트: Gwendolyn Zhao = YouTube **Homepage Ranking**, Yilin Zheng = YouTube **WatchNext Ranking** 담당 → 두 surface(그 외 포함)가 co-training 대상으로 보임.
- 그래서 Table 2의 "single surface vs multi-surface" 비교가 핵심 — surface를 늘리면 데이터 다양성↑ → task 수를 줄여도 더 좋은 embedding.

## MulCo 시스템 구성
- **Multi-surface Co-training (MulCo)** 모델: 다양한 video discovery surface의 데이터/objective로 co-training.
- 구조:
  - **Base model**: 실제 online ranker와 동일 (같은 input feature)
  - **Content Tower**: 다양한 pre-trained content embedding (frame-level visual, audio 등)을 입력으로 받음
  - Candidate ID feature는 제거 → content feature만으로 학습되게 강제
- 학습 task는 online ranker와 동일한 satisfied engagement (CTR, watch time 등) → **task-aligned summarization & projection** 역할.

### Figure 1: MulCo System Setup

```
          [ MulCo (offline co-training) ]                 [ Prod Ranker (online) ]

            ┌───────────────────────┐                       ┌───────────────────────┐
            │   Top of model        │                       │   Top of model        │
            │   (MulCo tasks)       │                       │   (Prod tasks)        │
            └──────────▲────────────┘                       └──────────▲────────────┘
                       │                                               │
            ┌──────────┴────────────┐                       ┌──────────┴────────────┐
            │  Base   │   Content   │                       │       Ranker          │
            │  model  │   Tower ────┼──► MulCo emb ──┐      │       model           │
            └────▲────┴──────▲──────┘                │      └──────────▲────────────┘
                 │           │                       │                 │
                 │           │                       │          ┌──────┴──────┐
        ┌────────┴──┐   ┌────┴──────────┐            │          │ MulCo emb   │
        │ Ranker    │   │ Pre-trained   │            │          │ (feature)   │
        │ features  │   │ content emb.  │            │          └──────▲──────┘
        │ (no cand  │   │ (visual/audio │            │                 │
        │   ID)     │   │  /query ...)  │            │          ┌──────┴──────┐
        └────▲──────┘   └──────▲────────┘            │          │ Online &    │
             │                 │                     │          │ batch       │
             └────────┬────────┘                     │          │ inference   │
                      │                              │          └──────▲──────┘
            ┌─────────┴──────────┐                   │                 │
            │ Multiple surface   │                   │   Export saved  │
            │ training data      │                   └──── model ──────┘
            └────────────────────┘
```

### 2-stage 학습/서빙
- **Stage 1**: raw content embedding을 입력으로, content tower를 base model과 co-train.
- **Stage 2**: content tower를 freeze & export.
- 서빙:
  - **Batch inference**: 기존 비디오 backfill
  - **Online inference**: 신규 비디오, content feature 변화 모니터링
- 장점:
  - 학습이 time-sensitive하지 않아 모델을 키우기 쉬움(catch-up time 걱정↓)
  - Content tower inference를 여러 client가 공유 → 리소스 절약

### 2-stage, 뭐가 특별한가?
"무거운 feature extractor → freeze → 서빙"은 흔한 패턴이 맞음. 이 논문이 다른 점은 **1단계에서 무엇과 함께 어떤 objective로 학습하는가**:

1. **Co-training with a "shadow ranker"**
   - 일반적인 pretrain은 CLIP-style contrastive, masked modeling 같은 **self-supervised / 일반 objective**를 씀 → downstream task(satisfied engagement)와 어긋남.
   - 여기는 Content Tower를 **실제 ranker와 동일 구조의 base model 옆에 붙여서**, ranker의 실제 task(CTR, watch time 등)로 같이 학습.
   - 즉 embedding이 "content를 잘 표현"이 아니라 **"이 ranker가 이 surface에서 쓰기 좋게 content를 요약"** 하도록 학습됨 → Table 1의 +0.85% vs +0.01~0.14%.

2. **Base model이 'context'로 들어감**
   - Content tower 혼자 학습하면 이미 ranker의 다른 feature(ID, user history 등)가 잡는 신호까지 중복 학습.
   - Base model과 함께 학습하면 content tower는 **"다른 feature가 못 잡는 residual 정보"** 쪽으로 특화됨. (Serving ablation에서 MulCo emb가 다른 content feature와 **상호보완**이라고 나온 것과 연결)

3. **Online ranker에 못 태울 크기의 pre-trained embedding을 "projection"으로 압축**
   - Frame-level visual/audio 같은 raw embedding은 학습·서빙 비용상 online ranker에 직접 못 넣음.
   - Content tower가 그걸 받아 **ranker가 쓰기 적합한 저차원 embedding으로 projection** → 결과적으로 heavy pretrained → cheap frozen embedding의 **task-aligned distillation**.

4. **Multi-surface / multi-task를 하나의 tower에 몰아넣음**
   - 각 surface마다 별도 content tower를 학습하는 대신 공유 → 유지보수·리소스 절감 + 데이터 다양성으로 task 수 감소(Table 2).

5. **Offline이라 "ranker보다 훨씬 크게" 키울 수 있음**
   - Online ranker엔 못 넣을 point-wise attention, 큰 pre-trained embedding 등도 1단계에선 자유롭게 사용 (Table 3).

요약하면 특별한 건 "2-stage 자체"가 아니라, **1단계의 pretraining objective를 프로덕션 ranker의 실제 task와 정렬(task alignment)했다는 점 + 그걸 multi-surface로 공유 가능하게 구성했다는 점**.

## 실험 결과

### 1. Task Alignment (Table 1, Offline CTR)
| Embedding | Offline CTR |
| --- | --- |
| Raw visual + audio | +0.01% |
| Raw video query | +0.14% |
| **MulCo** | **+0.85%** |
→ Task 정렬이 오프라인 CTR에 크게 기여.

### 2. Task / Surface Selection (Table 2, Online)
| 구성 | Satisfied Eng | Freshness |
| --- | --- | --- |
| Single surface, 2 tasks | +0.07% | +0.20% |
| Single surface, 4 tasks | +0.13% | +0.24% |
| **Multi-surface, 2 tasks** | **+0.16%** | **+0.28%** |

- Task 수를 늘리면 좋지만 모든 task가 유용하진 않음. task selection 을 잘 해야함.
- **Multi-surface 데이터가 task 수를 줄여줌** → 2 task + multi-surface가 4 task single-surface보다 나음.

> **"task 수를 늘린다"의 의미 = multi-task (multi-objective)**
> - 하나의 MulCo 모델에 head를 여러 개 달고, 각 head마다 다른 label(CTR / watch time / like / completion 등)로 loss를 계산 → 가중합해서 역전파.
> - 여러 task로 동시에 학습하면 content tower가 한 task에 과적합되지 않고 더 일반적인 representation을 얻음.
> - 핵심 관찰: 다양성을 **label 축(task 수)** 으로 늘리는 것보다 **data 축(surface 수)** 으로 늘리는 게 더 효율적.

### 3. Model Scaling (Table 3)
| Scaling 방식 | MulCo Offline CTR | Ranker Offline CTR | Online Sat Eng |
| --- | --- | --- | --- |
| Larger raw content embedding | +0.8% | +0.2% | +0.34% |
| Point-wise attention | +0.5% | +0.2% | +0.34% |
| Larger output dim | +0.07% | +0.1% | +0.13% |
- Online에서 서빙하지 않으므로 크기/속도 제약이 적음 → scaling 자유도가 큼.
- 효과 크기: **raw embedding 확장 > point-wise attention > output dim 확장**.

## Serving Lessons
1. **Training-serving skew 방지**: Stage 2 export 후, 서빙 시 raw content feature 버전이 학습 때와 일치해야 함. 안 그러면 오히려 지표 하락.
2. **Feature 중요도**: Randomization/ablation 결과, MulCo embedding이 다른 content feature(text, metadata, ID 기반 등)보다 가장 중요한 content feature 중 하나로 자리잡음. 다른 content feature와 상호보완적.
3. **Staleness**: Offline 모델이라 오래되면 성능 저하 우려 → 정기 retraining 수행. 단, 실험상 시간에 따른 drift는 크지 않았음.

## 결론 및 Future Work
- MulCo는 **task-aligned co-training + 2-stage offline 구조**로 ranker의 content 이해를 강화.
- 주요 학습점:
  (1) Non-task-aligned pre-trained embedding 대비 유의미한 개선
  (2) Multi-surface 데이터가 task 수를 줄이는 데 도움
  (3) 다양한 scaling 전략으로 추가 gain 가능
  (4) 서빙 디테일(skew, staleness)이 성능에 중요
- Future: 모델 추가 scaling, 학습 효율화, strategic user segment용 data sampling, 더 많은 surface의 데이터/objective 포함.

## 개인 메모 / 포인트
- "Offline model로 분리 + online ranker와 동일 task로 co-train"이 핵심 아이디어. Pretrained → downstream task로 **frozen distillation/projection**하는 구조라 볼 수 있음.
- Multi-surface가 task 수를 대체한다는 관찰이 재미있음 → 데이터 다양성이 objective 다양성을 대체 가능.
- 2-stage에서 feature 버전 정합성이 실무적으로 가장 빠지기 쉬운 함정.
