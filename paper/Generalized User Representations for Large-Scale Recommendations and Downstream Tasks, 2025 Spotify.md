# Generalized User Representations for Large-Scale Recommendations and Downstream Tasks (Spotify, 2025)

- **저자**: Ghazal Fazelnia 외 (Spotify)
- **발표**: RecSys '25 (ACM, 2025)
- **DOI**: 10.1145/3705328.3748132

## 한 줄 요약

Spotify가 프로덕션에 배포한 **2-stage 프레임워크**로, autoencoder 기반 representation learning과 transfer learning을 결합해 다양한 추천 다운스트림 태스크에 재사용 가능한 **일반화된 사용자 임베딩**을 생성한다.

## 문제 정의

음악 스트리밍 추천 시스템의 특수한 어려움:
- 트랙 길이가 짧고, 한 세션에 다수 트랙이 피드백 없이 재생됨
- 사용자 욕구가 충돌적 (favorite 재방문 vs 신규 발견)
- 신규 유저(cold-start) 대응이 어려움
- 태스크별로 별도 모델을 만들면 시스템 복잡도 ↑, 사용자 경험 분절 (e.g., retrieval from long-term taste,bandits for new users)

**3가지 연구 질문 (RQ)**
- **RQ1**: 핵심 사용자 관심을 담으면서 다양한 다운스트림 태스크에 적응 가능한 모델을 어떻게 설계할 것인가?
- **RQ2**: cold-start 유저에 작동하면서 활동량이 늘면 더 좋아지는 모델은?
- **RQ3**: 임베딩 공간의 효용을 어떻게 평가할 것인가?

## 접근 방식 (2-stage)

### 모델 아키텍처 (Figure 1)

```
                              ┌─ time horizons ─┐
 Raw audio ───►  Audio         ┌─ 6 mo. ─┐
                 Modality ───► │  1 mo.  │ ──┐
                 Encoder       └─ 1 week ┘   │
                                             │
 Collab.    ───► Collab.        ┌─ 6 mo. ─┐  │
 features        Modality ───►  │  1 mo.  │ ─┤──► concat = x_u
 (playlist co-   Encoder        └─ 1 week ┘  │           │
  occurrence)                                │           │
                                             │           ▼
 Demographics ───────────────────────────────┤   ┌──────────────┐       ┌──► Ranking
 Context ────────────────────────────────────┤   │   Encoder    │       │
 New User Onboarding Signals ────────────────┘   │      │       │       ├──► Search
                                                 │      ▼       │       │
                                                 │     z_u ─────┼───────┼──► Music Rec.
                                                 │ (User repr.) │       │
                                                 │      │       │       └──► Discovery
                                                 │      ▼       │
                                                 │   Decoder    │
                                                 │      │       │
                                                 │      ▼       │
                                                 │     x̂_u      │
                                                 └──────────────┘
                                                 Denoising Auto-Encoder
```

그림에서 확인되는 핵심 구성:

- **두 가지 modality encoder** (사전학습된 상태로 활용)
  - **Audio Modality Encoder**: raw audio feature → audio embedding
  - **Collaborative Modality Encoder**: playlist co-occurrence 기반 → tracks & artists embedding
- **시간 horizon별 집계**: 각 modality 임베딩을 **6개월 / 1개월 / 1주** 단위로 집계 후 concatenate → 단·중·장기 취향을 동시에 반영
- **추가 입력 피처**: Demographics, Context, New User Onboarding Signals
- 위를 모두 합친 입력 벡터 **x_u** → **Denoising Auto-Encoder** (Encoder → 잠재 **z_u** → Decoder → 복원 x̂_u)
- 학습 후 **z_u** (User representation embedding) 를 Ranking / Search / Music Rec. / Discovery 등 다운스트림 태스크에 전이

> 단순 AE가 아닌 **Denoising AE** 라는 점, 그리고 multi-horizon 집계로 "장기 취향 vs 최근 관심"을 동시에 임베딩에 녹인 점이 그림의 포인트.

### Modality Encoder 상세

User repr 모델 입력으로 들어가기 전에, **트랙(아이템)을 80-dim 벡터로 임베딩**해주는 사전학습 모듈. 두 가지 갈래로 나뉘고, 각각 다른 신호를 인코딩한다.

| Encoder | 입력 | 어떤 신호를 인코딩? | 비유 |
| --- | --- | --- | --- |
| **Audio Modality Encoder** | Raw audio feature | **콘텐츠**: 음향적 특성 (장르·분위기·악기 등) | 트랙 자체의 "소리" |
| **Collab Modality Encoder** | Playlist co-occurrence 통계 | **행동**: 어떤 트랙들과 같이 소비되는가 | 트랙의 "사회적 맥락" |

핵심 특징:
- **사전학습 (pre-trained)** 되어 있음. user repr 모델 학습할 때는 frozen feature extractor로만 사용.
- 두 인코더 모두 트랙 단위로 **80-dim 임베딩**을 출력 (논문 [1, 4] 인용).
- **콘텐츠 (audio) + 협업 신호 (collab)** 두 갈래를 모두 활용 → 신곡(콘텐츠는 있는데 행동 데이터 없음)도, 인기곡(행동 데이터 풍부)도 모두 잘 표현.

그럼 유저 입력은 어떻게 만들어지나:

```
유저 u의 청취 이력 = [track_t1, track_t2, ..., track_tN]
                                │
            ┌───────────────────┴───────────────────┐
            ▼                                       ▼
   Audio Modality Encoder              Collab Modality Encoder
   (트랙별 80-dim audio emb.)          (트랙별 80-dim collab emb.)
            │                                       │
            ▼ 시간 윈도우 + mean pooling             ▼
   audio_6mo / 1mo / 1wk                  collab_6mo / 1mo / 1wk
            └───────────────────┬───────────────────┘
                                ▼ concat + 정적 feature
                              x_u  →  AE 학습
```

왜 두 개를 모두 쓰는지 (ablation 결과):
- Modality encoder 임베딩을 빼면 **AUC −4.2%, favorite-artist 클러스터 nDCG@50 −37.1%** 로 가장 큰 성능 하락.
- 즉 user repr의 표현력 대부분이 이 사전학습된 트랙 임베딩에서 옴.
- Audio만 쓰면 인기/맥락 신호 부족, Collab만 쓰면 신곡(cold-track) 약함 → **두 modality 보완적**.

> 정리하면: Modality encoder는 **트랙 → 80-dim 임베딩** 으로 변환하는 사전학습된 lookup 같은 역할. AE는 그 위에서 "유저별로 어떤 트랙들을, 언제, 얼마나 들었나"를 압축할 뿐, 트랙 자체의 표현은 이 모듈에 위임한 구조.

### 전체 파이프라인 한 눈에

```
[사전학습 단계]
  Modality encoder → 트랙별 80-dim 임베딩 (audio 1개, collab 1개)

[AE 입력 만들기 — 유저별]
  유저 청취 이력
    → 트랙 임베딩으로 변환
    → 시간 윈도우(6mo/1mo/1wk)로 자르기
    → 윈도우별 mean pooling
    → concat (+ demographics, context, onboarding)
    → x_u

[AE 학습 — 1 row = 1 user]
  x_u 입력 → 인코더 → z_u (압축) → 디코더 → x̂_u (복원, denoising)

[서빙]
  유저 이벤트 발생 → 위 과정 다시 돌려 새 x_u → 인코더 통과 → 최신 z_u
  → feature store 갱신 → 다운스트림이 읽어 사용
```

역할 분담이 깔끔:

- **Modality encoder** = "트랙이 무엇인가" 표현 (사전학습, frozen)
- **시간 윈도우 + mean pooling** = "유저가 무엇을, 언제 들었나" 통계화
- **AE** = 그 통계 벡터를 다운스트림에 잘 전이되는 **잠재 공간**으로 압축

→ User repr 모델 자체는 단순한 MLP-AE. 표현력은 **사전학습된 modality encoder + 멀티 윈도우 통계 설계**에서 나오고, AE는 차원 축소 + 정규화 역할을 담당.

### Stage 1: Autoencoder 기반 표현 학습
- **Modality encoder**가 트랙 feature를 80차원 벡터로 사전 처리 (co-occurrence 통계 + 오디오 feature 기반)
- 트랙 임베딩 + **사용자 데이터**(인구통계, onboarding 신호, contextual 정보) 를 입력으로 받아 denoising autoencoder 학습
- 결과 임베딩(z_u)을 다양한 추천 태스크의 input feature 로 재사용

### Stage 2: Transfer Learning
학습된 사용자 임베딩을 retrieval / ranking / generation 등 large-scale 태스크에 전이.

핵심 시스템 요소 4가지:

1. **Responsiveness (반응성)**
   - **Near-Real-Time (NRT) inference**: 사용자 활동 이벤트로 트리거되어 분 단위로 온라인 feature store에 저장
   - NRT + batch inference 결합으로 활성/비활성 유저 모두 커버

2. **Cold-Start Awareness**
   - Onboarding 시 선택한 선호 아티스트/언어를 기존 유저와 동일한 모델에 통과
   - 인구통계 정보와 결합 후 데이터가 쌓이면 점진적으로 행동 기반 입력으로 전환

3. **Stability**
   - 주기적 재학습이 필요하나 비동기 업데이트는 다운스트림에 혼란 유발
   - **Batch Management**: 각 업데이트에 unique batch ID 부여, 다운스트림 모델은 같은 batch 안에서만 비교/재학습. 업데이트 중에는 "legacy" batch로 서비스 지속
   - 👉 본질은 **유저 임베딩 버전 ↔ 다운스트림 모델 버전을 맞추는 것**. user repr을 재학습하면 임베딩 좌표계가 바뀌므로, 그걸 입력으로 쓰는 Ranking/Search/Home 등 다운스트림도 **같은 batch로 같이 재학습 → 한 세트로 배포**해야 함. 버전을 절대 섞지 않는다는 룰. ("batch"라는 단어가 헷갈리지만 실제로는 그냥 **버전 페어링 + 무중단 스왑** 운영 전략)

## 실험 결과

### 7일 listening prediction (established users)

본 모델은 NMF, PMF, LightFM, DLRM, VAE-CF, average embeddings 등 모든 베이스라인을 상회.

| Comparison vs | Accuracy | AUC |
| --- | --- | --- |
| NMF | +15.2% | +18.6% |
| PMF | +10.1% | +12.7% |
| LightFM | +2.3% | +3.9% |
| DLRM | +1.5% | +2.8% |
| VAE-CF | +4.0% | +5.7% |
| Average embeddings | +1.8% | +1.6% |

→ **RQ1** 검증: 딥러닝/AE 베이스라인까지 모두 능가.

### 4시간 cold-start prediction

| Comparison vs | Onboarding | Accuracy | AUC |
| --- | --- | --- | --- |
| Popularity heuristic | Completed | +26.2% | +27.0% |
| Popularity heuristic | Not Completed | +24.6% | +24.7% |
| Average embeddings | Completed | +5.0% | +5.1% |

→ **RQ2** 검증: onboarding 정보가 불완전해도 초기 의도를 잘 포착.

### Clustering 기반 임베딩 품질 (nDCG@50, vs average embeddings)

| Cluster heuristic | nDCG@50 |
| --- | --- |
| Same favorite artists | +2.9% |
| Same country of most listened artists | +5.5% |
| Same new user onboarding | +26.2% |

→ **RQ3** 검증: 행동적으로 의미 있는 그룹화.

### 프로덕션 다운스트림 적용 (온라인 A/B)

| Downstream Model | Metric 1 | Metric 2 |
| --- | --- | --- |
| Candidate Generation | Discoveries +2.9% | Shelf-level i2s +13% |
| Search | Overall search success +0.06% | Podcast success +0.76% |
| Home Ranking | Music discovery success +0.20% | Consumption share +0.05% |

- **Candidate generation**: 홈페이지 앨범/플레이리스트 후보 생성, 앨범 발견과 impression-to-stream 비율 큰 폭 상승
- **Search**: 검색 결과 re-rank, 특히 팟캐스트 검색 성공률 향상 (cross-modal 효과)
- **Home Ranking**: 친숙한 콘텐츠 → 새로운 발견으로 사용자 참여 이동
- **Artist preference modeling**: 인프라 + feature 비용 **50% 감소**, top-line 지표는 유지

## Ablation Study

| 제거한 입력 | 영향 |
| --- | --- |
| 신규 유저 onboarding 신호 | 같은 onboarding 클러스터 nDCG@50 **−13.8%** |
| Modality encoder embeddings | 7일 listening AUC **−4.2%**, favorite-artist 클러스터 nDCG@50 **−37.1%** |
| User static feature (등록 국가 등) | 국가 기반 클러스터 nDCG@50 **−12.1%** |

→ Modality encoder 임베딩이 시스템 성능의 가장 핵심적인 요소.

## 결론 및 향후 과제

- Autoencoder + Transfer learning 결합으로 cold-start와 established 유저 모두에 강한 일반화된 사용자 표현을 학습
- Spotify 프로덕션에 배포되어 personalized search, candidate generation, ranking 등에 활용 중
- 인프라 복잡도 감소 효과
- **Future work**: 가사·플레이리스트·메타데이터 등 추가 모달리티 통합, **LLM 임베딩 활용**, 팟캐스트/뉴스/커머스 도메인 확장

## 인사이트 / 메모

- "한 모델로 여러 태스크" 형태의 representation 재사용은 시스템 비용 절감(50%) + 모델 단순화 효과가 큼
- Cold-start와 established를 같은 모델로 처리하기 위해 onboarding 신호 → 행동 신호로 점진 전환하는 설계가 핵심
- Stability를 위한 **Batch Management(legacy/new batch 분리)** 는 production 추천 시스템에서 자주 놓치기 쉬운 운영 요소. 임베딩 재학습이 다운스트림을 깨뜨리지 않도록 batch ID로 동기화하는 패턴은 참고할 만함
- NRT inference로 이벤트 기반 임베딩 갱신을 한다는 점도 일반적인 daily/hourly batch와 차별화되는 포인트
