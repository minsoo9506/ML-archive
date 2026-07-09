# Semantic IDs for Music Recommendation

- **저자**: M. Jeffrey Mei, Samuel E. Sandberg, Oliver Bembom, Andreas F. Ehmann (SiriusXM/Pandora), Florian Henkel (Spotify)
- **연도**: 2025 (RecSys '25)
- **링크**: https://doi.org/10.1145/3705328.3748139 (arXiv:2507.18800)

---

## 1. 배경 & 문제 정의

음악 스트리밍 서비스는 수천만 개의 곡을 보유하지만, 유저가 실제로 듣는 곡은 일부에 불과하다. next-item(다음 곡) 추천 모델은 보통 **아이템마다 고유 embedding을 학습**한다.

> 카탈로그 크기 N, hidden dimension h일 때 아이템 표현에만 **N × h개 파라미터** 필요
> → 카탈로그가 크면 모델이 비대해져 메모리·실시간 추론 비용 부담, 학습도 어려움

**해결 아이디어**: 아이템마다 고유 embedding 대신, **content 기반 feature를 공유하는 "Semantic ID"** 를 사용. 비슷한 곡은 같은(혹은 비슷한) ID를 공유하게 만들어 학습 파라미터 수를 줄인다.

### 기존 접근과의 차이
- **Hashing**: 공유 공간으로 매핑하지만 충돌(collision)·일반화 위험. 해싱은 pseudo-random이라 비슷한 곡이 비슷한 ID를 받지 않음.
- **Matrix factorization**: 학습 전/후 user-item 행렬 분해. 학습 후 분해는 cold-start(피드백 없는 신곡)에 일반화 불가, 학습 전 분해는 모델 표현력 제약.
- **Semantic ID (본 논문)**: content feature가 있으면 공유 semantic 공간으로 아이템 표현 → 비슷한 곡끼리 ID 공유, 신곡에도 적용 가능. [13](산업용 추천)과 유사하나 **음악 도메인**에 적용하고, **random ID와 비교**한 점이 기여.

---

## 2. 모델 & 데이터셋

### 2.1 두 데이터셋

| | **Spotify** | **Pandora** |
|---|---|---|
| 출처 | 오픈소스 Sequential Skip Prediction [1] | 자체 데이터 (radio stations) |
| 단위(granularity) | Session | User |
| 피드백 종류 | play(positive) / skip(negative), **암묵적** | thumb-up / thumb-down, **명시적** |
| max seq length | 20 | 400 |
| max lookback | Same day | 1 year |
| # sequences | 3×10⁶ | 10⁷ |
| # tracks | 3×10⁶ | 10⁶ |

- **baseline 모델**: SASRec [4] 기반 transformer.
- **Pandora**: track embedding = song + artist + genre embedding 합으로 분해(decomposition) 가능.
- **Spotify**: artist/genre 없어 track embedding = song embedding 그 자체.
- baseline = codebook size가 곡 수와 같은 **"1차원" semantic ID** (= 각 곡 고유 ID)와 동치.

### 2.2 Semantic ID 구성

핵심: 추천 아이템을 **n-tuple of codewords**로 표현. 각 codeword는 크기 k짜리 독립 codebook n개에서 선택.

- 본 논문: **n = 4**, k는 변화시킴. 예) k = 64면 64⁴ ≈ 16M개 unique 아이템 표현 가능.
- **RQ-VAE**(residual VQ-VAE) [17]로 semantic ID 생성. (cf. [11])
- tie-breaking용 차원 1개 추가 → 총 n + 1차원. 이 마지막 ID는 학습 X, 앞쪽 4개(학습된) semantic ID가 같은 두 곡이 충돌할 때 증가시킴.

**두 버전 비교**:
- **v0**: 곡마다 random하게 ID 부여 (random hashing 유사, 아이템 정보 무시)
- **v1**: content feature 기반으로 **학습된** semantic ID (비슷한 곡이 비슷한 codeword 공유)

content feature:
- **Spotify**: 8차원 audio vector + track attributes(speechiness, danceability, energy 등). popularity는 content feature가 아니라 제외.
- **Pandora**: 자체 audio embedding + metadata(genre, release year) embedding [6,9].

---

## 3. 평가 방법

- ranking accuracy: positive/negative 피드백 쌍을 올바르게 순위 매기는 정확도.
- test set은 학습 기간 이후 **2주**로 시간 분리. test 기간에 positive 1개 + negative 1개 이상 있는 user/session만 포함.
- 이 test accuracy(전체 평균) = **stratified AUC**와 동등.

---

## 4. 결과

### 4.1 Offline (Figure 1, 2)

- **Spotify**: codebook size **k=4096**이면 baseline과 동등, **k=8192**면 baseline 능가 (v0, v1 모두).
- **Pandora (song-only)**: v1 semantic ID가 **k=32768**에서 baseline 동등.
- **파라미터 절감**: Pandora ~75%, Spotify 99% 감소 → 절감한 파라미터를 hidden dimension 등 모델 복잡도 증가에 재할당 가능.
- **v1 > v0**: song-only 케이스(Fig 1a,b)에서 학습된 ID가 random ID보다 정확도 높음.
- **단, song decomposition 시 v0 ≈ v1** (Fig 1c): artist/genre embedding이 이미 비슷한 곡을 공유시키는 역할을 해서, semantic ID가 학습됐든 random이든 차이 없음. (단, genre/artist embedding만 쓴 모델은 정확도 ~10% 낮음 → semantic ID 자체는 여전히 기여)

### 4.2 모델 복잡도 trade-off (Figure 2)

- 학습 파라미터를 늘리면(codebook k ↑ 또는 hidden dim h ↑) 정확도 증가.
- **Spotify**: h든 k든 둘 다 효과 비슷.
- **Pandora**: 같은 파라미터 수라면 **h보다 k(codebook size)를 키우는 게 유리**. (피드백 타입 차이 등이 원인으로 추정)

### 4.3 User input length (Figure 3) — 핵심 발견

> **저-피드백(low-feedback) 유저에서 semantic ID의 lift가 가장 크다.**

- Pandora: feedback이 적은 유저에서 가장 큰 향상 (일부 고-피드백 유저도 향상).
- Spotify: 모든 세션 길이에서 향상, 짧은 세션에서 더 큼.
- v1 > v0 차이도 song-only 케이스에서 두드러짐.

### 4.4 Online A/B Test (Table 2)

Pandora 리스너 1천만 명, 30일, control(baseline) vs test(semantic, k=16384, h=120). semantic 모델은 **파라미터 ~50% 절감, 메모리·학습비용 ~20% 감소**.

| Metric | 변화 | p-value |
|---|---|---|
| Listening hours | -0.08% | 0.53 (n.s.) |
| Song completion rate | -0.04% | 0.22 (n.s.) |
| **New releases (<120일) played** | **+0.81%** | ≪10⁻⁴ |
| **Distinct songs per seed** | **+1.82%** | ≪10⁻⁴ |
| **Distinct artists per seed** | **+0.51%** | ≪10⁻⁴ |
| **Track repetition** | **-1.26%** | ≪10⁻⁴ |

→ **핵심 비즈니스 지표(청취 시간 등)는 중립**이면서, **추천 다양성(diversity)·신곡 노출은 유의하게 증가**, 반복 재생은 감소. 파라미터를 절반으로 줄이고도 같은 성능 + 더 높은 다양성 달성.

- 효과는 유저 세그먼트마다 비균일 (Fig 4): 저-활동 유저는 song completion rate 향상. 저-활동 유저가 고-활동 유저로 전환되면 장기적 이득 가능성.

---

## 5. 결론

Semantic ID는 **정확도 손실 없이 content feature를 공유하고 모델 파라미터를 줄이는** 실용적 방법이다.
- 절감한 파라미터로 더 복잡한 모델 구성 가능 → 정확도·다양성 향상 + 학습/추론 비용 절감.
- 정확도 lift는 **저-피드백 유저에서 가장 큼**.
- 효과는 유저 세그먼트마다 비균일.
- **향후 연구**: cold-start(신곡)를 위한 **interpolated semantic ID** 탐구.

---

## 6. 메모 (배울 점)

- **Semantic ID 핵심 공식**: 아이템 = n개 codebook(각 크기 k)에서 뽑은 n-tuple codeword. k⁻ⁿ개 조합으로 거대 카탈로그를 적은 파라미터로 표현. n=4, k 조절.
- **RQ-VAE**로 content feature → discrete codeword 생성. (residual quantization으로 계층적 표현)
- **v0(random) vs v1(trained) 비교**가 깔끔한 ablation: semantic ID의 이득이 "content 정보" 덕인지 "파라미터 공유/정규화" 덕인지 분리. song-only면 trained가 우세, decomposition(artist/genre 이미 존재)이면 차이 없음 → **이미 공유 feature가 있으면 추가 학습 ID의 한계 효용이 작다.**
- **codebook size k vs hidden dim h**: 같은 파라미터 예산에서 어디에 투자할지가 도메인마다 다름 (Pandora는 k 선호).
- **비즈니스 임팩트 프레이밍**: 메인 KPI는 중립으로 지키면서 "파라미터 -50%, 비용 -20%, 다양성 +" 라는 효율/다양성 스토리로 가치 증명. neutral KPI + 비용절감도 충분히 성과.
- **저-피드백/cold 유저에서 가장 큰 효과** → content 기반 표현이 data-sparse 영역을 보강한다는 직관과 일치.

### 관련 논문
- SASRec [4] Kang & McAuley, 2018 (backbone)
- Semantic ID for industrial reco [13] Singh et al., RecSys '24
- RQ-VAE / SoundStream [17] Zeghidour et al., 2021
- Generative retrieval semantic ID [11] Rajput et al., NeurIPS 2023
- Negative Feedback for Music Reco [8] Mei et al., UMAP '24

---

## Appendix. Semantic ID 자세히 (예시로 이해하기)

### A.1 왜 필요한가 (문제부터)

보통 추천 모델은 **곡 하나당 embedding 벡터 하나**를 학습한다.

```
곡 A → [0.1, 0.5, -0.3, ...] (h차원)
곡 B → [0.7, -0.2, 0.4, ...]
곡 C → ...
```

곡이 100만 개, embedding이 128차원이면 → **100만 × 128 = 1.28억 개 파라미터**가 "곡 표현"에만 들어간다. 카탈로그가 클수록 모델이 비대해지고, 신곡(피드백 없는 곡)은 embedding을 학습할 수 없다 (**cold-start 문제**).

### A.2 핵심 아이디어: 곡을 "코드 조합"으로 표현

곡마다 고유 벡터를 주는 대신, **여러 개의 작은 사전(codebook)에서 코드 번호를 하나씩 뽑아 조합**해 곡을 표현한다. 우편번호나 자동차 번호판처럼.

논문 설정: codebook **n=4개**, 각 codebook 크기 **k**.

```
codebook 1 (크기 k):  코드 0 ~ k-1 중 하나
codebook 2 (크기 k):  코드 0 ~ k-1 중 하나
codebook 3 (크기 k):  코드 0 ~ k-1 중 하나
codebook 4 (크기 k):  코드 0 ~ k-1 중 하나
```

곡 하나 = 코드 4개짜리 튜플:

```
곡 A = (12, 5, 200, 47)
곡 B = (12, 5, 200, 89)
곡 C = (700, 3, 11, 2)
```

**k=64이면 64⁴ ≈ 1,600만 개**의 조합 → 작은 사전 4개(64×4 = 256개 코드)만으로 1,600만 곡 구분 가능. 이게 파라미터 절감의 원리.

학습되는 건 **각 코드의 embedding**이다:

```
codebook1의 코드 12 → embedding 벡터
codebook2의 코드 5  → embedding 벡터
...
곡 A의 표현 = (코드12 emb) + (코드5 emb) + (코드200 emb) + (코드47 emb)
```

저장할 embedding이 "100만 곡"이 아니라 "256개 코드"로 줄어든다.

### A.3 핵심: 비슷한 곡은 코드를 "공유"한다

곡 A=(12,5,200,47), 곡 B=(12,5,200,89)는 앞 3개 코드가 같다 → **A와 B가 비슷한 곡**(예: 같은 아티스트의 비슷한 분위기). 마지막 코드만 달라 둘을 구분.

이득 두 가지:
- **비슷한 곡이 embedding 공유** → 한 곡 학습이 비슷한 곡에도 전파됨
- **신곡도 표현 가능** → 신곡의 오디오/메타데이터로 코드만 뽑으면 됨. 학습된 코드 embedding 재활용 → cold-start에 강함

> 비유: 곡마다 새 단어를 외우는 대신, **자모(ㄱ,ㄴ,ㅏ,ㅓ) 조합으로 글자를 만드는** 것. 자모 몇 개만 배우면 무한히 많은 글자를 표현하고 처음 보는 글자도 읽을 수 있다.

### A.4 코드는 어떻게 정하나? → RQ-VAE

곡의 **content feature**(Spotify: 8차원 audio vector + danceability, energy, speechiness 등)를 입력받아 **RQ-VAE(Residual Quantized VAE)**가 코드 4개를 출력한다.

"Residual(잔차)"가 핵심 — 계층적으로 뽑는다:

```
1단계: 오디오 벡터로 codebook1에서 가장 가까운 코드 선택 (큰 그림: 예 "잔잔한 곡")
       → 표현 못 한 나머지(잔차) 계산
2단계: 그 잔차를 codebook2에서 가장 가까운 코드로 (좀 더 세밀: 예 "어쿠스틱")
3단계: 또 남은 잔차를 codebook3으로 (더 세밀)
4단계: codebook4로 (가장 세밀)
```

→ **앞 코드일수록 coarse(큰) 특징, 뒤 코드일수록 미세한 특징**. 비슷한 곡이 앞 코드부터 공유되는 이유.

#### RQ-VAE 학습: 입출력과 업데이트되는 파라미터

RQ-VAE는 이름대로 **autoencoder** — 학습 시엔 "입력을 코드로 압축했다 다시 복원"한다.

```
content feature x  →  [Encoder]  →  z (잠재벡터)
                                       │
                          [Residual Quantizer] (codebook 4개)
                                       │
                                    z_q (양자화 벡터) + 코드 (12,5,200,47)
                                       │
                              [Decoder]  →  x̂ (복원된 feature)
```

**Input / Output**
- **학습 시 입력**: 곡의 content feature `x` (Spotify면 8차원 audio vector + danceability/energy/speechiness 등을 이어붙인 벡터)
- **학습 시 복원 타깃**: 같은 `x` 를 복원한 `x̂`. 즉 입력=타깃인 **self-supervised** (라벨 불필요)
- **추론(실제 사용) 시 출력**: `x̂`가 아니라 중간에 나온 **코드 튜플 (12,5,200,47)** = 그 곡의 semantic ID

**Residual Quantizer 동작** (codebook 4개 = $C^1 \sim C^4$, 각 k개 벡터)

```
r_0 = z                                  # 첫 잔차 = 인코더 출력
코드1 = argmin_j || r_0 - C¹[j] ||       # codebook1에서 가장 가까운 코드
r_1 = r_0 - C¹[코드1]                     # 남은 잔차
코드2 = argmin_j || r_1 - C²[j] || ;  r_2 = r_1 - C²[코드2]
코드3 = argmin_j || r_2 - C³[j] || ;  r_3 = r_2 - C³[코드3]
코드4 = argmin_j || r_3 - C⁴[j] ||
z_q = C¹[코드1] + C²[코드2] + C³[코드3] + C⁴[코드4]   # 최종 양자화 벡터
```

**업데이트되는 파라미터 3종 (동시 학습)**

| 파라미터 | 무엇 | 어떤 loss로 |
|---|---|---|
| **Encoder 가중치** | x → z 매핑 NN | reconstruction + commitment |
| **Decoder 가중치** | z_q → x̂ 매핑 NN | reconstruction |
| **Codebook 벡터** C¹~C⁴ | 각 코드의 embedding (사전 내용) | codebook loss (또는 EMA) |

**Loss 구성** (VQ-VAE 계열 표준)

$$L = \underbrace{\|x - \hat{x}\|^2}_{\text{① reconstruction}} + \underbrace{\sum_i \|\text{sg}[r_{i-1}] - C^i[\text{코드}_i]\|^2}_{\text{② codebook}} + \beta\underbrace{\sum_i \|r_{i-1} - \text{sg}[C^i[\text{코드}_i]]\|^2}_{\text{③ commitment}}$$

- **① reconstruction**: 복원 정확도 → encoder + decoder 학습
- **② codebook loss**: 선택된 코드 벡터를 잔차 쪽으로 끌어당김 → codebook 학습
- **③ commitment loss**: encoder 출력이 코드에서 멀어지지 않게 → encoder 학습 (β는 가중치)
- `sg[·]` = **stop-gradient**. 같은 항으로 양쪽이 서로 끌어당기면 불안정하니, ②는 codebook만 / ③은 encoder만 업데이트하도록 gradient를 한쪽씩 끊음
- **argmin은 미분 불가** → **straight-through estimator**로 우회: forward는 `z_q` 사용, backward는 `z_q`의 gradient를 그대로 `z`에 복사해 encoder까지 전파
- (codebook을 loss 대신 **EMA 업데이트**하는 변형도 흔함 — 선택된 코드에 할당된 `z`들의 이동평균으로 갱신, 더 안정적)

#### 2단계 파이프라인

RQ-VAE 학습은 **추천 모델과 별개로 먼저(offline) 진행**한다.
1. RQ-VAE 학습 → 모든 곡을 통과시켜 semantic ID(코드 튜플)를 뽑아둠
2. 그 semantic ID를 **SASRec 추천 모델의 입력**으로 사용 (곡 embedding = 코드 embedding 합)

### A.5 v0 vs v1 (ablation 핵심)

| | 코드 부여 방식 | 의미 |
|---|---|---|
| **v0** | 곡마다 **랜덤** 코드 배정 | content 정보 무시 — 단순 "압축" |
| **v1** | RQ-VAE로 content 기반 **학습된** 코드 | 비슷한 곡이 비슷한 코드 공유 |

v1 > v0 이면 → "코드 공유에 담긴 **content 의미**가 도움"이라는 증거 (song-only에서 실제로 v1 > v0).
반대로 song+artist+genre 분해 시 v0≈v1 → **artist/genre embedding이 이미 비슷한 곡을 묶어** semantic ID의 추가 의미 정보가 적었기 때문.

### A.6 tie-breaking 차원 (n+1번째)

앞 4개 코드가 우연히 완전히 같은 두 곡(충돌)이 생길 수 있다. 5번째 ID를 0,1,2... 증가시켜 구분 (학습 안 하는 단순 번호표).

```
곡 A = (12, 5, 200, 47, 0)
곡 D = (12, 5, 200, 47, 1)  ← 앞 4개 충돌, 마지막만 다르게
```

### 한 줄 정리

**"곡 100만 개 = embedding 100만 개" → "코드 256개의 조합으로 곡 100만 개 표현"** 으로 바꾼 게 semantic ID, 그 코드를 content 기반으로 똑똑하게 뽑는 게 RQ-VAE.
