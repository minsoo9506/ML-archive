# The Future is Sparse: Embedding Compression for Scalable Retrieval in Recommender Systems

- **저자**: Petr Kasalický, Martin Spišák, Vojtěch Vančura, Daniel Bohuněk, Rodrigo Alves, Pavel Kordík (Recombee / Czech Technical University in Prague / Charles University)
- **연도**: 2025 (Under review, arXiv:2505.11388)
- **링크**: https://arxiv.org/abs/2505.11388
- **코드**: https://github.com/recombee/CompresSAE

---

## 1. 배경 & 문제 정의

Recombee(추천-as-a-service 플랫폼)는 O(10⁸) 규모 카탈로그를 다루며, 텍스트/이미지/영상/행동 등 다양한 modality의 **dense embedding**으로 아이템·유저를 표현한다. Embedding 품질이 높을수록(차원이 클수록) 추천 성능은 좋아지지만, 그만큼 메모리·연산·지연시간 비용이 커진다.

> 실제 사례: production SBERT(512차원)를 Nomic embedding(768차원)으로 교체했더니 CTR **+4.86%** 상승. 하지만 아이템 1억 개 기준 embedding table이 **204.8GB → 307.2GB**로 증가.

Cold-start 아이템(상호작용 없는 롱테일)이 많을수록 content 기반 embedding 의존도가 커지고, 이는 메모리 문제를 더 악화시킨다.

### 기존 압축 기법과의 비교
| 기법 | 장점 | 단점 |
|---|---|---|
| **Quantization** (int4 등) | 압축률-정확도 balance 좋음 | 하드웨어 지원 필요, quantization-aware retraining 필요 |
| **SVD/PCA** | 효율적 | 정확도 손실 큼 |
| **Matryoshka Representation Learning** | truncation만으로 다양한 차원 지원 | **backbone encoder를 재학습**해야 함 (비용 큼) |
| **Offloading/caching** (Meta 등) | 100TB급 테이블도 처리 | 복잡한 엔지니어링, 플랫폼 종속 |
| **Sparse Autoencoder (SAE, 본 논문)** | encoder 재학습 불필요, 사후(post-hoc) 압축 | — |

**핵심 아이디어**: 이미 학습된 encoder의 출력(dense embedding)은 그대로 두고, **그 위에 SAE를 얹어** dense embedding을 **고차원이지만 희소(sparse)한 벡터**로 변환한다. 차원은 늘리되 대부분 0으로 만들어(k개만 non-zero) 저장/연산 비용을 줄이는 접근 — Matryoshka(차원 축소)와 반대되는 철학.

---

## 2. 방법: CompresSAE

### 2.1 구조

Pretrained encoder가 만든 d차원 dense embedding들의 corpus `E ∈ R^(N×d)` 위에서 SAE를 학습 (encoder 자체는 건드리지 않음 → 계산 비용 저렴, 유연함).

- **Encoder** `f_enc`: `s = φ(W_enc·x̄ + b_enc, k)`, `x̄ = x/‖x‖₂` (입력을 cosine 유사도에 맞춰 정규화)
- **Decoder** `f_dec` (linear, bias 없음): `x̂ = W_dec·s`
- `φ(·, k)`: **절댓값 기준 상위 k개만 남기고 나머지는 0**으로 만드는 함수. 활성화 함수이자 sparsification 메커니즘 역할을 겸함 (ReLU/TopK 대신). 음수도 보존한다는 점이 특징 — 원 벡터의 "방향"을 살리는 데 유리.
- Decoder weight `W_dec`는 **row-normalize** → 출력 스케일 일정하게 유지.

### 2.2 기존 SAE(해석가능성 지향)와의 차이

- 기존 연구([Gao et al. 2024], [Wen et al. 2025])는 LLM 해석가능성을 위해 설계되어 **ℓ2 reconstruction loss** `‖x - x̂‖²` 를 사용.
- 본 논문은 **retrieval(검색)** 이 목적이므로 **cosine distance**를 직접 최소화하는 loss를 사용:

$$L_{cosine}(x, \hat x) = 1 - \frac{x^\top \hat x}{\|x\|_2 \|\hat x\|_2}$$

**왜 L2 대신 cosine인가.** L2 loss는 벡터의 **크기(길이)와 방향을 둘 다** 맞춰야 줄어드는데, retrieval은 결국 cosine similarity로 아이템을 랭킹하므로 **방향만** 맞으면 충분하다 (아래 "왜 방향만 중요한가" 참고). 예를 들어 `x=[4,0]`, `x̂=[1,0]`은 방향이 완전히 같지만 L2 loss는 `(4-1)²=9`로 "많이 틀렸다"고 벌점을 주는 반면, cosine loss는 `1-cos(0°)=0`으로 정확히 "완벽하다"고 판단한다. L2로 학습하면 모델이 검색에 안 쓰이는 "길이" 정보를 복원하는 데 32개뿐인 nonzero 슬롯의 capacity를 낭비하게 되므로, retrieval에는 cosine loss가 태스크에 더 직접적으로 맞는 목적함수다.

**dead neuron 방지 트릭.** `φ(·, k)`는 절댓값 top-k만 남기고 나머지를 강제로 0으로 만드는 함수다. 그런데 특정 latent 차원(예: 4096개 중 하나)이 어떤 입력에서도 top-32 안에 한 번도 못 들면, 그 차원은 항상 0으로 잘려 decoder에 기여하지 못하고 **gradient도 전혀 받지 못해 영원히 죽은 채로 남는다** (dead neuron). 이를 막기 위해 최종 loss를 `f(x;θ,k)`(top-32 복원) 하나가 아니라, 더 관대한 `f(x;θ,4k)`(top-128 복원)의 loss까지 더해서 사용한다:

$$L = L_{cosine}(x, f(x;\theta,k)) + L_{cosine}(x, f(x;\theta,4k))$$

top-32에는 못 들었어도 top-128에는 드는 뉴런이라면 `4k` 항에서 gradient를 받을 기회가 생겨, 배포 시점(k=32)엔 안 쓰이더라도 학습 중엔 완전히 죽지 않도록 안전장치를 두는 것 (Gao et al. 2024의 기법을 응용).

**입력 정규화 방식의 단순화.** 기존 방식(Gao et al., Wen et al.)은 데이터셋 전체의 **차원별 평균/표준편차**로 입력을 표준화(standardize)한 뒤 인코딩하고, decoder 출력 후 그 통계값으로 다시 **rescale**해서 원래 스케일을 복원한다. 본 논문은 이런 데이터셋 통계 없이, 벡터 각각을 자신의 L2 norm으로 나눠 단위 벡터로 만들 뿐이다 (`x̄ = x/‖x‖₂`). Cosine similarity 자체가 벡터 길이에 무관하므로, 애초에 길이를 1로 고정하고 방향만 학습시키면 충분하다 — 통계량을 저장·적용하는 부가 로직이 필요 없는 더 단순한 설계.

**참고: 왜 retrieval에서는 방향만 중요한가.** 임베딩 벡터의 길이(norm)는 흔히 텍스트 길이·인기도·인코더의 우연한 활성화 크기 등 의미와 무관한 요인에 영향을 받는다. 예를 들어 짧은 설명글의 아이템 A=[0.9,0.1]과 긴 설명글의 아이템 B=[1.8,0.2]는 방향이 거의 같아 의미상 비슷한 아이템이지만, 단순 L2 거리로 쿼리와 비교하면 벡터가 더 긴 B가 훨씬 안 비슷하다고 잘못 판단된다. 이런 이유로 실무 dense retrieval(문장/아이템 임베딩 검색)은 거의 항상 cosine similarity(또는 정규화 후 dot product)를 써서 "의미와 무관한 길이 차이"를 지우고 방향만으로 비교한다. 본 논문의 압축 embedding도 결국 이 cosine 기반 검색에 쓰이므로, "방향을 얼마나 잘 보존하는가"가 곧 실질적 검색 품질이 된다.

### 2.3 학습

- Encoder/원본 데이터 접근 불필요 — **미리 계산된 embedding batch만으로 학습** 가능.
- Batch size 100,000, Adam optimizer, H100 GPU 1장 기준 **~500 step(~100초)** 만에 수렴 → 매우 가벼운 학습.

---

## 3. 추론(Inference): 두 가지 검색 모드

압축된 sparse embedding은 **Compressed Sparse Row(CSR)** 형식으로 저장. `k`개 nonzero면 값+인덱스로 `2·k·4 bytes`만 필요.
예: 768차원 dense → 4096차원 sparse(k=32)로 압축 시 **12배 압축**.

**(1) Sparse 압축 공간에서 직접 검색**
- Sparse 벡터끼리 dot product는 `O(k)` — 임베딩 차원과 무관하게 빠름 (SpMV 커널, pgvector의 HNSW 등과 호환).
- 빠르지만 근사적.

**(2) 복원(reconstructed) 공간에서 검색 — kernel trick**
- Decoder가 linear이므로, sparse 표현 `s_x, s_y`만으로 dense 복원 공간에서의 cosine similarity를 근사 계산 가능:

$$\frac{x^\top y}{\|x\|_2\|y\|_2} \approx \frac{s_x^\top K s_y}{\sqrt{s_x^\top K s_x}\sqrt{s_y^\top K s_y}}, \quad K = W_{dec}^\top W_{dec} \in \mathbb{R}^{s\times s}$$

- `s_x`, `s_y`가 각각 k개 nonzero이므로 복잡도는 `O(k²)` — 여전히 효율적이면서 **더 정확** (실험상 가장 좋은 trade-off).

---

## 4. 실험 결과

미디어 도메인 글로벌 고객사의 proprietary 데이터셋(카탈로그 O(10⁸))으로 실험.

### 4.1 Offline
- **학습 속도**: 약 15초 만에 동일 크기 Matryoshka의 recall@100을 추월.
- **압축률-정확도 trade-off** (Figure 3 center): 특히 **고압축 구간에서 CompresSAE가 Matryoshka보다 우세** — CompresSAE는 **4배 더 큰 Matryoshka 모델과 동등한 성능**을 냄.
- **복원 공간에서의 검색**(kernel trick)이 전체적으로 **가장 좋은 trade-off**.

### 4.2 Online A/B Test
8.5M 유저 규모, 4개 variant 비교: SBERT(512d, baseline) / Nomic(768d) / Nomic+Matryoshka(64d) / Nomic+CompresSAE(4096d, k=32, sparse 공간에서 직접 검색).

| Model | 차원 | CTR lift (vs SBERT) | 100M embedding 저장 용량 |
|---|---|---|---|
| SBERT (baseline) | 512 | — | 204.8 GB |
| Nomic | 768 | +4.86% | 307.2 GB |
| Nomic + Matryoshka | 64 | +1.89% | 25.6 GB |
| Nomic + **CompresSAE** | 4096 (nonzero 32) | **+3.44%** | 25.6 GB |

- CompresSAE(12배 압축)는 압축 안 한 Nomic 대비 CTR **-1.35%**만 손해.
- 동일 용량(25.6GB)의 Matryoshka 대비 **+1.52%** 통계적으로 유의하게 우세.
- 즉, **같은 메모리 예산에서 Matryoshka보다 나은 성능**, 원본 대비도 손실 최소.

---

## 5. 결론

- **CompresSAE**: dense embedding을 고차원·희소 벡터로 변환하는 경량 SAE. cosine similarity 보존에 특화된 loss/activation 설계.
- **Encoder 재학습 불필요** (post-hoc 압축) → Matryoshka 대비 훨씬 유연하고 저렴하게 적용 가능.
- 메모리/연산을 크게 줄이면서도 downstream 성능(CTR) 손실 최소화, 동급 용량의 Matryoshka보다 우수.
- **결론 메시지**: "sparse화(고차원+sparsity)"가 "저차원화(truncation)"보다 나은 압축 전략이 될 수 있다.

---

## 6. 메모 (배울 점)

- **Matryoshka vs Sparse의 철학 차이**: Matryoshka는 "차원을 줄이는" 압축(dense, low-dim), CompresSAE는 "차원은 늘리되 대부분을 0으로 만드는" 압축(sparse, high-dim). 후자가 표현력을 더 잘 보존한다는 것이 핵심 주장 — 뉴런 수는 많지만 각 샘플마다 소수만 활성화되는 것이 정보 손실을 줄이는 데 유리하다는 SAE 계열의 공통 직관.
- **재학습 불필요라는 실용적 이점**: production에서 이미 서빙 중인 backbone encoder(SBERT, Nomic 등)를 건드리지 않고 그 출력 위에만 SAE를 얹으면 됨 → 배포 리스크와 재학습 비용을 크게 낮춤. Matryoshka는 backbone을 재학습해야 하므로 이미 서빙 중인 시스템에 적용하기 부담스러움.
- **Retrieval 특화 설계 3가지**: (1) 입력 normalize로 cosine 지향, (2) `φ(·,k)`가 절댓값 top-k + 부호 보존 → 방향 정보 유지, (3) cosine reconstruction loss. "해석가능성용 SAE"를 그대로 안 쓰고 태스크(retrieval)에 맞게 재설계한 점이 기여.
- **두 가지 추론 모드 trade-off**: sparse 공간 직접 검색(빠름, `O(k)`, 근사) vs kernel trick으로 복원 공간 검색(`O(k²)`, 더 정확). 시스템 요구사항(latency vs 정확도)에 따라 선택 가능한 유연성.
- **학습이 매우 가벼움**(H100 1장, ~100초) → 대규모 catalog에도 자주 재학습하며 운영하기 부담 없음.
- **A/B test에서 "동급 용량 비교"로 증명**: 단순히 "우리 방법이 좋다"가 아니라 "같은 메모리 예산일 때 Matryoshka보다 낫다"는 fair comparison으로 실무적 설득력 확보.

### 관련 논문
- Matryoshka Representation Learning, [Matryoshka Representation Learning, NeurIPS 2022](./Matryoshka%20Representation%20Learning,%20NeurIPS%202022.md) — 본 논문이 극복하고자 하는 baseline 압축 기법
- Scaling and evaluating sparse autoencoders (Gao et al., 2024) — CompresSAE 아키텍처의 기반이 된 해석가능성용 SAE (OpenAI, dead-neuron 방지 기법 차용)
- Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation (Wen et al., 2025) — SAE + contrastive loss로 임베딩 압축, 본 논문이 직접 비교/확장하는 선행 연구
- k-Sparse Autoencoders (Makhzani & Frey, 2014) — self-supervised dictionary learning의 원류
- Toward 100TB Recommendation Models with Embedding Offloading, RecSys 2024 Meta — 압축이 아닌 offloading/caching으로 대응하는 대안 접근

---

## Appendix. CompresSAE 자세히 (예시로 이해하기)

### A.1 왜 필요한가

Dense embedding은 모든 차원이 (거의) 다 값을 가진다. 예: 768차원 벡터라면 768개 float 모두 저장·전송·연산해야 함. 아이템 1억 개면 `1억 × 768 × 4byte ≈ 307GB`.

**Matryoshka의 해법**: 차원 자체를 줄인다 (768 → 64). 문제는 이렇게 되려면 애초에 "앞쪽 64차원만 잘라도 의미 있게 동작"하도록 encoder를 그렇게 학습시켜야 함 → **encoder 재학습 필수**.

**CompresSAE의 해법**: 차원은 오히려 늘리되(768 → 4096), 그중 대부분(4096-32=4064개)을 0으로 만든다. 0인 값은 저장할 필요가 없으므로 실제 저장 비용은 `32개 값 + 32개 인덱스`만 있으면 됨 → 사실상 32차원 dense와 비슷한 저장 비용이지만 표현력은 4096차원.

```
Dense:    [0.12, -0.05, 0.33, 0.01, ..., 0.09]   (768개 모두 저장)
Sparse:   [0, 0, 0.41, 0, 0, ..., -0.28, 0, 0, 0.15, 0, ...]  (4096개 중 32개만 저장)
```

### A.2 학습 과정 (Autoencoder)

CompresSAE는 이미 학습된 encoder(SBERT, Nomic 등)의 **출력 embedding만** 가지고 학습한다. 원본 텍스트/이미지나 그 encoder 자체는 필요 없음.

```
dense embedding x (768d)
   │  정규화: x̄ = x/‖x‖₂
   ▼
[Encoder: W_enc·x̄ + b_enc]  →  4096차원 벡터
   │
[φ(·, k)]: 절댓값 top-32만 남기고 나머지 0
   ▼
sparse embedding s (4096d, nonzero 32개)
   │
[Decoder: W_dec·s]  (단순 행렬곱, bias 없음)
   ▼
복원된 dense embedding x̂ (768d)
```

**loss**: 원본 `x`와 복원 `x̂`의 **cosine distance**를 최소화 (`1 - cos(x, x̂)`). 즉 "원래 벡터가 가리키던 방향을 32개의 0이 아닌 값만으로도 최대한 비슷하게 재현하라"고 학습시키는 것.

- `k`와 `4k` 두 가지 sparsity로 동시에 reconstruction을 시켜서 loss를 더함 → 특정 latent 차원이 한 번도 top-k에 안 뽑혀서 죽어버리는(dead neuron) 현상을 방지.

### A.3 φ(·, k) 함수 — TopK와의 차이

일반적 TopK activation은 **값이 큰 것만** 남긴다(음수는 버려짐). CompresSAE의 `φ`는 **절댓값이 큰 것**을 남기고 부호는 유지한다.

```
원 벡터 일부: [3.0, -5.0, 0.1, 2.0, -0.2]
일반 TopK(k=2, 양수 기준): [3.0, 0, 0, 2.0, 0]     ← -5.0(가장 큰 정보량) 손실!
φ(·, k=2, 절댓값 기준):    [0, -5.0, 0, 2.0, 0]     ← 부호 보존, 정보량 큰 값 유지
```

방향 정보(코사인 유사도의 핵심)를 보존하려면 음수도 중요하므로 이 설계가 retrieval에 유리하다.

### A.4 검색 시 두 가지 옵션 — 비유

우편번호 비유로 보면: 압축된 sparse 벡터 `s`는 "우편번호"고, 원래 dense 벡터 `x`는 "정확한 주소"다.

- **옵션 1 (sparse 공간 직접 비교)**: 우편번호끼리만 비교 → 빠르지만 대략적인 유사도.
- **옵션 2 (kernel trick, 복원 공간 비교)**: 우편번호를 이용해 "실제 주소가 얼마나 가까울지"를 수학적으로 정확히 역산 (`K = W_dec^T W_dec`라는 4096×4096 커널 행렬을 미리 계산해두면, sparse 벡터 2개의 32개 nonzero 값만으로 `O(k²)` 만에 계산 가능) → 더 느리지만 훨씬 정확.

실무에서는 후보를 빠르게 좁힐 때 옵션 1, 최종 정밀 랭킹엔 옵션 2를 쓰는 식의 조합도 가능.

### A.5 임베딩 norm(길이)은 실제로 무엇을 담고 있나

2.2절에서 "norm은 의미 정보가 아니다"라고 했는데, 그럼 대체 무엇을 담고 있는 걸까? 몇 가지 잘 알려진 메커니즘이 있다.

**(1) 학습 중 업데이트 횟수 (= 인기도/빈도)**

학습은 gradient로 벡터를 조금씩 움직이는 과정이다. **자주 등장(자주 상호작용)하는 아이템일수록 gradient update를 더 많이 받아** 초기값(0 근처의 작은 랜덤값)에서 더 멀리 밀려나며 norm이 커지는 경향이 있다. 실제로 추천시스템의 popularity bias 연구에서, 인기 아이템의 임베딩이 비인기 아이템보다 체계적으로 norm이 큰 현상이 관찰된다. 이건 "인기 아이템이 의미적으로 더 풍부해서"가 아니라 **단순히 학습 신호를 더 많이 받았기 때문**이다.

**(2) 입력의 "신뢰도/품질" 신호**

얼굴 인식(face recognition) 임베딩 연구에서 잘 알려진 현상: 같은 사람이라도 선명한 사진은 임베딩 norm이 크고, 흐릿하거나 가려진 사진은 norm이 작게 나온다. "누구인지"(방향)는 비슷하게 잡히지만, **네트워크가 입력을 얼마나 확신하는지가 norm에 반영**된다. 이를 역이용해 norm을 품질 신호로 쓰는 모델(AdaFace 등)도 있다.

**(3) 문맥의 "일관성" (word2vec 연구, Schakel & Wilson 2015)**

- `the`, `a`, `is` 같은 기능어는 거의 모든 문맥에서 등장 → 매번 업데이트 방향이 제각각이라 gradient가 서로 상쇄되며 norm이 작게 수렴.
- `photosynthesis` 같은 특정 단어는 항상 비슷한 좁은 문맥에서만 등장 → gradient가 같은 방향으로 누적되며 norm이 크게 자람.

즉 norm이 "**이 단어/아이템이 얼마나 예측 가능하고 특정적인 문맥에서 쓰이는가**"라는, 의미 카테고리와는 다른 축의 정보를 담게 된다.

**(4) 입력 길이/pooling 방식**

Sentence embedding은 보통 토큰들을 pooling(평균 등)해서 하나의 벡터로 만드는데, 입력 길이나 pooling 방식에 따라 벡터의 스케일 특성이 달라질 수 있다. 의미가 비슷한 짧은 문장과 긴 문서도 pooling 과정 차이로 norm이 다르게 나올 수 있다.

**결론**: norm은 "무엇을 의미하는가"보다 "**학습 중 이 벡터가 얼마나 많이, 얼마나 일관되게 움직였는가**"(노출 빈도, 문맥 일관성, 입력 품질/길이 등)를 반영하는 부산물에 가깝다. 검색처럼 "의미가 비슷한가"만 알고 싶은 태스크에서는 이 norm을 지워버리고(cosine similarity) 방향만 보는 게 노이즈를 없애는 방법이 된다.

### 한 줄 정리

**"임베딩 차원을 줄여서 압축"(Matryoshka)** 이 아니라 **"차원은 늘리되 대부분 0으로 만들어서 압축"(CompresSAE)** 하는 접근. encoder 재학습 없이 사후에 붙일 수 있고, 동일 저장 용량 기준으로 Matryoshka보다 검색 성능이 더 좋았다.
