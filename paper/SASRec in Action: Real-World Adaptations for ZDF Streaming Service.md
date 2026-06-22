# SASRec in Action: Real-World Adaptations for ZDF Streaming Service

- **저자**: Venkata Harshit Koneru, Sebastian Loth, Andreas Grün (ZDF), Xenija Neufeld (Accso)
- **연도**: 2025 (RecSys '25)
- **링크**: https://doi.org/10.1145/3705328.3748097

---

## 1. 배경 & 문제 정의

**ZDF**는 독일 공영방송으로, 스트리밍 플랫폼에서 시리즈·다큐·뉴스 등을 제공한다. 추천 품질은 두 가지 축으로 모니터링:

- **KPI (비즈니스 지표)**: 클릭, **viewing volume(시청량)**
- **PVM (Public Value Metrics, 공영가치 지표)**: 추천 콘텐츠의 **다양성**과 **popularity(인기도)**

추천 백본으로 **SASRec**(Self-Attentive Sequential Recommendation)을 여러 use case에서 사용 중. 그러나 SASRec 같은 sequential recommender는 **popularity bias**(인기 아이템만 자주 추천) 문제가 있다.

> popularity bias → 추천의 관련성·개인화 저하 → 오히려 시청량 감소로 이어질 수 있음

이 논문은 popularity bias를 줄이면서 KPI(viewing volume)를 끌어올리려는 시도. **negative sampling 전략**과 **데이터 증강(RepPad)** 조합을 3개 use case에서 A/B 테스트로 검증.

### 3가지 Use Case

| Use Case | 설명 | 추천 개수 | 추론(inference) 입력 |
|----------|------|-----------|----------------------|
| **Next Video** | 현재 보던 영상 기반 다음 영상 1개 추천 | 1개 | - |
| **DKDI** (Das Könnte Dich Interessieren = "관심 있을 만한 것") | 홈 상단, 유저의 **관심사/시청 이력** 기반 | 최대 25개 (가로 리스트) | 최근 **10개** 아이템 |
| **Weil-Du** ("당신이 ...을 봤기 때문에") | 홈 하단, **마지막으로 본 아이템**과 유사한 추천 | 최대 25개 (가로 리스트) | **1개** 아이템 (보통 최근 것) |

---

## 2. 실험 셋업

### 2.1 기반 기술 (이전 연구 [4]에서 가져옴)

- **gBCE loss**: vanilla SASRec의 BCE 대신 generalized Binary Cross-Entropy 사용 → overconfidence 완화
- **top-k negative item 선택** [6]

→ 이 조합(vanilla 대비 KPI 우수)이 본 논문의 **baseline = Variant 1**.

### 2.2 새로 추가한 기법

**① RepPad (Repeated Padding)** [1]
- 짧은 history를 0으로 채우는(zero-padding) 대신, **짧은 이력을 반복**해서 idle 입력 공간을 채움
- positive item의 제시 방식을 조정해 학습 정확도 향상
- ZDF에 특히 유효한 이유: **쿠키 삭제 등으로 유저 이력이 매우 짧음**

**② neg (mixed negative sampling)** [6]
- **uniform + in-batch sampling** 혼합 전략
- 이전 실험 [4]에서 popularity bias를 줄이는 효과 확인됨

### 2.3 세 가지 모델 변형 (online A/B test, 23일간)

| 변형 | 구성 | 핵심 차이 |
|------|------|-----------|
| **Variant 1** | SASRec gBCE | baseline |
| **Variant 2** | SASRec gBCE **neg RepPad** | mixed neg sampling + RepPad (최신 novel 접근) |
| **Variant 3** | SASRec gBCE **RepPad** | RepPad만 추가 |

> Variant 2 vs 3 비교로 **neg sampling 효과**를 분리하고, Variant 1 vs 3 비교로 **RepPad 효과**를 분리할 수 있음.

---

## 3. A/B 테스트 결과 (Figure 1)

각 그래프: x축 = 평균 popularity(top-1 추천), y축 = 누적 viewing volume(%)

### Next Video
- Variant 1, 3은 비슷한 수준
- **Variant 2가 가장 낮은 popularity + 가장 높은 viewing volume** ✅
- 해석: **낮은 popularity ↔ 높은 engagement**. 인기도 감소의 주역은 **neg sampling**(Variant 2 only). RepPad(2,3 공통)는 popularity에 유의미한 영향 없음
- → **mixed neg sampling이 popularity bias 완화의 핵심 driver**

### DKDI
- Variant 2가 popularity는 유의하게 낮지만, **세 변형의 viewing volume은 거의 동일**

### Weil-Du
- **baseline(Variant 1)이 popularity·viewing volume 모두 최고** ✅
- → Weil-Du에서는 **popularity가 유저 선택에 긍정적 역할** (인기 아이템이 오히려 도움)

---

## 4. Viewing Volume에 영향을 주는 핵심 요인 (회귀 분석)

DKDI와 Weil-Du는 여러 아이템을 추천하므로, **추천 위치(가로 position)**도 변수로 작용.

**Multiple Linear Regression** 수행:
- 종속변수: position별 평균 viewing volume (Min-Max 스케일링)
- 독립변수: **Reco-position**(추천 위치) + **Popularity**(인기도 quantile)
- 모든 모델 r² > 0.9 (높은 설명력)

### Table 1 회귀 계수 요약

| Use Case | 변수 | Variant 1 | Variant 2 | Variant 3 |
|----------|------|-----------|-----------|-----------|
| **DKDI** | Reco-position | -1.65 *** | -0.66 ** | -1.75 *** |
| | Popularity | **-0.54 ***** | +0.55 (n.s.) | **-0.60 ***** |
| **Weil-Du** | Reco-position | -0.90 *** | -0.68 ** | -0.68 *** |
| | Popularity | **+0.17 ***** | +0.19 (n.s.) | +0.21 (n.s.) |

(*** p<0.01, ** p<0.05, n.s. = 유의하지 않음)

**해석:**

1. **Reco-position**: 모든 경우 강한 음의 효과 → **오른쪽에 위치할수록 viewing volume 감소** (가시성 제약 + 네비게이션 노력). 당연한 결과.

2. **Popularity 효과는 use case마다 다름**:
   - **Weil-Du**: 모든 변형에서 **양수** → popularity bias 증거 없음. 오히려 인기 아이템이 시청량에 도움 (Variant 1은 유의)
   - **DKDI**: Variant 1, 3에서 **유의한 음수** → **popularity bias 존재**. Variant 2는 양수지만 유의하지 않아 결론 보류(inconclusive)
   - → DKDI에서는 **Variant 2가 KPI/PVM 간 최선의 trade-off** (비슷한 viewing volume + 낮은 popularity) → 선호됨

### 왜 DKDI와 Weil-Du가 다른가? (핵심 인사이트)

> **추론에 사용하는 아이템 개수의 차이** 때문.

- **Weil-Du = 1개 아이템 추론**: sequential model에서 인기 아이템은 다른 인기 아이템과 강하게 연결됨. 입력이 1개뿐이므로 그 아이템이 인기일 때 **더 많은 학습 데이터의 혜택**을 받음 → bias 안 생김
- **DKDI = 10개 아이템 추론**: 여러 입력 중 **하나의 highly popular item이 모델의 attention을 분산(distract)** → popularity bias 발생 → 이를 막을 메커니즘(neg sampling) 필요

---

## 5. 결론 & 시사점

데이터 증강(RepPad) + negative sampling 수정의 효과는 **use case에 따라 다르며, 특히 추론에 쓰는 아이템 개수에 의존**한다.

| Use Case | 최선의 모델 | 이유 |
|----------|-------------|------|
| **Next Video** (top-1만 제시) | **Variant 2** (RepPad + mixed neg) | viewing volume 최고, popularity 최저 |
| **Weil-Du** (다중 추천, 입력 1개) | **Variant 1** (baseline) | 어떤 수정도 개선 없음, popularity bias 없음 |
| **DKDI** (다중 추천, 입력 10개) | **Variant 2** (미미한 개선) | trade-off 측면에서 선호 |

**핵심 메시지: 모델 선택은 use case의 요구사항에 따라 달라진다.** 범용 best 모델은 없음.

**향후 연구:**
- popularity bias가 use case 요구사항에 따라 어떻게 발생하는지
- 추론 시 single vs multiple item 입력의 영향
- 유저에게 single vs multiple 추천 제시의 영향
- 추천 리스트의 **수직(vertical) 페이지 배치**와 유저 기대의 정합성

---

## 6. 메모 (배울 점)

- **RepPad**: 짧은 user history 문제(쿠키 삭제 등 현실적 제약)에 대한 실용적 데이터 증강. zero-padding을 이력 반복으로 대체.
- **Mixed negative sampling (uniform + in-batch)**: popularity bias 완화의 실질적 driver. loss(gBCE)보다 sampling 전략이 더 결정적이었음.
- **같은 모델도 use case 셋업(입력 아이템 수, 추천 개수, 페이지 위치)에 따라 효과가 정반대**일 수 있다 → 단일 오프라인 metric으로 일반화 금지, use case별 A/B 필수.
- 공영방송 맥락에서 **KPI(viewing volume)와 PVM(popularity/diversity)의 trade-off**를 명시적으로 관리하는 점이 특징.

### 관련 논문
- SASRec [3] Kang & McAuley, 2018
- gSASRec (gBCE) [5] Petrov & Macdonald, 2023
- RepPad [1] Dang et al., 2024
- Optimized Negative Sampling [6] Wilm et al., 2023
- ZDF 이전 연구 (popularity bias 완화) [4] Koneru et al., 2024
