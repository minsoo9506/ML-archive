# LADDER: LLM-Annotated Data for Dogfooded Evaluation of Rankings

- **Author**: Mattia Ottoborgo (Trustpilot, Copenhagen)
- **Venue**: RecSys 2025, September 22–26, Prague, Czech Republic
- **링크**: https://doi.org/10.1145/3705328.3748094

---

## 1. 문제 정의

### 배경
- Trustpilot의 **Company Profile Page**에서 보여주는 리뷰 목록은 유저 engagement에 결정적
- 유저가 회사 웹사이트로 전환되려면 **고품질·최신·진정성 있는·의미적으로 관련 있는** 리뷰가 상단에 노출되어야 함
- 어떤 리뷰가 relevant한지, 어떤 순서로 보여줄지 → **Learning-to-Rank (LTR)** 문제

### LTR 데이터셋 구축의 한계
- **수동 어노테이션**: 대규모로 하기엔 시간/비용 prohibitive
- **암묵적 피드백** (Useful 버튼 클릭, view time 등): 데이터 불균형, 정확도 부족
- → **LLM을 어노테이터로 활용**하여 정확도와 확장성 동시 해결

---

## 2. LADDER 방법론

### 2.1 전체 파이프라인 (Figure 1)

```
Generate annotation rules
  → Generate LLM-annotated dataset
  → Supervised training (point-wise LTR)
  → Dog-fooding (internal evaluation)
  → Online experiment
```

### 2.2 LLM 어노테이션 설계

- **LLM**: Gemini 사용
- 각 리뷰에 대해 **0 (non-relevant) ~ 100 (very relevant)** 점수 부여 (pointwise)
- **Chain-of-Thought (CoT) prompting** + **enriched context** 사용
- 단순 점수만이 아니라 "사람이 평가할 때의 기준"을 LLM에 명시적으로 주입

### 2.3 4가지 평가 기준 (도메인 지식 주입)

내부 user study (리뷰를 Useful / Somewhat useful / Not useful로 분류)에서 도출된 4가지:

| 기준 | 설명 |
|---|---|
| **Authenticity** | 진정성 있어야 highly relevant. 콘텐츠 외적 feature → 프롬프트 엔지니어링이 핵심 |
| **Quality** | 문법/스타일이 좋고 불필요한 반복이 없을 것 |
| **Recency** | 회사의 현재 상태를 반영할 수 있도록 최근에 작성된 것 |
| **Length** | 너무 짧지도 길지도 않게 — engaging한 길이 |

**핵심 인사이트**: Quality·Length 같은 콘텐츠 기반 feature는 LLM이 자연스럽게 평가하지만,
**Authenticity (진정성)** 같은 비콘텐츠 feature를 점수에 반영시키는 것이 가장 어려웠고
이를 위한 **prompt engineering이 결정적**이었음.

---

## 3. 실험

### 3.1 비교 후보 모델

| 모델 | 설명 |
|---|---|
| **Baseline** | 기존 production 모델, 휴리스틱 기반 (labelled data로 학습 X) |
| **LADDER (LTR)** | LLM-annotated dataset으로 학습된 point-wise LTR |
| **LLM-based score** | LLM이 직접 매긴 pointwise 점수로 정렬 |

→ LADDER vs LLM-score 비교로 "LTR의 generalization 능력 vs LLM 어노테이션 자체의 alignment"를 분리해서 확인

### 3.2 평가 메트릭

- **Quality**: top-k 리뷰의 평균 작문 품질 (0~1)
- **Recency**: top-k 리뷰와 가장 최신 리뷰 간 일자 차이 평균
- **Authenticity**: top-k 리뷰의 평균 진정성 점수 (0~1)
- **Fairness**: top-k 리뷰의 평균 별점과 회사 Trustscore의 차이
- **NDCG**: 위치 가중 relevance 평가
- **Impact**: 새 알고리즘 top-k의 평균 relevance vs baseline top-k 평균 relevance 차이

### 3.3 오프라인 테스트

**모델 메트릭**:
- Quality: 약간 감소 (baseline이 quality에 과적합되어 있었음 — 예상된 trade-off)
- **Authenticity: 크게 향상** (Figure 4)
- Fairness: 안정적 유지
- **NDCG: 0.7259 → 0.7636 향상**

**Impact % (Table 1)** — 가짜 리뷰 비율 높은 회사일수록 효과 큼:

| Top K | Test Sample | Restricted Sample (fake review 많은 회사) |
|---|---|---|
| 5 | 11.25% | **14.90%** |
| 10 | 11.26% | **16.68%** |
| 20 | 9.99% | 13.77% |

### 3.4 Dog-fooding (내부 평가)

- **Round-based pairwise 비교 게임**: 50명 유저에게 같은 회사의 두 ranking을 보여주고 선호 선택
- 선택된 알고리즘 +1, 선택받지 못한 쪽 -1, 보여지지 않은 쪽 0
- 회사명·평점·위치·카테고리를 제공해 informed decision 유도
- **결과 (Figure 5)**: 초반 변동 후 안정화 → **LADDER (supervised) > LLM scores > Baseline**
- 시사점: LTR이 LLM의 raw 점수보다 더 나은 generalization을 보임

### 3.5 Online experiment ("Do No Harm" test)

- 약 **14,000개 비즈니스** 대상 A/B 테스트
- **"See all reviews" 버튼 클릭률 5% 감소** (Figure 2)
- 해석: top-4 리뷰만 보고도 유저가 필요한 정보를 얻을 수 있게 됨 → ranking 품질 개선의 증거

---

## 4. 핵심 기여

1. **LLM 어노테이션을 통한 LTR 데이터셋 구축 파이프라인** 제시 — 수백 시간의 수동 어노테이션 절감
2. **도메인 지식 (4가지 평가 기준)을 LLM 프롬프트에 통합**하는 방법론
3. **3단계 검증 프레임워크**: 모델 메트릭 → dog-fooding → online experiment
4. **Authenticity처럼 콘텐츠 외적 feature를 LLM이 평가하도록 만드는** prompt engineering 노하우
5. 실제 production에서 **검증된 비즈니스 임팩트** (5% click 감소)

---

## 5. 비판적 고찰

### 5.1 Pointwise LTR의 한계
- LADDER는 point-wise LTR. 그런데 ranking은 본질적으로 상대적인 문제 → pairwise/listwise 대비 약점 존재 가능
- 하지만 LLM이 "0~100 점수"로 어노테이션하기 때문에 자연스럽게 pointwise와 호환

### 5.2 LLM as Judge의 일반적 위험
- LLM (Gemini) 자체의 bias가 데이터셋 전체에 주입됨 → ground truth가 아닌 "LLM이 생각하는 ground truth"
- 다만 dog-fooding에서 LADDER가 LLM raw score보다도 선호됨 → LTR이 noise는 smoothing, signal은 흡수했다고 해석 가능

### 5.3 Authenticity 평가의 신뢰성
- 진정성 평가는 본질적으로 LLM의 텍스트 분석만으로는 한계 (메타데이터, 행동 신호 필요)
- 논문에서도 prompt engineering에 가장 많은 노력이 들었다고 언급 → 재현성에 의문

### 5.4 평가 메트릭 자체의 순환성
- **Impact 메트릭**: "새 모델 기준 relevance score"로 계산 → 당연히 새 모델이 이김 (self-referential)
- NDCG도 LLM이 매긴 label 기준 → LLM에 잘 맞게 학습된 LTR이 유리한 게 당연

### 5.5 작은 dog-fooding 샘플
- 50명 내부 직원 → selection bias 존재
- 실제 Trustpilot 일반 사용자와의 선호 차이 가능성

---

## 핵심 요약 (한 줄)

> LLM (Gemini)에 도메인 평가 기준(authenticity, quality, recency, length)을 CoT 프롬프트로 주입해 대규모 리뷰 ranking 데이터셋을 자동 생성하고, 이를 point-wise LTR로 학습시켜 Trustpilot 프로덕션에서 "See all reviews" 클릭률 5% 감소를 달성한 사례 — LLM-as-annotator를 LTR 파이프라인에 실용적으로 통합한 reference.
