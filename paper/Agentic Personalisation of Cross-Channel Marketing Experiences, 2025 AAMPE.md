# Introduction & Motivation
- CRM 도 개인화가 필요하다.
- 에이전트 방식으로 확장 가능하고 동적인 사용자 수준의 메시징 전략 개인화를 위한 일반적인 방법론
- 에이전트는 순차적 의사결정 모듈의 조합으로 구성되며, 개별 사용자 수준에서 작동하고, 사용자 경험의 다양한 차원에 걸쳐 시간이 지남에 따라 변화하는 사용자 선호에 적응함
  - leveraging methods from econometrics causal inference
  - contextual bandits 

# Methodology & Contributions
- 목표: 마케팅 커뮤니케이션의 개인화 (누구에게, 무엇을, 어떻게, 언제)
- 핵심 아이디어: 메시지의 속성(어조, 이모지, 가치 제안 등)을 모듈화하여 행동 공간을 관리 가능한 크기로 줄임
- 방법론적 기반: 의사결정 & 인과추론을 이용하여 high-dimensional yet structured treatment

## measuring outcomes
- 목표: 단순 상관이 아닌 **incrementality(증분 효과)** — 메시징이 실제로 결과를 유발했는지를 측정
- 이벤트를 두 종류로 분류: **goal event**(전환, 구독 갱신 등) vs **proxy event**(앱 오픈, 페이지뷰, 클릭 등)

### Outcome 정의 (Eq.1)
- 사용자 $u$의 기간 $[t_i, t_j]$ 동안의 outcome:

$$Y^u_{[t_i, t_j]} = \sum_{(t,e) \in \mathcal{D}_u} w_e \cdot w_t$$

  - $\mathcal{D}_u$: 해당 사용자의 (timestamp, event) 튜플 집합
  - $w_e$: 이벤트별 가중치 (log-likelihood ratio)
  - $w_t$: 시간 가중치 (exponential decay)

### Event Weight (이벤트 가중치)
- Granger causality 개념 기반의 log-likelihood ratio:

$$w_e = \ln \frac{P(e \mid \text{Goal})}{P(e \mid \neg\text{Goal})}$$

  - goal event에 선행하는 이벤트일수록 높은 가중치 부여

### Difference-in-Differences (DiD) for ITE (Eq.2)
- Individual Treatment Effect를 추정하기 위해 DiD 설계 사용:

$$\Delta Y = (Y^T_{\text{post}} - Y^T_{\text{pre}}) - (Y^C_{\text{post}} - Y^C_{\text{pre}})$$

  - $T$: treatment 그룹 (메시지 수신), $C$: control 그룹
  - treatment 전후 차이에서 control 전후 차이를 빼서 시간적/계절적 효과 제거

### 시간 윈도우 정의 (Eq.3)

$$\text{pre} = [t_{\text{int}} - t_\Delta,\ t_{\text{int}}), \quad \text{post} = [t_{\text{int}},\ t_{\text{int}} + t_\Delta)$$

  - $t_{\text{int}}$: 개입(intervention) 시점
  - $t_\Delta$: 튜닝 가능한 윈도우 파라미터
  - Interrupted Time Series의 sparse modification + 주기적 organic engagement 보정 → 형식적 DiD 설계

## controls
- 인과 효과를 측정하기 위해 (단순 상관이 아닌), 시간적·계절적 효과를 보정하고 이로 인한 변동성을 제거해야 함
- ITE 추정을 위해 개입받지 않은 사용자 집합 $C \subseteq \mathcal{U}$를 control로 정의
- **Bias-Variance 트레이드오프**:
  - 가장 유사한 사용자 1명만 사용 → low bias, high variance
  - 전체 사용자 사용 → high bias, low variance
  - 최적: treated 사용자 $T$에 대한 **nearest neighbours 집합**을 control로 사용
- 매칭된 control을 통해 DiD의 $Y^C$ 항을 구성하여 시간적/계절적 교란 효과를 제거

## decision-making
- **Thompson Sampling (TS)** 기반의 explore-exploit 균형
- 사용자 상태(context) $x$가 주어졌을 때, 각 action $a$에 대해 ITE 샘플링 후 최대값 선택:

$$\Delta \tilde{Y}_a \sim \hat{P}(\Delta Y \mid A=a;\ X=x)$$

$$a^* = \arg\max_{a \in \mathcal{A}} \Delta \tilde{Y}_a$$

- **사후분포 모델링**:
  - outcome $Y$가 연속형이고 Gaussian 가정 → $\Delta Y$도 Gaussian (선형결합) → closed-form update 가능
  - 해석 용이성을 위해 $\Delta Y$를 이진화하여 **Beta-Bernoulli 모델** 사용
- **Empirical Bayes Prior**: 특정 context-action 쌍에 데이터가 없을 때, 유사 사용자의 행동으로 분포 파라미터를 impute (user-based collaborative filtering에서 영감)

## synthesising messages
- 메시지를 하나의 discrete action으로 취급하지 않고, **모듈형 action set으로 분해**: 시간, 빈도, 채널, 메시지(tone-of-voice, emoji 사용 등)
- **Wolpertinger Architecture** 적용:
  1. Thompson Sampling이 각 action set에서 action을 선택
  2. 선택된 action 조합을 사용자가 수신 가능한 메시지 카탈로그에 매핑
  3. 가장 잘 매칭되는 메시지를 전송
- 이 구조로 개별 action 조합의 **조합적 폭발(combinatorial explosion)을 회피**

## human-in-the-loop
- 카피라이팅 관련 action set(tone-of-voice, value proposition, CTA 등)은 **마케팅/프로덕트 팀이 직접 관리·갱신·개선**
- 비즈니스가 발송 내용에 대한 통제권을 유지하면서, 전문가의 인간적 역량이 시스템 성능을 보강
- LLM 미사용으로 confabulation 방지 → **business guardrails by design**
- 핵심 철학: 마케터를 대체하는 것이 아니라, 마케터의 역량을 **증강(augment)** — 인간이 정의한 action space 내에서 자율적으로 운영

# Empirical Results & Discussion

## 실험 셋업
- 대상: 멀티서비스 앱 (ride-hailing + food delivery), 여러 시장에서 운영
- 규모: **640만 사용자**, 3주간 RCT (현재는 1.5억 사용자에 배포됨)
- 채널: push notification + in-app messaging (WhatsApp, email, SMS는 이번 실험 미포함)
- **Treatment**: 기존 rule-based 메시지 위에 에이전트 기반 개인화(시간, 콘텐츠, 빈도) 적용
- **Control**: population-level 휴리스틱으로 최적화된 표준 rule-based 메시지만 수신
- 4개의 product feature에 대해 평가

## 평가 지표
- **Intent**: 상위 퍼널 이벤트(페이지뷰, 클릭 등)를 수행한 사용자 비율
- **Conversion**: 실제 기능을 사용/전환한 사용자 비율
- **GMV (Gross Merchandise Value)**: 전환의 직접적 비즈니스 임팩트

## 주요 결과 (99% 신뢰구간, 절대 증가)

| Product Feature | Intent | Conversion | GMV (상대 증가) |
|---|---|---|---|
| Transactional 1 | [+0.70%, +0.83%] | [+0.42%, +0.48%] | [+14.12%, +22.62%] |
| Transactional 2 | [+1.76%, +1.92%] | [+0.30%, +0.36%] | [+25.35%, +43.39%] |
| Transactional 3 | [+1.30%, +1.44%] | [+0.07%, +0.13%] | [-0.31%, +22.55%] |
| Account Creation | [+2.37%, +2.45%] | [+0.21%, +0.25%] | N/A |

- 모든 use case에서 상위·하위 퍼널 이벤트 모두 유의미한 개선
- 640만 사용자 규모에서 이 수치는 실질적으로 큰 비즈니스 임팩트

## 방법론적 세부사항
- Propensity score matching으로 최소 1회 메시지 수신 사용자와 미수신 control 사용자를 매칭
- 3개 transactional feature: 거래 완료(주문 완료)가 end-of-funnel event, 장바구니 담기가 intent signal
- Account creation feature: 계정 생성이 goal, 약관/상품 페이지 조회가 intent signal

## Discussion
- 에이전트 프레임워크가 orchestration 단계의 **운영 오버헤드를 크게 줄임**
- 마케터가 수동으로 관리할 수 있는 것보다 **훨씬 많은 메시지 변형을 배포** 가능
- 마케터를 대체하는 것이 아니라 역량을 증강(augment)하는 도구

# Conclusions & Outlook
- 마케팅 커뮤니케이션의 수동 orchestration이라는 병목을 에이전트 프레임워크로 해결
- 640만 사용자 대상 실험에서 모든 use case에서 상·하위 퍼널 이벤트 유의미 개선, GMV 두 자릿수 상대 증가
- 운영 오버헤드를 크게 줄이고, 수동 대비 훨씬 많은 메시지 변형 배포 가능
- 마케터 대체가 아닌 증강(augment) — human-in-the-loop으로 비즈니스 guardrails 유지
- 현재 전체 사용자(1.5억)에 배포되어 운영 중이며, 추가 채널·use case로 확장 가능한 모듈형 구조
- 사용자 중심 마케팅 시스템을 위한 **실용적이고 확장 가능한 한 걸음(a practical, scalable step forward)**으로 포지셔닝

# 기억에 남는 점
- 일반적인 단순 conversion, click 을 outcome 으로 하지 않고 DID 기법으로 ITE 를 구해서 outcome 을 계산했다는 점
- 메시지 발송 관련 다양한 action 을 독립적인 TS 로 구해서 조합을 했다는 점 -> 한계점이지만 기술적 & 현실적으로 납득 가능
- 모든 과정을 자동화하지 않고 마케터가 생성한 Pool 내에서 constraint 를 두고 발송했다는 점 -> 자동화로 인해 발생할 수 있는 위험을 줄이고 과정과 결과에 대해 가시적인 접근이 가능

# appendix
## Thompson Sampling
- Multi-armed bandit 문제에서 **explore-exploit 균형**을 맞추는 확률적 알고리즘 (1933, Thompson)
- 핵심 아이디어: 각 action의 보상 분포에서 **샘플을 뽑고**, 가장 높은 샘플 값을 가진 action을 선택

### 알고리즘 흐름
1. 각 action $a$에 대해 보상의 사후분포 $P(\theta_a \mid \text{data})$를 유지
2. 매 시점 $t$마다:
   - 각 action의 사후분포에서 파라미터를 샘플링: $\tilde{\theta}_a \sim P(\theta_a \mid \text{data})$
   - 샘플 값이 최대인 action 선택: $a_t = \arg\max_{a} \tilde{\theta}_a$
   - 보상 $r_t$를 관측하고 사후분포 업데이트

### Beta-Bernoulli 모델 (이진 보상)

#### Bayes' Theorem 유도
- posterior ∝ likelihood × prior 는 정의가 아니라 **조건부 확률의 정의**로부터 유도됨:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)} \quad \text{(조건부 확률의 정의)}$$

$$P(A \cap B) = P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A) \quad \text{(대칭성)}$$

$$\therefore P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)} \quad \text{(Bayes' theorem)}$$

- $A = \theta$, $B = \text{data}$로 놓으면:

$$\underbrace{P(\theta \mid \text{data})}_{\text{posterior}} = \frac{\overbrace{P(\text{data} \mid \theta)}^{\text{likelihood}} \cdot \overbrace{P(\theta)}^{\text{prior}}}{\underbrace{P(\text{data})}_{\text{evidence (상수)}}}$$

- $P(\text{data})$는 $\theta$에 의존하지 않으므로: $\text{posterior} \propto \text{likelihood} \times \text{prior}$

#### Bernoulli Likelihood
- 보상 $r \in \{0, 1\}$이 성공 확률 $\theta$인 Bernoulli 분포를 따름:

$$P(r \mid \theta) = \theta^r (1-\theta)^{1-r}$$

- $n$번 관측 후 likelihood ($s$: 성공 횟수, $f$: 실패 횟수, $n = s + f$):

$$P(r_1, \dots, r_n \mid \theta) = \theta^s (1-\theta)^f$$

#### Beta Prior
- $\theta$의 사전분포로 Beta 분포 사용:

$$P(\theta) = \text{Beta}(\theta \mid \alpha, \beta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}$$

  - $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$: 정규화 상수 (Beta 함수)
  - $\alpha, \beta > 0$: shape 파라미터

#### Conjugate 관계 (왜 Beta-Bernoulli인가)
- Bayes' rule로 사후분포를 구하면:

$$P(\theta \mid \text{data}) \propto P(\text{data} \mid \theta) \cdot P(\theta)$$

$$\propto \theta^s (1-\theta)^f \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1} = \theta^{\alpha+s-1}(1-\theta)^{\beta+f-1}$$

- 이것은 다시 Beta 분포 형태 → **사후분포도 Beta**:

$$P(\theta \mid \text{data}) = \text{Beta}(\alpha + s,\ \beta + f)$$

- 즉, prior와 posterior가 같은 분포족(Beta)에 속함 → **conjugate prior**
- 관측 1개씩 순차 업데이트:

$$\alpha \leftarrow \alpha + r_t, \quad \beta \leftarrow \beta + (1 - r_t)$$

#### 직관적 해석
  - $\alpha$: 성공 횟수 + prior 가상 성공, $\beta$: 실패 횟수 + prior 가상 실패
  - 사후 평균: $E[\theta \mid \text{data}] = \frac{\alpha}{\alpha + \beta}$ (관측이 쌓일수록 표본 비율에 수렴)
  - 사후 분산: $\text{Var}[\theta] = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ (데이터 증가 시 감소 → 불확실성 축소)
  - $\text{Beta}(1,1) = \text{Uniform}(0,1)$: 무정보 사전분포로 초기화

### 왜 Thompson Sampling인가?
- **자연스러운 탐색**: 불확실성이 큰 action일수록 분산이 커서 높은 샘플이 뽑힐 확률 증가 → 자동으로 탐색
- **데이터 축적 시 수렴**: 관측이 쌓이면 사후분포가 좁아져 최적 action에 집중 (exploit)
- **구현 단순**: 샘플링 한 번 + argmax만으로 의사결정
- UCB 등 대비 empirical하게 우수한 성능, 특히 non-stationary 환경에서 강점

### 본 논문에서의 적용
- 각 (context $x$, action $a$) 쌍에 대해 ITE($\Delta Y$)의 사후분포를 유지
- $\Delta Y$를 이진화 → Beta-Bernoulli 모델로 closed-form 업데이트
- Empirical Bayes prior로 cold-start 문제 대응 (유사 사용자 데이터로 $\alpha, \beta$ 초기화)

## 전체 파이프라인 흐름

```
유저 context x 입력
  → 각 action set(when, how often, where, what)에서 TS로 arm 선택
  → 선택된 조합을 메시지 카탈로그에 매칭 (Wolpertinger)
  → 메시지 발송
  → 유저 이벤트 관측 → outcome Y 계산 (가중합)
  → DiD로 ITE(ΔY) 추정 → 이진화
  → Beta-Bernoulli posterior 업데이트
  → 다음 의사결정에 반영
```

### Outcome 계산 상세
- outcome $Y$는 단순 0/1이 아닌 이벤트의 **가중합** (log-likelihood ratio × exponential decay)으로 연속값
- DiD로 ITE($\Delta Y$) 계산 후, **이진화**하여 Beta-Bernoulli에 입력

### Action Set 구성 (4개 dimension)
- **When**: 발송 시간대
- **How often**: 발송 빈도
- **Where**: 채널 (push, in-app, email 등)
- **What**: 메시지 내용 (tone-of-voice, emoji, value proposition, CTA 등)

각 dimension에서 TS로 독립적으로 arm을 선택한 뒤, 조합을 직접 생성하는 것이 아니라 마케터가 사전에 만들어 놓은 **메시지 카탈로그에서 가장 가까운 메시지를 매칭**(Wolpertinger)하여 발송

## ΔY 이진화
- 논문은 "We binarise ΔY and use a Beta-Bernoulli model for interpretability"라고만 언급, 구체적 방법 미명시
- 가장 유력한 방법: sign 기반

$$r = \mathbb{1}[\Delta Y > 0]$$

  - $\Delta Y > 0$ → 메시지가 긍정적 증분 효과 → $r = 1$
  - $\Delta Y \leq 0$ → 효과 없거나 부정적 → $r = 0$
- 이렇게 하면 Beta-Bernoulli의 $\theta$가 **"해당 action이 해당 context에서 긍정적 증분 효과를 줄 확률"**로 해석됨
- 효과의 크기(magnitude) 정보는 버려짐 (ΔY=+0.01이든 +100이든 둘 다 $r=1$) — 해석 가능성을 위한 트레이드오프

## Action Set 독립 가정의 한계와 Wolpertinger
- 실제로 action set 간 독립이 아님 (예: "긴급한 tone" + "주 1회" → 모순, "할인 내용" + "새벽 3시" → 비효과적)
- 그러나 모든 조합을 하나의 action으로 모델링하면 action space 폭발 (각 dimension에 5개 arm만 있어도 $5^4 = 625$개)
- **실용적 트레이드오프**로 독립 가정 채택

### Wolpertinger Architecture의 역할
- 원래 RL에서 large discrete action space 문제를 다루기 위해 설계된 구조
- 마케터가 미리 만들어 놓은 **메시지 카탈로그**(실현 가능한 메시지 pool)가 존재
- TS가 각 dimension에서 독립 선택한 "이상적 조합"을 카탈로그에서 가장 가까운 메시지에 매칭
- 카탈로그에는 마케터가 검수한 합리적인 메시지만 존재 → **feasibility constraint** 역할
- 즉, 독립 가정으로 인한 비합리적 조합이 나와도 카탈로그 매칭 단계에서 자연스럽게 걸러짐

### 예시

**Step 1: 마케터가 메시지 카탈로그를 미리 만들어 놓음**
```
메시지 A: tone=친근, emoji=O, value_prop=할인, 시간=오후, 빈도=주2회, 채널=push
메시지 B: tone=공식적, emoji=X, value_prop=신기능, 시간=아침, 빈도=주1회, 채널=email
메시지 C: ...
```

**Step 2: TS가 각 dimension에서 독립적으로 arm 선택**
```
when → 오후
how often → 주2회
where → push
what(tone) → 친근, what(emoji) → O, what(value_prop) → 할인
```

**Step 3: 이 "이상적 조합"을 카탈로그에서 매칭**
- 카탈로그의 모든 메시지와 비교 → 메시지 A가 가장 가까움 → 메시지 A 발송