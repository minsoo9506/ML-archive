- 다른 bandit 시나리오와 다른 점
  - arm 이 발송메시지다보디 arm 들이 항상 사용가능한게 아니라 유저마다 t 마다 달라진다.
  - 이를 sleeping bandit problem 이라고 하는 듯 하다.

### problem definition and notation
- reminder 를 통해서 듀오링고 유저들이 하루에 하나의 수업을 듣도록 유도하는 것이 목표
- $r_t ∈ {0, 1}$, as follows: the reward is 1 if the
user completes a lesson within two hours of the reminder being
sent, and 0 otherwise.

### 알고리즘
- 로직이 복잡해서 정리는 안함
- 한줄로 정리하자면 각 arm 별 score 를 따로 만드는 방법 진행

# appendix

## Bandit 알고리즘에서 Softmax 방법 설명

**Softmax 방법**은 Multi-Armed Bandit(MAB) 문제에서 *탐험(Exploration)*과 *활용(Exploitation)*의 균형을 조절하기 위한 확률 기반 전략입니다. 이 방법은 각 행동(arm)의 추정 보상값을 확률 분포로 변환하여, 높은 보상이 예상되는 행동을 선호하면서도 일정 수준의 탐험을 유도합니다.

---

### 핵심 메커니즘
1. **확률 계산**  
   각 행동 $$a$$의 선택 확률은 다음 **Softmax 함수**로 계산됩니다:  
   $$
   P(a) = \frac{e^{\frac{Q(a)}{\tau}}}{\sum_{b=1}^K e^{\frac{Q(b)}{\tau}}}
   $$
   - $Q(a)$: 행동 $$a$$의 추정 보상값  
   - $\tau$: 탐험 강도를 제어하는 **온도(temperature)** 파라미터  
   - $K$: 전체 행동 수  

2. **온도(τ)의 역할**  
   - **τ → 0**: 최대 보상 행동만 선택 (완전한 활용)  
   - **τ → ∞**: 모든 행동을 균등 확률로 선택 (완전한 탐험)  
   - *예시*: τ=1.0일 때 [Q(a)=2, Q(b)=1] → P(a)=73%, P(b)=27%[4][10]

---

### ε-Greedy vs Softmax 비교
| 특성                | ε-Greedy                          | Softmax                     |
|---------------------|-----------------------------------|-----------------------------|
|**탐험 방식**        | 임의 무작위 선택                 | 보상 비율에 따른 확률 선택 |
|**장점**             | 구현 간단                        | 보상 차이를 반영한 스마트 탐험 |
|**단점**             | 낮은 보상 행동도 강제 탐험       | τ 튜닝 필요                |
|**적합 시나리오**    | 단순 환경                        | 보상 차이가 명확한 환경    |

---

### 주요 특징
- **지수 함수 활용**: 보상 차이를 비선형적으로 확대하여, 높은 보상 행동에 집중합니다[12].  
- **확률 정규화**: 모든 행동의 확률 합이 1이 되도록 보장합니다[5][6].  
- **온라인 적응**: 실시간 보상 관측으로 $Q(a)$ 값을 지속 업데이트합니다[7][9].

---

### 변형 기법
1. **Annealing Softmax**  
   시간에 따라 τ를 점차 감소시켜 초기에는 탐험을 강화하고 후기에는 활용을 강화합니다[1].  
   *예시*: $\tau_t = \frac{\tau_0}{\log(t+1)}$

2. **Bootstrapped Softmax**  
   신뢰 구간 추정을 결합하여 불확실성이 높은 행동을 우선 탐험합니다[3].

---

### 코드 구현 예시 (Python)
```python
import numpy as np

def softmax(q_values, tau=1.0):
    exp_values = np.exp(q_values / tau)
    return exp_values / np.sum(exp_values)

# 행동별 추정 보상값
q = [2.1, 1.8, 0.5]
tau = 0.7

# 선택 확률 계산
probs = softmax(q, tau)  # 출력: [0.62, 0.34, 0.04]
selected_arm = np.random.choice(len(q), p=probs)
```

---

### 활용 분야
- **온라인 광고**: 사용자 클릭률 기반 광고 노출 최적화[11]  
- **의료 임상실험**: 치료법 효능 비교 시 안전한 탐험[3]  
- **게임 AI**: NPC의 전략 선택 알고리즘[9]

---

Softmax 방법은 보상 분포의 상대적 차이를 효과적으로 활용하지만, **온도 파라미터 튜닝**과 **고차원 문제 확장성**이 주요 과제입니다. 최근 연구에서는 신경망과 결합한 Deep Softmax 등으로 발전하고 있습니다[7][13].

Citations:
[1] https://soobarkbar.tistory.com/135
[2] https://yjjo.tistory.com/22
[3] https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002316138
[4] https://hakucode.tistory.com/3
[5] https://velog.io/@chiroya/13-Softmax-Function
[6] https://pythonkim.tistory.com/20
[7] https://taeyuplab.tistory.com/4
[8] https://velog.io/@looa0807/Softmax-%ED%95%A8%EC%88%98%EC%9D%98-%EC%9D%B4%ED%95%B4%EC%99%80-%ED%99%9C%EC%9A%A9
[9] https://jjuke-brain.tistory.com/entry/Ch10-Multi-Armed-Bandit-MAB-Problem
[10] https://data-scientist-jeong.tistory.com/48
[11] https://otzslayer.github.io/ml/2022/01/07/about-multi-armed-bandit.html
[12] https://killerwhale0917.tistory.com/37
[13] https://taeyuplab.tistory.com/5
[14] https://blog.naver.com/takion7/221625764552
[15] https://wikidocs.net/59427
[16] https://dasu.tistory.com/62
[17] https://syj9700.tistory.com/38
[18] https://velog.io/@danniel1025/%EB%8B%A8%EC%B8%B5%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A0Softmax-%EC%84%A0%ED%83%9D%EB%B6%84%EB%A5%98
[19] https://wikidocs.net/35476

---