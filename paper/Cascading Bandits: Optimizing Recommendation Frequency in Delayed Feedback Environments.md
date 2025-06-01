- individual messages from a selected list are sent to a user periodically
- We first show a dynamic programming solution to finding the optimal message sequence in deterministic scenarios, in which the reward is allowed to vary with different messages

### background
- cascading bandit
  - sending a sequence of messages to individual users 인 상황에서 많이 사용됨
  - 근데 이들의 주요 가정이 immediate user feedback 이고 frequency 에 대한 고려가 없다. 유저에게 1번 message 를 보내는 것도 아닌데.
- 논문에서는 기존 cascading 방법과 다른 5가지 차이점을 제시한다.
  - 발송 메시지의 순서 고려
  - 발송 메시지 갯수도 유동적으로
  - 각 메시지 발송이 유저 별로 차이가 있으니 초반 발송 정보를 이용하여 후반 발송에 이용
  - 메시지 클릭 후 reward 는 다양
  - delayed feedback 고려

### problem formulation
- $N$ 개의 message 존재
- $\bold{S} = (S_1, S_2, ..., S_m)$ 은 message list
  - $\kappa(i)=j$: $i$ 위치를 $j$ message 로 mapping 하는 index function
  - $\theta(i)=j$: 위의 inverse 
- 유저는 message list 를 주기적으로 받는다 (or 스스로 업데이트)
- message list 를 받고 관심있는 것을 클릭한다. 이 때, $m$ 개의 list 안에서 다음 message 를 확인할 확률을 $q(m)$, 떠날 확률은 $1-q(m)$ 이라고 한다.
  - assume $q(m)$ 은 non-increasing function with $m$
- message $i$ 의 attraction probability 를 $v_i$ 라고 한다. 이 때 유저가 클릭해서 return 을 $R_i$ 라고 한다.
- 유저는 time $t$ 에 도착한다. ($t=1,...,T$)
- 각 유저마다 고정된 time window $D$ 가 있고 이 기간동안 message 들이 보내진다. message 는 해당 time window 에서 순서대로 $m/D$ 의 빈도로 보내진다.
- 유저는 한번에 1개의 message 를 받는다.

- $m$ 개의 message list 가 있다. 이 때, message $i$ 를 확인할 확률은 $w_i(m)$ 이다.

$$w_i(m) = \prod_{j=1}^{\theta(i) - 1}((1 - v_{\kappa(j)})q(m))$$

- 이는 $i$ 번째 message 를 보기전까지 유저가 클릭을 하지 않았으며 system 에 머물러 있음을 의미한다.
- learning agent a list of messages with appropriate dissemination frequency 를 정하기 위해서는 $E[U(\bold{S}, \bold{v}, \bold{R}, q(m))]$ 를 최대화하는 optimization 문제를 풀어야한다.

논문에서는 optimal sequence $\bold{S}$ 를 찾는 과정을 증명한다. 정리는 하지 않기로 한다.

### learning with delayed feedback
- An online learning algorithm for cascading bandits with delayed feedback 방법 제안

### personalized recommendation

### numerical experiment