### Introduction
- realtime-batch hybrid ranking approach that combines both realtime user action signals and batch user representations
  - realtime 은 추천에서 당연히 중요하다.
  - longer user action sequence 도 중요하지만 컴퓨팅 리소스가 많이들고 latency 도 증가하는 단점이 존재한다.
    - utilized hashing and nearest neighbor search in long user sequences 
    - encodes users’ past actions over an extended time frame to a user embedding to represent longterm user interest
    - ...

### Methodology
####  Pinnability: Pinterest Homefeed ranking model
- 아키텍처
  - high level 아키텍처는 Wide and Deep learning model 이다.
  - embedding layers to project categorical features to dense features, and perform batch normalization on numerical features.
  - realtime-batch hybrid 형태로 encodes the user action history (TransAct, PinnerFormer)
  - feature cross using a full-rank DCN V2 to explicitly model feature interactions
  - 마지막은 fully conntected layer 이고 각 output 은 cand pin 에 대해 0~1 확률
- loss function: weighted cross-entropy, multi-label classificaiton
- 특정 비즈니스 니즈에 따라 training example 에 대해 user-dependent (state x locaiton x gender) weight 사용

#### Realtime User Action Sequence Features
- 실시간 생각해서 유저 pin 에 대한 최근 액션 100개 사용, 없으면 0으로 padding, 최근 액션이 앞에 위치
  - 시간, 액션 타입, 32dim PinSage embedding (pin 에 대한 임베딩)

#### TransAct
- feature encoding
  - pin 에 대한 action 이 다양하고 이를 고려하기 위해 trainable embedding table 로 action 을 low dim vector 화 한다.
  - 따라서 final encoded user action sequence feature 는 concat(W_actions, W_pins) $\in R^{|S| * (d_{pinsage} + d_{action})}$
- early fusion
  - 추천 모델 앞부분에서 user, item feature 를 merge 하는 것을 의미한다.
  - 이게 ranking performance 에서 좋았고 concat 방법 사용한다.
  - resulting sequence feature with early fusion $U \in R^{|S| * d}$ where $d = (d_{action} + 2 d_{pinsage})$
- sequence aggregation model
  - transformer-based 아키텍처를 사용한다. (transformer encoder layer 2 and 1 head)
  - positional encoding 은 미사용한다. (오프라인 실험해보니 효과가 별로)
- random time window mask
  - 모든 최근 액션 시쿼스를 사용하면 모델이 최근 유저의 액션과 유사한 콘텐츠만 추천하는 문제가 발생한다. (rabbit hole effect) -> diversity 부족
  - 그래서 self-attention 이전에 transformer encoder 에서 time window mask 사용한다.
  - 각 forward pass 마다 0~24 사이를 time window random sample 해서 최근 action masking (training 만 사용, inference 시는 미사용)
- transformer output compression
  - transformer encoder 의 output matrix 는 $O = (o_0 : o_{|S|-1}) \in R^{|S| * d}$ 이다.
  - 이때 앞부분 K 개의 스퀀스 $(o_0 : o_{K-1})$ 를 뽑아서 유저의 최근 interest 를 구한다.
  - 전체 output maxtrix 에 대해 max pooling 하여 $R^d$ 차원의 max pooling vector 를 구한다. 이는 유저의 long term interest 정보를 담고 있다.
  - 따라서 최종적으로는 $R^{(K+1) * d}$ 의 vector 를 만들어서 DCN v2 의 feature crossing layer 에서 사용된다.

#### Model productionizatin
- retraining
  - Pinnability 는 일주일에 2번 훈련
- gpu serving (CUDA 라서 어렵다)
  - Fuse CUDA kernels
  - Combine memory copies
  - Form larger batches
  - Utilize CUDA graphs
- realtime feature processing
  - Flink, Kafka, 자체 feature store 사용
  - 서빙시 유저 request 가 발생하면 processor 가 모델에 맞게 convert 진행

#### 그외 내용들도 유익함