### Methodology
#### Ranking model
- 홈피드 랭킹 모델
  - point-wise multi-task learning 아키텍처
  - cross entorypy loss (multi-label classification)
  - next-action prediction 을 auxiliary task 로 진행

#### TransAct V2: Transformer for Lifelong User Action Sequence
- real-time sequence 는 유저의 short-term 액션에 대한 예측을 하는 경향이 있고 relevance 측면에서는 좋지만 유저의 long-term 은 고려하지 않는다. 그러다 보면 diversity 가 부족해진다. 그러면 retention 과 장기적 관점에서 유저 경험이 나빠진다. 따라서 user long-term interest 도 함께 고려해야한다.
- Lifelong User Sequence Features (LL)
  - 1년 이상 긴시간 동안 user interaction (explicit user action)
  - sequence 길이는 90th percentile of users' past 2 years of action history lengths weighted by visiting frequency
  - LL sequence token 은 4개의 feature (real-time, impression sequence 도 동일)
    - 액션 시간
    - 액션 타입 (multi-hot vector, 같은 Pin 에 대한 interaction 이 여러번인 경우)
    - 액션 발생 장소 (홈피드, 검색 등)
    - 32dim PinSage embedding (pin embedding)
      - affine quantization to convert the original 32-dimensional fp16 PinSage embedding into a 32-dimensional int8 vector
- Modeling Lifelong User Sequence (nearest neighbor search)
  - candidate item 을 anchor 로 하여 3가지 sequence 를 NN 한다.
    - lifelong, real-time, impression
    - NN 은 candidate PinSage embedding $e_c$ 와 sequence 들의 Pinsage embedding 간의 유사도로 진행
  - 추가로 candidate item 과 상관없이 항상 최근 액션 $S_{RT}[:r]$ 를 유지한다.
- Feature encoding
  - user sequence 의 데이터 (액션 타입, 발생 장소) 는 trainable embedding table 을 사용한다.
  - positional encoding 은 learnable parameter 이다
  - 따라서 final encoded user sequence 는 $F=CONCAT(E_{PinSage}(S_{all}), e_c) + E_{act} + E_{surf} + E_{pos} \in R^{|S| * d}$
- Transformer Encoder
  - single attention head, transformer encoder 2 layer, 차원은 64 차원이다.
  - ff 는 32차원이다.
  - output 은 2개의 downstream task 에서 사용된다.
    - linear layer + max pooling layer -> multi-head prediction
    - next action prediction

#### Next Action Loss
- user 의 next action prediction 을 auxiliary task 로 하니 랭킹 성능이 좋아졌다.
- next action prediciton task
  - 최근 유저 액션 sequence 를 이용하여 다음 pin 에 대해 긍부정 분류, contrastive loss manner
  - sampled softmax loss function
- key modeling design of NAL
  - negative sample 2가지 실험했는데 후자가 랭킹 효과가 좋았다.
    - in-batch random negative sampling
    - impression-based negative sampling (유저에게 노출되었으나 액션이 없는)
#### Serving and Logging System Design
- cost
  - $L$: length of $S_{LL}$, $N$: average number of items per ranking request
  - feature storage cost $O(L)$
    - $O(10^4)$ 값
  - network cost $O(NL)$
##### data pipelines
- 랭킹모델의 모든 피처는 서빙시 캐시되고 훈련데이터로 로깅한다. 하지만 $LL$ 이 너무 길어서 caching 과 network transfer 에서 비효율적이다.
- 그래서 NN feature loggging strategy 를 썼다. 전체 sequence 가 아니라 relevant NN feature 만 훈련데이터에 로깅한다.
- training & serving
  - 훈련시는 logged data 에서 NN 을 가져오고 서빙시에는 full LL sequence 를 가져온다.
##### serving optimizations
- c++, cuda, Triton
- request level de-duplication
- fused sequence dequantization
- single kernel unified trasformer
- pinned memory arena