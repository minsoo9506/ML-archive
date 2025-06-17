- social search: 검색 시 의도가 query text 뿐만 아니라 유저의 다양한 context 에도 영향을 받는다.

### Model
- search retrieval 을 recall optimization 으로 생각했다. recall@k 를 최대한 높이는 것.
- embedding model 훈련시 recall object 에 대한 approximate 로 triplet loss 를 사용했다.
- 모델 구조는 two tower 형태로 query, doc encoder 를 사용했다. 각 feature 는 text 뿐만 아니라, 위치, 유저 context 등의 다양한 정보를 사용했다.

#### evaluation metrics
- averaged recall@k over 10000 search sessions

#### loss function
- triplet loss
  - query, pos doc, neg doc
  - random 샘플을 사용하여 triplet loss 를 위한 negative 쌍을 형성하면, recall optimization 에 근사할 수 있다.
  - 훈련 데이터에서 각 pos 샘플에 대해 n개의 neg 샘플을 샘플링하면, 모델은 후보 풀 크기가 n일 때 top 1 position recall 을 최적화하게 된다.
  - 실제 제공하는 후보 풀 크기가 N이라고 가정할 때, 우리는 대략적으로 top K ≈ N/n  recall 을 최적화하고 있다.
- contrastive learning

#### unified embedding model
- two tower model 후 임베딩 사용
- cosine distance
- categorical feature 들은 embedding look-up layer 를 통해 encoder 에 들어간다.
  - multi-hot vector 의 경우는 avg 한다.

#### training data mining
- positive: click
- negative: random sample
- random sample 만 사용했을 때보다 non-click impression 을 neg 로 사용했을때 성능이 낮았다.

#### feature engineering
- text 만 썼을 때보다 다른 다양한 feature 더 쓰는게 성능에 좋았다.
- text feature
  - charactor n-gram: word n-gram 보다 vocab size 도 작고 subword 가 oov 문제에 효과적이다.
  - 단순 boolean term matching 보다 fuzzy text match, optionalization 방법 사용
- location feature
  - query, doc 둘 다에서 사용
    - query side, we extracted searcher city, region, country, and language
    - document side, we added publicly available information, such as explicit group location tagged by the admin
- social embedding feature
  - facebook 의 social graph 를 최대한 활용하기 위해 social graph 에 기반하여 user, entiry 들 embedding model 의 결과를 사용했다. (따로 embedding 을 만들었다는 뜻)

#### serving
- ANN: Faiss 사용
- System Implementation
  - 서빙 관련 embedding quantizatio 등 다양한 설명
  - query embedding 은 real-time inference 하고 doc embedding 은 offline 으로 저장해둔거 사용
- query and index selection
  - 효율성을 위해 일부 query 에서는 EBR 하지 않고 그냥 결과는 보내는 경우도 있음

#### later-stage optimization
- facebook search ranking 은 multi stage 이고 더 좋은 성능을 위해 2가지 방법
  - retrieval 에서 사용한 embedding 정보를 ranking feature 로도 사용, embedding 자체보다 그냥 similarity 만 해도 좋았다.
  - EBR 로 recall 은 높이지만 term matching 에 비해 precision 이 낮다. 이를 보완하기 위해 human rater 들의 retrieval 결과를 평가하는 loop 를 만들고 훈련에 사용했다.

#### advanced topics
- hard mining → 성능 향상이 있었다.
  - hard negative mining
  - query 가 동일하면 결과가 비슷 → 다른 context feature 를 잘 학습 못한듯 → negative 가 너무 쉬운듯
  - online hard negative mining
    - 각 batch 에서 dynamic 하게 hard negative 선택
    - batch 내 doc 중에서 similarity 가 높은 doc 을 hard negative 로 선택
- offline hard negative mining
  - 각 query 에 대해 topk 결과 뽑기 → 그중에서 hard selection strategy 로 hard negative 뽑기 → 재학습
- hard selection strategy insight
  - hardest example 이 best 는 아니고 101-500 정도가 좋다.
  - easy negative 도 필요하다. → random, hard 를 섞어서 훈련시켰고 100:1 비율이 좋았다.
- hard positive mining
  - production 에서 failed search session 의 결과를 이용했다.
- embedding ensemble
  - we need hard examples to improve model precision, but easy example are also important to represent the retrieval space
  - 그래서 multi-stage 로 접근한다.
  - weighted concatenation
    - 여러개의 모델의 임베딩을 weight(evalset의 성능)ed concatenation
    - recall 성능 향상할 수 있었다.
  - cascade model
    - 이전 모델의 output 을 순차적으로 사용
    - unified embedding model 단독 사용보다 text embedding 으로 candidate 를 pre-select 하고 unified embedding - model 로 re-rank 한 결과가 text matching precision 측면에서 좋았다.
