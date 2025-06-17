- inverted index-based lexical matching falls short in several key aspects
- 단어의 일반화, 동음어, 반의어에 대한 이해 부족
- 형태 변형에 취약 (woman vs women)
- 오타에 민감
- 그러니까 유저 행동 데이터로 모델 만들어서 검색하자.
- 유저 행동 데이터에는 semantic information 이 자연스레 들어있다.

### Model
- 모델 구조

- loss function
  - hinge loss
  - contrastive learning
  - 처음에는 2part hinge loss 사용했더니 아래처럼 유사도 분포가 겹치고 negative 에서 two peak 가 보임
    - 검색 후 구매(positive), 랜덤(negative)
  - 그래서 3part hinge loss 로 하니 아래처럼 잘 구분되었다.

- tokenization method
  - 다양한 방법을 함께 사용
  - 다양한 토크나이징 후 각 임베딩을 weighted sum 했다고 한다.
  - handling unseen word
    - hashing trick 을 사용
    - 일반적으로 oov 는 그냥 동일한 unk token embedding 을 사용하는데 그게 아니라 hash function 을 이용하여 어느 정도 다른 embedding 을 사용하게 하는듯
    - vocab size 에 5-10배 크기의 bin size 사용

### data
- purchased, impressed but not purchased, random
- 각 query 에 대해 product 는 1, 6, 7 비율로 훈련


### experiments
- metrics
  - matching: recall@100, MAP
  - ranking: NDCG, MRR
- 임베딩은 end-to-end 로 훈련했다고 한다.

### training acceleration
- 관련 팁들 설명