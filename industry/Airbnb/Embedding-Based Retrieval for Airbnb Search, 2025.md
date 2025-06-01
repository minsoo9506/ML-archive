- The goal of Airbnb Search is to surface the most relevant listings for each user’s query

### Training Data Construction
- contrastive learning 으로 학습했다.
- 데이터는 query, home pair 이다.
- 유저는 최종 예약 전까지 다양한 액션을 한다. (home 정보 클릭, 리뷰 클릭, 장바구니 등등)
- 최종 예약 home 을 positive label 로 사용하고 negative label 은 유저들이 검색과정에서 액션이 있었는데 예약까지 못한 home 으로 사용한다.

### Model Architecture
- 사용한 모델은 two-tower 모델이다.
  - (home) listing, query tower
- 각 tower 에서 사용한 feature 는
  - listing: historical engagement, amenities, and guest capacity etc
  - query: geographic search location, number of guests, and length of stay etc
- listing embedding 은 offline daily batch 로 생성한다. 이를 통해 online latency 를 줄일 수 있다.
- query embedding 만 real-time 으로 계산한다.

### Online Serving
- 다양한 ANN 알고리즘에서 2개로 후보군: inverted file index (IVF) and hierarchical navigable small worlds (HNSW)
  - IVF 가 속도는 좀 더 좋고 성능은 HNSW 가 좀 더 좋다.
- Airbnb home 은 데이터 업데이트가 잦고 검색이 다양한 filter 가 있다보니 IVF 가 좋았다.
- IVF 알고리즘에서는 cluster 를 이용하는데 이 때 cluster size uniformity 에 retrieval 결과가 영향을 준다.
  - dot product 보다 euclidean 이 even 했다고 한다.