## introduction
### query recommendations
- 검색할때 실시간으로 연관검색어, 자동완성 뜨는 거
- 전통적인 방법들과 다르게 prompt-based LLM 으로 생성하는 접근

## related work
### query recommendations
- query log 를 이용하여 ML 방법들은
  - still require access to a set of high-quality query logs, which are expensive to collect, maintain, and might not be available or be representative in a cold-start situation
- query log 가 없는 경우
  - query 와 관련있는 document 데이터를 이용하여 유저가 작성한 query 다음에 나올 단어 예측
- LLM
  - fine tune 없이도 좋은 성능
  - prompting in information retrieval
  - IR 분야에서도 다양하게 LLM 사용됨

## proposed approach
- 그냥 prompt 만 사용한다. few-shot prompt 사용.
- To sum up, this approach completely eradicates the cold-start issues, it demonstrates that there is no need for large collections of documents, to generate query recommendations, and the need to construct a query-specific architecture
experimental setup

## results and analysis
- RQ1: 우리가 제안한 GQR 시스템이 기존의 쿼리 추천 시스템과 비교하여 관련성 있고 유용한 쿼리 추천을 생성할 수 있는가?
- RQ2: 우리 GQR 시스템이 생성한 쿼리가 다른 시스템이 생성한 쿼리보다 사용자에게 더 매력적인가?
- RQ3: 우리 GQR 시스템이 드문 쿼리, 즉 롱테일 쿼리에 대한 추천을 생성하는가?
  - 새로운 방법이 좋다.
- RQ4: 쿼리 로그가 생성적 쿼리 추천에서 여전히 가치를 제공하는가?
  - 그렇다고 한다 그냥 GQR 이 아니라 retrieval 기능을 넣은 RA-GQR 이 더 좋았다고 한다.
- RA-GQR?
  - prompt 를 dynamic 하게 만든다.
  - 먼저 input query 가 주어지면 pre-trained embedding model, FAISS 를 통해 유사한 과거 유저들 query 들을 가져와서 prompt 에 추가한다.