- 검색하면 list-results, map-results 가 노출된다.
- list-results 는 순서대로 노출이되고 CTR 도 상위 랭크일수록 높다.
- 하지만 map-results 는 지도에 한번에 노출된다. (no decay of user attention by ranking position)
- 지도에서 유저가 클릭한 pin 의 갯수도 생각보다 많지 않다. 그렇다면 좋은 추천 장소를 낭비하고 있다고 할 수 있다.

### Uniform User Attention
- 기존 대비 실험을 해보니 제한적으로 topk 를 노출할수록 후기별점이 더 좋았고 취소율도 줄었다. 또한, 탐색을 위한 클릭수도 줄었다.

### Tiered User Attention
- 상위 topk 는 pin 을 크게, 하위는 작게 노출하여 실험했다.
- 마찬가지의 결과를 얻었다.

### Discounted User Attention
- 마지막으로는 화면에서 CTR 을 통해 user attention 이 어떤 위치에서 높고 낮은지 확인했고 중앙일수록 높았다.
- 그래서 re-center 방법으로 노출 위치를 상위 topk 위주로 했더니 역시나 비슷한 결과를 얻었다.

### Conclution
- 다양한 실험을 통해 노출 랭킹이 없는 경우에도 유저 만족도를 높이는 접근이 재밌었다.