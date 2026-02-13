**Matryoshka Representation Learning (MRL)**은 2022년 NeurIPS에서 발표된 논문으로, 하나의 임베딩(Embedding) 벡터 안에 여러 크기의 작은 벡터들을 중첩된(Nested) 형태로 학습시키는 기법입니다.

러시아 인형 '마트료시카'에서 이름을 따온 것처럼, 큰 인형(큰 차원 벡터) 안에 작은 인형(작은 차원 벡터)들이 차례로 들어있는 구조를 가집니다.

---

## 1. 핵심 개념: 왜 필요한가?

일반적인 딥러닝 모델은 고정된 차원(예: 768차원, 1024차원)의 임베딩을 생성합니다.

* **문제점:** 저사양 환경(모바일 등)에서는 고차원 벡터를 계산하고 저장하는 것이 부담스럽고, 반대로 고사양 환경에서는 더 정밀한 벡터가 필요할 수 있습니다.
* **MRL의 해결책:** 하나의 모델이 **가변적인 차원**을 가질 수 있게 합니다. 예를 들어 1024차원 벡터를 뽑은 뒤, 앞부분의 64차원만 잘라 써도 충분히 정확한 결과를 얻을 수 있도록 설계합니다.

해당 논문에서는 2가지 application 에 초점을 맞췄습니다.
- large-scale classification
  - 작은 임베딩으로도 동일한 정확도, 작은 임베딩이라 빠르고 리소스 효율적 
- retrieval
  - 작은 임베딩으로 retrieval 하고 큰 임베딩으로 ranking

## 2. 작동 원리 (Mechanism)

MRL은 학습 시점에 **중첩된 손실 함수(Nested Loss)**를 사용합니다.

* **Multi-scale Loss:** 모델이 출력하는 전체 벡터 에 대해서만 학습하는 것이 아니라, 미리 정해둔 여러 차원(예: 8, 16, 32, ..., 1024)의 **접두사(Prefix)**들에 대해 각각 손실 함수를 계산합니다.
* **Total Loss:** 각 차원별 손실값을 모두 더해 최종 손실로 사용합니다.
* **결과:** 이 과정을 통해 벡터의 **앞부분(낮은 차원)**에는 데이터의 가장 중요한 **핵심 정보(Coarse-grained)**가 담기고, **뒷부분**으로 갈수록 **세부 정보(Fine-grained)**가 추가되는 구조가 완성됩니다.

## 3. 주요 장점 및 성과

1. **유연한 배포 (Adaptive Deployment):** 환경에 따라 임베딩 사이즈를 자유롭게 조절할 수 있습니다. (예: 검색 시에는 64차원으로 후보군을 빠르게 추리고, 상위 결과만 1024차원으로 정밀하게 재정렬)
2. **효율성:** 논문에 따르면 ImageNet-1K 분류 작업에서 원래 크기의 **1/14 수준인 차원**만 사용해도 정확도 손실이 거의 없었으며, 실제 검색 속도를 최대 **14배**까지 향상시켰습니다.
3. **성능 유지:** 개별 차원별로 따로 학습시킨 모델들과 비교해도 성능이 뒤처지지 않거나 오히려 더 뛰어난 정확도를 보여주었습니다.

## 4. 실제 활용 예시

최근 **OpenAI의 신규 임베딩 모델(text-embedding-3-small/large)**이 이 MRL 기술을 채택하여 화제가 되었습니다. 사용자는 API 호출 시 `dimensions` 파라미터를 조절하여 성능과 비용 사이의 최적점을 직접 선택할 수 있습니다.

## 5. Loss Function 상세

### 기본 세팅

모델 $F(\cdot; \theta)$가 입력 $x$에 대해 $d$차원 임베딩 벡터 $z = F(x; \theta) \in \mathbb{R}^d$를 출력한다고 하자.

MRL에서는 **중첩 차원 집합(nesting dimensions)**을 미리 정의한다:

$$\mathcal{M} = \{m_1, m_2, \ldots, m_L\} \quad \text{where } m_1 < m_2 < \cdots < m_L = d$$

예: $\mathcal{M} = \{8, 16, 32, 64, 128, 256, 512, 1024\}$

각 $m \in \mathcal{M}$에 대해 벡터의 **처음 $m$개 차원(prefix)**을 잘라낸 것을 $z_{1:m}$으로 표기한다.

### MRL Loss

MRL의 전체 loss는 각 차원별 loss의 **가중합**이다:

$$\mathcal{L}_{\text{MRL}}(\theta) = \sum_{m \in \mathcal{M}} c_m \cdot \mathcal{L}\big(f_m(z_{1:m}); \, y\big)$$

- $z_{1:m}$: 전체 임베딩 $z$의 **앞 $m$차원 prefix**
- $f_m$: 차원 $m$에 대응하는 **linear classifier** (또는 projection head). 각 granularity마다 별도의 head를 둠
- $\mathcal{L}$: task에 맞는 **기본 loss** (classification이면 softmax cross-entropy, retrieval이면 contrastive loss 등)
- $c_m$: 각 차원별 **가중치** (논문에서는 모두 동일하게 $c_m = 1$로 설정해도 충분히 잘 동작한다고 보고)
- $y$: ground-truth label

### 학습 흐름

```
전체 임베딩 z (d=1024)
├── z[1:8]    → f_8  → Loss_8   (가장 coarse한 정보)
├── z[1:16]   → f_16 → Loss_16
├── z[1:32]   → f_32 → Loss_32
├── z[1:64]   → f_64 → Loss_64
├── z[1:128]  → f_128 → Loss_128
├── z[1:256]  → f_256 → Loss_256
├── z[1:512]  → f_512 → Loss_512
└── z[1:1024] → f_1024 → Loss_1024 (가장 fine-grained)

Total Loss = Loss_8 + Loss_16 + ... + Loss_1024
```

**핵심 포인트:**
1. **Backbone은 공유**: 모든 차원에서 동일한 encoder $F$의 gradient가 흘러들어옴
2. **Linear head만 별도**: $f_m$은 학습 시에만 사용하고, 추론 시에는 버림 (임베딩만 사용)
3. **Prefix 구조**: $z_{1:8} \subset z_{1:16} \subset \cdots \subset z_{1:1024}$이므로, 앞쪽 차원은 **모든 scale의 loss에 의해 동시에 최적화**됨 → 자연스럽게 앞쪽에 핵심 정보가 집중

### 왜 잘 동작하는가?

- 앞쪽 차원(예: 1~8)은 **모든** $m \in \mathcal{M}$의 loss에 포함되므로, 가장 많은 gradient signal을 받는다. 따라서 가장 discriminative한 정보를 인코딩하도록 강하게 압박받는다.
- 뒤쪽 차원은 큰 $m$에서만 loss에 포함되므로, 앞쪽이 놓친 **잔여(residual) 정보**를 보충하는 역할을 한다.
- 결과적으로 **정보의 중요도 순서가 차원 순서에 자연스럽게 정렬**된다.

### 추론 시

학습이 끝나면 $f_m$ head는 모두 버리고, 상황에 맞게 원하는 차원만 잘라서 사용한다:

```python
embedding = model.encode(x)        # [1024]
small_emb = embedding[:64]         # 모바일/빠른 검색용
large_emb = embedding[:512]        # 정밀 비교용
full_emb  = embedding              # 최고 성능용
```

## 6. PyTorch 구현 예시

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MRL(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes, nesting_dims=(8, 16, 32, 64, 128, 256)):
        super().__init__()
        self.backbone = backbone              # 임의의 encoder (ResNet, ViT 등)
        self.nesting_dims = nesting_dims      # M = {8, 16, 32, ..., 256}

        # 각 차원별 linear classifier (학습 시에만 사용)
        self.heads = nn.ModuleDict({
            str(m): nn.Linear(m, num_classes) for m in nesting_dims
        })

    def forward(self, x):
        z = self.backbone(x)  # [B, embed_dim] 전체 임베딩

        logits = {}
        for m in self.nesting_dims:
            prefix = z[:, :m]             # 앞 m차원만 잘라냄
            logits[m] = self.heads[str(m)](prefix)

        return z, logits


def mrl_loss(logits, targets):
    """각 차원별 CE loss의 합"""
    total = 0
    for m, logit in logits.items():
        total += F.cross_entropy(logit, targets)
    return total


# --- 사용 예시 ---
if __name__ == "__main__":
    embed_dim = 256
    num_classes = 100
    nesting_dims = (8, 16, 32, 64, 128, 256)

    backbone = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, embed_dim),
    )

    model = MRL(backbone, embed_dim, num_classes, nesting_dims)

    # 학습
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, num_classes, (4,))

    z, logits = model(x)
    loss = mrl_loss(logits, y)
    loss.backward()

    print(f"loss: {loss.item():.4f}")
    print(f"full embedding: {z.shape}")        # [4, 256]
    print(f"prefix 8-dim:   {z[:, :8].shape}") # [4, 8]  ← 이것만 써도 OK
```