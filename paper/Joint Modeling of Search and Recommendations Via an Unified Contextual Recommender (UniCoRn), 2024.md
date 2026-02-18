# UniCoRn: Joint Modeling of Search and Recommendations (RecSys 2024)

**저자:** Moumita Bhattacharya, Vito Ostuni, Sudarshan Lamkhede (Netflix)

**링크:** [arXiv](https://arxiv.org/abs/2408.10394) | [ACM RecSys 2024](https://dl.acm.org/doi/10.1145/3640457.3688034)

---

## 핵심 동기

검색(Search)과 추천(Recommendation) 시스템은 보통 **별도의 모델**로 개발·운영되는데, 이로 인해 유지보수 복잡도와 기술 부채(technical debt)가 증가한다. 이 논문은 검색과 추천을 **"같은 동전의 양면"**으로 보고, 하나의 통합 모델로 두 태스크를 동시에 처리하는 방법을 제안한다.

---

## 문제 정의 (Unified Formulation)

검색과 추천의 근본적 차이는 **컨텍스트 입력**에 있다:
- **검색**: 명시적 텍스트 쿼리 기반
- **추천**: 사용자 컨텍스트 / 소스 아이템 기반

UniCoRn은 이를 **공유 컨텍스트 정의**로 통합한다:
- User ID, Query, Country, Source Entity ID, Task Type
- 출력: 특정 엔티티에 대한 **긍정적 engagement 확률 점수**

---

## 모델 아키텍처

| 구성 요소 | 설명 |
|-----------|------|
| **Feature** | Context-specific (쿼리 길이, 소스 엔티티 임베딩) + Context-Target 결합 (특정 쿼리에 대한 타겟 클릭 수 등) |
| **Embedding** | 범주형 피처에 대한 학습 가능한 임베딩 레이어 |
| **네트워크** | Residual connection + Feature crossing |
| **Loss** | Binary Cross-Entropy |
| **Optimizer** | Adam |

### Context Imputation 전략 (핵심 기법)

누락된 컨텍스트를 태스크별로 채워 cross-task learning을 강화한다:
- **검색 태스크**에서 소스 엔티티가 없을 때 → null 값 대입
- **추천 태스크**에서 쿼리가 없을 때 → 소스 엔티티의 **display name 토큰**으로 대체

이를 통해 missing context를 줄이고, 태스크 간 학습 효과를 극대화한다.

---

## Multi-Task Learning

하나의 통합 데이터셋(모든 태스크의 engagement 데이터)으로 학습하며, 3가지 메커니즘으로 cross-task learning을 촉진한다:

1. **Task-aware contextualization**: task type을 피처로 포함하여 태스크 간 trade-off 학습
2. **Context imputation**: 누락 컨텍스트를 채워 피처 커버리지 향상
3. **Feature crossing**: 태스크 간 교차 학습 강화

**결과**: 개별 모델보다 통합 모델이 더 큰 데이터셋에서 학습하면서 **각 태스크가 보조 태스크로부터 이점**을 얻어 성능이 향상됨

---

## Personalization (단계적 도입)

검색에 개인화를 무분별하게 적용하면 **쿼리 관련성(relevance)을 해칠 수 있고**, 키 입력마다 결과를 반환해야 하는 **엄격한 지연 시간 제약**이 있다. 이를 해결하기 위해 3단계로 점진적 도입:

| 단계 | 방식 | 특징 |
|------|------|------|
| Phase 1 | **Semi-personalized** | 사용자 클러스터를 컨텍스트 피처로 사용 (캐싱 가능) |
| Phase 2 | **Intermediate** | 별도 추천 모델의 출력을 입력 피처로 활용 |
| Phase 3 | **End-to-end** | 사전학습된 user/item representation 모델을 UniCoRn과 jointly fine-tuning |

**성과**: 비개인화 → 완전 개인화 시 **검색 7%, 추천 10% 성능 향상**

---

## 실제 배포 (Netflix)

단일 UniCoRn 모델이 현재 Netflix의 다음 서비스들을 동시에 지원한다:
- **Netflix Search** (쿼리 기반 검색)
- **Personalized Pre-query Canvas** (검색 전 개인화 추천)
- **More Like This Canvas** (유사 콘텐츠 추천)

---

## 주요 기여 정리

1. 검색과 추천을 **하나의 모델로 통합** 가능함을 실증
2. Context imputation을 통한 **cross-task knowledge sharing** 메커니즘 제안
3. 쿼리 관련성과 개인화 간 **trade-off를 관리**하는 단계적 개인화 전략
4. 기술 부채와 시스템 관리 오버헤드 **대폭 감소**
5. Netflix 프로덕션에서 **실제 검증 완료**
