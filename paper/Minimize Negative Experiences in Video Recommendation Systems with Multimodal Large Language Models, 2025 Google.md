# Minimize Negative Experiences in Video Recommendation Systems with Multimodal Large Language Models

- RecSys 2025, Google (YouTube)
- Suman Malani, Youwei Zhang, Liang Liu

## 핵심 요약

YouTube 숏폼 추천에서 **설문(survey) 피드백** 기반으로 "매우 부정적인 경험(highly negative experience)"을 탐지·억제하는 방법을 다룬 논문. 설문 데이터는 ultra-sparse / imbalanced / noisy하다는 한계가 있는데, 이를 **MLLM(13B) teacher 모델 fine-tuning → silver label 생성 → 경량 student 랭킹 모델(HNRM)에 knowledge distillation** 하는 구조로 극복. 온라인 A/B에서 **설문 부정 경험률 -8Y% 감소, engagement +7Z% 증가**를 달성하면서 sparse 데이터 한계를 넘어 모델 스케일링을 가능하게 함.

## 문제 정의

- 부정적 경험 감소는 장기 사용자 만족도와 양질 콘텐츠 유통에 중요
- 시청 후 **interstitial survey**로 피드백 수집 → 사용자/아이템 전반에 걸친 random uniform 샘플링 가능, 세션별 개인화 만족도 포착
- 비요청(unsolicited) 피드백(싫어요, 신고 등)은 부정 피드백을 잘 주는 소수 사용자에게 데이터가 편향됨 → 설문이 더 균형적
- 그러나 설문 모델링의 고유 난제:
  - **Ultra sparsity**: 제한된 설문 노출 + 낮은 응답률 + positive(부정 경험) 비율의 심한 불균형
  - **Large epistemic noise**: 명시적 설문에서 오는 노이즈, 주요 사용자/아이템 코호트의 과소대표 (calibration 분석에서 확인)
- 기존 방식은 generalization plateau에 도달. 전통적 regularization, 모델 크기 확대, 데이터 추가로도 개선 안 됨 → 오히려 calibration error 증가 및 A/B 지표 악화. 데이터 볼륨에 의해 스케일링이 제한됨

## 접근 방법

핵심 동기: epistemic noise 최소화(분포 변화에 robust), generalization 향상으로 모델 스케일링 가능, 과소대표 사용자/아이템에 대한 예측 품질 개선.

- MLLM은 사전학습 지식을 활용해 도메인 특화 부정 경험 분류에 강점을 보임
- 그러나 fine-tuned MLLM을 직접 HNRM으로 서빙하는 것은 비용이 과도함
- → **Knowledge Distillation**: fine-tuned MLLM을 teacher로 두고 silver label을 생성, 이를 경량 student(HNRM)에 distill
- Student는 teacher 대비 **99.9% predictive coverage** (동일 샘플에서 student/teacher AUC 비율) 달성

## Survey Distribution (설문 수집 설계)

- 영상 노출 후 **interstitial survey**를 선택적으로 노출
- aleatoric 노이즈와 응답 편향 최소화를 위한 장치:
  - 설문 상호작용 후 일정 기간 사용자 **보호(cooldown)**, 제품 전반의 설문 분포 제한
  - 설문 dismissal 및 response cooldown으로 사용자 짜증 최소화
  - 배치 내 영상에 대한 **무작위 트리거링**으로 positional bias 제한
- **2단계 설문**: 1~2점(낮은 smiley)을 선택하면 follow-up 옵션으로 부정 응답을 구체적으로 포착

## Contextual Understanding (Teacher 전용 컨텍스트)

기존 랭킹 모델은 user / item / cross 정보를 쓰지만, 두 가지 추가 컨텍스트가 중요:

### Post Engagement Session Information
- 영상 상호작용 **이후에만** 얻을 수 있는 정보: 후속 시청의 watch time, 좋아요/싫어요/댓글, 댓글 engagement time 등
- 추론 시점에는 사용 불가하므로 온라인 서빙 student에는 못 쓰지만, **teacher에는 의도적으로 포함**

### Community Information
- 사용자는 커뮤니티 기여와 과거 아이템 상호작용에 영향받음
- engagement용 dense feature/임베딩은 **커뮤니티 sentiment와 과거 상호작용**을 직접 포착 못 함
- 댓글 데이터, 시청 이력 설명/transcript 등으로 의도 구분: "과거 시청 이력상 개인적으로 관심 없음" vs "주제 자체의 문화적 민감성"

## Fine-tuning MLLM (Teacher)

- 데이터 sparsity가 오히려 장점 → 추가 feature 가공/변환/주석이 비용 효율적, 최적화된 하드웨어 활용 가능
- prompt engineering이 아닌 **supervised fine-tuning** 선택 (개인화된 도메인 지식이 필요하기 때문)
- 내부 사전학습 **13B 파라미터** 모델 사용 (image/video 임베딩 활용) → latency 감소, 학습 속도/개발 속도 향상
- 데이터 파이프라인: post engagement feature, dense feature, text, image/video/user/cross-user query 임베딩으로 세션 데이터 augment
- **Embedding projection**으로 raw 데이터(영상 프레임 등) 없이 컨텍스트 추가 → 학습 latency/비용 폭증 방지
- baseline 대비 teacher **AUC +3%, AUC-PR +1%** 향상, calibration은 baseline과 동등

## Knowledge Distillation for Sparse Tasks (Student)

- Teacher inference 파이프라인을 **매일** 실행하여 silver label 생성 → student 랭킹 학습 데이터에 merge
- Student는 teacher보다 훨씬 단순(user 임베딩 + sparse/dense feature만)하지만, 기존 production HNRM baseline보다는 큼
  - dense feature는 teacher의 부분집합 (post engagement 정보는 서빙 시 사용 불가)
- **Loss 공식** (L = binary cross entropy):

  `loss = α · L(p_{u,i}, ȳ_{u,i}) + β · L(p_{u,i}, y_{u,i})`

  - `ȳ_{u,i}`: silver(teacher) 예측, `y_{u,i}`: 실제 설문 응답 라벨
  - α, β로 silver task와 survey task의 가중치 조정
- Silver task 덕분에 student를 더 큰 네트워크로 학습 가능 → **dense 파라미터 20배 이상**, 더 많은 step, 더 큰 batch size
- 성과 (production baseline 대비):
  - **AUC +2.46%** 추가 향상
  - **99.9% predictive coverage**
  - **Expected calibration error -49% 감소**
  - full epoch 학습 및 HNRM 스케일업 달성
- 두 가지 핵심 learning:
  1. silver label이 설문 응답 라벨보다 더 많은 정보를 인코딩 → 두 task를 co-training하고 가중치를 튜닝하면 teacher divergence를 보정하며 성능 개선
  2. silver label은 샘플별 teacher가 학습한 규칙성을 전달 → 시간에 따른 분포 변화에 더 유연

### 왜 silver를 survey와 동등하게 합치지 않고 가중치(α, β)로 분리했나

- **두 라벨의 신뢰도가 다름**: survey는 유저가 실제로 답한 ground truth(양은 적지만 진짜), silver는 teacher의 예측(양은 많지만 틀릴 수 있음)
- 구분 없이 같은 정답으로 합치면 생기는 문제:
  - silver가 압도적으로 많아 student가 사실상 **teacher를 그대로 복제** → teacher의 bias/노이즈까지 물려받아 teacher를 못 넘어섬
  - 적은 양의 진짜 survey 신호가 대량 silver에 **희석**됨
- 그래서 silver와 survey를 **별도 task로 분리**하고 α, β로 균형 조정:
  - silver(α): **일반화·데이터 풍부화** 담당 → 큰 모델 학습 가능(dense 파라미터 20배)
  - survey(β): **진짜 정답으로의 anchor/보정** 담당 → 논문 표현으로 *"teacher divergence 보정"*
- silver는 hard label(0/1)이 아니라 **teacher의 soft prediction(샘플별 학습된 regularity)**으로 다루는 게 distillation의 정석(Hinton et al. 2015) → hard label로 박아넣는 것과 성격이 다르므로 별도 task + 가중치로 처리

## Live Experiment Results

- 수십억 사용자 규모의 숏폼 시스템에서 **2주간 A/B 실험** (랭킹 stage)
- 결과: **설문 부정 경험률 -8Y% 감소**, **engagement +7Z% 증가** (Y, Z는 익명화를 위한 양의 상수)
- 주요 아이템/사용자 코호트 전반에서 개선 확인

## Conclusion & Future Work

- ultra-sparse/noisy한 설문 모델링을 generalization 향상으로 개선 → 세션 표현 개선 및 데이터 볼륨에 묶이던 모델의 스케일링 가능
- 향후 방향:
  1. **Large Recommender Models** 같은 더 나은 사전학습 모델로 user-item 상호작용 이해 강화
  2. silver data를 더 많이 생성하여 sparsity 문제 완화
  3. **active distillation**으로 추가 노이즈 감소 및 효율적 학습
