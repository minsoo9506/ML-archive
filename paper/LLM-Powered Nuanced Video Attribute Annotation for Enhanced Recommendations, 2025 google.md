# Introduction
- 대규모 숏폼 비디오 추천 시스템(YouTube)에서 "LLM-as-annotators" 방식을 통해 콘텐츠의 미묘한(nuanced) 속성(예: "vibe" — authentic, inspiring, calming, energetic)을 어노테이션하는 사례 연구
- 기존 ML 분류기의 두 가지 한계:
  - 신규 분류기 개발에 긴 사이클이 필요
  - 표면적(high-level) 이해에 머물러 개인화에 필요한 뉘앙스를 놓침
- LLM의 world knowledge와 reasoning을 활용해 개발 사이클 단축 + 미묘한 속성 어노테이션 가능
- 산업 규모 적용 시 핵심 과제:
  1. 미묘하지만 중요한 콘텐츠 속성의 정의 및 정제
  2. 일일 수백만~수천만 비디오에 대한 고품질·저비용·저지연 어노테이션 스케일링
  3. 풍부한 어노테이션을 온라인 추천 모델에 통합하여 사용자 경험 개선

# Method
End-to-end 워크플로우는 3단계로 구성:

## 1) Defining Target Attributes & Evaluation
- 주관적/미묘한 속성(vibes)에 대해 명확하고 일관된 정의 수립이 핵심
- **Golden Set**: 내부 전문가 rater들이 반복적 토론/캘리브레이션을 통해 정렬된 고품질 수동 어노테이션 세트
  - Inter-Rater Reliability 개선 + edge case 발굴로 정의 정제
  - 예: "authentic" vibe 초기 프롬프트가 편집이 많은 vlog 스타일을 잘못 제외 → 편집 수준보다 크리에이터의 진정성 있는 표현을 우선시하도록 정의 수정
- 평가 전략:
  - 오프라인: Golden Set 기반 Precision/Recall/F1
  - 온라인: A/B 테스트로 실제 임팩트를 ground truth로 활용 → 정의/프롬프트를 반복적으로 정제
- 레슨런:
  - LLM 어노테이션 품질은 내부 rater 정렬(=인간의 모호성 감소)에 강하게 의존
  - 오프라인→온라인 실험 사이클의 속도 극대화가 임팩트의 관건

## 2) Offline Bulk Annotation
- 신규/트렌딩/고임팩트 등 우선순위 비디오(10^5–10^6/일) 대상으로 LLM 직접 어노테이션
  - 입력: 샘플링된 비디오 프레임 + 비디오 설명 + 프롬프트
- **Inference 최적화** (baseline 대비 2-3x throughput):
  - Model quantization (GPTQ)
  - Batch size tuning
  - Model sharding (Megatron-LM)
- **Knowledge Distillation으로 전체 코퍼스(10^7/일)로 확장**:
  - LLM 어노테이션 + 확률 점수 → "Silver Set" teacher label
  - 경량 student DNN을 사전계산된 비디오 임베딩 같은 컴팩트 피처로 from-scratch 학습
  - 약간의 품질 손실로 대규모 throughput / 낮은 지연·비용 달성
- 워크플로우 순서: 우선순위 코퍼스에 LLM 직접 어노테이션 → A/B로 임팩트 검증 → distillation으로 전체 확장

## 3) Online Personalized Recommendation
- **Personalized Restricted Retrieval** 방식으로 통합:
  - 프로덕션 대규모 Transformer 기반 sequential retrieval 모델 위에서, 각 attribute의 콘텐츠 vocabulary 내에서 restrictive nearest neighbor 검색(SCANN)
  - 어떤 attribute를 트리거할지는 사용자 의도 모델/휴리스틱이 결정 (사용자의 attribute 친화도 예측)
  - attribute 내 아이템 선택은 sequential retrieval 모델이 담당
- LLM-annotated corpus와 retrieval stack의 긴밀한 통합으로 **신규 프롬프트 → 온라인 A/B까지 1주일 이내** 달성
- 다른 통합 방식 실험 중: 랭킹 모델 피처로 사용, 다양성(diversity) 차원으로 사용

## Workflow Diagram
```
[Attribute Defs + Golden Set] → [Optimized LLM Annotator] → [LLM Silver Set] → [Distilled Student Model]
                                          ↓
              [Video Corpus & Features] → [Model Inference] → [Annotated Video Corpus]
                                                                      ↓
                                              [Personalized Restricted Retrieval] → [Recommended Videos]
```

# Results
## Offline Annotation Quality
- 어떤 nuanced attribute에 대해:
  - **Gemini 2.5 Pro**: F1 **81.33%** (P 85.03%, R 77.94%)
  - **External crowd-sourced human raters**: F1 **63.21%** (P 76.82%, R 53.69%)
- LLM이 human rater를 능가 → 인간 라벨 기반의 전통 ML 분류기 상한선을 돌파
- Multimodal Gemini + flexible prompting은 태스크별 fine-tuning 불필요로 운영 효율성 크게 향상

## Online A/B Experimentation
- Personalized Restricted Retrieval로 LLM-annotated attribute 적용 시:
  - User participation in content creation: **+0.49%**
  - Satisfied consumption: **+0.21%**

# Conclusion
- LLM-as-annotators 접근으로 미묘한 콘텐츠 속성에 대한 깊은 이해 + 개발 사이클 단축
- 핵심 성공 요소:
  - 오프라인 평가 + 온라인 A/B를 결합한 반복적 정제
  - Knowledge distillation을 통한 전체 코퍼스 확장
  - 온라인 추천 스택과의 긴밀한 통합으로 실험 속도 극대화
- 향후 과제: 추천 모델과의 더 깊은 통합, 변화하는 콘텐츠 환경에 대한 지속 적응
