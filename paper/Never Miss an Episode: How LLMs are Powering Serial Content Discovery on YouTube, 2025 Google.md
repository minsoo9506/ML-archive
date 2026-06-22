# Never Miss an Episode: How LLMs are Powering Serial Content Discovery on YouTube

- RecSys 2025, Google (YouTube)
- Aditee Kumthekar, Li Wei, Andrea Bettale, Mahesh Sathiamoorthy, Zrinka Puljiz, Aditya Mahajan

## 핵심 요약

YouTube에서 **연속 시청 콘텐츠(Serial Playlist)**를 식별하기 위해 모델 학습 없이 **few-shot LLM 프롬프트**만으로 분류기를 구축한 사례. 기존의 수작업 정규표현식(REG-EXP) 기반 시스템을 대체하여, 최소한의 엔지니어링 투자로 golden set 기준 **정밀도 69% / 재현율 100%**를 달성. 연속 콘텐츠 식별량을 **25% 증가**시키고, 라이브 실험에서 **만족 engagement +0.39%**의 긍정적 효과를 확인.

## 문제 정의

> **분류 단위는 "영상 1개"가 아니라 "플레이리스트 1개"다.**
> 크리에이터가 이미 만들어 둔 플레이리스트(영상들이 하나의 set으로 묶여 있고 위→아래 순서도 이미 매겨진 상태)를 입력으로 받아, 그 **플레이리스트 전체에 메타 라벨을 붙이는 분류** 작업이다. LLM이 새 플레이리스트를 만드는 게 아니라, 기존 플레이리스트에 "이건 연속물이고 다음 화는 이쪽 방향"이라는 라벨을 달아 추천 엔진이 이어보기/다음 화 추천에 쓰도록 한다.

- YouTube 플레이리스트 중 일부는 **Episodic**(반복되는 테마/캐릭터/포맷)이고, 그 중 일부는 순서대로 봐야 하는 **Serial**(연속) 콘텐츠
  - **Episodic: Serial** — 에피소드 간 서사가 이어져 순서대로 시청해야 함 (예: Hell's Kitchen Season 20)
  - **Episodic: Non-Serial** — 함께 보지만 순서는 무관 (예: Hot Ones)
  - **Not Episodic** — 그 외 (예: Chill Songs)
- 연속 콘텐츠를 잘못 식별하면 "다음 에피소드"로 엉뚱한 영상을 추천하게 됨 (Figure 1)
- 정확히 식별하면 시청자의 시리즈 **발견 / 이어보기 / 계속보기** 경험을 개선 가능

## 기존 방식 (Prior Art)

기존 시스템은 두 가지 소스로 연속 데이터를 확보:

1. **크리에이터가 직접 Serial로 태깅** → 부정확하고 악용 소지 있음
2. **수작업 정규표현식(REG-EXP)** → 엔지니어가 새 패턴 발견할 때마다 수동 유지보수

- REG-EXP는 **고정밀 / 저재현율** 시스템 → 플레이리스트에 추가되지 않은 연속 콘텐츠의 **30%만** 포착
- 국제(외국어) 콘텐츠나 새로운 패턴에 대한 **일반화 능력 부족**
- BERT 같은 전통적 텍스트 분류기는 수천 개의 균형 잡힌 라벨링 데이터가 필요 → 구축 비용이 큼

## 접근 방법 (Our Approach)

- **few-shot 프롬프트 + 소수의 수작업 큐레이션 예시**로 multi-headed classifier 구성
  - 플레이리스트가 (1) Episodic 인지, (2) Serial 인지, (3) Serial이면 순서(ascending/descending)가 무엇인지 예측
  - **Order는 "순서를 새로 정하는 것"이 아니라**, 플레이리스트에 이미 나열된 순서가 에피소드 번호 기준 오름차순(1화가 위)인지 내림차순(최신화가 위)인지를 읽어내는 것. 크리에이터마다 정렬 방향이 달라서, 지금 보는 화의 **진짜 다음 화**가 리스트의 위쪽인지 아래쪽인지 계산하는 데 필요 (Figure 1의 "다음 에피소드 오추천" 문제 해결)
- 입력: 플레이리스트의 앞쪽 영상 제목 몇 개
- 출력 예: `Rater Evaluation: Episodic - Yes <> Serial - Yes <> Order - Descending`
- **오프라인 배치 파이프라인**에서 소형 LLM으로 추론 → 추천 엔진으로 Serial Playlist 전달
- 리소스/품질 트레이드오프를 고려해 모델 크기 선택

## 평가 및 학습 (Evaluations and Learnings)

수백 개 규모의 수작업 라벨링 golden set 구축, **Gemini V2 S 모델**로 프롬프트 튜닝.

### 1. 분류 목표의 명확성이 결정적

- 모호한 목표인 "Episodic"보다 명확한 목표인 "Serial"에서 정밀도가 더 높음
- LLM은 **순서 시청 여부(Serial) 판단을 Episodic 여부 판단보다 훨씬 쉽게** 수행

| Objective | Highest Precision |
|---|---|
| Episodic | 66% |
| Serial | 77% |

### 2. 페르소나/추론 추가는 효과 없거나 부정적

- Persona text 예: "Imagine you are a film critic..."
- Reasoning text 예: "Please show your work with reasoning."
- Serial 정밀도는 거의 변화 없으나, **Episodic 정밀도는 크게 하락** → 라이브 실험에 미적용

| Prompt Technique | Episodic Precision | Serial Precision |
|---|---|---|
| No Persona + No Reasoning | 67.62% | 77.16% |
| No Persona + Reasoning | 66.55% | 76.01% |
| Persona + No Reasoning | 55.26% | 77.15% |
| Persona + Reasoning | 55.38% | 76.39% |

### 3. 프로덕션 배포 결과 (vs REG-EXP)

- 크리에이터가 Serial로 태깅한 플레이리스트 중 **30%만 실제 Serial**임을 LLM이 식별
- REG-EXP 대비 **연속 플레이리스트 식별량 25% 증가**
- LLM이 추가로 잡아낸 재현율(recall) 향상의 출처:
  1. **74%** — 외국어 연속 플레이리스트
  2. **12%** — 수작업 REG-EXP에 없던 새로운 정규식 유형
  3. **14%** — 실제로는 연속이 아닌 false positive

## 리소스 트레이드오프 (Resource Trade-off)

소형 모델(8B)이 대형 모델(24B) 대비 **절반의 리소스로 비슷한 정밀도/재현율** 달성:

| Model | Accuracy | Precision | Recall | Resources |
|---|---|---|---|---|
| Gemini V2 S (24B) | 82% | 76.39% | 96.29% | 256 TPUs |
| Gemini V2 XS (8B) | 79% | 69% | 100% | 128 TPUs |

## 라이브 실험 결과

| Technique | Satisfied Engagement | Daily Active Users |
|---|---|---|
| LLM classification | +0.39% | +0.02% |

→ 시리즈의 다음 에피소드를 찾는 어려움을 해소하여 연속 콘텐츠 engagement가 유의미하게 상승.

## 향후 방향 (Future Directions)

- 언어(영어 vs 비영어), 장르(학습, 엔터테인먼트 등)별 **구조화된 평가셋** 구축
- **멀티모달 피처**(썸네일, OCR) 통합 및 **auto-tuning** 실험
- 수작업 평가셋 한계를 넘기 위해 **별도 LLM 프롬프트로 평가 데이터 생성**

## Conclusion

- few-shot LLM 추론으로 더 정확한 연속 플레이리스트 탐지 + 식별량 증가 달성
- BERT 분류기(대량 라벨 필요)나 fine-tuned LLM보다 **비용 효율적이고 안정적인 대안**
- 텍스트 기반 분류에서는 **명확한 목표 정의**가 예측 정확도의 핵심
- Persona/Reasoning 추가가 모호한 목표("Episodic")의 정밀도를 개선하지 못함
- 프로덕션에는 **Gemini XS(소형) 모델 + 오프라인 추론**으로 최소 리소스로 충분한 정밀도 확보
