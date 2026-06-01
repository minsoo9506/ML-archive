# Metadata Generation and Evaluation using LLMs - Case Study on Canonical Titles

- RecSys 2025, Indeed
- Sinan Zhu, Sanja Simonovikj, Darren Edmonds, Yang Sun
- **Keywords**: `canonical title generation`, `LLM annotation`, `text normalization`, `embedding-based similarity`, `two-stage deduplication`, `prompt engineering`, `LLM fine-tuning`, `occupation taxonomy`, `autocomplete / query suggestion`

## 핵심 요약

구직 플랫폼의 autocomplete 품질을 높이기 위해, raw job title을 **정규화된 canonical title**로 자동 생성하는 LLM 기반 프레임워크. 수작업 vocabulary에 의존하던 기존 방식(NormTitles)을 대체하여, LLM 정규화 + 임베딩 기반 2단계 dedup으로 canonical title set을 구축. **offline 정확도 +18.6%**, **online A/B에서 selection rate 최대 +316.7%** 향상.

## 문제 정의

- Job title은 표현이 제각각: `sr data scientist` vs `data scientist senior`, 또는 `superstar software engineer` 같은 과장된 형태
- 이런 불일치가 autocomplete 제안 품질과 사용자 만족도를 저하
- 기존 방식(전문가가 canonical title을 정의하고 raw variant를 매핑)은 **수작업·정적 vocabulary**에 의존 → 노동집약적, 확장 어려움, 오류 발생
- 목표: raw title 데이터로부터 **자동·확장 가능하게** canonical title을 생성

# Methodology: Canonical Title Generation

3단계로 구성: 불필요 정보 제거 → 일관된 포맷 강제 → 의미적으로 동일한 title 제거.

## 1. Normalization (정규화)

- 초기 zero/few-shot 프롬프팅의 문제: **포맷 불일치** + 실제 데이터에 anchor되지 않은 비현실적 title
- 해결: Indeed 독점 데이터(en-US 시장의 resume·job에서 추출한 raw title)를 occupation별로 수집해 정규화
  - rare title 제외, 비영어 문자·잘못된 기호·과도한 노이즈 제거
- **gpt-4o-2024-08-06** 모델로 raw title을 표준 포맷으로 변환 (사전 정의된 exclusion/inclusion 기준 적용)
  - 핵심 책임·seniority는 보존, pay·location 등 비핵심 정보는 제거
  - 예: `Software Engineer Level 2`, `SWE II` → `Software Engineer II`
  - occupation context로 generic title 명확화: `Senior Engineer` + "Back End Developer" occupation → `Senior Back End Engineer`

## 2. Generic Title Removal (일반 title 제거)

- bulk 정규화 후, Indeed taxonomist가 큐레이션한 occupation 정의를 기반으로 LLM이 추가 generic title 식별
- occupation의 핵심 책임·기능을 대표하지 못하는 title은 generic으로 판단해 canonical set에서 제외

## 3. Deduplication (2단계 중복 제거)

- **1단계**: 임베딩 거리 기반 **K-means 클러스터링** → 유사 역할의 title을 묶고, 같은 클러스터 내 title 쌍 생성
- **2단계**: LLM이 사전 정의 규칙으로 의미적 동일 여부 판정
  - 동일/유의어 조건: 핵심 기능 중복 + 동일 seniority + 동일 work setting/contract type
- 동일 그룹 내에서는 **raw title 빈도가 가장 높은 단일 title만 유지**

## Canonical Title 분포

- **922개 occupation**에 걸쳐 **33,952개** unique canonical title 생성
- occupation당 평균 45개(중앙값 40), 최소 1개 ~ 최대 258개 → 역할 다양성·granularity의 큰 편차

# Evaluation

offline 평가를 위해선 **정답이 있는 데이터셋**이 필요한데 기존에 없으므로, 평가 pair를 직접 만들고 라벨링하여 데이터셋을 구축함.

- **Job Title Equivalence Classification Task**: 두 raw title `t1`, `t2`가 같은 근본적 역할인지(→ 같은 canonical title로 매핑되어야 하는지) binary 분류
  - 라벨 1 = 같은 역할, 0 = 다른 역할 (표현·부가정보가 달라도 본질이 같으면 1)
  - 의도: canonical set의 **granularity(세분화 정도)**가 적절한지 측정 — "같은 건 묶고, 다른 건 구분"이 잘 되는지
  - inference: 각 raw title의 최근접 canonical title을 **임베딩 거리**로 찾고, 두 결과가 같으면 1 → 정확도가 높으면 canonical set이 동의어 title을 잘 모은다는 의미

### 평가 데이터셋 구축 과정

1. **평가 pair 생성**: occupation 내 raw title을 클러스터링 후 within/between-cluster 쌍을 섞어 샘플링(쉬운/어려운 쌍 다양성 확보), occupation당 수백 개 쌍
2. **라벨링 (fine-tuned LLM 라벨러)**: 사람이 다 하기엔 양이 많아 LLM을 라벨러로 사용
   - 전문가 in-the-loop 라벨링은 비효율, 순수 프롬프팅은 human과 <70% 일치
   - ~3,000 쌍 추가 라벨링으로 LLM fine-tune (occupation당 3쌍씩, 유사도 버킷별, 최종 ~5,500 쌍)
   - 9개 프롬프트 변형 테스트 → 4개 선별 → fine-tuning 시 프롬프트 민감도 감소(정확도 차 <1%)
   - 최종 프롬프트(<120 tokens)로 holdout에서 **human과 93% 일치**, 이후 occupation당 수백 datapoint 라벨링 → 최종 **~347,000 datapoint**

> **두 종류의 LLM 구분 (헷갈리기 쉬움)**
> - **생성용 LLM (gpt-4o)**: raw title → canonical title을 *만드는* 모델 (= 평가 대상)
> - **라벨러 LLM (fine-tuned)**: 평가 데이터셋의 *정답을 붙이는* 모델 (= 사람 대신 채점 도구)
> - 장점: 사람 라벨링 없이 대규모 정답 데이터셋을 싸게 확보해 평가 자동화/확장
> - 한계: 정답 자체가 LLM 산출물이라 모델 편향이 평가에 섞일 수 있음 → holdout human 93% 일치로 신뢰성 확보

- **결과**: 약 347,000 datapoint에서 canonical set이 normTitle baseline 대비 정확도 **+18.6% (65% → 77.1%)**
  - baseline: 기존 normTitles 서비스로 raw → normTitle 변환 후 같은지 비교
  - occupation별 분석: 82.6%가 의미있는 향상(평균 +0.172), 13.9% 하락(-0.072), 3.5% 변화 없음

## Online Evaluation (A/B)

- 두 플랫폼(jobseeker onboarding, job posting interface)에서 실험
- 처치군: canonical title로 강화된 autocomplete 제안 / 대조군: 기존 시스템
- 결과 (selection rate 증가):
  - **Jobseeker Onboarding: +316.7%** (95% CI 314.8–318.5%)
  - **Job Posting: +163.2%** (95% CI 160.3–166.0%)

# Cost

- 922개 occupation 전체의 canonical title 생성 비용 약 **$2,500**
- 평가용 fine-tuning + inference 비용 **<$250** (대부분 inference, 실험·프롬프트 최적화 비용 제외)

# Conclusion & Future Work

- 비정형 텍스트의 일관된 canonicalization이 필요한 모든 도메인(상품 카탈로그, 강의, 뉴스 등)에 일반화 가능한 확장형 LLM 접근법
- 남은 과제:
  - LLM이 생성한 title이 지나치게 generic할 수 있어 **human oversight** 필요
  - 임베딩 기반 dedup이 **seniority가 다른 title을 잘못 제거**할 수 있음
- 향후 방향: job title을 **구조적으로 분해**(domain, specialty, seniority 등)하여 더 정밀한 canonical set·dimension별 dedup 구현
