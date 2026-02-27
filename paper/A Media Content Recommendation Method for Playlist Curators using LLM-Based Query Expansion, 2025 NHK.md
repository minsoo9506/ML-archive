# Introduction
- playlist curation 은 좋지만 사람이 하기 리소스가 많이 든다.
- 해당 논문에서는 item ordering 보다 **aiding curators in initial content identification and selection** 에 집중한다.
- Unlike typical recommendations optimized for individual end-userpreferences, playlist curation demands **테마의 다양성과 포괄성**
- our method boosts precision at the critical top of the list (P@10 improved from 0.79 to 0.98), improves overall P@50 by 22 percentage points, and increases the diversity of the retrieved content

# Proposed Method

## System Architecture

시스템은 **Content-Indexing Phase**(콘텐츠 색인)와 **Playlist-Generation Phase**(플레이리스트 생성) 두 단계로 구성된다.

---

### Phase 1: Content-Indexing Phase

> 콘텐츠 데이터를 벡터화하여 검색 가능한 상태로 준비하는 단계

```
Content Data ──Register──▶ Content Data DB
     │
     └──Embed & Index (v_c with ID)──▶ Content Vector DB
```

1. **Content Data**를 **Content Data DB**에 등록(Register)
2. 각 콘텐츠를 임베딩하여 벡터 $v_c$와 ID를 **Content Vector DB**에 색인(Index)
   1. title + description 을 임베딩

---

### Phase 2: Playlist-Generation Phase

> 큐레이터가 입력한 플레이리스트 테마를 기반으로 후보 콘텐츠를 검색하는 단계

```
Playlist Theme ──────────────────────────────────┐
(Title & Description)                            │
     │                                           │
     ▼                                           │
   LLM (Query Expansion)                         │
     │                                           │
     ▼                                           │
Expanded Queries: q_1 ··· q_x                    │
     │                                           │
     ▼                                           ▼
Generate Search Vectors          Original Theme Query: q_meta
  v_meta, v_1 ··· v_x                            │
     │                                           │
     ▼                                           │
┌─────────────────────┐                          │
│  Content Vector DB  │◀─────────────────────────┘
└─────────────────────┘
     │
     │  Retrieve Top-k Content IDs
     │  by sim(v, v_c)
     ▼
┌─────────────────────┐
│  Content Data DB    │
└─────────────────────┘
     │
     │  Fetch / Load
     ▼
Candidate Content for Playlist
```

**흐름 요약:**

| 순서 | 단계 | 설명 |
|:---:|------|------|
| 1 | **테마 입력** | 큐레이터가 Playlist Theme (제목 & 설명)을 입력 |
| 2 | **LLM Query Expansion** | LLM이 테마를 분석하여 확장 쿼리 $q_1 \cdots q_x$ 를 생성 |
| 3 | **Original Theme Query** | 원본 테마에서 직접 $q_{meta}$도 함께 생성 |
| 4 | **벡터 생성** | 모든 쿼리를 임베딩하여 검색 벡터 $v_{meta}, v_1 \cdots v_x$ 생성 |
| 5 | **유사도 검색** | Content Vector DB에서 $sim(v, v_c)$ 기반 Top-k 콘텐츠 ID 검색 |
| 6 | **콘텐츠 로드** | Content Data DB에서 해당 ID의 콘텐츠를 Fetch/Load |
| 7 | **결과 반환** | 플레이리스트 후보 콘텐츠(Candidate Content) 제공 |

# Experiments
## 주요 실험 내용
- 일본 tv 프로그램 12,127개 데이터셋 사용
-  Content vectors were pre-computed using `intfloat/multilingual-e5-large` and indexed with `Faiss`
-  5 expanded queries from the `google/gemma-2-27b-it` LLM
-  총 6개의 seed vector 로 각각 top50 retrieve 하고 similarity score 높은 최종 top50 선정
-  prompt
  -  (1) Phrase-based Queries, to ensure semantic richness
  -  (2) Semantic Rephrasing, to avoid using the exact input words
  -  (3) Forced Diversity, to ensure queries were dissimilar
  -  The prompt also included two few-shot examples

## 결과
- baseline (query expansion 이 없는 방식) 보다 precision, diversity 대부분 좋았다.

## prompt
```text
You are an expert playlist curator. Given a playlisttitle and an optional description , generate 5 appropriatesearch queries to find programs for that playlist.

### INSTRUCTIONS & CONSTRAINTS :
- ** Phrase - based Queries :** Queries must be descriptivephrases (e.g., " programs about ancient history ") , notjust lists of keywords.
- ** Semantic Rephrasing :** Crucially , you must AVOIDusing the exact words from the input title / description .Rephrase the concept using synonyms or related terms.
- ** Forced Diversity :** The 5 generated queries must beas dissimilar from each other as possible . Approach thetheme from different angles (e.g., target audience ,content format , emotional tone ).

### FEW - SHOT EXAMPLES :
** Example 1:**
** Input :**
- ** Title :** " Weekend Watch : Emotional Movies "
- ** Description :** "A collection of movies for a quietweekend that will move you to tears . From human dramas toromances , enjoy these heartwarming stories ."
** Output :**
- Tearjerker films to get lost in on a day off
- Heart-wrenching visual works depicting human drama- Love stories that resonate with adults
- A collection of films that leave you with a gentlefeeling
- Timeless masterpieces to watch during a long holiday

** Example 2:**
** Input :**
- ** Title :** " Beginner 's Yoga Lessons "
- ** Description :** " Simple yoga lessons perfect forbeginners to start today . Learn poses for relaxation andweight loss with easy-to-follow instructions."
** Output :**
- A safe yoga course for the inexperienced
- An introductory program of stretches to relax mind andbody
- Yoga poses for those aiming for a slimmer figure
- Simple exercises to improve flexibility
- Relaxing yoga to do before bed
```

# 기억에 남는 점
- LLM 을 활용한 query expansion 은 다양하게 응용할 수 있을 것 같다.