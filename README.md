요즘은 주로 추천, LLM 관련하여 공부합니다.

# Index

- [Index](#index)
- [📑 Paper Reference](#-paper-reference)
  - [Recommendation System](#recommendation-system)
      - [Algorithm](#algorithm)
      - [Algorithm - text, image](#algorithm---text-image)
      - [Algorithm - session-based, sequential](#algorithm---session-based-sequential)
      - [Algorithm - graph](#algorithm---graph)
      - [Retrieval](#retrieval)
      - [Embedding](#embedding)
      - [LookAlike](#lookalike)
      - [Bandit](#bandit)
      - [LLM, Agent](#llm-agent)
      - [Diversity](#diversity)
      - [Calibration](#calibration)
      - [Bias](#bias)
      - [Explainable](#explainable)
      - [Cold Start](#cold-start)
      - [User Modeling](#user-modeling)
      - [Causality](#causality)
      - [Push, CRM](#push-crm)
      - [Ads](#ads)
      - [Search, Query, IR](#search-query-ir)
      - [System](#system)
      - [Survey](#survey)
  - [Imbalanced Learning, Anomaly Detection](#imbalanced-learning-anomaly-detection)
      - [Survey](#survey-1)
      - [Perfomance Measure](#perfomance-measure)
      - [Cost-sensitive](#cost-sensitive)
      - [Sampling](#sampling)
      - [Ensemble Learning](#ensemble-learning)
      - [Feature Selection](#feature-selection)
      - [Learning feature representations of normality](#learning-feature-representations-of-normality)
      - [Time Series and Streaming Anomaly Detection](#time-series-and-streaming-anomaly-detection)
  - [Causality](#causality-1)
      - [Heterogeneous treatment effect estimation, uplift](#heterogeneous-treatment-effect-estimation-uplift)
  - [LLM](#llm)
      - [LLM judge](#llm-judge)
      - [Embedding](#embedding-1)
- [📑 Other Reference](#-other-reference)
  - [conference](#conference)
  - [카카오 💬](#카카오-)
      - [추천, 검색](#추천-검색)
      - [LLM, AI](#llm-ai)
      - [이상치탐지](#이상치탐지)
      - [XAI](#xai)
      - [플랫폼](#플랫폼)
      - [ML](#ml)
  - [카카오엔터테인먼트 🎤](#카카오엔터테인먼트-)
      - [추천](#추천)
      - [LLM, AI](#llm-ai-1)
  - [카카오헬스케어 💪🏼](#카카오헬스케어-)
      - [LLM](#llm-1)
  - [카카오뱅크 💰](#카카오뱅크-)
      - [LLM](#llm-2)
  - [카카오페이, 카카오페이손해보험 💰](#카카오페이-카카오페이손해보험-)
      - [LLM, AI](#llm-ai-2)
      - [이상치탐지](#이상치탐지-1)
      - [플랫폼](#플랫폼-1)
  - [카카오게임즈 🎮](#카카오게임즈-)
      - [이탈방지](#이탈방지)
      - [AI](#ai)
  - [카카오모빌리티 🚕](#카카오모빌리티-)
      - [AI](#ai-1)
  - [당근 🥕](#당근-)
      - [추천, 검색](#추천-검색-1)
      - [이상치탐지](#이상치탐지-2)
      - [플랫폼](#플랫폼-2)
  - [하이퍼커넥트 👩‍❤️‍👨](#하이퍼커넥트-️)
      - [추천](#추천-1)
      - [AI, ML](#ai-ml)
      - [개발](#개발)
  - [라인 💬](#라인-)
      - [추천](#추천-2)
      - [ML](#ml-1)
  - [네이버 🔍](#네이버-)
      - [추천, 검색](#추천-검색-2)
      - [AI, LLM](#ai-llm)
      - [광고](#광고)
      - [기타](#기타)
  - [네이버웹툰 📖](#네이버웹툰-)
      - [추천, 검색](#추천-검색-3)
      - [타겟팅](#타겟팅)
      - [LLM, AI](#llm-ai-3)
  - [네이버페이 💰](#네이버페이-)
      - [인과추론](#인과추론)
      - [LLM, AI, Agent](#llm-ai-agent)
  - [넥슨 🎮](#넥슨-)
      - [추천](#추천-3)
      - [인과추론](#인과추론-1)
  - [요기요 🍚](#요기요-)
      - [추천](#추천-4)
  - [무신사 👔](#무신사-)
      - [추천, 검색](#추천-검색-4)
  - [라이너 🔍](#라이너-)
      - [추천, 검색](#추천-검색-5)
  - [오늘의집 🏠](#오늘의집-)
      - [추천, 검색](#추천-검색-6)
  - [우아한형제들(배민) 🍚](#우아한형제들배민-)
      - [추천](#추천-5)
      - [LLM, AI](#llm-ai-4)
      - [이상치탐지](#이상치탐지-3)
      - [기타](#기타-1)
      - [플랫폼](#플랫폼-3)
  - [컬리 🍚](#컬리-)
      - [추천](#추천-6)
  - [토스 💰](#토스-)
      - [추천](#추천-7)
      - [플랫폼](#플랫폼-4)
      - [데이터분석](#데이터분석)
  - [Spotify 🎤](#spotify-)
      - [추천, 검색](#추천-검색-7)
  - [Pinterest 🎆](#pinterest-)
      - [추천, 검색](#추천-검색-8)
      - [이상치탐지](#이상치탐지-4)
  - [Ebay 🛒](#ebay-)
      - [추천, 검색](#추천-검색-9)
  - [Nvidia 💻](#nvidia-)
      - [추천, 검색](#추천-검색-10)
    - [인과추론](#인과추론-2)
  - [Uber 🚕](#uber-)
      - [추천, 검색](#추천-검색-11)
  - [Google 🔍](#google-)
      - [추천, 검색](#추천-검색-12)
      - [이상치탐지](#이상치탐지-5)
  - [X (twitter) 📱](#x-twitter-)
      - [추천, 검색](#추천-검색-13)
  - [Meta (facebook) 📱](#meta-facebook-)
      - [추천, 검색](#추천-검색-14)
  - [Netflix 🎬](#netflix-)
      - [추천, 검색](#추천-검색-15)
      - [이상치탐지](#이상치탐지-6)
      - [인과추론](#인과추론-3)
  - [Yelp 🍚](#yelp-)
      - [추천, 검색](#추천-검색-16)
  - [Swiggy 🛒](#swiggy-)
      - [추천, 검색](#추천-검색-17)
  - [Airbnb ✈️](#airbnb-️)
      - [추천, 검색](#추천-검색-18)
  - [Lyft 🚕](#lyft-)
      - [이상치탐지](#이상치탐지-7)
  - [Whatnot 🛒](#whatnot-)
      - [추천, 검색](#추천-검색-19)
  - [Others blog](#others-blog)
      - [추천](#추천-8)
      - [인과추론](#인과추론-4)
- [🧑🏻‍💻 Study \& Practice \& Project](#-study--practice--project)
  - [Language](#language)
  - [ML engineering](#ml-engineering)
  - [Recommendation System](#recommendation-system-1)
  - [Imbalanced Learning, Anomaly Detection](#imbalanced-learning-anomaly-detection-1)
  - [Causality](#causality-2)
  - [NLP](#nlp)
  - [XAI](#xai-1)
  - [Time series](#time-series)
  - [Graph](#graph)
  - [etc](#etc)

# 📑 Paper Reference
## Recommendation System
#### Algorithm
- [Collaborative Filtering for Implicit Feedback Data, 2008](http://yifanhu.net/PUB/cf.pdf)
- [BPR: Bayesian Personalized Ranking from Implicit Feedback, UAI 2009](https://arxiv.org/abs/1205.2618)
- [Context-Aware Recommender Systems, 2011](https://www.researchgate.net/publication/220605653_Context-Aware_Recommender_Systems)
- [Fatorization Machines, 2009](https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf)
- [Wide & Deep Learning for Recommender Systems, 2016](https://arxiv.org/abs/1606.07792)
- [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017](https://arxiv.org/abs/1703.04247)
- [Neural Collaborative Filtering, 2017 IWWWC](https://arxiv.org/abs/1708.05031)
- [Deep content-based music recommendation, 2013 NIPS](https://papers.nips.cc/paper_files/paper/2013/hash/b3ba8f1bee1238a2f37603d90b58898d-Abstract.html)
- [AutoRec: Autoencoders Meet Collaborative Filtering, 2015 WWW](https://dl.acm.org/doi/10.1145/2740908.2742726)
- [Training Deep AutoEncoders for Collaborative Filtering, 2017](https://arxiv.org/abs/1708.01715)
- [Variational Autoencoders for Collaborative Filtering, 2018](https://arxiv.org/abs/1802.05814)
- [Deep Neural Networks for YouTube Recommendations, 2016 RecSys](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45530.pdf)
- [Deep Learning Recommendation Model for Personalization and Recommendation Systems (DLRM), 2019](https://arxiv.org/pdf/1906.00091.pdf)
- [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems 2020](https://arxiv.org/pdf/2008.13535.pdf)
- [Real-time Personalization using Embeddings for Search Ranking at Airbnb, KDD 2018](https://dl.acm.org/doi/pdf/10.1145/3219819.3219885)
- [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations, 2019](https://storage.googleapis.com/gweb-research2023-media/pubtools/5716.pdf)
- [Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations, 2020](https://storage.googleapis.com/gweb-research2023-media/pubtools/6090.pdf)
#### Algorithm - text, image
- [Joint Training of Ratings and Reviews with Recurrent Recommender Nerworks, 2017 ICLR](https://openreview.net/pdf?id=Bkv9FyHYx)
- [Image-based Recommendations on Styles and Substitutes, 2015 SIGIR](https://arxiv.org/abs/1506.04757)
- [VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback, 2016 AAAI](https://arxiv.org/abs/1510.01784)
- [Recommending What Video to Watch Next: A Multitask Ranking System, 2019 RecSys](https://dl.acm.org/doi/10.1145/3298689.3346997)
#### Algorithm - session-based, sequential
- [Session-based Recommendations with Recurrent Neural Networks, 2015 ICLR](https://arxiv.org/abs/1511.06939)
- [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer, 2019](https://arxiv.org/abs/1904.06690)
- [SASRec: Self-Attentive Sequential Recommendation, 2018](https://arxiv.org/abs/1808.09781)
- [Positive, Negative and Neutral: Modeling Implicit Feedback in Session-based News Recommendation, SIGIR 2022](https://arxiv.org/pdf/2205.06058.pdf)
- [TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest, 2023](https://arxiv.org/abs/2306.00248) `Pinterest` `TransAct`
- [TransAct V2: Lifelong User Action Sequence Modeling on Pinterest Recommendation, 2025](https://arxiv.org/abs/2506.02267) `Pinterest` `TransAct v2`
#### Algorithm - graph
- [PageRank: Standing on the shoulders of giant, 2010](https://arxiv.org/pdf/1002.2858.pdf)
- [DeepWalk: Online Learning of Social Representations, 2014](https://arxiv.org/pdf/1403.6652.pdf)
- [SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS, 2017](https://arxiv.org/pdf/1609.02907.pdf)
- [Inductive Representation Learning on Large Graphs, 2017](https://arxiv.org/pdf/1706.02216.pdf)
- [Graph Attention Networks, 2018](https://arxiv.org/pdf/1710.10903.pdf)
- [Graph Convolutional Neural Networks for Web-Scale Recommender Systems, 2018 Pinterest](https://arxiv.org/pdf/1806.01973.pdf) `PinSAGE`
#### Retrieval
- [Recommender Systems with Generative Retrieval, NeurIPS 2023 Google](https://arxiv.org/pdf/2305.05065) `Semantic ID` `RQ-VAE` `generative retrieval` `sequential model`
#### Embedding
- [Matryoshka Representation Learning, NeurIPS 2022](https://arxiv.org/pdf/2205.13147) `embedding` `MRL`
- [Contrastive Conditional Embeddings for Item-based Recommendation at E-commerce Scale, 2025 Rakuten](https://dl.acm.org/doi/epdf/10.1145/3705328.3748095) `item embedding` `constrastive learning`
#### LookAlike
- [Finding Users Who Act Alike: Transfer Learning for Expanding Advertiser Audiences, KDD 2019](https://www.pinterestcareers.com/media/gvnpojec/transferlearning-kdd2019.pdf)
#### Bandit
- [Explore, Exploit, and Explain: Personalizing Explainable Recommendations with Bandits, 2018 spotify](https://static1.squarespace.com/static/5ae0d0b48ab7227d232c2bea/t/5ba849e3c83025fa56814f45/1537755637453/BartRecSys.pdf) `epsilon-greedy`, `explanation`
- [Deep neural network marketplace recommenders in online experiments, 2018](https://arxiv.org/abs/1809.02130) `epsilon-greedy`, `hybrid item-representation`
- [A Batched Multi-Armed Bandit Approach to News Headline Testing, 2019](https://arxiv.org/pdf/1908.06256) `thomson-sampling`, `batched MAB`
- [A Contextual-Bandit Approach to Personalized News Article Recommendation, 2012](https://arxiv.org/abs/1003.0146) `LinUCB`
- [Contextual User Browsing Bandits for Large-Scale Online Mobile Recommendation, 2020 Alibaba](https://arxiv.org/pdf/2008.09368) `UBM-LinUCB`, `contextual combinatorial bandit`
- [An Empirical Evaluation of Thompson Sampling, 2011 Yahoo](https://papers.nips.cc/paper_files/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf)
- [An Efficient Bandit Algorithm for Realtime Multivariate Optimization, 2018 Amazon](https://arxiv.org/abs/1810.09558) `thomson-sampling`
- [Cascading Bandits: Learning to Rank in the Cascade Model, 2015](https://arxiv.org/pdf/1502.02763) `cascade bandit`
- [Carousel Personalization in Music Streaming Apps with Contextual Bandits, 2020 Deezer](https://arxiv.org/pdf/2009.06546) `cascade bandit`, `semi-personalized`
- [Cascading Bandits: Optimizing Recommendation Frequency in Delayed Feedback Environments, 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/f95606d8e870020085990d9650b4f2a1-Paper-Conference.pdf) `cascade bandit`
- [Deep Bayesian Bandits: Exploring in Online Personalized Recommendations, 2020 Twitter](https://arxiv.org/pdf/2008.00727) `deep Bayesian bandits`, `ad display`
- [A Sleeping, Recovering Bandit Algorithm for Optimizing Recurring Notifications, 2020 Duolingo](https://research.duolingo.com/papers/yancey.kdd20.pdf)
#### LLM, Agent
- [LLM-Based Aspect Augmentations for Recommendation Systems, 2023](https://openreview.net/pdf?id=bStpLVqv1H) `item aspect generation`
- [Language-Based User Profiles for Recommendation, 2024](https://arxiv.org/abs/2402.15623) `LFM`
- [Harnessing Large Language Models for Text-Rich Sequential Recommendation, 2024](https://arxiv.org/abs/2403.13325) `text sequential summarize` `SFT`
- [The Unequal Opportunities of Large Language Models: Revealing Demographic Bias through Job Recommendation, 2023](https://dl.acm.org/doi/pdf/10.1145/3617694.3623257) `bias`
- [Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction, 2023 google research](https://arxiv.org/pdf/2305.06474) `prediction`
- [LLMs for User Interest Exploration in Large-scale Recommendation Systems, 2024 google](https://arxiv.org/pdf/2405.16363) `hybrid` `SFT`
- [Comparing Human and LLM Ratings of Music-Recommendation Quality with User Context, 2024](https://openreview.net/pdf?id=QyLYF7SOpS) `LLM-as-a-judge`
- [Playlist Search Reinvented: LLMs Behind the Curtain, 2024 amazon music](https://assets.amazon.science/cb/60/a2cd86b646508da2cc99792481de/playlist-search-reinvented-llms-behind-the-curtain.pdf) `content enrichment` `synthesizing training data` `judges for evaluation`
- [A Multi-Agent Conversational Recommender System, 2024](https://arxiv.org/pdf/2402.01135) `multi-agent` `conversational rec sys`
- [BETTER GENERALIZATION WITH SEMANTIC IDS: A CASE STUDY IN RANKING FOR RECOMMENDATIONS, 2024 google](https://arxiv.org/pdf/2306.08121) `id-based`
- [A Media Content Recommendation Method for Playlist Curatorsusing LLM-Based Query Expansion, 2025 NHK](https://dl.acm.org/doi/epdf/10.1145/3705328.3748129) `media content` `query expansion`
- [Balancing Fine-tuning and RAG: A Hybrid Strategy for Dynamic LLM Recommendation Updates, 2025 google](https://arxiv.org/abs/2510.20260) `LLM based Rec` `interest cluster prediction`
#### Diversity
- [Algorithmic Effects on the Diversity of Consumption on Spotify, 2020](https://www.cs.toronto.edu/~ashton/pubs/alg-effects-spotify-www2020.pdf)
#### Calibration
- [Calibrated Recommendations, 2018 Netflix](https://dl.acm.org/doi/pdf/10.1145/3240323.3240372)
- [Calibrated Recommendations as a Minimum-Cost Flow Problem, 2023 Spotify](https://abdollahpouri.github.io/assets/docs/wsdm2023.pdf)
- [Calibrated Recommendations with Contextual Bandits, 2025 Spotify](https://arxiv.org/pdf/2509.05460)
#### Bias
- Lessons Learned Addressing Dataset Bias in Model-Based Candidate Generation at Twitter, 2020 KDD IRS
- [Popularity-Opportunity Bias in Collaborative Filtering, WSDM 2021](https://dl.acm.org/doi/pdf/10.1145/3437963.3441820)
- [Managing Popularity Bias in Recommender Systems with Personalized Re-ranking, 2019](https://arxiv.org/pdf/1901.07555)
- [The Unfairness of Popularity Bias in Recommendation, 2019](https://arxiv.org/pdf/1907.13286)
- [An Empirical Study of Selection Bias in Pinterest Ads Retrieval, 2023 Pinterest](https://dl.acm.org/doi/epdf/10.1145/3580305.3599771) `selection bias` `online ad` `retrieval`
#### Explainable
- [Faithfully Explaining Rankings in a News Recommender System, 2018](https://arxiv.org/pdf/1805.05447)
#### Cold Start
- [Cold Starting a New Content Type: A Case Study with Netflix Live, 2025 netflix](https://dl.acm.org/doi/epdf/10.1145/3705328.3748112) `new type content cold start`
#### User Modeling
- [Exploring the longitudinal effects of nudging on users’ music genre exploration behavior and listening preferences, 2022](https://dl.acm.org/doi/pdf/10.1145/3523227.3546772)
- [Personalizing Benefits Allocation Without Spending Money: Utilizing Uplift Modeling in a Budget Constrained Setup, Recsys2022](https://dl.acm.org/doi/10.1145/3523227.3547381)
#### Causality
- [Inferring the Causal Impact of New Track Releases on Music Recommendation Platforms through Counterfactual Predictions, RecSys2020](https://labtomarket.files.wordpress.com/2020/08/recsys2020lbr.pdf?utm_source=LinkedIn&utm_medium=post&utm_campaign=monday_posting&utm_term=2023_07_24)
#### Push, CRM
- [Personalized Push Notifications for News Recommendation, 2019 DGP media](https://proceedings.mlr.press/v109/loni19a/loni19a.pdf) `location considered`
- [Predicting which type of push notification content motivates users to engage in a self-monitoring app, 2018](https://www.sciencedirect.com/science/article/pii/S2211335518301177) `statistics analysis` `heavy user, heavy content`
- [Near Real-time Optimization of Activity-based Notifications, 2018 LinkedIn](https://dl.acm.org/doi/pdf/10.1145/3219819.3219880)
- [Notification Volume Control and Optimization System at Pinterest, 2018 Pinterest](https://dl.acm.org/doi/pdf/10.1145/3219819.3219906) `noti volume`
- [Agentic Personalisation of Cross-Channel Marketing Experiences, 2025 AAMPE](https://arxiv.org/abs/2506.16429) `CRM` `causality` `DID` `bandit` `human-in-loop`
#### Ads
- [Cross-Batch Aggregation for Streaming Learning from LabelProportions in Industrial-Scale Recommendation Systems, 2025 Google](https://dl.acm.org/doi/epdf/10.1145/3705328.3748115) `LLP`
- [Decoupled Entity Representation Learning for Pinterest Ads Ranking, 2025 Pinterest](https://arxiv.org/abs/2509.04337) `embedding` `cross domain`
#### Search, Query, IR
- [Query2doc: Query Expansion with Large Language Models, 2023 Microsoft Research](https://arxiv.org/pdf/2303.07678) `query expansion`
- [Query Expansion by Prompting Large Language Models, 2023 Google Research](https://arxiv.org/pdf/2305.03653)
- [An Interactive Query Generation Assistant using LLM-based Prompt Modification and User Feedback, 2023](https://arxiv.org/pdf/2311.11226)
- [InPars: Data Augmentation for Information Retrieval using Large Language Models, 2022](https://arxiv.org/pdf/2202.05144)
- [Generating Query Recommendations via LLMs, 2024 Spotify](https://arxiv.org/pdf/2405.19749) `query expansion` `prompt`
- [Semantic Product Search, 2019 Amazon](https://arxiv.org/pdf/1907.00937) `product search` `semantic` `contrastive learning` `tokenization`
- [Embedding-based Retrieval in Facebook Search, 2020 Facebook](https://arxiv.org/pdf/2006.11632) `social search` `two-tower` `ANN` `negative sampling`
- [Unified Embedding Based Personalized Retrieval in Etsy Search, 2024 Etsy](https://arxiv.org/pdf/2306.04833) `two-tower` `negative sampling` `ANN`
- [Embedding based retrieval for long tail search queries in ecommerce, 2025](https://www.arxiv.org/pdf/2505.01946) `long tail` `llm synthetic data`
- [Towards Personalized and Semantic Retrieval: An End-to-End Solution for E-commerce Search via Embedding Learning, 2020](https://arxiv.org/pdf/2006.02282) `two-tower` `negative sampling` `emb retrieval system`
- [Improving Pinterest Search Relevance Using Large Language Models, 2025 Pinterest](https://arxiv.org/pdf/2410.17152) `KD` `LLM` `enriched text` `augumented data`
#### System
- [Building a Scalable, Effective, and Steerable Search and Ranking Platform, 2024 zalando](https://arxiv.org/pdf/2409.02856) `Platform` `candidate generation` `ranking` `position debias`
- [Joint Modeling of Search and Recommendations Via an Unified Contextual Recommender (UniCoRn), 2024](https://arxiv.org/pdf/2408.10394) `unified model`
#### Survey
- [Deep Learning based Recommender System: A Survey and New Perspectives, 2019](https://arxiv.org/abs/1707.07435)
- [A Survey on Causal Inference for Recommendation, 2024](https://arxiv.org/abs/2303.11666)
- [Fairness and Diversity in Recommender Systems: A Survey, 2024](https://arxiv.org/pdf/2307.04644)
- [Recommender Systems in the Era of Large Language Models (LLMs), 2024](https://arxiv.org/pdf/2307.02046)

## Imbalanced Learning, Anomaly Detection
#### Survey
- [Learning From Imbalanced Data: open challenges and future directions (survey article 2016)](https://link.springer.com/article/10.1007/s13748-016-0094-0)
- [Deep Learning for Anomaly Detection A Review, 2020](https://arxiv.org/pdf/2007.02500.pdf)
- [Autoencoders, 2020](https://arxiv.org/pdf/2003.05991.pdf)
#### Perfomance Measure
- [The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets](https://pubmed.ncbi.nlm.nih.gov/25738806/)
- [The Relationship Between Precision-Recall and ROC Curves](https://www.biostat.wisc.edu/~page/rocpr.pdf)
- [Predicting Good Probabilities With Supervised Learning](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)
- [Properties and benefits of calibrated classifiers](http://www.ifp.illinois.edu/~iracohen/publications/CalibrationECML2004.pdf)
- [The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets](https://www.researchgate.net/publication/273155496_The_Precision-Recall_Plot_Is_More_Informative_than_the_ROC_Plot_When_Evaluating_Binary_Classifiers_on_Imbalanced_Datasets)
####  Cost-sensitive
 - [An optimized cost-sensitive SVM for imbalanced data learning](https://webdocs.cs.ualberta.ca/~zaiane/postscript/pakdd13-1.pdf)
 - [Metacost : a general method for making classifiers cost-sensitive (KDD 99)](https://homes.cs.washington.edu/~pedrod/papers/kdd99.pdf)
 - [The influence of class imbalance on cost-sensitive learning (IEEE 2006)](https://ieeexplore.ieee.org/document/4053137)
 - [Learning and Making Decisions When Costs and Probabilities are Both Unknown (2001)](https://cseweb.ucsd.edu/~elkan/kddbianca.pdf)
#### Sampling
- [SMOTE, 2002](https://arxiv.org/pdf/1106.1813.pdf)
- [SMOTE for learning from imbalanced data : progress and challenges, 2018](https://www.jair.org/index.php/jair/article/view/11192)
- [Influence of minority class instance types on SMOTE imbalanced data oversampling](https://www.researchgate.net/publication/320625181_Influence_of_minority_class_instance_types_on_SMOTE_imbalanced_data_oversampling)
- [Calibrating Probability with Undersampling for Unbalanced Classification .2015](https://www3.nd.edu/~dial/publications/dalpozzolo2015calibrating.pdf)
- [A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data](https://www.researchgate.net/publication/220520041_A_Study_of_the_Behavior_of_Several_Methods_for_Balancing_machine_Learning_Training_Data)
- [Dynamic Sampling in Convolutional Neural Networks for Imbalanced Data Classification](https://users.cs.fiu.edu/~chens/PDF/MIPR18_CNN.pdf)
#### Ensemble Learning
- [Self-paced Ensemble for Highly Imbalanced Massive Data Classification, 2020](https://arxiv.org/abs/1909.03500)
#### Feature Selection
- [Ensemble-based wrapper methods for feature selection and class imbalance learning, 2010](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.706.4216&rep=rep1&type=pdf)
- A comparative study of iterative and non-iterative feature selection techniques for software defect prediction
####  Learning feature representations of normality
 - [Outlier Detection with AutoEncoder Ensemble, 2017](https://saketsathe.net/downloads/autoencoder.pdf)
 - [Auto-Encoding Variational Bayes ,2014](https://arxiv.org/abs/1312.6114)
 - [Deep Variational Information Bottleneck, ICLR 2017](https://arxiv.org/abs/1612.00410)
 - [Extracting and Composing Robust Features with Denoising Autoencoders, 2008](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)
 - [Generatice Adversarial Nets, NIPS 2014](https://papers.nips.cc/paper/2014/hash/5ca3e9b122f61f8f06494c97b1afccf3-Abstract.html)
 - [Least Squares Generative Adversarial Networks ,2016](https://arxiv.org/abs/1611.04076)
 - [Adversarial Autoencoders, 2016](https://arxiv.org/abs/1511.05644)
 - [Generative Probabilistic Novelty Detection with Adversarial Autoencoders , NIPS 2018](https://papers.nips.cc/paper/2018/file/5421e013565f7f1afa0cfe8ad87a99ab-Paper.pdf)
 - [Deep Autoencoding Gaussian Mixture Model For Unsupervised Anomaly Detection, ICLR 2018](https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf)
 - [Anomaly Detection with Robust Deep Autoencoders, KDD 2017](https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p665.pdf)
#### Time Series and Streaming Anomaly Detection
- [Anomaly Detection In Univariate Time-Series : A Survey on the state-of-the-art](https://arxiv.org/abs/2004.00433)
- [USAD : UnSupervised Anomaly Detection on multivariate time series, KDD2020](https://dl.acm.org/doi/10.1145/3394486.3403392)
- [Variational Attention for Sequence-to-Sequence Models, 2017](https://arxiv.org/abs/1712.08207)
- [A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based Variational Autoencoder (2017)](https://arxiv.org/abs/1711.00614)
- [Outlier Detection for Time Series with Recurrent Autoencoder Ensembles , 2019](https://www.ijcai.org/proceedings/2019/0378.pdf)
- [Robust Anomaly Detection for Multivariate time series through Stochastic Recurrent Neural Network, KKD 2019](https://github.com/NetManAIOps/OmniAnomaly)
- [Time Series Anomaly Detection with Multiresolution Ensemble Decoding, AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17152)
- [An Improved Arima-Based Traffic Anomaly Detection Algorithm for Wireless Sensor Networks ,2016](https://journals.sagepub.com/doi/pdf/10.1155/2016/9653230)
- [Time-Series Anomaly Detection Service at Microsoft, 2019](https://arxiv.org/abs/1906.03821)
- [Time Series Anomaly Detection Using Convolutional Neural Networks and Transfer Learning, 2019](https://arxiv.org/pdf/1905.13628.pdf)
- [Abuse and Fraud Detection in Streaming Services Using Heuristic-Aware Machine Learning, 2022 Netflix](https://arxiv.org/pdf/2203.02124.pdf)
- [Are Transformers Effective for Time Series Forecasting?, 2022](https://arxiv.org/pdf/2205.13504.pdf)

## Causality
#### Heterogeneous treatment effect estimation, uplift
- [Causal Inference and Uplift Modeling A review of the literature, 2016](https://proceedings.mlr.press/v67/gutierrez17a/gutierrez17a.pdf) [review](./paper_review/Causal%20Inference%20and%20Uplift%20Modeling%20A%20review%20of%20the%20literature.md)
- [Double machine learning for treatment and causal parameters, 2016](https://www.econstor.eu/bitstream/10419/149795/1/869216953.pdf)
- [Metalearners for estimation heterogeneous treatment effects using machine learning, 2019](https://www.pnas.org/doi/epdf/10.1073/pnas.1804597116)
- [Estimation and Inference of Heterogeneous Treatment Effects using Random Forests, 2018](http://bayes.acs.unt.edu:8083/BayesContent/class/rich/articles/Estimation_And_Inference_Of_Heterogeneous_Treatment_Effects_Using_Random_Forests.pdf)

## LLM
#### LLM judge
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, 2023](https://arxiv.org/abs/2306.05685v4)
- [CheckEval: A reliable LLM-as-a-Judge framework for evaluating text generation using checklists, 2024](https://arxiv.org/abs/2403.18771)
- [A Survey on LLM-as-a-Judge, 2024](https://arxiv.org/abs/2411.15594)

#### Embedding
- [M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation, 2025](https://arxiv.org/pdf/2402.03216) `m3-embedding`

# 📑 Other Reference
## conference
- 네이버, 카카오, 당근, 우아한형제들, 토스 2024 conference [review](./industry/2024_conference/)
- 네이버, 우아한형제들, 토스 2025 conference [review](./industry/2025_conference/)

아래 대부분은 ML/AI 와 관련되었지만 컨퍼런스에서 product 관련 발표내용도 배울 점이 많습니다.

## 카카오 💬
#### 추천, 검색
- 카카오 AI 추천: 카카오페이지와 멜론으로 살펴보는 카카오 연관 추천
- 카카오 AI 추천: 토픽모델링과 MAB를 이용한 카카오 개인화 추천
- 카카오 AI 추천: 협업필터링 모델 선택 시의 기준에 대하여
- 카카오 AI 추천: 카카오의 콘텐츠 기반 필터링
- 우리 생활 속 추천 시스템, 어떻게 발전해왔고, 어떻게 발전해나가고 있는가? (카카오 김성진 팀장, 2021.12)
- 추천 기술이 마주하고 있는 현실적인 문제들 (카카오 김성진 리더, 2021.10)
- if(kakao)2019 멜론 플레이리스트 자동 생성
- if(kakao)2020 맥락과 취향 사이 줄타기
- 브런치 추천의 힘에 대한 6가지 기술
- if(kakao)dev2022 Sequential Recommendation System 카카오 서비스 적용기
- if(kakao)dev2022 Explainable Recommender System in 카카오웹툰
- [커뮤니티로 진화한 오픈채팅, AI로 슬기롭게 연결하다, ifkakao2025](https://www.youtube.com/watch?v=ELVOiHaTj_Y&list=PLwe9WEhzDhwG1H81qHrjc05sj75cGa1fi&index=3)
- [LLM을 활용한 유저 프로파일링과 개인화 추천, ifkakao2025](https://www.youtube.com/watch?v=GaEnbqahnEc&list=PLwe9WEhzDhwG1H81qHrjc05sj75cGa1fi&index=13)
#### LLM, AI
- [이미지까지 이해하는 Multimodal LLM의 학습 방법 밝혀내기, ifkakao2024](https://if.kakao.com/session/12)
- [나만의 프로필 이미지를 만드는 Personalized T2I 모델 개발기, ifkakao2024](https://if.kakao.com/session/17)
- [AI Agent 기반 스마트 AI 마이 노트, ifkakao2024](https://if.kakao.com/session/21)
- [업무 효율화를 위한 카카오 사내봇 개발기, ifkakao2024](https://if.kakao.com/session/26)
- [AI 를 통해 스팸을 대응하는 카카오의 노력, ifkakao2024](https://if.kakao.com/session/30)
- [LLM으로 음성인식 성능 개선하기, ifkakao2024](https://if.kakao.com/session/34)
- [CodeBuddy 와 함께하는 AI 코드리뷰, ifkakao2024](https://if.kakao.com/session/35)
- [AI Assistant와 통합 지식베이스를 통한 AI Native Company 구현, ifkakao2024](https://if.kakao.com/session/41)
- [밑바닥부터 시작하는 LLM 개발기, ifkakao2024](https://if.kakao.com/session/48)
- [AI 기반 광고 콘텐츠 모니터링 기술 개발기, ifkakao2024](https://if.kakao.com/session/59)
- [빠르고 비용효율적으로 LLM 서빙하기, ifkakao2024](https://if.kakao.com/session/53)
- [서비스에 LLM 부스터 달아주기: 요약부터 AI Bot 까지, ifkakao2024](https://if.kakao.com/session/66)
- [‘선물하기 와인탐험’ LLM 대화형 서비스 개발기, ifkakao2024](https://if.kakao.com/session/71)
- [필요한 순간 먼저 말을 걸어주는 온디바이스 AI, ifkakao2025](https://www.youtube.com/watch?v=4wwsyiLmVkA&list=PLwe9WEhzDhwG1H81qHrjc05sj75cGa1fi&index=15)
- [카나나 앱 메이트 개발기, ifkakao2025](https://www.youtube.com/watch?v=V-jlscSg_AA&list=PLwe9WEhzDhwG1H81qHrjc05sj75cGa1fi&index=16)
- [사용자 발화에서 응답까지: 그래프 기반 에이전트로 동작하는 AI 서비스, ifkakao2025](https://www.youtube.com/watch?v=XQo0DHuqgVM&list=PLwe9WEhzDhwGaTnz0fR4Zb817vVCh6T2_&index=2)
- [데이터 뚝딱! AI메이트 메타데이터 생성 비법, ifkakao2025](https://www.youtube.com/watch?v=F4G2E9Np6Wo&list=PLwe9WEhzDhwGaTnz0fR4Zb817vVCh6T2_&index=3)
- [All About LLM: 카카오 AI메이트, 학습부터 서빙까지의 모든 것, ifkakao2025](https://www.youtube.com/watch?v=U0IqPwxydHE&list=PLwe9WEhzDhwGaTnz0fR4Zb817vVCh6T2_&index=4)
- [LLM은 있지만 다시 학습하고 싶어 - Kanana-2 개발기 (~ing), ifkakao2025](https://www.youtube.com/watch?v=B7sc-OlnkCE&list=PLwe9WEhzDhwGaTnz0fR4Zb817vVCh6T2_&index=15)
- [데이터는 없지만 LLM은 학습하고 싶어 - Code, Math 데이터 개발기, ifkakao2025](https://www.youtube.com/watch?v=cL_UjE38_8Q&list=PLwe9WEhzDhwGaTnz0fR4Zb817vVCh6T2_&index=16)
- [LLM은 있지만 컨텍스트가 짧아 - Long Context 실전 적용기, ifkakao2025](https://www.youtube.com/watch?v=mFHjPLa_d-4&list=PLwe9WEhzDhwGaTnz0fR4Zb817vVCh6T2_&index=17)
- [눈으로 보고, 귀로 듣고, 입으로 말하는 AI – 통합 멀티모달 언어모델 Kanana-o 개발기, ifkakao2025](https://www.youtube.com/watch?v=S4CMtHfscis&list=PLwe9WEhzDhwGaTnz0fR4Zb817vVCh6T2_&index=18)
- [화면을 이해하고 행동하는 AI - GUI Agent 개발기, ifkakao2025](https://www.youtube.com/watch?v=k-8CFgBhhu4&list=PLwe9WEhzDhwGaTnz0fR4Zb817vVCh6T2_&index=19)
- [카카오톡 AI 에이전트를 위한 온디바이스 모델 최적화 및 적용, ifkakao2025](https://www.youtube.com/watch?v=8ggESRKrfZ4&list=PLwe9WEhzDhwGaTnz0fR4Zb817vVCh6T2_&index=20)
- [AI Coding Agent: 실험을 통해 확인한 생산성 그 이상의 가능성, ifkakao2025](https://www.youtube.com/watch?v=VdElizwWE4s&list=PLwe9WEhzDhwGaTnz0fR4Zb817vVCh6T2_&index=21)
- [국내 최초 오픈소스 가드레일: Kanana-safeguard, ifkakao2025](https://www.youtube.com/watch?v=hIherrdkvnA&list=PLwe9WEhzDhwGaTnz0fR4Zb817vVCh6T2_&index=22)
- [LLM as a Judge를 활용한 CodeBuddy 성능 평가, 2025](https://tech.kakao.com/posts/690)
#### 이상치탐지
- [그래프 기반 악성 유저군 탐지: 온라인 광고 도메인에서의 적용, ifkakao2024](https://if.kakao.com/session/18)
#### XAI
- [AI를 설명하면서 속도도 빠르게 할 순 없을까? SHAP 가속화 이야기 (feat. 산학협력), ifkakao2024](https://if.kakao.com/session/16)
#### 플랫폼
- [메시지 광고 추천 딥러닝 인퍼런스 서버 개선 - Jvm Onnx Runtime에서  Nvidia Triton 도입까지, ifkakao2024](https://if.kakao.com/session/15)
- [카카오 광고 AI 추천 MLOps 아키텍쳐 - Feature Store 편, ifkakao2024](https://if.kakao.com/session/20)
- [AI 기반 광고 추천 파이프라인에서 스파크 스트리밍의 배포 및 모니터링 전략, ifkakao2024](https://if.kakao.com/session/33)
- [GenAI를 위한 Gateway, ifkakao2025](https://www.youtube.com/watch?v=HxTUBCyCnYM&list=PLwe9WEhzDhwGaTnz0fR4Zb817vVCh6T2_&index=23)
#### ML
- [데이터 분석과 머신러닝을 통한 유저 방문 맛집 발굴하기, ifkakao2024](https://if.kakao.com/session/14)

## 카카오엔터테인먼트 🎤
#### 추천
- [최애 작품 이용권 선물해주는 ‘Helix 푸시’ 개발기 (2024), ifkakao2024](https://if.kakao.com/session/19)
- [음악과 감정을 배우는 AI 모델의 여정: DJ 말랑이, ifkakao2025](https://www.youtube.com/watch?v=7X9whn-SeFk&list=PLwe9WEhzDhwGaTnz0fR4Zb817vVCh6T2_&index=10)
#### LLM, AI
- [지연 시간 순삭! LLM 추론 구조와 효율적 애플리케이션 설계, ifkakao2024](https://if.kakao.com/session/24)
- [웹툰 크리에이터들과의 상생을 위한 숏츠 생성 AI Agent - 헬릭스 숏츠, ifkakao2025](https://www.youtube.com/watch?v=iMR2F5z2n3Y&list=PLwe9WEhzDhwG1H81qHrjc05sj75cGa1fi&index=2)

## 카카오헬스케어 💪🏼
#### LLM
- [생성형 AI를 활용한 개체명 인식(NER), ifkakao2024](https://if.kakao.com/session/22)

## 카카오뱅크 💰
#### LLM
- [이 문자가 스미싱인 이유는? - 스미싱 탐지를 위한 LLM 개발 및 평가,ifkakao2024](https://if.kakao.com/session/23)
- [Prompt Attack에 대항하는 공든 요새 쌓기: lab.fortress, ifkakao2025](https://www.youtube.com/watch?v=Ui6oW-OpI-g&list=PLwe9WEhzDhwGaTnz0fR4Zb817vVCh6T2_&index=5)

## 카카오페이, 카카오페이손해보험 💰
#### LLM, AI
- [LLM 서빙하기, ifkakao2024](https://if.kakao.com/session/25)
- [문서 검토는 이제 Document AI로 한방에!,ifkakao2024](https://if.kakao.com/session/31)
#### 이상치탐지
- [FDS에 지속 성장하는 ML 적용 이야기, ifkakao2024](https://if.kakao.com/session/29)
#### 플랫폼
- [AI 플랫폼 하드웨어부터 코드까지: GPU, LLMOps, Agentic Coding으로 완성하는 AI 플랫폼, ifkakao2025](https://www.youtube.com/watch?v=TofE_ld_pFI&list=PLwe9WEhzDhwGaTnz0fR4Zb817vVCh6T2_&index=10)

## 카카오게임즈 🎮
#### 이탈방지
- [통계를 이용해 이탈을 방지할 수 있을까?, SMART STATS 개발기, ifkakao2024](https://if.kakao.com/session/74)
#### AI
- [김대리, SQL 팡션? 사용하지 마세요. MCP 쓰세요: AI 기술을 이용한 데이터 민주화, ifkakao2025](https://www.youtube.com/watch?v=2v6zhEFh3HI&list=PLwe9WEhzDhwGaTnz0fR4Zb817vVCh6T2_&index=13)
- [AI를 활용한 자동 게임보안 검수 시스템, ifkakao2025](https://www.youtube.com/watch?v=JNO5_JdCR9E&list=PLwe9WEhzDhwGaTnz0fR4Zb817vVCh6T2_&index=14)

## 카카오모빌리티 🚕
#### AI
- [자율주행 AI 실차 적용기: 서비스를 위해 우리가 만들고 있는 자율주행, ifkakao2025](https://www.youtube.com/watch?v=N7CVydbY2gE&list=PLwe9WEhzDhwGaTnz0fR4Zb817vVCh6T2_&index=11)
- [실시간 경로탐색에 Multi-armed Banadit 기반 강화학습 도입하기 (feat. SCI급 논문게재), ifkakao2025](https://www.youtube.com/watch?v=d9c32iTNqS8&list=PLwe9WEhzDhwGaTnz0fR4Zb817vVCh6T2_&index=12)

## 당근 🥕
#### 추천, 검색
- 딥러닝 개인화 추천 (당근마켓, 2019)
- [RAG를 활용한 검색 서비스 만들기, 2025](https://medium.com/daangn/rag%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EA%B2%80%EC%83%89-%EC%84%9C%EB%B9%84%EC%8A%A4-%EB%A7%8C%EB%93%A4%EA%B8%B0-211930ec74a1)
- [검색어에 숨겨진 의도를 더 정확하게, 검색을 바꾸는 AI 실험들 — 당근 AI Show & Tell #5, 2025](https://medium.com/daangn/%EA%B2%80%EC%83%89%EC%96%B4%EC%97%90-%EC%88%A8%EA%B2%A8%EC%A7%84-%EC%9D%98%EB%8F%84%EB%A5%BC-%EB%8D%94-%EC%A0%95%ED%99%95%ED%95%98%EA%B2%8C-%EA%B2%80%EC%83%89%EC%9D%84-%EB%B0%94%EA%BE%B8%EB%8A%94-ai-%EC%8B%A4%ED%97%98%EB%93%A4-14d01677c273)
- [LLM을 활용한 스마트폰 시세 조회 서비스 구축, 2025](https://medium.com/daangn/llm%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%9C-%EC%8A%A4%EB%A7%88%ED%8A%B8%ED%8F%B0-%EC%8B%9C%EC%84%B8%EC%A1%B0%ED%9A%8C-%EC%84%9C%EB%B9%84%EC%8A%A4-%EA%B5%AC%EC%B6%95-bd4650ec67f4)
- [당근 추천 알고리즘 - 홈피드 후보모델 파헤치기, 2024당근테크밋업](https://www.youtube.com/watch?v=qYo0R2nv1PQ&list=PLaHcMRg2hoBryC2cZkhyEin5MrnEJhMUl&index=3)
- [수십억 개 연결이 존재하는 당근 그래프에서 GNN 학습하기, 2024당근테크밋업](https://www.youtube.com/watch?v=R7ecb7xKDj0&list=PLaHcMRg2hoBryC2cZkhyEin5MrnEJhMUl&index=7)
- [중고거래 시멘틱서치 도입기: 꽁꽁 얼어붙은 키워드 위로 벡터가 걸어다닙니다, 2024당근테크밋업](https://www.youtube.com/watch?v=bWfWFAMbJQ4&list=PLaHcMRg2hoBryC2cZkhyEin5MrnEJhMUl&index=4)
#### 이상치탐지
- [ㅎㅖ어져서 팝니ㄷr ☆: LLM과 임베딩 유사도로 빠르게 패턴을 바꾸는 업자 잡아내기, 2024당근테크밋업](https://www.youtube.com/watch?v=UGjRhqZygHg&list=PLaHcMRg2hoBryC2cZkhyEin5MrnEJhMUl&index=1)
- [연간 LLM 호출 비용 25% 절감, 인턴이 도전한 시맨틱 캐싱 도입 기록, 2025](https://medium.com/daangn/llm-%ED%98%B8%EC%B6%9C-%EB%B9%84%EC%9A%A9-00-%EC%A0%88%EA%B0%90-%EC%9D%B8%ED%84%B4%EC%9D%B4-%EB%8F%84%EC%A0%84%ED%95%9C-%EC%8B%9C%EB%A7%A8%ED%8B%B1-%EC%BA%90%EC%8B%B1-%EB%8F%84%EC%9E%85-%EA%B8%B0%EB%A1%9D-d53d951ac10a)
#### 플랫폼
- [당근페이 데이터플랫폼 구축기, 2024당근테크밋업](https://www.youtube.com/watch?v=abdIqj9dxww&list=PLaHcMRg2hoBryC2cZkhyEin5MrnEJhMUl&index=2)
- [지표 통합과 탐색: KarrotMetrics와 Explorer로 가치 있는 의사결정하기, 2024당근테크밋업](https://www.youtube.com/watch?v=I_i3jbQn_tg&list=PLaHcMRg2hoBryC2cZkhyEin5MrnEJhMUl&index=5)
- [추천 서빙 시스템 아키텍처: 높은 생산성을 위한 아키텍쳐 및 ML Flywheel, 2024당근테크밋업](https://www.youtube.com/watch?v=Cs09fzdJo5Y&list=PLaHcMRg2hoBryC2cZkhyEin5MrnEJhMUl&index=6)
- [온콜, 알림만 보다가 죽겠어요, 2024당근테크밋업](https://www.youtube.com/watch?v=4XpZpplWJBw&list=PLaHcMRg2hoBryC2cZkhyEin5MrnEJhMUl&index=8)

## 하이퍼커넥트 👩‍❤️‍👨
#### 추천
- [비용 효율적인 Click-Through Rate Prediction 모델로 하쿠나 라이브 추천시스템 구축하기, 2021](https://hyperconnect.github.io/2021/04/26/hakuna-recsys-gb.html)
- [아자르에서 AI 기반 추천 모델의 타겟 지표를 설정하는 방법 (feat. 아하 모멘트), 2024](https://hyperconnect.github.io/2024/04/26/azar-aha-moment.html)
- [협업 필터링을 넘어서: 하이퍼커넥트 AI의 추천 모델링, 2024](https://hyperconnect.github.io/2024/10/21/beyond-collaborative-filtering.html)
- [이벤트 기반의 라이브 스트리밍 추천 시스템 운용하기, 2022](https://hyperconnect.github.io/2022/01/24/event-driven-recsys.html)
- [아자르에서는 어떤 추천 모델을 사용하고 있을까?, 2024](https://hyperconnect.github.io/2024/11/19/azar-recommendation-model.html)
- [AI 실시간 추천 시스템을 위한 Flink 기반 스트림 조인 서비스 구축기, 2025](https://hyperconnect.github.io/2025/06/11/azar-flink-real-time-stream-join-service.html)
- [클릭 한 번으로 실험 시작! 이터레이션 사이클을 단축하는 추천 실험 시스템 개발기, 2025](https://hyperconnect.github.io/2025/08/26/azar-recsys-experiment-framework.html)
#### AI, ML
- [왜 막상 배포하면 효과가 없지? 타겟 지표에 맞는 ML모델 train/eval 설계하기, 2025](https://hyperconnect.github.io/2025/11/28/how-to-set-ml-objective.html)
- [온디바이스 AI 얼굴 식별 파이프라인 최적화, 2026](https://hyperconnect.github.io/2026/01/23/On-device-Face-Verification-Pipeline-Optimization.html)
#### 개발
- [1:1 비디오 채팅 서비스는 E2E 회귀 테스트를 어떻게 자동화할까?, 2025](https://hyperconnect.github.io/2025/06/12/how-to-automate-e2e-test-on-1on1-video-chat-app.html)

## 라인 💬
#### 추천
- [LINE Timeline의 새로운 도전 (2020.04)](https://engineering.linecorp.com/ko/blog/line-timeline-discover-ml-recommendation)
- [오프라인과 온라인 A/B 테스트를 통해 오픈챗 추천 모델 개선하기, 2023](https://techblog.lycorp.co.jp/ko/improve-openchat-recommendation-model-with-offline-and-online-ab-test)
- [Milvus: LINE VOOM의 실시간 추천 시스템을 위한 대규모 벡터 DB 구축기, 2025](https://techblog.lycorp.co.jp/ko/large-scale-vector-db-for-real-time-recommendation-in-line-voom)
#### ML
- [머신러닝을 활용한 오픈챗 클린 스코어 모델 개발기, 2020](https://engineering.linecorp.com/ko/blog/line-openchat-cleanscore)
- [오픈챗 해시태그 예측을 위한 다중 레이블 분류 모델 개발하기, 2024](https://techblog.lycorp.co.jp/ko/multi-label-classification-model-for-openchat-hashtag-prediction)
- [오픈챗 메시지들로부터 트렌딩 키워드 추출하기, 2025](https://techblog.lycorp.co.jp/ko/extracting-trending-keywords-from-openchat-messages)

## 네이버 🔍
#### 추천, 검색
- (Deview2021) BERT로 만든 네이버 플레이스 비슷한 취향 유저 추천 시스템
- (Deview2021) Knowledge Graph에게 맛집과 사용자를 묻는다: GNN으로 맛집 취향 저격 하기
- (Deview2020) 유저가 좋은 작품(웹툰)을 만났을 때
- (Deview2020) 추천시스템 3.0: 딥러닝 후기시대에서 바이어스, 그래프, 그리고 인과관계의 중요성
- [홈피드: 네이버의 진입점에서 추천피드를 외치다! 추천피드 도입 고군분투기, DAN24](https://tv.naver.com/v/67443984)
- [네이버 검색이 이렇게 좋아졌어? LLM의 Re-Ranking Ability 검색에 이식하기, DAN24](https://tv.naver.com/v/67444172)
- [서치피드: SERP를 넘어 SURF로! 검색의 새로운 물결, DAN24](https://tv.naver.com/v/67444300)
- [검색과 피드의 만남: LLM으로 완성하는 초개인화 서비스, DAN24](https://tv.naver.com/v/67444402)
- [클립 크리에이터와 네이버 유저를 연결하기: 숏폼 컨텐츠 개인화 추천, DAN24](https://tv.naver.com/v/67444550)
- [LLM 기반 추천/광고 파운데이션 모델, DAN24](https://tv.naver.com/v/67445059)
- [사용자 경험을 극대화하는 AI 기반 장소 추천 시스템 : LLM과 유저 데이터의 융합, DAN24](https://tv.naver.com/v/67445325)
- [LLM for Search: 꽁꽁 얼어붙은 검색 서비스 위로 LLM이 걸어다닙니다, DAN24](https://tv.naver.com/v/67452448)
- [사람을 대신해야 진짜 AI지? : LLM 기반 임베딩부터 검색 품질 자동 평가 모델까지, DAN24](https://dan.naver.com/24/sessions/591)
- [SQM으로 네이버 검색 품질 췍↗!, DAN24](https://tv.naver.com/v/67324891)
- [[팀네이버 컨퍼런스 DAN25] 끊임없이 즐거움, 네이버 클립으로 빠져들다: 개인화·VLM·에이전트가 완성하는 네이버 클립 경험, 2025](https://www.youtube.com/watch?v=rLFFX9S9IEk&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=34) `숏폼 추천` `알림 발송` `추천풀 구축` `랭킹`
- [[팀네이버 컨퍼런스 DAN25] From Trend to You: 숏텐츠 키워드 추천의 개인화, 2025](https://www.youtube.com/watch?v=gbxSLh9gsBQ&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=63) `키워드 추출` `랭킹` `LLM Ranker`
- [[팀네이버 컨퍼런스 DAN25] 20년 된 카페의 도전: AI 추천 피드 실험기, 2025](https://www.youtube.com/watch?v=Wbv7Drj14Mo&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=72) `피드 추천`
- [[팀네이버 컨퍼런스 DAN25] F-Solid: RAG에 최적화된 검색아키텍쳐, 2025](https://www.youtube.com/watch?v=T5jgiMYJe4g&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=3) `검색`
- [[팀네이버 컨퍼런스 DAN25] 발췌에서 생성으로: 네이버 검색의 진화, 2025](https://www.youtube.com/watch?v=fE3RlcXpPnk&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=47) `AI 브리핑` `검색` `LLM`
- [[팀네이버 컨퍼런스 DAN25] 검색 서비스에 최적화된 LLM 만들기: 데이터, 학습, 서비스 적용 사례, 2025](https://www.youtube.com/watch?v=VXpaG2iRaXg&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=58) `검색 특화 LLM` `SFT` `RL`
- [[팀네이버 컨퍼런스 DAN25] LLM 라우터: 맥락을 이해하고 분류하는 똑똑한 AI 분류기, 2025](https://www.youtube.com/watch?v=n1C42aEmniQ&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=61)
- [[팀네이버 컨퍼런스 DAN25] NextN Search: 끊임없이 진화하는 네이버 검색, 2025](https://www.youtube.com/watch?v=Cms1auuJi6A&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=64) `VLM` `LLM` `검색` `개인화 추천` `컨텐츠 품질 평가`
- [[팀네이버 컨퍼런스 DAN25] Place AI Briefing: 검색부터 요약까지, 2025](https://www.youtube.com/watch?v=jfFxBhN4bpY&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=65) `검색` `요약` `LLM` `임베딩` `reranker`
- [[팀네이버 컨퍼런스 DAN25] Station M: LLM 그냥 쓰지마세요. 네이버 AI 검색의 모델 오케스트레이션, 2025](https://www.youtube.com/watch?v=k88ps71nnnw&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=66) `LLM`
- [[팀네이버 컨퍼런스 DAN25] 같은 검색 다른 경험: 검색 품질을 넘어 사용자의 만족으로, 2025](https://www.youtube.com/watch?v=jE7wst0YT9w&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=73) `검색 지표` `평가 정량화`
#### AI, LLM
- [eFoundation: 상품을 딱 잘 표현하는 임베딩을 만들었지 뭐야 ꒰⍢꒱ 완전 럭키비키잔앙 ☘︎, DAN24](https://tv.naver.com/v/67444878)
- [어? GPU 그거 어떻게 쓰는건가요? : AI 서빙 헤딩팟 실전 노하우 (feat. AI 이어북)](https://tv.naver.com/v/67327010)
- [속도와 효율의 레이스! : LLM 서빙 최적화의 모든것.](https://tv.naver.com/v/67337608)
- [[팀네이버 컨퍼런스 DAN25] 네이버 PersonA - 지금 나를 이해하는 AI (부제 : LLM 기반 사용자 메모리 구축과 실시간 사용자 로그 반영 시스템 구현), 2025](https://www.youtube.com/watch?v=ZhaZPpCEd38&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=22) `LLM 메모리` `LLM` `개인화 콘텐츠`
- [[팀네이버 컨퍼런스 DAN25] Media AI로 창작되고 소비되는 네이버 동영상, 2025](https://www.youtube.com/watch?v=Nr3Jgnw1LUg&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=29) `vlm` `숏폼생성`
- [[팀네이버 컨퍼런스 DAN25] Papago Speech Translation을 소개 합니다, 2025](https://www.youtube.com/watch?v=uU-pWGk2js8&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=37) `speech translation` `model`
- [[팀네이버 컨퍼런스 DAN25] 데이터 속 숨은 트렌드, LLM이 답하다: 랭킹 기반 플레이스 트렌드 분석 시스템](https://www.youtube.com/watch?v=n2CmC_ZUE-U&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=68) `트렌드 랭킹` `텍스트 마이닝` `키워드 추출` `설명가능성` `LLM` `KD`
- [[팀네이버 컨퍼런스 DAN25] LLM 조각하기: HyperCLOVA-X SEED 모델 학습의 기술, 90x 학습 효율화, 효율적인 LLM 구조, 2025](https://www.youtube.com/watch?v=sLDzTrSkkEQ&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=69)
- [[팀네이버 컨퍼런스 DAN25] Multi-AI Agent 개발: 레퍼런스 아키텍처와 구현 전략, 2025](https://www.youtube.com/watch?v=r6wo53WIpa4&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=70)
#### 광고
- [[팀네이버 컨퍼런스 DAN25] 단 하나의 문장으로: Foundation Model이 답해주는 차세대 Audience Targeting, COMPASS, 2025](https://www.youtube.com/watch?v=2bBHLT2FOeY&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=48) `광고 타겟팅` `FM model` `text2segment`
#### 기타
- [당신의 Python 모델이 이븐하게 추론하지 못하는 이유 [CPU 추론/모델서빙 Python 딥다이브], DAN24](https://tv.naver.com/v/67452152)
- [[팀네이버 컨퍼런스 DAN25] 물과 실험은 셀프입니다: 대규모 검색 서비스의 셀프서브 프로세스 정착기, 2025](https://www.youtube.com/watch?v=cw_gCB8TxrY) `AB test` `metric store` `query view` `사내플랫폼`
- [[팀네이버 컨퍼런스 DAN25] 사용자에 대한 이해로부터 : 검색과 홈피드 서비스의 성장 문법, 2025](https://www.youtube.com/watch?v=t5VwHdnqBFI&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=76) `성장 측정`

## 네이버웹툰 📖
#### 추천, 검색
- [내 손 안의 알딱핑! (੭˃ᴗ˂)੭ 네게 맞는 웹툰을 알아서 딱! 추천해줄게, DAN24](https://tv.naver.com/v/67445183)
- [[팀네이버 컨퍼런스 DAN25] 실시간 추천-CRM 통합 모델로 완성하는 개인화 UX, 2025](http://youtube.com/watch?v=8C1gPGGeRMU&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=49) `추천` `이탈방지` `MMoE`
#### 타겟팅
- [글로벌 웹툰의 ML 기반 CRM 실전 적용기 : Uplift Model과 Survival Model을 활용한 타겟팅 고도화 (네이버 웹툰), DAN24](https://tv.naver.com/v/67320890)
#### LLM, AI
  - [[팀네이버 컨퍼런스 DAN25] 대화형 AI로 바꾼 데이터 협업: 데이터 활용의 장벽을 없애다, 2025](https://www.youtube.com/watch?v=K18evP4Y-1o&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=16) `사내 AI Agent`

## 네이버페이 💰
#### 인과추론
- [Uplift Modeling을 통한 마케팅 비용 최적화 (with Multiple Treatments), 2024](https://medium.com/naverfinancial/uplift-modeling%EC%9D%84-%ED%86%B5%ED%95%9C-%EB%A7%88%EC%BC%80%ED%8C%85-%EB%B9%84%EC%9A%A9-%EC%B5%9C%EC%A0%81%ED%99%94-with-multiple-treatments-5e4e3824b9df)
#### LLM, AI, Agent
- [[팀네이버 컨퍼런스 DAN25] 부동산 매물 검색의 혁명 대화형 AI로 구현하는 검색-분석-투자 통합 생태계, 2025](https://www.youtube.com/watch?v=6F-LZ1DjJik&list=PLCct9_ckBx7-ZsGXBneookwydQNrjO1PW&index=21) `대화형 검색봇`

## 넥슨 🎮
#### 추천
- NDC21-데이터분석, 추천알고리즘 offline A/B 테스트 (feat: PAIGE 프로야구 서비스)
#### 인과추론
- [업리프트 모델링을 통해 게임 광고 전환율 향상시키기, 2023](https://www.intelligencelabs.tech/372bcb48-af74-4a4f-a2b5-57df9e45fcb9)

## 요기요 🍚
#### 추천
- 데이터야놀자2022, 뭐먹지 빌런을 무찌르는 GNN 기반 개인화 추천

## 무신사 👔
#### 추천, 검색
- [무신사가 카테고리숍 추천을 하는 방법, 무신사 2023](https://medium.com/musinsa-tech/%EB%AC%B4%EC%8B%A0%EC%82%AC%EA%B0%80-%EC%B9%B4%ED%85%8C%EA%B3%A0%EB%A6%AC%EC%88%8D-%EC%B6%94%EC%B2%9C%EC%9D%84-%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95-a45b219685ea)
- [검색어 분석을 통한 상품 정렬 개선, 무신사 2021](https://medium.com/musinsa-tech/%EA%B2%80%EC%83%89%EC%96%B4-%EB%B6%84%EC%84%9D%EC%9D%84-%ED%86%B5%ED%95%9C-%EC%83%81%ED%92%88-%EC%A0%95%EB%A0%AC-%EA%B0%9C%EC%84%A0-b92ded2923c3)
- [AI와 함께하는 패션 큐레이션 — 무신사 2.0 시나리오 기반 추천 시스템 개발, 2024](https://medium.com/musinsa-tech/ai%EC%99%80-%ED%95%A8%EA%BB%98%ED%95%98%EB%8A%94-%ED%8C%A8%EC%85%98-%ED%81%90%EB%A0%88%EC%9D%B4%EC%85%98-%EB%AC%B4%EC%8B%A0%EC%82%AC-2-0-%EC%8B%9C%EB%82%98%EB%A6%AC%EC%98%A4-%EA%B8%B0%EB%B0%98-%EC%B6%94%EC%B2%9C-%EC%8B%9C%EC%8A%A4%ED%85%9C-%EA%B0%9C%EB%B0%9C-db7020b20b68)
- [당신이 보는 첫 화면은 어떻게 정해질까? 무신사 홈 배너 개인화 추천 이야기, 2025](https://medium.com/musinsa-tech/%EB%8B%B9%EC%8B%A0%EC%9D%B4-%EB%B3%B4%EB%8A%94-%EC%B2%AB-%ED%99%94%EB%A9%B4%EC%9D%80-%EC%96%B4%EB%96%BB%EA%B2%8C-%EC%A0%95%ED%95%B4%EC%A7%88%EA%B9%8C-%EB%AC%B4%EC%8B%A0%EC%82%AC-%ED%99%88-%EB%B0%B0%EB%84%88-%EA%B0%9C%EC%9D%B8%ED%99%94-%EC%B6%94%EC%B2%9C-%EC%9D%B4%EC%95%BC%EA%B8%B0-33b96721db54)

## 라이너 🔍
#### 추천, 검색
- [TikTok for Text! 라이너 앱 Session-based Recommender 구축기, 라이너 2023](https://blog.getliner.com/sessrec/)
- [Bag-of-Tricks for Recommendation: Recency, Clustering 그리고 Item Shuffling, 2022](https://blog.liner.space/bag-of-tricks-for-recommendation)
- [신뢰성 있는 문서를 골라주기 위한 Liner Ranker, 2024](https://liner.com/ko/blog/liner-ranker)

## 오늘의집 🏠
#### 추천, 검색
- [유사 이미지 추천 개발 #1 비슷한 공간 - 콘텐츠 추천을 위한 이미지 유사도 모델 개발 과정](https://www.bucketplace.com/post/2023-05-22-%EC%9C%A0%EC%82%AC-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%B6%94%EC%B2%9C-%EA%B0%9C%EB%B0%9C-1-%EB%B9%84%EC%8A%B7%ED%95%9C-%EA%B3%B5%EA%B0%84/)
- [유사 이미지 추천 개발 #2 비슷한 상품 - 커머스 상품 추천을 위한 유사도 모델 개발 과정](https://www.bucketplace.com/post/2023-07-13-%EC%9C%A0%EC%82%AC-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%B6%94%EC%B2%9C-%EA%B0%9C%EB%B0%9C-2-%EB%B9%84%EC%8A%B7%ED%95%9C-%EC%83%81%ED%92%88/)
- [개인화 추천 시스템 #1. Multi-Stage Recommender System, 2024](https://www.bucketplace.com/post/2024-03-26-%EA%B0%9C%EC%9D%B8%ED%99%94-%EC%B6%94%EC%B2%9C-%EC%8B%9C%EC%8A%A4%ED%85%9C-1-multi-stage-recommender-system/)
- [개인화 추천 시스템 #2. Personalized Content Ranking, 2024](https://www.bucketplace.com/post/2024-07-10-%EA%B0%9C%EC%9D%B8%ED%99%94-%EC%B6%94%EC%B2%9C-%EC%8B%9C%EC%8A%A4%ED%85%9C-2-personalized-content-ranking/)
- [개인화 추천 시스템 #3. 모델 서빙, 2025](https://www.bucketplace.com/post/2025-03-14-%EA%B0%9C%EC%9D%B8%ED%99%94-%EC%B6%94%EC%B2%9C-%EC%8B%9C%EC%8A%A4%ED%85%9C-3-%EB%AA%A8%EB%8D%B8-%EC%84%9C%EB%B9%99/)
- [개인화 추천 시스템 #4. Feature Store, 2025](https://www.bucketplace.com/post/2025-12-17-%EA%B0%9C%EC%9D%B8%ED%99%94-%EC%B6%94%EC%B2%9C-%EC%8B%9C%EC%8A%A4%ED%85%9C-4-feature-store/)
- [오늘의집 렌즈 - 텍스트를 넘어 이미지로 확장되는 검색 경험, 2025](https://www.bucketplace.com/post/2025-12-02-%EC%98%A4%EB%8A%98%EC%9D%98%EC%A7%91-%EB%A0%8C%EC%A6%88-%ED%85%8D%EC%8A%A4%ED%8A%B8%EB%A5%BC-%EB%84%98%EC%96%B4-%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%A1%9C-%ED%99%95%EC%9E%A5%EB%90%98%EB%8A%94-%EA%B2%80%EC%83%89-%EA%B2%BD%ED%97%98/) `이미지 검색` `객체 탐지`
- [오늘의집 검색 - 모든 언어와 도메인을 아우르는 일반화된 Retrieval 시스템 만들기, 2025](https://www.bucketplace.com/post/2026-01-14-%EC%98%A4%EB%8A%98%EC%9D%98%EC%A7%91-%EA%B2%80%EC%83%89-%EB%AA%A8%EB%93%A0-%EC%96%B8%EC%96%B4%EC%99%80-%EB%8F%84%EB%A9%94%EC%9D%B8%EC%9D%84-%EC%95%84%EC%9A%B0%EB%A5%B4%EB%8A%94-%EC%9D%BC%EB%B0%98%ED%99%94%EB%90%9C-retrieval-%EC%8B%9C%EC%8A%A4%ED%85%9C-%EB%A7%8C%EB%93%A4%EA%B8%B0/) `임베딩 검색` `임베딩 학습`

## 우아한형제들(배민) 🍚
#### 추천
- [(2023 우아콘) 추천시스템 성장 일지: 데이터 엔지니어 편](https://www.youtube.com/watch?v=x49PqlAQC3U&list=PLgXGHBqgT2TundZ81MAVHPzeYOTeII69j&index=13)
- [(2023 우아콘) 여기, 주문하신 '예측' 나왔습니다: 추천/ML에서 '예측'을 서빙한다는 것에 대하여](https://www.youtube.com/watch?v=OXAABJWUgx4&list=PLgXGHBqgT2TundZ81MAVHPzeYOTeII69j&index=15)
- [실시간 반응형 추천 개발 일지 1부: 프로젝트 소개, 2024](https://techblog.woowahan.com/17383/)
- [실시간 반응형 추천 개발 일지 2부: 벡터 검색, 그리고 숨겨진 요구사항과 기술 도입 의사 결정을 다루는 방법, 2025](https://techblog.woowahan.com/21027/)
- [그래프, 텍스트 인코더를 활용한 실시간 추천 검색어 모델링, 우아콘2024](https://www.youtube.com/watch?v=FPdJ24JfKmw&list=PLgXGHBqgT2Tu7H-ita_W0IHospr64ON_a&index=41&pp=iAQB)
- [취향 저격 맛집 추천, 더 똑똑하게: 추천 모델 성장 일지, 우아콘2024](https://www.youtube.com/watch?v=zRLS3_vD1FM&list=PLgXGHBqgT2Tu7H-ita_W0IHospr64ON_a&index=9&pp=iAQB)
#### LLM, AI
- [AI 데이터 분석가 '물어보새' 등장: 데이터 리터러시 향상을 위한 나만의 데이터 분석가, 우아콘2024](https://www.youtube.com/watch?v=_QPhoKItI2k)
- [Fine-tuning 없이, 프롬프트 엔지니어링으로 메뉴 이미지 검수하기, 우아콘2024](https://www.youtube.com/watch?v=YjdZL3Sc9hA&list=PLgXGHBqgT2Tu7H-ita_W0IHospr64ON_a&index=10&pp=iAQB)
- [GPT를 활용한 카탈로그 아이템 생성, 2025](https://techblog.woowahan.com/21294/)
- [AI 데이터 분석가 '물어보새', 그다음: 더 넓고 깊은 지식 공유하기, 우아콘 2025](https://www.youtube.com/watch?v=dLiJXVUF7bQ&list=PLgXGHBqgT2TuPhUGHTe1oFkWJDu3u6-Si&index=9) `사내봇` `멀티 에이전트`
- [AI Platform 2.0: LLMOps로 만드는 비즈니스 레버리지, 우아콘 2025](https://www.youtube.com/watch?v=AhbFFOZUc5s&list=PLgXGHBqgT2TuPhUGHTe1oFkWJDu3u6-Si&index=13) `LLMOps`
- ["어제 쓴 프롬프트, 오늘도 그대로 쓰시나요?", 우아콘 2025](https://www.youtube.com/watch?v=YcA31dmSNMk&list=PLgXGHBqgT2TuPhUGHTe1oFkWJDu3u6-Si&index=37) `프롬프트` `프롬프트 학습법`
#### 이상치탐지
- [배민 앱 리뷰 품질을 향상시킨 방법은? 머신 러닝 X 네트워크 탐지 모델 도입](https://techblog.woowahan.com/11829/)
#### 기타
- [음식 픽업하러 산 넘고 강 건널 수 없으니가: 배달 데이터를 활용해 최적의 지역 클러스터링하기, 우아콘2024](https://www.youtube.com/watch?v=Ub1kL0OB5n8&list=PLgXGHBqgT2Tu7H-ita_W0IHospr64ON_a&index=6)
- [당신에게 배달 시간이 전달되기까지: 불확실성을 다루는 예측 시스템 구축 과정, 우아콘2024](https://www.youtube.com/watch?v=SkliEsGRuSQ&list=PLgXGHBqgT2Tu7H-ita_W0IHospr64ON_a&index=12&pp=iAQB)
- [자율주행 로봇을 위한 머신러닝 모델의 추론 성능을 최적화하기, 우아콘2024](https://www.youtube.com/watch?v=zOJQ4l6cooQ&list=PLgXGHBqgT2Tu7H-ita_W0IHospr64ON_a&index=25&pp=iAQB)
- [로봇 ML 모델의 경량화 1부: 훈련 후 양자화, 2024](https://techblog.woowahan.com/18980/)
- [로봇 ML 모델의 경량화 2부: 양자화 인식 훈련, 2024](https://techblog.woowahan.com/21176/)
- [프로덕트 전략, 어떻게 시작해야 할까?, 2025](https://techblog.woowahan.com/21115/)
#### 플랫폼
- [우아한 데이터 허브, 일 200억 건 데이터 안전하게 처리하는 대용량 시스템 구축하기, 우아콘2024](https://www.youtube.com/watch?v=AtmI56DGhi4&list=PLgXGHBqgT2Tu7H-ita_W0IHospr64ON_a&index=8)
- [장애 같은데? 일단 STOP!: 배달서비스 장애 감지/차단 시스템 구축 경험담, 우아콘2024](https://www.youtube.com/watch?v=NKbmLyWlVpg&list=PLgXGHBqgT2Tu7H-ita_W0IHospr64ON_a&index=28&pp=iAQB)
- [Kafka 파티션 증설 없이 처리량 한계 돌파하기: Parallel Consumer 적용기, 우아콘 2025](https://www.youtube.com/watch?v=UhnERp2AYRo&list=PLgXGHBqgT2TuPhUGHTe1oFkWJDu3u6-Si&index=4)
- [데이터 연금술: 로우 데이터를 파이프라인을 통해 '금'으로 바꾸는 법, 우아콘 2025](https://www.youtube.com/watch?v=tHBPV_iwh00&list=PLgXGHBqgT2TuPhUGHTe1oFkWJDu3u6-Si&index=5)
- [배달의 민족 주문~!: ServerSentEvents로 실시간 알림 전송하기, 우아콘 2025](https://www.youtube.com/watch?v=VCE79fStzA4&list=PLgXGHBqgT2TuPhUGHTe1oFkWJDu3u6-Si&index=6)
- [배달의민족 모든 푸드주문 데이터, 서비스 중단 없이 DocumentDB 전환하기, 우아콘 2025](https://www.youtube.com/watch?v=ROzMnlS1mKQ&list=PLgXGHBqgT2TuPhUGHTe1oFkWJDu3u6-Si&index=11)
- [진화하는 스트리밍 플랫폼: 견고함을 향한 여정과 인사이트, 우아콘 2025](https://www.youtube.com/watch?v=-EP6Wd0qUvE&list=PLgXGHBqgT2TuPhUGHTe1oFkWJDu3u6-Si&index=12)

## 컬리 🍚
#### 추천
- [함께 구매하면 좋은 상품이에요! - 장바구니 추천 개발기 1부, 2024](https://helloworld.kurly.com/blog/cart-recommend-model-development/)
- [함께 구매하면 좋은 상품이에요! - 장바구니 추천 개발기 2부, 2024](https://helloworld.kurly.com/blog/cart-recommend-model-development_second/)

## 토스 💰
#### 추천
- [기반 데이터가 부족해도 OK! 커머스 추천 시스템 제작기, SLASH24](https://www.youtube.com/watch?v=LAD6LYnkPsA&list=PL1DJtS1Hv1PiGXmgruP1_gM2TSvQiOsFL&index=31)
#### 플랫폼
- [ML 플랫폼으로 개발 속도와 안정성 높이기, SLASH24](https://www.youtube.com/watch?v=-im8Gzmf3TM&list=PL1DJtS1Hv1PiGXmgruP1_gM2TSvQiOsFL&index=13)
- [Feature Store로 유연하게 ML 고도화하기, SLASH24](https://www.youtube.com/watch?v=-u3rhd7k2JQ&list=PL1DJtS1Hv1PiGXmgruP1_gM2TSvQiOsFL&index=20)
#### 데이터분석
- [TMC25 | Product - 글로벌 수준의 DA 조직을 만들기 위해 꼭 갖춰야 할 것, 2025](https://www.youtube.com/watch?v=GC5Einv-2iQ)
- [TMC25 | Product - 토스 데이터 분석가가 하는 진짜 A/B테스트, 2025](https://www.youtube.com/watch?v=BljJ1jZ7e3U) `푸시`
- [TMC25 | Product - Metric Review, 실행을 이끌다, 2025](https://www.youtube.com/watch?v=g6XLJgMlvNY)
- [TMC25 | Product - ‘전환율’이 아닌 ‘정답률’, 2025](https://www.youtube.com/watch?v=xxQmEG2_M6A)
- [TMC25 | Product - Data Driven 팀, 이렇게 시작했습니다, 2025](https://www.youtube.com/watch?v=61PYnUA-f_Y) `조직`
- [TMC25 | Product - 데이터 분석팀 리더는 처음이라, 2025](https://www.youtube.com/watch?v=1WnsdCrD1sM) `조직`
- [TMC25 | Product - 중층적, 고객중심적, 동적으로 접근하는 제품 성장, 2025](https://www.youtube.com/watch?v=Peyr3WAaNBY&list=PL-245_cWUfl98ZlF6xAZDVFc8HP33GJA5&index=27) `프로덕트 분석` `인과관계`

## Spotify 🎤
#### 추천, 검색
- [The Rise (and Lessons Learned) of ML Models to Personalize Content on Home, 2021](https://engineering.atspotify.com/2021/11/the-rise-and-lessons-learned-of-ml-models-to-personalize-content-on-home-part-i/)
- [Introducing Natural Language Search for Podcast Episodes, spotify 2022](https://engineering.atspotify.com/2022/03/introducing-natural-language-search-for-podcast-episodes/)
- [Modeling Users According to Their Slow and Fast-Moving Interests, spotify 2022](https://research.atspotify.com/2022/02/modeling-users-according-to-their-slow-and-fast-moving-interests/)
- [Researching how less-streamed podcasts can reach their potential, 2022](https://research.atspotify.com/2022/1/researching-how-less-streamed-podcasts-can-reach-their-potential)
- [Socially-Motivated Music Recommendation, 2024](https://research.atspotify.com/2024/06/socially-motivated-music-recommendation/)
- [Personalizing Audiobooks and Podcasts with graph-based models, 2024](https://research.atspotify.com/2024/05/personalizing-audiobooks-and-podcasts-with-graph-based-models/)
- [Teaching Large Language Models to Speak Spotify: How Semantic IDs Enable Personalization, 2025](https://research.atspotify.com/2025/11/teaching-large-language-models-to-speak-spotify-how-semantic-ids-enable) `Semantic IDs` `LLM`
- [Spillover Detection for Donor Selection in Synthetic Control Models, 2025](https://research.atspotify.com/2025/11/spillover-detection-for-donor-selection-in-synthetic-control-models)

## Pinterest 🎆
#### 추천, 검색
- [The machine learning behind delivering relevant ads, Pinterest 2021](https://medium.com/pinterest-engineering/the-machine-learning-behind-delivering-relevant-ads-8987fc5ba1c0)
- [Feature Caching for Recommender Systems w/ Cachelib, 2024](https://medium.com/pinterest-engineering/feature-caching-for-recommender-systems-w-cachelib-8fb7bacc2762)
- [Advancements in Embedding-Based Retrieval at Pinterest Homefeed, 2025](https://medium.com/pinterest-engineering/advancements-in-embedding-based-retrieval-at-pinterest-homefeed-d7d7971a409e) `two-tower` `conditional retrieval`
- [Foundation Model for Personalized Recommendation, 2025](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39) `foundation model`
- [Improving Pinterest Search Relevance Using Large Language Models, 2025](https://medium.com/pinterest-engineering/improving-pinterest-search-relevance-using-large-language-models-4cd938d4e892) `KD` `enriched text` `augmented label`
- [Modernizing Home Feed Pre-Ranking Stage, 2025](https://medium.com/pinterest-engineering/modernizing-home-feed-pre-ranking-stage-e636c9cdc36b) `pre-ranking` `sample selection bias` `serving` `early funnel log`
- [Deep Multi-task Learning and Real-time Personalization for Closeup Recommendations, 2025](https://medium.com/pinterest-engineering/deep-multi-task-learning-and-real-time-personalization-for-closeup-recommendations-1030edfe445f)
- [Next-Level Personalization: How 16k+ Lifelong User Actions Supercharge Pinterest’s Recommendations, 2025](https://medium.com/pinterest-engineering/next-level-personalization-how-16k-lifelong-user-actions-supercharge-pinterests-recommendations-bd5989f8f5d3) `TransActV2` `lifelong user sequence` `Next Action Loss`
- [Unlocking Efficient Ad Retrieval: Offline Approximate Nearest Neighbors in Pinterest Ads, 2025](https://medium.com/pinterest-engineering/u~nlocking-efficient-ad-retrieval-offline-approximate-nearest-neighbors-in-pinterest-ads-6fccc131ac14)
- [Improving Quality of Recommended Content through Pinner Surveys, 2025](https://medium.com/pinterest-engineering/improving-quality-of-recommended-content-through-pinner-surveys-eebca8a52652) `quality model` `survey`
#### 이상치탐지
- [Warden: Real Time Anomaly Detection at Pinterest](https://medium.com/pinterest-engineering/warden-real-time-anomaly-detection-at-pinterest-210c122f6afa)
- [Fighting Spam using Clustering and Automated Rule Creation](https://medium.com/pinterest-engineering/fighting-spam-using-clustering-and-automated-rule-creation-1c01d8c11a05)

## Ebay 🛒
#### 추천, 검색
- [Building a Deep Learning Based Retrieval System for Personalized Recommendations, ebay 2022](https://tech.ebayinc.com/engineering/building-a-deep-learning-based-retrieval-system-for-personalized-recommendations/)

## Nvidia 💻
#### 추천, 검색
- [Recommender Systems, Not Just Recommender Models, nvidia merlin](https://medium.com/nvidia-merlin/recommender-systems-not-just-recommender-models-485c161c755e)
### 인과추론
- [Using Causal Inference to Improve the Uber User Experience, 2019](https://www.uber.com/en-KR/blog/causal-inference-at-uber/)

## Uber 🚕
#### 추천, 검색
- [Innovative Recommendation Applications Using Two Tower Embeddings at Uber, Uber 2023](https://www.uber.com/en-KR/blog/innovative-recommendation-applications-using-two-tower-embeddings/)
- [Enhancing Personalized CRM Communication with Contextual Bandit Strategies, 2025](https://www.uber.com/en-IN/blog/enhancing-personalized-crm/)

## Google 🔍
#### 추천, 검색
- [Scaling deep retrieval with TensorFlow Recommenders and Vertex AI Matching Engine](https://cloud.google.com/blog/products/ai-machine-learning/scaling-deep-retrieval-tensorflow-two-towers-architecture?hl=en)
#### 이상치탐지
- [Unsupervised and semi-supervised anomaly detection with data-centric ML, google blog](https://ai.googleblog.com/2023/02/unsupervised-and-semi-supervised.html)

## X (twitter) 📱
#### 추천, 검색
- [GraphJet: Real-Time Content Recommendations at Twitter, 2016](https://www.vldb.org/pvldb/vol9/p1281-sharma.pdf)
- [Model-based candidate generation for account recommendations, X 2022](https://blog.twitter.com/engineering/en_us/topics/insights/2022/model-based-candidate-generation-for-account-recommendations)
- [A hybrid approach to personalize notification volume, 2022](https://blog.x.com/engineering/en_us/topics/insights/2022/a-hybrid-approach-to-personalize-notification-volume)

## Meta (facebook) 📱
#### 추천, 검색
- [Scaling the Instagram Explore recommendations system, meta 2023](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/)
- [How machine learning powers Facebook’s News Feed ranking algorithm, meta 2021](https://engineering.fb.com/2021/01/26/ml-applications/news-feed-ranking/)

## Netflix 🎬
#### 추천, 검색
- [Innovating Faster on Personalization Algorithms at Netflix Using Interleaving, 2017](https://netflixtechblog.com/interleaving-in-online-experiments-at-netflix-a04ee392ec55)
#### 이상치탐지
- [Machine Learning for Fraud Detection in Streaming Services](https://netflixtechblog.medium.com/machine-learning-for-fraud-detection-in-streaming-services-b0b4ef3be3f6)
#### 인과추론
- [A Survey of Causal Inference Applications at Netflix](https://netflixtechblog.com/a-survey-of-causal-inference-applications-at-netflix-b62d25175e6f)

## Yelp 🍚
#### 추천, 검색
- [Search Query Understanding with LLMs: From Ideation to Production, 2025](https://engineeringblog.yelp.com/2025/02/search-query-understanding-with-LLMs.html)

## Swiggy 🛒
#### 추천, 검색
- [Contextual Bandits for Ads Recommendations, 2022](https://bytes.swiggy.com/contextual-bandits-for-ads-recommendations-ec210775fcf)
- [Smart Push notifications (Multi-Armed Bandits at Swiggy: Part-4)](https://bytes.swiggy.com/smart-push-notifications-multi-armed-bandits-at-swiggy-part-4-f5698f2af0a6)

## Airbnb ✈️
#### 추천, 검색
- [Embedding-Based Retrieval for Airbnb Search, 2025](https://medium.com/airbnb-engineering/embedding-based-retrieval-for-airbnb-search-aabebfc85839)
- [Improving Search Ranking for Maps, 2024](https://medium.com/airbnb-engineering/improving-search-ranking-for-maps-13b03f2c2cca)

## Lyft 🚕
#### 이상치탐지
- [Full-Spectrum ML Model Monitoring at Lyft](https://eng.lyft.com/full-spectrum-ml-model-monitoring-at-lyft-a4cdaf828e8f)
- [Building a large scale unsupervised model anomaly detection system — Part 1](https://eng.lyft.com/building-a-large-scale-unsupervised-model-anomaly-detection-system-part-1-aca4766a823c)
- [Building a large scale unsupervised model anomaly detection system — Part 2](https://eng.lyft.com/building-a-large-scale-unsupervised-model-anomaly-detection-system-part-2-3690f4c37c5b)

## Whatnot 🛒
#### 추천, 검색
- [Evolving Feed Ranking at Whatnot, 2025](https://medium.com/whatnot-engineering/evolving-feed-ranking-at-whatnot-25adb116aeb6)

## Others blog
#### 추천
- Transformers4Rec: A flexible library for Sequential and Session-based recommendation
- [[22'Recsys] BERT4Rec 구현의 진실에 관하여 : A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation](https://mytype9591.tistory.com/m/6)
- [Scaling deep retrieval with TensorFlow Recommenders and Vertex AI Matching Engine](https://cloud.google.com/blog/products/ai-machine-learning/scaling-deep-retrieval-tensorflow-two-towers-architecture?hl=en)
- [DLRM github](https://github.com/facebookresearch/dlrm)
- [DeepCTR github](https://github.com/shenweichen/DeepCTR)
- [Two Tower Model Architecture: Current State and Promising Extensions, 2023](https://blog.reachsumit.com/posts/2023/03/two-tower-model/)
- [추천 시스템 서비스 적용을 위한 Elastic Search 도입기, 2022](https://blog.dramancompany.com/2022/11/%EC%B6%94%EC%B2%9C-%EC%8B%9C%EC%8A%A4%ED%85%9C-%EC%84%9C%EB%B9%84%EC%8A%A4-%EC%A0%81%EC%9A%A9%EC%9D%84-%EC%9C%84%ED%95%9C-elastic-search-%EB%8F%84%EC%9E%85%EA%B8%B0/)
- [Recommendation Systems • Bias](https://aman.ai/recsys/bias/)
- [Search: Query Matching via Lexical, Graph, and Embedding Methods, 2021](https://eugeneyan.com/writing/search-query-matching/)
- [Patterns for Personalization in Recommendations and Search, 2021](https://eugeneyan.com/writing/patterns-for-personalization/)
- [Real-time Machine Learning For Recommendations, 2021](https://eugeneyan.com/writing/real-time-recommendations/)
- [System Design for Recommendations and Search, 2021](https://eugeneyan.com/writing/system-design-for-discovery/)
- [Improving Recommendation Systems & Search in the Age of LLMs, 2025](https://eugeneyan.com/writing/recsys-llm/)
- [Push Notifications: What to Push, What Not to Push, and How Often, 2023](https://eugeneyan.com/writing/push/)
#### 인과추론
- [Brady Neal - Causal Inference](https://www.youtube.com/c/BradyNealCausalInference/playlists)
- [인과추론의 데이터과학](https://www.youtube.com/c/%EC%9D%B8%EA%B3%BC%EC%B6%94%EB%A1%A0%EC%9D%98%EB%8D%B0%EC%9D%B4%ED%84%B0%EA%B3%BC%ED%95%99/playlists)
- [EconML/CausalML KDD 2021 Tutorial](https://causal-machine-learning.github.io/kdd2021-tutorial/)
- [Causal Inference for the Brave and True](https://matheusfacure.github.io/python-causality-handbook/01-Introduction-To-Causality.html)
- [Dowhy 가이드 실습 pap gitbook](https://playinpap.gitbook.io/dowhy/)
- [Causal-Inference-and-Discovery-in-Python](https://github.com/PacktPublishing/Causal-Inference-and-Discovery-in-Python)
- [For effective treatment of churn, don’t predict churn, 2019](https://medium.com/bigdatarepublic/for-effective-treatment-of-churn-dont-predict-churn-58328967ec4f)
- [Causal inference 123, Microsoft Shujuan(Jane) Huang 2020](https://medium.com/@shujuanhuang)

# 🧑🏻‍💻 Study & Practice & Project

개인적으로 공부하는 모든 내용을 작성하지는 않고 주로 코드나 기록으로 남아있는 경우만 index 용으로 작성합니다.

## Language
- c++ 기본 공부, 2020 [repo](https://github.com/minsoo9506/c-and-cpp)
- Python 심화 공부, 2023 [repo](https://github.com/minsoo9506/advanced-python-study)
- Java 기본 공부, 2025
- Rust 기본 공부, 2026 [repo](https://github.com/minsoo9506/rust-basic-study)

## ML engineering
- Deployment of ML Models 공부, 2022 [repo](https://github.com/minsoo9506/fraudDetection-python-package)
- Bentoml tutorial, 2022 [repo](https://github.com/minsoo9506/BentoML-model-serving)
- Airflow 공부, 2022 [review](https://minsoo9506.github.io/categories/airflow/)
- Docker, K8s 공부, 2023 [review](https://minsoo9506.github.io/categories/)
- ML Engineer를 위한 MLOps 튜토리얼, 2023 [repo](https://github.com/minsoo9506/mlops-project-level0)
- ML Testing Monitoring 공부, 2023 [repo](https://github.com/minsoo9506/ML-testing-monitoring)
- FastAPI 공부, 2023 [repo](https://github.com/minsoo9506/FastAPI-study)
- 디자인 패턴 공부, 2024 [repo](https://github.com/minsoo9506/design-pattern-study)
- 머신러닝 디자인 패턴 공부, 2025 [repo](https://github.com/minsoo9506/machine-learning-design-pattern)
- 백엔드 프로그래밍 (FastAPI), 2026 [repo](https://github.com/minsoo9506/TILAPP)

## Recommendation System
- 추천 모델 구현, 2023 [repo](https://github.com/minsoo9506/RecModel)
- bandit 공부, 2025 [repo](https://github.com/minsoo9506/bandit-study)

## Imbalanced Learning, Anomaly Detection
- Dacon 신용카드 사용자 연체 예측 AI 경진대회, 2022 [code](./practice/Dacon%20신용카드%20사용자%20연체%20예측/)
- Kaggle Credit Card Fraud Detection, 2022 [code](./practice/Kaggle%20Credit%20Card%20Fraud%20Detection/)
- 네트워크임베딩 대학원수업 기말 프로젝트 (Anomaly Detection with Graph Embedding Ensemble) [pdf](./practice/Anomaly_Detection_with_Graph_Embedding_Ensemble.pdf)
- 모델 구현 (라이브러리화), 2023 [repo](https://github.com/minsoo9506/catchMinor)

## Causality
- Brady Neal - Causal Inference 공부 [review](https://minsoo9506.github.io/categories/causality/)
- Causal Inference for the Brave and True 공부 [review](./practice/Causal_Inference_for_the_Brave_and_True_practice/)
- DoWhy tutorial [review](./practice/DoWhy_tutorial/)
- Heterogeneous Treatment Effect Estimation tutorial [review](./practice/heterogeneous_treatment_effect_estimation_tutorial/)

## NLP
- NLP 공부, 2020 [repo](https://github.com/minsoo9506/NLP-study/commits/master/)
- hugginface text classification, 2022 [repo](https://github.com/minsoo9506/huggingface-text-classification)

## XAI
- XAI 기본 공부, 2019 [repo](https://github.com/minsoo9506/Interpretable-Machine-Learning)

## Time series
- time series 공부, 2021 [repo](https://github.com/minsoo9506/time-series-study)

## Graph
- network embedding 공부, 2021 [repo](https://github.com/minsoo9506/network-embedding)
- cs224w 공부, 2022 [repo](https://github.com/minsoo9506/cs224w-study)

## etc
- 개인 블로그 공부 정리 [blog](https://minsoo9506.github.io/categories/)