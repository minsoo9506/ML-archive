요즘은 추천시스템 위주의 공부를 하고 있습니다.

# Index
- Paper: 논문의 주요 내용을 정리합니다.
- Industry: 회사들의 tech blog 주요 내용을 정리합니다.
- Practice & Project: 공부하는 과정에서 기록으로 남기고 싶은 내용을 정리합니다.
- Reference: 다양한 레퍼런스 아카이빙합니다.

# 📄 Paper
### Recommendation System
- Factorization Machines, 2010
  - [`paper`](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf), [`review`](./paper/2023_08_28_Factorization_Machine.md), [`code`](https://github.com/minsoo9506/RecModel)
- Wide & Deep Learning for Recommender Systems, 2016
  - [`paper`](https://arxiv.org/abs/1606.07792), [`review`](./paper/2023_09_02_Wide_and_Deep.md)
- Neural Collaborative Filtering, 2017
  - [`paper`](https://arxiv.org/pdf/1708.05031.pdf), [`review`](./paper/2023_09_07_NCF.md), [`code`](https://github.com/minsoo9506/RecModel) 
- DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017
  - [`paper`](https://arxiv.org/abs/1703.04247), [`review`](./paper/2023_08_30_DeepFM.md), [`code`](https://github.com/minsoo9506/RecModel)
- Real-time Personalization using Embeddings for Search Ranking at Airbnb, KDD 2018
  - [`paper`](https://dl.acm.org/doi/pdf/10.1145/3219819.3219885), [`review`](./paper/2023_09_10_Search_Ranking_Airbnb_2018.md)
- Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations, RecSys 2019
  - [`paper`](https://research.google/pubs/pub48840/), [`review`](./paper/2023_09_21_Two_Tower_in_batch.md), [`code`](https://github.com/minsoo9506/RecModel)
- DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems 2020
  - [`paper`](https://arxiv.org/pdf/2008.13535.pdf), [`review`](./paper/2023_10_03_DCN2.md)

### Imbalanced Learning, Anomaly Detection
- Deep Learning for Anomaly Detection A Review, 2020
  - [`paper`](https://arxiv.org/pdf/2007.02500.pdf), [`review`](./paper/Deep_Learning_for_Anomaly_Detection_A_Review.md)
- Deep Autoencoding Gaussian Mixture Model For Unsupervised Anomaly Detection, ICLR 2018
  - [`paper`](https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf), [`review`](./paper/DAGMM.pdf)
- Adversarial Autoencoders, 2016
  - [`paper`](https://arxiv.org/abs/1511.05644), [`review`](./paper/Adversarial_Autoencoders.pdf)
- USAD : UnSupervised Anomaly Detection on multivariate time series, KDD2020
  - [`paper`](https://dl.acm.org/doi/10.1145/3394486.3403392), [`review`](./paper/USAD.pdf)

# 🏢 Industry
### Recommendation System
#### Spotify
- The Rise (and Lessons Learned) of ML Models to Personalize Content on Home, 2021
  - [`blog`](https://engineering.atspotify.com/2021/11/the-rise-and-lessons-learned-of-ml-models-to-personalize-content-on-home-part-i/), [`review`](./industry/Spotify/2023_10_22_Spotify_The_Rise_(and%20Lessons%20Learned)_of_ML_Models_to_Personalize_Content_on_Home.md)
- Introducing Natural Language Search for Podcast Episodes, 2022
  - [`blog`](https://engineering.atspotify.com/2022/03/introducing-natural-language-search-for-podcast-episodes/), [`review`](./industry/Spotify/2023_10_14_Spotify_Introducing_Natural_Language_Search_for_Podcast_Episodes.md)
- Modeling Users According to Their Slow and Fast-Moving Interests, 2022
  - [`blog`](https://research.atspotify.com/2022/02/modeling-users-according-to-their-slow-and-fast-moving-interests/), [`review`](./industry/Spotify/2023_10_16_Spotify_Modeling_Users_According_to_Their_Slow_and_Fast-Moving_Interests.md)

#### Pinterest
- The machine learning behind delivering relevant ads, 2021
  - [`blog`](https://medium.com/pinterest-engineering/the-machine-learning-behind-delivering-relevant-ads-8987fc5ba1c0), [`review`](./industry/Pinterest/2023_10_28_Pinterest_The_machine_learning_behind_delivering_relevants_ads.md)
- Pinterest Home Feed Unified Lightweight Scoring: A Two-tower Approach, 2021
  - [`blog`](https://medium.com/pinterest-engineering/pinterest-home-feed-unified-lightweight-scoring-a-two-tower-approach-b3143ac70b55), [`review`](./industry/Pinterest/2023_11_02_Pinterest_Home_Feed_Unified_Lightweight_Scoring_A_Two-tower_Approach.md)
- Query Rewards: Building a Recommendation Feedback Loop During Query Selection, 2022
  - [`blog`](https://medium.com/pinterest-engineering/query-rewards-building-a-recommendation-feedback-loop-during-query-selection-70b4d20e5ea0), [`review`](./industry/Pinterest/2023_11_12_Pinterest_Query_Rewards_Building_a_Recommendation_Feedback_Loop_During_Query_Selection.md)
- How Pinterest Leverages Realtime User Actions in Recommendation to Boost Homefeed Engagement Volume, 2022
  - [`blog`](https://medium.com/pinterest-engineering/), [`review`](./industry/Pinterest/2023_12_10_How_Pinterest_Leverages_Realtime_User_Actions_in_Recommendation_to_Boost_Homefeed_Engagement_Volume_2022.md)

#### Meta(Facebook)
- How Instagram suggests new content, 2020
  - [`blog`](https://engineering.fb.com/2020/12/10/web/how-instagram-suggests-new-content/), [`review`](./industry/Meta(Facebook)/2023_12_17_How_Instagram_suggests_new_content_2020.md)

#### LinkedIn
- The AI Behind LinkedIn Recruiter search and recommendation systems, 2019
  - [`blog`](https://engineering.linkedin.com/blog/2019/04/ai-behind-linkedin-recruiter-search-and-recommendation-systems), [`review`](./industry/LikedIn/2023_12-17_The_AI_Behind_LinkedIn_Recruiter_search_and_recommendation_systems.md)
- Near real-time features for near real-time personalization, 2022
  - [`blog`](https://engineering.linkedin.com/blog/2022/near-real-time-features-for-near-real-time-personalization), [`review`](./industry/LikedIn/2024_01_03_Near_real-time_features_for_near_real-time_personalization.md)
- Community building recommender for group chats in LinkedIn Messaging, 2022
  - [`blog`](https://engineering.linkedin.com/blog/2022/community-building-recommender-for-group-chats-in-linkedin-messa), [`review`](./industry/LikedIn/2024_01_27_Community_building_recommender_for_group_chats_in_LinkedIn_Messaging_2022.md)

### Conference
- 네이버, 카카오, 당근, 우아한형제들, 토스 2024 conference [`review`](./industry/2024_conference/)

# 🧑🏻‍💻 Practice & Project
### Recommendation System
- 추천 모델 구현, 2023 [`repository`](https://github.com/minsoo9506/RecModel)
- bandit 공부, 2025  [`repository`](https://github.com/minsoo9506/bandit-study)

### ML engineering
- udemy 강의 deployment of ml models, 2022 [`repository`](https://github.com/minsoo9506/fraudDetection-python-package)
- bentoml tutorial, 2022 [`repository`](https://github.com/minsoo9506/BentoML-model-serving)
- mlops tutorial, 2023 [`repository`](https://github.com/minsoo9506/mlops-project-level0)
- udemy 강의 ml testing monitoring, 2023 [`repository`](https://github.com/minsoo9506/ML-testing-monitoring)
- FastAPI study, 2023 [`repository`](https://github.com/minsoo9506/FastAPI-study)

### Imbalanced Learning, Anomaly Detection
- Dacon 신용카드 사용자 연체 예측 AI 경진대회, 2022 [`code`](./practice/Dacon%20신용카드%20사용자%20연체%20예측/)
- Kaggle Credit Card Fraud Detection, 2022 [`code`](./practice/Kaggle%20Credit%20Card%20Fraud%20Detection/)
- 네트워크임베딩 대학원수업 기말 프로젝트 (Anomaly Detection with Graph Embedding Ensemble) [`pdf`](./practice/Anomaly_Detection_with_Graph_Embedding_Ensemble.pdf)
- 모델 구현 (라이브러리화), 2023 [`repository`](https://github.com/minsoo9506/catchMinor)

### Causality
- Brady Neal - Causal Inference [`review`](https://minsoo9506.github.io/categories/causality/)
- Causal Inference for the Brave and True [`review`](./practice/Causal_Inference_for_the_Brave_and_True_practice/)
- DoWhy tutorial [`review`](./practice/DoWhy_tutorial/)
- Heterogeneous Treatment Effect Estimation tutorial [`review`](./practice/heterogeneous_treatment_effect_estimation_tutorial/)

### NLP
- nlp study, 2020 [`repository`](https://github.com/minsoo9506/NLP-study/commits/master/)
- hugginface text classification, 2022 [`repository`](https://github.com/minsoo9506/huggingface-text-classification)

### etc
- 개인 블로그 공부 정리 [`blog`](https://minsoo9506.github.io/categories/)

# 📑 Paper Reference
### Recommendation System
- Algorithm
  - Collaborative Filtering for Implicit Feedback Data, 2008
  - BPR: Bayesian Personalized Ranking from Implicit Feedback, UAI 2009
  - Context-Aware Recommender Systems, 2011
  - Neural Collaborative Filtering, 2017 IWWWC
  - Fatorization Machines, 2009
  - Wide & Deep Learning for Recommender Systems, 2016
  - DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017
  - AutoRec: Autoencoders Meet Collaborative Filtering, 2015 WWW
  - Training Deep AutoEncoders for Collaborative Filtering, 2017
  - Variational Autoencoders for Collaborative Filtering, 2018
  - Deep content-based music recommendation, 2013 NIPS
  - [Deep Learning Recommendation Model for Personalization and Recommendation Systems (DLRM), 2019](https://arxiv.org/pdf/1906.00091.pdf)
  - [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems 2020](https://arxiv.org/pdf/2008.13535.pdf)
  - [Real-time Personalization using Embeddings for Search Ranking at Airbnb, KDD 2018](https://dl.acm.org/doi/pdf/10.1145/3219819.3219885)
  - [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations, 2019](https://storage.googleapis.com/gweb-research2023-media/pubtools/5716.pdf)
  - [Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations, 2020](https://storage.googleapis.com/gweb-research2023-media/pubtools/6090.pdf)
- Algorithm - text, image
  - Joint Training of Ratings and Reviews with Recurrent Recommender Nerworks, 2017 ICLR
  - Image-based Recommendations on Styles and Substitutes, 2015 SIGIR
  - VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback, 2016 AAAI
  - [Deep Neural Networks for YouTube Recommendations, 2016 RecSys](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45530.pdf)
  - Recommending What Video to Watch Next: A Multitask Ranking System, 2019 RecSys
- Algorithm - session-based, sequential
  - Session-based Recommendations with Recurrent Neural Networks, 2015 ICLR
  - [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer, 2019](https://arxiv.org/abs/1904.06690)
  - [SASRec: Self-Attentive Sequential Recommendation, 2018](https://arxiv.org/abs/1808.09781)
  - [Positive, Negative and Neutral: Modeling Implicit Feedback in Session-based News Recommendation, SIGIR 2022](https://arxiv.org/pdf/2205.06058.pdf)
- Algorithm - graph
  - [PageRank: Standing on the shoulders of giant, 2010](https://arxiv.org/pdf/1002.2858.pdf)
  - [DeepWalk: Online Learning of Social Representations, 2014](https://arxiv.org/pdf/1403.6652.pdf)
  - [SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS, 2017](https://arxiv.org/pdf/1609.02907.pdf)
  - [Inductive Representation Learning on Large Graphs, 2017](https://arxiv.org/pdf/1706.02216.pdf)
  - [Graph Attention Networks, 2018](https://arxiv.org/pdf/1710.10903.pdf)
  - [Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://arxiv.org/pdf/1806.01973.pdf)
- LookAlike
  - [Finding Users Who Act Alike: Transfer Learning for Expanding Advertiser Audiences, KDD 2019](https://www.pinterestcareers.com/media/gvnpojec/transferlearning-kdd2019.pdf)
- Diversity
  - Algorithmic Effects on the Diversity of Consumption on Spotify, WWW 2020
- Bias
  - Lessons Learned Addressing Dataset Bias in Model-Based Candidate Generation at Twitter, 2020 KDD IRS
  - [Popularity-Opportunity Bias in Collaborative Filtering, WSDM 2021](https://dl.acm.org/doi/pdf/10.1145/3437963.3441820)
  - [Managing Popularity Bias in Recommender Systems with Personalized Re-ranking, 2019](https://arxiv.org/pdf/1901.07555)
  - [The Unfairness of Popularity Bias in Recommendation, 2019](https://arxiv.org/pdf/1907.13286)
- User Modeling
  - [Exploring the longitudinal effects of nudging on users’ music genre exploration behavior and listening preferences, 2022](https://dl.acm.org/doi/pdf/10.1145/3523227.3546772)
  - [Personalizing Benefits Allocation Without Spending Money: Utilizing Uplift Modeling in a Budget Constrained Setup, Recsys2022](https://dl.acm.org/doi/10.1145/3523227.3547381)
- Causality
  - [Inferring the Causal Impact of New Track Releases on Music Recommendation Platforms through Counterfactual Predictions, RecSys2020](https://labtomarket.files.wordpress.com/2020/08/recsys2020lbr.pdf?utm_source=LinkedIn&utm_medium=post&utm_campaign=monday_posting&utm_term=2023_07_24)
- Survey
  - Deep Learning based Recommender System: A Survey and New Perspectives, 2019
  - [A Survey on Causal Inference for Recommendation, 2024](https://arxiv.org/abs/2303.11666)
  - [Fairness and Diversity in Recommender Systems: A Survey, 2024](https://arxiv.org/pdf/2307.04644)

### Imbalanced Learning, Anomaly Detection
- Survey
  - [Learning From Imbalanced Data: open challenges and future directions (survey article 2016)](https://link.springer.com/article/10.1007/s13748-016-0094-0)
  - [Deep Learning for Anomaly Detection A Review, 2020](https://arxiv.org/pdf/2007.02500.pdf)
  - [Autoencoders, 2020](https://arxiv.org/pdf/2003.05991.pdf)
- Perfomance Measure
  - [The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets](https://pubmed.ncbi.nlm.nih.gov/25738806/)
  - [The Relationship Between Precision-Recall and ROC Curves](https://www.biostat.wisc.edu/~page/rocpr.pdf)
  - [Predicting Good Probabilities With Supervised Learning](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)
  - [Properties and benefits of calibrated classifiers](http://www.ifp.illinois.edu/~iracohen/publications/CalibrationECML2004.pdf)
  - [The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets](https://www.researchgate.net/publication/273155496_The_Precision-Recall_Plot_Is_More_Informative_than_the_ROC_Plot_When_Evaluating_Binary_Classifiers_on_Imbalanced_Datasets)
-  Cost-sensitive
   - [An optimized cost-sensitive SVM for imbalanced data learning](https://webdocs.cs.ualberta.ca/~zaiane/postscript/pakdd13-1.pdf)
   - [Metacost : a general method for making classifiers cost-sensitive (KDD 99)](https://homes.cs.washington.edu/~pedrod/papers/kdd99.pdf)
   - [The influence of class imbalance on cost-sensitive learning (IEEE 2006)](https://ieeexplore.ieee.org/document/4053137)
   - [Learning and Making Decisions When Costs and Probabilities are Both Unknown (2001)](https://cseweb.ucsd.edu/~elkan/kddbianca.pdf)
- Sampling
  - [SMOTE, 2002](https://arxiv.org/pdf/1106.1813.pdf)
  - [SMOTE for learning from imbalanced data : progress and challenges, 2018](https://www.jair.org/index.php/jair/article/view/11192)
  - [Influence of minority class instance types on SMOTE imbalanced data oversampling](https://www.researchgate.net/publication/320625181_Influence_of_minority_class_instance_types_on_SMOTE_imbalanced_data_oversampling)
  - [Calibrating Probability with Undersampling for Unbalanced Classification .2015](https://www3.nd.edu/~dial/publications/dalpozzolo2015calibrating.pdf)
  - [A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data](https://www.researchgate.net/publication/220520041_A_Study_of_the_Behavior_of_Several_Methods_for_Balancing_machine_Learning_Training_Data)
  - [Dynamic Sampling in Convolutional Neural Networks for Imbalanced Data Classification](https://users.cs.fiu.edu/~chens/PDF/MIPR18_CNN.pdf)
- Ensemble Learning
  - [Self-paced Ensemble for Highly Imbalanced Massive Data Classification, 2020](https://arxiv.org/abs/1909.03500)
- Feature Selection
  - [Ensemble-based wrapper methods for feature selection and class imbalance learning, 2010](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.706.4216&rep=rep1&type=pdf)
  - A comparative study of iterative and non-iterative feature selection techniques for software defect prediction
-  Learning feature representations of normality
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
- Time Series and Streaming Anomaly Detection
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

### Causality
- Heterogeneous treatment effect estimation, uplift
  - [Causal Inference and Uplift Modeling A review of the literature, 2016](https://proceedings.mlr.press/v67/gutierrez17a/gutierrez17a.pdf) [`review`](./paper_review/Causal%20Inference%20and%20Uplift%20Modeling%20A%20review%20of%20the%20literature.md)
  - [Double machine learning for treatment and causal parameters, 2016](https://www.econstor.eu/bitstream/10419/149795/1/869216953.pdf)
  - [Metalearners for estimation heterogeneous treatment effects using machine learning, 2019](https://www.pnas.org/doi/epdf/10.1073/pnas.1804597116)
  - [Estimation and Inference of Heterogeneous Treatment Effects using Random Forests, 2018](http://bayes.acs.unt.edu:8083/BayesContent/class/rich/articles/Estimation_And_Inference_Of_Heterogeneous_Treatment_Effects_Using_Random_Forests.pdf)

# 📑 Other Reference
### Recommendation System
- 카카오
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
- 카카오엔터테인먼트
  - [최애 작품 이용권 선물해주는 ‘Helix 푸시’ 개발기 (2024)](https://if.kakao.com/session/19)
- 당근
  - 딥러닝 개인화 추천 (당근마켓, 2019)
- 번개장터
  - 번개장터 추천시스템 개발 및 도입후기 (2017)
- 하이퍼커넥트
  - [비용 효율적인 Click-Through Rate Prediction 모델로 하쿠나 라이브 추천시스템 구축하기, 2021](https://hyperconnect.github.io/2021/04/26/hakuna-recsys-gb.html)
  - [아자르에서 AI 기반 추천 모델의 타겟 지표를 설정하는 방법 (feat. 아하 모멘트), 2024](https://hyperconnect.github.io/2024/04/26/azar-aha-moment.html)
  - [협업 필터링을 넘어서: 하이퍼커넥트 AI의 추천 모델링, 2024](https://hyperconnect.github.io/2024/10/21/beyond-collaborative-filtering.html)
  - [이벤트 기반의 라이브 스트리밍 추천 시스템 운용하기, 2022](https://hyperconnect.github.io/2022/01/24/event-driven-recsys.html)
  - [아자르에서는 어떤 추천 모델을 사용하고 있을까?, 2024](https://hyperconnect.github.io/2024/11/19/azar-recommendation-model.html)
- 라인
  - [머신러닝을 활용한 오픈챗 클린 스코어 모델 개발기, 2020](https://engineering.linecorp.com/ko/blog/line-openchat-cleanscore)
  - [LINE Timeline의 새로운 도전 (2020.04)](https://engineering.linecorp.com/ko/blog/line-timeline-discover-ml-recommendation)
  - [오픈챗 해시태그 예측을 위한 다중 레이블 분류 모델 개발하기, 2024](https://techblog.lycorp.co.jp/ko/multi-label-classification-model-for-openchat-hashtag-prediction)
  - [오프라인과 온라인 A/B 테스트를 통해 오픈챗 추천 모델 개선하기, 2023](https://techblog.lycorp.co.jp/ko/improve-openchat-recommendation-model-with-offline-and-online-ab-test)
  - [오픈챗 해시태그 예측을 위한 다중 레이블 분류 모델 개발하기, 2024](https://techblog.lycorp.co.jp/ko/multi-label-classification-model-for-openchat-hashtag-prediction)
- 네이버
  - (Deview2021) BERT로 만든 네이버 플레이스 비슷한 취향 유저 추천 시스템
  - (Deview2021) Knowledge Graph에게 맛집과 사용자를 묻는다: GNN으로 맛집 취향 저격 하기
  - (Deview2020) 유저가 좋은 작품(웹툰)을 만났을 때
  - (Deview2020) 추천시스템 3.0: 딥러닝 후기시대에서 바이어스, 그래프, 그리고 인과관계의 중요성
- 넥슨
  - NDC21-데이터분석, 추천알고리즘 offline A/B 테스트 (feat: PAIGE 프로야구 서비스)
- 요기요
  - 데이터야놀자2022, 뭐먹지 빌런을 무찌르는 GNN 기반 개인화 추천 - 윤기태님
- 무신사
  - [무신사가 카테고리숍 추천을 하는 방법, 무신사 2023](https://medium.com/musinsa-tech/%EB%AC%B4%EC%8B%A0%EC%82%AC%EA%B0%80-%EC%B9%B4%ED%85%8C%EA%B3%A0%EB%A6%AC%EC%88%8D-%EC%B6%94%EC%B2%9C%EC%9D%84-%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95-a45b219685ea)
  - [검색어 분석을 통한 상품 정렬 개선, 무신사 2021](https://medium.com/musinsa-tech/%EA%B2%80%EC%83%89%EC%96%B4-%EB%B6%84%EC%84%9D%EC%9D%84-%ED%86%B5%ED%95%9C-%EC%83%81%ED%92%88-%EC%A0%95%EB%A0%AC-%EA%B0%9C%EC%84%A0-b92ded2923c3)
- 라이너
  - [TikTok for Text! 라이너 앱 Session-based Recommender 구축기, 라이너 2023](https://blog.getliner.com/sessrec/)
  - [Bag-of-Tricks for Recommendation: Recency, Clustering 그리고 Item Shuffling, 2022](https://blog.liner.space/bag-of-tricks-for-recommendation)
  - [신뢰성 있는 문서를 골라주기 위한 Liner Ranker, 2024](https://blog.liner.space/line-ranker)
- 오늘의집
  - [유사 이미지 추천 개발 #1 비슷한 공간 - 콘텐츠 추천을 위한 이미지 유사도 모델 개발 과정](https://www.bucketplace.com/post/2023-05-22-%EC%9C%A0%EC%82%AC-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%B6%94%EC%B2%9C-%EA%B0%9C%EB%B0%9C-1-%EB%B9%84%EC%8A%B7%ED%95%9C-%EA%B3%B5%EA%B0%84/)
  - [유사 이미지 추천 개발 #2 비슷한 상품 - 커머스 상품 추천을 위한 유사도 모델 개발 과정](https://www.bucketplace.com/post/2023-07-13-%EC%9C%A0%EC%82%AC-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%B6%94%EC%B2%9C-%EA%B0%9C%EB%B0%9C-2-%EB%B9%84%EC%8A%B7%ED%95%9C-%EC%83%81%ED%92%88/)
  - [개인화 추천 시스템 #1. Multi-Stage Recommender System, 2024](https://www.bucketplace.com/post/2024-03-26-%EA%B0%9C%EC%9D%B8%ED%99%94-%EC%B6%94%EC%B2%9C-%EC%8B%9C%EC%8A%A4%ED%85%9C-1-multi-stage-recommender-system/)
  - [개인화 추천 시스템 #2. Personalized Content Ranking, 2024](https://www.bucketplace.com/post/2024-07-10-%EA%B0%9C%EC%9D%B8%ED%99%94-%EC%B6%94%EC%B2%9C-%EC%8B%9C%EC%8A%A4%ED%85%9C-2-personalized-content-ranking/)
- 우아한형제들(배민)
  - [개인화 추천 시스템 #3. 모델 서빙, 2025](https://www.bucketplace.com/post/2025-03-14-%EA%B0%9C%EC%9D%B8%ED%99%94-%EC%B6%94%EC%B2%9C-%EC%8B%9C%EC%8A%A4%ED%85%9C-3-%EB%AA%A8%EB%8D%B8-%EC%84%9C%EB%B9%99/) 
  - [(2023 우아콘) 추천시스템 성장 일지: 데이터 엔지니어 편](https://www.youtube.com/watch?v=x49PqlAQC3U&list=PLgXGHBqgT2TundZ81MAVHPzeYOTeII69j&index=13)
  - [(2023 우아콘) 여기, 주문하신 '예측' 나왔습니다: 추천/ML에서 '예측'을 서빙한다는 것에 대하여](https://www.youtube.com/watch?v=OXAABJWUgx4&list=PLgXGHBqgT2TundZ81MAVHPzeYOTeII69j&index=15)
  - [실시간 반응형 추천 개발 일지 1부: 프로젝트 소개, 2024](https://techblog.woowahan.com/17383/)
  - [실시간 반응형 추천 개발 일지 2부: 벡터 검색, 그리고 숨겨진 요구사항과 기술 도입 의사 결정을 다루는 방법, 2025](https://techblog.woowahan.com/21027/)
- 컬리
  - [함께 구매하면 좋은 상품이에요! - 장바구니 추천 개발기 1부, 2024](https://helloworld.kurly.com/blog/cart-recommend-model-development/)
  - [함께 구매하면 좋은 상품이에요! - 장바구니 추천 개발기 2부, 2024](https://helloworld.kurly.com/blog/cart-recommend-model-development_second/)
- 토스
  - [SLASH 24 - 기반 데이터가 부족해도 OK! 커머스 추천 시스템 제작기, 2024](https://www.youtube.com/watch?v=LAD6LYnkPsA&list=PL1DJtS1Hv1PiGXmgruP1_gM2TSvQiOsFL&index=31)
- Spotify
  - [The Rise (and Lessons Learned) of ML Models to Personalize Content on Home, 2021](https://engineering.atspotify.com/2021/11/the-rise-and-lessons-learned-of-ml-models-to-personalize-content-on-home-part-i/)
  - [Introducing Natural Language Search for Podcast Episodes, spotify 2022](https://engineering.atspotify.com/2022/03/introducing-natural-language-search-for-podcast-episodes/)
  - [Modeling Users According to Their Slow and Fast-Moving Interests, spotify 2022](https://research.atspotify.com/2022/02/modeling-users-according-to-their-slow-and-fast-moving-interests/)
  - [Socially-Motivated Music Recommendation, 2024](https://research.atspotify.com/2024/06/socially-motivated-music-recommendation/)
  - [Personalizing Audiobooks and Podcasts with graph-based models, 2024](https://research.atspotify.com/2024/05/personalizing-audiobooks-and-podcasts-with-graph-based-models/)
- Pinterest
  - [The machine learning behind delivering relevant ads, Pinterest 2021](https://medium.com/pinterest-engineering/the-machine-learning-behind-delivering-relevant-ads-8987fc5ba1c0)
- Ebay
  - [Building a Deep Learning Based Retrieval System for Personalized Recommendations, ebay 2022](https://tech.ebayinc.com/engineering/building-a-deep-learning-based-retrieval-system-for-personalized-recommendations/)
- nvidia
  - [Recommender Systems, Not Just Recommender Models, nvidia merlin](https://medium.com/nvidia-merlin/recommender-systems-not-just-recommender-models-485c161c755e)
- Uber
  - [Innovative Recommendation Applications Using Two Tower Embeddings at Uber, Uber 2023](https://www.uber.com/en-KR/blog/innovative-recommendation-applications-using-two-tower-embeddings/)
- Google
  - [Scaling deep retrieval with TensorFlow Recommenders and Vertex AI Matching Engine](https://cloud.google.com/blog/products/ai-machine-learning/scaling-deep-retrieval-tensorflow-two-towers-architecture?hl=en)
- X (twitter)
  - [Model-based candidate generation for account recommendations, X 2022](https://blog.twitter.com/engineering/en_us/topics/insights/2022/model-based-candidate-generation-for-account-recommendations)
- meta (facebook)
  - [Scaling the Instagram Explore recommendations system, meta 2023](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/)
  - [How machine learning powers Facebook’s News Feed ranking algorithm, meta 2021](https://engineering.fb.com/2021/01/26/ml-applications/news-feed-ranking/)
- Netflix
  - [Innovating Faster on Personalization Algorithms at Netflix Using Interleaving, 2017](https://netflixtechblog.com/interleaving-in-online-experiments-at-netflix-a04ee392ec55)
- Others
  - Transformers4Rec: A flexible library for Sequential and Session-based recommendation
  - [[22'Recsys] BERT4Rec 구현의 진실에 관하여 : A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation](https://mytype9591.tistory.com/m/6)
  - [Scaling deep retrieval with TensorFlow Recommenders and Vertex AI Matching Engine](https://cloud.google.com/blog/products/ai-machine-learning/scaling-deep-retrieval-tensorflow-two-towers-architecture?hl=en)
  - [DLRM github](https://github.com/facebookresearch/dlrm)
  - [DeepCTR github](https://github.com/shenweichen/DeepCTR)
  - [Two Tower Model Architecture: Current State and Promising Extensions, 2023](https://blog.reachsumit.com/posts/2023/03/two-tower-model/)
  - [추천 시스템 서비스 적용을 위한 Elastic Search 도입기, 2022](https://blog.dramancompany.com/2022/11/%EC%B6%94%EC%B2%9C-%EC%8B%9C%EC%8A%A4%ED%85%9C-%EC%84%9C%EB%B9%84%EC%8A%A4-%EC%A0%81%EC%9A%A9%EC%9D%84-%EC%9C%84%ED%95%9C-elastic-search-%EB%8F%84%EC%9E%85%EA%B8%B0/)
  - [Recommendation Systems • Bias](https://aman.ai/recsys/bias/)

### Imbalnaced Learning, Anomaly Detection
- 우아한형제들(배민)
  - [배민 앱 리뷰 품질을 향상시킨 방법은? 머신 러닝 X 네트워크 탐지 모델 도입](https://techblog.woowahan.com/11829/)
- Netflix
  - [Machine Learning for Fraud Detection in Streaming Services](https://netflixtechblog.medium.com/machine-learning-for-fraud-detection-in-streaming-services-b0b4ef3be3f6)
- Pinterest
  - [Warden: Real Time Anomaly Detection at Pinterest](https://medium.com/pinterest-engineering/warden-real-time-anomaly-detection-at-pinterest-210c122f6afa)
  - [Fighting Spam using Clustering and Automated Rule Creation](https://medium.com/pinterest-engineering/fighting-spam-using-clustering-and-automated-rule-creation-1c01d8c11a05)
- Lyft
  - [Full-Spectrum ML Model Monitoring at Lyft](https://eng.lyft.com/full-spectrum-ml-model-monitoring-at-lyft-a4cdaf828e8f)
  - [Building a large scale unsupervised model anomaly detection system — Part 1](https://eng.lyft.com/building-a-large-scale-unsupervised-model-anomaly-detection-system-part-1-aca4766a823c)
  - [Building a large scale unsupervised model anomaly detection system — Part 2](https://eng.lyft.com/building-a-large-scale-unsupervised-model-anomaly-detection-system-part-2-3690f4c37c5b)
- Others
  - [Fraud Detection on Blockchain based on Machine learning (medium article)](https://jysden.medium.com/fraud-detection-on-ethereum-with-graph-neural-network-f434e5b9aef5)
  - [Unsupervised and semi-supervised anomaly detection with data-centric ML, google blog](https://ai.googleblog.com/2023/02/unsupervised-and-semi-supervised.html)

### Causality
- Netflix
  - [A Survey of Causal Inference Applications at Netflix](https://netflixtechblog.com/a-survey-of-causal-inference-applications-at-netflix-b62d25175e6f)
- Uber
  - [Using Causal Inference to Improve the Uber User Experience, 2019](https://www.uber.com/en-KR/blog/causal-inference-at-uber/)
- 넥슨
  - [업리프트 모델링을 통해 게임 광고 전환율 향상시키기, 2023](https://www.intelligencelabs.tech/372bcb48-af74-4a4f-a2b5-57df9e45fcb9)
- 네이버페이
  - [Uplift Modeling을 통한 마케팅 비용 최적화 (with Multiple Treatments), 2024](https://medium.com/naverfinancial/uplift-modeling%EC%9D%84-%ED%86%B5%ED%95%9C-%EB%A7%88%EC%BC%80%ED%8C%85-%EB%B9%84%EC%9A%A9-%EC%B5%9C%EC%A0%81%ED%99%94-with-multiple-treatments-5e4e3824b9df)
- Others
  - [Brady Neal - Causal Inference](https://www.youtube.com/c/BradyNealCausalInference/playlists)
  - [인과추론의 데이터과학](https://www.youtube.com/c/%EC%9D%B8%EA%B3%BC%EC%B6%94%EB%A1%A0%EC%9D%98%EB%8D%B0%EC%9D%B4%ED%84%B0%EA%B3%BC%ED%95%99/playlists)
  - [EconML/CausalML KDD 2021 Tutorial](https://causal-machine-learning.github.io/kdd2021-tutorial/)
  - [Causal Inference for the Brave and True](https://matheusfacure.github.io/python-causality-handbook/01-Introduction-To-Causality.html)
  - [Dowhy 가이드 실습 pap gitbook](https://playinpap.gitbook.io/dowhy/)
  - [Causal-Inference-and-Discovery-in-Python](https://github.com/PacktPublishing/Causal-Inference-and-Discovery-in-Python)
  - [For effective treatment of churn, don’t predict churn, 2019](https://medium.com/bigdatarepublic/for-effective-treatment-of-churn-dont-predict-churn-58328967ec4f)
  - [Causal inference 123, Microsoft Shujuan(Jane) Huang 2020](https://medium.com/@shujuanhuang)

### LLM
- 우아한형제들(배민)
  - [GPT를 활용한 카탈로그 아이템 생성, 2025](https://techblog.woowahan.com/21294/)

### etc
- 우아한형제들(배민)
  - [로봇 ML 모델의 경량화 1부: 훈련 후 양자화, 2024](https://techblog.woowahan.com/18980/)
  - [로봇 ML 모델의 경량화 2부: 양자화 인식 훈련, 2024](https://techblog.woowahan.com/21176/)
  - [프로덕트 전략, 어떻게 시작해야 할까?, 2025](https://techblog.woowahan.com/21115/)