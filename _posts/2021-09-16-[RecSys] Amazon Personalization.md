---
title: "[RecSys] Amazon Personalization 살펴보기"
categories:
  - RecSys
tags:
  - aws
  - recommendation
---
### Amazon Personalization for the world


> Amazon에서 제공하는 개인화 추천 솔루션인 Amazon Personalize에 대해 알아봅니다.  


<center><img src="/assets/materials/recsys/amazon_personalize/personalize_logo.png" align="center" alt="drawing" width="400"/></center>   


<br/>


>  **1. AWS에서 2019년에 런치한 개인화 추천 서비스**
>
>  **2. 머신러닝 엔지니어 없이도 자동화된 실시간 추천 솔루션을 시스템에 적용할 수 있음**
>
>  **3. user meta, item meta, interaction 세 가지의 데이터셋을 사용가능, interaction은 필수**


<br/>

----

**본 포스트는 Anoop Deoras(Amazon)의 발표 영상을 토대로 작성한 노트입니다.**
[영상은 여기에서 보실 수 있습니다.](https://www.youtube.com/watch?v=2s7vUQDQPNY&list=PL-4RaB3L_GT9jx1zLo9liJNCQbAmLJ5zs&index=14)


1. Intro  
	- 2019년에 런치된 개인화 추천 서비스  
	- misson: 모든 개발자가 사용할 수 있게 머신러닝을 제공하는 것  
	- AWS ML stack의 3가지 레이어  
	1) bottom: Frameworks, AMIs, Docker Images  
	2) middle: Sagemaker, ML/DS practitioners, Notebooks, HPO, Model zoos..  
	3) top: AI services: fully managed, Pre-trained and auto-trained models WITHOUT ML/DS practitioners  


2. Amazon Personalization
	- 개인화된 추천을 제공하기 위한 fully managed service
	- 매우 유관한 추천을 거의 실시간에 제공
	- custom & private한 추천모델을 사용자의 고객에 맞추어 제공, ML/DS 전문가 필요 없음
	- user meta, item meta, interaction 데이터셋을 사용할 수 있음


3. 로데이터에서 고객 API Endpoint까지
	- 원클릭 솔루션을 제공하기 위해 노력함 (로데이터에서 개인용 커스텀 딥러닝 모델 API까지)
	- 라지 스케일이며, 거의 실시간이고, contextual하며 시간 순에 민감한 딥러닝 모델을 빌드함
	- 내부적으로(under-the-hood), Amazon Personalize는 아래와 같은 작업을 처리  
	1) 데이터 조사(inspection): 결측치, 이상치 검사  
	2) featurization: 텍스트, 카테고리와 같은 비정형 데이터 처리, 수치 변수 표준화 등  
	3) HPO: 베이지안 최적화를 사용해 딥러닝 모델의 HPO  
	4) model training & optimization& hosting  
	5) real time feature store for near realtime reco  


4. Personalize 사용하기  
	1) Data ingestion  
	2) Training  
	3) Inference  

5. 6개의 use cases  
	1) user specific reco: 이번 발표의 포커스  
	2) coldstart users    
	3) coldstart items   
	4) similar items  
	5) personalized ranking  
	6) popularity  


6. User Personalization - key features  
	1) user impression: impression data를 모델링, 노출되었으나 클릭되지 않은 아이템  
	2) item exploration: 새로운 아이템과 유저가 좋아할만한 아이템 사이를 밸런싱  
	3) filtering of event: 이벤트 기준에 따라서 추천할 아이템을 포함하거나 배제  
	4) filtering based on metadata: 아이템이나 유저의 메타데이터 기준에 따라 추천할 아이템을 포함하거나 배제  
	5) cold start: 새로운 유저나 아이템에 대한 추천을 포함  


7. 딥러닝 추천 모델에 대한 일반적인 아키텍처  
	- 크게 두 개의 레이어로 구성됨: exploration, exploitation  


8. Exploitation layer
	- exploitation layer에서는 item id embedding과 item meta data 기반 representation을 사용해 아이템을 표현
	- 한 유저를 $n$개의 아이템과 이벤트로써 표현한다고 해보자. 유저는 $x_1,x_2,...,x_n$으로 표현될 수 있을 것  
	- $x_i$는 i번째 아이템의 표현인데, 두 개로 나누어져 얻어짐. 하나는 $e_i$로 표시되는 item id embedding, 다른 하나는 $f_i$로 표시되는 item meta based representation임. 이들은 각각 $E$와 $F$라는 임베딩 매트릭스에서  $x_i$의 내적을 통해 lookup됨  
	- 이 둘 $(e_i, f_i)$이 결합됨으로써 $z_i$가 구해짐. $z_i = W(e_i  \mid f_i))$  이는 일종의 latent representation임  
	- 이 latent representation을 RNN, transformer, FNN 등의 알고리즘에 태워서 h_u라는 유저 representation을 구해냄
	- 이 $h_i$를 아까 사용했던 $E$와 $F$의 전치행렬에 내적해줌으로써 vocab의 score를 구할 수 있음.
	- 이들은 각각 O, O로 표현됨

9. Exploration layer
	- $x_1,x_2,...,x_{vocab}$이 있을 때, 최신성(recency, 지금까지 아이템이 받아온 impression)과 경향성(propensity, 아이템의 유저에 대한 경향)을 살펴보고 모든 아이템에 대한 score(일종의 exploration score)를 얻게 될 것임
	- 이 결과로 얻은 $O$와 앞에서 얻은 두 $O$를 목적함수에 넣은 뒤 최적화
	- 이렇게 함으로써 유관하고(relevant), 최신성에 편향되고(recency biased), 새로운 컨텐츠를 탐색하는 추천 결과를 제공함


10. User personalization DL model challenges
	- scale: 어떻게 이 딥러닝 시퀀셜 모델을 수백만의 아이템이 포함된 카탈로그에 대해 scale up할 수 있는가?
	- near real time reco: 어떻게 유저의 최신 행동에 대해 뒤쳐지지 않은 추천을 제공한다고 확신할 수 있나?
	- contextual bandit: 어떻게 탐색과 활용의 트레이드오프 지점을 확신할 수 있는가?
	- unstructured text info: 카탈로그 온톨로지(장르, 서브장르)를 구축하는 것은 매우 비쌈. 로우 텍스트를 그대로 처리할 수는 없을까?
	- business objectives: 비즈니스 목표를 직접적으로 최적화할 수는 없을까?(스트리밍 시간, 매출..)

11. Scaling Deep Learning model: partition function, importance sampling, negatives
	- 시퀀셜 추천 모델은 이벤트에 대한 언어 모델로 볼 수 있음
	- 유저 맥락/표현 $h_t$가 있을 때, 다음 이벤트/아이템 y_t에 대한 조건부 확률은 다음과 같이 표현됨: $P(y_t \mid h_t)=\frac{e^{-\epsilon(y_t, h_t)}}{Z(h_t)}$  
	- $Z(h_t)$는 추적 가능하지만, 학습 중에 연산 비용이 비쌈  
	- energy based model: $P(X=z) = \frac{e^{-\epsilon(z)}}{Z}$, 여기서 $z$는 energy, $Z$는 partition function
	- 경사하강법 기반의 최적화 알고리즘의 핵심은 기울기를 취하는 것임. 여기서는 log likelihood를 취해준 뒤 기울기를 구함.
	- log likelihood 후 미분하면 2개의 텀으로 나뉨. 이 중 오른쪽은 energy의 기울기의 기댓값이 됨. 여기서 'average'는 Gibbs sampling으로 추정될 수 있음.
	- 그러나 $P(.)$에서 샘플링하려면 $P(.)$을 추정해야하는데 여기에는 우리가 모르는 Z가 포함되어 있음
	- 이 텀에 surrogate function Q를 도입함으로써 문제를 처리함. $Q$는 샘플링이 쉬운 함수로 선택.
	- 내부 텀에 대하여 평균을 취해주면, 우리가 알고싶은 expectation에 대한 unbiased and constant 한 추정치를 얻을 수 있음.
	- 따라서 정리된 식에, $Q$에서 뽑은 $m$개의 샘플을 대입하여 쉽게 정리 가능
	- 그러나 여전히 $P(.)$가 내부 텀에 포함되어 있음. 샘플링할 필요는 없으나, 평가를 위해서는 필요함. 그러나 이는 연산이 매우 비쌈. (매 가중치 스텝마다 해주어야 하므로)
	- 이 문제를 해결하기 위해 biased estimator를 사용함.

13. 조슈아 밴지오의 아이디어
	- 조슈아의 논문에서 아이디어를 얻었음.
	- $P(.)$을 상수곱(multiplicative constant)으로 명시적(explicitly) 표현할 수 있는가? YES.
	- 여기서의 아이디어는 $(1/W)w(y_i)$를 가중치로 사용하는 것임.
	- 이제 핵심적인 질문은, surrogate $Q$로 무엇을 써야하느냐, 그리고 $m$은 몇개나 되어야 하느냐임.


14. 실험
	- vocab size가 중요: 소-중규모의 보캡 수에 달하면 연산이 매우 복잡해짐. 따라서 특정 역치를 기점으로 importance sampling을 사용하는 것이 필요함, 그 아래 사이즈의 데이터셋은 일반적인 추론을 수행
	- importance sampling에 의해 쓰루풋이 수천배 차이나고 결과적으로 성능에도 영향을 미침.
	- Q 선택: working 했던 분포는 standard unigram popularity distribution임.
	- 샘플수 m: vocab의 sqrt가 잘 먹히는듯 (like a charm)
	- 샘플링 방법: 샘플링은 학습에 미니배치마다 stochasticity를 주기때문에 효율적임. alias sampling은 처음에 시간이 조금 소요되나 이후에는 O(1)의 속도를 보여주므로 매우 효율적인 샘플링기법임

15. Cold start
	- coldstart를 해결하기 위해, 크게 두가지 접근을 취함.  
	1) unstructured textual inputs: item meta data를 사용함으로써 문제해결  
	2) Exploitation/Exploration tradeoff  

16. Item content embeddings
	- 아이템 설명과 같은 텍스트 데이터는 SOTA NLP 모델에 입력되어 dense representation을 만들어냄
	- 이 NLP 모델은 추천 모델과 결합적으로 학습/파인튜닝될 수 있음 (모델 내부에 포함될수도, 바깥에 있을 수도 있음) -> usecase에 따라서 nlp 모델과 같이 파인튜닝 될지 아닐지를 결정
	- 상품 설명을 사용한 모델과 그렇지 않은 모델을 비교
	- 아이템당 인터랙션의 수가 적은 데이터셋일수록, 상품 설명 텍스트 메타데이터는 성능에 도움을 줌 (amazon prime pantry dataset에 실험)
	- 그러나 메타데이터 생성은 매우 비용이 많이 드는 과정이고, 모든 사용자(Amazon Personalize 사용자)가 이것을 하려고 하지는 않을 것
	- 비정형 텍스트 메타데이터를 사용해서, 메타데이터 필요없이 모델을 만들 수 있다면 좋지 않을까?
	- 실험 결과, 수작업한 메타데이터에 비해 텍스트를 사용한 결과가 모든 지표에서 좋았음

17. 비정형 텍스트 데이터에 대한 best practice
	- 편집상에서 검증된, 간결/유관/정보적인 설명이 각각의 아이템에 대해 제공될 경우, 그리고 가장 적절한 설명이 텍스트 서두에 나올 경우가 권장됨
	- 결측이 많은 텍스트 칼럼은 텍스트를 포함시킨 작업의 긍정적인 영향을 감소시킴(pantry 데이터셋에 적용시킨 결과, 비율이 늘수록 성능은 떨어지지만 그럼에도 90퍼 결측인 상황에도 텍스트 없는 상황보다 3배 높은 성능)
	- 마크업이나 공백을 사전에 처리해 주는 작업이 유용
	- 다국어를 위해서는 영어모델보단 다국어 nlp모델 사용



18. Exploit/Explore
	- 환경은 변화함. 아이템이 새로 들어오고 나가고, 유저의 취향은 변화하기 마련
	- 변치않는 목표: 유저의 취향에 맞는 아이템 제공
	- exploration을 아이템 신선도, 노출, 유저 친밀도(user affinity)의 함수로 만들자

19. 시뮬레이션
	- 매 라운드마다 1명의 유저와 10개의 아이템을 생성
	- 각 스텝마다 각 유저에게 1개의 추천을 제공, 300 스텝 진행
	- 100번째 스텝에서, 유저의 관심사가 역전됨
	- 각 스텝에서의 유저 반응이 모델 재학습을 위해 수집됨, 모델이 10 스텝마다 업데이트됨
	- exploration이 없는 모델의 경우 유저 관심이 변한 후의 reward를 회복하지 못함.
	- exploitation only: greedy하며, exploration에 reluctant함

20. Streaming Events
	- 자동 모델 업데이트: 전체 학습(full training)은 아이템 카탈로그에 대한 모델의 시각을 고정시킴(freeze)
	- 환경은 빠르게 변하며, 새로운 아이템과 그들의 속성 역시 매우 빠르게 변화함
	- Amazon Personalize는 새로운 아이템과 그들의 최신 메타데이터를 포함하도록 모델을 업데이트하고, exploration 과정을 유저로부터의 암묵적/명시적인 피드백을 사용해 조정함
	- 이를 위해서는 앞서 살펴본 E와 F 매트릭스를 들여다보아야 함. 새로운 아이템이 들어오면, 이 매트릭스들의 내부 파라미터를 조정해서 출력 스코어에 반영되도록 해야함
	- E는 아이템 id 임베딩이고, F는 아이템 메타데이터 임베딩임.
	- $O=E^T * h_u + b$ 그리고 $O=F^T * h_u + b$ 로 나타낼 수 있음 (output)
	- $E$ 혹은 $F$는 embedding size * vocab size이고, $O$는 vocab size * 1임
	- 새로운 아이템이 추가되면, E와 F에 새로운 추가적인 차원 하나가 필요하고, $O$/$O$는 그들에 대한 새로운 스코어를 가지게 될 것 (우리가 자동으로 모델을 업데이트하기 위해 필요한 새로운 아이템에 대한 스코어)  
	- 모든 새로운 아이템에 대해, 이미 존재하는 아이템 스페이스의 이웃을 찾기 위해 메타데이터를 사용함으로써 스마트한 초기화 수행 -> 전체 재학습 없이도 사용 가능한 성능 얻어냄


22. Automatic updates vs model training: Best practice
	- 주기적으로 모델을 학습하되 자동 업데이트를 사용해서 최신 아이템 메타데이터의 트렌드를 반영
	- full training 사이에 발생한 유저의 implicit feedback을 사용, exploration을 조정
	- 그러나 full training은 비쌈
	- incremental training에 대한 실험이 진행중


23. Near Real time recommendation
	- Amazon Personalize에 streaming event를 보내기 위해 event tracker를 사용할 수 있음
	- event tracker는 event tracker가 설치된 서비스에서 발생한 가장 최신의 유저 액션을 제공해줌
	- 이벤트 아이디, 발생 시간, 이벤트 타입, 특징(아이템 아이디 등) 을 전송해줌

24. 정리
	- ML code는 전체 시스템의 매우 일부분에 불과함
	- ML 시스템 전반은 Amazon에게 맡기고, 고객에게 더 중요한 일에 집중할 수 있도록 해줌
	- 아마존에 고객 이슈를 대응하는 24/7 고객 팀이 있음.


----------------


**개선을 위한 여러분의 피드백과 제안을 코멘트로 공유해 주세요.**
**내용에 대한 지적, 혹은 질문을 환영합니다.**  
