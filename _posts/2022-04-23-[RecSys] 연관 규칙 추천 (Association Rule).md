---
title: "(작성중) [RecSys] 연관 규칙 추천 (Association Rule Mining)"
categories:
  - Recommender Systems
tags:
  - Business Cases
  - Recommender Systems
---

### 추천 시스템의 가장 원시적이지만 강력한 Baseline으로 사용되는 Association Rule에 대해 배워봅니다.  


> 기저귀를 샀던 김씨는 맥주도 사던데, 당신도 기저귀를 샀으니 맥주를 사는거 어때요?  


<center><img src="/assets/materials/recsys/inmobi_nfm/inmobi.jpg" align="center" alt="drawing" width="500"/></center>   


>  **1. 온라인 유저 행동 예측을 위해, 선형 모델으로부터 시작해 다양한 구조의 Factorization Machine 기반 모델을 적용**
>
>  **2. FM(Factorization Machine), FFM(Field-aware Factorization Machine), DeepFM(Deep neural net with FM), NFM(Neural Factorization Machine), DeepFFM, NFFM(Neural Feature-aware Factorization Machine)을 적용, 각각의 실험 결과와 장단점을 설명. 결론적으로는 NFFM이 우수한 성능을 보임**
>
>  **3. 이러한 모델을 학습하고 서빙하기 위한 디테일한 설정을 소개**

<br/>

----

#### Contents

<br/>

1.  [연관 규칙 추천이란?](#arm)
2.  [연관 규칙 추천의 로직](#logic)
3.  [코드 예제](#example)

<br />



<a id="arm"></a>

## 1. 연관 규칙 추천(Assocication Rule based Recommendation)이란?  
연관 규칙 추천이란, 어쩌면 이 글을 읽는 많은 분들이 데이터 마이닝, 혹은 빅데이터의 개론 강의에서 한 번쯤은 접해봤을 "기저귀와 맥주" 예시와 관련이 있습니다. 얼핏 어울리지 않는 "기저귀"와 "맥주"가 같이 자주 팔린다는 패턴을 활용해 추천을 한 일화, 혹시 들어 보셨나요? 장을 보러 온 부모님이 자신들이 마실 맥주를 구매하는 것이 매우 지당한 이야기 같지만, 데이터 없이 이러한 연관을 떠올린다는 것은 매우 힘든 일일 것입니다.  


이처럼 연관 규칙 추천의 접근법은, 얼른 말해 "A를 사면 B도 산다"는 규칙을 찾는 것인데요. 이를 찾아내는 방법은 전체 거래 내역을 살펴 보아, "A를 사는 사람은 B도 산다"라는 패턴에서 규칙을 찾아내는 것입니다. 정리하자면, **"A를 사는 사람은 B도 사던데, 당신은 A를 샀으니 B라는 상품은 어떠세요?"** 하며 추천하는 접근이죠.   


이처럼 다양한 item이 등장하는 전체 거래에서 **특정 item이 연결되는 방법, 그리고 그 이유를 결정하는 규칙을 발견**하기 위한 학습 방법론입니다. 이는 Market Basket Analysis, Affinity Analysis로도 알려져 있습니다.  


<a id="logic"></a>

## 2. 연관 규칙 추천의 로직



<a id="example"></a>

## 3.




3. 비지도학습
	- 내재적인 특성을 탐색
	- 잠재적인 분포를 추정
	- 밀도 추정, 군집화, 특성 탐지(novelty detection)

4. 지도학습
	- x와 y사이의 관계를 찾음
	- y=f(x)라는 잠재적인 함수를 추정
	- 회귀, 분류

5. Association Rule Mining(ARM)
	-

6. ARM 구현
	- 물건의 구매 여부만을 고려하고, 수량은 고려하지 않음. 즉 빵 10개와 우유 1개, 그리고 빵 1개와 우유 1개가 동일하게 취급됨
	- item list(각 행이 transaction을 나타내고 그 안에 상품 list가 포함된 형태) 혹은 item matrix(거래 수*아이템 수 사이즈의 매트릭스에 0,1로 표기)

7. ARM의 용어
	- Antecedent: "IF" 부분을 나타냄
	- Consequent: "THEN" 부분을 나타냄
	- Item set: antecedent와 consequent로 구성된 아이템 집합
	- antecedent와 consequent는 disjoint함 (즉, 공통된 아이템이 하나도 없음)
	- (라면, 콜라) -> (밥, 참치) : O
	- (라면, 콜라) -> (라면, 참치) : X

8. ARM 규칙 생성
	- 많은 규칙이 가능함(예를 들어 거래 1에 대해)
	1) 만약 계란이 구매되면, 라면도 같이 구매된다
	2) 만약 계란과 라면이 구매되면, 참치캔도 같이 구매된다
	3) 만약 참치캔이 구매되면, 계란이 같이 구매된다 등
	- 매우 많은 수라서 6개의 아이템으로 구성된 인벤토리라 할지라도 수백개의 규칙이 가능해져버림
	- 불필요한(성능이 낮은) 규칙을 배제하기 위해, 다양한 성능 지표를 적용하여 연산 효율화

9. Support(지지도)
	- Support(A->B)= P(A) or P(A,B)
	- 기본적으로는 A라는 조건이 등장할 확률을 의미하지만(전자, P(A)), 대부분의 패키지에서는 후자로 구현함. (P(A,B))
	- 빈번히 등장하는 아이템 집합을 찾기 위해 사용
	- 지지도가 높을수록, 이 규칙을 적용할 가능성이 높아짐 (우리의 거래 내역에서 자주 등장하는 조합이니 자주 노출시킴)

10. Confidence(신뢰도)
	- Confidence(A->B)= P(A,B)/P(A) = P(B|A)
	- A가 주어졌을 때 B의 조건부 확률을 나타냄
	- 유의미한 규칙을 만들어내기 위해 사용

11. Lift(향상도)
	- lift(A->B)=P(A,B)/P(A)*P(B)
	- 생성된 규칙의 유용도를 나타내기 위해 사용
	- lift가 1이면 A와 B는 통계적으로 독립
	- lift > 1은 A와 B 사이에 긍정적인 관계
	- lift < 1은 A와 B 사이에 부정적인 관계
	- lift가 1.25라면, 라면과 밥이 독립이라고 가정했을 때에 비해서 0.25개가 더 팔렸다. 고로 그만큼 효과적인 규칙이다
	- confidence로 부족한 이유: 장바구니에 기본 아이템이 포함되어 있을 경우, 가령 노가리 호프집에서 맥주의 경우 맥주가 포함된 거래에서 모두 confidence는 높지만 lift는 낮을 것임

12. 규칙 생성 : Brute Force
	- 이상적으로는 모든 조합을 고려해서 계산하는 것이 좋음
	- 그러나 아이템 수가 늘어남에 따라 연산량에 기하급수적으로 증가
	- 모든 규칙을 리스트업하고 각각 confidence, support를 계산
	- 그 후 최소한의 threshold를 넘지 못하는 규칙을 제거
	- 그러나 이는 연산적으로 불가능한 알고리즘임

13. 규칙 생성 : A priori
	- 오직 빈번하게 등장하는 아이템 셋에 대해서만 고려함
	- support로 기준점을 잡음
	1) 아이템 셋 빈도 P(A)의 기준
	2) antecedent와 consequent 모두를 포함하고 있는 거래의 수(%)
	- anti-monotone property: 한 아이템셋의 support는 그의 부분집합의 support를 넘지 못한다는 support의 특성. 달리 말해 minimum support를 넘지 못하는 아이템 조합의 superset은 모두 minimum support를 넘지 못함
	- 고로 아이템 2개짜리 조합을 고려했을 때 minimum support를 넘지 못했다면, 다음 아이템 3개짜리 조합을 고려할 때 볼 필요도 없어짐
	- 이와 같은 규칙을 적용함에 따라 연산량을 크게 줄일 수 있음

14. 실제 사용 시 고려할 점
	- support와 lift 둘 중에 하나를 골라야 될 경우 맥락에 따라 판단이 달라져야함
	- support가 높으면 규칙을 적용할 가능성은 높지만, lift를 사용하면 효과는 확실함
	- 즉 support는 노출의 가능성, lift는 전환의 성공 확률을 각각 중점으로 뒀을 때 더 가중치를 둬야할 성능지표임







### 1.들어가며

InMobi는 글로벌한 온라인 타겟 광고를 제공하는 기업입니다. 이러한 기업의 매출 성과를 위해서는 온라인 상의 유저 행동을 예측하는 것이 매우 중요한데요. 좋은 성능의 추천 모델은 곧 수십억의 매출 향상과 직결되기 때문에, 비즈니스 성공을 위해 우수한 모델을 개발하는 일이 필요한 상황이었다고 합니다.  

Inmobi의 엔지니어들이 최종적으로 NFFM (Neural-network Field-aware Factorization Machine) 모델을 개발하기 까지 다양한 Factorization Mahcine 기반 모델을 적용한 이야기를 공유해 주었고, 이 포스트에서는 그 영상에 담긴 이야기를 간략하게 정리해 보도록 하겠습니다.  

<br />

<a id="background"></a>

### 2. 배경 설명

<br />

#### 2-1. Existing context and challenges  

- 일반적으로 linear/logistic 모델과 tree-based 모델을 주로 사용함.  
- 실제 적용될 경우 두 모델은 각각의 장단점을 가짐.  
- Linear Regression: unseen combination에 대해 잘 일반화함, 때때로 underfit될 가능성 존재, 더 적은 RAM 필요  
- Tree model: unseen combination에 대해 잘 일반화 못함, 때때로 overfit되며, 종종 RAM을 터지게할 수 있음. 특히 매우 많은 수의 feature를 사용할 경우 메모리 이슈 발생.  
- 우리는 이 두 모델의 가운데에 있는 어떤 모델을 찾아 성능을 극대화하고 싶음.  

<br>

#### 2-2. Why think of NN for CVR/VCR prediction  

- LR(Linear Regression)에 cross feature를 사용하는 것은 현 문제에 적합하지 않았음  
- 또한 때때로 학습과 예측 단계에서 다루기 까다로워짐(cumbersome)  
- 여기서 언급된 모든 주된 예측 작업은 복잡한 곡선을 따름  
- LR 모델은 interaction term이 제한되어 있어 트리 기반에 비해 개선의 여지가 컸음  
- 몇몇 효과적인 모델을 적용해 보았으나 트리 기반의 모델을 이길 수 없었음.  
- 우리의 팀은 피처들 간의 고계(high-order) interaction을 찾아내기 위해서는 뉴럴넷이 필요하다고 판단했음  
- 뉴럴넷은 unseen combination에 일반화하는 성능도 가지고 있음.  

<br>

#### 2-3. Challenges involved  

- 전통적으로 뉴럴넷은 분류 문제에 더욱 활용되고 있음  
- 우리의 예측을 regression으로 모델링하고 싶었음    
- 대부분의 피처가 카테고리형이었고 이 말은 one-hot encoding을 사용해야함을 의미함.  
- 효과적인 학습을 위해 아주 많은 데이터를 요구하기 때문에 NN모델은 좋지 않은 성능을 내기 마련이었음.  
- 몇몇 피처는 매우 많은 수를 포함하고 있었고 이는 학습을 더 어렵게 함.  
- 모델은 학습과 서빙을 위한 운용이 쉬워야함  
- spark는 custom 뉴럴넷에 적합하지 않았음  
- 모델은 쉽게 디버깅되고, 비즈니스 변화를 설명할 수 있어야함  
- 뉴럴넷을 오랫동안 사용하지 않았던 이유는 이해가 부족해서였음  

<br />

<a id="fm-models"></a>

### 3. Factorization Machine 계열 모델의 실험 결과

<br />

#### 3-1. **FM(Factorization Machine)**  

- 각각의 카테고리형 변수에 대해 k차원의 latent vector가 있음  
- 이 k는 하이퍼 파라미터임  
- 예를 들어 3개의 카테고리 변수 PV(publisher latent vector), AV(Advertiser latent vector), GV(Gender latent vector)가 있고, 각 각 카테고리 변수에는 3(퍼블리셔가 SONY인지, CNBC인지, ESPN인지)/4(아디다스인지, 나이키인지, 코카콜라인지, P&G인지)/2(남성인지 여성인지)의 cardinality를 갖는다고 할 때 최종 예측 확률(pCVR)은 $PV^T \cdot AV+AV^T \cdot GV+GV^T \cdot PV$로 계산함. (여기서 $\cdot$는 내적)
 - 즉, 각 카테고리에 해당하는 피처간의 유사도를 계산하여 총합을 내림.   


 <br>
 <center><img src="/assets/materials/recsys/inmobi_nfm/fm.png" align="center" alt="drawing" width="600"/></center>   


 <font size="2"><center> 출처: inmobi youtube 영상 (https://www.youtube.com/watch?v=MMTuoFFRCCs) </center>  </font>   
 <br>


#### 3-2. FM(Factorization Machine) 계속  

- 각각의 feature value에 대해 k차원의 representation을 사용함  
- 모든 피처들에 대한 second-order interaction을 잡아냄($A^TB=\mid A\mid \cdot \mid B\mid \cdot cos(\theta)$)  
- 기본적으로 쌍곡선(hyperbola)의 결합의 총합이 최종 예측에 사용됨  
- LR 모델보다는 효과가 좋지만 여전히 트리 기반의 모델보다는 강력하지 못함  
- 영화의 매출을 예측하는 모델을 예시로 들어 설명해 보겠음.  
   - feature는 영화, 성별, 도시를 들 수 있고, latent feature는 액션, 코미디, 호러 등을 들 수 있음.  
   - second-order interaction의 직관: 모든 latent feature에 대해, 모든 original feature쌍에 대해 이 pair들을 감안했을 때 해당 latent feature가 얼마나 매출에 영향을 미치는가를 반영  
   - 최종 예측값은 모든 latent feature에 걸친 선형 총합(linear sum)임  
   - 두 벡터간의 내적을 취해주는 것은 latent feature space내의 유사도를 구해주는 것  


<br>



#### 3-3. **FFM(Field-aware Factorization Machine)**  

- FM의 진화된 형태로, 한 feature에서 latent vector를 만들 때 다른 카테고리(Field)에 대한 latent vector를 각각 만들어서 학습시킴.  
- 최종 예측 확률(pCVR)은 $PV_A^T \cdot AV_P + AV_G^T \cdot GV_A+GV_P^T \cdot PV_G$로 계산  
- 손실함수는 RMSE를 사용할 수 있음. 이를 역전파를 사용해 학습하여 각각 그리고 모든 interaction의 값을 학습할 수 있음  
- FM과 마찬가지로 k차원의 latent vector를 사용하지만 각각의 cross feature에 대해 개별적인 latent vector를 갖는 것이 차이점임  
- FM처럼 second order interaction이지만 자유도(degree of freedom)가 더 높음.  
- 직관: latent feature들이 다른 cross feature들과 다르게 상호작용함.  
- FM보다 훨씬 낫지만, 트리기반 모델을 이기지는 못함  


<br>
<center><img src="/assets/materials/recsys/inmobi_nfm/ffm.png" align="center" alt="drawing" width="600"/></center>   


<font size="2"><center> 출처: inmobi youtube 영상 (https://www.youtube.com/watch?v=MMTuoFFRCCs) </center>  </font>   
<br>


#### 3-4. **Deep neural net with FM: DeepFM**  

- FM과 딥러닝 아키텍쳐를 함께 사용하는 모델로, sparse feature에서 dense embedding을 뽑아 각각 hidden layer와 FM에 투입함. 그 결과값을 addition으로 결합하여 sigmoid를 태우고 확률값의 형태로 출력  
- sigmoid(FM+ NeuralNet(PV:+AV:+GV))=pCVR  
- 이 모델은 NN모델과 FM모델의 결합으로, 최종 출력값은 두 모델의 출력값의 합임.  
- 여기서는 전체 그래프를 한번에 최적화함.  
- 이 모델은 FM에서 얻은 latent vector를 두번째 최적화로써 뉴럴넷에 태우는 것보다 더 나은 성능을 보임(FNN)  
- FM보다는 낫지만 FFM보다는 나쁜 성능  
- 직관: FM은 second order interaction을 찾고, 뉴럴넷은 latent vector를 사용해 higher order nonlinear interaction을 찾음.  


<br>
<center><img src="/assets/materials/recsys/inmobi_nfm/deepfm.png" align="center" alt="drawing" width="600"/></center>   


<font size="2"><center> 출처: inmobi youtube 영상 (https://www.youtube.com/watch?v=MMTuoFFRCCs) </center>  </font>   
<br>


#### 3-5. **Neural Factorization Machine: NFM**  

- sparse vector를 입력받아 입력된 feature에 상응하는 latent vector를 bi-interaction pooling layer를 거침으로써 second order interaction의 정보를 구하고, 이를 뉴럴넷에 통과시킴  
- raw latent vector 대신에 second order feature를 뉴럴넷에 통과시킴.  
- NeuralNet(PV. $\cdot$ AV.+AV. $\cdot$ GV.+GV. $\cdot$ PV)=pCVR  
- 직관: 뉴럴넷은 second order interaction을 입력받아 higher order nonlinear interaction을 찾음.  
- DeepFM보다 더 나은 성능을 보임. 이는 아래의 두 이유때문임.  
  1) 네트워크의 크기가 더 작음으로써 수렴이 더 빨리 이루어짐.  
  2) 뉴럴넷이 second order interaction을 입력받아 higher order interaction으로 쉽게 변형할 수 있음.  
- 그러나 여전히 FFM보다는 낫지 않은 성능  



<br>
<center><img src="/assets/materials/recsys/inmobi_nfm/nfm.png" align="center" alt="drawing" width="600"/></center>   


<font size="2"><center> 출처: inmobi youtube 영상 (https://www.youtube.com/watch?v=MMTuoFFRCCs) </center>  </font>   
<br>


#### 3-6. **DeepFFM**  

- DeepFM의 간단한 업그레이드 버전임.  
- DeepFM, FFM보다 더 성능이 좋음.  
- 학습이 느림  
- FFM part가 예측의 heavy lifting에 큰 부분을 차지. 이는 더 빠른 gradient convergence 때문으로 보임.  
- 직관: latent vector를 취해 뉴럴넷에 넣어 high order interaction을 얻고 FFM으로 second order interaction을 학습함  



<br>
<center><img src="/assets/materials/recsys/inmobi_nfm/deepffm.png" align="center" alt="drawing" width="600"/></center>   


<font size="2"><center> 출처: inmobi youtube 영상 (https://www.youtube.com/watch?v=MMTuoFFRCCs) </center>  </font>   
<br>



<br />

<a id="nffm"></a>

### 4. NFFM(Neural Feature-aware Factorization Machine) 구현 디테일

<br />

#### 4-1. **NFFM(Neural Feature-aware Factorization Machine)**  

- NFM의 FFM 버전, simple upgrade of NFM으로 볼 수 있음  
- **최종적으로 적용한 모델**
- 다른 어느 모델보다 유의미하게 좋은 성능  
- DeepFFM보다 더욱 빠른 수렴  
- 직관: FFM으로부터 second order interaction을 얻어 이를 뉴럴넷에 태움으로써 higher order nonlinear interaction을 얻음.  



<br>
<center><img src="/assets/materials/recsys/inmobi_nfm/nffm.png" align="center" alt="drawing" width="600"/></center>   


<font size="2"><center> 출처: inmobi youtube 영상 (https://www.youtube.com/watch?v=MMTuoFFRCCs) </center>  </font>   
<br>

<br>

#### 4-2. Implementation details  

- Hyperparameters: k, lambda, num of layers, num of nodes in layers, activation functions  
- Tensorflow로 구현됨  
- Adam optimizer  
- L2 regularization, no dropout (추후 테스트 해볼 예정)  
- no batch norm (추후 테스트 해볼 예정)  
- 1 layer에 100개 노드로도 충분히 작동하고 잘 수렴함, 수가 늘어날수록 성능은 좋을 수 있으나 학습 시간이 길어짐  
- Relu activation: 빠른 수렴  
- k=16 (2의 제곱수로 실험해봄) 논문에는 수가 늘어날수록 성능이 좋아진다는 말이 있었으나 학습 시간 대비 효과가 없어 16으로 선정  
- 두 사용 사례에 모두 손실함수로써 weighted RMSE을 적용함. 여기서 가중치는 특정 combination을 볼 확률을 의미  
- unseen feature value에 대한 예측은, 같은 필드 내의 다른 latent vector를 평균내어 사용  
<br>

<br>

#### 4-3. implementing at low-latency, high-scale  
- MLeap: Spark와 Tensorflow로 학습된 모델을 지원하는 프레임워크로, 트리 기반 모델 학습을 위해 스파크로, 뉴럴넷 기반 모델 학습을 위해 텐서플로로 모델을 구현하는 데 도움을 줌  
- offline training and challenges: yarn 클러스터에서 TF 모델을 학습할 수 없으므로, HDFS에서 데이터를 끌어오는 게이트웨이로써 GPU 머신을 사용하여 GPU에서 모델을 학습  
- Online serving challenges: TF serving은 상당히 낮은 처리량(throughput)을 가지고(즉 느리고), 우리의 QPS에서 잘 scale되지 않았음. 우리는 decent TTL을 가진 로컬 LRU 캐시를 사용해 TF serving을 scale up함  
  - 이 아이디어는 먼저 대부분의 경우에 캐시로 접근하고, 처리가 안되면 그때야 TF 서빙으로 가서 값을 받아옴으로써 처리를 효율적으로 하는 것  

- (참고: throughput이 높다는 것은 가벼운 모델을 의미하고, throughput이 낮다는 것은 무거운 모델을 의미함)

<br>  

#### 4-4. Future Works  

- hybrid binning NFFM: FE에서 사용되는 binning을 적용해 NFFM  
- 분산처리된(distributed) training and serving  
- Dropouts & Batch norm  
- latent vector를 interpret 할 방법 (t-SNE 사용 등)  


<br>


----------------


**개선을 위한 여러분의 피드백과 제안을 코멘트로 공유해 주세요.**
**내용에 대한 지적, 혹은 질문을 환영합니다.**  


**출처**  

https://www.youtube.com/watch?v=MMTuoFFRCCs
