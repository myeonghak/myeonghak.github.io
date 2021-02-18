---
title: "[General ML Train set] batch 1. MLE와 MAP (작성중)"
categories:
  - General Machine Learning
tags:
  - MLE
  - MAP
---
### MLE(Maximum Likelihood Estimation)와 MAP(Maximum a Posteriori)


> 파라미터를 추정하는 방법론 중 하나인 MLE와 MAP의 개념에 대해서 알아봅니다.


<center><img src="/assets/materials/generalML/batch1/batch1.png" align="center" alt="drawing" width="400"/></center>    



>  **1. MLE와 MAP는 특정 확률분포, 혹은 머신러닝 모델의 파라미터를 추정하는 방법론이다**
>
>  **2. MLE는 파라미터 Θ(theta)가 주어졌을 때 우리가 가진 데이터셋이 발생할 가능성(likelihood)를 극대화하도록 유도하는 접근법이다**
>
>  **3. MAP는 **


<br/>

----

#### Contents

<br/>

1.	[압정 던지기 예시](#thumbtack-problem)
2.	[이항 분포](#binomial-distribution)
3.  [Maximum Likelihood Estimation](#MLE)
4.	[Simple Error Bound와 PAC learning](#simple-upper-bound)
5.	[사전지식 결합하기, 베이즈 정리](#incorporation-prior)
6.	[Maximum a Posteriori Estimation](#MAP)

<br />

<a id="thumbtack-problem"></a>
### 압정 던지기 예시
어떤 도박장에서, 압정을 던지는 게임을 하고 있다고 생각해 봅시다. 만약 윗면(뾰족한 면이 위) 혹은 아랫면에 돈을 걸고 맞추면 2배의 돈을 벌게 됩니다.  

어떤 갑부가 찾아와서, 우리에게 과학적이고 엔지니어링적인 도움을 요청했다고 해봅시다. 아주 큰 돈을 주면서 말이죠! 갑부의 질문은 다음과 같습니다:  

압정을 던졌을 때 나오는 윗면이 확률은 어떻게되지?!  



<center><img src="/assets/materials/generalML/batch1/thumbtack.png" align="center" alt="drawing" width="300"/></center>    

이 질문에 답하기 위해, 우리는 아마 몇번 던져보는 시도를 먼저 해볼 것 같습니다.  

5번을 던졌더니 3/5가 윗면이 나오고, 2/5가 아랫면이 나왔다고 해 보죠. 이에 따라서 배팅하면 된다고 말해주면 되겠지만, 뭔가 찜찜합니다. 너무 감에 의존하는 것 같지 않나요?  

우리가 제시한 답이 정말 해답인지, 그렇다면 어떤 근거로 그 답이 최적이라고 할 수 있는지 한번 알아봅시다.  

그 전에, 먼저 알아야 할 몇 가지 개념들이 있습니다.

-----

<a id="binomial-distribution"></a>

### 이항 분포 (Binomial Distribution)  

이항 분포는 discrete probability distribution(이산적 확률 분포)의 하나로, 각 시행이 $\theta$의 성공 확률과 1-$\theta$의 실패 확률을 가질 때 $n$개의 독립된 시행을 통해 성공한 횟수가 따르는 분포를 이릅니다.  

간단히 말해, 압정 던지기와 같은 실험을 해볼 때, 3/5(앞에서의 $\theta$)의 확률로 윗면이 나올(정의에서의 "성공") 때, 압정을 100번(정의에서의 n)을 던졌을 경우 윗면이 나온(정의에서의 "성공") 횟수가 따르는 분포라는 말입니다.  

각 시행은 i.i.d (independent and identically distributed) 즉 독립적이고 동일하게 분포되어 있습니다. 다음의 특성을 말합니다.
  - independent한 시행: 이전의 압정 던지기가 다음에 영향을 미치지 않음
  - 베르누이 분포에 따라 identically distributed: 압정의 손상이 없어서 위와 아래가 일정한 확률로 출현함  

자, 이제 우리의 압정 던지기 문제로 돌아가 봅시다. 이항 분포를 설명한 것은 압정 던지기라는 시행이 따르는 분포가 이항분포이기 때문이라는 것을 눈치 채셨을 것입니다.  

  - head가 나올 경우를 승으로 간주하고, 그 확률이 $\theta$ 라고 할 때, $D$(즉 우리가 가진 Dataset)=HHTHT라는 결과가 나왔다고 해 봅시다.
	- 이런 상황이 나올 확률을 계산 해 보면, 다음처럼 정리될 것 같네요.
    $$\theta^{3}\times (1-\theta )^{2}$$
  - 일반화해서 정리하면, 아래처럼 나타낼 수 있겠죠.  
    $$P(D|\theta ) = \theta^{(head)}\times (1-\theta )^{(tail)}$$  

  -  이런 상황에서, 어떻게하면 head가 나올 확률은 3/5이고 tail의 확률은 2/5라는 결론으로 도달할 수 있을까요?  



-----


<a id="MLE"></a>
### Maximum Likelihood Estimation
- 데이터: 관측한 시퀀스 데이터 D, a_H와 a_T로 이루어져 있음
- 우리의 hypothesis: 압정 던지기의 결과는 theta의 이항분포를 따른다
- 우리의 가설을 더욱 강하게 하는 법
1) 관측치에 대한 더 나은 분포를 찾는 것: 가능하지만, 더욱 엄밀한 검토가 필요
2) theta의 최고의 후보군을 만들어 내는 것: theta를 가장 plausible하게 만드는 조건은 무엇인가?
- 최고의 후보군을 만드는 방법 중 하나로 MLE를 들 수 있음. 이는 관측된 데이터의 확률을 최대화하는 theta를 찾아내는 것.
- theta^hat=argmax_thata P(D|theta) 즉, MLE인 theta^hat은 theta가 주어졌을 때 D라는 관측이 등장할 확률을 극대화해주는 theta를 가리킴

MLE calculation
- theta^hat=argmax_thata P(D|theta) =argmax_theta theta^head수*(1-theta)^tail수가 될것인데, 이는 처리하기 어려움
- 이를 해결하기 위해 흔히 사용하는 테크닉으로, 단조증가함수인 log를 양변에 취해줌. 단조증가 함수의 특성으로 인해 log를 취한 전이나 후나 같은 지점에서 MLE가 구해짐
- 이 이후에는 maximization problem임. theta에 대해 미분해서 최대화되는 theta 지점을 찾아주면 MLE가 구해짐.
- 결과적으로 theta^hat=a_H/(a_H+a_T)가 됨. 즉 이 사례에서는 3/5가 되는 것! (여기서 a_H는 H의 등장 횟수, a_T는 T의 등장 횟수)

시행 횟수(Number of trials)
- 만일 50번의 시행을 더 했더니 Head가 30번이 나왔다. 그렇다면?
- MLE를 통한 theta 값은 동일하게 0.6이 나온다. 그럼 아무것도 얻은 것이 없는건가?
- 그렇지 않다. 우리의 0.6이라는 theta^hat의 값은 estimation임. 따라서 이 추정의 오차를 줄였다고 할 수 있음


<a id="simple-upper-bound"></a>
### Simple Error Bound와 PAC learning

Simple Error Bound
- theta^hat을 우리의 MLE 추정치, N을 시행 횟수, theta*를 우리의 true parameter, trial error>0이라 할 때
- Hoeffding’s inequality에 의해 확률에 대한 simple upper bound를 가짐.
- 이는 풀어 설명하면, theta^hat와 theta*의 차이가 특정 error bound보다 클 확률은 시행 횟수 N이 커질수록, error bound가 커질수록 작아진다.
- 따라서 시행이 늘어났으므로 실제 파라미터 값과 추정 파라미터 값의 차이인 오차가 더 작아졌을 것이라는 주장을 할 수 있음!
- 이제 역으로, 오차가 0.1이 넘는 확률이 0.01%보다 낮게끔 만들 수 있는 N을 구할 수 있음

PAC learning(Probably Approximate Correct)
- probably(0.01% 확률)
- approximate(0.1 오차)
- 즉 0.01% 확률로 0.1이 넘는 오차가 나타날 theta^hat이 바로 0.6입니다 라고 하는 것이 PAC learning의 결과물임


<a id="incorporation-prior"></a>
### 사전지식 결합하기


사전 지식의 결합(Incorporating prior knowledge)
- 베이지안적인 접근법
- 앞에서의 압정 예시를 그대로 들고옴. 베이즈가 등장해서, 0.6이라는 확률이 맞는가? 50:50이 아닐까? 하는 의문을 제기하게 됨.
- 여기서 50:50일거라는 의견은 사전지식으로, 이를 결합하여 추정에 활용할 수 있음을 주장함
- P(theta|D)=P(D|theta)*P(theta)/P(D)
- Posterior=Likelihood*Prior Knowledge/Normalizing Constant
- 앞서, MLE 접근에서 P(D|theta)를 구했음. 여기에 P(theta)라는 사전지식을 가미해보면 좋을 것 같음.

베이지안 관점에서의 더 많은 공식
- P(theta|D) £(비례기호, proportion) P(D|theta)*P(theta) -> P(D)는 이미 주어진 상수이므로 소거하되, 등호를 비례기호로 바꾸어줌
- P(D|theta)=theta^a_H*(1-theta)^a_T로 나타낼 수 있음
- 이제 P(theta)를 나타내는 것이 관건임. 앞에서 나타낸 방식과 유사하게, 특정 분포에 의존해서 표현해야함.
- 여기서 베이즈는 beta dist.를 쓸 것을 제안함.
- 베타 분포는 0-1사이에 cdf(cumulative density function)가 confine되어 있어 확률을 나타내기 용이함.
- 이 분포에서는 alpha와 beta라는 parameter를 입력으로 받음.
- 이 식을 정리하면, P(D|theta)*P(theta)을 나타낼 수 있음. P(theta)항의 B(alpha, beta)는 theta에 독립적인 constant term이므로 이는 위에서 P(D)를 처리한 것과 같이 proportion으로 처리할 수 있음. (소거됨)
- 정리하면, MLE의 식과 유사한 꼴이 됨!


<a id="MAP"></a>
### Maximum a Posteriori Estimation

Maximum a Posteriori Estimation
- MLE에서는 likelihood를 최대화했지만, MAP에서는 사후확률을 최대화하는 전략을 취함!
- 즉 MLE는 P(D|theta)를 극대화하는 theta^hat을 찾고,MAP는 P(theta|D)를 극대화하는 theta^hat을 찾음
- 동일한 방식으로 log 변환 후 미분하여 극점을 사용한 최적화를 수행함
- 때문에 결과값은 MLE나 MAP나 동일함. 관점이 다른 것임!
-MAP에서 구한 공식에서는 alpha, beta에 대한 사전지식을 반영하여 theta^hat의 값을 변화시킬 방법이 있음. 가령 회장이 alpha와 beta가 반반이라고 생각하면 이를 반영할 수 있음.
- 그러나 MLE에는 그러한 방법이 없음.

결론
- 시행이 아주아주 많아지면, alpha와 beta의 영향은 점점 fade away하고 a_H와 a_T의 영향이 dominant해짐. 결국 MLE와 MAP의 결과는 같아지게 됨
- 관측치가 많지 않을 경우에는 우리의 사전지식인 alpha와 beta를 반영할 수 있음. 올바른 도메인 지식에 의해 도출될 경우 유용할 것임!

----------------

<a id="conclusion"></a>
### 닫으며  

지금까지 MLE와 MAP의 개념에 대해 알아 보았습니다.

**개선을 위한 여러분의 피드백과 제안을 코멘트로 공유해 주세요.**
**내용에 대한 지적, 혹은 질문을 환영합니다.**  

해당 포스트는 카이스트 문일철 교수님의 동의를 얻고, 강좌 내용과 자료, 그리고 reference의 자료를 참고하여 작성되었습니다. 원 강좌는 [여기](https://www.youtube.com/playlist?list=PLbhbGI_ppZISMV4tAWHlytBqNq1-lb8bz)에서 보실 수 있습니다.

**자료 공유를 허락해 주신 문일철 교수님께 감사의 말씀을 전합니다.**
