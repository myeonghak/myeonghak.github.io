---
title: "(작성중) [RecSys] Multi-Armed Bandits"
categories:
  - Recommender Systems
tags:
  - Recommender Systems  
  - MAB  
  - Reinforcement Learning  
---

### 최대의 이득을 찾는 전략을 학습하는 강화학습 방식의 추천 알고리즘인 MAB에 대해 살펴 봅니다.     


> 어떤 슬롯머신이 어떤 수익률을 가지는지 모를 때,  
> 탐색(Exploration)과 활용(Exploitation)을 적절히 사용하여  
> 최적의 수익을 찾아내고자 하는 강화학습 알고리즘!  

﻿


<center><img src="/assets/materials/recsys/mab/mab_1.png" align="center" alt="drawing" width="500"/></center>   

<br>

>  **1. 연관 규칙 추천은 "A를 사면 B도 산다"는 규칙을 찾는 것**  
>
>  **2. "A를 사면"은 조건절(antecedent), "B도 산다"를 결과절(Consequent)이라고 부름**  
>
>  **3. 오직 빈번하게 등장하는 아이템 셋에 대해서만 고려하는 A priori 알고리즘을 적용해 빠른 규칙 생성이 가능**  
>   
>  **4. 파이썬의 mlxtend 라이브러리를 통해 쉬운 구현이 가능함**  




<br/>

----

#### Contents  

<br/>

1.  [Multi-Armed Bandits이란?](#mab)  
2.  [예시 상황 - 철용좌의 이야기](#story)  
3.  [탐색과 활용 (Exploration & Exploitation)](#ex-n-ex)
4.  [보상 최대화 알고리즘](#terms)  
5.  [코드 예제](#example)  

<br />



<a id="mab"></a>

## 1. Multi-Armed Bandits이란?  

 Multi-Armed Bandits은, 여러개(Multi)의 레버(Arm)를 가진 여러대의 슬롯머신(Bandits)이라는 뜻입니다.

이 알고리즘의 유래가, 과거 카지노에서 어떤 슬롯머신에 게임을 해야 최대한 많은 수익을 얻어낼 수 있을까? 하는 고민에서 출발했기 때문이라고 합니다.  

조금 더 기술적으로 정의하자면, 아래 세 가지의 제약이 *모두* 주어졌을 때의 문제 (multi-armed bandit problem)를 해결하는 알고리즘입니다.    
1) 한정된 자원 상황 하에 여러 개의 상충하는 선택을 내려야하는 경우  
2) 어떠한 선택이 얼마 만큼의 이득을 얼마 만큼의 편차로 제공하는지를 알 수 없을 때 (그러나 같은 선택을 여러번 내림에 따라 불확실성이 줄어들 때)  
3) 이 선택들의 예상되는 최대 이득이 극대화되도록 하고자 할 때  

이제부터는, 우리의 타짜 철용좌가 자신의 마지막 승부를 거는 상황을 예시로 들어 더욱 쉬운 말로 설명해 보고자 합니다.  





<a id="story"></a>

## 2. 예시 상황  

<center><img src="/assets/materials/recsys/mab/kwak_1.jpeg" align="center" alt="drawing" width="300"/></center>   

<br>


> 무너진 줄만 알았던 마포대교를 무사히 건넌 철용좌는 마지막 재산 1,000만원을 끌어 안고 인근의 유명한 도박장에 도착했습니다. 3대의 슬롯머신이 눈 앞에 놓여있고, 기계마다 수익률이 다르게 책정되어 있다는 사실을 알게 됩니다. 여기서 우리의 철용좌가 재기를 위해 최대한의 이익을 낼 수 있는 방법은 무엇일까요?  

<br>  

여기서 철용좌는 아직 3개의 bandit의 보상을 전혀 알지 못합니다. 철용좌는 마지막 자존심을 되찾을 수 있을까요?  

<center><img src="/assets/materials/recsys/mab/mab_2.png" align="center" alt="drawing" width="500"/></center>   

<br>


평소 머신러닝에 관심이 많던 철용좌는 이 마지막 기회를 최대한 살리기 위해, 강화학습의 기초적인 형태인 Multi-Armed Bandit 알고리즘을 적용해보고자 합니다. 그가 떠올린 기초적인 강화학습의 플로우는 다음과 같습니다. 일반적인 강화학습과는 달리, 환경이 변하지 않는다는 점에서 더욱 문제가 단순해졌습니다.    

<br>

<center><img src="/assets/materials/recsys/mab/mab_3.png" align="center" alt="drawing" width="500"/></center>   

<br>


여기에서 Agent는 우리의 강화학습 모델입니다. 이 Agent가 어떠한 액션(예를 들면, 버튼 2번을 누른다!)을 취하면, Environment(환경)은 우리에게 어떠한 보상을 주게 됩니다.


간단히 말해서, Agent가 선택을 잘 했으면 "잘했어~"하고 칭찬 해주고,  
반대로 잘못된 선택을 하면 "응, 왜 그거 골랐어~" 하고 혼내주는 것이지요.  


이러한 과정 끝에, Agent는 최종 시점에서 최대의 누적 수익을 얻기위해 자신의 전략(Policy)을 점점 수정해 나가고, 이러한 플로우가 잘 구성되어 있는 경우에 최적의 전략을 발견하면서 학습을 종료하게 됩니다.


앞서 말했던 칭찬과 면박은 각각 Reward의 수치값으로 표현됩니다. 잘했어~ 는 돈을 더 주는 것이고, 못했어~는 돈을 잃게 만드는 것이지요.  

여기서 보상은 두 가지의 형태를 가질 수 있을 것입니다.  

1) 일정한 보상 (Stationary Reward)  

만약 보상이 일정하다면 문제는 아주 간단하겠지요? 가령,
  - ① 번 버튼은 500원,  
  - ② 번 버튼은 100원,  
  - ③ 번 버튼은 -1000원이라고 해봅시다.  

그렇다면 몇번의 탐색 끝에 ① 번 버튼이 최고라는 것을 알게 될 것이고, 우리의 Agent는 ① 번을 남은 기회동안 눌러 최대의 보상을 얻게 될 것입니다.

2) 변칙적인 보상 (Non-Stationary Reward)  

하지만 현실은 역시 녹록치 않죠.  

  - ① 번 버튼의 보상은 첫 회에 500원이었다가 시간이 흐름에 따라 한 회마다 -50원씩 줄어듦  
  - ② 번 버튼은 100원이었다가 -10원, +20원, -30원, +40원...   
  - ③ 번 버튼은 -1000원에서 한 회마다 500원씩 늘어남   

이렇게 다양한 방식으로 보상이 변화할 수도 있을 것 같습니다. 이러한 상황을 변칙적인 보상(Non-Stationary Reward)라고 합니다.  

멋진 말로, 보상의 분포가 시간의 흐름에 따라 변화한다고 표현합니다. (Reward distribution changes over time)  

이 경우에는 조금 더 복잡한 접근이 필요하겠죠?  

﻿어떻게하면 이렇게 알 수 없는 보상을 정확히, 최소한의 손실로 알아낼 수 있을까요? 이를 위해, **탐색과 활용(Exploration and Exploitation) 전략**을 취하게 됩니다.  


<br>


<a id="ex-n-ex"></a>
## 3. 탐색과 활용 (Exploration & Exploitation)

우리의 철용좌는 단 1,000번 만을 시도할 돈만 가지고 있기 때문에, 제한된 기회 내에서 최대한의 수익을 얻어내야 합니다.

그런데, 3개 중 1개의 가장 우수한 수익을 주는 슬롯머신이 있다고 할 때,  
- 3개 중 어떤 슬롯이 가장 많은 이득을 줄 지 알 수 없는 상황에서 한가지에 모두 쏟아버리기에는 위험할 것 같고(exploitation only),
- 또 반대로 계속 3개를 골고루 골라버리기에는 최대의 이득을 못 누릴 것 같습니다. (exploration only)  

여기서, 두 번째 전략을 **탐험(Exploration)**, 첫 번째 전략을 **활용(Exploitation)** 이라고 합니다. 직관적으로는 다음과 같은 의미를 갖습니다.  

1) 탐험(Exploration): 어떤 버튼이 얼마의 보상을 주는지 탐험해보기  

2) 활용(Exploitation): 알아낸 보상을 바탕으로 돈을 뽑아먹기  

이 두 가지 전략을 적절히 섞어서 쓰는 것이 필요할텐데, 어떻게 풀어낼 수 있을까요? 이제 그 방법에 대해 알아보겠습니다.  



## 4. 보상 최대화 알고리즘  


탐색과 활용을 잘 절충해서, 보상을 최대화하는 세 가지 접근법에 대해 배워보겠습니다.


### 4-1. Greedy 알고리즘


다시 철용좌의 이야기로 돌아가 봅시다. 1,000번에 걸쳐 슬롯을 돌려 볼 돈이 있을 때, 30번 정도를 ① ② ③ 번 버튼에 각각 10번씩 투자한 결과가 다음과 같다고 합시다.  

﻿

<br>

<center><img src="/assets/materials/recsys/mab/mab_4.png" align="center" alt="drawing" width="500"/></center>   

<br>


이럴 때, 오케이, ① 번으로 나머지 970번 가즈아!


<br>

<center><img src="/assets/materials/recsys/mab/kwak_2.jpeg" align="center" alt="drawing" width="300"/></center>   

<br>


하는게 Greedy 알고리즘이라고 할 수 있겠습니다.  

가장 수익이 잘 나오는 버튼을 **"탐욕적으로(greedy)"** 소비하는 알고리즘이라 greedy라는 이름이 붙은듯합니다.  

이를 수식으로 표현하면 아래와 같습니다!


$q_*(a) \doteq \mathbb{E}[R_t \mid A_t = a] . $  

또한, $Q_t(a)$는 다음과 같이 정의됩니다.  

$Q_t(a) \doteq \frac{ sum \space of \space rewards \space  when \space a \space taken \space prior \space to \space t}{number \space of \space times \space a \space taken \space prior \space to \space t} = \frac{\Sigma_{i=1}^{t-1} R_i \cdot \mathbb{1}_{A_i=a}}{\Sigma_{i=1}^{t-1} \mathbb{1}_{A_i=a}}$  










----------------


**개선을 위한 여러분의 피드백과 제안을 코멘트로 공유해 주세요.**
**내용에 대한 지적, 혹은 질문을 환영합니다.**  


**출처**  
