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
2.  [예시 상황](#story)  
3.  [연관 규칙 생성 알고리즘](#logic)  
3.  [코드 예제](#example)  

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





<a id="dataset"></a>

## 2. 예시 상황  







----------------


**개선을 위한 여러분의 피드백과 제안을 코멘트로 공유해 주세요.**
**내용에 대한 지적, 혹은 질문을 환영합니다.**  


**출처**  
