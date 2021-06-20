---
title: "[XAI] GAM(Generalized Linear Models) 알고리즘 및 pyGAM 소개"
categories:
  - XAI
tags:
  - XAI
---

> GAM(Generalized Additive Models)의 파이썬 구현체인 pyGAM을 소개합니다.


<center><img src="/assets/materials/XAI/pyGAM/pygam_image.png" align="center" alt="drawing" width="400"/></center>    



>  **1. GAM=GLM(generalized linear model)+spline+smoothing penalty**
>
>  **2. 가법적인(additive) 구조 안에 비선형 함수를 적합시킬 수 있어 학습 성능과 설명 가능성을 동시에 달성함**
>
>  **3. pyGAM은 파이썬으로 짜여진 GAM 알고리즘으로 사이킷런과 잘 호환됨**


<br/>

----


1. Generalized Additive Models for python   
- 파이썬으로 짜여진 GAM 알고리즘으로 사이킷런과 잘 호환됨.  
 - 그리드 서치, 랜덤 서치 등의 하이퍼 파라미터 튜닝도 가능   


2. 빌딩 전력 예측 시나리오   
- 다양한 모델이 선택지가 될 수 있음   
- 그러나 실무진은 회의적임. 어떻게 믿을 수 있나? 곧 여름인데 새로운 계절적인 특성도 반영할 수 있나?  
 - 설명 가능성이 좋은 선형 모델은 너무 bias가 커서 우리의 데이터에 잘 맞지 않음.
 - 더 복잡한 모델은 블랙박스임.  
 - ARIMAX는 단지 계수만 보여줄 뿐임.  

3. GAM의 등장   
- gam은 독립적인 피처 함수의 합으로 이루어져있음   
- 선형 모델은 피처벡터 내에 있는 차원마다 하나의 계수를 갖는 구조임. 각 차원마다 하나의 선형 모델을 갖게 되며 이들을 모두 합한 형태가 최종적인 모델임   
- GAM 역시 이와 유사한데, 각 피처별로 비선형 함수를 더한다는 점이 차이점임.  


4. GAM이란   
- GAM=GLM(generalized linear model)+spline+smoothing penalty   
- spline: 쭉 0이다가 짧은 구간동안 0이 아닌 값이 나오는 함수. 저차원 다항식에서 나옴. 따라서 가장 단순한 형태의 spline은 box function임   
- spline의 재밌는 점은 두 개의 spline을 convolve하면 한 차원 높은 spline이 된다는 것임. 두 개의 box function을 convolve하면 1차원 spline을 얻을 수 있음. n차원 spline을 미분할 경우 n-1차 도함수를 가지게 되고, n+1개의 segment을 가짐.  
 - 이를 반복함으로써 3차원 spline을 얻으면 거기서 멈춤. 3차원 spline을 만들면 연속적인 2차 도함수를 얻을 수 있는데 이 그림이 시각적으로 설득적이기 때문. 3차원 spline을 얼마나 더하더라도 다 똑같은 연속적인 2차 도함수를 가지게 됨.  
- spline을 무한히 convolve하면 가우시안 함수가 나옴.  

5. Spline 더하기   
- 모든 spline을 일정한 grid로 배열하고, 저마다의 coefficient를 부여함으로써 가중치를 부과함.   
- 그 뒤 전부를 더해주게 되면, 우리의 feature function이 완성됨 - 비선형적으로 보이지만 선형적인 함수의 결합으로 만들어낸 모델임.  
 - 우리는 사후적으로 spline function에 대해 알고 있기 때문에, X에 대해서 각 spline function을 평가할 수 있음.   
- 따라서 spline function 기호를 수식에서 x hat으로 교체할 수 있고, 이를 행렬 연산으로 치환하면 GAM은 거대한 GLM이라고 할 수 있음  




6. GLM으로서의 GAM   
- linear model이라는 특성을 통해 GAM을 사용할 때 우리는 다양한 선형 모델의 특징을  그대로 사용가능  -> 신뢰구간, 예측구간, non-normal observation, p-value를 사용한 피처 선택, GCV를 통한 빠른 Cross Validation   
- 또, GAM의 고유한 성질을 이용 가능  -> 스무딩을 통한 직관적인 모델 정규화, 자동적으로 얻어지는 모델 비선형성, 여러 제약들(볼록성, 단조성, 주기성), 통제된 extrapolation   

7. Smoothing   
- 모델에  구불거림(wiggliness)를 추가하는 것은 앞의 선형식에 2차 도함수의 적분을 더해줌으로써 이룰 수 있음. 2차 도함수는 곡률을 의미하는 것을 상기   
- 그 앞에 lambda라는 하이퍼파라미터를 통해 강하게 구불거림을 penalize할지 약하게 할지 결정 가능. 즉 극단적으로 크면 선형식이 되고 극단적으로 작으면 오버피팅됨   
- 이 람다는 각 피처별로 하나씩 존재함. 4개 변수가 있으면 람다도 네개.  
 - 보통은 Leave-One-Out Cross Validation으로 최적의 람다를 찾음.   





**개선을 위한 여러분의 피드백과 제안을 코멘트로 공유해 주세요.**  
**내용에 대한 지적, 혹은 질문을 환영합니다.**  

참고: https://youtu.be/XQ1vk7wEI7c
