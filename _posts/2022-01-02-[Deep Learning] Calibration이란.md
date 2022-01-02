---
title: "[Deep Learning] 딥러닝 모델의 Calibration이란?"
categories:
  - Deep Learning
tags:
  - Deep Learning
  - Liquid Neural Network
---
### 딥러닝 모델의 Calibration이란?


> 분류 문제에서 출력값이 0-1 사이의 값으로 나올 때, 이 값이 모델의 확신 정도(confidence)를 잘 나타낼까요?


<center><img src="/assets/materials/DeepLearning/Calibration/calib_paper.png" align="center" alt="drawing" width="400"/></center>   


<font size="2"><center> 출처: On Calibration of Modern Neural Networks 논문 (https://arxiv.org/pdf/1706.04599.pdf) </center>  </font>   
<br>

<br/>


## calibration이란?
- **모형의 출력값이 실제 confidence (또는 이논문에서 calibrated confidence 로 표현) 를 반영하도록 만드는 것**

- calibration의 measure: 1) Reliability Diagram, 2) Expected Calibration Error (ECE), 3) Negative log likelihood  


-  현대 딥러닝 기법과 Calibration 의 관계  

	- Model capacity 와 Regularization 방법이 miscalibration 과 관계가 있다는 실험 결과를 제시  

	1) Model capacity: 많은 layer 를 가질 수록 traninig set 의 특징을 더욱 잘 학습하고, generalization 도 더 좋다는 것을 보여주었습니다.  

	2) Batch Normalization (BN): BN 의 도입은 calibration을 개선하나, 그 원인을 알 수 없음.  

	3) Weight decay : weight decay 를 크게 줄 수록 오히려 ece 가 좋아지며, weight decay 가 적을 수록 ece 가 증가함.  

<br/>

##  Calibration 방법  

- Post-processing calibration 은 모델의 예측 확률로부터 Calibrated probability 를 구하는 과정임.  

- 이 방법은 validation set 이 필수적으로 필요  

- Post-processing calibration의 목적은 모델의 예측값 p 로부터 calibrated probability q 를 구하는 것  

- 아래에는 이중 분류 상황에서의 calibration 방법론을 간략히 소개  


**1. Histogram binning**  

1) 예측값을 M 개의 bin 으로 쪼갬.  
-> bin 을 쪼개는 방법에는 같은 interval 로 쪼개는 방법, sample 수에 따라서 쪼개는 방법이 있음.    

2) 이후에 Bin 마다 Score 를 구함.  

3) Score 는 bin-wise squared loss 를 최소화하는 방법으로 선택됨.  


**2. Platt scaling**  
- Platt scaling 은 1999년 Platt 이 제시한 방법으로 SVM 의 출력을 '확률화' 하기 위한 방법으로 제시되었음  

- Platt scaling 은 histogram binning 과는 다르게 parametric 방법  

- 모형의 출력을 logistic regression의 입력값으로 넣음. 이 때,
q_i=σ(az_i+b)의 수식에서 a와 b는 validation set에 대해 NLL을 최소화 하는 방향으로 학습이 됨.  


**3. Temperature scaling**
- Temperature scaling 은 Platt scaling 에 기초한 방법  

- K 개의 label 이 있는 다중 분류 문제에서 Temperature scaling 방법에서는 단일 scalar parameter T 를 이용해 logit vector z 를 아래와 같이 변환  

- 저자는 T 를 temperature 라고 부릅니다. T 는 soft max function 을 "soften" 시키는 일을 함.  

- T 의 장점은 argmax 를 바꾸지는 않는다는 것임.  

- 즉, T 는 모델의 정확도에는 영향을 주지 않고, Calibration 에만 영향을 줌.  



----------------


**개선을 위한 여러분의 피드백과 제안을 코멘트로 공유해 주세요.**
**내용에 대한 지적, 혹은 질문을 환영합니다.**  


**출처**
https://3months.tistory.com/490
https://seing.tistory.com/41
https://arxiv.org/pdf/1706.04599.pdf
https://rocreguant.com/calibration-for-deep-learning-models/1734/  
