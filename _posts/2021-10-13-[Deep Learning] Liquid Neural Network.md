---
title: "[Deep Learning] Liquid Neural Network란?"
categories:
  - Deep Learning
tags:
  - Deep Learning
  - Liquid Neural Network
---
### Liquid Neural Network란?


> 시계열의 흐름에 따라 동적인 특성을 잡아내는 Liquid Neural Network에 대해 간단히 살펴봅니다.  


<center><img src="/assets/materials/DeepLearning/LNN/LNN.png" align="center" alt="drawing" width="400"/></center>   


<font size="2"><center> 출처: Liquid Time-constant Networks 논문 (https://arxiv.org/pdf/2006.04439.pdf) </center>  </font>   
<br>

<br/>


- 여러 AI 방법론은 입력 데이터와 출력 예측 사이의 고정된 맵핑만을 학습한다는 한계가 있음. 즉, 변화하는 맥락/환경을 제대로 잡아내지 못하고 그저 각각의 순간에 따른 판단을 내린다는 것. 객체 탐지 등의 문제에서 프레임마다 각각 판단을 내릴 뿐 사람이 하듯이 흐름에 따른 판단을 내리지는 못함
- 이러한 문제를 해결하기 위해 매우 큰 데이터셋을 수집하여 active learning하는 방식으로 접근함. 계속 재 라벨링하고, 재배치하고, 재학습하고.. 그러나 이러한 방식보다 더 나은 방법이 있을 것 같음
- 여기서 LNN(Liquid Neural Network)이 등장함. LNN은 RNN의 일종이라고 생각할 수 있음. RNN이 시퀀셜 데이터에 매우 좋은 성능을 보였듯이, LNN 역시 시간의 흐름에 따른 변화를 잡아내는 데 좋은 성과를 보임.
- 직관적으로 설명하자면 LNN은 시계열 내의 한 순간에서 공간적으로 동적인 히든 스테이트를 만들어 냄으로써 작동함.
- 매 순간마다 다음 스텝의 히든 스테이트에서 어떻게 변화할지를 예측하게 함
- 이로써 변화하는 환경에 따라 적응하는 예측 모델을 만들 수 있음
- 비디오 데이터에 대한 예측/분류 모델의 노이즈에 대한 강건성을 확보할 수 있는 효과적인 방법론으로 제안됨
- 시간의 흐름에 따른 판단을 내림으로써 설명 가능성과 맥락에 대한 판단 능력을 모델에 제공할 수 있음
- 매우 큰 연산량이 한계점으로 지적됨

----------------


**개선을 위한 여러분의 피드백과 제안을 코멘트로 공유해 주세요.**
**내용에 대한 지적, 혹은 질문을 환영합니다.**  


**출처**
https://www.youtube.com/watch?v=IlliqYiRhMU  
https://blog.roboflow.com/liquid-neural-netowrks/  
https://arxiv.org/pdf/2006.04439.pdf
