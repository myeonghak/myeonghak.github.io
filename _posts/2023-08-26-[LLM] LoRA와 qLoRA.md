---
title: "(WIP) [LLM] PEFT 기법 - LoRA와 qLoRA"
categories:
  - Large Language Models
  - Natural Language Processing
tags:
  - PEFT
  - LLM Finetuning
  - quantization
---

### Open Source LLM의 민주화를 연 파인튜닝 기법인 LoRA와 QLoRA 기법에 대해서 알아봅니다.  


> 원제
> **LoRA**: Low-Rank Adaptation of Large Language Models   
> **QLoRA**: Efficient Finetuning of Quantized LLMs      

<center><img src="/assets/materials/LLM/lora_qlora/lora_01.png" align="center" alt="drawing" width="400"/></center>   

<br>

----

**본 포스트는 LoRA와 QLoRA 논문을 리뷰한 내용과 huggingface transformers를 사용해 파인튜닝 하는 내용에 대한 설명을 포함하고 있습니다.**
원문은 여기에서 보실 수 있습니다.  
[LoRA paper](https://arxiv.org/abs/2106.09685),[QLoRA paper](https://arxiv.org/abs/2305.14314)      

----
#### Contents  

1. [Intro - 이 많은 파라미터를 굳이 다 학습해야하나?](#intro)  
2. [LoRA: 저차원으로 압축해보자!](#lora)  
3. [QLoRA: 양자화된 모델에 LoRA 태우기](#qlora)
4. [Fine-Tuning 실습: Huggingface API 활용](#implementation)
<br />

<a id="intro"></a>
## 1. Intro - 이 많은 파라미터를 굳이 다 학습해야하나?  

언어 모델의 크기가 성능에 비례한다는 실험적 결과가 점차 누적되면서 (LLM Scaling laws) LLM의 사이즈는 꾸준히 증가해 왔습니다.  
그러던 중 Meta의 LLAMA를 필두로 다양한 size의 LLM들이 open source로 공개되고 있는데요.  
지금에 와서(2023년 10월) 우리가 마주하게 된 open source LLM 모델의 파라미터 수는 적게는 7B, 많게는 180B에 이르게 되었습니다.  
그런데 그림의 떡이라고 할까요, 막상 공개되어도 우리같은 가난한 영세 개발자들은 써 볼 수가 없습니다.  
4080 한장 구하면 감사합니다 하는 와중에 80GB, 320GB VRAM을 얻어야만 모델을 학습할 수 있다니 애초에 가능한 선택지 같지 않을텐데요.  

이러한 상황에서 조금 더 효율적으로 모델을 파인튜닝 해보자! 하고 제안된 것이 PEFT 기법입니다.  
PEFT는 말 그대로 Parameter Efficient Fine-Tuning의 약자로, 소수의 파라미터로 원하는 Task를 학습시키는 전략입니다.  
이 많은 파라미터를 굳이 다 학습해야해? 라는 단순한 아이디어에서 시작된 참신한 해결책이라고 할 수 있겠습니다.  
한편 이와 다르게 Memory Efficient한 방식으로 Fine Tuning을 수행하는 기법도 연구되고 있습니다.  


<a id="lora"></a>
## 2. LoRA: 저차원으로 압축해보자!    

오늘 이야기 할 LoRA는 대형 언어 모델(LLM)을 파인튜닝할 때, 전체 파라미터를 파인튜닝 (Full Fine-tuning) 하는 대신 소수의 추가적인 파라미터만을 업데이트하여 효율적으로 파인튜닝하는 기법입니다.  
이 기법은 pretrained된 모델의 weight를 고정(freeze)하고, 이 큰 pretrained weight를 저 랭크(Low Rank)를 갖는 두 개의 행렬을 사용해 간접적으로 최적화하는 아이디어에 기반합니다.  
논문에서는 135B의 파라미터를 가진 원본 12,888차원의 정보를 1,2차원의 저차원에서 표현이 가능하다는 것이 실험을 통해 확인했다고 주장합니다. 이를 통해 파인튜닝에 필요한 시간과 공간 복잡도를 크게 줄일 수 있습니다.  
이 기법은 학습에 필요한 파라미터의 수를 0.01% 수준까지 줄이면서도 성능을 보존할 수 있어 혁신적입니다.  
GPT-3 모델을 학습할 때 필요한 메모리 량을 1.2TB에서 350GB 수준으로 줄일 수 있다고 합니다.  
그 외에도 이 rank를 줄일수록 큰 압축이 가능한데, rank=4로 설정한 Transformer 모델의 경우 350GB 소요량을 35MB로 10,000배 압축할 수 있다고 하네요.  

### LoRA의 장점  

이어서 LoRA가 왜 좋은지 알아보겠습니다.  

**1) Memory Efficiency**

- LoRA Adapter를 사용함으로써, GPU 메모리를 효율적으로 활용하여 Fine-tuning에 필요한 GPU 메모리 필요량을 줄일 수 있음  

- Full Fine-tuning의 경우, LLM 모델 자체의 파라미터 외에도 gradient 및 학습 관련 정보를 GPU에 로드해야 하므로 GPU 메모리가 크게 필요함    

- 반면, LoRA는 본 모델에 비해 더 작은 일부 LoRA 파라미터와 관련된 기울기 값만을 보존하므로 메모리가 효율적으로 사용됨  


**2) Full Fine-tuning의 일반화**  

- low rank 파라미터인 **r**의 값을 키워감에 따라, 기존의 모델을 근사화하는 것과 유사하다고 볼 수 있음
- 적은 수의 파라미터로 전체 모델을 근사화하는 것이 가능 -> 원본 모델의 정보를 denoise하여 잘 녹여낼 수 있는 구조를 가짐  
- 이는 Matrix Factorization과 Dimensionality Reduction 기법의 아이디어와 유사하다고 볼 수 있음  

**3) No Additional Inference Latency**  
- 다른 Downstream Task를 수행하고 싶다고 할 때, 새로운 LoRA 모델만을 학습하여 이들을 활용하면 됨
- 즉, 다양한 task를 번갈아가면서 수행할 때 발생하는 지연이 매우 적음 (no additional inference latency)  



### Low-Rank-Parameterized Update Matrices  

**$h = W_0x + \Delta Wx = W_0 + BAx$**

해당 수식은 맨 초반 부분의 논문 그림 자료를 그대로 수식화한 것인데요, 그 풀이는 아래와 같습니다.  
- 다음 레이어에서의 hidden state인 $h$는  
- 이전 레이어에서의 input tensor인 x를 각각  
- Freeze된 pretrained weight matrix인 W_0과 $\Delta W$에 각각 곱해준 것의 합과 같은데,
- $\Delta W$는 Low Rank Matrix인 $A$와 $B$를 순차적으로 곱해준 것과 동일하다  

이를 좀 더 들여다 보기 위해, 실제 peft 라이브러리를 적용해 생성한 LoRA 장착 모델의 구조를 살펴보겠습니다.  

- (추가 필요)



### LoRA in Detail
LoRA를 조금 더 자세히 들여다보겠습니다. 이 글을 보시는 분들은 모두 아시다시피, 모든 뉴럴넷은 행렬 연산을 포함합니다.  
저자들은 [선행 연구](https://arxiv.org/abs/2012.13255)에서 주장하듯이 파인튜닝 과정에서 중요 정보를 포함하는 내재 차원(intrinsic dimensionality)이 있을 것이라고 예상하며, 이 점에서 착안하여 더 파라미터 효율적으로 원 행렬의 정보를 adaptation할 수 있을 것이라는 가설을 세웠습니다.  
여기서 Adaptation이란, pretrained 모델이 가지고 있는 지식을 downstream task에 활용할 수 있도록 녹여내는 과정을 의미합니다.  

$d*k$ 형태의 원 weight matrix $W_0$가 있을 때, 이를 마치 auto-encoder처럼 복원해내는 두 개의 low-rank 행렬 $A$와 $B$를 도입합니다.  
여기서 $A$는 $r*k$, $B$는 $d*r$의 형태를 갖습니다.

#### LoRA의 학습 과정  
여기서 핵심 아이디어가 등장합니다.
$W0$를 freze하고, $A$와 $B$의 파라미터만을 업데이트한다고 말씀드렸는데요. 이 때, downstream task를 수행하기 위해 loss function을 최적화해 나가며 역전파 하는 과정을 주목할 만 합니다.  
나이브하게 말하자면, 문제를 해결하기 위해 필요한 $W_0$의 방대한 정보 중 필요한 핵심 지식이 $A$와 $B$에 녹아들게 되는 거죠.  
중요한 점은 첫 순전파 & 역전파 과정이 이루어질 때 random generate된 weight에서 생성된 정보가 노이즈로 작용하지 않도록 **$B$의 파라미터를 0으로 초기화**한다는 것입니다. 이러한 조치를 통해, 첫번째 순전파 시 LoRA 레이어의 정보가 전혀 전달되지 않습니다.  
이러한 과정을 통해 첫 번째 역전파부터 파라미터가 업데이트 됨으로써 adaptation이 일어나게 됩니다. (adaptation을 수행하는 주체인 LoRA 모델을 Adapter라 부르는 이유이기도 합니다.)  




<a id="qlora"></a>
## 3. QLoRA: 양자화된 모델에 LoRA 태우기  

LoRA를 통해 파인 튜닝 과정에서 업데이트하는 파라미터의 수를 획기적으로 줄임으로써, 메모리/연산 효율적으로 우리의 파인튜닝 과정을 수행할 수 있었습니다. 그런데, 가난한 우리네 영세 개발자들은 이 모델 자체 마저도 메모리에 올릴 수 없는 상황에 처한 경우가 많은데요. 이러한 문제를 해결하기 위해 제한된 GPU 메모리 자원으로도 큰 언어 모델을 로드할 수 있도록 하는 방법이 제안되었는데, 이게 바로 **양자화(Quantization)** 기법입니다. 이에 대한 내용은 아래에서 더 살펴 보도록 하겠습니다.    

QLoRA는 **양자화**된 대형 언어 모델을 효율적으로 파인튜닝하기 위한 기법입니다. 간단하게 말해, 양자화와 LoRA를 더한 것이라고 이해할 수 있겠죠.

### QLoRA Overview

언어 모델의 크기 증가에 따라 파인튜닝 시 VRAM 요구량이 크게 증가해 왔습니다. 파라미터

양자화 (Quantization) 기법이 도입되었으나, 이는 추론 과정에서만 작동하였고, 파인튜닝 과정에서는 적용할 수 없었음.
QLoRA는 16/32비트의 FP precision으로 학습된 모델들을 양자화한 4비트 모델로 파인튜닝하며, 성능 저하를 최소화한 LoRA 방식을 도입합니다.  

양자화 (Quantization):
추론 단계에서 적용하는 경량화 기법.
16비트로 표현된 가중치는 양자화 상수로 quantize 후 다시 16-bit 데이터 타입으로 de-quantize하여 모델 가중치로 사용.
양자화된 LLM 모델은 Hugging Face의 bitsandbytes 라이브러리를 통해 로드할 수 있음.
QLoRA에서 적용된 기법:

4비트 NormalFloat 양자화:
양자화할 때, 각 양자화 구간에 동일한 수의 값이 들어가게끔 이상치가 발생시키는 문제를 최소화.
이중 양자화 (Double Quantization):
양자화 상수를 다시 양자화하여 메모리 절약.
페이징 옵티마이저:
시퀸스 길이가 긴 미니 배치 처리 시 그라데이션 체크포인트 메모리 스파이크를 피하기 위해 NVIDIA 통합 메모리를 활용.


<a id="implementation"></a>
## 4. Fine-Tuning 실습: Huggingface API 활용

실제로 LoRA와 QLoRA를 어떻게 적용하는지, 그리고 이들의 코드를 어떻게 구현하는지에 대한 내용을 살펴보겠습니다.  
현실적으로 colab 환경에서만 실행이 가능한데, 이를 위해서는 코랩 프로를 구독하고, A100 GPU를 할당 받아야하는 번거로움이 있습니다.  




----------------


**개선을 위한 여러분의 피드백과 제안을 코멘트로 공유해 주세요.**
**내용에 대한 지적, 혹은 질문을 환영합니다.**  


**출처**  

https://www.youtube.com/watch?v=G2PoGAyg-1k&t=1672s
