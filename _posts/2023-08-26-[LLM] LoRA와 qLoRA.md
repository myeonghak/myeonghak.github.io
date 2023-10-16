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

Memory Efficiency:

LoRA는 GPU 메모리를 효율적으로 사용하여 Fine-tuning에 필요한 GPU 메모리 필요량을 줄인다.  

Full Fine-tuning의 경우, 파라미터 외에도 gradient와 학습 관련 정보를 GPU에 로드해야 하므로 GPU 메모리가 크게 필요하다.
반면, LoRA는 일부 LoRA 파라미터와 관련된 기울기 값만 보존하므로 메모리가 효율적으로 사용된다.
Full Fine-tuning의 일반화:

적은 수의 파라미터로 전체 모델을 근사화하는 것이 가능하다.
이는 Matrix Factorization과 Dimensionality Reduction 기법의 아이디어와 유사하다.

No Additional Inference Latency:

LoRA 모델을 사용하여 다양한 tasks를 수행할 때 추가적인 지연이 거의 없다.
Low-Rank-Parameterized Update Matrices:

여기서 AW는 Low Rank Matrix인 A와 B를 순차적으로 곱한 것과 동일하다.  

LoRA in Detail:

모든 뉴럴넷은 행렬 연산을 포함한다.
선행 연구에서는 intrinsic dimensionality가 있을 것이라고 예상하며, 파라미터를 효율적으로 사용하여 원 행렬의 정보를 adaptation할 수 있을 것이라는 가설을 세웠다.
Adaptation은 pretrained 모델의 지식을 downstream task에 활용할 수 있도록 하는 과정을 의미한다.
d×k 형태의 원 weight matrix가 있을 때, 이를 복원하는 두 개의 low-rank 행렬 A와 B를 도입한다.

A는 r×k의 형태를, B는 d×r의 형태를 가진다. W0를 고정하고, A와 B의 파라미터만을 업데이트하여 downstream task의 loss function을 최적화한다.



<a id="qlora"></a>
## 3. QLoRA: 양자화된 모델에 LoRA 태우기  

QLoRA는 양자화된 대형 언어 모델을 효율적으로 파인튜닝하기 위한 기법입니다. QLoRA는 16/32비트의 FP precision으로 학습된 모델들을 양자화한 4비트 모델로 파인튜닝하며, 성능 저하를 최소화한 LoRA 방식을 도입합니다.  

QLoRA Overview:

대형 모델의 크기 증가에 따라 파인튜닝 시 VRAM 요구량이 크게 증가함.
양자화 (Quantization) 기법이 도입되었으나, 이는 추론 과정에서만 작동하였고, 파인튜닝 과정에서는 적용할 수 없었음.
QLoRA는 16/32비트의 FP precision으로 학습된 모델들을 양자화한 4비트 모델로 파인튜닝하며, 성능 저하를 최소화한 LoRA 방식을 도입함.
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
