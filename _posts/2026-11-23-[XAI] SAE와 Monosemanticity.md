---
title: "[XAI] SAE와 Monosemanticity - LLM 내부를 사후에 분해하는 방법"
categories:
  - XAI
tags:
  - Sparse Autoencoder
  - Mechanistic Interpretability
  - Superposition
  - LLM
---

### 뉴런 하나에 여러 개념이 겹쳐 있는 LLM의 활성값을, Sparse Autoencoder로 "한 가지 뜻만 갖는" feature 단위로 풀어내는 방법에 대해 알아봅니다.


> 다루는 글  
> **Toy Models of Superposition** (Elhage et al., Anthropic, 2022)  
> **Towards Monosemanticity**: Decomposing Language Models With Dictionary Learning (Bricken et al., Anthropic, 2023)  
> **Sparse Autoencoders Find Highly Interpretable Features in Language Models** (Cunningham et al., 2023)  
> **Scaling Monosemanticity**: Extracting Interpretable Features from Claude 3 Sonnet (Templeton et al., Anthropic, 2024)  

<center><img src="/assets/materials/XAI/sae/sae_05.png" align="center" alt="Claude 3 Sonnet에서 추출한 feature 지도와 sycophantic praise feature의 활성 예시, 그리고 이 feature를 고정했을 때의 응답" width="700"/></center>

<font size="2"><center> 출처: Templeton et al., Scaling Monosemanticity (https://transformer-circuits.pub/2024/scaling-monosemanticity/) </center>  </font>

<br/>


>  **1. LLM의 뉴런은 서로 관계없는 여러 개념에 동시에 반응하는 polysemantic한 경우가 많고, 그 원인으로 뉴런 수보다 많은 feature를 겹쳐 저장하는 superposition이 지목됨**
>
>  **2. SAE는 이미 학습된 모델의 활성값(activation)을 받아 "복원 오차 + L1 sparsity" 를 최소화하도록 따로 학습하는 작은 autoencoder이며, 모델 가중치는 전혀 건드리지 않음**
>
>  **3. 1층 transformer(512 뉴런 -> 최대 131,072 feature)에서 시작해 Claude 3 Sonnet(최대 약 3,400만 feature)까지 확장되었고, feature를 고정(clamp)해 모델 행동을 바꾸는 steering으로 해석을 검증함**
>
>  **4. 다만 복원 오차, feature splitting, dead feature, 막대한 비용, "활성값은 설명하지만 가중치는 설명하지 않는다" 는 한계가 분명하며, 2025년에는 SAE의 실효성을 두고 논쟁이 이어짐**

<br/>

----

**본 포스트는 Anthropic의 Transformer Circuits 글들과 Cunningham et al.(2023)을 읽고 정리한 노트입니다. 간단한 PyTorch toy 실험도 함께 담았습니다.**  
원문은 글 하단의 출처에서 보실 수 있습니다.

----
#### Contents

1. [Intro - 해석은 어디에서 일어나는가](#intro)
2. [Polysemanticity와 Superposition](#superposition)
3. [Sparse Autoencoder의 구조와 학습 목표](#sae)
4. [Toy 실험: 겹쳐진 feature를 SAE로 되찾아 보자!](#toy)
5. [Feature는 어떻게 해석하고 검증할까?](#interpret)
6. [Towards Monosemanticity: 1층 transformer에서의 결과](#towards)
7. [Scaling Monosemanticity: Claude 3 Sonnet으로의 확장](#scaling)
8. [한계와 그 이후의 흐름](#limits)
9. [마치며](#conclusion)
<br />

<a id="intro"></a>
## 1. Intro - 해석은 어디에서 일어나는가

이 블로그의 XAI 카테고리에서는 예전에 [LIME](https://myeonghak.github.io/xai/XAI-LIME%28Local-Interpretable-Model-agnostic-Explanation%29-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98/), [SHAP](https://myeonghak.github.io/xai/XAI-SHAP/), 그리고 [Captum으로 추천 모델 설명하기](https://myeonghak.github.io/xai/XAI-Captum%EC%9C%BC%EB%A1%9C-%EC%B6%94%EC%B2%9C-%EB%AA%A8%EB%8D%B8-%EC%84%A4%EB%AA%85%ED%95%98%EA%B8%B0/) 같은 글을 다뤘는데요. 이 방법들은 공통적으로 **모델을 블랙박스로 두고 입력과 출력 사이의 관계** 를 보는 방식입니다. "이 예측에 어떤 입력 변수가 얼마나 기여했나?" 를 묻는 것이죠.

그런데 LLM 시대가 되면서 질문이 조금 바뀌었습니다. "어떤 토큰이 중요했나" 를 넘어서, **"모델 안에서 지금 어떤 개념이 켜져 있나?"** 를 알고 싶어진 것인데요. 이번 시리즈는 이 흐름을 **"해석이 어디에서 일어나는가"** 라는 축으로 정리해 보려고 합니다.

| 단계 | 해석의 대상 | 모델 가중치 | 대표 방법 |
|---|---|---|---|
| 1 | 입력과 출력의 관계 (사후) | 건드리지 않음 | LIME, SHAP, Integrated Gradients |
| 2 | **내부 활성값의 분해 (사후)** | **건드리지 않음** | **SAE (이번 글)** |
| 3 | 가중치 자체를 해석 가능하게 학습/편집 | 학습 또는 편집 | circuits, ROME, weight-sparse transformer |
| 4 | 가중치를 직접 작성 | 사람이 작성 | RASP/Tracr, iWSP |

이번 글은 그중 두 번째 단계인 **Sparse Autoencoder(SAE)** 를 다룹니다. 미리 강조해 두면, **SAE는 모델의 가중치를 전혀 수정하지 않습니다.** 이미 학습이 끝난 모델을 그대로 돌리면서 중간 활성값을 수집하고, 그 활성값을 설명하는 별도의 작은 모델을 옆에 붙여 학습하는 방식이에요. 즉 여전히 "사후(post-hoc)" 해석인데, 보는 위치가 입력/출력에서 모델 내부로 한 칸 들어왔다고 생각하시면 됩니다.


<a id="superposition"></a>
## 2. Polysemanticity와 Superposition

### 뉴런 하나가 왜 여러 뜻을 가질까?

Mechanistic interpretability는 신경망을 이해 가능한 부품으로 쪼개고, 부품들의 상호작용으로 전체 동작을 설명하려는 분야입니다. 가장 자연스러운 부품 후보는 당연히 **뉴런** 이겠죠. 문제는 뉴런이 생각보다 "한 가지 일" 만 하지 않는다는 점입니다.

Towards Monosemanticity 논문은 재미있는 예시를 하나 드는데요, 논문에서 분석한 작은 언어 모델의 한 뉴런이 **학술 인용, 영어 대화, HTTP 요청, 그리고 한국어 텍스트** 에 동시에 반응했다고 합니다. (한국어가 여기서 등장할 줄은 몰랐네요.) 이렇게 서로 관계없는 여러 입력에 반응하는 뉴런을 **polysemantic** 하다고 하고, 반대로 한 가지 개념에만 반응하는 단위를 **monosemantic** 하다고 부릅니다.

### Superposition 가설

Polysemanticity의 유력한 원인으로 제시된 것이 **superposition** 입니다. 모델이 표현하고 싶은 "feature" 의 수가 뉴런(차원) 수보다 훨씬 많을 때, 각 feature를 하나의 뉴런에 배정하는 대신 **여러 뉴런의 선형 결합(방향)** 으로 저장한다는 가설인데요.

<center><img src="/assets/materials/XAI/sae/sae_01.png" align="center" alt="superposition 가설: 관측되는 작은 모델은 더 크고 disentangled된 가상의 모델을 저차원으로 투영한 것" width="650"/></center>

<font size="2"><center> 출처: Bricken et al., Towards Monosemanticity (https://transformer-circuits.pub/2023/monosemantic-features) </center>  </font>

위 그림이 이 가설을 잘 보여줍니다. 위쪽의 "가상의 disentangled 모델" 에서는 뉴런 하나가 feature 하나에 대응하지만, 우리가 실제로 관측하는 모델은 이것을 저차원으로 **투영한(projected)** 것이라는 시각입니다. 투영된 공간에서는 feature들이 거의 직교(almost orthogonal)하는 방향으로 겹쳐 있으니, 개별 뉴런 입장에서 보면 여러 feature가 섞여 보이는 것이죠.

이게 어떻게 가능할까요? 핵심은 **sparsity** 입니다. Toy Models of Superposition(Elhage et al., 2022)은 작은 toy 모델로 실험해, 학습 데이터에서 feature들이 **드물게(sparse) 등장할수록** 모델이 자연스럽게 superposition을 학습한다는 것을 보였습니다. 대부분의 feature가 동시에 켜지지 않으니, 서로 완전히 직교하지 않은 방향에 겹쳐 담아도 간섭(interference)이 크지 않은 것인데요. compressed sensing에서 sparse 신호를 적은 측정값으로 복원할 수 있는 것과 같은 원리입니다.

### 그럼 처음부터 superposition 없는 모델을 만들면?

Anthropic 팀도 처음에는 이 방향을 시도했다고 합니다. 학습 중 활성값을 sparse하게 강제하고, 심지어 1-hot 활성값까지 밀어붙여 봤는데요. 결과는 superposition은 사라졌지만 **뉴런은 여전히 polysemantic** 했습니다.

논문은 이를 간단한 계산으로 설명합니다. 뉴런이 1개뿐이고 서로 배타적인 feature A/B/C/D가 있을 때,

- 뉴런이 A에만 반응하면 cross-entropy loss는 $\frac{3}{4}\ln 3 \approx 0.8$
- 뉴런이 A와 B에 함께 반응하면 loss는 $\ln 2 \approx 0.7$

즉 **cross-entropy로 학습하는 모델은 superposition이 불가능한 상황에서도 polysemantic한 쪽을 더 선호** 할 수 있다는 것입니다. 그래서 Anthropic은 "아키텍처를 바꿔서 monosemantic한 LLM을 만드는 것" 대신, **학습이 끝난 모델의 활성값을 사후에 분해하는 dictionary learning** 쪽으로 방향을 잡게 됩니다. (참고로 가중치 수준에서 해석 가능성을 확보하려는 다른 시도들은 다음 글에서 다룹니다.)


<a id="sae"></a>
## 3. Sparse Autoencoder의 구조와 학습 목표

### Dictionary learning으로서의 SAE

활성값 벡터 $\mathbf{x}$ 가 "소수의 feature 방향들의 가중합" 으로 표현된다고 가정해 봅시다.

$$
\mathbf{x} \approx \mathbf{b} + \sum_i f_i(\mathbf{x})\, \mathbf{d}_i
$$

여기서 $\mathbf{d}_i$ 는 feature $i$의 방향(unit vector), $f_i(\mathbf{x})$ 는 그 feature의 활성 강도입니다. superposition 가설이 맞다면 방향 $\mathbf{d}_i$ 의 개수는 뉴런 수보다 많아야 하고(**overcomplete**), 각 데이터 포인트에서 켜지는 $f_i$ 는 극히 일부여야(**sparse**) 하겠죠. 이건 고전적인 sparse dictionary learning 문제 그대로입니다.

SAE는 이 문제를 신경망으로 근사하는 방법입니다. 구조는 정말 단순한데요.

<center><img src="/assets/materials/XAI/sae/sae_02.png" align="center" alt="SAE 구조: d차원 활성값을 m차원 sparse feature로 인코딩한 뒤 다시 d차원으로 복원" width="650"/></center>

- **Encoder**: $f(\mathbf{x}) = \text{ReLU}(W_{enc}(\mathbf{x} - \mathbf{b}_{dec}) + \mathbf{b}_{enc})$  
- **Decoder**: $\hat{\mathbf{x}} = \mathbf{b}_{dec} + \sum_{i=1}^{F} f_i(\mathbf{x})\, W_{dec,\cdot i}$  
- decoder 가중치의 각 열 $W_{dec,\cdot i}$ 가 곧 feature $i$의 **방향** 이 되고, 은닉층 크기 $F$ 가 **dictionary size** 임

### 손실 함수: 복원 + L1 sparsity

학습 목표는 "원래 활성값을 잘 복원하되, 켜지는 feature 수는 최소한으로" 입니다. Scaling Monosemanticity 기준으로 쓰면 아래와 같습니다.

$$
\mathcal{L} = \mathbb{E}_{\mathbf{x}}\left[ \lVert \mathbf{x} - \hat{\mathbf{x}} \rVert_2^2 + \lambda \sum_i f_i(\mathbf{x}) \cdot \lVert W_{dec,\cdot i} \rVert_2 \right]
$$

- 첫 항은 **복원 오차(MSE)**, 두 번째 항은 feature 활성값에 대한 **L1 penalty** 임
- L1 항에 decoder 열의 norm을 곱하는 이유는, $f_i$ 를 작게 만들고 decoder 열을 크게 키워서 L1 penalty를 "속이는" 꼼수를 막기 위함임 (Cunningham et al.은 대신 decoder 행을 정규화하고, encoder와 decoder를 전치 관계로 묶는 tied weights를 사용함)
- $\lambda$ 는 복원 품질과 sparsity 사이의 trade-off를 조절하는 핵심 하이퍼파라미터임

앞에서 cross-entropy 모델은 polysemantic을 선호할 수 있다고 했는데, SAE는 **MSE로 학습** 하기 때문에 이 문제를 피해 간다는 점도 논문에서 짚고 있습니다. 재밌는 포인트죠.

### Dead feature 문제

SAE를 학습하다 보면 어떤 feature는 학습 도중 아예 켜지지 않게 되는데, 이를 **dead feature(dead neuron)** 라고 합니다. dictionary를 크게 잡아 놓아도 상당수가 죽어 버리면 그만큼 낭비겠죠. Towards Monosemanticity에서는 주기적으로 죽은 feature를 찾아, **현재 SAE가 잘 복원하지 못하는 데이터 포인트 쪽으로 encoder 가중치를 재초기화(resampling)** 하는 방법을 썼습니다. 이렇게 하면 해석 가능한 feature 수가 늘고 loss도 줄었다고 합니다.

여기서 다시 한 번 정리하면, 이 모든 학습은 **LLM과 별개인 SAE 파라미터($W_{enc}, W_{dec}$, bias)** 에 대해서만 일어납니다. LLM은 활성값을 뽑아내는 용도로만 forward를 돌릴 뿐, 가중치는 그대로입니다.


<a id="toy"></a>
## 4. Toy 실험: 겹쳐진 feature를 SAE로 되찾아 보자!

말로만 들으면 감이 잘 안 오니, 정답을 아는 synthetic 데이터로 직접 확인해 보겠습니다. 설정은 이렇습니다.

- 정답 feature 16개를 8차원 공간의 랜덤 방향에 배정함 (feature 수 > 차원 수 -> superposition)
- 각 feature는 5% 확률로만 켜짐 (샘플당 평균 0.8개만 활성)
- 8차원 "뉴런" 활성값만 보고 32차원 SAE를 학습한 뒤, 정답 방향을 얼마나 되찾는지 cosine similarity로 측정함
- $\lambda$ 를 바꿔 가며 trade-off를 확인함

```python
import torch
import torch.nn.functional as F
torch.manual_seed(0)

# 1) 정답 feature 16개를 8차원 공간에 겹쳐 넣은(superposition) synthetic 데이터
n_feat, d, N = 16, 8, 50_000
true_dirs = F.normalize(torch.randn(n_feat, d), dim=1)
coef = (torch.rand(N, n_feat) < 0.05).float() * torch.rand(N, n_feat)  # 각 feature는 5%만 켜짐
X = coef @ true_dirs                                                   # (N, 8) "뉴런" 활성값

# 2) SAE: x -> f = ReLU(W_enc(x - b_dec) + b_enc) -> x_hat = f @ W_dec + b_dec
class SAE(torch.nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.b_dec = torch.nn.Parameter(torch.zeros(d))
        self.enc = torch.nn.Linear(d, m)
        self.W_dec = torch.nn.Parameter(torch.randn(m, d) * 0.1)
    def forward(self, x):
        f = torch.relu(self.enc(x - self.b_dec))
        return f @ F.normalize(self.W_dec, dim=1) + self.b_dec, f   # decoder 방향은 unit norm

def train(lam, m=32, steps=6000):
    torch.manual_seed(1)
    sae = SAE(d, m)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    for _ in range(steps):
        x = X[torch.randint(0, N, (1024,))]
        x_hat, f = sae(x)
        loss = ((x - x_hat) ** 2).sum(1).mean() + lam * f.abs().sum(1).mean()  # MSE + L1
        opt.zero_grad(); loss.backward(); opt.step()
    return sae

# 3) 평가: 정답 방향별로 가장 가까운 decoder 방향과의 cosine similarity
print(f"true avg L0 = {(coef > 0).float().sum(1).mean():.2f}")
print("lambda | recovered(cos>0.9) | mean max-cos | avg L0 | FVU   | alive")
for lam in [0.003, 0.03, 0.1, 0.3]:
    sae = train(lam)
    with torch.no_grad():
        x_hat, f = sae(X)
        best = (true_dirs @ F.normalize(sae.W_dec, dim=1).T).max(1).values
        fvu = ((X - x_hat) ** 2).sum() / ((X - X.mean(0)) ** 2).sum()
    print(f"{lam:<6} | {(best > 0.9).sum().item():>2}/16              | {best.mean():.3f}"
          f"        | {(f > 0).float().sum(1).mean():.2f}   | {fvu:.3f} | {(f > 0).any(0).sum().item()}/32")
```

실제 실행 결과는 다음과 같습니다.

```
true avg L0 = 0.80
lambda | recovered(cos>0.9) | mean max-cos | avg L0 | FVU   | alive
0.003  |  4/16              | 0.808        | 5.56   | 0.000 | 30/32
0.03   |  6/16              | 0.861        | 3.98   | 0.006 | 30/32
0.1    | 13/16              | 0.938        | 2.30   | 0.034 | 30/32
0.3    | 16/16              | 0.990        | 1.07   | 0.115 | 25/32
```

결과가 꽤 교과서적으로 나왔는데요.

- **$\lambda$ 가 작을 때 (0.003)**: 복원은 거의 완벽(FVU, 즉 설명하지 못한 분산 비율이 0.000)하지만, 샘플당 5.56개의 feature가 켜지고 정답 방향은 16개 중 4개만 찾음. 복원만 잘하는 "dense한 아무 기저" 를 학습한 셈임
- **$\lambda$ 가 클 때 (0.3)**: 16개 정답 방향을 모두 cosine 0.9 이상으로 되찾고(평균 0.990), 평균 L0도 1.07로 실제 값 0.8에 가까워짐. 대신 복원 오차가 11.5%로 커지고, 32개 중 7개 feature가 죽음
- 즉 **해석 가능성(sparsity)과 복원 충실도(faithfulness) 사이에 trade-off가 있고**, 그 사이 어딘가를 사람이 골라야 함

<center><img src="/assets/materials/XAI/sae/sae_03.png" align="center" alt="왼쪽: 뉴런별로 여러 정답 feature 성분이 섞여 있는 모습, 오른쪽: λ=0.3에서 정답 feature와 SAE decoder 방향의 cosine similarity 행렬" width="700"/></center>

왼쪽 그림은 8개 뉴런 각각에 여러 정답 feature의 성분이 고르게 섞여 있는 모습, 즉 **뉴런 단위로는 polysemantic** 한 상태를 보여줍니다. 오른쪽은 $\lambda=0.3$ 에서 학습된 SAE의 decoder 방향과 정답 feature 사이의 cosine similarity인데, 대각선에 진한 빨간 칸이 줄지어 있죠. 뉴런 기저에서는 뒤섞여 있던 feature가 SAE 기저에서는 하나씩 분리된 것입니다.

물론 실제 LLM에는 이런 "정답" 이 없다는 게 가장 큰 차이입니다. 그래서 다음 질문이 중요해집니다. 정답이 없을 때 feature가 맞게 뽑혔는지 어떻게 알 수 있을까요?


<a id="interpret"></a>
## 5. Feature는 어떻게 해석하고 검증할까?

Towards Monosemanticity의 저자들도 "test loss처럼 한 눈에 볼 수 있는 지표를 찾으려 했지만 만족스러운 것을 찾지 못했다" 고 솔직하게 적고 있습니다. 그래서 여러 증거를 겹쳐서 판단하는데요, 크게 네 가지로 정리할 수 있습니다.

**1) 활성 예시(activating examples) 보기**
- feature가 가장 강하게 켜지는 데이터셋 문장들을 모아서 사람이 공통점을 찾음
- 단, 최댓값 예시만 보면 polysemantic한 뉴런도 monosemantic해 보일 수 있음. 그래서 Towards Monosemanticity는 활성 범위를 11개 구간으로 나누어 **약하게 켜지는 구간까지 골고루** 샘플링해 평가함

**2) 출력 쪽 효과(logit weights) 보기**
- feature 방향이 다음 토큰 logit을 어떻게 바꾸는지 확인함. "아랍 문자에서 켜지는 feature" 라면 아랍 문자 토큰의 확률을 올려야 앞뒤가 맞겠죠
- SAE 학습에는 출력 쪽 정보를 전혀 쓰지 않으므로, 이건 일종의 독립적인 검증이 됨

**3) Automated interpretability**
- 사람이 수천, 수백만 개를 다 볼 수 없으니 LLM에게 맡김 (Bills et al., 2023의 방식)
- LLM이 활성 예시를 보고 feature 설명을 쓰고 -> 그 설명만 가지고 처음 보는 텍스트에서 활성값을 예측하게 한 뒤 -> 실제 활성값과의 상관을 점수로 씀
- Towards Monosemanticity는 Claude를, Cunningham et al.은 GPT-4를 설명 생성에 사용함

**4) 개입(intervention)과 steering**
- 해석이 맞다면, 그 feature를 인위적으로 켜거나 끄면 모델 행동도 해석대로 바뀌어야 함
- Scaling Monosemanticity는 residual stream을 "SAE 복원값 + 복원 오차" 로 나눈 뒤, 복원값 안에서 특정 feature 값만 고정(clamp)하고 오차 항은 그대로 둔 채 forward를 진행함
- 여기서도 바뀌는 것은 **추론 중의 활성값** 이지 가중치가 아님

이 마지막 방법을 대중에게 가장 인상적으로 보여준 사례가 바로 **Golden Gate Claude** 입니다. 2024년 5월, Anthropic은 Golden Gate Bridge feature를 강하게 켠 Claude를 24시간 동안 연구 데모로 공개했는데요. 10달러를 어떻게 쓸지 물으면 금문교 통행료를 내라고 하고, 자기가 어떻게 생겼을지 물으면 금문교처럼 생겼다고 답하는 식이었습니다. Anthropic은 이것이 system prompt도, fine-tuning도 아닌 **내부 활성값에 대한 정밀한 개입** 이라는 점을 강조했습니다.


<a id="towards"></a>
## 6. Towards Monosemanticity: 1층 transformer에서의 결과

### 실험 설정

| 항목 | Transformer | SAE |
|---|---|---|
| 구조 | attention block 1개 + ReLU MLP block 1개 | ReLU(up) 1층 + Linear(down) 1층 |
| 크기 | MLP 512 뉴런 | 512 (1x) ~ 131,072 (256x) feature |
| 데이터 | The Pile, 1,000억 토큰 | MLP 활성값 80억 개 |
| Loss | autoregressive log-likelihood | L2 복원 + L1 |

- "가장 단순하지만 우리가 제대로 이해하지 못하는 언어 모델" 로 1층 transformer를 골랐다고 함
- 같은 설정에서 seed만 다른 두 transformer(A, B)를 학습해, 서로 다른 모델에서 비슷한 feature가 나오는지(universality)도 확인함
- 주로 분석한 것은 4,096개 feature를 가진 **A/1** 실행임

### 주요 결과

**1) 뉴런에서는 보이지 않던 feature가 나타남**
- 대표 예시가 **아랍 문자 feature(A/1/3450)**. 아랍 문자는 전체 학습 토큰의 0.13%에 불과하지만, 이 feature가 켜진 토큰의 81%를 차지함
- DNA 서열, base64, 히브리 문자 feature 등도 상세 분석함. 특히 히브리 문자 feature는 **어떤 뉴런의 top 활성 예시에도 나타나지 않는**, 뉴런 기저에서는 사실상 보이지 않는 feature였음

**2) feature가 뉴런보다 훨씬 해석하기 쉬움**

<center><img src="/assets/materials/XAI/sae/sae_04.png" align="center" alt="사람 평가자의 해석 가능성 점수 분포: feature 중앙값 12, 뉴런 중앙값 0" width="650"/></center>

<font size="2"><center> 출처: Bricken et al., Towards Monosemanticity (https://transformer-circuits.pub/2023/monosemantic-features) </center>  </font>

- 블라인드 평가자가 162개 feature/뉴런의 412개 활성 구간을 rubric으로 채점함
- **뉴런 중앙값 0점** (무엇을 나타내는지 가설조차 세울 수 없음) vs **feature 중앙값 12점** (확신 있고 구체적이며 일관된 가설)
- logit 쪽 automated interpretability에서도, 설명만 보고 해당 feature가 올릴 토큰을 맞히는 정확도가 feature 평균 74%, 뉴런 평균 58% (무작위 50%)

**3) 모델 기능의 상당 부분을 설명함**
- MLP 활성값을 SAE 복원값으로 바꿔 끼웠을 때, A/1은 MLP가 제공하는 log-likelihood loss 감소분의 **79%** 를 회복함
- feature를 131,072개로 늘린 A/5는 **94.5%** 까지 회복함
- 다만 저자들 스스로 "MLP를 통째로 0으로 만든 경우" 를 기준으로 삼기 때문에 과대평가일 수 있다고 경고함

**4) Dead / ultralow density feature**
- A/1의 4,096개 중 168개는 1억 개 예시에서 한 번도 켜지지 않은 dead feature, 292개는 100만 번에 한 번도 안 켜지는 ultralow density feature로 분석에서 제외함

**5) Feature splitting과 universality**
- dictionary 크기를 512에서 131,000개 이상으로 늘리면, 작은 dictionary의 feature가 더 세분화된 여러 feature로 **쪼개짐(feature splitting)**. 예를 들어 base64 feature 하나가 큰 dictionary에서는 세 개로 나뉨
- 저자들은 이를 같은 대상을 다른 "해상도" 로 보는 것이라 표현함
- seed가 다른 두 모델에서 학습한 feature들이, 각 모델 자신의 뉴런보다 서로 더 비슷하게 나옴 (universality)
- 고작 512개 뉴런 안에서 SAE를 키울수록 계속 새로운 feature가 발견됨

### Cunningham et al.(2023)의 독립적인 확인

거의 같은 시기에 EleutherAI 쪽 연구자들도 비슷한 결론을 냈습니다.

- Pythia-70M, Pythia-410M의 **residual stream**(각각 512, 1024차원)에 tied-weight SAE를 학습함
- automated interpretability 점수로 비교했을 때, SAE feature가 default basis(뉴런), random direction, PCA, ICA보다 해석 가능성이 높음. 다만 그 격차는 뒤쪽 layer로 갈수록 줄어 layer 4에서는 ICA와 비슷하고, 마지막 layer에서는 개선이 미미했음
- **Indirect Object Identification(IOI)** 과제에서 activation patching을 해 보니, SAE feature는 PCA 성분보다 **더 적은 수만 patch해도** 목표 행동을 재현함. 즉 행동의 원인을 더 촘촘하게 국소화함

1층 장난감 모델이 아니라 여러 층을 가진 실제 언어 모델에서도 통한다는 점에서 의미가 있는 결과입니다.


<a id="scaling"></a>
## 7. Scaling Monosemanticity: Claude 3 Sonnet으로의 확장

Towards Monosemanticity가 나왔을 때 가장 큰 의문은 "이게 상용 규모 모델에서도 될까?" 였습니다. 약 8개월 뒤 나온 Scaling Monosemanticity가 그 답인데요.

### 설정

- 대상: Anthropic의 production 모델 **Claude 3 Sonnet** (모델 크기는 비공개)
- 위치: 모델 **중간 layer의 residual stream**. MLP보다 차원이 작아 SAE 비용이 싸고, layer를 넘나드는 superposition 문제를 줄이는 데 유리하며, 추상적인 feature가 많을 것으로 봤기 때문
- 크기: **1,048,576 (~1M), 4,194,304 (~4M), 33,554,432 (~34M)** feature의 세 가지 SAE
- 세 SAE 모두 토큰당 평균 활성 feature 수는 **300개 미만**, 활성값 분산의 **최소 65%** 를 설명함
- dead feature 비율은 1M에서 약 2%, 4M에서 약 35%, **34M에서 약 65%**. 34M SAE의 살아 있는 feature는 약 1,200만 개 수준임

### Scaling law로 SAE 학습하기

흥미로운 점은 SAE 학습 자체를 LLM처럼 scaling law 문제로 다뤘다는 것입니다. 연산량은 주로 "feature 수 x 학습 step 수" 로 결정되는데, compute-optimal 설정에서 loss가 compute에 대해 대략 **power law** 로 줄어들었고, 최적 feature 수와 학습 step 수도 power law로 늘어났습니다. 34M SAE의 학습 step 수는 이 분석으로 정했다고 하네요.

### 무엇을 찾았나

- **추상적인 feature**: 유명 인물, 국가와 도시, 코드의 type signature 등. 상당수가 **다국어(multilingual)** 이고 텍스트와 이미지 모두에 반응하는 **multimodal** feature였음
- **Golden Gate Bridge feature(34M/31164353)**: 이 feature를 최대 활성값의 10배로 고정하자, 모델이 스스로를 금문교라고 소개하기 시작함
- **코드 오류 feature(1M/1013764)**: Python뿐 아니라 C, Scheme의 버그에도 반응하지만 영어 문장 오타에는 반응하지 않음. 버그 없는 코드에서 이 feature를 강하게 켜면 존재하지 않는 에러 메시지를 지어내고, 버그 있는 코드에서 음수로 고정하면 버그가 없었을 때의 결과를 예측함
- **안전 관련 feature**: 보안 취약 코드, backdoor, 편향, 거짓말과 기만, 권력 추구, 아첨(sycophancy), 위험 콘텐츠 등. 예컨대 sycophantic praise feature(1M/847723)를 5배로 고정하면 과장된 칭찬을 쏟아냄 (맨 위 그림)
- 저자들은 이런 feature가 "존재한다" 는 것과 모델이 실제로 그렇게 행동한다는 것은 다르다며, 과대 해석을 경계함

### Feature는 뉴런과 다른가?

- 1M SAE의 feature 일부를 이전 모든 layer의 뉴런과 비교했더니, **82%의 feature는 가장 상관이 높은 뉴런과도 상관계수가 0.3 이하** 였음
- 즉 SAE feature는 단순히 특정 뉴런을 다시 찾아낸 것이 아님

### Feature splitting과 커버리지

- 1M SAE의 "San Francisco" feature가 4M에서는 2개, 34M에서는 **11개** 의 세부 feature로 쪼개짐
- Claude 3 Sonnet은 런던의 모든 자치구(borough)를 나열할 수 있었지만, 34M SAE에서 feature로 찾을 수 있었던 것은 **약 60%** 에 불과함
- 학습 데이터에서 개념이 등장하는 빈도와, 그 개념 전용 feature가 생기는 데 필요한 dictionary 크기 사이에 체계적인 관계가 관찰됨. 10억 토큰에 한 번 나오는 개념이라면 살아 있는 feature가 10억 개 수준은 되어야 전용 feature를 기대할 수 있다는 뜻이라, 모든 feature를 찾는 일이 얼마나 비싼지 짐작하게 함


<a id="limits"></a>
## 8. 한계와 그 이후의 흐름

### SAE의 한계

**1) 복원 오차(reconstruction error)**
- 34M SAE도 분산의 "최소 65%" 를 설명하는 수준임. 나머지는 SAE가 설명하지 못하는 영역으로 남음
- steering 실험에서도 이 오차 항은 손대지 않고 그대로 더해 줌. 즉 해석은 모델의 **일부** 에 대한 것임

**2) Feature splitting과 "정답 개수" 의 부재**
- dictionary 크기에 따라 같은 개념이 1개가 되기도, 11개가 되기도 함. 어느 해상도가 "진짜" feature인지는 정해져 있지 않음
- Scaling Monosemanticity도 "무엇이 ground truth 목표인지 분명하지 않다" 는 점(Inability to Evaluate)을 한계로 꼽음

**3) Shrinkage와 dead feature**
- L1 penalty는 0이 아닌 활성값까지 체계적으로 작게 추정하는 shrinkage 문제가 있음
- dictionary가 커질수록 dead feature 비율이 크게 늘어남 (34M에서 약 65%)

**4) 비용**
- 저자들은 중간 layer 하나에서도 "모든 feature" 에 한참 못 미친다고 보며, 모든 layer의 feature를 다 찾으려면 **원래 모델을 학습하는 데 든 compute보다 더 많은 compute** 가 필요할 수 있다고 적고 있음
- 여러 층에 feature가 걸쳐 있는 cross-layer superposition도 아직 풀리지 않은 문제임

**5) 활성값을 설명하지, 가중치를 설명하지 않음**
- SAE는 "이 시점에 어떤 개념이 켜져 있는가" 를 알려 주지만, "모델의 가중치가 그 개념을 어떤 계산으로 만들어 내는가" 는 직접 알려 주지 않음
- 게다가 SAE는 원래 모델과 별개로 학습된 **대리 모델** 이므로, SAE가 찾은 feature가 모델이 실제로 쓰는 계산 단위라는 보장(faithfulness)이 없음. steering이나 patching 같은 개입 실험으로 간접 검증할 뿐임

### 그 이후의 흐름 (간단히)

- **Gemma Scope (Lieberum et al., 2024)**: Google DeepMind가 Gemma 2 2B, 9B의 모든 layer와 sub-layer, 그리고 27B의 일부 layer에 대해 학습한 **JumpReLU SAE** 를 공개함. 연구 커뮤니티가 SAE를 직접 학습하는 비용 부담을 덜어 준 셈임
- **Transcoders (Dunefsky et al., 2024)**: 활성값을 복원하는 대신, dense한 MLP 층을 더 넓고 sparse하게 활성화되는 MLP 층으로 근사(입력 -> 출력)하도록 학습해 MLP를 거치는 가중치 기반 circuit 분석을 가능하게 한 방법임. 논문은 sparsity, faithfulness, 해석 가능성 면에서 SAE와 최소 동등한 수준이라고 보고함. Anthropic의 Circuit Tracing(2025)도 cross-layer transcoder를 사용함
- **2025년의 회의론**: Google DeepMind mechanistic interpretability 팀은 2025년 3월 Alignment Forum 글에서, 분포 밖(OOD) 유해 의도 탐지 과제에서 **SAE 기반 probe가 단순한 dense linear probe보다 성능이 낮았다** 는 negative result를 공유하고, **근본적인 SAE 연구의 우선순위를 당분간 낮추겠다** 고 밝힘. 다만 SAE가 쓸모없다는 뜻은 아니며 도구 상자의 하나로 계속 쓸 것이라고 덧붙였고, 데이터셋의 spurious correlation을 찾는 디버깅에는 꽤 유용했다고 함

정리하면 SAE는 "LLM 안을 들여다보는 현미경" 으로서는 분명한 성과를 냈지만, 그것만으로 모델을 완전히 설명하거나 downstream 과제에서 기존 방법을 압도하지는 못했다는 것이 현재까지의 대체적인 평가로 보입니다.


<a id="conclusion"></a>
## 9. 마치며

SAE를 처음 접했을 때 가장 인상적이었던 건, 문제 설정 자체는 고전적인 sparse dictionary learning인데 그걸 LLM의 활성값에 수십억 개 단위로 밀어붙였다는 점이었습니다. 직접 toy 실험을 돌려 보니 $\lambda$ 하나에 따라 "복원은 완벽하지만 해석 불가능한 기저" 와 "해석은 깔끔하지만 정보를 일부 버리는 기저" 사이를 오가는 게 눈에 보여서, 논문들이 왜 그렇게 평가 지표 문제로 고민했는지 조금은 실감이 나더라고요.

또 하나 짚어 두고 싶은 점은, SAE가 아무리 정교해져도 결국 **이미 학습된 모델을 바깥에서 관찰하는 사후 해석** 이라는 사실입니다. 가중치는 그대로 둔 채 활성값만 분해하기 때문에, "모델이 왜 이런 표현을 갖게 되었는지", "가중치가 어떤 계산을 하는지" 에는 한 발짝 떨어져 있습니다. LIME과 SHAP이 입력과 출력 사이를 봤다면, SAE는 그 시선을 모델 내부의 활성값까지 끌고 들어온 것이고, 다음 단계는 자연스럽게 **가중치 그 자체** 가 되겠죠.

다음 글 [가중치 수준의 해석가능성](https://myeonghak.github.io/xai/XAI-%EA%B0%80%EC%A4%91%EC%B9%98-%EC%88%98%EC%A4%80%EC%9D%98-%ED%95%B4%EC%84%9D%EA%B0%80%EB%8A%A5%EC%84%B1/)에서는 circuits 분석, ROME 같은 모델 편집, weight-sparse transformer, 그리고 RASP/Tracr처럼 **가중치를 학습하거나 편집하는 단계에서 해석 가능성을 확보하려는 시도** 들을 살펴보겠습니다. 시리즈 마지막 글 [iWSP와 설명가능성](https://myeonghak.github.io/xai/XAI-iWSP%EC%99%80-%EC%84%A4%EB%AA%85%EA%B0%80%EB%8A%A5%EC%84%B1/)에서는 아예 가중치를 직접 작성하는 접근과 그것이 금융, 의료, 공공 분야에 주는 의미를 다룰 예정입니다.

----------------


**개선을 위한 여러분의 피드백과 제안을 코멘트로 공유해 주세요.**
**내용에 대한 지적, 혹은 질문을 환영합니다.**


**출처**  
[Elhage et al., Toy Models of Superposition (2022)](https://transformer-circuits.pub/2022/toy_model/)  
[Bricken et al., Towards Monosemanticity: Decomposing Language Models With Dictionary Learning (2023)](https://transformer-circuits.pub/2023/monosemantic-features)  
[Cunningham et al., Sparse Autoencoders Find Highly Interpretable Features in Language Models (2023)](https://arxiv.org/abs/2309.08600)  
[Templeton et al., Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet (2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/)  
[Anthropic, Golden Gate Claude (2024)](https://www.anthropic.com/news/golden-gate-claude)  
[Lieberum et al., Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2 (2024)](https://arxiv.org/abs/2408.05147)  
[Dunefsky et al., Transcoders Find Interpretable LLM Feature Circuits (2024)](https://arxiv.org/abs/2406.11944)  
[Anthropic, Circuit Tracing: Revealing Computational Graphs in Language Models (2025)](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)  
[Smith et al., Negative Results for SAEs On Downstream Tasks and Deprioritising SAE Research (GDM Mech Interp Team, 2025)](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks)  
