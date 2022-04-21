---
title: "[Deep Learning] 차원 축소 - PCA와 AutoEncoder"
categories:
  - Deep Learning
tags:
  - PCA
  - AutoEncoder
  - Dimensionality Reduction
---



> 고차원 데이터를 차원축소 할 때는 AutoEncoder보다 PCA가 유리할 수 있음

<center><img src="/assets/materials/DeepLearning/pca_ae/PCA.png" align="center" alt="drawing" width="500"/></center>   




<br/>

----




> (...) Highly complex data with perhaps thousands of dimensions the autoencoder has a better chance of unpacking the structure and storing it in the hidden nodes by finding hidden features.  
> (...) What is interesting is the autoencoder is better at reconstructing the original data set than PCA when k is small, however the error converges as k increases  


- K가 작을 때 AE는 reconstruction error가 적었으나, 5를 넘어가자 error가 늘어났음.  

- 고차원일때는 PCA가 더 유리할 수 있다  

- PCA와 AE의 차이  
	1) AE에서, bottleneck layer의 사이즈를 결정할 가이드라인은 없다. PCA로는 top k개의 컴포넌트를 사용할 수 있음. PCA를 써서 k를 결정할수도 있을 것임.  
	2) AE는 k가 작을 때 PCA보다 잘 작동하는 경향이 있음. 즉 비슷한 정확도가 더 적은 컴포넌트로 성취될 수 있으며 따라서 더 작은 데이터셋으로 가능하다는 것. 이는 아주 큰(아마 feature의 수 기준으로) 데이터셋을 다룰 때 중요함.  
	3) PCA 결과를 시각화할 때, 2 혹은 3 컴포넌트가 주로 사용됨. 다른 컴포넌트는 볼 수 없다는 것이 단점임. 다른 컴포넌트를 보려면 따로 시각화해야해 오래걸림.
		AE는 전체 데이터를 2 or 3차원으로 축소시킴으로써 전체 정보를 포함할 수 있고, 이는 시간을 절약할 수 있음.  
	4) AE는 PCA보다 연산이 많음. 그럼에도, 아주 큰 데이터 셋에서는 메모리에 저장할 수 없어서 PCA가 사용될 수 없음 (아마 행렬연산이 필요하기 때문)
		AE 구축이 메모리 한계를 해결하며 쉽게 배치화 될 수 있을 것임.   



<br/>



#### [Appendix] 타이틀 plotly 코드  

타이틀 플롯에 사용한, plotly로 간단한 interactive 3d 시각화를 보여주는 예제코드입니다.  


<br/>

```python
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

import plotly.offline as pyo
# import plotly.graph_objs as go
# Set notebook mode to work in offline
pyo.init_notebook_mode()

df = px.data.iris()
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

pca = PCA(n_components=3)
components = pca.fit_transform(X)

components_df=pd.DataFrame(components)

components_df["species"]=df["species"]

total_var = pca.explained_variance_ratio_.sum() * 100

fig = px.scatter_3d(
    data_frame=components_df, x=0, y=1, z=2, color='species',
    title=f'Total Explained Variance: {total_var:.2f}%',
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}   
)
fig.update_traces(marker_size = 2)
fig.show()
```




----------------


**개선을 위한 여러분의 피드백과 제안을 코멘트로 공유해 주세요.**
**내용에 대한 지적, 혹은 질문을 환영합니다.**  


**출처**  
https://www.r-bloggers.com/2018/07/pca-vs-autoencoders-for-dimensionality-reduction/  
