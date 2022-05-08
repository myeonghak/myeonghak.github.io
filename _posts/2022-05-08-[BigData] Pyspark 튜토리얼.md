---
title: "[BigData] (작성중) Pyspark 튜토리얼"
categories:
  - Big Data
tags:
  - Pyspark
  - data preprocessing
  - distributed computing
---

### 인-메모리 기반 분산 처리로 대용량 데이터를 빠르게 조작/분석할 수 있는 툴인 Pyspark에 대해 알아봅니다.  


> 2억 줄짜리 테이블을, Pandas로 언제 분석해!  

<center><img src="/assets/materials/BigData/pyspark_tutorial/pyspark_01.png" align="center" alt="drawing" width="500"/></center>   




<br/>


----

#### Contents  

<br/>

1.  [들어가며 - Pyspark과 친해지길 바라](#intro)  
2.  [Pyspark 개요 - 따분하지만 중요한 밑그림](#summary)  
3.  [Pyspark 전처리 1 - 간단한 DF 조작과 결측치 처리](#prepro1)
4.  [Pyspark 전처리 2 - Groupby 함수](#prepro2)
5.  [Pyspark MLlib으로 머신러닝 모델 학습하기](#ml)

<br />



<a id="intro"></a>
## 1. 들어가며 - Pyspark과 친해지길 바라  

> 입사 전: Spark? 그걸 왜 쓰지 ㅋㅋ 그냥 Pandas로 뚝딱뚝딱 하면 되는데!  
> 입사 후: 아니 로우가 2억개..? 한번 돌리는데 2시간..? 연산 끝나니 퇴근시간..?  

현재 재직 중인 카드사에서는 다양하고 방대한 결제 정보 데이터가 축적되고 있습니다. 무슨 이런 데이터가 죄다 있지..? 싶을 정도로, 예전에는 생각하기 어려운 양의 데이터를 매일 만지고 있는데요.  
입사 후에 가장 어려운 점이 이 대용량 데이터를 전처리하고 조작하는 작업을 위해 Pyspark를 배워야한다는 점이었습니다.  
Pandas에 익숙해진 저와 동료들은 레거시 코드를 보면서 한참을 고통받았더랬죠. 이건 왜 이렇게 짜여 있는거지..? 이건 뭐하는 함수지? 그때마다 느꼈던 건.. Pandas는 정말로 친절한 친구였습니다.  

하지만 조금씩 Pyspark이라는 친구를 알아가면서, 이 친구와 친해지면 얼마나 편해질 수 있는지 피부로 와닿기 시작했습니다. Pandas로 한참이 걸리던 연산이 Pyspark로 뿅! 하고 해결되는 것을 보고 너.. 생각보다 좋은 아이였구나? 하는 긍정적인 감정이 쌓이다 보니 이 친구를 더 알아가고 싶더라구요.  

오늘은 저처럼 Pandas에 익숙한 분석가가 Pyspark를 쉽게 이해하고 다룰 수 있도록 설명하는 포스트를 준비해 보았습니다.  



----------------


**개선을 위한 여러분의 피드백과 제안을 코멘트로 공유해 주세요.**
**내용에 대한 지적, 혹은 질문을 환영합니다.**  


**출처**  
