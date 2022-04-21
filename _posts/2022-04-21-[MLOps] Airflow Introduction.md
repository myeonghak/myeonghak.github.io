---
title: "[MLOps] Airflow Introduction"
categories:
  - MLOps
tags:
  - airflow
---



> Workflow & Data Pipeline 관리 플랫폼인 Airflow를 소개합니다.

<center><img src="/assets/materials/recsys/uber_eats/logo.png" align="center" alt="drawing" width="500"/></center>   


>  **Airflow란, 워크플로우와 데이터 파이프라인을 프로그램적으로 인증, 스케줄링, 모니터링하는 플랫폼**

<br/>

----

**본 포스트는 다음 영상을 토대로 작성한 노트입니다.**
[영상은 여기에서 보실 수 있습니다.](https://youtu.be/AHMm1wfGuHE)  



<br/>  


<br/>

1. Airflow란?  
	- 워크플로우와 데이터 파이프라인을 프로그램적으로 인증, 스케줄링, 모니터링하는 플랫폼.  

<br/>

2. 워크플로우란?  
	- 일련의 작업  
	- 스케줄에 의해 시작되거나 특정 이벤트에 의해 발생됨(trigger)  
	- 흔히 빅데이터 처리 파이프라인에서 사용됨  

<br/>

3. 전형적인 워크플로우  
	- 원천으로부터 데이터 다운로드  
	- (프로세스 내 특정 위치로) 데이터 전송  
	- 프로세스 종료를 모니터링  
	- 결과를 받고, 레포트를 생성  
	- 생성된 레포트를 이메일로 전송  

<br/>

4. 전통적인 ETL 프로세스  
	- DB에서 데이터를 끌어와 HDFS로 보내 데이터를 처리함. 이러한 과정을 스크립트로 처리  
	- HDFS: Hadoop Distributed File System  
	- 이 스크립트를 cronjob으로 스케줄링함  

<br/>

5. 전통적인 프로세스의 문제점  
	- 실패: 실패하면 다시 살려야함  
	- 모니터링: 성공했나? 그럼 언제 끝났어?  
	- 의존성: 먼저 끝나야할 작업이 먼저 안끝나면?  
	- 확장성: 두 다른 머신을 돌려야할 경우엔 어떡해?  
	- 전개: 새로운 버전을 업데이트 계속 해야되면?  
	- 과거 데이터 처리: 과거 데이터를 복구/재처리 해야하면?  

6. 아파치 에어플로우  
	- Airbnb에 의해 개발된 워크플로우 관리 시스템  
	- 태스크를 정의하고 관리하는 파이썬기반 프레임워크  
	- 워크 노드에 걸쳐 실행, 스케줄링, 배포  
	- 현재와 과거 실행을 조회하고 로그 관리  
	- 다양한 플러그인을 관리 가능  
	- REST API로 정의 가능  
	- DB와 잘 연동함  

7. Airflow DAG  
	- DAG란 Directed Acyclic Graph를 의미하며, 독립적으로 실행 가능한 다중의 태스크로 구성됨  
	- DAG은 task로 구성됨  


----------------


**개선을 위한 여러분의 피드백과 제안을 코멘트로 공유해 주세요.**
**내용에 대한 지적, 혹은 질문을 환영합니다.**  


**출처**  
https://youtu.be/AHMm1wfGuHE
