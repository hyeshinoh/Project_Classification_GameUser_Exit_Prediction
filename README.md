#### 분류 프로젝트
# 엔씨소프트 ‹블레이드 앤 소울› 게임 유저 이탈 예측

## 1. Introduction
게임 ‘블레이드 앤 소울’ 유저 이탈 예측 모형
- 엔씨소프트의 MMORPG 게임 ‘블레이드앤소울’ 유저의 활동데이터를 활용하여 어떤 유저가 향후 게임서비스에서 이탈하는지, 언제 이탈하는지 예측하는 모델 개발

2018 빅콘테스트 챔피언리그 문제
- 대회 데이터 공개: 7/18
- 제출 마감: 9/14

의의
- 고객 이탈은 다양한 도메인의 CRM에서 중요하게 다루는 문제
- 게임 데이터: 현실 사회와 유사한 가상 세계에서의 인간 행동에 대한 고품질 데이터가 제공되는 context 

### 1.1 문제 및 평가 기준
- 문제: 게임 유저의 향후 이탈 여부 및 시점 예측
	- ‘이탈’의 정의: 4주 이상 접속하지 않으면 이탈로 판단
	- 제공 데이터 시점 이후 12주 동안의 접속 이력으로 판단
- target label: 이탈 여부, 이탈 시점에 따른 4개 클래스
	- week: 1주일 내에 이탈
	- month: 2-4주 내에 이탈
	- 2month: 5-8주 내에 이탈
	- retained: 잔존
- 평가 기준: 예측 성능(F1 score)
	- 각 클래스별 precision과 recall을 계산한 후 전체에 대한 조화 평균

### 1.2 데이터 소개
데이터 규모
- Train data: (계정 아이디 기준) 10만 명의 게임 활동 데이터
- Test data: (계정 아이디 기준) 4만 명의 게임 활동 데이터

데이터 구성
- 주요 활동 정보 “activity”: 게임 내 주요 활동량을 유저별 1주일 단위로 집계
- 결제 정보 “payment”: 사용자의 결제 정보를 1주일 단위로 집계
- 사회관계 정보 “party”: 유저간 상호작용 및 사회관계에 대한 정보 (예측 대상이 아닌 유저 포함)
- 사회관계 정보 “guild”: 유저간 상호작용 및 사회관계에 대한 정보 (예측 대상이 아닌 유저 포함)
- 사회관계 정보 “trade”: 유저간 상호작용 및 사회관계에 대한 정보 (예측 대상이 아닌 유저 포함)
  
## 2. EDA & Feature Engineering
Overview
- 각각 다른 schema를 가진 data에서 예측 대상인 유저id(acc_id) 기준으로 하여 feature variable 생성
- Activity, payment data의 경우
	- 한 유저가 week 별로 여러 개의 관측치를 가지고 있음
	- 이를 변수마다 week별 변수로 확장(w1~w8)하거나 groupby하여 변수 생성
- Party, guild와 trade data의 경우
	- party와 guild의 경우 개인 유저 레벨이 아닌 그룹(party, guild) 레벨의 data
	- 전체 사회관계를 담기 위해 train id에 대해 sampling되어 있지 않음
	- party 멤버 id와 guild 멤버id에서 개별id를 추출하여 참여 횟수 등의 변수 생성 
	- trade의 경우 전체 trade 리스트 중 train id가 구매/판매한 데이터만 이용해 변수 생성
	- party와 trade 전체 데이터를 network로 분석하여 변수 생성

현재 feature variables 총 536개
- [변수 설명](https://docs.google.com/spreadsheets/d/1mm9PTYYPBvEwT4YOv-zK9bUCs2nIcGCh_BwxN9m97Iw/edit?usp=sharing)
- Modeling에서는 feature간 상관관계 등을 고려하여 선택 사용


## 3. Modeling
1. 앙상블 기법 중 하나인 stacking을 사용 
	- 1단계: random forest, XGBoost 등 사용
  	- 2단계: XGBoost
2. Keras 단일 모델 ([jupyter notebook](https://colab.research.google.com/drive/1DLjjuAU87UHCXkUEDoBE12SuTrRRXCTi))

## 4. Performance
- 현재 중간 평가 결과: stacking 사용 모델의 f-1 score 최고점수 0.7264

## 5. Conclusion
4개 클래스(Week-Month-2Month-Retained)별로 예측 성능에 차이 발생
- Week과 Retained은 상대적으로 잘 예측됨
- Month와 2Month는 서로 구분하기가 쉽지 않음
- Month의 recall과 2Month의 precision이 떨어지는 경향이 있음
	- Month로 분류해야하는 유저를 2Month로 분류하고 있다는 의미

Week과 Retained의 분류 성능이 높다는 점에 착안하여 한 label씩 분류하면서 단계적으로 진행하는 hierarchical한 접근법을 시도하였으나 최종 성능은 크게 효과적이지 않았음
- Week과 Retained 단계에서 정확도가 완벽할 정도로 굉장히  높은 경우에만 기대만큼 효과적일 것이라는 점을 알게됨
