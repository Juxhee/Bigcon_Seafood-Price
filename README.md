# Bigcon_Seafood-Price

## 오징어, 연어, 흰다리 새우 수입가격 예측을 통한 최적의 가격 예측 모형 도출


### Part1. 새로운 수입 가격 생성
##
#### 이슈사항
- 하루 동안의 거래 데이터가 여러 개 존재
- 거래 정보(독립변수)를 반영한 하루 동안의 거래 가격 1개를 산출하기 위해 Transformer의 multi-head attention 활용

![image](https://user-images.githubusercontent.com/60679596/147020781-86360397-3e0b-4260-84ce-95b11fedf3d4.png)

![image](https://user-images.githubusercontent.com/60679596/147020826-fa5b92c7-5596-45c7-add5-14b75a833140.png)


##
### Part2. 수산물 가격 예측
##
#### - fbProphet과 Neural Prophet 사용하여 RMSE가 낮은 모델 채택

![image](https://user-images.githubusercontent.com/60679596/147020998-22367c04-1115-4de2-80ad-49728fe0fbe8.png)

![image](https://user-images.githubusercontent.com/60679596/147021019-7b25d831-c0e9-4d43-a970-3d664a7e018f.png)

## 

### Part3. 모델링 결과 
##

![image](https://user-images.githubusercontent.com/60679596/147021356-3449e8a0-3ef9-4f36-b6ce-99a23e2cc76c.png)


