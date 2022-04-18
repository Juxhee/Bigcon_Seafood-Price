## 2021 빅콘테스트 데이터분석분야 챔피언리그 수산 Biz 

- 오징어, 연어, 흰다리 새우 수입가격 예측을 통한 최적의 가격 예측 모형 도출

</br>



#### 데이터셋
- 제공 데이터 : 2019, 2020, 2021년 동안 수산물의 생산국, 수입국, 수입 가격 등 시계열 데이터
- 추가 수집 데이터 : 2019, 2020, 2021년 동안 오징어, 연어, 흰다리 새우의 서식지 수온 데이터, 해당 거래 중량 데이터

#### 데이터 전처리 
- 명목형 변수 : multi-hot encoding
- 수치형 변수 : 스케일링 작업 진행

</br>

#### 모델링
#### Part1. 새로운 수입 가격 생성

#### 이슈사항
- 하루 동안의 거래 데이터가 여러 개 존재
- 거래 정보(독립변수)를 반영한 하루 동안의 거래 가격 1개를 산출하기 위해 Transformer의 multi-head attention 활용
- 거래 정보를 attention 계산을 거친 뒤 마지막 fully-connected layer로 하나의 값으로 출력하여 이 값을 거래 하나에 대한 가중치로 활용

![image](https://user-images.githubusercontent.com/60679596/147020781-86360397-3e0b-4260-84ce-95b11fedf3d4.png)

![image](https://user-images.githubusercontent.com/60679596/147020826-fa5b92c7-5596-45c7-add5-14b75a833140.png)
</br>

```python
# num_layer만큼 attention 계산하고, 마지막은 fully-connected layer를 통해 차원 축소하여 하나의 값으로 계산하여 이를 거래 하나에 대한 가중치로 활용
class ModelTrunk(keras.Model):
    def __init__(self, name='ModelTrunk', num_heads=2, head_size=128, ff_dim=None, num_layers=10, dropout=0, **kwargs):
      super().__init__(name=name, **kwargs)
      if ff_dim is None:
          ff_dim = head_size
      self.dropout = dropout
      self.attention_layers = [AttentionBlock(num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)]
      self.dense2 = keras.layers.Dense(1, kernel_initializer='he_normal')

    def call(self, inputs):
      x = inputs
      for attention_layer in self.attention_layers:
          x = attention_layer(x)
      x = self.dense2(x)
      return x
```

##
#### Part2. 수산물 가격 예측

#### - 오징어, 연어, 흰다리 새우에 대해 각각 fbProphet과 Neural Prophet 사용하여 RMSE가 낮은 모델 채택

![image](https://user-images.githubusercontent.com/60679596/147020998-22367c04-1115-4de2-80ad-49728fe0fbe8.png)

![image](https://user-images.githubusercontent.com/60679596/147021019-7b25d831-c0e9-4d43-a970-3d664a7e018f.png)

## 

#### Part3. 모델링 결과 


![image](https://user-images.githubusercontent.com/60679596/147021356-3449e8a0-3ef9-4f36-b6ce-99a23e2cc76c.png)


