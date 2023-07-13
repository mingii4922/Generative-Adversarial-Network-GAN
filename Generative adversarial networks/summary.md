# Generative adversarial networks

## Generative adversarial networks, ACM, 2020


GAN은 2014년, Ian Goodfellow의 "Generative Adversarial Network"라는 논문에서 처음 제시됨
적대적인 과정을 통해 생성 모델을 추정하기 위한 새로운 프레임워크를 제안
- generative model(생성자, G): 학습 데이터의 분포를 capture
- discriminator model(판별자, D): 학습 데이터인지 생성된 데이터인지 판별
  * 두 명의 플레이어가 min-max game을 하는 것과 같다고 생각하면 됨

---
### Introduction

<img src='https://github.com/mingii4922/Generative-Adversarial-Network-GAN-/assets/79297596/2e5da048-1201-4e72-a515-9d14f7b1b208' height=300 width=800></center>

- 딥러닝은 discriminative model(고차원, 풍부한 feature의 입력을 class label에 mapping)에서 두드러진 성공을 보여줌
- backpropagation, dropout 등의 다양한 알고리즘과 각 layer의 gradient를 network 전체에 잘 전달할 수 있게 도와주는 activation function을 기반으로 딥러닝은 우수한 성능을 보여줌

> 하지만 MLE(Maximum likehood estimation)과 같은 전략에서 나오는 확률론적인 계산의 어려움으로 인해 generative model은 intractable problem이 존재
  * intractable problem: 보통 $O(N^2)$이상의 time complexity를 가지면 intractable(다루기 어렵다)하다고 말함

---
<img src='https://github.com/mingii4922/Generative-Adversarial-Network-GAN-/assets/79297596/b74e3420-cb46-4f15-9b2c-3229e5300622' height=300 width=800></center>

- generative model은 샘플링 된 데이터가 어느 분포에서 나왔는지 추정하는 것이 목적
  * MLE는 확률 밀도 함수를 모델링하는 방법 중 하나
- 결국 실제 데이터(**빨간 점**)들의 확률 분포(**초록 분포**)를 알고 있을 때, generative model(**검정 분포**)로 실제 데이터의 분포를 근사하는 방향으로 학습

---
<img src="https://github.com/mingii4922/Generative-Adversarial-Network-GAN-/assets/79297596/4140a581-34a6-4f91-9240-cfa2174bb012" height=300 width=800></center>


- discriminative model은 실제 데이터와 생성된 데이터를 분류
- discriminative model은 경찰, generative model은 위조지폐를 만드는 사람에 비유하면 쉬움
  - 이런 게임으로 인한 두 model간의 경쟁은 위조 데이터와 진짜 데이터를 구분이 불가능한 방향으로 학습하게 지도가능

<img src="https://github.com/mingii4922/Generative-Adversarial-Network-GAN-/assets/79297596/9a3ad842-c0d9-4ce2-8976-e437408e5fe0" height=100 width=800></center>

|수식|설명|
|---|---|
|$x$|$P_{data}(x)$부터 샘플링한 학습 데이터|
|$z$|$P_z(z)로부터 샘플링한 데이터(noise)|
|$P_{data}(x)$|실제 데이터의 분포|
|$P_z(z)$|노이즈의 분포|
|$G(z)$| generative model: 노이즈 $z$로 생성한 데이터|
|$D(x)$| discriminator model: 입력 데이터 x가 학습 데이터일 확률 [0,1]|

---

<img src="https://github.com/mingii4922/Generative-Adversarial-Network-GAN-/assets/79297596/104d1e11-5fa9-4d43-85d5-26262ca98581" height=100 width=350></center>

> - Generator: 
if $D(G(z)) == 1:$
  - generator는 loss가 최소화 되는 방향으로 학습을 진행(데이터를 잘 생성하도록)
  - noise $z$로부터 생성한 데이터 $G(z)$가 discriminator를 완벽하게 속임
  - (discriminator는 생성된 데이터가 실제 데이터라고 100% 확신)
    - $log(1-D(G(z))) = log(1-1) = log(0) = - \infty$ 

---

<img src="https://github.com/mingii4922/Generative-Adversarial-Network-GAN-/assets/79297596/6b8e89bd-f55d-4128-8a44-c3f1fe1b53fc" height=100 width=600></center>

> - Discriminator: 
if $D(G(z)) == 0, D(x) == 1:$
  - noise $z$로부터 생성한 데이터 $G(z)$와 실제 데이터 $x$를 discriminator가 완벽하게 구별 가능
  - (discriminator는 생성된 데이터가 가짜 데이터라고 100% 확신)
  - $log(D(x)) + log(1-D(G(z))) = log(1) + log(1-0) = 2log(1)$

---
### Architecture


<img src="https://github.com/mingii4922/Generative-Adversarial-Network-GAN/assets/79297596/111b8e33-6440-4c49-aaec-f2aaf14f0a4e" height=200 width=700></center>

- 실제 동작과정은 다음과 같음
  * Generator의 input으로 latent vector $z$를 넣어 가짜 이미지를 생성하고, 실제 이미지와 generator로 생성된 가짜 이미지를 discriminator가 비교(판별)하는 구조
  * discriminator를 통해 추출된 결과(loss)를 가지고 generator와 discriminator가 역전파를 통해 network 학습

- generative model과 discriminative model을 모두 충분히 학습하고 나면 $p_{data} = p_(g)$가 되는 지점에 도달 (global optimality of $p_g = p_{data}$ )
  * discriminative model은 실제 데이터와 가짜 데이터를 구분할 수 없게 됨 ($D(x) = \frac{1}{2}$)

---

### Adversarial network

* 매 iter에서 discriminator를 최적화 시키고 generator를 업데이트 시키는 것은 불가능하며, 유한한 크기의 데이터 셋 x를 가졌을 대 discriminator가 overfitting 될 수 있음
  * 따라서 k번 만큼 discriminator를 업데이트하고, 한번의 generator를 업데이트하는 방식으로 진행

<img src="https://github.com/mingii4922/Generative-Adversarial-Network-GAN/assets/79297596/2ad4fd78-ae95-4635-b532-a4fba38c4015" height=300 width=700></center>


> 1. 학습 초기: generator는 실제 데이터와 큰 차이를 보이기 때문에 discriminator가 높은 확률로 정답을 맞춤(빠르게 학습됨)
   * **(위 그림)**: $log(1-D(G(z)))$를 minimize 하는 수식을 $log(D(G(Z)))$를 maximize 하는 방식으로 변경

---

#### 장점
- Markov chain이 필요하지 않고 역전파만 사용하여 학습할 수 있음
- 학습 중에 추론이 필요하지 않고 미분 가능한 다양한 함수를 모델에 사용가능
#### 단점
- $p_{g}(x)$를 명시적으로 표현할 수 없음
- 학습 중 discriminator가 generator와 잘 동기화되어야 함
- generator가 충분한 다양성을 갖기 위해 많은 $z$ 값을 동일한 $x$ 값으로 축소하는 model collapse 현상이 발생할 수 있음
  * 따라서 discriminator를 업데이트하지 않고 generator를 너무 많이 훈련하는 것은 안됨
