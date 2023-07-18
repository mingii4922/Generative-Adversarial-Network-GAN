# Conditional GAN

### 3D conditional generative adversarial networks for high-quality PET image estimation at low dose

---
### Abstract

- GAN은 생성모델을 적대적 학습을 통해 훈련시키는 방법으로 연구됨.
- 본 논문은 간단하게 생성자와 판별자에 조건(condition)을 사용하기 위한 data y를 입력으로 추가하는 방식을 제안
- GAN 모델이 어떻게 multi-modal 모델로서의 학습을 할 수 있는지 이해하는 것이 핵심
- 결과적으로 conditional data를 입력에 추가하면서 생성하는 과정을 제어하고자 노력함

---
### Keyword

- Markov chain
- Multi-modal
- Modality-Biased: multi-modality 모델에서 하나 이상의 modality가 다른 modality에 비해 중요한 역할**(편향)**을 하는것
  - ex) 이미지 주석


----
#### Markov chain

- Markov property: 과거 상태들과 현재 상태가 주어졌을 때, 미래의 상태는 과거 상태들과 독립적이며 현재 상태 정보에 의해서만 결정된다는 성질
- Discrete time: 시간이 연속적으로 변하는 것이 아닌 이산적으로 변하는 시간을 의미
- Stochastic process: 시간에 따라 어떤 사건이 발생할 확률이 변화하는 과정

> - 0차 Markov chain: 동전 던지기와 같이 n번째 상태가 n+1번째 상태에 영향을 주지 않음 **(독립)**
- 1차 Markov chain: n번째 상태가 n+1번째 상태를 결정할 때 영향을 미침
- 2차 Markov chain: n-1, n번째 상태가 n+1번째 상태를 결정할 때 영향을 미침

---
#### Multi-modal 접근 연구

- Devise: A deep visual-semantic embedding model, 2013, NIPS
  - image와 text 간의 semantic 정보를 잘 반영하는 이미지 검색 시스템을 구현하기 위한 딥러닝 모델 제안
- Multimodal Learning with Deep Boltzmann Machines, 2012, NIPS
  - Deep Boltzmann Machines(DBM)을 활용해 멀티모달 학습에 대한 연구 진행한 논문

<img src='https://github.com/mingii4922/Generative-Adversarial-Network-GAN/assets/79297596/a20a7977-94c4-4263-83c5-d931bd08e8e5' width=800></center>

단순하게 이미지에 대한 hidden state feature와 테스트에 대한 hidden state feature를 joint representation 하는 방식


---

### Conditional probability


<img src='https://github.com/mingii4922/Generative-Adversarial-Network-GAN/assets/79297596/dbf856a7-9dc5-4b21-9b48-c2ad36da1cbd' width=800></center>

- random variable(확률 변수): ex) 주사위를 던졌을 때 특정 case가 발생할 때 그 변수: 1,2,3,4,5,6
  - 주사위 게임과 같은 상황은 **이산확률 변수**, 범위 내의 모든 실수 값이 확률에 포함되면 **연속확률 변수**
- probability mass function: 확률 변수에 의해서 정의된 실수를 확률 (0~1) 사이에 대응 시키는 함수를 말함

<img src='https://github.com/mingii4922/Generative-Adversarial-Network-GAN/assets/79297596/ff395df0-bf9b-471f-aa33-cb6bc62d24dd' width=300></center>

- Conditional probability: 어떤 사건이 일어나는 경우 다른 사건이 일어날 확률(조건부 확률)

$$P(B|A) = \frac{P(A \cap B)}{P(A)}$$


---
### Conditional Adversarial Nets

- GAN 모델은 생성자와 판별자에 추가적인 condition data y를 부여함으로써 conditional model로 확장할 수 있음
- conditional data y는 형태가 정해져있지 않으며, class label 이나 다른 modality로 적용될 수 있음
- 생성자는 noise $P_{z}(z)$와 conditional data y가 공동 결합으로 표현됨
- 판별자는 x와 y를 이용하여 real과 fake를 판별함

#### Distribution 관점

<img src='https://github.com/mingii4922/Generative-Adversarial-Network-GAN/assets/79297596/391c3fd6-acc8-48ca-9f88-2830438d6982' width=600></center>

- 실제 데이터 분포(x축) x가 있을 때, 데이터의 특징에 따라 분포를 구할 수 있고, 우리는 금발머리를 가진 여성$x_3$의 density를 구할 수 있음
  - 추가적으로 실제 가지고 있는 데이터에서 이상한 이미지들을 $x_4$에 있다고 가정한다면? 전체 데이터 셋에서 적은 양의 이미지를 가지고 있다고 해석할 수 있음
- 결과적으로 어떠한 특징(conditional 정보 y)가 있다면 데이터 분포에서 우리가 원하는 이미지를 찾을 수 있을 것임



#### Loss

<img src='https://github.com/mingii4922/Generative-Adversarial-Network-GAN/assets/79297596/3b8f1901-7c3c-4a88-8b72-7c993e44c34b' width=800></center>

- $P(x|y)$: conditional data y를 조건으로 주었을 때 data x 가 생성될 확률
- 생성자(G) 입장에서는 뒤의 수식만을 고려하며, 조건 y를 주었을 때 z를 가지고 y의 조건에 맞게 이미지를 생성함
- 판별자(D) 입장에서는 조건 y를 가지고 data x 가 G에 의해서 만들어졌는지 실제 이미지인지 파악)

#### Architecture

<img src='https://github.com/mingii4922/Generative-Adversarial-Network-GAN/assets/79297596/fffff44f-521a-4f83-8adb-8225b4eb994e' width=600></center>

- noise z로부터 얻은 feature와 conditional data y를 통해 얻은 feature를 가지고 generator가 이미지를 생성
  - ex) MNIST (data 1)을 cGAN으로 생성할 때
    - noise z = $D^{100 \times 1}$ -> $D^{200 \times 1}$
    - conditional data y = $D^{10 \times 1}$ -> $D^{1000 \times 1}$
    : 0~9까지 10개의 class 원핫인코딩 (0,1,0,0,0,0,0,0,0,0)
    - z feature + y feature = $D^{1200 \times 1}$ -> Generator -> data 1 이미지 생성
    
---

### Conclusion

- 2014년에 나온 conditional network의 초기단계에 가깝지만 CGAN의 잠재력을 증명했고 유용한 방안들이 이후 나오게 됨
- 향후 다양한 멀티모달(태그)를 동시에 사용해서 더 우수한 성능을 낼 수 있을 것이라 예상함
