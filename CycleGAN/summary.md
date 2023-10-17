# CycleGAN

### Unpaired image-to-image translation using cycle-consistent adversarial networks CVPR, 2017

---
### Introduction

<img src='https://github.com/mingii4922/Generative-Adversarial-Network-GAN/assets/79297596/318d050a-7b3d-43af-94cc-33cecd3cc7a2' width=700></center>

- 데이터셋을 구성할 때 구조적으로 동일한 이미지 2개가 쌍으로 맺어진 데이터셋을 구하기는 쉽지 않음.
  - 위에 그림과 같이 paired 데이터 보다 unpaired 데이터를 구하기 훨씬 수월함
  - 이는 이전에 Pix2Pix 논문(쌍으로 된 데이터 셋에서 작동하는 GAN기반 image 생성)에서 나온 방법의 한계와 동일

- 본 논문에서는 pair-image를 사용하지 않고 단지 X domain과 Y domain 데이터를 활용해 두 domain 간의 이미지를 변환하는 법을 학습함

- CycleGAN의 contribution
> - 쌍이 아닌(unpaired) 데이터로부터 도메인 간 이미지 변환 학습
> - Cycle-consistent한 적대적 네트워크 구성
---
### Keywords

- paired dataset: 정확히는 도메인이 다른 두 이미지에 대해서 같은 좌표의 pixel이 서로 일치해야함**(동일한 위치를 나타내야함)**
- Style transfer: pre-trained model의 feature map을 활용해서 gram matric 통계를 일치시켜 한 이미지의 content와 다른 이미지의 style을 합성하는 방법
  - Gram matric: 특정 layer에 대해 feature map의 channel 간 상관 관계의 값을 찾는 행렬

---
### Formulation

<img src='https://github.com/mingii4922/Generative-Adversarial-Network-GAN/assets/79297596/dc70a416-2297-46af-af27-a0af1c7e07cd' width=700></center>

> $D_X$: F를 통해 생성된 이미지 $F(Y)$와 실제 이미지 $(X)$를 구분하는 판별자(discriminator)
$D_Y$: G를 통해 생성된 이미지 $G(X)$와 실제 이미지 $(Y)$를 구분하는 판별자(discriminator)
$X$: $X$ domain의 이미지 collection
$Y$: $Y$ domain의 이미지 collection
$G$: $X$ domain의 이미지를 통해 $Y$ domain의 이미지 생성 $(G(X)=Y)$하는 생성자(generator)
$F$: $Y$ domain의 이미지를 통해 $X$ domain의 이미지 생성 $(F(Y)=X)$하는 생성자(generator)

- 총 2개의 generator와 2개의 discriminator가 존재함

#### Loss function-1

<img src='https://github.com/mingii4922/Generative-Adversarial-Network-GAN/assets/79297596/c6dc6352-e6b9-43e7-88fa-78524e82f2db' width=700></center>

- Discriminator 학습: 입력된 이미지가 $Y$ domain에 존재하는지? -> $logD_Y(y)$
  - 0 or 1의 값을 가짐 (이진 분류)
  - 결과적으로 $D_Y(y)$가 항상 1이 되도록 학습을 진행함: 실제 이미지 $y$를 넣으면 discriminator가 항상 1로 판별
  
- Generator 학습: $D_Y(G(x))$ 기준으로 설명
  - $G(x)$: $G$를 통해 $X$ domain의 이미지 x가 $Y$ domain으로 변환한 것을 의미
    - $G$가 잘 학습되었다면? $D_Y(G(x))$의 결과는 1 -> 잘 생성되어서 판별자를 속임
- 논문에서는 adversarial loss를 통해 실험해보았지만, 

#### Full objective function
<img src='https://github.com/mingii4922/Generative-Adversarial-Network-GAN/assets/79297596/6202180b-26c4-4e8a-a907-7958ae269823' width=700></center>
- Cycle-consistency loss: $G$에 의해 생성된 $G(x)$가 $F$에 의해 다시 원래 입력 $x$로 돌아갈 수 있도록 이미지를 변형시키는 network를 만드는 것이 목적 -> $F(G(x)) = x$

---
### Conclusion

- 기존 pix-to-pix에서 발전하여 unpaired 데이터셋에 대해 매우 효과적으로 작동함
- 본 논문에 포함된 모든 objective function은 기능상 필요함(cycle-consistency-loss)
- 도메인 간 스타일 변환에 특화되어 있기 때문에 형태가 변환되어야 하는 주제에는 낮은 성능을 보임

---
### Limitations
<img src='https://github.com/mingii4922/Generative-Adversarial-Network-GAN/assets/79297596/b6ec0b99-6810-4b30-8bca-b3319bdacc9d' width=700></center>

- 색상과 질감을 변환하는데는 효과적이지만, 기하학적인 변환이 불가능하기 때문에 dog-cat으로 변형하는 경우 생성이미지를 원하는대로 만들지 못함
- Dataset에 없는 분포를 가진 이미지를 가지고 생성하게 된다면, 성능이 하락함 (사람-얼룩말)
- 여전이 Paired dataset과의 결과 차이가 존재함
- Object의 경계를 인식하지 못하며, 사람의 객체에 얼룩말 형태의 무늬를 입히는 것을 발견
