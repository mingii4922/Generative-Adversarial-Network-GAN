# Generative adversarial networks

## Generative adversarial networks, ACM, 2020


GAN은 2014년, Ian Goodfellow의 "Generative Adversarial Network"라는 논문에서 처음 제시됨
적대적인 과정을 통해 생성 모델을 추정하기 위한 새로운 프레임워크를 제안
- generative model(생성자, G): 학습 데이터의 분포를 capture
- discriminator model(판별자, D): 학습 데이터인지 생성된 데이터인지 판별
  * 두 명의 플레이어가 min-max game을 하는 것과 같다고 생각하면 됨
  
<img src="https://github.com/mingii4922/Generative-Adversarial-Network-GAN-/assets/79297596/4140a581-34a6-4f91-9240-cfa2174bb012" height=200 width=800></center>

---
### Introduction

<img src='https://github.com/mingii4922/Generative-Adversarial-Network-GAN-/assets/79297596/2e5da048-1201-4e72-a515-9d14f7b1b208' height=200 width=800></center>

- 딥러닝은 discriminative model(고차원, 풍부한 feature의 입력을 class label에 mapping)에서 두드러진 성공을 보여줌
- backpropagation, dropout 등의 다양한 알고리즘과 각 layer의 gradient를 network 전체에 잘 전달할 수 있게 도와주는 activation function을 기반으로 딥러닝은 우수한 성능을 보여줌
---
- 하지만 MLE(Maximum likehood estimation)과 같은 전략에서 나오는 확률론적인 계산의 어려움으로 인해 generative model은 intractable problem이 존재
  * intractable problem: 보통 $O(N^2)$이상의 time complexity를 가지면 intractable(다루기 어렵다)하다고 말함


---
### Background
