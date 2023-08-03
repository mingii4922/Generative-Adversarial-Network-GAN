# Pix2Pix

### Image-to-image translation with conditional adversarial networks, 2017, CVPR

<img src='https://github.com/mingii4922/Generative-Adversarial-Network-GAN/assets/79297596/9e5a0135-5cfe-4ad2-8fd5-2308aa2e2972' width=900></center>

  
----
### Abstract

본 논문은 Conditional Generative Adversarial Network(CGAN)을 사용하여 한 이미지를 다른 domain(style)의 이미지로 변환하는 방법을 제안하였고 기존의 방식들은 특정 task마다 loss나 architecture를 specific하게 design해주었던 것과 달리 하나의 structure를 사용하여 모든 tasks에 적용할 수 있도록 base model을 제공했다는 점이 큰 contribution이라고 할 수 있다.

- 목표: image-to-image translation의 모든 문제에 대한 공통의 프레임워크 개발
- Contribution
> 1. 다양한 문제에서 conditional GAN이 합리적인 결과를 산출한다는 것을 증명함
> 2. 좋은 결과를 얻기에 충분히 간단한 프레임워크를 제시하고, 몇가지 중요한 모듈 선택의 효과를 분석함

----

### Keyword

- Image-to-image translation: 충분한 데이터가 주어졌을 때, 한 장면을 **가능한 다른 표현**으로 변환하는 작업
- SSIM(structural similarity index measure): 두 이미지의 유사도를 luminance, contrast, structure 3가지 요소를 이용하여 비교하는 방법
- Feature matching loss: 어떤 함수를 통해 feature map을 추출한 후, feature map 간의 거리를 계산하여 이미지 간의 유사성을 측정함

#### PatchGAN

- 이미지를 패치단위로 나눠 예측
- 최종 discriminator의 출력은 모든 패치에 대한 값을 평균 냄
> "연산량 감소 + 속도 증가 + 임의로 큰 이미지에 적용 가능" 하다는 장점들이 존재

----
### 선행연구

- [9] A. Buades, B. Coll, and J.-M. Morel. A non-local algorithm for image denoising. In CVPR, 2005.
  -  이미지의 모든 픽셀에 대해 비슷한 패턴을 갖는 영역의 평균 사용하는 non-local algorithm 제안
- [11]T. Chen, M.-M. Cheng, P. Tan, A. Shamir, and S.-M. Hu. Sketch2photo: internet image montage. ACM Transactions on
Graphics (TOG), 28(5):124, 2009.
  - 변환할 최적의 이미지를 이미지 검색 시스템을 통해 찾는 방법
- [18] D. Eigen and R. Fergus. Predicting depth, surface normals and semantic labels with a common multi-scale convolutional
architecture. In ICCV, 2015.
  - semantic labeling을 다중스케일 합성곱 신경망 구조를 이용하는 방법 제안
- [58] S. Xie and Z. Tu. Holistically-nested edge detection. In ICCV, 2015
  - 이미지에서 물체의 경계를 검출할 수 있는 심층신경망인 HED를 사용하여 윤곽선을 검출하는 방법 제안
- [62] R. Zhang, P. Isola, and A. A. Efros. Colorful image colorization. ECCV, 2016.
  - 깊은 CNN 구조를 사용하여 색상의 분포를 추정하는 알고리즘 제안

---
### GAN

- GAN은 데이터에 적응하는 loss를 학습하기 때문에 blurry 이미지의 경우, discriminator가 가짜 이미지로 판별하여 이미지 생성 task를 해결함
- CGAN은 입력 이미지에 조건(condition)을 적용하여 그에 상응하는 출력이 나오도록 유도하며, 이는 image-to-image translation에 적합함

---
### Method

#### Generator

<img src='https://github.com/mingii4922/Generative-Adversarial-Network-GAN/assets/79297596/beace557-3b9a-47b1-9ded-6626a034aad8' width=700></center>

- Encoder-decoder에 skip connection이 추가되어 feature를 직접 공유하는 **U-Net**을 사용
  - 이전 연구들은 autoencoder를 사용하지만, 많은 이미지 변환 문제의 경우 입출력 간 많은 양의 low-level information을 공유해야 함

<img src='https://github.com/mingii4922/Generative-Adversarial-Network-GAN/assets/79297596/d00b505e-2b0f-4c3b-9d43-fff250c2dfa1' width=700></center>

- 일반적인 image-to-image translation 문제를 해결하기 위해 입력과 출력이 외관상 달라보이지만, 입력과 출력이 모두 동일한 기본 구주의 렌더링 형태를 가진 형태로 구성됨
  - 이는 이미지의 구조가 대략적으로 정렬되어 있다는 것을 고려하여 설계함

#### Discriminator

<img src='https://github.com/mingii4922/Generative-Adversarial-Network-GAN/assets/79297596/c90a48b3-3ef8-4afe-83b7-560dc622f1d6' width=500></center>

- High-frequency structure를 모델링하도록 제한하고, L1 term이 low frequency 정보를 잡아내도록 역할 분담
- **PatchGAN**을 사용하여 local image patch로 이미지를 제한적으로 관찰하며, 고주파 구조에 집중하도록 설계

#### Objective


<img src='https://github.com/mingii4922/Generative-Adversarial-Network-GAN/assets/79297596/60e912af-3565-4b88-90d4-717cee80ace3' width=900></center>

$$\begin{aligned}
  L_{GAN}(G,D) &= \mathbb{E}_{y} [log D(y)] + \\
               &= \mathbb{E}_{x,z} [log(1-D(G(x,z))].\\
  L_{L1}(G) &= \mathbb{E}_{x,y,z}[||y-G(x,z)||_{1}].
               
\end{aligned}$$


#### Loss function

- 해당 논문은 이전 논문과 달리, L1 distance를 사용함
  - L2 distance는 블러효과를 발생한다는 단점이 있음
  
- L1 norm(Mahatten distance): 벡터의 모든 원소들의 차이의 절댓값을 합한 값
- L2 norm(Euclidean distance): 벡터들의 모든 원소들의 차이의 제곱의 합을 루트로 씌워준 값





#### Optimization

- Generator를 학습시킬 때 $log(1-D(x, G(x,z)))$에 최소화하지 않고, $log(D(x, G(x,z)))$에 최대화하는 방향으로 학습
- Discriminator 최적화시, objective를 2로 나눠 generator보다 천천히 학습하게 의도
- Minibatch SGD, adam optimizer 사용

#### Metrics

- 전통적인 지표는 결과물의 결합 통계량을 평가하지 못하고, structured loss가 보고자 하는것을 측정하지 못한다는 단점 존재
  - 결합 통계량: 다변량 확률변수로 결합된 함수의 기댓값(픽셀 하나하나가 확률변수)

- AMT perceptual studies
  - 사람들이 직접 생성된 이미지와 실제 이미지에 대해 real/fake를 판단하는 평가방법
  >  - 각 사진마다 1초동안만 보여주고 이미지가 사라진 후, 무제한 시간으로 응답할 시간을 제공
    - 처음 10개의 사진으로 연습한 후, 결과에 대한 피드백 제공
    - 한 명당 한 개의 실험에만 참여 가능
    - 한 실험에는 40개의 사진으로 구성
    - 각 알고리즘 마다 약~50명이 검증함

- FCN-score
  - Pre-trained semantic classifier(FCN-8S)를 이용해 생성된 이미지들의 semantic segmentation 결과를 통해 realistic한지 판단
