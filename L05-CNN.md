ⓒ 2018 JMC

**NB Viewer** :

http://nbviewer.jupyter.org/github/datalater/cs231n-2017/blob/master/L05-CNN.ipynb

---

## 20180203 Study Notes

**39**:

+ filter와 가장 비슷한 부분이 activation map에서 하얀색으로 드러난다.
+ CNN에서 쓰는 convolution은 사실 correlation이고 신호처리에서 사용하는 기존의 convolution은 flipped correlation이다.

**40**:

+ FC : 선형분류기 (separable linear classifier)

**55**:

+ 입력 이미지 픽셀 사이를 패딩하는 기법도 있는데 그것을 dilated CNN이라고 한다.
+ 픽셀 사이에 패딩을 하면 이미지를 더 넓게 볼 수 있는 장점이 있다.

**72**:

+ max-pooling : 인식 가능한 정도로 feature를 추출하고 연산량을 줄인다.
+ global average pooling : attention 찾을 때 쓰는 풀링 방법. 개를 찾는 문제에서 어디에서 개를 찾고 있는 것인지 확인할 때 쓴다.


**END**

---

## L05 Convolution Neural Networks

**3: 지난 시간 복습**

지난 시간에 linear score function을 이야기했습니다.
$f = Wx$로 표시했었죠.
그리고 2-layer 신경망에 대해서 이야기했습니다.
2-layer 신경망은 linear layer 2개를 연결한 것이고 그 사이에는 non-linearity가 있었죠.
그리고 mode problem에 대해서 이야기했습니다.
mode problem이란 다른 종류의 자동차 또는 비행기 같은 여러 가지 클래스를 찾는 intermidiate template을 학습하는 것을 말합니다.
그리고 입력 이미지를 여러 가지 template이 비교해보면서 각 클래스에 대한 final score를 출력했었죠.

**4: 오늘의 주제 CNN**

오늘 이야기할 것은 CNN입니다.
CNN은 기존에 이야기 했던 2-layer 신경망과 기본적인 원리는 같습니다.
다만 convolutional layer를 학습해서 spatial structure(공간 구조)를 유지하는 게 CNN의 특징입니다.

**5**:

CNN을 이야기하기 전에 역사적인 흐름을 볼게요.
예전에는 layer가 1개인 신경망을 사용했고 feedforward로 원하는 결과가 나올 때까지 가중치를 직접 조절하는 방식을 사용했습니다.

**6**:

그러다가 layer를 여러 개 쌓아서 다층 신경망을 만들었고, 현재의 신경망과 모양새가 비슷하지만 이때까지도 backpropagation이 없었고 다층 신경망을 학습시킬 어떤 방법도 없었습니다.

**7**:

그 후에 1986년 Rumelhart에 의해 backpropagation이 소개되었습니다.
chain rule과 update rule을 적용하게 된 거죠.
이때 신경망 구조를 학습시킬 수 있는 원리가 처음 발견되었습니다.

**8**:

그러다 그후에도 매우 큰 신경망은 여전히 학습하기 어려웠습니다.
새로운 것들이 등장하지 않았고 신경망을 대중적으로 사용하지도 못했기 때문에 인공지능 분야가 잠시 주춤했었죠.
2006년에서야 딥뉴럴넷을 효율적으로 훈련시킬 수 있다는 게 Hinton에 의해 증명되었습니다.
그러나 이때까지도 현재의 신경망 훈련과는 차이가 있습니다.
이때는 backpropagation을 쓰려면 initialization이 매우 중요했기 때문이죠.
그래서 제한된 볼츠만 머신(RBM)을 hidden layer가 통과하는 pre-training 절차를 먼저 거칩니다.
RBM을 거치면 각 레이어를 반복적으로 훈련시켜서 어떤 initialization weight가 나오거든요.
이런 방식으로 모든 hidden layer를 pre-training을 거치게 해서 전체 신경망을 initialize 하고 거기서부터 backpropagation을 하고 fine tuning을 하는 거죠.

**9**:

결과가 잘 나오기 시작한 것은 2012년이었습니다.
음성 인식 분야였죠.
제프리 힌튼 교수 연구실에서 연구한 음성 인식이었습니다.
그 후에 이미지 인식에서 제프리 힌튼 교수 연구실의 연구생 알렉스가 AlexNet을 만들었는데 ImageNet classification에서 처음으로 convolution neural network 모델을 소개했고, 기존의 모델보다 성능이 훨씬 더 잘 나왔습니다.
그 후부터 CNN이 다양한 분야에서 매우 널리 사용되기 시작했습니다.

**10**:

convolution이 뭔지 설명할게요.
1950년에 시각적인 인지를 담당하는 세포를 이해하려는 연구가 있었습니다.
Hubel과 Wiesel이 고양이를 대상으로 실험했죠.
고양이 뇌에 전기장치를 연결하고, 시각적으로 다른 자극을 주고 어떤 반응이 있는지 실험했습니다.
가령 대각선을 보여줄 때와, 동그라미를 보여줄 때, 서로 방향이 다른 대각선을 보여줄 때 고양이 뇌의 뉴런에서 나타나는 반응이 달랐던 거죠.

**11**:

그들이 연구한 결과, 2가지 결론이 있었습니다.

첫째, 가까이 붙어 있는 시각 세포는 비슷한 이미지를 인식한다는 겁니다.
멀리 있는 세포와 가까이 있는 세포는 서로 다른 이미지를 인식하는 거죠.

가령, 사람이 가장 바깥 쪽이 청색이고 중심으로 갈수록 붉은 색으로 칠해진 과녁 형태의 그림을 보고 공간 맵핑(spatial mapping)을 취해 시각 피질로 맵핑한다면, 시각 피질에서도 가장 바깥쪽 세포가 청색을 인식하고 가장 중심에 있는 세포가 붉은 색을 인식한다는 겁니다.

**12**:

둘째, 시각 피질의 뉴런이 계층적 구조(hierarchical organization)를 가지고 있다는 것입니다.

다른 종류의 시각적 자극을 보면 뉴런에서 신호가 가장 먼저 들어오는 레이어의 세포는 매우 간단한 세포이고 뒤로 갈수록 복잡한 세포입니다.

간단한 세포는 빛의 방향을 인식합니다.
좀 더 복잡한 세포는 빛의 방향과 움직임을 인식합니다.
매우 복잡한 세포는 물체의 끝점의 움직임을 인식합니다.

세포의 복잡성이 증가할수록 인식의 수준도 높아지는 거죠.

**13**:

이후 1980년에 neuocognitron이 발표되었습니다.
Fukushima가 발표한 뉴로코그니트론은 휴블과 위즐이 알아낸 간단한 세포와 복잡한 세포의 계층적 인식 구조의 원리를 적용한 첫 번째 신경망 모델이었습니다.

뉴로코그니트론은 simple cell과 complex cell이 교대되는 레이어로 구성되어 있습니다.
simple cell은 조절 가능한 parameter가 있고 그 뒤에 연결되는 complex cell은 pooling과 비슷한 작업을 수행합니다.
pooling은 simple cell이 조금 수정되었을 때 그것에 민감하지 않게(invariant) 만드는 작업입니다.

**14**:

그리고 Yann LeCun이 CNN에 backpropagation과 gradient-based learning을 적용해서 document recognition을 하는 모델을 소개했는데 매우 성능이 잘 나왔습니다.
특히 숫자(digit)로 구성되는 우편 코드(zip code) 인식이 매우 뛰어났고 실제 우편 산업에 널리 적용되었습니다.

그러나 더 나아가서 더 복잡한 데이터에 적용되지는 못했습니다.
숫자는 상대적으로 형태도 복잡하지 않고 인식할 종류 (0-9) 또한 간단했기 때문이죠.

**15**:

2012년에 제프리 힌튼 연구실의 알렉스가 현대적인 CNN 모델을 발표합니다.
바로 AlexNet인데요, 사실 좀 더 깊은 모델일 뿐 얀 르쿤이 소개한 모델과 크게 다르지는 않습니다.
그러나 AlexNet이 발표된 당시에는 ImageNet dataset이나 웹 이미지라는 매우 양이 큰 데이터와 GPU의 병렬 컴퓨팅 파워를 활용할 수 있었다는 게 차이점이었죠.

**16-24**:

다시 현재 시점으로 돌아와서 보면 ConvNet은 정말 많이 쓰입니다.

+ 이미지 classification (분류)
+ 이미지 retrieval (검색)
+ 이미지 detection (사물인식)
+ 이미지 segmentation (세분화)

참고로 이미지 segmentation은 bounding box를 찾는 detection에서 더 나아가서 모든 픽셀의 바깥 경계에 레이블을 매기는 작업입니다.

또한 ConvNet은 자율주행자동차에도 쓰이는데, 이는 모두 GPU의 병렬 컴퓨팅 파워가 발전해서 ConvNet에서 필요한 연산이 효율적으로 빨라진 덕택입니다.

+ face recognition
+ video classification
+ pose recognition
+ image recognition in reinforcement learning

그리고 face recognition에도 쓰입니다.
face recognition은 사람의 얼굴을 입력하고 그 얼굴 이미지가 어떤 사람인지에 대한 likelihood를 출력하는 것입니다.

> **Note**: likelihood : 데이터 x가 주어졌을 때 특정 class일 확률

+ medical image에 대한 해석과 진단
+ galaxiy classification
+ street sign recognition
+ whale recognition (from Kaggle challenge)
+ map image segmentation (지도 이미지에 대한 segmentation)

지금까지 이야기한 recognition, classfication, detection 외에 다른 분야에도 ConvNet이 적용된다.

+ image captioning (이미지에 대해 설명하는 문장 출력하기)
+ neural style transfer

---

**25-27: Fully Connected Layer**

지난 시간에 Fully Connected Layer에 대해서 이야기했습니다.
fully connected layer는 들어온 input 이미지의 모든 픽셀이 다 계산됩니다.
이미지가 32x32x3 사이즈라면, 그것을 1차원으로 쭉 펼친 3072x1 차원 벡터값이 layer의 weight와 곱해지는 것이죠.
그리고 출력될 값이 10개라면 다음과 같은 형태가 됩니다.

+ input $x$: 32x32x3 $\rightarrow$ 3072x1
+ weight $W$ : 10x3072
+ dot product $Wx$ : (10x3072) x (3072x1)

$$\mathbf{input} : 32 \times 32 \times 3 \rightarrow \mathbf{Wx} : (10 \times 3072) \cdot (3072 \times 1) \rightarrow \mathbf{activation} : 1 \times 10$$

**28-30: Convolution Layer**

convolution layer가 fully connected layer와 다른 가장 큰 차이점은 spatial structure(공간 구조)를 보존한다는 점입니다.

convolution layer는 fc layer와 달리 input을 1차원의 긴 벡터로 펼치지 않습니다.
3차원 구조를 그대로 유지합니다.
그리고 convolve를 합니다.

**31: convolve**

convolve란 input 이미지에 filter를 연산하는 것을 말합니다.
가령 필터의 크기가 5x5x3이라고 합시다.
이 필터를 3차원 이미지에 slide 하여 필터와 이미지가 겹치는 각 공간마다 dot product를 계산합니다.

필터가 이미지와 convolve 할 때는 항상 이미지 부피의 가득 채워서 합니다.
그래서 필터가 convolve 하는 영역은 이미지의 일부 공간이 됩니다.

+ image $x$ : 32x32x3 (but we use part of input as big as filter)
+ filter $w$ : 5x5x3
+ dot product $w^{T}x$ : (1x75) x (75x1)

위 convolve 과정을 image의 좌측 상단부터 우측 하단까지 진행합니다.
필터를 이미지 전체에 convolve 하는 것을 slide 한다고 말합니다.
입력 이미지에 하나의 필터를 slide 하면, 이미지의 가장자리의 값은 필터의 중심 포인트가 될 수 없기 때문에 28x28x1 이미지가 생성됩니다.

**32-34: activation map**

필터를 이미지 전체에 slide 한 결과물을 activation map이라고 합니다.

그리고 convolution layer에서는 필터를 여러 개 사용합니다.
왜냐하면 각각의 필터는 입력 이미지의 특정 template을 학습하기 때문에 여러 개 있을수록 입력 이미지의 다양한 특징들을 분석할 수 있기 때문입니다.

가령 filter를 6개 사용했다면 activation map이 6개 나옵니다.
따라서 28x28x6 이미지가 convolution layer의 output이 됩니다.

**35-36: ConvNet**

신경망은 linear layer를 여러 개로 층층이 쌓은 것입니다.
마찬가지로, ConvNet은 convolution layer가 여러 개로 구성된 하나의 나열인데, convolution layer 사이사이마다 activation function이 산재하고 있습니다.

ConvNet의 일반적인 구조는 다음과 같습니다.

+ $\mathbf{[(CONV-RELU) * N - POOL?] * M - (FC - RELU) * K, SOFTMAX}$


**37-38**: **ConvNet as a hierarchical feature learning**

ConvNet을 구성하는 여러 개의 convolution layer는 각각 여러 개의 필터를 갖고 있습니다.
그리고 각 필터마다 하나의 activation map을 만들어내고요.

그래서 ConvNet에 층층이 쌓인 convolution layer를 보면 결국 hierarchical filter를 학습한다는 것을 알 수 있습니다.

+ earlier layer : represent low-level features that you're looking for (e.g. edge)
+ middle layer : represent mid-level features that you're looking for (e.g. corners and blobs)
+ later layer : represent high-level features that you're looking for (e.g. concepts)

이러한 과정은 Hubel과 Wiesel이 고양이 뇌 실험에서 발견한 연구결과인 simple cell이 low-level feature를 학습하고 뒤로 갈수록 complex cell이 high-level feature를 학습한다는 점과 일치한다는 것을 알 수 있습니다.

참고로 그리드 안에 있는 이미지는 backpropagation 결과 뉴런이 가장 활성화되는 값을 뜻하며, 각 뉴런이 찾고 있는 이미지의 특징이라고 보면 됩니다.

**39**:

가령 특정 방향을 나타내는 edge를 나타내는 뉴런은 input을 넣고, 해당 뉴런의 필터를 slide 하면 이미지 상에서 그 방향에 대한 edge가 나타내는 영역에서 훨씬 더 높은 값이 나올 것입니다.

**40: 입력 이미지와 ConvNet**

입력 이미지를 넣었을 때 ConvNet을 전체적으로 본 것입니다.
입력 이미지를 여러 개로 구성된 layer 나열에 넣습니다.
가장 먼저 convolutional layer를 통과합니다.
conv layer 뒤에는 바로 non-liner layer를 통과합니다.
여기서는 RELU layer를 썼습니다.
이런 방식으로 입력 이미지가 CONV-RELU-CONV-RELU를 통과하고 그 다음에는 pooling layer를 통과합니다.
pooling layer는 기본적으로 activation map의 사이즈를 줄여서 다운샘플링하는 것을 말합니다.
입력 이미지가 CONV-RELU-CONV-RELU-POOL을 여러 번 거친 후에는 fully connected layer를 통과하여 final score를 출력합니다.

**41-52: example of how the spatial dimensions work out**

+ Output size : (N-F) / stride + 1

**53-55: zero pad to the border**

Q.  what is the output below?

+ input : 7x7
+ filter : 3x3
+ stride : 1
+ pad with 1 pixel border

A. 7x7 output!

+ N = 9, F = 3, stride = 1
+ (9 - 3) / 1 + 1 = 7

Q. 필터 사이즈가 5일 때와 7일 때 zero padding을 몇을 해야 원래 이미지와 같은 크기의 activation map이 나오는가?

A. (F-1)/2 만큼 zero padding 하면 된다.

+ F = 3 => zero pad with 1
+ F = 5 => zero pad with 2
+ F = 7 => zero pad with 3

**56: why zero pad**

zero padding을 해서 input size를 유지하려는 이유는 여러 개의 conv layer로 구성된 CNN을 사용할 때 입력 이미지가 conv layer를 거칠 때마다 activation map의 사이즈가 줄어든다면 기존의 이미지를 나타낼 수 있는 정보를 잃어버리게 되기 때문입니다.
필터의 측면에서 생각해봐도 기존에 학습한 edge의 정보가 사이즈가 줄어들어서 뒤로 갈수록 사라진다면 당연히 좋지 않겠죠.

**58-60: example of zero-paded output**

Q. whats the output volume size below?

+ input volume : 32x32x3
+ filte size : 5x5x3
+ number of filters : 10
+ stride : 1
+ pad : 2

A. 32x32x10

+ (32 + 2 * 2 - 5) / 1 + 1 = 32 spatially

+ so, 32x32x10

Q. what's the number of parameters in this layer?

A. each filter has 5 * 5 * 3 weights + 1 bias = 76, so 76 * 10 = 3

**61-62: Common numbers of the Conv Layer**

+ 필터 사이즈 : 3 or 5
+ 스트라이드 : 1 or 2
+ 필터 개수 : powers of 2, e.g. 32, 64, 128, 512
+ 패딩 사이즈 : whatever will preserve your spatial extent

parameter sharing

+ parameter sharing : activation map을 구성하는 모든 뉴런이 하나의 filter로 만들어진다 == 하나의 filter의 parameter를 share 해서 activation map이 만들어진다.
+ fully connected layer는 convolution layer와 달리 parameter를 share 하지 않고 각기 다른 parameter를 사용하므로 parameter 수가 훨씬 더 많다.


**63: Does 1x1 convolution make sense?**

1x1 convolution, 완벽히 말이 된다.

각 필터가 input volume의 entire depth까지 적용되면서 dot product 되기 때문에 말이 된다.

+ input volume : 56x56x64
+ filter size : 1x1x64
+ number of filters : 32
+ activation map : 56x56x32

**64-65: example of conv layer in Torch, Caffe**

**66: brain/neuron view of CONV Layer**

`@@@resume`: 50분24초 (https://youtu.be/bNb2fEVKeEo?t=50m24s)


---




---

**END**
