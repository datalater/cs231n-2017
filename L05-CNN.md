
ⓒ 2017 JMC

**NB Viewer** : http://nbviewer.jupyter.org/github/datalater/cs231n-2017/blob/master/L05-CNN.ipynb

---

## L05 Convolutional Neural Networks

### 01 Fully Connected Layer vs. Convolutional Layer

**Layer == function**:

1. input neuron
2. $Wx$
3. output neuron == activation

각 레이어는 input neuron을 weight matrix와 multiplication한 후 그 결과값을 output neruon으로 리턴한다.
output neuron은 input neuron이 해당 layer의 weight와 연산된 것이므로 activation이라고도 부른다.
즉, layer는 쉽게 말하면 일종의 함수이다.

**Fully Connected Layer**:

1. input neuron : flatten
2. $Wx$ : dot product with entire input neuron
3. output neuron : number of output neurons == number of target classes

1\) 가장 먼저 input neuron을 하나의 행으로 펼친다.
예를 들어, 32x32x3 이미지는 1x3072 벡터가 된다.
2\) $Wx$를 하기 위해 1x3072 벡터와 3072x10 weight matrix를 곱한다.
input neuron 전체가 weight matrix와 dot product 된다.
3\) output neuron은 펼쳐진 input 전체가 weight matrix의 하나의 column과 dot product되어 1x10 activation으로 생성된다.

즉 32x32x3 size를 가진 input neuron이 weight matrix multiplication을 통해 1x10 output neroun으로 바뀌었다.

+ **intuition** : input neuron을 c개의 관점을 가진 weight matrix와 dot product 시키기 때문에 그 결과값인 output neuron은 input neuron을 c개의 관점으로 분석한 정보(~=score)로 변환된다. (c = number of target classes)

**Convolutional Layer**:

컨볼루션 레이어는 fully connected layer와 다르게, input neuron의 공간 정보(spatial structure)를 보존한다.
따라서 32x32x3 이미지를 하나의 행으로 펼치지 않고 시작한다.

1. input neuron : default
2. $Wx$ : dot product at every spatial location
3. output neuron

1\) input neroun은 default 상태로 둔다.
2\) $Wx$를 할 때 weight matrix가 small filter로 대체된다.
예를 들어, 5x5x3 filter라면, filter는 이미지 전체의 공간을 모두 훑으면서 dot product를 한다.
fully connected layer가 weight matrix와 input neuron을 곱할 때 input neuron 전체를 한번에 곱했다면, convolutional layer는 input neuron의 특정 공간만큼만 filter와 곱한다.
filter는 위와 같이 그리드 방식으로 전체 input neuron을 훑어서 output neuron을 리턴한다.
이때 특정 공간의 정보를 온전히 담기 위해 filter는 input neuron의 full depth까지 훑는다. (그래서 filter size(5x5x3)와 input size(32x32x3)의 마지막 차원의 값은 항상 같다)
그런데 보통 filter는 한 개가 아니라 여러 개를 사용한다.
왜냐하면 filter는 특정 정보(feature)를 찾아내는 하나의 관점이고, 하나의 현상을 제대로 분석하려면 여러 개의 관점으로 분석하는 것이 유리하듯이, 여기서도 여러 개의 filter로 input neuron을 분석하는 게 유리하기 때문이다.
즉 filter가 k개라면 activation도 k개 만들어진다.
이렇게 만들어진 activation k개 각각을 activation map이라고 부른다.

+ **intuition**: input neuron의 특정 공간을 filter와 dot product 시키기 때문에, 그 결과값인 output neuron은 input neuron의 특정 공간에 대한 정보를 담고 있게 된다. 이때 filter는 그리드 방식으로 input neuron을 모두 훑어서, input neuron에서 filter의 특징이 두드러지게 나타난 곳이 어딘지 감지한다. 그리고 convolutional layer는 여러 개의 filter를 사용므로 하나의 input neuron을 여러 개의 특징으로 분석할 수 있게 된다.

### 02 Hierarchical Filters in ConvNet

**ConvNet**:

CNN은 여러 개의 activation function과 여러 개의 convolutional layer가 층층이 쌓여서 구성하는 아키텍처이다.
학습이 끝나면 각각의 convolutional layer의 filter들은 서로 다른 수준으로 feature(input neuron의 특정 정보)를 학습한다.
앞단에 있는 conv layer의 filter는 edge 같은 low-level feature를 represent한다.
중간에 있는 conv layer의 filter는 corner나 blob 같은 좀 더 복잡한 mid-level feature를 represent한다.
뒷단에 있는 conv layer의 filter는 blob 보다 더 복잡한 high-level feature를 represent한다.

`@@@resume`: https://youtu.be/bNb2fEVKeEo?t=27m





---


---
