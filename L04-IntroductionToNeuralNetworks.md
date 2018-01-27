ⓒ 2017 JMC

**NB Viewer** : http://nbviewer.jupyter.org/github/datalater/cs231n-2017/blob/master/L04-IntroductionToNeuralNetworks.ipynb

---

## 20180127 Study Notes


---

## L04 Introduction to Neural Networks

**5**:

지난 시간에는 함수 $f$와 parameter $W$를 이용해서 어떻게 classifier를 정의해야 할지에 대해 이야기했습니다.
이때 함수 $f$는 데이터 $x$를 input으로 받고 output으로 각 class에 대한 score 벡터를 내뱉습니다.

여기서 loss function $L_i$을 정의할 수 있고, 그 예롤 SVM loss function을 이야기했었습니다.
loss function은 모델이 만들어낸 score가 얼마나 만족스러운지 또는 불만족스러운지에 대한 함수였습니다.
개별 loss를 사용해서 total loss function을 정의하는데, 기본적으로 최종 loss function $L$은 data loss term과 regularization term으로 구성합니다.
regularization은 모델이 얼마나 simple한지 모델의 복잡도를 측정하는 기준으로써, test data에서 성능이 잘 나오기 위해 간단한 모델을 선호하는 것이 regularization의 목적입니다.

그리고 이제 우리가 원하는 건 가장 낮은 loss에 해당하는 parameter W를 찾는 것입니다.
loss function 값을 최소화하기 위해 W에 대한 gradient L(gradient of L with respect to W)을 구합니다.

**6**:

이를 구하는 과정이 optimization입니다.
optimization의 한 가지 방법이 gradient descent로써 가장 경사가 급하게 내려오는 방향으로 반복적으로 스텝을 밟아가면서 가장 loss가 낮은 곳으로 도달하는 것입니다.
이때 방향은 negative gradient descent에 해당합니다.
gradient descent가 사선으로 움직이면서 minimum loss에 도달하는 것을 지난 시간에 확인했습니다.

**7**:

그리고 gradient를 계산하는 2가지 방법에 대해 이야기했습니다.

첫 번째 방법은 numerical gradient(수치 미분)입니다.
아주 작은 값($f(x+h)$)을 추가하는 방식인 finite difference approximation을 사용해서 구합니다.
수치 미분은 매우 느리고, 정확한 값이 아니라 근사값이지만, 매우 간편하게 작성할 수 있습니다.

두 번째 방법은 analytic gradient(해석 해)입다.
analytic gradient의 정확한 수식(expression)을 구할 수 있으면 매우 빠르고 정확하게 계산할 수 있습니다.
반면에 수식을 얻기 위해서 반복적인 수학 계산과 미분을 사용해야 하기 때문에 실수가 발생할 가능성이 높습니다(error-prone).

실제로 우리가 원하는 것은 analytic gradient를 도출해서 사용하는 것이지만, 그 과정에서 올바르게 수식을 구했는지를 점검하기 위해 numerical gradient를 사용해서 체크합니다.

**8**:

그래서 오늘은 임의의 복잡한 함수에 대해 analytic gradient를 어떻게 구할 수 있는지 computational graph(계산 그래프)라는 도구를 사용해서 이야기합니다.

계산 그래프란, 어떤 함수든 표현할 수 있는 그래프이며 각 노드는 우리가 거쳐야 하는 하나의 계산을 의미합니다.

그림은 linear classifier를 사용한 multiclass SVM loss function을 계산 그래프로 표현한 것입니다.
먼저 그래프 첫 단에 input x와 parameter W가 있고 두 행렬 W와 x의 곱셈 노드가 등장하는데 그 노드의 output은 score 벡터가 됩니다.
그 다음에 등장하는 계산 노드는 hinge loss를 나타냅니다.
hinge loss는 data loss인 $L_i$를 뜻하구요.
또한 첫 단에 있는 W가 Regularization 계산 노드를 거쳐서 data loss와 regularization term이 합해지는 덧셈 노드를 거칩니다.
그래서 그래프 최종 단에는 regularization term과 data loss의 합인 total loss인 $L$이 나오게 됩니다.

계산 그래프를 사용하는 이유는 backpropagation(오차역전파) 기술을 사용할 수 있기 때문입니다.
backpropagation은 모든 변수에 대한 gradient를 계산하기 위해 계산 그래프에서 recursively(재귀적으로) chain rule을 적용하는 기술입니다.

**9-11**:

이 방법은 특히 매우 복잡한 함수를 다룰 때 유용합니다.
예를 들면 CNN에서도 매우 유용합니다.
매우 복잡해서 input이 수많은 layer를 통과하면서 여러 번의 변형을 거쳐야 최종 loss가 나옵니다.

**12-23**:

backpropagation은 어떻게 작동할까요?
아주 간단한 예제로 살펴보겠습니다.

1. $f = (x+y)z$
+ 함수가 있으면 먼저 계산 그래프로 나타냅니다.
2. $q = x + y$, $f = qz$
+ 중간에 계산되는 값에 변수 이름을 부여하고 식으로 나타냅니다.
3. $\frac{\partial q}{\partial x} = 1$, $\frac{\partial q}{\partial y} = 1$, $\frac{\partial f}{\partial q} = z$, $\frac{\partial f}{\partial z} = q$
+ 중간 변수의 각 입력값에 대한(ex. gradient of q w.r.t. x and y) gradient를 구합니다.
4. $\frac{df}{dx} = \frac{\partial f}{\partial q} \cdot \frac{\partial q}{\partial x}$, $\frac{df}{dy} = \frac{\partial f}{\partial q} \cdot \frac{\partial q}{\partial y}$
+ 최종 출력부터 차례대로 chain rule을 recursive하게 적용해서 각 입력 변수에 대한 grdient를 계산합니다.
+ 입력 변수가 중간 변수를 거쳐야 하는 최종 출력에 도달할 경우 chain rule을 적용합니다.
+ 예를 들어, $y$가 $f$에 미치는 영향을 계산하려면 $(x+y)$가 $f$에 미치는 영향을 계산하고 $y$가 $(x+y)$에 미치는 영향을 계산한 후 chain rule에 근거하여 서로 곱하면 된다.

**24-29 upstream gradient $\times$ local gradient**:

computational graph 관점에서 보면 기본 단위를 중간 변수로 잡을 수 있다.
중간 변수는 항상 입력 변수 $x$와 $y$가 있고 계산 노드 $f$가 있고 출력 변수 $z$가 있다고 일반화할 수 있다.
출력 변수 $z$는 다음 노드의 입력 변수가 된다.
이렇게 계산 그래프의 기본 단위를 정의한 후 다음과 같이 backpropagation을 유도할 수 있다.

1. $\left[\frac{\partial z}{\partial x}, \frac{\partial z}{\partial y}\right]$
+ 출력 변수($z$)의 각 입력 변수($x$,$y$)에 대한 local gradient를 구한다.
2. $\left[\frac{\partial L}{\partial z} \times \frac{\partial z}{\partial x}, \frac{\partial L}{\partial z} \times \frac{\partial z}{\partial y}\right]$
+ 다음 노드의 출력 변수($L$)의 현재 노드의 입력 변수($x$,$y$)에 대한 gradient를 구한다.
+ 중간 변수($z$)를 거쳐야 하므로 chain rule에 근거하여 gradient를 곱셉한다.
+ 의미로 정리하면 upstream gradient $\times$ local gradient가 된다.

**30-43 backpropagation through more complex example**:

`@@@resume : 20:40~`

upstream gradient와 local gradient를 구해서 차례대로 backpropagation을 하는 과정이 곧 전체 함수의 입력 변수에 대한 analytic gradient를 구한 것이다.

**44**:

computational graph를 그릴 때 여러 계산 노드를 압축시켜서 하나의 노드로 만들어서 표현할 수도 있다.
예를 들면 sigmoid 함수값을 계산하려면 4개의 개별 계산 노드를 거쳐야 하는데, 그냥 sigmoid 함수 통째로 1개의 계산 노드로 표현할 수도 있다.

`@@@resume : 26:30~`




---

**END**
