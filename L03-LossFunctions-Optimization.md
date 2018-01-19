
ⓒ 2017 JMC

**NB Viewer** : http://nbviewer.jupyter.org/github/datalater/cs231n-2017/blob/master/L03-LossFunctions-Optimization.ipynb

---

## 20180120 Study Notes

---

## L03 Loss Functions and Optimization

**05**:

이미지 분류가 컴퓨터로 하기 힘든 이유

**06**:

data-driven 접근법으로 KNN 소개.
이미지 데이터셋 CIFAR-10 소개. 10개의 카테고리 존재.
KNN이 decision boundary를 정하는 방법 == training data를 기준으로 최근접 포인트 다수결.
K나 distance metric을 몇으로 정해야 할지에 대한 문제를 이야기하면서 hyperparameter setting을 cross-validation으로 한다는 이야기 했음.
cross-validation은 데이터를 training, validation, test set으로 나눠서 한다.

**07**:

linear classifier는 앞으로 배울 신경망의 가장 기초가 되는 구성 단위라고 얘기했음.
linear classifier는 parametric classifier의 한 종류이다.
training dat의 모든 정보가 parameter matrix인 W에 요약된다.
W는 training 하면서 정해진다.

이미지 분류에서 linear classifier가 어떻게 작동하는지 복습해보면, 이미지를 1차원 칼럼 벡터로 쭉 펼친다.
가령, CIFAR-10이미지를 다룬다면, 32x32x3 차원의 input 이미지는 3072 벡터로 표현되고 그게 x이다.
W는 parameter matrix인데 이미지를 나타내는 x와 행렬 곱셈을 해서 (클래스 개수만큼) 10차원 칼럼 벡터를 뱉어낸다.
10차원이 값은 10개의 카테고리 각각에 대한 score이다.
score가 클수록 input 이미지가 해당 카테고리에 속할 확률이 높다고 해석한다.

그리고 linear classification을 2가지 관점으로 해석할 수 있다.
(1) 첫 번째는 각 클래스마다 template을 학습하는 것이다.
weight matrix의 각 row가 각 클래스에 대한 score에 영향을 미치는 값이고, 그 row를 정사각형 이미지로 바꾸면 그게 곧 그 클래스에 대한 template(견본) 이미지라고 볼 수 있기 때문이다.
(2) 두 번째는 픽셀에 대한 고차원 공간에서 linear decision boundary를 학습하는 것이다.

**08**:

여기까지가 지난 시간에 한 것이고 W를 어떻게 결정해야 할지에 대해서는 말하지 않았다.
최적의 W를 알아내려면 training data를 어떻게 사용해야 할까.

만약 W값이 임의의 어떤 값이라고 해보자.
그 상태에서 input 이미지가 어떤 class에 속할지 나타내는 score를 볼 수 있다.
고양이를 나타내는 이미지의 score를 보면 cat에 대한 점수는 2.9인데 개구리일 점수는 3.78로 더 높다.
그러므로 이 classifier는 잘 못하고 있는 거다.
우리가 원하는 것은 true class가 나머지 class에 비해 가장 높은 점수를 갖는 것이다.
자동차에 대한 이미지는 automobile이 가장 높으므로 잘 나왔다.
개구리에 대한 이미지는 개구리일 점수가 낮으므로 좋지 않다.
그런데 이렇게 일일이 스코어를 눈으로 확인하면서 비교하는 것은 너무 힘든 일이다.
사람의 수작업 없이 가장 좋은 W를 자동으로 구하는 알고리즘을 짜려면, W의 좋고 나쁨에 대한 정도를 수치로 나타낼 수 있어야 한다.

즉, 이러한 역할을 하는 함수는 W값을 이용해서 socre를 보고 W가 얼마나 나쁜지 수치적으로 나타내야 하는데 이를 loss function이라고 부른다.
본 강의에서는 이미지 분류에서 사용할 수 있는 여러 가지 loss function을 살펴본다.

이게 loss function의 아이디어인데, 그 다음에 우리가 필요한 건 모든 가능한 W값을 찾아보는 효율적인 절차가 필요하고 나쁜 정도가 가장 적은 W를 찾아낼 수 있어야 한다.
이러한 과정을 optimization이라고 한다.

**09**:

score를 좋고 나쁨의 기준으로 보면, 고양이 이미지는 score가 나쁘고, car 이미지는 score고 좋고, frog 이미지는 score가 굉장히 나쁘다.

**10**:

방금 말한 내용을 수학적으로 정리해서 말하면,
loss function을 이야기할 때 먼저 (x, y)로 이루어진 training set이 있고 N개가 있다고 상상한다.
x는 알고리즘의 input, 즉 이미지의 픽셀값이고 y는 알고리즘이 분류해야 할 정답인 lable이다.
CIFAR-10은 class가 10개이므로 label의 값은 0~9 사이의 정수값이 될 것이다.

각 training set에 대한 loss는 $L_i$라고 나타낸다.
$f(x_i, W)$는 prediction function을 나타낸다.
prediction function의 결과값인 score와 label y의 차이를 구하는 함수가 $L_i$로 써있는 loss function이다.
그리고 아래 첨자 i가 없는 최종(final) loss function $L$은 모든 N개의 input에 대한 loss를 평균을 낸 값이 된다.
이러한 방식은 이미지 분류뿐 아니라 다른 분야에서도 적용되는 매우 일반적인 loss function이다.
loss function은 한마디로 (x와 y) 쌍으로 이루어진 input 데이터를 넣고 paramter matrix W의 나쁜 정도를 측정하는 것이다.
loss function을 통해 가능한 W의 경우들을 살펴보고 training set에 대해 가장 loss가 최소화되는 W를 구하게 된다.

### Loss function 예시 :: Multiclass SVM loss

**11**:

이미지 분류에서 잘 작동하는 loss function 중 하나가 mutliclass svm loss이다.
기계학습에서 SVM 이라고 하면 binary class에 대해서 적용하는데 class가 여러 개일 때는 multiclass라고 이름 붙인다.

개별 training set에 대한 loss인 $L_i$는 어떻게 구하냐면, correct category의 score와 incorrect category의 score를 비교한다.
이때 correct category가 incorrect category보다 그냥 큰 것이 아니라 특정 차이 이상으로 더 커야 한다.
이를 safety margin이라고 하는데 슬라이드에서는 1로 설정되었다.
그래서 만약에 correct category의 score가 incorrect category의 score보다 1 이상으로 더 크면 loss가 0이 된다.
이렇게 하나의 input 이미지에 대해 모든 incorrect category와 비교해서 나온 모든 loss를 합하면 그게 $L_i$가 된다.
그 다음에는 이렇게 나온 모든 loss를 N개로 나눠주면 final loss $L$이 된다.

**12**:

방금 표현한 SVM loss는 case by cas로 if, then statement로 나타냈는데, 이를 max function을 사용하면 한 줄로 나타낼 수 있다.

이렇게 0과 loss를 max function에 넣는 방식을 hinge loss라고 부른다.
hinge라는 이름은 max function을 그래프로 나타냈을 때 나온 모양에서 따온 것이다.
그림에서 $S_{y_i}$는 true category에 대한 score를 나타내고 $S_j$는 false category에 대한 score를 나타낸다.
그래프는 x축의 값인 $S_{y_i}$가 x축 상에서 어떤 값이든 간에 $S_j$보다 safety margin보다 크면 loss를 나타내는 y축의 값은 항상 0이 되는 것을 나타내고 있다.
true category에 대한 score가 커질수록 loss는 safety margin에 도달할 때까지 linear하게 줄어들고 그 후 loss는 0이 된다.

**13~17**:

SVM loss 예시.
고양이 이미지에 대해 true category score와 false category score를 비교해서 loss를 계산해보니 2.9가 나왔다.
즉, 현재 input에 대한 classifier의 나쁜 정도는 2.9가 된다.
car 이미지에 대한 loss는 0이고, frog 이미지에 대한 loss는 12.9가 된다.
즉 3가지 이미지가 전체 training set이라면 final loss $L$은 (2.9 + 0 + 12.9)/3 = 5.27이 된다.

**18**:

car는 이미 loss가 0이므로 car에 대한 score가 높아지더라도 loss는 변하지 않는다.

**19**:

min = 0. max = infinity.

**20**:

모든 score가 0이라면, loss 계산할 때 safety margin만 남는다.
즉 class가 c개라면 true calss 1개를 제외한 나머지 c-1개의 false class에 대한 loss가 모두 1씩 발생하므로 loss는 c-1이 된다.

c(number of classes) - 1.


**21**:

true class에 대한 score도 loss에 포함시켜서 (true_class - true_class + safety_margin)을 계산하면 항상 safety margin이 남으므로 하나의 이미지에 대한 loss는 1씩 증가한다.

**22**:

Doesn't change.
mean을 하든 sum을 하든 loss function의 값을 constant로 rescaling 하는 것이고, rescaling은 classifier를 바꾸지 않는다.

**23**:

제곱을 하면 큰 loss가 작은 loss에 비해 훨씬 더 큰 loss를 만들게 되므로 다른 classifier가 만들어진다.
큰 loss에 더 민감한 classifier가 된다.

loss function은 알고리즘이 어떤 에러에 민감하고 어떤 에러에 둔감해지는지 결정하므로 매우 중요하다.

**24**:

Mutlicalss SVM Loss: Example code

```python
def L_i_vectorized(x, y, W):
    scores = W.dot(x)
    margins = np.maximum(0, scores - scores[y] + 1)
    margins[y] = 0 # zero out the margins corresponding to the correct class
    loss_i = np.sum(margins)
    return loss_i
```

**25**:

There are definitely other Ws.

**26**:

반례를 들어보녀 2W만 해도 L=0이 된다.

**27**:

2W에 대한 예시.
true class score가 false class score 보다 이미 1보다 더 크면, 각각의 값을 2배를 해도 여전히 1보다 크므로 loss는 여전히 0이 되어 변하지 않는다.

그러면 이상하다.
loss가 최소화되는 W가 여러 개라면, classifier가 어떻게 W를 고를 수 있을까?

**28**:

지금까지 말한 내용은 classifier에거 training set에서 loss가 0이 되는 W를 찾으라고 말한 것이다.
이를 data loss라고 한다.
그러나 실제로 우리가 중요하게 여기는 것은 training set을 정확하게 예측하는 게 아니다.
머신 러닝의 핵심은 training set으로 classifier를 찾고 test data에 적용하는 것이다.
즉 training set에 대한 performance보다는 test set에 대한 performance가 더 중요하다.
그래서 classifier에게 training set에 대한 loss를 줄이라고만 말하면 이상한 경우가 생긴다.

**29**:

파란 점은 training set 데이터이고, 그것을 잇고 있는 파란 선은 training set에 fit한 classifier를 뜻한다.
참고로 multiclass svm에서는 linear classifier를 말하고 있었지만 이 그림은 liner classifier가 아닌 general한 경우를 나타낸다.
즉 data loss만 신경쓰라고 classifier에게 말하면, training set에 완벽하게 일치하기 위해 매우 구불구불한 선이 나온다.

**30**:

만약 training set의 흐름과 유사한 새로운 데이터로 초록색 점이 나타나면, 파란색 classifier는 완전히 틀리게 된다.

**31**:

그래서 차라리 classifier는 차라리 곧은 초록색 선이 되는 게 낫다.
training set에 완벽하게 fit한 매우 구불구불한 복잡한 선이 되느니.
이는 머신러닝에서 매우 핵심이 되는 포인트이다.

**32~33**:

그래서 머신러닝에서는 모델이 test data에 잘 작동하도록 심플하게 만들기 위해 regularization[레귤러리제이션] 개념을 사용한다.

Loss function에 data loss에다가 레귤러리제이션 term도 추가한다.
그래서 classifier에게 training set에 fit하면서 최대한 심플한 W를 고르라고 말하는 것이다.

이러한 간단함에 대한 개념을 오컴의 면도날이라고 한다.
과학적 발견에서 적용되는 개념인데, 현상을 설명하는 여러 가지 가설 중에 하나를 골라야 한다면 간단한 가설이 새로운 현상을 일반화하기에 제일 좋다는 것이다.

머신러닝에서도 이러한 원리를 적용하여 loss function에 레귤러리제이션 페널티를 적용한다.

그래서 표준적인 loss function은 data loss와 regularization을 포함한다.
그리고 hyperparameter lambda는 두 가지에 대한 trade-off를 나타낸다.
람다가 클수록 regularization에 비중을 더 두게 된다.
람다는 hyperparameter이므로 모델을 훈련시킬 때 조정해야 하는 중요한 hyperparameter 중 하나이다.

**34**:

실제로 사용되는 regularization term에는 여러 가지 종류가 있다.

가장 흔히 사용되는 게 L2 regularization이다.
L2 regularization은 weight decay라고 부른다.
(벡터의 값이 골고루 비슷한 값을 갖는다)
L2 regularization은 weight vector에 유클리디안 거리를 사용하는 것이다.
미분을 쉽게 하기 위해 제곱을 하거나 1/2 제곱하기도 한다.
어쨌든 L2 regularization의 아이디어는 weight vector의 유클리디안 거리에 페널티를 부여하는 것이다.

L1 regularization은 weight vector의 L1 거리에 페널티를 부여한다.
L1 regularization을 쓰면 weight matrix W에 sparsity를 부여하는 효과가 있다.
(벡터의 값이 거의 0을 갖게 된다)

여러 가지 regularization이 있는데, 결국 모두 모델의 복잡도를 줄이기 위한 노력이다.


**35**:

L2 regularization의 예시를 보자.
linear classfication 관점에서 x와 w1 또는 x와 w2를 dot product하면 결과가 똑같이 나온다.
즉 w1이나 w2나 classifier에게 똑같은 w가 된다.
그러면 w1과 w2에 L2 regularization을 적용하면 무엇을 더 선호하게 될까?

L2는 w2를 더 선호한다.
L2를 적용하면 w2의 길이가 더 짧기 때문이다.
(각 원소의 제곱?)

L2를 적용하면 linear classifier는 weight decay가 되므로 모든 값이 골고루 분포된다.
즉 L1과 비교하면 L2를 적용한 W는 input 벡터와 dot product 될 때 모든 값이 살아있으므로 골고루 영향을 발휘하게 된다.
만약에 input x의 모든 값을 적절히 봐야 하는 경우라면 L2를 적용한 W가 더 robust할 것이다.
반대로 L1을 적용한 W는 input x의 특정 성분만 더욱 고려하게 된다.

왜냐하면 L1 regularization은 weight에 0이 많을수록 더 선호하기 때문이다.
즉 L1을 사용하면 sparse solution을 선호하게 된다.

+ L1: weight vector에 zero가 많으면 simple하다.
+ L2: weight vector에 값이 골고루 퍼져 있으면 simple하다.

**36**:

베이지안 이론을 적용하면 L2 regularization은 W에 가우시안 prior를 사용하는 MAP inference와 같은 결과를 만든다.

`~38:10`







**END**

---




---
