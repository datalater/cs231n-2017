ⓒ 2018 JMC

**NB Viewer** :

http://nbviewer.jupyter.org/github/datalater/cs231n-2017/blob/master/L07-TrainingNN2.ipynb

---

## 20180210 NEXT

+ TensorFlow : 조민철

## 20180210 Study Notes

**25**:

Nesterov에서 gradient는 일종의 correction 기능을 한다.

**27**:

correction

**32**:

RMSProp : decay_rate

(사진)

**34**:



**END**

---

## L07 Training Neural Networks Part II

**9: Data Preprocessing이 중요한 이유**

normalization을 하지 않으면 classification loss가 weight matrix의 변화에 따라 매우 민감해져서 optimize 하기 힘들어집니다.

왼쪽 그림과 오른쪽 그림 모두 linear classifier가 origin을 지납니다.
그런데 왼쪽 그림은 데이터가 normalized 되지 않아서 origin으로부터 멀리 떨어져 있습니다.
이때 linear classifier (weight matrix)가 origin을 기준으로 살짝만 움직여도 끝부분은 훨씬 더 크게 움직이기 때문에 classification loss가 매우 크게 변화합니다.
반면, 오른쪽 그림처럼 데이터가 normalized 되어서 평균이 0이라면 linear classifier가 origin을 기준으로 살짝 움직여도 classification loss가 크게 달라지지 않습니다.

loss가 parameter vector에 매우 민감하게 달라지면 learning이 힘들어집니다.

그리고 이는 linear classifer에만 해당되는 것이 아닙니다.
신경망에서도 layer의 input이 normalized 되지 않으면 weight matrix가 약간만 달라져도 layer의 output이 크게 달라지게 됩니다.
그러면 learning이 힘들어집니다.

**10: batch normalization**

normalization이 중요하기 때문에 batch normalization에 대해서도 이야기했습니다.

batch normalization은 신경망 안에 additional layer를 추가해서 layer의 output(intermediate activation)이 zero mean과 unit variance가 되도록 강제로 만드는 것입니다.

batch normalization은, forward pass를 할 때 우리는 mini batch를 사용해서 평균 및 표준편차를 추정하며, 그 추정치를 이용해서 forward pass 되는 데이터를 normalize 합니다.

그리고 scale과 shift parameter를 사용해서 layer의 expressivity(표현력)을 증가시킵니다.

**11: loss 및 accuracy graph 활용법**

epoch에 따라 loss가 줄어들고 있다면 신경망이 잘 작동하고 있는 겁니다.
그런데 만약 train set과 validation set의 정확도를 찍어봤는데 train은 계속 증가하는 반면 validation은 어느 시점부터 정체되어 있다면 overfitting을 의심해야 하며 additional regularization을 사용해야 할지도 모릅니다.

**12: hyperparameter search**

grid search와 random search를 비교했었습니다.
그리고 이론적으로는 random search가 더 나을 수도 있다고 말했었죠.
왜냐하면 모델의 성능이 특정 hyperparameter에 유난히 민감한 상황이라면 random search를 사용하는 게 전체 hyperparameter space를 더 잘 커버할 수 있기 때문입니다.

그리고 coarse to fine search에 대해서도 말했습니다.
hyperparameter optimization을 할 때 처음에는 hyperparameter의 범위를 매우 크게 잡았다가 몇 회 train을 반복한 이후에는 train 결과를 보고 좋은 hyperparameter의 범위를 좁힐 수 있습니다.
이런 식으로 계속 반복하여 점점 hyperparameter 범위를 좁히는 것이죠.
중요한 것은 처음 시작할 때 매우 넓은 범위를 잡아야 한다는 겁니다.
그래야 가능한 공간을 최대한 많이 탐색하게 되기 때문이죠.

**13: Today**

대부분의 사람들이 요즘 사용하는 optimization 알고리즘이 무엇인지 자세히 얘기해보겠습니다.

그리고 이전 강의에서 잠깐 소개했었던 regularization도 다루겠습니다.
regularization은 train error와 test error의 gap을 줄이기 위한 목적을 갖고 있습니다.

그 다음은 transfer learning입니다.

**14: Optimization**

신경망 훈련에 있어서 핵심 전략은 optimization입니다.

**15-16: Problems with SGD (1)**

SGD를 사용하면 한 가지 문제가 있습니다.

만약 loss가 수평 방향으로는 매우 느리게 변화하고 수직 방향으로는 매우 민감하게 변화하는 상황에 맞닥뜨렸다고 합시다.

참고로 이러한 상황을 loss function이 high condition number(Hessian matrix의 signular value에 대한 smallest와 largest의 ratio)를 갖고 있다고 말합니다.

> **Note**:  high condition number problem

loss function의 landscape가 taco shell처럼 오른쪽으로 길게 찌그러진 타원형인 거죠.

taco shell 같은 loss function에서 SGD는 지그재그 패턴을 보입니다.
tao shell 같은 모양의 loss function에서는 gradient의 방향이 최소값 방향으로 정렬되지 않기 때문입니다.

When you compute the gradient and take a step, you might step sort of over this line and sort of zigzag back and forth.
In effect, you get very slow progress along the horizontal dimension, which is the less sensitive dimension, and you get this zigzagging, nasty, nasty zigzagging behavior across the fast-changing dimension.
This is undesirable behavior.

특히 이러한 문제는 고차원일수록 흔하게 나타납니다.

이 슬라이드의 그림은 w가 2개일 때인데 보통 신경망은 parameter가 수백만에서 수천만 개입니다.
즉 SGD가 움직일 수 있는 방향이 수백만 개인거죠.
서로 다른 수백만 개의 방향이 있는 상태에서 만약 smallest gradient 방향과 largest gradient 방향이 매우 차이가 크다면, SGD는 나이스하게 움직이질 못합니다.

즉 실제로 고차원 문제에서는 SGD 잘 작동하지 못하는 심각한 위험이 있는 겁니다.

**17-19: Problems with SGD (2) Saddle point**

SGD의 또 다른 문제점은 local minima 또는 saddle point입니다.

local minima는 global minima보다 loss가 크지만 gradient가 0가 되는 위치를 말합니다.

saddle point는 어느 방향에서는 loss가 증가하지만 다른 방향에서는 loss가 감소하는 위치를 의미합니다.

예를 들어보죠.
우측에서 좌측으로 내려오는 두 개의 곡선이 있고 둘 다 중간에 푹 들어가는 계곡이 있습니다.
그래프의 x축은 parameter이고 y축은 loss 값입니다.
SGD가 이 두 곡선에서 어떻게 작동할까요?

SGD는 local minima인 계곡에서 멈출 것입니다.
왜냐하면 계곡은 gradient가 0이기 때문이죠.

그러나 그림은 1차원일 경우를 묘사한 것이고 실제 우리가 다루게 되는 고차원에서는 saddle point가 큰 문제가 되지 않습니다.

예를 들어 parameter가 1억 개 있다고 해봅시다.
local mimima는 1가지 방향일 때 gradient=0이 되지만, 나머지 방향도 모두 gradient=0이 되기는 쉽지 않습니다.
즉 이러한 지점은 saddle point이죠.
saddle point는 어떤 뜻인가요?
현재 지점에서 특정 방향으로는 loss가 증가하고 다른 방향으로는 loss가 감소한다는 뜻입니다.
parameter가 1억 개 있다면 실제로 saddle point는 거의 모든 위치에서 발생할 것입니다.

만약 그 지점이 local minima라면 움직일 수 있는 방향이 1억 개가 되더라도 어느 방향으로 움직여도 loss가 증가한다는 것입니다.
실제로 고차원이라면 이러한 local minima는 매우 매우 드물 것이고 즉 문제가 될 가능성은 거의 없다는 뜻이 되겠죠.

즉 실제 신경망에서 맞닥뜨리게 되는 문제는 saddle point이지 local minima가 아닌 겁니다.

그리고 문제는 saddle point 딱 그 지점뿐만 아니라 saddle point 부근에서도 발생합니다.

parameter value가 saddle point 근처에 오면 gradient가 매우 작기 때문에 optimization의 속도가 매우 작아집니다.

이건 매우 큰 문제죠.

**20 Problems with SGD (3) stochastic**

SGD의 또 다른 문제점은 이름에 포함된 S 때문입니다.

SGD는 stochastic gradient descent의 약자입니다.
loss function은 일반적으로 수많은 데이터 example에 대한 loss를 계산해서 정의됩니다.

그런데 training example의 개수인 N이 백만 개 된다고 해봅시다.
N이 클수록 loss를 매번 계산하는 것은 매우 expensive할 겁니다.

그래서 우리는 example의 small mini batch를 사용해서 loss를 추정하고, gradient를 추정합니다.
즉, 추정치라는 것은 true information이 아니라는 거죠.
우리는 true information 대신에 노이즈가 섞인 추정치를 사용합니다.

gradient 추정치에 노이즈가 섞이면 optimization의 방향은 구불구불하게 움직일 것이고 minima에 도달하기 가지 오랜 시간이 걸리게 됩니다.

그래서 이러한 문제점을 해결하고자 하는 아이디어가 여러 가지 나왔습니다.

**21: SGD + Momentum**

It's very simple.
The idea is that we maintain a velocity over time, and we add our gradient estimates to the velocity.
And then we step in the direction of the velocity, rather than stepping in the direction of the gradient.

We also have this hyperparameter rho now, which corresponds to friction.

Now at every time step, we teake our current velocity, we decay the current velocity by the firction constant, rho, which is often something high, like .9 is a common choice.

We take our current velocity, we decay it by friction and we add in our gradient.

Now we step in the direction of our velocity vector, rather than the direction of our raw gradient vector.

**22: SGD+Momentum helps for local minima, saddle point, gradient noise**

If you think about what happens at local minima or saddle points, then if we're imagining velocity in this system, then you kind of have this physical interpretation of this ball kind of rolling down the hill, picking up speed as it comes down.

**Now once we have velocity, then even when we pass the point of local minima, the point will still have velocity, even if doesn't have gradient.
So then we can hopefully get over this local minima and continue downward**.

**There's this similar intuition near saddle points, where even though the gradient around the saddle point is very small, we have this velocity vector that we've built up as we roll downhill.
That can hopefully carry us through the saddle point and let us continue rolling all the way down**.

If you think about what happens in poor conditioning, now if we were to have these kind of zigzagging approximations to the gradient, then those zigzags will hopefully cancel each other out pretty fast once we're using momentum.
**This well effectively reduce the amount by which we step in the sensitive direction, whereas in the horizontal direction, our velocity will just keep building up, and will actually accelerate our descent across that less sensitive dimension**.

So adding momentum here can actually help us with this high condition number problem, as well.

Finally on the right, we've repeated the same visualization of gradient descent with noise.
Here, the black is this vanilla SGD, which is sort of zigzagging all over the place, where the blue line is showing now SGD with momentum.
**You can see that because we're adding it, we're building up this velocity over time, the noise kind of gets averaged out in our gradient estimates**.

Now SGD ends up taking a much smoother path towards the minima, compared with the SGD, which is kind of meandering due to noise.

**23: SGD + Momentum - Original picture**

The red is our current point.
At current point, we have some red vector, which is the direction of gradient, or rather our estimate of the gradient at the current point.

Green is now the direction of our velocity vector.
Now when we do the momentum update, we're actually stepping according to a weighted average of these two.
This helps overcome some noise in our gradient estimate.

**24: SGD + Momentum - Nesterov accelerated gradient**

There's slight variation of momentum that you sometimes see, called Nesterov accelerated gradient, also sometimes called Nesterov momentum.

That switches up this order of things a little bit.
In sort of normal SGD momentum, we imagine that we estimate the gradient at our current point, and then take a mix of our velocity and our gradient.

With Nesterov accelerated gradient, you do something a little bit different.
Here, you start at the red point.
You step in the direction of where the velocity would take you.
You evaluate the gradient at that point.
Then you go back to your original point, and kind of mix together those two.

This is kind of a funny interpretation, but you can imagine that you're kind of mixing together information a little bit more.
If your velocity direction was actually a little bit wrong, it lets you incorporate gradient information from a little bit larger parts of the objective landscape.

This also has some really nice theoretical properties when it comes to convex optimization, but those guarantees go a little bit out the windwo once it comes to non-convex problems like neural networks.

**25: SGD + Momentum - Nesterov Momentum equation**

Writing it down in equations, Nesterov momentum looks something like this, where now to update our velocity, we take a step, according to our previous velocity, and evaluate that gradient there.

Now when we take our next step, we actually step in the direction of our velocity that's incorporating information from these multiple points.

`@@@resume:` https://youtu.be/_JB0AO7QxSA?t=30m2s


---






---

**END**
