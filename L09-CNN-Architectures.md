ⓒ 2018 JMC

**NB Viewer** :

http://nbviewer.jupyter.org/github/datalater/cs231n-2017/blob/master/L07-TrainingNN2.ipynb

---

## 20180303 NEXT


## 20180303 Study Notes




**END**

---

## L09 CNN Architectures

**26**:

+ VGG : much deeper networks, much smaller filters

훨씬 더 깊은 신경망을 구성하고 훠린 더 작은 사이즈의 필터를 사용한다.

AlexNet이 8개의 layer을 사용한 반면 VGG는 16개부터 19개까지 layer를 증가시킨다.
특징적인 점은 모든 conv layer는 3x3 필터를 사용한다는 것이다.
3x3 필터를 사용한다는 것은 그만큼 주변 픽셀을 아주 조금만 살펴본다는 것이다.

구조는 매우 간단한데 3x3 conv layer를 2~3개씩 쌓고 그 사이마다 pooling layer가 들어가 있다.
이미지넷 챌린지에서 top-5 error rate가 7.3%에 불과한 좋은 성적을 거뒀다.

**27**:

+ VGG가 smaller filter를 사용하는 이유? [비슷한 receptive field를 가지면서 파라미터 수가 더 적고 비선형성을 더 많이 줄 수 있는 깊은 신경망이기 때문에]

우선 3x3 필터를 사용하면 파라미터 수가 적어진다.
그래서 더 큰 사이즈의 필터를 사용하는 것보다 3x3 conv layer를 여러 개 깊게 쌓을 수 있는 이점이 있다.
결국 3x3 conv layer를 여러 개 쌓는 것은 7x7 conv layer 하나를 사용하는 것과 똑같은 effective receptive field를 갖는 것과 같은 효과가 있다.

+ 3x3 conv를 3개 사용했을 때 effective receptive field는 얼마일까? [7x7]

첫 번째 레이어의 receptive field는 3x3이다.
두 번째 레이어의 뉴런 하나는 첫 번째 레이어 아웃풋의 3x3을 보게 된다.
그런데 첫 번째 레이어 아웃풋의 각 뉴런 하나는 3x3을 본 것이므로 original input 기준으로는 두 번째 레이어의 뉴런이 original input을 5x5로 보고 있는 것이다.
같은 방식으로 세 번째 레이어의 뉴런 하나는 original input을 7x7로 보는 것이다.
따라서 정답은 [7x7]이 된다.

결국 VGG가 매우 작은 사이즈인 3x3 필터를 사용하는 conv layer를 3번 쌓으면 7x7 필터를 1번 사용하는 것과 receptive field는 동일하지만 신경망 관점에서 볼 때 훨씬 더 깊고, non-linearity를 더 많이 부여할 수 있다.
또한 파라미터의 수 또한 적다.
왜냐하면 3x3 conv layer를 3번 사용하면 각 레이어마다 파라미터 수는 3x3에다가 input depth C와 채널 수를 똑같이 보존한다는 가정 하(if we're going to preserve the total number of channels)에 output feature map의 수 C를 곱하면 3x3xCxC가 된다.
이러한 레이어가 3개 있으므로 총 파라미터 수는 3x(3x3xCxC)가 된다.
반면 7x7 conv layer를 1번 사용하면 7x7xCxC가 된다.
따라서 파라미터 수는 3x3을 3번 사용하는 게 훨씬 더 적다.

**31-32**:

Total memory and Total params

한 번 forward pass 하는데 드는 메모리 용량이 거의 100MB이다.
한 이미지당 100MB 메모리를 사용하므로 메모리를 꽤 많이 사용하는 모델이다.

전체 파라미터 수는 138 million으로 AlexNet 60 million 보다 2배 더 많다.

**33-34**:

메모리가 사용된 분포를 보면 초기 레이어에서 많이 사용되었음을 알 수 있다.
왜냐하면 초기 레이어일수록 input의 많은 공간 영역을 커버해야 하기 때문이다.

반면 파라미터가 사용된 분포를 보면 마지막 레이어에서 많이 사용되고 있음을 알 수 있다.
Fully Connected Layer는 엄청난 양의 파라미터를 사용한다.
왜냐하면 dense connection을 사용하기 때문이다.

그래서 이후에는 fc layer를 제거해서 파라미터 수를 확연하게 줄어들게 만든 모델을 보게 될 것이다.

**35**:

VGG는 2014년 이미지넷 챌린지에서 분류에서는 2등, localization에서는 1등을 했다.

> **Note**: localization : 이미지를 단순히 분류하는 것뿐만 아니라 해당 물체가 있는 곳에 bouding box를 그리는 것. detection은 여러 물체가 있지만 localiation에서는 한 가지 종류의 물체만 있다고 가정한다.

AlexNet과 비슷한 훈련 과정을 거쳤다.
Local Response Normalization (LRN)은 별로 도움 되지 않아서 쓰지 않았다.

실제로 VGG-16보다 VGG-19가 성능이 약간 더 좋지만 메모리 사용량이 더 많으므로 어느 것을 사용해도 괜찮고 VGG-16이 더 흔하게 쓰인다.

최상의 결과를 얻으려면 AlexNet이 한 것처럼 여러 모델을 평균 내는 모델 앙상블을 사용한다.

출력층 바로 이전 레이어인 FC7 레이어는 좋은 feature representation을 보인다.
그리고 FC7 레이어를 사용하면 다른 태스크에서도 일반화가 잘 된다.
따라서 다른 데이터에서 feature를 뽑아낼 때도 사용될 수 있다.











---






---

**END**
