ⓒ 2018 JMC

**NB Viewer** :

http://nbviewer.jupyter.org/github/datalater/cs231n-2017/blob/master/L05-CNN.ipynb

---

## NEXT

+ L07 앞 부분 - optimization 수학적 부분
+ L07 뒷 부분 - 드랍아웃, 앙상블

## 20180203 Study Notes

**시각세포**:

+ VGG-Net은 gray scale로 보는데 이게 인간의 뇌가 시각을 인식하는 것을 묘사한 것이다.

인간의 뇌 세포

+ rod cell : 흑백 판단
+ ganglion cell : 특정 시야 판단 (receptive field)
+ 신호가 오면 noise를 제거하기 위해 최대 신호만 뽑는다. 최대 신호만 뽑는 게 max-pooling이다. CNN의 연산량이 FC에서 90% 발생한다. FC를 빼면 연산량이 많지 않다.

**16-17**:

+ sigmoid : historically popular, 그러나 기울기가 0이 되는 saturation problem이 있다.

**19**:

+ sigmoid : not zero-centered. 항상 양수가 되므로 문제가 된다.

**21**:

+ 항상 양수가 되면 지그재그로만 쫓아갈 수밖에 없다.
+ df/dw로 미분했을 때 f가 양수이면 w가 1사분면 방향으로 이동할 수 있고, f가 음수이면 w가 3사분면 방향으로이동할 수밖에 없어서 zig zag 패턴이 생긴다.

**23**:

+ 학습이 엄청 오래 걸린다.

**24**:

+ ReLU는 생물학적으로 sigmoid 보다 더 유사하다. 또 학습을 해보면 더 빠르게 수렴이 된다.

**27**:

+ relu layer에 x가 0이하면 아예 activate 되지도 않고 update 되지도 않는다.

**29**:

+ relu에서 x가 0이하일 때도 살려야 할 때가 있는데 그게 GAN이다.

**31**:

+ ELU: computation이 좀 오래 걸린다.

**32**:

+ Maxout : 네트워크를 2개로 구성한다.

**33**:

+ Too Long Don Read: 그냥 relu 써라.

**34-35**:

+ Data Preprocessing : 정규화를 안 하면 수렴 속도가 느려진다.

**37**:

+ 컴퓨터 공학 관점에서 데이터를 정규화하지 않으면 iteration이 많아질 때 반드시 overflow가 발생한다.

**39**:

+ 짧게 얘기하면 AlexNet에서는 평균값을 뺐는데 VGGNet에서는 채널의 평균을 뺐다. 그러면 그레이스케일이 된다.

**41**:

+ 대칭을 깨는 게 우리의 목적이다. 그래서 weight를 random으로 initialization을 한다.

**45**:

+ 기껏 input을 정규분포로 만들었는데 layer를 거치면서 정규분포가 깨져버리게 된다.

**47**:

+ saturated : gradient가 0이 된다.

**48**:

+ Xavier initialization : 현명한 초기화 방법.

**50**:

+ Xavier 쓰면 반만 남으므로 표준편차가 더 커지게 만든다. 편차가 잘 퍼지게 만든다.
+ 그림의 의미: 뉴런이 많이 죽는다.

**53**:

+ Batch Normalization :

**55**:

+ row: sample
+ column을 정규화한다.

**56**:

+ BN을 activation 앞에 두자. 왜냐하면 평균을 0으로 분산을 1로 만들어주기 때문에 0 사이에서는 수렴이 잘되기 때문이다.

**58**:

+ 스케일링. 쉬프트.
+ 기존 x로 되돌릴 수 있다.
+ batch normalization이 뉴럴넷에서 좋을 수도 나쁠 수도 있다. 감마와 베타는 학습이 가능한 값이다. 만약 BN을 적용한게 뉴럴넷에 나쁘면 감마와 베타를 넣어서 BN을 쓰지 않도록 다시 기존의 x로 돌아갈 수 있게 한다. 이렇게 감마와 베타를 넣어서 뉴럴넷에 좋거나 나쁠 때를 판단하여 좋은 방향으로 사용할 수 있게 유동성을 주는 것이다.

**59**:

+ 엡실론이 안 들어가면 sigma가 0일 때 바로 overflow가 발생하다. 그래서 sigma가 0이 되더라도 강제로 값이 나오게 하는 일종의 프로그래밍 팁이다.

---

**62**:

+ (1) 데이터가 있으면 전처리를 먼저 하자.

**63**:

+ (2) 그리고 아키텍쳐를 선택하자.

**64**:

+ loss가 줄어드는지 체크해보자.

**66**:

+ overfit이 되는지 안 되는지 잘 작동하는지 확인해보자.

**71**:

+ loss가 안 바뀌면 어떡할까.
+ learning rate를 키워봐라.

**80**:

+ 랜덤 서치 vs. 그리드 서치
+ 랜덤이 더 좋다.

**86**:

+ gap이 없으면 잘 안되고 있는 것이다.
+ training accuracy가 0.9 이상이 안되면 모델을 폐기해야 한다. 비즈니스를 할 수 없기 때문이다.

**88**:

+ 데이터 전처리: 이미지 채널별로 mean을 빼라. 우리 실제 눈이 그렇게 되어있기 때문에.

**발표자님의 강의 추천**:

+ Sanfrancisco - Practical Deep Learning 최고의 강의. CS231n 보다 더 좋다.

**퍼실 보충**:

+ 모두연노트에 2018.2.3. 필기함.


**END**

---

## L06 Training Neural Networks Part I



---

**END**
