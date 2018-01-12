
ⓒ 2017 JMC

**NB Viewer** : http://nbviewer.jupyter.org/github/datalater/cs231n-2017/blob/master/L02-ImageClassification.ipynb

---

## L02 Image Classification

### 01 이미지 분류가 어려운 이유

**Semantic Gap**:

컴퓨터가 이미지 분류를 하기 어려운 이유는 semantic gap이 존재하기 때문이다.
이미지가 사람이 볼 때는 고양이지만, 컴퓨터가 볼 때는 픽셀값이기 때문에 '고양이'라는 label과 픽셀값 사이에는 커다란 갭이 존재한다.
이를 semantic gap이라고 한다.

### 02 이미지 분류를 하는 방법

**Not rule-based but Data-Driven Approach**:

+ (1) Collect a dataset of images and labels
+ (2) Use Machine Learning to train a classifier
+ (3) Evaluate the classifier on new images

```python
def train(images, labels):
# Machine learning!
return model
```

```python
def predict(model, test_images):
# Use model to predict labels
return test_labels
```

규칙을 하나하나 만들기에는 고려해야 할 요소가 너무나 많아서 통제가 불가능할 정도이다.
가능한 방법은 데이터에 기반한 접근법을 사용하는 것이다.
(1) 이미지와 레이블로 구성된 데이터셋을 모으고, (2) 머신러닝으로 classifier를 훈련시킨 다음, (3) 새로운 이미지에 대해 classifier를 적용시키는 3단계 방법을 사용하면 된다.

### 03 First classifier: Nearest Neighbor

**방법론**:

1. train에서 모든 데이터와 레이블을 암기한다.

```python
def train(images, labels):
# Machine learning!
return model
```

2. test 이미지와 가장 비슷한 training 이미지를 찾고, 해당 레이블로 예측한다.

```python
def predict(model, test_images):
# Use model to predict labels
return test_labels
```

**Distance Metric to compare images**:

L1 distance를 사용하여 pixel-wise absolute value differences를 모두 합한다.

**K-Nearest Neighbors**:

가장 가까운 training data 한 개의 label을 그대로 복사하는 것이 아니라, 가장 가까운 training data K개 중에서 다수결로 label을 결정하는 알고리즘이다.
K-NN을 쓰면 single nearest neighbor를 쓰는 것보다 decision boundary가 더 부드러워져서 일반화가 더 잘되고 따라서 더 나은 결과를 도출할 수 있다.

**Difference between Distance Metrics**:

+ L1 distance :
    + $d_1(I_1,I_2) = \Sigma_p{|I^{p}_{1}-I^{p}_{2}|}$
    + depends on your choice of coordnate system
+ L2 distance :
    + $d_2(I_1,I_2) = \sqrt{\Sigma_p{(I^{p}_{1}-I^{p}_{2})}^2}$
    + not depends on the coordinate system

L1 distance는 coordinate system, 즉 좌표에 따라 거리를 잰다.
가령, 4 by 4 grid가 있을 때, 대각선으로 가로지르지 않고 →←↑↓ 방향으로만 움직이고, 그 움직인 횟수가 곧 거리가 된다 (grid 위에서 움직이는 거리로 재기 때문에 block 단위로 설계된 도시 이름을 따서 Manhattan distance라고 부른다).
반면에 L2 distance는 좌표와 상관이 없고 대각선으로도 가로지를 수 있다(대각선 길이를 잴 때 사용하는 방법의 이름을 따서 Euclidean distance라고 부른다).
그러므로 대상을 회전시키면 L1 distance는 달라지지만 L2 distance는 달라지지 않는다.
따라서 벡터를 구성하는 첫 번째 요소와 두 번째 요소의 의미가 서로 다르다면, 즉 개별 요소(좌표축)의 의미가 포함되어 있는 좌표를 고려해야 한다면, L1 distance를 사용하도록 접근하는 게 더 일리 있다.
그러나 distance metric 같은 hyperparameter는 해결하려는 문제 상황과 데이터셋에 따라 최적의 값이 달라지므로, 실제로는 실험 결과가 가장 나은 것을 선택하는 것이 정답이다.

**hyperparameters**:

+ 후보1. 데이터에서 가장 좋은 결과값을 내는 hyperparameter set을 고른다.

그렇다면 hyperparameter는 어떻게 결정해야 할까?
쉽게 떠오르는 아이디어는 데이터에서 가장 좋은 결과값을 내는 hyperparameter를 고르는 것이다.
그러나 이는 옳지 않은 방법이며 절대로 해서는 안된다.
왜냐하면 training data에 overfitting 되기 때문이다.

+ 후보2. 데이터셋을 training set과 test set으로 나눈 후 training set에서 여러 가지 hyperparameter를 실험한 후, test set에 적용했을 때 가장 성능이 좋은 hyperparameter set을 고른다.

이러한 방법 또한 절대 해서는 안되는 끔찍한 방법이다.
test set의 핵심은 우리의 모델이 이전에 보지 못한 데이터에서 어떤 성능을 낼지에 대한 추정치를 알려준다는 점이다.
만약 여러 조합의 hyperparameter를 사용해서 알고리즘을 훈련시키고, test data에서 가장 성능이 좋은 최고의 hyperparameter를 고른다면, 그 특정 test set에서는 꽤 좋은 성능을 내는 hyperparameter라고 볼 수 있다.
그러나 그 특정 test set에 대한 성능이 이전에 보지 못한 실세계의 데이터에 대한 성능을 대표한다고 더 이상 볼 수 없다.
따라서 이 방법은 안 좋은 방법이며 해서는 안된다.

+ 정답. trainig set, validation set, test set으로 나눈 후, validation set에서 최적의 성능을 내는 hyperparameter set을 고른다.

더 나은 방법은 데이터를 3가지 셋으로 나누는 것이다.
대부분의 데이터를 training set에 할당하고, validation set과 test set을 만든다.
그 다음에는 다양한 조합의 hyperparameter를 사용해서 training set에서 훈련시키고, validation set에 대해 평가한 다음, validation set에서 가장 성능이 좋은 hyperparameter set을 고른다.
그 다음 디버깅을 포함하여 모델에 대한 모든 개발이 완료된 후에야 validation set에서 가장 성능이 좋은 모델을 test set에서 딱 한 번 돌린다.
그렇게 나온 수치들이 최종 페이퍼에 써야할 값이며, 당신이 만든 모델이 이전에 보지 못한 실세계의 데이터에 대해 어떤 성능을 낼지 알려주는 값이다.
실제로 validation set과 teste set을 엄격하게 분리하는 것이 매우 중요하다.
본 수업의 T.A.인 Justin Johnson 같은 경우에도 연구 논문 마감 일주일 전에 test set을 딱 한 번 사용한다.
test set을 엄격하게 관리하는 것이 정말로 매우 중요하다.

**K-NN on images never used**:

그러나 K-NN은 3가지 이유 때문에 이미지 분류에서 절대로 사용되지 않는다.
첫째, test time이 매우 느리다.
둘째, L1이든 L2이든 픽셀 간의 거리를 재는 것이 이미지끼리 의미적으로 유사한지 판단하는 유의미한 기준이 되지 못하기 때문이다 (vectorial distance functions do not correspond to perceptual similarity between images).

**K-NN and 'the curse of dimensionality'**:

셋째, 차원의 저주와 관련이 있다.
K-NN classifier가 어떻게 분류를 했는지 그 원리를 되돌아보면, 모든 training data point를 전체 공간에 뿌린 다음, 뿌려진 training data points를 이용해서 공간을 나누는 것이었다.
즉 K-NN이 잘 작동하려면 전체 공간을 조밀조밀하게 덮을 수 있는 training example이 필요하다.
그렇지 않고 training example이 없는 넓은 공간이 있다면 nearest neighbor가 너무 멀리 있게 되고 그렇다면 testing point와 유사하다고 볼 수 없기 때문이다.
따라서 문제는, K-NN이 잘 작동하기 위해 공간을 빽빽하게 덮으려면, 그만큼 많은 수의 training example이 필요하다는 것이고, 공간의 차원이 증가하면 필요한 training example의 수가 기하급수적으로 증가한다는 것이다.
필요한 데이터가 기하급수적으로 증가한다는 것은 매우 안 좋은 현상이다.
고차원 공간을 빽빽하게 채울 수 있는 충분한 데이터 수를 구하는 것은 기본적으로 거의 불가능하기 때문이다.
이러한 '차원의 저주'가 K-NN을 사용할 때 고려해야 할 점이다.

### 04 Linear Classification (inProgress)



One of the most basic building blocks that we'll see in different types of deep learning applications is this linear classifier. So I think it's actually really important to have a good understanding of what's happening with linear classification. Because these will end up generalizing quite nicely to whole neural networks. So another example of kind of this modular nature of neural networks comes from some research in our own lab on image captioning, just as a little bit of a preview. So here the setup is that we want to input an image and then output a descriptive sentence describing the image. And the way this kind of works is that we have one convolutional neural network that's looking at the image, and a recurrent neural network that knows about language. And we kind of just stick these two pieces together like Lego blocks and train the whole thing together and end up with a pretty cool system that can do some non-trivial things. And we'll work through the details of this model as we go forward in the class, but this just gives you the sense that, these deep learning networks are kind of like Legos and this linear classifier is kind of like the most basic building blocks of these giant networks. But that's a little bit too exciting for lecture two, so we have to go back to CIFAR-10 for the moment.

K-NN에서는 parameter가 없었다. parametric model은 training data의 지식을 요약한 후, 요약한 지식을 parameter에 집어 넣는다. 그래서 test할 때는 training data가 필요 없게 된다. parameter만으로 test할 수 있으므로 훨씬 더 효율적이고 computing power가 약한 모바일에서도 모델을 돌릴 수 있는 것이다.

parameter와 data를 결합하는 가장 쉬운 방법은 multiplication이며 이것이 바로 linear classifier가 하는 역할이다.

bias term은 training data와 직접 연산되지는 않지만 class 간의 independent preference를 부여한다. 만약 준비한 데이터셋에 개보다 고양이가 훨씬 더 많다면, 고양이에 대한 bias element 값은 개보다 더 클 것이다.

So the way that the linear classifier works is that we take this 2 by 2 image, we stretch it out into a column vector with 4 elements, and now, in this example, we are just restricting to 3 classes, cat, dog, and ship, because you can't fit 10 on a slide, and now our weight matrix is going to be 4 by 3, so we have 4 pixels and 3 classes. And now, again, we have a 3 element bias vector that gives us data independent bias terms for each category. Now we see that the cat score is going to be inner product between the pixels of our image and this row in the weight matrix added together with this bias term. So, when you look at it this way, you can kind of understand linear classification as almost a template matching approach. Where each of the rows in the matrix correspond to some template of the image. And now the inner product or dot product between the row of the matrix and the column giving the pixels of the image, computing this dot product kind of gives us a similarity between this template for the class and the pixels of our image. And then bias just, again, gives you the data independence scaling offset to each of the classes.


If we think about linear classificaiton from this viewpoint of template matching, we can actually take the rows of that weight matrix and unravel them back into images and actually visualize those templates as images. And this gives us some sense of what a linear classifier might actually be doing to try to understand our data. So, in this example, we've gone ahead and trained a linear classifier on our images. And now on the bottom we're visualizing what are those rows in that learned weight matrix corresponding to each of the 10 categories in CIFAR-10. And in this way we kind of get a sense of for what's going on in these images.


So the problem is that the linear classifier is only learning one template for each class. So if there's sort of variations in how that class might appear, it's trying to average out all those different variations, all those different appearances, and just one single template to recognize each of those categories. We can also see this pretty explicitly in the horse classifier. And then, if you look carefully, the horse actually seems to have maybe two heads, one head on each side. And I've never seen a horse with two heads. But the linear classifier is just doing the best that it can, because it's only allowed to learn one template per category. And as we move forward into neural networks and more complex models, we'll be able to achieve much better accuracy be cause they no longer have this restriction of just learning a single template per category.


Another viewpoint of the linear classifier is to go back to this idea of images as points and high dimensional space. And you can imagine that each of our images is something like a point in this high dimensional space. And now the linear classifier is putting in these linear decisions boundaries to try to draw linear separation between one category and the rest of the categories. So maybe up on the upper-left hand side we see these training examples of airplanes and throughout the process of training the linear classifier will go and try to draw this blue line to separate out with a single line the airplane class from all the rest of the classes. But when you think about linear classification in this way, from this high dimensional point of view, you can start to see again what are some of the problems that might come up with linear classification. And it's not too hard to construct examples of datasets where a linear classifier will totally fail.

**linear classification summary**:

+ At this point, we kind of talked about what is functional form corresponding to a linear classifier. And we've seen that this functional form of matrix vector multiply corresponds this idea of template matching and learning a single template for each category in your data. And then once we have this trained matrix, you can use it to actually go and get your scores for any new training example. But what we have not told you is how do you actually go about choosing the right W for you dataset. We've just talked about what is the functional form and what is going on with this thing.

**END**

---
---

## Quote

+ The Problem: Semantic Gap
    + This idea of a cat, or this label of a cat, is a semantic label that we're assuming to this image, and there's this huge gap between the semantic idea of a cat and these pixel values that the computer is actually seeing.
+ Challenges:
    + Viewpoint variation
    + Illumination
    + Deformation
    + Occlusion
    + Background Clutter
    + Intraclass variation

+ Not Rule-based but Data-Driven Approach
    + (1) Collect a dataset of images and labels
    + (2) Use Machine Learning to train a classifier
    + (3) Evaluate the classifier on new images

```python
def train(images, labels):
    # Machine learning!
    return model
```

```python
def predict(model, test_images):
    # Use model to predict labels
    return test_labels
```

+ With N examples, how much time do you expect to train and test using a nearest neighbor classifier?
    + training = O(1)
    + test = O(N)

+ decision regions of a nearest neighbor classifier

**L1 distance vs. L2 distance**

+ L1 distance, underneath this, this is actually a circle according to the L1 distance and it forms this square shape thing around the origin, where each of the points on this, on the square, is equidistant from the origin according to L1, whereas with the L2 or Euclidean distance then this circle is a familiar circle, it looks like what you'd expect.
+ So one interesting thing to point out between these two metrics in particulalr is that the L1 distance depends on your choice of coordinates system. So if you were to rotate the coordinate frame that would actually change the L1 distance between points. Whereas changing the coordinate frame in the L2 distance doesn't matter, it's the same thing no matter what what your coordinate frame is.
+ Maybe if your input features, if the individual entries in your vector have some important meaning for your task, then maybe somehow L1 might be a more natural fit. But if it's just a generic vector in some space and you don't know which of the different elements, you don't know what they actually mean, then maybe L2 is slightly more natural.
+ And another point here is that by using different distance metrics we can actually generalize the K-nearest neighbor classifier to many, many different types of data, not just vectors, not just images. So, for example, imagine you wanted to classify pieces of text, then the only thing you need to do, to use K-nearest neighbor, is to specify some distance function that can measure distances between maybe two paragraphs or two sentences or something like that.
+ So simply by specifying different distance metrics we can actually apply this algorithm very generally to basically any type of data. Even though it's a kind of simple algorithm, in general, it's a very good thing to try first when you're looking at a new problem.
+ So then, it's also kind of interesting to think about what is actually happening geometrically if we choose different distant metrics. So here we see the same set of points on the left using the L1, or Manhattan distance, and then, on the right, using the L2, or Euclidean distance. And you can see that the shapes of these decision boundaries actually change quite a bit between the two metrics. So when you're looking at L1 these decision boundaries tend to follow the coordinate axes. And this is again because the L1 depends on our choice of coordinate system. Where the L2 sort of doens't really care about the coordinate axis, it just puts the boundaries where they should fall naturally.
+ If you know that you have a vector, and maybe the individual elements of the vector have meaning. Like maybe you're classifying employees for some reason and then the different elements of that vector correspond to diffrent features or aspects of an employee. Like their salary or the number of years they've been working at the company or something like that. So I think when your individual elements actually have some meaning, is where I think maybe using L1 might make a little bit more sense.
+ But in general, again, this is a hyperparameter and it really depends on your problem and your data so the best answer is  just to try them both and see what works better.

**hyperparameter**

+ they are not necessarily learned from the training data, instead these are choies about your algorithm that you make ahead of time and there's no way to learn them directly from the data.
+ So, the question is how do you set these things in practice?
+ And they turn out to be very problem-dependent. And the simple thing that most people do is simply try different values of hyperparameters for you data and for your problem, and figure out which one works best.
+ Even this idea of trying out different values of hyperparameters and seeing what works best, there are many different choices here. What exactly does it mean to try hyperparameters and see what works best?
+ Well, the first idea you might think of is simply choosing the hyperparameters that give you the best accuracy or best performance on your training data.
+ This is actually a really terrible idea. You should never do this. In the concrete case of the nearest neighbor classifier, for example, if we set K=1, we will always classify the training data perfectly. So if we use this strategy we'all always pick K=1, but, as we saw from the examples earlier, in practice it seems that setting K eqauls to larger values might cause us to misclassify some of the training data, but, in fact, lead to better performance on points that were not in the training data. And ultimately in machine learning we don't care about fitting the training data, we really care about how our classifier, or how our method, will perform on unseen data after training. So this is a terrible idea, don't do this.
+ So another idea that you might think of, is maybe we'll take our full dataset and we'll split it into some training data and some test data. And now I'll try training my algorithm with different choices of hyperparameters on the training data and then I'll go and apply that trained classifier on the test data and now I will pick the set of hyperparameters that cause me to perform best on the test data. This seems like maybe a more reasonable strategy, but, in fact, this is also a terrible idea and you should never do this. Because, again, the point of machine learning systems is that we want to know how our algorithm will perform. So the point of the test set is to give us some estimate of how our method will do on unseen data that's coming out from the wild. And if we use this strategy of training many different algorithms with different hyperparameters, and then, selecting one which does the best on the test data, then, it's possible, that we may have just picked the right set of hyperparameters that cuased our algorithm to work quite well on this testing set, but now our performance on this test set will no longer be representative of our performance of new, unseen data. So, again, you should not do this, this is a bad idea, you'll get in trouble if you do this.
+ What is much more common, is to actually split your data into three different sets. You'll partition most of your data into a training set and then you'll create a validation set and a test set. And now what we typically do is go and train our algorithm with many different choices of hyperparameters on the training set, evaluate on the validation set, and now pick the set of hyperparameters which performs best on the validation set. And now, after you've done all your development, you've done all your debugging, after you've done everything, then you'd take the best performing classifier on the validation set and run it once on the test set. And now that's the number that goes into your paper, that's the number that goes into your report, that's the number that actually is telling you how your algorithm is doing on unseen data. And this is actually really really important that you keep a very strict separation between the validation data and the test data. So, for example, when we're working on research papers, we typically only touch the test set at the very last minute. So, when I'm writing papers, I tend to only touch the test set for my problem in maybe the week before the deadline or so to really insure that we're not being dishonest here and we're reporting a number which is unfair. So, this is super important and you want to make sure to keep your test data quite under control.
+ So another strategy for setting hyperparameters is called cross validation. And this is used a little bit more commonly for small data sets, not used so much in deep learning. So here the idea is we're going to take our test data, or we're going to take our dataset, as usual, hold out some test set to use at the very end, and now, for the rest of the data, rather than splitting it into single training and validation partition, instead, we can split our training data into many different folds. And now, in this way, we've cycled through choosing which fold is going to be the validation set. So now, in this example, we're using five fold cross validation, so you would train your algorithm with one set of of hyperparameters on the first four folds, evaluate the performance on fold four, and now go and retrain your algorithm on folds one, two, three and five, evaluate on fold four, and cycle through all the different folds. And, when you do it this way, you get much higher confidence about which hyperparameters are going to perform more robustly. So this is kind of the gold standard to use, but, in practice in deep learning when we're training large models, and training is very computationally expensive, these doesn't get used too much in practice.

**Test set can be representative**:

+ The question is, whether the test set, is it possible that the test set might not be representative of data out there in the wild?
+ This defintely can be a problem in practice, the underlying statistical assumption here is that your data are all independently and identically distributed, so that all of your data points should be drawn from the same underlying probability distribution. Of course, in practice, this might not always be the case, and you definitely can run into cases where the test set might not be super representative of what you see in the wild. So this is kind of problem that dataset creators and dataset curators need to think about. But when I'm creating datasets, for example, one thing I do, is I'll go and collect a whole bunch of data all at once, using the exact same methodology for collecting the data, and then afterwards you go and partition it randomly between train and test. One thing that can screw you up here is maybe if you're collecting data over time and you make earlier data, that you collect first, be the training data, and the later data that you collect be the test data, then you actually might run into this shift that coulud cause problems. But as long as this partition is random among your entire set of data points, then that's how we try to all alleviate this problem in practice.

**The Curse of Dimensionality**:

+ Another, sort of, problem with the k-nearest neighbor classifier has to do with something we call the curse of dimensionality. So, if you recall back this viewpoint we had of the k-nearest neighbor classifier, it's sort of dropping paint around each of the training data points and using that to sort of partition the space. So that means that if we expect the k-nearest neighbor classifier to work well, we kind of need our training examples to cover the space quite densely. Otherwise our nearest neighbors could actually be quite far away and might not actually be very similar to our testing points. And the problem is, that actually densely covering the space, means that we need a number of training examples, which is exponential in the dimension of the problem. So this is very bad, exponential growth is always bad, and basically, you're never going to get enough images to densely cover this space of pixels in this high dimensional space. So that's maybe another thing to keep in mind when you're thinking about using k-nearest neighbor.

**Linear Classification**:

+ Linear classification is, again, quite a simple learning algorithm, but this will become super important and help us build up to whole neural networks, and whole convolutional networks. So, one analogy people often talk about when working with neural networks is we think of them as being kind of like Lego blocks. That you can have different kinds of components of neural networks and you can stick these components together to build these large different towers of convolutional networks. One of the most basic building blocks that we'll see in different types of deep learning applications is this linear classifier. So I think it's actually really important to have a good understanding of what's happening with linear classification. Because these will end up generalizing quite nicely to whole neural networks. So another example of kind of this modular nature of neural networks comes from some research in our own lab on image captioning, just as a little bit of a preview. So here the setup is that we want to input an image and then output a descriptive sentence describing the image. And the way this kind of works is that we have one convolutional neural network that's looking at the image, and a recurrent neural network that knows about language. And we kind of just stick these two pieces together like Lego blocks and train the whole thing together and end up with a pretty cool system that can do some non-trivial things. And we'll work through the details of this model as we go forward in the class, but this just gives you the sense that, these deep learning networks are kind of like Legos and this linear classifier is kind of like the most basic building blocks of these giant networks. But that's a little bit too exciting for lecture two, so we have to go back to CIFAR-10 for the moment.
+ In linear classificaiton, we're going to take a bit of a different approach from k-nearest neighbor. So, the linear classifier is one of the simplest examples of what we call a parametric model. So now, our parametric model actually has two different components. It's going to take in this image, maybe, of a cat on the left, and this, that we usually write as X for our input data, and also a set of parameters, or weights, which is usaually called W, also sometimes theta, depending on the literature. And now,  we're going to write down some function which takes in both the data, X, and the parameters, W, and this'll spit out now 10 numbers describing what are the scores corresponding to each of these 10 categories in CIFAR-10. With the interpretation that, like the larger score for cat, indicates a large probability of that input X being cat.
+ So, in the k-nearest neighbor setup, there was no parameters, instead, we just kind of keep around the whole training data, the whole training set, and use that at test time. But now, in a parametric approach, we're going to summarize our knowledge of the training data and stick all that knowledge into these parameters, W. And now, at test time, we no longer need the actual training data, we can throw it away. We only need these parameters, W, at test time. So this allows our models to now be more efficient and actually run on maybe small devices like phones. | K-NN에서는 parameter가 없었다. parametric model은 training data의 지식을 요약한 후, 요약한 지식을 parameter에 집어 넣는다. 그래서 test할 때는 training data가 필요 없게 된다. parameter만으로 test할 수 있으므로 훨씬 더 효율적이고 computing power가 약한 모바일에서도 모델을 돌릴 수 있는 것이다.
+ So, kind of, the whole story in deeep learning is coming up with the right structure for this function, F. You can imagine writing down different functional forms for how to combine weights and data in different complex ways, and these could correspond to different network architectures. But the simplest possible example of combining these two things is just, maybe, to multiply them. And this is a linear classifier. | parameter와 data를 결합하는 가장 쉬운 방법은 multiplication이며 이것이 바로 linear classifier가 하는 역할이다.
+ Also sometimes, you'll typically see this, we'll often add a bias term which will be a constant vector of 10 elements that does not interact with the training data, and instead just gives us some sort of data independent preferences for some classes over another. So you might imagine that if your datset was unbalanced and had many more cats than dogs, for example, then the bias elements corresponding to cat would be higher than the other ones. | bias term은 training data와 직접 연산되지는 않지만 class 간의 independent preference를 부여한다. 만약 준비한 데이터셋에 개보다 고양이가 훨씬 더 많다면, 고양이에 대한 bias element 값은 개보다 더 클 것이다.
+ So the way that the linear classifier works is that we take this 2 by 2 image, we stretch it out into a column vector with 4 elements, and now, in this example, we are just restricting to 3 classes, cat, dog, and ship, because you can't fit 10 on a slide, and now our weight matrix is going to be 4 by 3, so we have 4 pixels and 3 classes. And now, again, we have a 3 element bias vector that gives us data independent bias terms for each category. Now we see that the cat score is going to be inner product between the pixels of our image and this row in the weight matrix added together with this bias term. So, when you look at it this way, you can kind of understand linear classification as almost a template matching approach. Where each of the rows in the matrix correspond to some template of the image. And now the inner product or dot product between the row of the matrix and the column giving the pixels of the image, computing this dot product kind of gives us a similarity between this template for the class and the pixels of our image. And then bias just, again, gives you the data independence scaling offset to each of the classes.
+ If we think about linear classificaiton from this viewpoint of template matching, we can actually take the rows of that weight matrix and unravel them back into images and actually visualize those templates as images. And this gives us some sense of what a linear classifier might actually be doing to try to understand our data. So, in this example, we've gone ahead and trained a linear classifier on our images. And now on the bottom we're visualizing what are those rows in that learned weight matrix corresponding to each of the 10 categories in CIFAR-10. And in this way we kind of get a sense of for what's going on in these images.
+ So the problem is that the linear classifier is only learning one template for each class. So if there's sort of variations in how that class might appear, it's trying to average out all those different variations, all those different appearances, and just one single template to recognize each of those categories. We can also see this pretty explicitly in the horse classifier. And then, if you look carefully, the horse actually seems to have maybe two heads, one head on each side. And I've never seen a horse with two heads. But the linear classifier is just doing the best that it can, because it's only allowed to learn one template per category. And as we move forward into neural networks and more complex models, we'll be able to achieve much better accuracy be cause they no longer have this restriction of just learning a single template per category.
+ Another viewpoint of the linear classifier is to go back to this idea of images as points and high dimensional space. And you can imagine that each of our images is something like a point in this high dimensional space. And now the linear classifier is putting in these linear decisions boundaries to try to draw linear separation between one category and the rest of the categories. So maybe up on the upper-left hand side we see these training examples of airplanes and throughout the process of training the linear classifier will go and try to draw this blue line to separate out with a single line the airplane class from all the rest of the classes. But when you think about linear classification in this way, from this high dimensional point of view, you can start to see again what are some of the problems that might come up with linear classification. And it's not too hard to construct examples of datasets where a linear classifier will totally fail.

**linear classification summary**:

+ At this point, we kind of talked about what is functional form corresponding to a linear classifier. And we've seen that this functional form of matrix vector multiply corresponds this idea of template matching and learning a single template for each category in your data. And then once we have this trained matrix, you can use it to actually go and get your scores for any new training example. But what we have not told you is how do you actually go about choosing the right W for you dataset. We've just talked about what is the functional form and what is going on with this thing.

**END**

---




---
