
ⓒ 2018 JMC

---

# MNIST tutorial using TF

## TensorFlow의 기본 원리

+ (1) 그래프 준비하기 (construction phase)
+ (2) 그래프 실행하기 (execution phase)

## 00 기본 세팅

**모듈 임포트**

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
```

**tf 버전확인**

```python
print(tf.__version__) # 1.3.0
```

**mnist 데이터 다운로드**

```python
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
```

다운로드 파일

+ train-images-idx3-ubyte.gz: training set images
+ train-labels-idx1-ubyte.gz: training set labels
+ t10k-images-idx3-ubyte.gz: test set images
+ t10k-labels-idx1-ubyte.gz: test set labels

개별 이미지 파일

```python
print(len(mnist.train.images[0])) # 784
print(len(mnist.train.labels[0])) # 10
```

+ images : 28x28x1 = 784차원
+ labels : 10차원

---

## Part I. 그래프 준비하기 - softmax classifier

**01 placeholder**

```python
X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
```

**02 predictions**

```python
W = tf.Variable(tf.random_normal(shape=[784, 10]))
b = tf.Variable(tf.random_normal(shape=[10]))

predictions = tf.matmul(X, W) + b
```

## Part I. 그래프 준비하기 - Neural Networks

**02 predictions**

```python
W1 = tf.Variable(tf.random_normal(shape=[784, 256]))
b1 = tf.Variable(tf.random_normal(shape=[256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([256, 256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([256, 10]))
b3 = tf.Variable(tf.random_normal([10]))

predictions = tf.matmul(L2, W3) + b3
```

---

**03 cost function and optimizer**

```python
cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
```

**04 accuracy**

```python
is_correct = tf.equal(tf.argmax(predictions , axis=1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
```

## Part II. 그래프 실행하기

**05 training epoch/batch**

```python
epochs = 15
batch_size = 100

total_batch = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        avg_cost = 0

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print("Epoch:", "%04d" % (epoch + 1), "cost =", "{:.9f}".format(avg_cost))
```

+ **epoch={k}** : 파라미터 업데이트에 전체 데이터가 온전히 사용된 횟수가 {k}번이다.
+ **batch size={k}** : 한 번 파라미터 업데이트 할 때마다 사용되는 데이터 개수가 {k}개이다.
+ **number of iterations={k}** : 파라미터 업데이트 횟수가 {k}번이다.
+ 전체 데이터가 1000개이고 배치 사이즈가 {500}이라면 iteration을 {2}번하면 {1} epoch가 된다.

**06 test accuracy**

```python
print("Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
```


---
