
ⓒ JMC 2017

- [CS231N Video - Playlist](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)  
- [CS231N Notes - Backpropgation](http://cs231n.github.io/optimization-2/)

---

## Backpropagation

### I. Introduction

**Problem statement**:

우리가 풀어야 할 문제는 함수 $f(x)$에서 input $x$가 함수값 $f$에 미치는 영향을 구하는 것이다.
함수값에 대한 변수의 영향력을 gradient라고 하며 $\triangledown f(x)$라고 표기한다.

+ function: $f(x)$
+ input: $x$
+ gradient of $f$ at $x$: $\triangledown f(x)$

**Motivation**:

성능이 뛰어난 Neural Network 모델을 만들려면 loss를 최소화하는 parameter를 구해야 한다.
parameter를 어떻게 변화시켜야 loss가 줄어들까.
parameter값이 loss function에 미치는 영향(=변화율)인 gradient를 구하면 해결할 수 있다.

+ function: $f(x) = L$(loss function)
+ input: $x$
  1. training data: $(x_i, y_i)$
  2. weights: $W, b$
+ gradient:
  1. gradient of $f$ at training data $(x_i, y_i)$ = $\triangledown L(x_i, y_i)$
  2. gradient of $f$ at weights $(W, b)$ = $\triangledown L(W, b)$

### II. Simple expression and interpretation of the gradient

**Gradient = partial derivative**:

+ function : $f(x, y) = xy$
+ input : $x, y$
+ gradient:
  1. gradient of $f$ at $x$ = partial derivative for $x$ = $y$
  2. gradient of $f$ at $y$ = partial derivative for $y$ = $x$
+ gradient:
  + $\triangledown f$ = vector of partial derivatives
  + $\triangledown f$ = $[\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}]$
  + $\triangledown f$ = $[y, x]$

**Interpretation**:

+ derivative 의미 : 특정 변수가 함수값에 미치는 영향(=변화율)
+ $f(x+h)=f(x)+h\frac{df(x)}{dx}$
  + tells $x$가 $h$만큼 증가하면, $f$값은 "$x$의 gradient 값$ \times h = h\frac{df(x)}{dx}$"만큼 증가한다.

**Example**:

(1) Multiplication

+ $f(x,y) = xy = -12$
+ $x = 4, y=-3$
+ gradient on $x$ (=partial derivative on $x$)
  + $\frac{\partial f}{\partial x} = y = -3$
  + tells $x$값이 $h$만큼 증가하면, $f$값은 "$x$의 gradient 값$ \times h = h\frac{df(x)}{dx}=hy=-3h$"만큼 감소한다.
+ gradient on $y$ (=partial derivative on $y$)
  + $\frac{\partial f}{\partial y} = x = 4$
  + tells $y$값이 $h$만큼 증가하면, $f$값은 "$y$의 gradient 값$ \times h = h\frac{df(y)}{dy}=hx=4h$"만큼 증가한다.

(2) Addition

+ $f(x,y)=x+y$
+ gradient on $x$
  + $\frac{\partial f}{\partial x} = 1$
  + tells $x$값이 $h$만큼 증가하면, $f$값은 "$x$의 gradient 값$ \times h = h\frac{df(x)}{dx}=h$"만큼 (1:1비율로) 증가한다.
+ gradient on $y$ (=partial derivative on $y$)
  + $\frac{\partial f}{\partial y} = 1$
  + tells $y$값이 $h$만큼 증가하면, $f$값은 "$y$의 gradient 값$ \times h = h\frac{df(y)}{dy}=h$"만큼 (1:1비율로) 증가한다.

(3) Max

+ $f(x,y)=max(x,y)$
+ gradient on $x$
  + if, $x \ge y \Rightarrow \frac{\partial f}{\partial x} = 1$
    + tells $x$값이 $h$만큼 증가하면, $f$값은 "$x$의 gradient 값$ \times h = h\frac{df(x)}{dx}=h$"만큼 (1:1비율로) 증가한다.
  + if, $x < y \Rightarrow \frac{\partial f}{\partial x} = 0$
    + tells $x$값이 $h$만큼 증가하면, $f$값은 "$x$의 gradient 값$ \times h = h\frac{df(x)}{dx}=0$"만큼 증가한다. 즉, $x$값의 변화는 $f$값에 아무런 영향을 미치지 않는다.
+ gradient on $y$
  + if, $y \ge x \Rightarrow \frac{\partial f}{\partial y} = 1$
    + tells $y$값이 $h$만큼 증가하면, $f$값은 "$y$의 gradient 값$ \times h = h\frac{df(y)}{dy}=h$"만큼 (1:1비율로) 증가한다.
  + if, $y < x \Rightarrow \frac{\partial f}{\partial y} = 0$
    + tells $y$값이 $h$만큼 증가하면, $f$값은 "$y$의 gradient 값$ \times h = h\frac{df(y)}{dy}=0$"만큼 증가한다. 즉, $y$값의 변화는 $f$값에 아무런 영향을 미치지 않는다.
+ gradient $\triangledown f$ = $[\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}]$
  + 더 큰 input값의 gradient는 1, 더 작은 input 값의 gradient는 0이다.


### III. Compound expressions with chain rule



---