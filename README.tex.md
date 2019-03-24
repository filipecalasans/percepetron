# *Perceptron*

Our ultimate goal is to mathematically formulate a MLP, however there is a simple type of neural network that will help you to build the foundation to understand MLPs. If you think you already comfortable with this concept, you might want to skip to [MLP](https://github.com/filipecalasans/mlp)

# Preamble 

You may be asking yourself: why do we need another Perceptron/MLP explanation in the internet? This repository provides my thought process after reading several materials when I tried to implement a MLP myself. At the time, I was able to understand and implement it only after a lot of reading, and trial and error. So, as I felt the necessity to be exposed to different ways of explaining the same topic, I think others may face the same situation.

Hope this document can help you on your learning journey. Good Luck !

# Mathematical Formulation

*Perceptron* is single neuron Neural Network(NN) as shown in the picture bellow.

 <p align="center"> 
    <img src="doc/perceptron.png" alt="Perceptron">
 </p>

Mathematically speaking, this neuron produces the following output:

$$out(t) = \sigma( \sum_{i=1}^{n} w_{i}*x_{i} + b )$$

In other words, the output of a neuron is given by a linear combination of its inputs:

$$\sum_{i=1}^{n} w_{i}*x_{i} :(1)$$

Adjusted by an offset, called baias, which produces the output $a$:

$$a = \sum_{i=1}^{n} w_{i}*x_{i} + b :(2)$$

Then, the final output is calculated passing $a$ as argument of the function denominated **Activation Function**:

$$z = out(t) = \sigma(a) :(3)$$

If you remind of Linear Algebra, the equation *(2)* looks like the hyperplane equation $(4)$. Indeed it is a hyperplane. Moreover, the equation give us a notion of how far the data vector $X<x1,x2,x3,...,x_n>$ is from the hyperplane:

$$\sum_{i=1}^{n} w_{i}*x_{i} + b = 0 :(4)$$

Using *Perceptron*, we can create a classifier that given an example characterized by the input $X<x1,x2,x3,...,x_n>$, it returns if the example is **Class** **A = 0** or **B = 1**, using as decisive factor how far the point is from the hyperplane. If you noticed, this is the role of the **Activation Function** in the equation $(3)$. In the image,you can notice that the function used is a step function, but we'll see later there are better **Activation Functions** that we can use.

The step function is given by $f(a) = 0, if a < threshold, f(x)=1, if a > threashold$. This is how we would classify our examples mathematically.

### Now, you should be wondering: How does perceptron "learns" the best hyperplane? 

Indeed, the challenge in Machine Learning is: how do algorithms "learn"?The *Perceptron* classifier is a *supervised learning algorithm*, therefore we must provide a set or examples beforehand, from which it will calculate the best possible hyperplane that separates the examples into two different classes. As you noticed, a single neuron is capable of classifying only two classes. Another characteristic of the *Perceptron* is that it works well only with linearly separable datasets.

Two sets of points are said to be linear separable if there is at least one hyperplane that can separate them in two classes. In two dimensional spaces, you can think as a line that can separate the points on a plane on two different sides. You can read more in [Linear separability - Wikepedia.](https://en.wikipedia.org/wiki/Linear_separability)


## Stochastic Gradient Descent (SGD) - How NNs Learn

Neural Networks, including *Perceptron* and *MLP*, apply the method *Stochastic Gradient Descent (SGD)* on their learning process. SGD is an iterative method for optimizing a differentiable objective function, a stochastic approximation of gradient descent optimization. You can find a more formal explanation in [Wikepedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).

It may sound confusing, even intimidating. But don't worry we'll get there.

Simplifying, SGD is an algorithm to estimate an unknown function. SGD comes from optimization field. In optimization, the ultimate goal is to estimate a function trying to minimize the *Cost function*, which is the measure of how far we are from our goal. If you have ever studied optimization problems, that might sound familiar to you.

The concept of *Cost Function* is also applicable to NNs. It  mathematically represents how far we are from the ultimate goal. The ultimate goal in Classification problems is to classify the training examples correctly.

Let's make a hypothetical experiment. Let's say we have a data set with 10 examples, given by: 

$$Xi = <x1, x2, x3, ...., Xn, Y> :(5)$$

where, $<x1, x2, x3, ...., Xn>$ is the input and *Y* is the correct class for the example. Now, we randomly generate a set of initial weights $<w1, w2, w3, ..., wn>$ and biases $<b1, b2, b3,..., bn>$. We should be able to describe how far we are from classifying the examples correctly, so we can take the best action to improve our classifier. That is the point that **Cost Function** comes in handy. One very popular **Cost Function** is the quadratic error difference, given by:

$$C(w, b) = \|Y -Ŷ\|^2 :(6)$$

This formula tells that for a given set of wights and biases $(w,b)$, the cost is the square of the distance between the right classification $Y$ and the estimated classification $Ŷ$. On 1-dimensional classifiers, such as *Perceptron*, the distance is simply the difference; On N-dimensional problems the cost is the module of the vectorial distance between the two vectors.

In this context, SGD is a method to update $(w,b)$ interactively towards one of the minimum of the function *$C(w,b)$ hopping that it will turn our classifier better, or it will converge towards a minimum. SGD defines the following two update equations, also called in this article learning equations:

$$w_i(t+1) = w_i(t) - \eta\frac{\partial C}{\partial w_i} :(7)$$

$$b_i(t+1) = b_i(t) - \eta\frac{\partial C}{\partial b_i} :(8)$$

These two equations tell that every interaction of the algorithm we must update the weights and biases by a fraction *$\eta$* of the partial derivative, but in the opposite direction. That makes $C(w,b)$ to advance towards a local minimum. It turns out that a function can have multiples minimums, and depending of the initial values we may be trapped on a local minimum, instead of a global one. There are some techniques to mitigate that, however it beyond the scope of this article.

## Formulating the Perceptron

Let's apply what ye have discussed so far to formulate the *Perceptron*.

$$(9): Estimated Output: Ŷ = \sigma(a) = \sigma( \sum_{i=1}^{n} w_{i}*x_{i} + b )$$

$$(10): Cost Function: C(w,b) = \|Y - Ŷ\|^2$$

*Perceptrons* have uni-dimensional output, so we are going to skip the vectorial notation. Re-wrinting it, we have:

$$(11): Cost Function: C(w,b) = (y-ŷ)^2 = (y-\sigma(a))^2$$

Learning Equations:

$$w_i(t+1) = w_i(t) - \eta\frac{\partial C}{\partial w_i} :(12)$$

$$b(t+1) = b(t) - \eta\frac{\partial C}{\partial b} :(13)$$

The key part to understand the next step is to remember the **Chain Rule Derivative**, which is given by:

$${\frac{df(g(x))}{dx}} = {\frac{df(x)}{dg(x)}} {\frac{dg(x)}{dx}}$$

Applying $(10)$ in $(12)$, we have:

$$w_i(t+1) = w_i(t) - \eta\frac{\partial }{\partial w_i}[(y-ŷ)^2] :(14)$$

Let's call the derivative of $D$:

$$D = \frac{\partial }{\partial w_i}[(y-ŷ)^2)] = \frac{\partial }{\partial w_i}[(y-\sigma(w))^2] :(15)$$

If you notice, we have written $D$ on a way that it would be evident the **Chain Rule**.

Let's call $y-\sigma(w) = g(w)$.

Applying the **Chain Rule**, we have:

$$D = \frac{\partial g(w)^2}{\partial g(w)}\frac{\partial g(w)}{\partial w_i} = \frac{\partial g(w)^2}{\partial g(w)}\frac{\partial}{\partial w_i}[y - \sigma(w)] = -2g(w)\sigma'(w) = -2(y-\sigma(w))\sigma'(w) :(16)$$

Notice that $y$ is constant, therefore its derivatives regarding $w_i$, and $\sigma(w_i)$ are zero.

Finally, we can update the Learning Equation $(17)$ to:

$$w_i(t+1) = w_i(t) + 2\eta[y-\sigma(w)]\sigma'(w) :(18)$$

Do you remember that SGD requires a differentiable objective function? Now, you can understand why. As you must have noticed, SGD depends on both **Cost Function** and **Activation Function** derivatives. That is
the reason why we do not utilize the step function as **Cost Function**. Since it has a singularity on $x=0$ we have no way to calculate the derivatives on several points of the space.

Applying the same approach, we can deduce the learning equation for $b$.

$$D = \frac{\partial g(b)^2}{\partial g(b)}\frac{\partial g(b)}{\partial b} = \frac{\partial g(b)^2}{\partial g(b)}\frac{\partial}{\partial b}[y - \sigma(b)] = -2g(b)\sigma'(w) = -2[y-\sigma(b)]\sigma'(b) :(16)$$

Remember:

$$\sigma'(b) = \frac{\partial}{\partial b}[\sigma( \sum_{i=1}^{n} w_{i}*x_{i} + b )] = 1 :(20)$$

Therefore,

$$b_i(t+1) = b_i(t) + 2\eta[y-\sigma(b)] (21)$$

We have now the two Learning Equations that we can use to implement the algorithm:

$$w_i(t+1) = w_i(t) + 2\eta\sigma(a)\sigma'(a)] :(22)$$

$$b(t+1) = b(t) + 2\eta[y-\sigma(b)] (23)$$

## Choosing the Activation Function

We are interested on finding an **Activation Function** that looks like a step function, but at the same time is continuous and differentiable in $-\infty <x <+\infty$. Sigmoid, also called logistic function, is one of the widely used functions due to having these properties.

Sigmoid function is given by:

$$\sigma(x) = \frac{1}{e^-x} : (24)$$

With the following derivative:

$$\sigma'(x) = \sigma(x)(1 - \sigma(x)) : (25)$$

NOTE: The sigmoid is easily differentiable using **Chain Rule**, this is also one of the reasons for its popularity. You can google it if you are curious how to calculate the derivative.

## Notes on Matrix Representation

The aspect that I had difficult the most when I tried to implement NNs in Python was to translate the equations to matrix representation. Sure, we could iterate over each index and calculate one weight per iteration. However, we would be limiting ourself. The main reason we should use matrix representation is because the numeric libraries are optimized for matrix representation. Moreover, they try to take advantage of hardware optimization when possible.

Let's re-write the equations we have learned so far on matrix representation. First we will work with a particular example $(3 inputs, 1 output)$, so you can visualize the dimensions, then we will write the algebraic notation generalizing this particular case.

$$
X = \begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
\end{bmatrix}
,
W =\begin{bmatrix}
w_1 \\
w_2 \\
w_3 \\
\end{bmatrix}
,
B =\begin{bmatrix}
b \\
\end{bmatrix}
,
A =\begin{bmatrix}
a_1 \\
\end{bmatrix}
,
Ŷ =\begin{bmatrix}
y_1 \\
\end{bmatrix}
$$

Output $A$ of the Neuron is given by:

$$
\begin{bmatrix}
a_1 \\
\end{bmatrix}
= 
\begin{bmatrix}
w_1 & w_2 &w_3
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
\end{bmatrix}
+ 
\begin{bmatrix}
b \\
\end{bmatrix}
$$

$$
Algebraic: A = W^TX + B :(28)
$$

After applying the **Activation Function**, we have:
$$
Z = 
\begin{bmatrix}
\sigma(a1) \\
\end{bmatrix}
$$

$$
Algebraic: Z = \sigma(A) :(30)
$$

Notice, from vectorial calculus that:

$$
\frac{\partial Z}{\partial w_i} = 
\begin{bmatrix}
\frac{\partial Z}{\partial w1} \\
\\
\frac{\partial Z}{\partial w2} \\
\\
\frac{\partial Z}{\partial w3}
\end{bmatrix}
=
\begin{bmatrix}
\sigma'_{w1}(a) \\
\\
\sigma'_{w2}(a) \\
\\
\sigma'_{w3}(a)
\end{bmatrix}
$$

The Learning Equations can be re-written as:

$$
\begin{bmatrix}
w_1(t+1) \\
w_2(t+1) \\
w_3(t+1) \\
\end{bmatrix}
=
\begin{bmatrix}
w_1(t) \\
w_2(t) \\
w_3(t) \\
\end{bmatrix}
+ 2\eta
\begin{bmatrix}
y1 - ŷ1
\end{bmatrix}
\begin{bmatrix}
\sigma'_{w1}(a) \\
\\
\sigma'_{w2}(a) \\
\\
\sigma'_{w3}(a)
\end{bmatrix}
$$


$$
\begin{bmatrix}
b(t+1) \\
\end{bmatrix}
=
\begin{bmatrix}
b(t) \\
\end{bmatrix}
+ 2\eta
\begin{bmatrix}
y1 - ŷ1
\end{bmatrix}
$$

$$
Algebraic: W(t+1) = W(t) + 2\eta(Y-Ŷ)\sigma'(A)
$$

$$
Algebraic: B(t+1) = B(t) + 2\eta(Y-Ŷ)
$$

## Implementation

We are going to use the library Numpy to implement the matrix operation in Python.

### Implementation details

We are going to use a slightly different activation function, in order to our example to converge.

```py
def f_sigmoid(x):
    """The sigmoid function."""
    return 10.0/(10.0+math.exp(-x))

def df_sigmoid(x):
    """Derivative of the sigmoid function."""
    return f_sigmoid(x)*(10.0-f_sigmoid(x))
```

We define a class *Perceptron* with the following parameters:

```py
class Percepetron:
    
  def __init__(self, input_size, eta=0.01, threshold=1e-3):
    """
      Generate the random initial weights
    """
    self.w = np.random.uniform([-1, 1, input_size+1])
    self.fnet = np.vectorize(f_sigmoid)
    self.dfnet = np.vectorize(df_sigmoid) 
    self.eta = eta
    # Minimum error before stopping training
    self.threshold = threshold
    # Current squarer of error
    self.sqerror = 0
```

Notice that we generate a initial set of random weights in the interval $[-1,1]$. We also store the weights and bais in the same array, in order to optimize calculation. The array $W$ is given by $W=<w1, w2, b>$. Additionally, we also utilize the function *vectorize* in *numpy* to call the functions *fnet* and *dfnet* for each element in the vector.

```py
class Percepetron:

   """
      ...
   """

    def apply_learning_equation(self, dataset):
    """
      Training Equations
      w_1(t+1) = w_1(t) + 2*eta*(y-sigma(net))*sigma'_{w1}(net)
      w_2(t+1) = w_2(t) + 2*eta*(y-sigma(net))*sigma'_{w2}(net)
      b(t+1) = b(t) + 2*eta*(y-sigma(net))
      sigma(net) = fnet(net)
      sigma'(net) = dfnet(net)
    """
    col = dataset.shape[0]

    # dataset = [x1, x2, ..., xn, Y]
    # x = [x1, x2, ..., xn, 1]
    # y = [Y]
    x = np.append(dataset[:-1], 1)
    y = dataset[-1]

    # Transpose X, Y
    x = x.T
    y = y.T

    net = np.matmul(self.w, x)

    y_o = self.fnet(net)
    error = y - y_o
    
    self.sqerror += np.linalg.norm(error)**2
    delta = 2*error*self.dfnet(x)

    self.w[:-1] += (2.0*self.eta*delta[:-1])
    self.w[-1] += (2.0*self.eta*error)
```

The function *apply_learning_equation* receives as argument an example from the dataset as $dataset=<x1, x2, y>$ and updates the weights and bias. This is done repeatedly until the
minimum error or a maximum number of iterations is reached. We monitor the square error to know if the training is working.

```py
def train(self, dataset):
    
    print("Training ...")

    # dataset = [ [x1_1, x1_2, ..., x1_n, y1],
    #             [x2_1, x2_2, ..., x2_n, y2], 
    #             ...
    n = dataset.shape[0]
    iteration = 0

    while True:
      iteration += 1
      self.sqerror = 0.0
      np.apply_along_axis(self.apply_learning_equation, axis=1, arr=dataset)
      if (self.sqerror/n) < self.threshold:
        break
      if iteration > 50000:
         break
```

## Example

We are going to validate the implementation training the Perceptron to learn the logic *OR*. The examples are provided in the file **OR.dat**.

```py
def train_and_test_and(filename):

  dataset = np.loadtxt(open(filename, "rb"), delimiter=" ")
  
  print("DataSet: {}".format(dataset))
  dimension = dataset.shape[1]

  p = Percepetron(dimension)
  p.train(dataset)

  a = 0
  b = 0
  print("{} OR {}: {}".format(a,b, p.test(np.array([a, b]))))

  a = 0
  b = 1
  print("{} OR {}: {}".format(a,b, p.test(np.array([a, b]))))

  a = 1
  b = 0
  print("{} OR {}: {}".format(a,b, p.test(np.array([a, b]))))

  a = 1
  b = 1
  print("{} OR {}: {}".format(a,b, p.test(np.array([a, b]))))

if __name__ == "__main__":
  train_and_test_and("OR.dat")
```

You can try to train using different datasets as a challenge. We provide the logic *AND* in the file **AND.dat**. Bare in mind that the problem must be linearly separable. 

**Challenge:** Try to train the *Perceptron* to learn the logic *XOR* and verify if the dataset is linearly separable.

**Challenge 2:** Tune the parameters *Activation Function* and *eta*, so the training converges faster. 
