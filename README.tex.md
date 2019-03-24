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

\begin{equation*}
$out(t) = \tau( \sum_{i=1}^{n} w_{i}*x_{i} + b )$
\end{equation*}

In other words, the output of a neuron is given by a linear combination of its inputs:

\begin{equation*}
$\sum_{i=1}^{n} w_{i}*x_{i} :(1)$
\end{equation*}

Adjusted by an offset, called baias, which produces the output $a$:

\begin{equation*}
$a = \sum_{i=1}^{n} w_{i}*x_{i} + b :(2)$
\end{equation*}

Then, the final output is calculated passing $a$ as argument of the function denominated **Activation Function**:

\begin{equation*}
$z = out(t) = \tau(a) :(3)$
\end{equation*}

If you remind of Linear Algebra, the equation *(2)* looks like the hyperplane equation $(4)$. Indeed it is a hyperplane. Moreover, the equation give us a notion of how far the data vector $X<x1,x2,x3,...,x_n>$ is from the hyperplane:

\begin{equation*}
$\sum_{i=1}^{n} w_{i}*x_{i} + b = 0 :(4)$
\end{equation*}

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

\begin{equation*}
$Xi = <x1, x2, x3, ...., Xn, Y> :(5)$
\end{equation*}

where, $<x1, x2, x3, ...., Xn>$ is the input and *Y* is the correct class for the example. Now, we randomly generate a set of initial weights $<w1, w2, w3, ..., wn>$ and biases $<b1, b2, b3,..., bn>$. We should be able to describe how far we are from classifying the examples correctly, so we can take the best action to improve our classifier. That is the point that **Cost Function** comes in handy. One very popular **Cost Function** is the quadratic error difference, given by:

\begin{equation*}
$C(w, b) = \|Y -Ŷ\|^2 :(4)$
\end{equation*}

This formula tells that for a given set of wights and biases $(w,b)$, the cost is the square of the distance between the right classification $Y$ and the estimated classification $Ŷ$. On 1-dimensional classifiers, such as *Perceptron*, the distance is simply the difference; On N-dimensional problems the cost is the module of the vectorial distance between the two vectors.

In this context, SGD is a method to update $(w,b)$ interactively towards one of the minimum of the function *$C(w,b)$ hopping that it will turn our classifier better, or it will converge towards a minimum. SGD defines the following two update equations, also called in this article learning equations:

\begin{equation*}
$w_i(t+1) = w_i(t) - \eta\frac{\partial C}{\partial w_i} :(6)$
\end{equation*}

\begin{equation*}
$b_i(t+1) = b_i(t) - \eta\frac{\partial C}{\partial b_i} :(7)$
\end{equation*}

These two equations tell that every interaction of the algorithm we must update the weights and biases by a fraction *$\eta$* of the partial derivative, but in the opposite direction. That makes $C(w,b)$ to advance towards a local minimum. It turns out that a function can have multiples minimums, and depending of the initial values we may be trapped on a local minimum, instead of a global one. There are some techniques to mitigate that, however it beyond the scope of this article.

## Formulating the Perceptron

Let's apply what ye have discussed so far to formulate the *Perceptron*.

\begin{equation*}
$(7) Estimated Output: Ŷ = \tau(a) = \tau( \sum_{i=1}^{n} w_{i}*x_{i} + b )$
\end{equation*}

\begin{equation*}
$(8) Cost Function: C(w,b) = \|Y - Ŷ\|^2$
\end{equation*}

*Perceptrons* have uni-dimensional output, so we are going to skip the vectorial notation. Re-wrinting it, we have:

\begin{equation*}
$(9) Cost Function: C(w,b) = (y-ŷ)^2 = (y-\tau(a))^2$
\end{equation*}

Learning Equations:

\begin{equation*}
$w_i(t+1) = w_i(t) - \eta\frac{\partial C}{\partial w_i} :(10)$
\end{equation*}

\begin{equation*}
$b_i(t+1) = b_i(t) - \eta\frac{\partial C}{\partial b_i} :(11)$
\end{equation*}

The key part to understand the next step is to remember the **Chain Rule Derivative**, which is given by:

\begin{equation*}
$\frac{df(g(x))}{dx} = \frac{df(x)}{dg(x)}\frac{dg(x)}{dx}$
\end{equation*}

Applying $(9)$ in $(10)$, we have:

\begin{equation*}
$w_i(t+1) = w_i(t) - \eta\frac{\partial }{\partial w_i}[(y-ŷ)^2] (11)$
\end{equation*}

Let's call the derivative of $D$:

\begin{equation*}
$D = \frac{\partial }{\partial w_i}[(y-ŷ)^2)] = \frac{\partial }{\partial w_i}[(y-\tau(w))^2] :(12)$
\end{equation*}

If you notice, we have written $D$ on a way that it would be evident the **Chain Rule**.

Applying the **Chain Rule**, we have:

\begin{equation*}
$D = \frac{\partial}{\partial \tau(w)}[(y-\tau(w))^2]\frac{\partial}{\partial w_i}[y - \tau(w_i)] = 2\tau(w)\tau'(w) :(13)$
\end{equation*}

Notice that $y$ is constant, therefore its derivatives regarding $w_i$, and $\tau(w_i)$ are zero.

Finally, we can update the Learning Equation $(10)$ to:

\begin{equation*}
$w_i(t+1) = w_i(t) - 2\eta\tau(a)\tau'(a)] :(14)$
\end{equation*}

Do you remember from the SGD section, that SGD required a differentiable objective function? Now, you can understand why. As you must have noticed, SGD depends on both **Cost Function** and **Activation Function** derivatives. That is
the reason why we do not utilize the step function in practice. Since it has a singularity on $x=0$ we have now way to calculate the derivatives we need on several points of the space.

Applying the same concepts, we can demonstrate that the learning equation for $b_i$ is:

\begin{equation*}
$D = \frac{\partial}{\partial \tau(b_i)}[(y-\tau(b_i))^2]\frac{\partial}{\partial b_i}[y - \tau(b_i)] = 2\tau(b_i)\tau'(b_i) :(15)$
\end{equation*}

Remember:

\begin{equation*}
$\tau'(b) = \tau'( \sum_{i=1}^{n} w_{i}*x_{i} + b )= 1 :(16)$
\end{equation*}

Therefore,

\begin{equation*}
$b_i(t+1) = b_i(t) - 2\eta\tau(a)] (17)$
\end{equation*}

We now have the two Learning Equations that we can use to implement the algorithm:

\begin{equation*}
$w_i(t+1) = w_i(t) - 2\eta\tau(a)\tau'(a)] :(18)$
\end{equation*}

\begin{equation*}
$b_i(t+1) = b_i(t) - 2\eta\tau(a)] (19)$
\end{equation*}


