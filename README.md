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

<p align="center"><img src="/tex/b7e9ad8adc6f8441041e3da923ed466b.svg?invert_in_darkmode&sanitize=true" align=middle width=205.48717365pt height=18.150897599999997pt/></p>

In other words, the output of a neuron is given by a linear combination of its inputs:

<p align="center"><img src="/tex/c6e96a9879596512e1ca747f468d060c.svg?invert_in_darkmode&sanitize=true" align=middle width=124.5451383pt height=18.150897599999997pt/></p>

Adjusted by an offset, called baias, which produces the output <img src="/tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode&sanitize=true" align=middle width=8.68915409999999pt height=14.15524440000002pt/>:

<p align="center"><img src="/tex/bb4580678c77d4c9117cbc9b45affedb.svg?invert_in_darkmode&sanitize=true" align=middle width=183.6546558pt height=18.150897599999997pt/></p>

Then, the final output is calculated passing <img src="/tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode&sanitize=true" align=middle width=8.68915409999999pt height=14.15524440000002pt/> as argument of the function denominated **Activation Function**:

<p align="center"><img src="/tex/585d690c417f302f41000ad7e6984a2e.svg?invert_in_darkmode&sanitize=true" align=middle width=162.94160685pt height=16.438356pt/></p>

If you remind of Linear Algebra, the equation *(2)* looks like the hyperplane equation <img src="/tex/c2e27d5dc3a5c37211768bd7e35bb67e.svg?invert_in_darkmode&sanitize=true" align=middle width=21.00464354999999pt height=24.65753399999998pt/>. Indeed it is a hyperplane. Moreover, the equation give us a notion of how far the data vector <img src="/tex/0fcf433dbe8b273336cdb1ea0866b4ed.svg?invert_in_darkmode&sanitize=true" align=middle width=168.2855526pt height=22.465723500000017pt/> is from the hyperplane:

<p align="center"><img src="/tex/351aaba2c6eea885cab2e44193802227.svg?invert_in_darkmode&sanitize=true" align=middle width=181.82796555pt height=18.150897599999997pt/></p>

Using *Perceptron*, we can create a classifier that given an example characterized by the input <img src="/tex/0fcf433dbe8b273336cdb1ea0866b4ed.svg?invert_in_darkmode&sanitize=true" align=middle width=168.2855526pt height=22.465723500000017pt/>, it returns if the example is **Class** **A = 0** or **B = 1**, using as decisive factor how far the point is from the hyperplane. If you noticed, this is the role of the **Activation Function** in the equation <img src="/tex/cf330257519e06f13c2ecab5e25c6d2a.svg?invert_in_darkmode&sanitize=true" align=middle width=21.00464354999999pt height=24.65753399999998pt/>. In the image,you can notice that the function used is a step function, but we'll see later there are better **Activation Functions** that we can use.

The step function is given by <img src="/tex/7990677e990316a3b2f1ac2952c73e75.svg?invert_in_darkmode&sanitize=true" align=middle width=386.0716827pt height=24.65753399999998pt/>. This is how we would classify our examples mathematically.

### Now, you should be wondering: How does perceptron "learns" the best hyperplane? 

Indeed, the challenge in Machine Learning is: how do algorithms "learn"?The *Perceptron* classifier is a *supervised learning algorithm*, therefore we must provide a set or examples beforehand, from which it will calculate the best possible hyperplane that separates the examples into two different classes. As you noticed, a single neuron is capable of classifying only two classes. Another characteristic of the *Perceptron* is that it works well only with linearly separable datasets.

Two sets of points are said to be linear separable if there is at least one hyperplane that can separate them in two classes. In two dimensional spaces, you can think as a line that can separate the points on a plane on two different sides. You can read more in [Linear separability - Wikepedia.](https://en.wikipedia.org/wiki/Linear_separability)


## Stochastic Gradient Descent (SGD) - How NNs Learn

Neural Networks, including *Perceptron* and *MLP*, apply the method *Stochastic Gradient Descent (SGD)* on their learning process. SGD is an iterative method for optimizing a differentiable objective function, a stochastic approximation of gradient descent optimization. You can find a more formal explanation in [Wikepedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).

It may sound confusing, even intimidating. But don't worry we'll get there.

Simplifying, SGD is an algorithm to estimate an unknown function. SGD comes from optimization field. In optimization, the ultimate goal is to estimate a function trying to minimize the *Cost function*, which is the measure of how far we are from our goal. If you have ever studied optimization problems, that might sound familiar to you.

The concept of *Cost Function* is also applicable to NNs. It  mathematically represents how far we are from the ultimate goal. The ultimate goal in Classification problems is to classify the training examples correctly.

Let's make a hypothetical experiment. Let's say we have a data set with 10 examples, given by: 

<p align="center"><img src="/tex/8d5b956c0f66255b73784d0c7fddc11f.svg?invert_in_darkmode&sanitize=true" align=middle width=236.98690785pt height=16.438356pt/></p>

where, <img src="/tex/9a7e8f68349d094d9f7c597453e4cd1f.svg?invert_in_darkmode&sanitize=true" align=middle width=159.809628pt height=22.465723500000017pt/> is the input and *Y* is the correct class for the example. Now, we randomly generate a set of initial weights <img src="/tex/59f5c7b361345f8237d1bab6a4e9e794.svg?invert_in_darkmode&sanitize=true" align=middle width=160.99315649999997pt height=21.18721440000001pt/> and biases <img src="/tex/49f0048c71ffb9701ef80d555a331f2a.svg?invert_in_darkmode&sanitize=true" align=middle width=140.36896005pt height=22.831056599999986pt/>. We should be able to describe how far we are from classifying the examples correctly, so we can take the best action to improve our classifier. That is the point that **Cost Function** comes in handy. One very popular **Cost Function** is the quadratic error difference, given by:

<p align="center"><img src="/tex/5de8ab4814e83fe396d857b2761e1dc6.svg?invert_in_darkmode&sanitize=true" align=middle width=184.45210244999998pt height=19.68035685pt/></p>

This formula tells that for a given set of wights and biases <img src="/tex/ef714b3dc87e11b2953977c10a4d6d43.svg?invert_in_darkmode&sanitize=true" align=middle width=39.35695994999999pt height=24.65753399999998pt/>, the cost is the square of the distance between the right classification <img src="/tex/91aac9730317276af725abd8cef04ca9.svg?invert_in_darkmode&sanitize=true" align=middle width=13.19638649999999pt height=22.465723500000017pt/> and the estimated classification <img src="/tex/29ca0449252d1ae4e25240e835c5107b.svg?invert_in_darkmode&sanitize=true" align=middle width=13.19638649999999pt height=31.141535699999984pt/>. On 1-dimensional classifiers, such as *Perceptron*, the distance is simply the difference; On N-dimensional problems the cost is the module of the vectorial distance between the two vectors.

In this context, SGD is a method to update <img src="/tex/ef714b3dc87e11b2953977c10a4d6d43.svg?invert_in_darkmode&sanitize=true" align=middle width=39.35695994999999pt height=24.65753399999998pt/> interactively towards one of the minimum of the function *<img src="/tex/1850bfb0ad9603625395b5c8bc51832a.svg?invert_in_darkmode&sanitize=true" align=middle width=52.28159639999999pt height=24.65753399999998pt/> hopping that it will turn our classifier better, or it will converge towards a minimum. SGD defines the following two update equations, also called in this article learning equations:

<p align="center"><img src="/tex/d331d380fd0a6d117e0da1a5c2c36099.svg?invert_in_darkmode&sanitize=true" align=middle width=212.19357885pt height=21.7756011pt/></p>

<p align="center"><img src="/tex/68e62b90c9f31524c7c7b4a8e7b4cdb8.svg?invert_in_darkmode&sanitize=true" align=middle width=201.0112797pt height=21.7756011pt/></p>

These two equations tell that every interaction of the algorithm we must update the weights and biases by a fraction *<img src="/tex/1d0496971a2775f4887d1df25cea4f7e.svg?invert_in_darkmode&sanitize=true" align=middle width=8.751954749999989pt height=14.15524440000002pt/>* of the partial derivative, but in the opposite direction. That makes <img src="/tex/1850bfb0ad9603625395b5c8bc51832a.svg?invert_in_darkmode&sanitize=true" align=middle width=52.28159639999999pt height=24.65753399999998pt/> to advance towards a local minimum. It turns out that a function can have multiples minimums, and depending of the initial values we may be trapped on a local minimum, instead of a global one. There are some techniques to mitigate that, however it beyond the scope of this article.

## Formulating the Perceptron

Let's apply what ye have discussed so far to formulate the *Perceptron*.

<p align="center"><img src="/tex/0201b21642f6e0a78ba870718aa4186b.svg?invert_in_darkmode&sanitize=true" align=middle width=397.5156174pt height=20.50235055pt/></p>

<p align="center"><img src="/tex/2b45367ea6b5aaddbbcc3c4a6bf7875e.svg?invert_in_darkmode&sanitize=true" align=middle width=283.49371709999997pt height=19.68035685pt/></p>

*Perceptrons* have uni-dimensional output, so we are going to skip the vectorial notation. Re-wrinting it, we have:

<p align="center"><img src="/tex/bd420a3171fcf8896e47dfaa69b3d20b.svg?invert_in_darkmode&sanitize=true" align=middle width=357.5272998pt height=17.4904653pt/></p>

Learning Equations:

<p align="center"><img src="/tex/ac1726d7f6b3d181bce77c917dfc5cc4.svg?invert_in_darkmode&sanitize=true" align=middle width=220.41278819999997pt height=21.7756011pt/></p>

<p align="center"><img src="/tex/1b7d9cf2347056068694b36d2261a399.svg?invert_in_darkmode&sanitize=true" align=middle width=209.23048905pt height=21.7756011pt/></p>

The key part to understand the next step is to remember the **Chain Rule Derivative**, which is given by:

<!-- <p align="center"><img src="/tex/3cdba63ed7ddceb3222c779f2bf511a8.svg?invert_in_darkmode&sanitize=true" align=middle width=346.94066054999996pt height=42.37444365pt/></p> -->
<img src="/tex/1c0a0c938c1cfceae74e0da6cb7dd099.svg?invert_in_darkmode&sanitize=true" align=middle width=139.85085015pt height=33.20539859999999pt/>

Applying <img src="/tex/a1fe456628ef54c400febb4c43522cc5.svg?invert_in_darkmode&sanitize=true" align=middle width=21.00464354999999pt height=24.65753399999998pt/> in <img src="/tex/ba5fe1a447c2f0050fee52d1db3dda81.svg?invert_in_darkmode&sanitize=true" align=middle width=29.22385289999999pt height=24.65753399999998pt/>, we have:

<p align="center"><img src="/tex/68d34f28cab035b6bdc5170627ef92ae.svg?invert_in_darkmode&sanitize=true" align=middle width=273.39630615pt height=21.7756011pt/></p>

Let's call the derivative of <img src="/tex/78ec2b7008296ce0561cf83393cb746d.svg?invert_in_darkmode&sanitize=true" align=middle width=14.06623184999999pt height=22.465723500000017pt/>:

<p align="center"><img src="/tex/78795bc119ed7adb1f207abad8954b3b.svg?invert_in_darkmode&sanitize=true" align=middle width=254.3981154pt height=25.0250055pt/></p>

If you notice, we have written <img src="/tex/78ec2b7008296ce0561cf83393cb746d.svg?invert_in_darkmode&sanitize=true" align=middle width=14.06623184999999pt height=22.465723500000017pt/> on a way that it would be evident the **Chain Rule**.

Applying the **Chain Rule**, we have:

<p align="center"><img src="/tex/a198f07999e502cd77e1ccd050d9d412.svg?invert_in_darkmode&sanitize=true" align=middle width=336.61350525pt height=25.0250055pt/></p>

Notice that <img src="/tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode&sanitize=true" align=middle width=8.649225749999989pt height=14.15524440000002pt/> is constant, therefore its derivatives regarding <img src="/tex/c2a29561d89e139b3c7bffe51570c3ce.svg?invert_in_darkmode&sanitize=true" align=middle width=16.41940739999999pt height=14.15524440000002pt/>, and <img src="/tex/298816eb1611d0204937793eb4b79ce8.svg?invert_in_darkmode&sanitize=true" align=middle width=39.07357244999999pt height=24.65753399999998pt/> are zero.

Finally, we can update the Learning Equation <img src="/tex/ba5fe1a447c2f0050fee52d1db3dda81.svg?invert_in_darkmode&sanitize=true" align=middle width=29.22385289999999pt height=24.65753399999998pt/> to:

<p align="center"><img src="/tex/6407f3ffcd9aa17aaae2a9a833922ffb.svg?invert_in_darkmode&sanitize=true" align=middle width=272.46292589999996pt height=16.4676534pt/></p>

Do you remember from the SGD section, that SGD required a differentiable objective function? Now, you can understand why. As you must have noticed, SGD depends on both **Cost Function** and **Activation Function** derivatives. That is
the reason why we do not utilize the step function in practice. Since it has a singularity on <img src="/tex/8436d02a042a1eec745015a5801fc1a0.svg?invert_in_darkmode&sanitize=true" align=middle width=39.53182859999999pt height=21.18721440000001pt/> we have now way to calculate the derivatives we need on several points of the space.

Applying the same concepts, we can demonstrate that the learning equation for <img src="/tex/d3aa71141bc89a24937c86ec1d350a7c.svg?invert_in_darkmode&sanitize=true" align=middle width=11.705695649999988pt height=22.831056599999986pt/> is:

<p align="center"><img src="/tex/23fac247f652b49e87f019335f7cf90e.svg?invert_in_darkmode&sanitize=true" align=middle width=334.4729619pt height=25.0250055pt/></p>

Remember:

<p align="center"><img src="/tex/8f11234123521f42e7d1fca449cb70d4.svg?invert_in_darkmode&sanitize=true" align=middle width=271.90788734999995pt height=18.150897599999997pt/></p>

Therefore,

<p align="center"><img src="/tex/8717184ab7f1c0ffad3c07cf688e3b39.svg?invert_in_darkmode&sanitize=true" align=middle width=216.17776455pt height=16.438356pt/></p>

We now have the two Learning Equations that we can use to implement the algorithm:

<p align="center"><img src="/tex/68fbb0db540a2d8256eae27b9b617ec0.svg?invert_in_darkmode&sanitize=true" align=middle width=272.46292589999996pt height=16.4676534pt/></p>

<p align="center"><img src="/tex/c1fccf6abfe754e7a4fddad843c2711f.svg?invert_in_darkmode&sanitize=true" align=middle width=216.17776455pt height=16.438356pt/></p>


