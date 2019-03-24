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

<p align="center"><img src="/tex/9a809df9fdc8249e9410f07dab9cd474.svg?invert_in_darkmode&sanitize=true" align=middle width=187.04989214999998pt height=44.89738935pt/></p>

In other words, the output of a neuron is given by a linear combination of its inputs:

<p align="center"><img src="/tex/db748de60c1603f7240a3e4a49c1c114.svg?invert_in_darkmode&sanitize=true" align=middle width=108.8211201pt height=44.89738935pt/></p>

Adjusted by an offset, called baias, which produces the output <img src="/tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode&sanitize=true" align=middle width=8.68915409999999pt height=14.15524440000002pt/>:

<p align="center"><img src="/tex/dd1c0900835973c70d36f0e4833028e3.svg?invert_in_darkmode&sanitize=true" align=middle width=166.5738921pt height=44.89738935pt/></p>

Then, the final output is calculated passing <img src="/tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode&sanitize=true" align=middle width=8.68915409999999pt height=14.15524440000002pt/> as argument of the function denominated **Activation Function**:

<p align="center"><img src="/tex/355600e9458a6807696bbd8e5f76a8ad.svg?invert_in_darkmode&sanitize=true" align=middle width=159.46331775pt height=16.438356pt/></p>

If you remind of Linear Algebra, the equation *(2)* looks like the hyperplane equation <img src="/tex/c2e27d5dc3a5c37211768bd7e35bb67e.svg?invert_in_darkmode&sanitize=true" align=middle width=21.00464354999999pt height=24.65753399999998pt/>. Indeed it is a hyperplane. Moreover, the equation give us a notion of how far the data vector <img src="/tex/0fcf433dbe8b273336cdb1ea0866b4ed.svg?invert_in_darkmode&sanitize=true" align=middle width=168.2855526pt height=22.465723500000017pt/> is from the hyperplane:

<p align="center"><img src="/tex/19822bec635c7168d482d3640699c162.svg?invert_in_darkmode&sanitize=true" align=middle width=166.10394735pt height=44.89738935pt/></p>

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

<p align="center"><img src="/tex/a4ba56e87b700707923b54d5957041ec.svg?invert_in_darkmode&sanitize=true" align=middle width=248.37228074999996pt height=16.438356pt/></p>

where, <img src="/tex/9a7e8f68349d094d9f7c597453e4cd1f.svg?invert_in_darkmode&sanitize=true" align=middle width=159.809628pt height=22.465723500000017pt/> is the input and *Y* is the correct class for the example. Now, we randomly generate a set of initial weights <img src="/tex/59f5c7b361345f8237d1bab6a4e9e794.svg?invert_in_darkmode&sanitize=true" align=middle width=160.99315649999997pt height=21.18721440000001pt/> and biases <img src="/tex/49f0048c71ffb9701ef80d555a331f2a.svg?invert_in_darkmode&sanitize=true" align=middle width=140.36896005pt height=22.831056599999986pt/>. We should be able to describe how far we are from classifying the examples correctly, so we can take the best action to improve our classifier. That is the point that **Cost Function** comes in handy. One very popular **Cost Function** is the quadratic error difference, given by:

<p align="center"><img src="/tex/7ba9cff103ca5b2e24e79fb4de1298a9.svg?invert_in_darkmode&sanitize=true" align=middle width=179.1991014pt height=19.68035685pt/></p>

This formula tells that for a given set of wights and biases <img src="/tex/ef714b3dc87e11b2953977c10a4d6d43.svg?invert_in_darkmode&sanitize=true" align=middle width=39.35695994999999pt height=24.65753399999998pt/>, the cost is the square of the distance between the right classification <img src="/tex/91aac9730317276af725abd8cef04ca9.svg?invert_in_darkmode&sanitize=true" align=middle width=13.19638649999999pt height=22.465723500000017pt/> and the estimated classification <img src="/tex/29ca0449252d1ae4e25240e835c5107b.svg?invert_in_darkmode&sanitize=true" align=middle width=13.19638649999999pt height=31.141535699999984pt/>. On 1-dimensional classifiers, such as *Perceptron*, the distance is simply the difference; On N-dimensional problems the cost is the module of the vectorial distance between the two vectors.

In this context, SGD is a method to update <img src="/tex/ef714b3dc87e11b2953977c10a4d6d43.svg?invert_in_darkmode&sanitize=true" align=middle width=39.35695994999999pt height=24.65753399999998pt/> interactively towards one of the minimum of the function *<img src="/tex/1850bfb0ad9603625395b5c8bc51832a.svg?invert_in_darkmode&sanitize=true" align=middle width=52.28159639999999pt height=24.65753399999998pt/> hopping that it will turn our classifier better, or it will converge towards a minimum. SGD defines the following two update equations, also called in this article learning equations:

<p align="center"><img src="/tex/e3edb60f5dbe2d0f213a5f886b110b11.svg?invert_in_darkmode&sanitize=true" align=middle width=216.52677914999998pt height=36.2778141pt/></p>

<p align="center"><img src="/tex/ad71ab9caf04fbb85c11d6b806a431e1.svg?invert_in_darkmode&sanitize=true" align=middle width=700.2746189999999pt height=219.38286315000002pt/></p>(9) Estimated Output: Ŷ = \tau(a) = \tau( \sum_{i=1}^{n} w_{i}*x_{i} + b )<p align="center"><img src="/tex/e7e1fce898b1583cb28cc71db94ffdd5.svg?invert_in_darkmode&sanitize=true" align=middle width=0.0pt height=0.0pt/></p>(10) Cost Function: C(w,b) = \|Y - Ŷ\|^2<p align="center"><img src="/tex/2d572b7d0902aa386e00f7298f25021b.svg?invert_in_darkmode&sanitize=true" align=middle width=700.2745398pt height=35.251144499999995pt/></p>(11) Cost Function: C(w,b) = (y-ŷ)^2 = (y-\tau(a))^2<p align="center"><img src="/tex/fa0bf65070ac4c7a4991859b6f503f42.svg?invert_in_darkmode&sanitize=true" align=middle width=145.34289495pt height=14.42921205pt/></p>w_i(t+1) = w_i(t) - \eta\frac{\partial C}{\partial w_i} :(12)<p align="center"><img src="/tex/e7e1fce898b1583cb28cc71db94ffdd5.svg?invert_in_darkmode&sanitize=true" align=middle width=0.0pt height=0.0pt/></p>b_i(t+1) = b_i(t) - \eta\frac{\partial C}{\partial b_i} :(13)<p align="center"><img src="/tex/b4f94f0579c130dda2e20c386bc401e6.svg?invert_in_darkmode&sanitize=true" align=middle width=700.2746553pt height=35.251144499999995pt/></p>{\frac{df(g(x))}{dx}} = {\frac{df(x)}{dg(x)}} {\frac{dg(x)}{dx}}<p align="center"><img src="/tex/8f8383f067ed6a8066b73f35c6ef4808.svg?invert_in_darkmode&sanitize=true" align=middle width=225.57137294999998pt height=16.438356pt/></p>w_i(t+1) = w_i(t) - \eta\frac{\partial }{\partial w_i}[(y-ŷ)^2] :(14)<p align="center"><img src="/tex/b5ba3d385441b10676fd6f8b2c573d21.svg?invert_in_darkmode&sanitize=true" align=middle width=211.69233359999998pt height=11.4155283pt/></p>D = \frac{\partial }{\partial w_i}[(y-ŷ)^2)] = \frac{\partial }{\partial w_i}[(y-\tau(w))^2] :(15)<p align="center"><img src="/tex/320c0819532c644a5439d2d83925a35d.svg?invert_in_darkmode&sanitize=true" align=middle width=636.71615205pt height=35.251144499999995pt/></p>D = \frac{\partial}{\partial \tau(w)}[(y-\tau(w))^2]\frac{\partial}{\partial w_i}[y - \tau(w)] = 2\tau(w)\tau'(w) :(16)<p align="center"><img src="/tex/e73edf82a4e51a8ab2b11c6341541612.svg?invert_in_darkmode&sanitize=true" align=middle width=600.81026115pt height=36.164383199999996pt/></p>w_i(t+1) = w_i(t) - 2\eta\tau(w)\tau'(w)] :(18)<p align="center"><img src="/tex/ef9ac930b29289ac57f185238c8dc2c4.svg?invert_in_darkmode&sanitize=true" align=middle width=700.27465035pt height=113.24201624999999pt/></p>D = \frac{\partial}{\partial \tau(b_i)}[(y-\tau(b_i))^2]\frac{\partial}{\partial b_i}[y - \tau(b_i)] = 2\tau(b_i)\tau'(b_i) :(19)<p align="center"><img src="/tex/1fcfea2bb42596ee31952d5179b2c5f8.svg?invert_in_darkmode&sanitize=true" align=middle width=81.55274655pt height=11.4155283pt/></p>\tau'(b) = \tau'( \sum_{i=1}^{n} w_{i}*x_{i} + b )= 1 :(20)<p align="center"><img src="/tex/80cd99aaa301f3ef0180000c3710b049.svg?invert_in_darkmode&sanitize=true" align=middle width=73.6075527pt height=14.611878599999999pt/></p>b_i(t+1) = b_i(t) - 2\eta\tau(a)] (21)<p align="center"><img src="/tex/79c9cc02fa0d18b9da329721f95e4901.svg?invert_in_darkmode&sanitize=true" align=middle width=627.6727677pt height=14.611878599999999pt/></p>w_i(t+1) = w_i(t) - 2\eta\tau(a)\tau'(a)] :(22)<p align="center"><img src="/tex/e7e1fce898b1583cb28cc71db94ffdd5.svg?invert_in_darkmode&sanitize=true" align=middle width=0.0pt height=0.0pt/></p>b_i(t+1) = b_i(t) - 2\eta\tau(a)] (23)$$


