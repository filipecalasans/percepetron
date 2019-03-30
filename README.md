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

<p align="center"><img src="/tex/4ed3291b949ab84de341c51e0169abad.svg?invert_in_darkmode&sanitize=true" align=middle width=187.98594045pt height=44.89738935pt/></p>

In other words, the output of a neuron is given by a linear combination of its inputs:

<p align="center"><img src="/tex/db748de60c1603f7240a3e4a49c1c114.svg?invert_in_darkmode&sanitize=true" align=middle width=108.8211201pt height=44.89738935pt/></p>

Adjusted by an offset, called baias, which produces the output <img src="/tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode&sanitize=true" align=middle width=8.68915409999999pt height=14.15524440000002pt/>:

<p align="center"><img src="/tex/dd1c0900835973c70d36f0e4833028e3.svg?invert_in_darkmode&sanitize=true" align=middle width=166.5738921pt height=44.89738935pt/></p>

Then, the final output is calculated passing <img src="/tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode&sanitize=true" align=middle width=8.68915409999999pt height=14.15524440000002pt/> as argument of the function denominated **Activation Function**:

<p align="center"><img src="/tex/b2f63fde1a43cd602c961e800b8a121f.svg?invert_in_darkmode&sanitize=true" align=middle width=160.39936605pt height=16.438356pt/></p>

If you remind of Linear Algebra, the equation *(2)* looks like the hyperplane equation <img src="/tex/c2e27d5dc3a5c37211768bd7e35bb67e.svg?invert_in_darkmode&sanitize=true" align=middle width=21.00464354999999pt height=24.65753399999998pt/>. Indeed it is a hyperplane. Moreover, the equation give us a notion of how far the data vector <img src="/tex/0fcf433dbe8b273336cdb1ea0866b4ed.svg?invert_in_darkmode&sanitize=true" align=middle width=168.2855526pt height=22.465723500000017pt/> is from the hyperplane:

<p align="center"><img src="/tex/19822bec635c7168d482d3640699c162.svg?invert_in_darkmode&sanitize=true" align=middle width=166.10394735pt height=44.89738935pt/></p>

Using *Perceptron*, we can create a classifier that given an example characterized by the input <img src="/tex/0fcf433dbe8b273336cdb1ea0866b4ed.svg?invert_in_darkmode&sanitize=true" align=middle width=168.2855526pt height=22.465723500000017pt/>, it returns if the example is **Class** **A = 0** or **B = 1**, using as decisive factor how far the point is from the hyperplane. If you noticed, this is the role of the **Activation Function** in the equation <img src="/tex/cf330257519e06f13c2ecab5e25c6d2a.svg?invert_in_darkmode&sanitize=true" align=middle width=21.00464354999999pt height=24.65753399999998pt/>. In the image,you can notice that the function used is a step function, but we'll see later there are better **Activation Functions** that we can use.

The step function is given by <img src="/tex/c9d57b49d0a3f8e431d9b620ec8eedee.svg?invert_in_darkmode&sanitize=true" align=middle width=385.3658489999999pt height=24.65753399999998pt/>. This is how we would classify our examples mathematically.

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

where, <img src="/tex/9a7e8f68349d094d9f7c597453e4cd1f.svg?invert_in_darkmode&sanitize=true" align=middle width=159.809628pt height=22.465723500000017pt/> is the input and *Y* is the correct class for the example. Now, we randomly generate a set of initial weights <img src="/tex/59f5c7b361345f8237d1bab6a4e9e794.svg?invert_in_darkmode&sanitize=true" align=middle width=160.99315649999997pt height=21.18721440000001pt/> and bias <img src="/tex/ff2f2a53c274e7b799118e6d5f855f98.svg?invert_in_darkmode&sanitize=true" align=middle width=41.75786009999999pt height=22.831056599999986pt/>. We should be able to describe how far we are from classifying the examples correctly, so we can take the best action to improve our classifier. That is the point that **Cost Function** comes in handy. One very popular **Cost Function** is the quadratic error difference, given by:

<p align="center"><img src="/tex/7ba9cff103ca5b2e24e79fb4de1298a9.svg?invert_in_darkmode&sanitize=true" align=middle width=179.1991014pt height=19.68035685pt/></p>

This formula tells that for a given set of wights and biases <img src="/tex/ef714b3dc87e11b2953977c10a4d6d43.svg?invert_in_darkmode&sanitize=true" align=middle width=39.35695994999999pt height=24.65753399999998pt/>, the cost is the square of the distance between the right classification <img src="/tex/91aac9730317276af725abd8cef04ca9.svg?invert_in_darkmode&sanitize=true" align=middle width=13.19638649999999pt height=22.465723500000017pt/> and the estimated classification <img src="/tex/29ca0449252d1ae4e25240e835c5107b.svg?invert_in_darkmode&sanitize=true" align=middle width=13.19638649999999pt height=31.141535699999984pt/>. On 1-dimensional classifiers, such as *Perceptron*, the distance is simply the squared difference; On N-dimensional problems the cost is the squared module of the vectorial distance between the vector <img src="/tex/91aac9730317276af725abd8cef04ca9.svg?invert_in_darkmode&sanitize=true" align=middle width=13.19638649999999pt height=22.465723500000017pt/> and <img src="/tex/29ca0449252d1ae4e25240e835c5107b.svg?invert_in_darkmode&sanitize=true" align=middle width=13.19638649999999pt height=31.141535699999984pt/>.

In this context, SGD is a method to update <img src="/tex/ef714b3dc87e11b2953977c10a4d6d43.svg?invert_in_darkmode&sanitize=true" align=middle width=39.35695994999999pt height=24.65753399999998pt/> interactively towards one of the minimum of the function *<img src="/tex/1850bfb0ad9603625395b5c8bc51832a.svg?invert_in_darkmode&sanitize=true" align=middle width=52.28159639999999pt height=24.65753399999998pt/> hopping that it will turn our classifier better, or it will converge towards a minimum. SGD defines the following two update equations, also called in this article learning equations:

<p align="center"><img src="/tex/e3edb60f5dbe2d0f213a5f886b110b11.svg?invert_in_darkmode&sanitize=true" align=middle width=216.52677914999998pt height=36.2778141pt/></p>

<p align="center"><img src="/tex/264896f33ad740b0213a9a1f6a9b7e71.svg?invert_in_darkmode&sanitize=true" align=middle width=202.7827065pt height=36.2778141pt/></p>

These two equations tell that every interaction of the algorithm we must update the weights and biases by a fraction *<img src="/tex/1d0496971a2775f4887d1df25cea4f7e.svg?invert_in_darkmode&sanitize=true" align=middle width=8.751954749999989pt height=14.15524440000002pt/>* of the partial derivative, but in the opposite direction. That makes <img src="/tex/1850bfb0ad9603625395b5c8bc51832a.svg?invert_in_darkmode&sanitize=true" align=middle width=52.28159639999999pt height=24.65753399999998pt/> to advance towards a local minimum. It turns out that a function can have multiples minimums, and depending of the initial values we may be trapped on a local minimum, instead of a global one. There are some techniques to mitigate that, however it is beyond the scope of this article.

## Formulating the Perceptron

Let's apply what ye have discussed so far to formulate the *Perceptron*.

<p align="center"><img src="/tex/d8ca7031841ffe3d69c77d4210237f9d.svg?invert_in_darkmode&sanitize=true" align=middle width=390.5371008pt height=44.89738935pt/></p>

<p align="center"><img src="/tex/7b8d0047f75d3c2c770fd682c5023068.svg?invert_in_darkmode&sanitize=true" align=middle width=303.50821214999996pt height=19.68035685pt/></p>

*Perceptrons* have uni-dimensional output, so we are going to skip the vectorial notation. Re-wrinting it, we have:

<p align="center"><img src="/tex/fd5bf95321b22e3e7e664e160c0ddc2c.svg?invert_in_darkmode&sanitize=true" align=middle width=393.03628155pt height=18.312383099999998pt/></p>

Learning Equations:

<p align="center"><img src="/tex/8c2ab502a7471706dbc383260fc1ad34.svg?invert_in_darkmode&sanitize=true" align=middle width=224.74598849999998pt height=36.2778141pt/></p>

<p align="center"><img src="/tex/73f624ee44730ad7c51823c76d82fed6.svg?invert_in_darkmode&sanitize=true" align=middle width=200.0563224pt height=33.81208709999999pt/></p>

The key part to understand the next step is to remember the **Chain Rule Derivative**, which is given by:

<p align="center"><img src="/tex/489ee83ae8532e861c880a7e07ddd0dd.svg?invert_in_darkmode&sanitize=true" align=middle width=167.1712416pt height=38.83491479999999pt/></p>

Applying <img src="/tex/ba5fe1a447c2f0050fee52d1db3dda81.svg?invert_in_darkmode&sanitize=true" align=middle width=29.22385289999999pt height=24.65753399999998pt/> in <img src="/tex/15ab5c1ec13963c53cc79907cd1a57aa.svg?invert_in_darkmode&sanitize=true" align=middle width=29.22385289999999pt height=24.65753399999998pt/>, we have:

<p align="center"><img src="/tex/f42522559706cef82b178baf3c84539b.svg?invert_in_darkmode&sanitize=true" align=middle width=291.4279269pt height=36.2778141pt/></p>

Let's call the derivative <img src="/tex/78ec2b7008296ce0561cf83393cb746d.svg?invert_in_darkmode&sanitize=true" align=middle width=14.06623184999999pt height=22.465723500000017pt/>:

<p align="center"><img src="/tex/ed7e36fa0fbdee99034a50fb8daa8210.svg?invert_in_darkmode&sanitize=true" align=middle width=328.56408915pt height=36.2778141pt/></p>

If you notice, we have written <img src="/tex/78ec2b7008296ce0561cf83393cb746d.svg?invert_in_darkmode&sanitize=true" align=middle width=14.06623184999999pt height=22.465723500000017pt/> on a way that it is evident the **Chain Rule**.

Let's call <img src="/tex/4a547888053313824f59344dea6bfb27.svg?invert_in_darkmode&sanitize=true" align=middle width=119.06382674999998pt height=24.65753399999998pt/>.

Applying the **Chain Rule**, we have:

<p align="center"><img src="/tex/26b0ac68253de9434c66358015fc2117.svg?invert_in_darkmode&sanitize=true" align=middle width=635.2464359999999pt height=39.887022449999996pt/></p>

Notice that <img src="/tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode&sanitize=true" align=middle width=8.649225749999989pt height=14.15524440000002pt/> is constant, therefore its derivatives regarding <img src="/tex/c2a29561d89e139b3c7bffe51570c3ce.svg?invert_in_darkmode&sanitize=true" align=middle width=16.41940739999999pt height=14.15524440000002pt/>, and <img src="/tex/aef8fbb420b71de8ebe75f7dc3d250bd.svg?invert_in_darkmode&sanitize=true" align=middle width=40.00962074999999pt height=24.65753399999998pt/> are zero.

Finally, we can update the Learning Equation <img src="/tex/28520695807194870a28e367e9a2af1d.svg?invert_in_darkmode&sanitize=true" align=middle width=29.22385289999999pt height=24.65753399999998pt/> to:

<p align="center"><img src="/tex/db3e913ef7c166c970359727464a91e1.svg?invert_in_darkmode&sanitize=true" align=middle width=314.58135555pt height=17.2895712pt/></p>

Do you remember that SGD requires a differentiable objective function? Now, you can understand why. As you must have noticed, SGD depends on both **Cost Function** and **Activation Function** derivatives. That is
the reason why we do not utilize the step function as **Cost Function**. Since it has a singularity on <img src="/tex/8436d02a042a1eec745015a5801fc1a0.svg?invert_in_darkmode&sanitize=true" align=middle width=39.53182859999999pt height=21.18721440000001pt/> we have no way to calculate the derivatives on several points of the space.

Applying the same approach, we can deduce the learning equation for <img src="/tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode&sanitize=true" align=middle width=7.054796099999991pt height=22.831056599999986pt/>.

<p align="center"><img src="/tex/b4f8ca80b4e09ab73842ddb3e0ddab2a.svg?invert_in_darkmode&sanitize=true" align=middle width=585.3145963499999pt height=39.887022449999996pt/></p>

Remember:

<p align="center"><img src="/tex/2cadc561c9d3f507ac3c557dcb94330a.svg?invert_in_darkmode&sanitize=true" align=middle width=283.2169131pt height=44.89738935pt/></p>

Therefore,

<p align="center"><img src="/tex/f88ced7d182be6ced9ef92718ee28468.svg?invert_in_darkmode&sanitize=true" align=middle width=246.70842570000002pt height=16.438356pt/></p>

We have now the two Learning Equations that we can use to implement the algorithm:

<p align="center"><img src="/tex/f381ca3599aca557a097ab5b30deecf0.svg?invert_in_darkmode&sanitize=true" align=middle width=312.1041957pt height=17.2895712pt/></p>

<p align="center"><img src="/tex/6a2ea19a0583c2dc5ffd69c39b4e5334.svg?invert_in_darkmode&sanitize=true" align=middle width=237.39719024999997pt height=16.438356pt/></p>

## Choosing the Activation Function

We are interested on finding an **Activation Function** that looks like a step function, but at the same time is continuous and differentiable in <img src="/tex/990b257e0959fef263bdbc9a4515b170.svg?invert_in_darkmode&sanitize=true" align=middle width=111.67792184999999pt height=19.1781018pt/>. Sigmoid, also called logistic function, is one of the widely used functions due to having these properties.

Sigmoid function is given by:

<p align="center"><img src="/tex/a274e2e6e3236f958a1d07cee6875695.svg?invert_in_darkmode&sanitize=true" align=middle width=129.0934194pt height=32.990165999999995pt/></p>

With the following derivative:

<p align="center"><img src="/tex/79fa85e82c7f1837c231a8ada327261e.svg?invert_in_darkmode&sanitize=true" align=middle width=207.03752355pt height=17.2895712pt/></p>

NOTE: The sigmoid is easily differentiable using **Chain Rule**, this is also one of the reasons for its popularity. You can google it if you are curious how to calculate the derivative.

## Notes on Matrix Representation

The aspect that I had difficult the most when I tried to implement NNs in Python was to translate the equations to matrix representation. Sure, we could iterate over each index and calculate one weight per iteration. However, we would be limiting ourself. The main reason we should use matrix representation is because the numeric libraries are optimized for matrix representation. Moreover, they try to take advantage of hardware optimization when possible.

Let's re-write the equations we have learned so far on matrix representation. First we will work with a particular example <img src="/tex/618665d8d764eca9c578a2a175f0061b.svg?invert_in_darkmode&sanitize=true" align=middle width=130.31361255pt height=24.65753399999998pt/>, so you can visualize the dimensions, then we will write the algebraic notation generalizing this particular case.

<p align="center"><img src="/tex/13461f75d11da71cde0726b5023d0f23.svg?invert_in_darkmode&sanitize=true" align=middle width=380.70228405pt height=59.1786591pt/></p>

Output <img src="/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/> of the Neuron is given by:

<p align="center"><img src="/tex/3f1486440d5360f417aa6888d6c808a0.svg?invert_in_darkmode&sanitize=true" align=middle width=237.955806pt height=59.1786591pt/></p>

<p align="center"><img src="/tex/e18af9b7380108f9969f163a1b874dfd.svg?invert_in_darkmode&sanitize=true" align=middle width=237.35981114999998pt height=18.7598829pt/></p>

After applying the **Activation Function**, we have:
<p align="center"><img src="/tex/d214c29d79da3c95df531d6877a7291a.svg?invert_in_darkmode&sanitize=true" align=middle width=87.69022185pt height=19.726228499999998pt/></p>

<p align="center"><img src="/tex/b0555c9199390d80ab1829bbeaac193c.svg?invert_in_darkmode&sanitize=true" align=middle width=196.0682823pt height=16.438356pt/></p>

Notice, from vectorial calculus that:

<p align="center"><img src="/tex/d257addb21af0ecbaede2879d45e9b85.svg?invert_in_darkmode&sanitize=true" align=middle width=192.63145935pt height=100.34585715pt/></p>

The Learning Equations can be re-written as:

<p align="center"><img src="/tex/ce9f08162de18eaa0896eb5ee05441b6.svg?invert_in_darkmode&sanitize=true" align=middle width=349.8402996pt height=98.63111444999998pt/></p>


<p align="center"><img src="/tex/6038e154ccad851b48acbbfdfea23a56.svg?invert_in_darkmode&sanitize=true" align=middle width=236.50666710000002pt height=19.726228499999998pt/></p>

<p align="center"><img src="/tex/7177e96421fe6dd3e0b448fd34e2d5b9.svg?invert_in_darkmode&sanitize=true" align=middle width=343.06229594999996pt height=19.68035685pt/></p>

<p align="center"><img src="/tex/f8e066434d0d10ab047623e23d086c9c.svg?invert_in_darkmode&sanitize=true" align=middle width=294.32359935pt height=19.68035685pt/></p>

## Implementation

We are going to use the library *Numpy* to implement matrix operation in *Python*.

### Implementation details

We are going also to use a slightly different activation function, in order to our example to converge.

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

Notice that we generate a initial set of random weights in the interval <img src="/tex/699628c77c65481a123e3649944c0d51.svg?invert_in_darkmode&sanitize=true" align=middle width=45.66218414999998pt height=24.65753399999998pt/>. We also store the weights and bais in the same array, in order to optimize calculation. The array <img src="/tex/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode&sanitize=true" align=middle width=17.80826024999999pt height=22.465723500000017pt/> is given by <img src="/tex/b29d13461ee82e1bfb4ff61ac73d506b.svg?invert_in_darkmode&sanitize=true" align=middle width=132.38952704999997pt height=22.831056599999986pt/>. Additionally, we also utilize the function *vectorize* in *numpy* to call the functions *fnet* and *dfnet* for each element in the vector.

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

The function *apply_learning_equation* receives as argument an example from the dataset, where <img src="/tex/3f57ca682c1bb77e982d41dc25f129e4.svg?invert_in_darkmode&sanitize=true" align=middle width=163.71004649999998pt height=22.831056599999986pt/>. Next we update the weights and bias. This step is repeated until the
minimum error or the maximum number of iterations is reached. We also keep monitoring the square error to know if the training is converging.

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

We are going to validate our implementation training the Perceptron to learn the logic *OR*. The examples are provided in the file **OR.dat**.

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

You can try to train using different datasets as a challenge. We provide the logic *AND* in the file **AND.dat**. But, bear in mind that the dataset must be linearly separable. 

**Challenge 1:** Try to train the *Perceptron* to learn the logic *XOR* and verify if the dataset is linearly separable.

**Challenge 2:** Tune the parameters in the *Activation Function* and *eta*, so the training converges faster.
