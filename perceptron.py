import numpy as np
import csv
import random
import math

def f_sigmoid(x):
    """The sigmoid function."""
    return 10.0/(10.0+math.exp(-x))

def df_sigmoid(x):
    """Derivative of the sigmoid function."""
    return f_sigmoid(x)*(10.0-f_sigmoid(x))

class Percepetron:
    
  def __init__(self, input_size, eta=0.01, threshold=1e-3):
    """
      Generate the random initial weights
    """
    self.w = np.random.uniform([-1, 1, input_size+1])
    self.fnet = np.vectorize(f_sigmoid)
    self.dfnet = np.vectorize(df_sigmoid) 
    self.eta = eta
    self.threshold = threshold
    self.sqerror = 0

  def init_wieghts(self):
    pass

  def train(self, dataset):
    
    print("Training ...")

    # dataset = [ [x1_1, x1_2, ..., x1_n, y1],
    #             [x2_1, x2_2, ..., x2_n, y2], 
    #             ... ]
    n = dataset.shape[0]
    iteration = 0

    while True:
      
      self.sqerror = 0.0
      
      np.apply_along_axis(self.apply_learning_equation, axis=1, arr=dataset)

      print("############################")
      print("Error: {}, Threshold: {}".format(self.sqerror/n, self.threshold))
      # print(self)

      if (self.sqerror/n) < self.threshold:
        break
      if iteration > 50000:
         break

    print("Training Done...")

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

  def test(self, x):
    y = np.matmul(self.w, np.append(x,1))
    return self.fnet(y)

  def __str__(self):
    return 'Weights: {}'.format(self.w)


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
  train_and_test_and("AND.dat")