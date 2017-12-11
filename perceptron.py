import numpy as np
import csv

class Percepetron:
    
  def __init__(self, w, theta):
    self.w = w
    self.theta = theta

  def __init__(self, input_size):
    self.w = np.zeros(input_size)
    self.theta = 0

  def init_wieghts(self):
    pass

  def train(self, dataset):
    col = dataset.shape[1]
    x = dataset[:,-(col-1):]
    y= dataset[:,(col-1):]

    # Training Equations
    # w_1(t+1) = w_1(t) - etha * dE2/dW_1
    # w_2(t+1) = w_2(t) - etha * dE2/dW_2
    # theta(t+1) = theta(t) - etha * dE2/dtheta
  
    print("# col: {}".format(col))
    print("X: {}".format(x))
    print("Y: {}".format(y))
    

  def __str__(self):
    return 'Weights: {}'.format(self.w)


def test(filename):

  dataset = np.loadtxt(open(filename, "rb"), delimiter=" ")
  
  print("DataSet: {}".format(dataset))

  p = Percepetron(3)
  p.train(dataset)

  print(p)

test("AND.dat")