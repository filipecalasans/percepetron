import numpy as np
import csv
import random

class Percepetron:
    
  def __init__(self, input_size):
    # Generate randomly
    self.w = np.random.uniform([-0.5, 0.5, input_size+1])
    self.fnet = self.f
    self.etha = 0.01
    self.threshold = 1e-3
    self.sqerror = 0

  def init_wieghts(self):
    pass

  def train(self, dataset):
    
    # Training Equations
    # w_1(t+1) = w_1(t) - etha * dE2/dW_1
    # w_2(t+1) = w_2(t) - etha * dE2/dW_2
    # theta(t+1) = theta(t) - etha * dE2/dtheta

    print("Training")

    # dataset = [ [x1_1, x1_2, ..., x1_n, y1],
    #             [x2_1, x2_2, ..., x2_n, y2], 
    #             ...
    #
    #
    n = dataset.shape[0]
    
    while True:
      
      self.sqerror = 0.0
      
      np.apply_along_axis(self.apply_net_function, axis=1, arr=dataset)
      
      print("############################")
      print("Error: {}, Threshold: {}".format(self.sqerror/n, self.threshold))
      print(self)

      if (self.sqerror/n) < self.threshold:
        break

  def apply_net_function(self, dataset):
    
    col = dataset.shape[0]

    # print("# col: {}".format(col))

    # [x1, x2, ..., xn, 1]
    x = np.append(dataset[-(col-1):],1)
    y = dataset[(col-1):]

    net = np.dot(x, self.w)

    # print("X = {}m y = {}".format(x, y))

    y_o = self.fnet(net)
    error = y - y_o
    
    self.sqerror += error**2
    
    print("Error: {}".format(error))

    # print("Y:{}, Yo: {}, error = {}, error^2:{} ".format(y, y_o, error, self.sqerror))

    # deltas = [2*error[0] - x_i for x_i in x]

    dE2 = error * -x
    
    # print("dE2: {}".format(dE2))
    # print("W(t): {}", self.w)
    
    self.w -= (self.etha * dE2)

    # print("W(t+1): {}", self.w)
    # print("sqerror {}".format(self.sqerror))
    # print("##################################################################")

  def f(self, net):
    if net >= 0.5: return 1.0
    
    return 0.0


  def __str__(self):
    return 'Weights: {}'.format(self.w)


def test(filename):

  dataset = np.loadtxt(open(filename, "rb"), delimiter=" ")
  
  print("DataSet: {}".format(dataset))

  p = Percepetron(2)
  p.train(dataset)

  print(p)

test("AND.dat")