import numpy
from atoms.kalmanFIlter import KalmanFilter

"""
Consider a discrete-time linear system of the form:

  X(k) = A*X(k-1) + B*U(k)
  Y(k) = C*X(k) 
  
Implement a KF algorithm to predict the evolution of the state X.
"""

print("Testing KalmanFilter class...")

k = KalmanFilter()
# help(k)
help(k.setup)
help(k.predict)

# define dimensions of the state, measurements and input
n_x = 2
n_u = 1
n_y = 2

var = {}
var.update({'X': numpy.array(n_x)})
var.update({'A': numpy.eye(n_x)})
var.update({'B': numpy.zeros([n_x, n_u])})
var.update({'U': numpy.ones([n_u])})
var.update({'Q': numpy.eye(n_x)})

# setup variables
k.setup(var)

# predict X and P at step k
k.predict()
