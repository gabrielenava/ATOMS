import numpy
from atoms.kalmanFIlter import KalmanFilter

k = KalmanFilter()
help(k)

var = {}
var.update({'X': numpy.array(2)})
var.update({'A': numpy.eye(2)})
var.update({'B': numpy.zeros(2, 1)})
var.update({'U': numpy.ones(1, 1)})
var.update({'Q': numpy.eye(2, 2)})

k.setup(var)
k.update()
