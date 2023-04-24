import numpy as np
from matplotlib import pyplot as plt
from atoms.atoms_helpers import Helpers
from atoms.kalmanFilter import KalmanFilter

"""
In the example code we use KF to track a mobile user connected to a wireless network. The discrete-time dynamical 
system is given by:

 X(k+1) = A*X(k) + B*U(k)
 
The matrix of measurement Y describes the estimated position of the mobile:
 
 Y(k) = C*X(k)
 
We plot the estimated, the real trajectory of the mobile user, and the measurements.
"""
logger = Helpers.init_logger()
logger.info('Example of Kalman Filter estimation.')

# time step of mobile user movement
dt = 0.1

# initialization of state matrices
A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
X = np.array([[0.0], [0.0], [0.1], [0.1]])
B = np.eye(X.shape[0])
U = np.zeros((X.shape[0], 1))

# process noise covariance matrix and initial state covariance
Q = np.eye(X.shape[0])
P = np.diag((0.01, 0.01, 0.01, 0.01))

# measurement matrices (state X plus a random gaussian noise)
Y = np.array([[X[0, 0] + abs(np.random.randn(1)[0])], [X[1, 0] + abs(np.random.randn(1)[0])]])
C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

# measurement noise covariance
R = np.eye(Y.shape[0])

# Set up the KF class and simulate the estimation process
logger.info('Setting up the KF class...')

var = {}
var.update({'X': X})
var.update({'A': A})
var.update({'B': B})
var.update({'U': U})
var.update({'Q': Q})
var.update({'C': C})
var.update({'R': R})

kf = KalmanFilter()
kf.setup(var)

# number of iterations in Kalman Filter
n_iter = 50
y_predictive_prob = np.zeros((n_iter, 1))
x_estimated = np.zeros((X.shape[0], n_iter))

# Applying the Kalman Filter
for i in np.arange(0, n_iter):
    kf.predict()
    x_est, y_predict = kf.update(Y)
    x_estimated[0:4, i] = x_est.reshape(1, -1)
    y_predictive_prob[i, 0] = y_predict.reshape(1, -1)
    Y = np.array([[X[0, 0] + abs(0.1 * np.random.randn(1)[0])], [X[1, 0] + abs(0.1 * np.random.randn(1)[0])]])

logger.info('Estimation complete.')

# Plot the results
logger.info('Plotting results ...')

time = np.arange(0, 50*dt, dt)
x_estimated = x_estimated.transpose()
plt.figure()
plt.plot(time, x_estimated)
plt.xlabel('Time')
plt.ylabel('State (X)')
plt.title('KF results')
plt.legend(['x1', 'x2', 'x3', 'x4'], loc='upper right')
plt.show()

logger.info('Done!')
