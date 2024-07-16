import numpy as np
from matplotlib import pyplot as plt
from atoms.atoms_helpers import Helpers
from atoms.kalmanFilter import KalmanFilter

"""
Example of using Kalman Filter to track a mobile user connected to a wireless network. 
The discrete-time dynamical system is given by:

 X(k+1) = A*X(k) + B*U(k)

The measurement matrix Y describes the estimated position of the mobile:

 Y(k) = C*X(k)

We plot the estimated trajectory, the real trajectory, and the noisy measurements.
"""

# Initialize logger
logger = Helpers.init_logger()
logger.info('Example of Kalman Filter estimation.')

# Time step of mobile user movement
dt = 0.01

# Initialization of state matrices
A = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
X = np.array([[0.0], [0.0], [0.1], [0.1]])  # Initial state
B = np.eye(X.shape[0])
U = np.zeros((X.shape[0], 1))  # Control input

# Process noise covariance matrix
Q = np.eye(X.shape[0])

# Measurement matrix (state X plus a random Gaussian noise)
C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

# Measurement noise covariance
R = np.eye(C.shape[0])

# Set up the Kalman Filter class
logger.info('Setting up the KF class...')
variables = {
    'X': X, 'A': A, 'B': B, 'U': U, 'Q': Q, 'C': C, 'R': R
}
kf = KalmanFilter(debug=True)
kf.setup(variables)

# Number of iterations in Kalman Filter
n_iter = 50
y_measured = np.zeros((C.shape[0], n_iter))
x_estimated = np.zeros((X.shape[0], n_iter))

# Applying the Kalman Filter
for i in range(n_iter):
    kf.predict()
    # Simulate measurement with some noise
    Y = C @ X + np.random.normal(0, 0.1, (C.shape[0], 1))
    y_measured[:, i] = Y.flatten()

    x_est, _ = kf.update(Y)
    x_estimated[:, i] = x_est.flatten()

    # Simulate real system's next state (for demonstration purposes)
    X = A @ X + B @ U

logger.info('Estimation complete.')

# Plot the results
logger.info('Plotting results ...')
time = np.arange(0, n_iter * dt, dt)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot each state variable
for i in range(X.shape[0]):
    row, col = divmod(i, 2)
    axs[row, col].plot(time, x_estimated[i, :], label='Estimated')
    axs[row, col].scatter(time, y_measured[i % 2, :], s=10, color='r', label='Measured', alpha=0.6)
    axs[row, col].set_xlabel('Time [s]')
    axs[row, col].set_ylabel(f'State X{i + 1}')
    axs[row, col].legend(loc='upper right')
    axs[row, col].grid(True)

fig.suptitle('Kalman Filter Results')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

logger.info('Done!')
