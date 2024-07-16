import numpy as np
from atoms.linearMPC import LinearMPC
from atoms.atoms_helpers import Helpers
from matplotlib import pyplot as plt

"""
Example of a linear MPC problem, implemented with the LinearMPC class.

Consider a discrete-time double integrator model of the form:

  x_1(k) = x_1(0) + sum_{j=1}^{k-1} x_2(j)
  x_2(k) = x_2(0) + sum_{j=1}^{k-1} u(j)

Rewrite the problem in its state-space form:

  y(k)   = [x_1(k); x_2(k)]
  y(k+1) = A*y(k) + B*u(k) = [1 dt; 0 1]*y(k) + [0; dt]*u(k)

Setup a constrained linear-quadratic MPC problem to stabilize the system using multiple shooting.
"""
logger = Helpers.init_logger()
logger.info('Example of linear MPC problem.')

# Define the time step
dt = 0.025

# Define state-space matrices for a double integrator
n_x = 2
n_u = 1
A = np.array([[1, dt], [0, 1]])
B = np.array([[0], [dt]])

# Define all variables needed for setting up the MPC problem
var = {}
var.update({'N': 20})
var.update({'A': A})
var.update({'B': B})
var.update({'Q_N': np.diag([200, 200])})
var.update({'Q': np.diag([2, 2])})
var.update({'R': 0.1 * np.eye(n_u)})
var.update({'x_r': np.array([1.0, 0.0])})
var.update({'x_0': np.array([0.0, 0.5])})
var.update({'x_min': np.array([-2, -10])})
var.update({'x_max': np.array([2, 10])})
var.update({'u_min': np.array([-5])})
var.update({'u_max': np.array([5])})

# Create the linear MPC object and setup
opti = LinearMPC(debug=True)
opti.setup(var)

# Simulate the problem in closed loop
time = 0
n_sim = 250
y = np.zeros((n_sim, n_x))
y_r = np.zeros((n_sim, n_x))
logger.info('Running simulation ...')

for i in range(n_sim):
    # Solve the problem
    u_star = opti.solve()

    # Apply first control input and update initial conditions
    y[i, :] = var['x_0']
    u = u_star[n_x * (var['N']+1):n_x * (var['N']+1) + n_u]
    x_0 = A.dot(var['x_0']) + B.dot(u)
    var.update({'x_0': x_0})

    # Update the reference trajectory
    time = time + dt
    x_r = np.array([1.0, 0.0])
    y_r[i, :] = x_r
    var.update({'x_r': x_r})
    opti.update(x_r=x_r, x_0=x_0)

# Plot the results
logger.info('Plotting results ...')
fig, axs = plt.subplots(2, 1)
cont = 0

for i in range(n_x):
    axs[i].plot(range(n_sim), y[:, i], range(n_sim), y_r[:, i])
    axs[i].set_xlabel('Iterations')
    axs[i].set_ylabel(f'x_{i+1}')
    axs[i].legend(['Measured', 'Reference'], loc='upper left')
    axs[i].grid(True)

fig.tight_layout()
plt.show()
logger.info('Done!')
