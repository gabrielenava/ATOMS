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
  y(k+1) = A*y(k) + B*u(k) = [1 1; 0 1]*y(k) + [0; 1]*u(k)

Setup a constrained linear-quadratic MPC problem to stabilize the system using multiple shooting.
"""
logger = Helpers.init_logger()
logger.info('Example of linear MPC problem.')

# set the state and input dimensions, initial time, and define matrices A and B
n_x = 2
n_u = 2
time_init = 0
A = np.block([[np.eye(n_x), np.eye(n_x)],
              [np.zeros((n_x, n_x)), np.eye(n_x)]])
B = np.block([[np.zeros((n_x, n_u))], [np.eye(n_u)]])

# define all variables needed for setting up the MPC problem
var = {}
var.update({'N': 6})
var.update({'A': A})
var.update({'B': B})
var.update({'Q_N': np.diag([20, 20, 10, 10], k=0)})
var.update({'Q': np.diag([2, 2, 1, 1], k=0)})
var.update({'R': 15*np.eye(n_u)})
var.update({'x_r': np.array([np.cos(2*np.pi*time_init), -np.cos(2*np.pi*time_init), 0, 0])})
var.update({'x_0': np.array([0.8, -0.8, 0, 0])})
var.update({'x_min': np.array([-2*np.pi, -2*np.pi, -100*np.pi/180, -100*np.pi/180])})
var.update({'x_max': np.array([2*np.pi,  2*np.pi,  100*np.pi/180,  100*np.pi/180])})
var.update({'u_min': np.array([-5, -5])})
var.update({'u_max': np.array([5, 5])})

# create the linear MPC object and setup
opti = LinearMPC(debug=True)
opti.setup(var)

# simulate the problem in closed loop
time = time_init
n_sim = 250
y = np.zeros((n_sim, 2*n_x))
y_r = np.zeros((n_sim, 2*n_x))
logger.info('Running simulation ...')

for i in range(n_sim):

    # solve the problem
    u_star = opti.solve()

    # apply first control input and update initial conditions
    y[i, :] = var['x_0']
    u = u_star[(var['N']+1)*2*n_x:(var['N']+1)*2*n_x+n_u]
    x_0 = A.dot(var['x_0']) + B.dot(u)
    var.update({'x_0': x_0})

    # update the reference trajectory
    time = time + 0.025
    x_r = np.array([np.cos(2*np.pi*time), -np.cos(2*np.pi*time), 0, 0])
    y_r[i, :] = x_r
    var.update({'x_r': x_r})
    opti.update(x_r=x_r, x_0=x_0)

# plot the results
logger.info('Plotting results ...')
fig, axs = plt.subplots(2, 2)
cont = 0

for i in range(n_x):
    for j in range(n_x):
        axs[i, j].plot(range(n_sim), y[:, cont], range(n_sim), y_r[:, cont])
        axs[i, j].set_xlabel('iters')
        axs[i, j].set_ylabel('data')
        # set legend
        axs[i, j].legend(['measured', 'reference'], loc='upper left')
        axs[i, j].grid(True)
        fig.tight_layout()
        cont = cont + 1

plt.show()
logger.info('Done!')
