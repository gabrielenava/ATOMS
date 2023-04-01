# Testing of the LinearMPC class from the ATOMS package
import unittest
import numpy as np
from atoms.linearMPC import LinearMPC


class TestLinearMPC(unittest.TestCase):

    def test_LinearMPC(self):

        # set the state and input dimensions, initial time, and define matrices A and B
        n_x = 2
        n_u = 2
        time_init = 0
        A = np.block([[np.eye(n_x), np.eye(n_x)],
                      [np.zeros((n_x, n_x)), np.eye(n_x)]])
        B = np.block([[np.zeros((n_x, n_u))], [np.eye(n_u)]])

        # check if the LinearMPC class is instantiated correctly
        opti = LinearMPC(debug=True)
        self.assertEqual(type(opti), LinearMPC)

        # check if the setup method is working correctly
        var = {}
        var.update({'N': 6})
        var.update({'A': A})
        var.update({'B': B})
        var.update({'Q_N': np.diag([20, 20, 10, 10], k=0)})
        var.update({'Q': np.diag([2, 2, 1, 1], k=0)})
        var.update({'R': 15 * np.eye(n_u)})
        var.update({'x_r': np.array([np.cos(2 * np.pi * time_init), -np.cos(2 * np.pi * time_init), 0, 0])})
        var.update({'x_0': np.array([0.8, -0.8, 0, 0])})
        var.update({'x_min': np.array([-2 * np.pi, -2 * np.pi, -100 * np.pi / 180, -100 * np.pi / 180])})
        var.update({'x_max': np.array([2 * np.pi, 2 * np.pi, 100 * np.pi / 180, 100 * np.pi / 180])})
        var.update({'u_min': np.array([-5, -5])})
        var.update({'u_max': np.array([5, 5])})

        opti.setup(var)
        self.assertEqual(list(opti.variables.keys()), ['N', 'Q', 'Q_N', 'n_x', 'x_0', 'x_r', 'P', 'q', 'A', 'l', 'u'])

        # simulate the problem in closed loop
        time = time_init
        n_sim = 2
        y = np.zeros((n_sim, 2 * n_x))
        y_r = np.zeros((n_sim, 2 * n_x))

        for i in range(n_sim):

            # solve the problem
            u_star = opti.solve()

            # apply first control input and update initial conditions
            y[i, :] = var['x_0']
            u = u_star[(var['N'] + 1) * 2 * n_x:(var['N'] + 1) * 2 * n_x + n_u]
            x_0 = A.dot(var['x_0']) + B.dot(u)
            var.update({'x_0': x_0})

            # update the reference trajectory
            time = time + 0.025
            x_r = np.array([np.cos(2 * np.pi * time), -np.cos(2 * np.pi * time), 0, 0])
            y_r[i, :] = x_r
            var.update({'x_r': x_r})

            opti.update(x_r=x_r, x_0=x_0)
            self.assertEqual(opti.variables['x_0'], [0.84653871, -0.84653871,  0.0512824,  -0.0512824])
            self.assertEqual(opti.variables['x_r'], [0.98768834, -0.98768834,  0., 0.])


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestLinearMPC('test_LinearMPC'))
