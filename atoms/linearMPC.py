import osqp
import numpy as np
from scipy import sparse as sp
from atoms.atoms_helpers import Helpers


class LinearMPC:
    """
    LinearMPC class: implements a multiple-shooting, constrained linear-quadratic MPC via OSQP.

        The class solves the following optimal control problem:

            minimize (x(N)-x_r)^T*Q_N*(x(N)-x_r) + sum_{k=1}^{N-1}[(x(k)-x_r)^T*Q*(x(k)-x_r) + u(k)^T*R*u(k)]
                  s.t.
                      x(k+1) = A*x(k) + B*u(k)
                      x_min <= x(k) <= x_max
                      u_min <= u(k) <= u_max

    Author: Gabriele Nava, gabriele.nava@iit.it
    Last updated on 01/04/2023
    """
    def __init__(self, debug=False):
        self.variables = {}
        self.debug = debug
        self.solver = osqp.OSQP()

        if debug:
            self.logger = Helpers.init_logger()

    def __str__(self):
        return f" LinearMPC class object \n" \
               f" Stored variables: {self.variables}"

    def setup(self, variables):
        """
        Cast the MPC problem to a QP.
        :param variables: list of variables to be passed to the QP solver. It must include:
            - A = discrete system state matrix such that x(k+1) = A*x(k) + B*u(k)
            - B = discrete system input matrix such that x(k+1) = A*x(k) + B*u(k)
            - x_min = lower limits on x
            - x_max = upper limits on x
            - u_min = lower limits on u
            - u_max = upper limits on u
            - x_r = reference state
            - x_0 = initial state
            - N = number of prediction steps
            - Q = weight on state error
            - Q_N = weight on final state error
            - R = weight on input
        """
        # demux variables
        N = variables['N']
        Q = variables['Q']
        R = variables['R']
        Q_N = variables['Q_N']
        x_r = variables['x_r']
        x_0 = variables['x_0']
        x_min = variables['x_min']
        x_max = variables['x_max']
        u_min = variables['u_min']
        u_max = variables['u_max']
        A = variables['A']
        B = variables['B']
        [n_x, n_u] = B.shape

        # save useful variables
        self.variables.update({'N': N, 'Q': Q, 'Q_N': Q_N, 'n_x': n_x, 'x_0': x_0, 'x_r': x_r})

        # create the Hessian matrix. Format:
        #
        # P = [ Q   0 ...  R  ... 0;
        #       0   Q ...  0  ... 0;
        #      ...  0  0  ...  0  R];
        #
        P = sp.block_diag([sp.kron(sp.eye(N), Q), Q_N, sp.kron(sp.eye(N), R)], format='csc')
        self.variables.update({'P': P})

        # create the gradient. Format:
        #
        # q = [-Q * x_r; ...; -Q_N * x_r; ...; 0]
        #
        # note: the term x_r^T*Q*x_r does not affect the QP solution, and it is ignored.
        #
        q = np.hstack([np.kron(np.ones(N), -Q.dot(x_r)), -Q_N.dot(x_r), np.zeros(N*n_u)])
        self.variables.update({'q': q})

        # constraints: linear dynamics and initial conditions
        #
        # x(0) - x_0 = 0 (initial conditions)
        # A*x(k) + B*u(k) - x(k+1) = 0 (dynamics)
        #
        # A_dyn = [-1   0 ... 0  0
        #           A  -1 ... 0  0
        #           0   0 ... A -1]
        #
        # B_dyn = [0  0 ... 0
        #          0  B ... 0
        #          0  0 ... B]
        #
        # leq = ueq = [-x0; 0; 0]
        #
        A_dyn = sp.kron(sp.eye(N+1), -sp.eye(n_x)) + sp.kron(sp.eye(N+1, k=-1), A)
        B_dyn = sp.kron(sp.vstack([sp.csc_matrix((1, N)), sp.eye(N)]), B)
        A_eq = sp.hstack([A_dyn, B_dyn])
        l_eq = np.hstack([-x_0, np.zeros(N*n_x)])
        u_eq = l_eq

        # constraints: lower and upper bounds
        A_ineq = sp.eye((N+1)*n_x + N*n_u)
        l_ineq = np.hstack([np.kron(np.ones(N+1), x_min), np.kron(np.ones(N), u_min)])
        u_ineq = np.hstack([np.kron(np.ones(N+1), x_max), np.kron(np.ones(N), u_max)])

        # compose OSQP constraints
        A_total = sp.vstack([A_eq, A_ineq], format='csc')
        l_total = np.hstack([l_eq, l_ineq])
        u_total = np.hstack([u_eq, u_ineq])
        self.variables.update({'A': A_total, 'l': l_total, 'u': u_total})

        # set up the OSQP problem
        self.solver.setup(P, q, A_total, l_total, u_total, warm_start=True)

        if self.debug:
            self.logger.debug('QP problem setup completed.')

    def update(self, **kwargs):
        """
        Update the MPC problem. Can update both the initial conditions and/or the reference state.
        Input can include:
            - x_0 = initial state
            - x_r = reference state
        """
        # demux variables
        N = self.variables['N']
        Q = self.variables['Q']
        Q_N = self.variables['Q_N']
        n_x = self.variables['n_x']
        x_0 = self.variables['x_0']
        x_r = self.variables['x_r']

        # update x_0 and x_r accordingly to the user input
        for k, v in kwargs.items():
            if k == 'x_0':
                x_0 = v
            if k == 'x_r':
                x_r = v

        # update initial state
        l_total = self.variables['l']
        u_total = self.variables['u']
        l_total[:n_x] = -x_0
        u_total[:n_x] = -x_0
        self.variables.update({'u': u_total, 'l': l_total})

        # update reference state
        q = self.variables['q']
        q[:n_x*N] = np.kron(np.ones(N), -Q.dot(x_r))
        q[n_x*N:n_x*(N+1)] = -Q_N.dot(x_r)
        self.variables.update({'q': q})

        self.solver.update(q=q, l=l_total, u=u_total)

        if self.debug:
            self.logger.debug('QP problem updated correctly.')

    def solve(self):
        """
        Solve the MPC problem.
        """
        res = self.solver.solve()

        # check solver status and return the solution
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')
        else:
            if self.debug:
                self.logger.debug('QP problem solved correctly.')
            return res.x
