'''
classdef LinearMPC < handle
    % LINEARMPC implement constrained linear-quadratic Model Predicitve Control.
    %           The expected input system is a linear time-invariant system
    %           of the form x(k+1) = A*x(k) + B*u(k).
    %
    % LinearMPC solves the following problem:
    %
    %   minimize (x(N) - x_r)^T*Q_N*(x(N) - x_r) + ...
    %       \sum_(k=0)^(N-1) (x(k) - x_r)^T*Q*(x(k) - x_r) + u(k)^T*R*u(k)
    %
    %      s.t. x(k+1) = A*x(k) + B*u(k)
    %           x_min <= x(k) <= x_max
    %           u_min <= u(k) <= u_max
    %           x(0) = \bar{x}
    %
    % Author: Gabriele Nava, gabriele.nava@iit.it
    % Jan. 2023
    %
    properties
        H
        g
        Aeq
        leq
        ueq
        ub
        lb
        opts
        debug
    end

    methods
        function obj = LinearMPC(varargin)

            obj.H     = [];
            obj.g     = [];
            obj.Aeq   = [];
            obj.leq   = [];
            obj.ueq   = [];
            obj.ub    = [];
            obj.lb    = [];
            obj.opts  = optimset();

            switch nargin
              case 1
                obj.debug = varargin{1};
              otherwise
                obj.debug = false;
            endswitch
        end

        function [] = setup(obj, var)

            % setup the MPC problem. Required inputs:
            %
            % var.N     = number of steps;
            % var.A     = matrix A of the system \dot{x} = Ax + Bu
            % var.B     = matrix B of the system \dot{x} = Ax + Bu
            % var.Q_N   = weight of final cost
            % var.Q     = weight of the step-by-step cost
            % var.R     = weight of the input cost
            % var.x_r   = reference state
            % var.x_0   = initial state
            % var.x_min = state lower bound
            % var.x_max = state upper bound
            % var.u_min = input lower bound
            % var.u_max = input upper bound
            % var.opts  = QP problem options
            %
            N     = var.N;
            A     = var.A;
            B     = var.B;
            Q_N   = var.Q_N;
            Q     = var.Q;
            R     = var.R;
            x_r   = var.x_r;
            x_0   = var.x_0;
            x_min = var.x_min;
            x_max = var.x_max;
            u_min = var.u_min;
            u_max = var.u_max;

            % compute the Hessian matrix
            %
            % input state of the MPC problem is:
            %
            %   x = [x(0); x(1); ...; x(N); u(0); ...; u(N-1)]
            %
            % so, the Hessian must be build as follows:
            %
            %   H = [Q   0 ... R ... 0;
            %        0   Q ... 0 ... 0;
            %              ...
            %        0   0 ... 0 ... R];
            %
            Hx    = kron(eye(N), Q);
            Hu    = kron(eye(N), R);
            Hxn   = Q_N;
            obj.H = blkdiag(Hx, Hxn, Hu);

            % compute the gradient
            %
            % the gradient must be build as
            %
            %   h = [-Q*x_r; ...; -Q_N*x_r; ...; 0]
            %
            % note: the term x_r'*Q*x_r does not affect the QP solution
            %
            gx    =  repmat(-Q*x_r, N, 1);
            gxn   = -Q_N*x_r;
            gu    =  zeros(N*size(B, 2), 1);
            obj.g = [gx; gxn; gu];

            % compute the constraints
            %
            % equality constraints: linear dynamics and initial conditions
            %
            %     0 = -x(k+1) + A*x(k) + B*u(k) (dynamics)
            %
            %     A_dyn = [-1   0 ... 0  0
            %               A  -1 ... 0  0
            %               0   0 ... A -1]
            %
            %     B_dyn = [0  0  ... 0
            %              0  B  ... 0
            %              0  0  ... B]
            %
            %     leq = ueq = [-x0; 0; 0]
            %
            A_dyn   = kron(eye(N+1), -eye(size(A,1))) + kron(diag(ones(N, 1), -1), A);
            B_dyn   = kron([zeros(1, N); eye(N)], B);
            obj.Aeq = [A_dyn, B_dyn];
            obj.leq = [-x_0; zeros(N*size(A,1), 1)];
            obj.ueq = obj.leq;

            % compute the bounds on state and input
            obj.lb  = [repmat(x_min, N+1, 1); repmat(u_min, N, 1)];
            obj.ub  = [repmat(x_max, N+1, 1); repmat(u_max, N, 1)];

            % setup the QP problem options
            obj.opts = var.opts;
        end

        function [] = update(obj, var)

            % update the MPC problem. Required inputs:
            %
            % var.N     = number of steps;
            % var.Q_N   = weight of final cost
            % var.Q     = weight of the step-by-step cost
            % var.x_r   = reference state
            % var.x_0   = initial state
            %
            N     = var.N;
            Q_N   = var.Q_N;
            Q     = var.Q;
            x_r   = var.x_r;
            x_0   = var.x_0;

            % update initial conditions
            obj.leq(1:length(x_0)) = -x_0;
            obj.ueq(1:length(x_0)) = -x_0;

            % recompute the gradient
            obj.g(1:length(x_r)*N) = repmat(-Q*x_r, N, 1);
            obj.g(length(x_r)*N+1:length(x_r)*(N+1)) = -Q_N*x_r;
        end

        function u_star = solve(obj)

            % solve the MPC problem
            [u_star, ~, info, ~] = qp([], obj.H, obj.g, [], [], obj.lb, obj.ub, ...
                                      obj.leq, obj.Aeq, obj.ueq, obj.opts);

            switch info.info
              case 0
                if obj.debug
                  disp('Global solution found.')
                endif
              case 1
                if obj.debug
                  warning('Problem is non convex, but local solution found.')
                endif
              case 2
                error('Problem is non convex and unbounded.')
              case 3
                error('Max number of iterations reached.')
              otherwise
                error('Problem is unfeasible.')
            end
        end
    end
end
'''''

import osqp
import numpy as np
from scipy import sparse as sp


class LinearMPC:
    """
    LinearMPC class: implements a constrained linear-quadratic MPC via OSQP.

        The class solves the following optimal control problem:

            minimize (x(N)-x_r)^T*QN*(x(N)-x_r) + sum_{k=1}^{N-1}[(x(k)-x_r)^T*Q*(x(k)-x_r) + u(k)^T*R*u(k)]
                  s.t.
                      x(k+1) = A*x(k) + B*u(k)
                      x_min <= x(k) <= x_max
                      u_min <= u(k) <= u_max
    """
    def __init__(self):
        self.variables = {}

    def __str__(self):
        return f" LinearMPC class object \n" \
               f" System state size: {self.variables} \n" \
               f" Input vector size: {self.variables} \n" \
               f" Number of steps: {self.variables}"
