classdef LinearMPC < handle
    % LINEARMPC implements a constrained linear-quadratic MPC using OSQP.
    %
    % LinearMPC solves the following problem:
    %
    %   min (x_N - x_r)^T*Q_N*(x_N - x_r) + ...
    %       \sum_(k=0)^(N-1) (x_k - x_r)^T*Q*(x_k - x_r) u_k^T*R*u_k
    %
    %      s.t. x_(k+1) = A*x_k + B*u_k
    %           x_min <= x_k <= x_max
    %           u_min <= u_k <= u_max
    %           x_0 = \bar{x}
    %
    % Author: Gabriele Nava, gabriele.nava@iit.it
    % Dec. 2022
    %
    properties
        P
        q
        A
        l
        u
        solver
    end

    methods
        function obj = LinearMPC()

            obj.P      = [];
            obj.q      = [];
            obj.A      = [];
            obj.l      = [];
            obj.u      = [];
            obj.solver = [];
        end

        function [] = setup(obj, var)

            % setup the MPC problem. Required inputs:
            %
            % var.N     = number of steps;
            % var.Ax    = matrix A of the system \dot{x} = Ax + Bu
            % var.Bu    = matrix B of the system \dot{x} = Ax + Bu
            % var.Q_N   = weight of final cost
            % var.Q     = weight of the step-by-step cost
            % var.R     = weight of the input cost
            % var.x_r   = reference state
            % var.x_0   = initial state
            % var.x_min = state lower bound
            % var.x_max = state upper bound
            % var.u_min = input lower bound
            % var.u_max = input upper bound
            %
            N     = var.N;
            Ax    = var.Ax;
            Bu    = var.Bu;
            Q_N   = var.Q_N;
            Q     = var.Q;
            R     = var.R;
            x_r   = var.x_r;
            x_0   = var.x_0;
            x_min = var.x_min;
            x_max = var.x_max;
            u_min = var.u_min;
            u_max = var.u_max;

            % compute the hessian
            %
            % input state y is:
            %
            %   y = [x(0); x(1); ...; x(N); u(0); ...; u(N-1)]
            %
            % so, the Hessian must be build as follows:
            %
            %   P = [Q   0 ... R ... 0;
            %        0   Q ... 0 ... 0;
            %              ...
            %        0   0 ... 0 ... R];
            %
            Px    = kron(eye(N), Q);
            Pu    = kron(eye(N), R);
            Pxn   = Q_N;
            obj.P = blkdiag(Px, Pxn, Pu);

            % compute the gradient
            %
            % must be build as
            %
            %   q = [-Q*x_r; ...; -Qf*x_r; ...; 0]
            %
            % the term x_r'*Q*x_r does not affect the gradient
            %
            qx    =  repmat(-Q*x_r, N, 1);
            qu    = -Q_N*x_r;
            qxn   =  zeros(N*size(Bu, 2), 1);
            obj.q = [qx; qxn; qu];

            % compute the constraints
            %
            % linear dynamics and initial conditions
            %
            %   expand the linear dynamics constraint to each state
            %
            %     0 = -x_(k+1) + A*x_k + B*u_k (dynamics)
            %
            %     A_dyn = [-1   0 ... 0  0
            %               Ax -1 ... 0  0
            %               0   0 ... Ax -1]
            %
            %     B_dyn = [0 0  ... 0
            %              0 Bu ... 0
            %              0 0  ... Bu]
            %
            %     leq = ueq = [-x0; 0; 0]
            %
            A_dyn = kron(eye(N+1), -eye(size(Ax,1))) + kron(diag(ones(N, 1), -1), Ax);
            B_dyn = kron([zeros(1, N); eye(N)], Bu);
            Aeq   = [A_dyn, B_dyn];
            leq   = [-x_0; zeros(N*size(Ax,1), 1)];
            ueq   = leq;

            % compute the bounds on state and input
            Aineq = eye((N+1)*size(Ax,1) + N*size(Bu,2));
            lineq = [repmat(x_min, N+1, 1); repmat(u_min, N, 1)];
            uineq = [repmat(x_max, N+1, 1); repmat(u_max, N, 1)];

            % formulate OSQP constraints
            obj.A = [Aeq; Aineq];
            obj.l = [leq; lineq];
            obj.u = [ueq; uineq];

            % setup the OSQP problem
            obj.solver = osqp();
            obj.solver.setup(obj.P, obj.q, obj.A, obj.l, obj.u, 'warm_start', true);
        end

        function [] = update(obj, x_0)

            % update the MPC problem.
            %
            % WARNING! For the moment, only the initial state x_0 can be
            % updated with this method. The reason is that the call to
            % osqp.update() does not always work fine for the Hessian and
            % gradient. If you need to update variables other than the
            % initial state, use obj.setup().
            %
            obj.l(1:length(x_0)) = -x_0;
            obj.u(1:length(x_0)) = -x_0;

            obj.solver.update('l', obj.l, 'u', obj.u);
        end

        function u_star = solve(obj)

            % solve the MPC problem
            sol    = obj.solver.solve();
            u_star = sol.x;

            if ~strcmp(sol.info.status, 'solved')
                error('OSQP did not solve the problem!')
            end
        end
    end
end
