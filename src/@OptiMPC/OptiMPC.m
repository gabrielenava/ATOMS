classdef OptiMPC < handle
    % OPTIQP wraps Quadratic Programming optimization with OSQP and QUADPROG
    %
    % Author: Gabriele Nava, gabriele.nava@iit.it
    % Dec. 2022
    %
    properties
        opti_type
        solver
        quadprog_var
    end

    methods
        function obj = OptiMPC(opti_type)

            obj.opti_type    = opti_type;
            obj.quadprog_var = struct;

            switch obj.opti_type

                case 'osqp'

                    obj.solver = osqp();

                case 'quadprog'

                    obj.solver           = [];
                    obj.quadprog_var.H   = [];
                    obj.quadprog_var.g   = [];
                    obj.quadprog_var.A   = [];
                    obj.quadprog_var.b   = [];
                    obj.quadprog_var.Aeq = [];
                    obj.quadprog_var.beq = [];
                    obj.quadprog_var.lb  = [];
                    obj.quadprog_var.ub  = [];
                    obj.quadprog_var.x0  = [];
                    obj.quadprog_var.options = [];

                otherwise
                    error('opti_type not recognized.')
            end
        end

        function [] = initialize(obj, P, q, A, l, u)

            % setup workspace. OSQP solver only
            obj.solver.setup(P, q, A, l, u, 'alpha', 1);
        end

        function [] = update(obj, P, q, A, l, u)

            % update workspace. OSQP solver only
            obj.solver.update('Px', nonzeros(triu(P)), 'Ax', nonzeros(A), 'q', q, 'l', l, 'u', u);
        end

        function [] = updateSelectedParams(obj, names, values)

            % update the selected parameters. 'names' is a cell array of
            % strings, while 'values' another cell array of doubles with
            % the corresponding values to update
            if length(names) ~= length(values)

                error('''names'' and ''values'' inputs do not have the same length.')
            else
                for k = 1:length(names)

                    switch names{k}

                        case 'P'
                            obj.solver.update('Px', nonzeros(triu(values{k})));

                        case 'q'
                            obj.solver.update('q', values{k})

                        case 'A'
                            obj.solver.update('Ax', nonzeros(values{k}));

                        case 'u'
                            obj.solver.update('u', values{k});

                        case 'l'
                            obj.solver.update('l', values{k});

                        otherwise
                            error('Parameter name not found.')
                    end
                end
            end
        end

        function [] = updateQuadprogVar(obj, variables)

            % update opti variables. QUADPROG only
            variables_names = fieldnames(variables);

            for k = 1:length(variables_names)

                obj.quadprog_var.(variables_names{k}) = variables.(variables_names{k});
            end
        end

        function u_star = solve(obj)

            % solve the QP problem
            switch obj.opti_type

                case 'osqp'
                    % assuming a call to OSQP
                    sol    = obj.solver.solve();
                    u_star = sol.x;

                    if ~strcmp(sol.info.status, 'solved')
                        error('OSQP did not solve the problem!')
                    end

                case 'quadprog'
                    % assuming a call to QUADPROG
                    var    = obj.quadprog_var;
                    u_star = quadprog(var.H, var.g, var.A, var.b, var.Aeq, ...
                        var.beq, var.lb, var.ub, var.x0, var.options);
            end
        end
    end
end
