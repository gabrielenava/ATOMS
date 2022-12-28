close all
clc
clear

addpath('./../src/')
disp('Testing OptiMPC class ...')

%% Call to OSQP

% Problem structure:
%
% x = argmin (1/2*x'*P*x + x'*q)
%
% s.t.
% l <= A*x <= u

% Define problem data
P = sparse([4, 1; 1, 2]);
q = [1; 1];
A = sparse([1, 1; 1, 0; 0, 1]);
l = [1; 0; 0];
u = [1; 0.7; 0.7];

% Create the solver object
opti = OptiMPC('osqp');

% Setup workspace and change alpha parameter
opti.initialize(P, q, A, l, u);

% Update problem
P_new = sparse([5, 1.5; 1.5, 1]);
A_new = sparse([1.2, 1.1; 1.5, 0; 0, 0.8]);
q_new = [2; 3];
l_new = [2; -1; -1];
u_new = [2; 2.5; 2.5];

opti.updateSelectedParams({'q','u','l'}, {q_new, u_new, l_new});
opti.update(P_new, q_new, A_new, l_new, u_new);

% Solve problem
u_star = opti.solve();

disp('u_star test 1 (OSQP):')
disp(num2str(u_star))

%% Call to QUADPROG

% Setup QP problem
var.H   = sparse([5, 1.5; 1.5, 1]);
var.g   = [2; 3];
var.A   = sparse([1.2, 1.1; 1.5, 0; 0, 0.8]);
var.b   = [2; 2.5; 2.5];
var.Aeq = [];
var.beq = [];
var.lb  = [-10; -5];
var.ub  = [10; 5];
var.x0  = [];

var.options = optimoptions('quadprog','Display','iter');

opti2 = OptiMPC('quadprog');
opti2.updateQuadprogVar(var);

u_star = opti2.solve();

disp('u_star test 2 (QUADPROG):')
disp(num2str(u_star))

disp('Done!')
rmpath('./../src/')
