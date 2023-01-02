close all
clear all
clc

addpath('./../src/')
disp('Testing LinearMPC class ...')

% Setup the problem
%
% consider a simple double integrator model of the form:
%
%  y       = [x; \dot{x}]
%  \dot{y} = [\dot{x}; \ddot{x} = u].
%
% we write the system in its state form:
%
%   \dot{y} = A*y + B*u = [0 1; 0 0]*y + [0; 1]*u
%
% and we try to setup a constrained linear-quadratic MPC problem to stabilize the
% system along the given trajectory y_r = y_r(t).
%

% write the model. we choose dim(x) = 2 and dim(u) = 2
n_x = 2;
n_u = 2;
A   = [zeros(n_x) eye(n_x); zeros(n_x) zeros(n_x)];
B   = [zeros(n_x, n_u); eye(n_x, n_u)];

% setup the MPC problem
var.N     = 10;
var.A     = A;
var.B     = B;
var.Q_N   = diag([2; 2; 2e3; 2e3]);
var.Q     = diag([1; 1; 1e3; 1e3]);
var.R     = 0.1*eye(n_u);
var.x_r   = [sin(0); 2*cos(0); cos(0); -2*sin(0)];
var.x_0   = [[0; 2]; [1; 0]];
var.x_min = [-2*pi; -2*pi; -100*pi/180; -100*pi/180];
var.x_max = [ 2*pi;  2*pi;  100*pi/180;  100*pi/180];
var.u_min = [-50; -50];
var.u_max = [ 50;  50];
var.opts  = optimset();

opti = LinearMPC(true);
opti.setup(var);

% simulate the problem in closed loop
time     = 0;
n_sim    = 30;
y        = zeros(n_sim, 2*n_x);
y_r      = zeros(n_sim, 2*n_x);

for k = 1:n_sim

    % solve the problem
    u_star = opti.solve();

    % apply first control input and update initial conditions
    y(k, :) = var.x_0;
    u       = u_star((var.N+1)*n_x+1:(var.N+1)*n_x+n_u);
    var.x_0 = A*var.x_0 + B*u;

    % update the reference trajectory
    y_r(k, :) = var.x_r;
    time      = time + 0.1;
    var.x_r   = [sin(2*pi*time); 2*cos(2*pi*time); cos(2*pi*time); -2*sin(2*pi*time)];

    opti.update(var);
endfor

% plot the results
figure(1)

for i = 1:2*n_x

  subplot(2,2,i)
  hold on
  grid on
  plot(0:n_sim-1, y(:,i), 'r', 'linewidth', 2)
  plot(0:n_sim-1, y(:,i), '.k', 'markersize', 12)
  plot(0:n_sim-1, y_r(:,i), '--b', 'linewidth', 2)
  xlabel('iters')
  ylabel(['state ' num2str(i)])
endfor

rmpath('./../src/')
disp('Done!')

