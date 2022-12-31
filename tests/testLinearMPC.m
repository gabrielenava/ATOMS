close all
clc
clear

addpath('./../src/')
disp('Testing LinearMPC class ...')

%% Setup the problem

% linearized model of a quadcopter
Ax = [1       0       0   0   0   0   0.1     0       0    0       0       0;
      0       1       0   0   0   0   0       0.1     0    0       0       0;
      0       0       1   0   0   0   0       0       0.1  0       0       0;
      0.0488  0       0   1   0   0   0.0016  0       0    0.0992  0       0;
      0      -0.0488  0   0   1   0   0      -0.0016  0    0       0.0992  0;
      0       0       0   0   0   1   0       0       0    0       0       0.0992;
      0       0       0   0   0   0   1       0       0    0       0       0;
      0       0       0   0   0   0   0       1       0    0       0       0;
      0       0       0   0   0   0   0       0       1    0       0       0;
      0.9734  0       0   0   0   0   0.0488  0       0    0.9846  0       0;
      0      -0.9734  0   0   0   0   0      -0.0488  0    0       0.9846  0;
      0       0       0   0   0   0   0       0       0    0       0       0.9846];

Bu = [ 0      -0.0726  0       0.0726;
      -0.0726  0       0.0726  0;
      -0.0152  0.0152 -0.0152  0.0152;
       0      -0.0006 -0.0000  0.0006;
       0.0006  0      -0.0006  0;
       0.0106  0.0106  0.0106  0.0106;
       0      -1.4512  0       1.4512;
      -1.4512  0       1.4512  0;
      -0.3049  0.3049 -0.3049  0.3049;
       0      -0.0236  0       0.0236;
       0.0236  0      -0.0236  0;
       0.2107  0.2107  0.2107  0.2107];

% setup the MPC problem
var.N     = 10;
var.Ax    = Ax;
var.Bu    = Bu;
var.Q_N   = diag([0 0 10 10 10 10 0 0 0 5 5 5]);
var.Q     = diag([0 0 10 10 10 10 0 0 0 5 5 5]);
var.R     = 0.1*eye(size(Bu,2));
var.x_r   = [0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0];
var.x_0   = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
var.x_min = [-pi/6; -pi/6; -Inf; -Inf; -Inf; -1; -Inf(6,1)];
var.x_max = [ pi/6;  pi/6;  Inf;  Inf;  Inf; Inf; Inf(6,1)];
var.u_min = [9.6; 9.6; 9.6; 9.6] - 10.5916;
var.u_max = [13; 13; 13; 13] - 10.5916;

opti = LinearMPC();
opti.setup(var);

% simulate in closed loop
n_sim = 20;
x     = zeros(n_sim,1);
x_0   = var.x_0;

for k = 1:n_sim

    % solve the problem
    u_star = opti.solve();

    % apply first control input and update initial conditions
    x(k) = x_0(3);
    u    = u_star((var.N+1)*size(Ax,1)+1:(var.N+1)*size(Ax,1)+size(Bu,2));
    x_0  = Ax*x_0 + Bu*u;
    
    opti.update(x_0);
end

% plot the result for the third component of x (the only one diff. from 0)
figure(1)
hold on
grid on
plot(0:n_sim-1, x, 'r', 'linewidth', 2)
plot(0:n_sim-1, x, '.k', 'markersize', 12)
title('State x(3)')
xlabel('iters')
ylabel('state')

disp('Done!')
rmpath('./../src/')
