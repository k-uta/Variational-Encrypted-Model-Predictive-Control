function [Ac, Bc] = invertedPendulum(m, l, g)
%INVERTEDPENDULUM Continuous-time linearized inverted pendulum (paper model).
%   [Ac, Bc] = vempc.invertedPendulum(m, l, g) returns the continuous-time
%   state-space matrices of an inverted pendulum linearized about the upright
%   equilibrium (small-angle approximation).
%
%   State  x = [theta; theta_dot]   (angle, angular velocity)
%   Input  u = torque applied at the pivot
%
%   This mirrors GoVempc/examples/pendulum.go and the numerical example of
%   the VEMPC paper (Section V).
    Ac = [0,   1;
          g/l, 0];
    Bc = [0;
          1/(m*l*l)];
end
