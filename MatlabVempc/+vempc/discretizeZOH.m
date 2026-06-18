function [Ad, Bd] = discretizeZOH(Ac, Bc, dt)
%DISCRETIZEZOH Zero-order-hold discretization via the matrix exponential.
%   [Ad, Bd] = vempc.discretizeZOH(Ac, Bc, dt) discretizes the continuous-time
%   system (Ac, Bc) with sampling period dt using the augmented matrix
%   exponential
%
%       [Ad Bd; 0 I] = expm([Ac Bc; 0 0] * dt).
%
%   This matches the ZOH discretization in GoVempc/examples/pendulum.go and
%   requires no Control System Toolbox (expm is a core MATLAB function).
    n = size(Ac, 1);
    m = size(Bc, 2);
    M = expm([Ac, Bc; zeros(m, n + m)] * dt);
    Ad = M(1:n, 1:n);
    Bd = M(1:n, n+1:n+m);
end
