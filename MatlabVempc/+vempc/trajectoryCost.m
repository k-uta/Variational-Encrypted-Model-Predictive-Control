function cost = trajectoryCost(xs, us, Q, R, Qf)
%TRAJECTORYCOST Realized LQ cost of a closed-loop trajectory.
%   cost = vempc.trajectoryCost(xs, us, Q, R, Qf) returns
%       sum_k ( x_k' Q x_k + u_k' R u_k ) + x_N' Qf x_N
%   over the realized history. Mirrors TrajectoryCost in the Go/Python cores.
    steps = size(us, 1);
    cost = 0.0;
    for k = 1:steps
        xk = xs(k, :).';
        uk = us(k, :).';
        cost = cost + xk.' * Q * xk + uk.' * R * uk;
    end
    xN = xs(steps + 1, :).';
    cost = cost + xN.' * Qf * xN;
end
