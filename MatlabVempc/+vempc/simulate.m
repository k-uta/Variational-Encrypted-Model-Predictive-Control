function [xs, us, info, elapsed] = simulate(x0, controllerFn, A, B, steps)
%SIMULATE Closed-loop rollout of a controller on x_{k+1} = A x_k + B u_k.
%   [xs, us, info, elapsed] = vempc.simulate(x0, controllerFn, A, B, steps)
%   rolls the system forward for `steps` steps. controllerFn has signature
%
%       [u, Useq, stepInfo] = controllerFn(x, warm)
%
%   where `warm` is the previous step's Useq (warm start, may be []). Returns
%   the state history xs ((steps+1) x n), input history us (steps x m), a
%   per-step info struct array, and the total wall-clock time.
%
%   Mirrors Simulate in vempc/core/mpc.py and GoVempc/core/mpc.go.
    n = size(A, 1);
    m = size(B, 2);
    xs = zeros(steps + 1, n);
    us = zeros(steps, m);
    info = repmat(struct('wSum', NaN, 'accept', NaN, 'iterMs', NaN), steps, 1);

    x = x0(:);
    xs(1, :) = x.';
    warm = [];

    tStart = tic;
    for k = 1:steps
        tIter = tic;
        [u, Useq, stepInfo] = controllerFn(x, warm);
        info(k).iterMs = toc(tIter) * 1000;

        if isstruct(stepInfo)
            if isfield(stepInfo, 'wSum'),   info(k).wSum = stepInfo.wSum;   end
            if isfield(stepInfo, 'accept'), info(k).accept = stepInfo.accept; end
        end

        warm = Useq;
        u = u(:);
        us(k, :) = u(1:m).';
        x = A * x + B * u(1:m);
        xs(k + 1, :) = x.';
    end
    elapsed = toc(tStart);
end
