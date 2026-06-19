function [vx, vu] = maxConstraintViolation(xs, us, Gx, hx, Gu, hu)
%MAXCONSTRAINTVIOLATION Largest positive state/input constraint violation.
%   [vx, vu] = vempc.maxConstraintViolation(xs, us, Gx, hx, Gu, hu) returns the
%   maximum positive violation of the per-step constraints Gx*x <= hx (over
%   x_1..x_N) and Gu*u <= hu (over u_0..u_{N-1}); 0 means feasible.
%   Mirrors MaxConstraintViolation in the Go/Python cores.
    vx = 0.0;
    if size(xs, 1) > 1 && ~isempty(Gx)
        resX = xs(2:end, :) * Gx.' - hx(:).';
        vx = max(0.0, max(resX(:)));
    end

    vu = 0.0;
    if size(us, 1) > 0 && ~isempty(Gu)
        resU = us * Gu.' - hu(:).';
        vu = max(0.0, max(resU(:)));
    end
end
