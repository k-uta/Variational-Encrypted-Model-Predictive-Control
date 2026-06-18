function [u0, Useq, wSum, trueAccept] = sampleVariationalControl( ...
        x0, variational, penalty, K, coeffs, bound, eta, seed)
%SAMPLEVARIATIONALCONTROL One online step of variational MPC (plaintext).
%   [u0, Useq, wSum, trueAccept] = vempc.sampleVariationalControl(...)
%   draws K tilted samples, computes the polynomial-surrogate feasibility
%   weights, and returns the Monte-Carlo estimator
%
%       Uhat = sum_i w_i U^{(i)},     u0 = first input block of Uhat.
%
%   wSum is the (unnormalized) weight mass and trueAccept is the number of
%   exactly feasible samples (diagnostic). Mirrors
%   GoVempc/solvers/sampling.go : SampleVariationalControl.
    if nargin < 8
        seed = [];
    end

    U = variational.sampleKappaTilde(x0, K, seed);     % K x Nm
    trueAccept = sum(penalty.isFeasible(U, x0, 1e-6));

    [weights, wSum] = variational.computeWeights(U, x0, coeffs, bound, eta);

    Nm = size(U, 2);
    mIn = variational.mpc.m;
    if wSum == 0
        u0 = zeros(mIn, 1);
        Useq = zeros(Nm, 1);
        return;
    end

    Uhat = (weights.' * U).';        % Nm x 1 weighted average
    Useq = Uhat;
    u0 = Uhat(1:mIn);
end
