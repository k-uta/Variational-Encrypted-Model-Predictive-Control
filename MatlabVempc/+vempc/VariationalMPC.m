classdef VariationalMPC
    %VARIATIONALMPC Variational reformulation of MPC with tilted Gaussian sampling.
    %
    %   Implements the exponentially tilted reference distribution of Theorem 1
    %   of the VEMPC paper. The quadratic cost is absorbed into the sampling
    %   distribution kappa_tilde = N(m_U(x0), Sigma_U), so that online weighting
    %   reduces to a polynomial (Chebyshev) feasibility surrogate.
    %
    %       Sigma_U = (Sigma_0^{-1} + (1/lambda) H)^{-1}                  (14b)
    %       m_U(x0) = -(1/lambda) Sigma_U S' x0                           (14a)
    %       U^{(i)} = m_U(x0) + L_U xi^{(i)},   xi ~ N(0, I)              (22)
    %
    %   Mirrors vempc/core/variational.py and GoVempc/core/variational.go.
    %
    %   Construction:
    %       v = vempc.VariationalMPC(mpc, penalty, lambda, Sigma0)

    properties
        mpc         % vempc.MPCProblem
        penalty     % vempc.ConstraintPenalty
        lambda      % temperature parameter
        Sigma0      % (Nm x Nm) prior covariance for kappa_0 = N(0, Sigma0)
        SigmaU      % (Nm x Nm) tilted covariance
        LU          % (Nm x Nm) lower-triangular Cholesky factor of SigmaU
    end

    methods
        function obj = VariationalMPC(mpc, penalty, lambda, Sigma0)
            obj.mpc = mpc;
            obj.penalty = penalty;
            obj.lambda = lambda;
            if nargin < 4 || isempty(Sigma0)
                Sigma0 = eye(mpc.Nm);
            end
            obj.Sigma0 = Sigma0;

            SigmaU = inv(inv(Sigma0) + mpc.H / lambda);   %#ok<MINV>
            SigmaU = (SigmaU + SigmaU.') / 2;             % symmetrize for chol
            obj.SigmaU = SigmaU;
            obj.LU = chol(SigmaU, 'lower');               % SigmaU = LU * LU'
        end

        function m = mU(obj, x0)
            % Tilted mean m_U(x0) = -(1/lambda) Sigma_U S' x0  (column vector).
            m = -(1 / obj.lambda) * (obj.SigmaU * (obj.mpc.S.' * x0(:)));
        end

        function U = sampleKappaTilde(obj, x0, K, seed)
            % Draw K samples U^{(i)} ~ N(m_U(x0), Sigma_U), one per row.
            if nargin >= 4 && ~isempty(seed)
                rng(seed, 'twister');
            end
            meanU = obj.mU(x0);                  % Nm x 1
            xi = randn(K, obj.mpc.Nm);           % K x Nm
            U = meanU.' + xi * obj.LU.';         % U_i = m + LU * xi_i
        end

        function [weights, wSum] = computeWeights(obj, U, x0, coeffs, bound, eta)
            % Polynomial-surrogate feasibility weights for tilted samples.
            % Working in log-space:  log w_i = -eta * sum_j [ h_l(g_ij) ]_above
            % where the threshold h_l(0) implements the Corollary-2 correction
            % so feasible samples keep weight ~ 1.
            K = size(U, 1);

            res = obj.penalty.constraintResidual(U, x0);   % K x p
            t = res / bound;
            h = vempc.chebVal(t, coeffs);                  % K x p, ~ ReLU(res)
            threshold = vempc.chebVal(0.0, coeffs);        % h_l(0)

            mask = h > threshold;
            s = sum(h .* mask, 2);                         % K x 1 aggregate score
            logW = -eta * s;

            maxLog = max(logW);
            if ~isfinite(maxLog)
                maxLog = 0.0;
            end
            w = exp(logW - maxLog);
            wSum = sum(w);
            if wSum > 0
                weights = w / wSum;
            else
                weights = zeros(K, 1);
            end
        end
    end
end
