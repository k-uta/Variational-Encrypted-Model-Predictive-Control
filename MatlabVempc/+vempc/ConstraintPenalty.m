classdef ConstraintPenalty
    %CONSTRAINTPENALTY Feasibility handling for the condensed constraints.
    %
    %   Computes the constraint residual g(U; x0) = G U - h(x0) and exact
    %   feasibility for batches of sampled control sequences. The polynomial
    %   (Chebyshev) surrogate used by the encrypted protocol is applied in
    %   vempc.VariationalMPC/computeWeights; this class supplies the residual
    %   and the diagnostic (hard) feasibility count.
    %
    %   Mirrors vempc/core/constraints.py and GoVempc/core/constraints.go.

    properties
        G               % (p x N*m) stacked constraint matrix, or []
        hFunc           % function handle x0 -> (p x 1) bound vector, or []
        hasConstraints  % logical
        p               % number of scalar constraints
    end

    methods
        function obj = ConstraintPenalty(G, hFunc)
            obj.G = G;
            obj.hFunc = hFunc;
            obj.hasConstraints = ~isempty(G) && ~isempty(hFunc);
            if obj.hasConstraints
                obj.p = size(G, 1);
            else
                obj.p = 0;
            end
        end

        function res = constraintResidual(obj, U, x0)
            % U: (K x N*m) batch of samples. Returns (K x p) residuals,
            % row i equal to G*U_i - h(x0).
            if ~obj.hasConstraints
                res = zeros(size(U, 1), 0);
                return;
            end
            h0 = obj.hFunc(x0);
            res = U * obj.G.' - h0(:).';   % implicit expansion over rows
        end

        function feas = isFeasible(obj, U, x0, tol)
            % Row-wise hard feasibility test (diagnostic only).
            if nargin < 4 || isempty(tol)
                tol = 1e-6;
            end
            if ~obj.hasConstraints
                feas = true(size(U, 1), 1);
                return;
            end
            res = obj.constraintResidual(U, x0);
            feas = all(res <= tol, 2);
        end
    end
end
