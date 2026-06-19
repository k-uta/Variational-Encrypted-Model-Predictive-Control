function coeffs = chebyshevReLUCoeffs(order, bound, nSamples)
%CHEBYSHEVRELUCOEFFS Least-squares Chebyshev fit of ReLU on [-bound, bound].
%   coeffs = vempc.chebyshevReLUCoeffs(order, bound, nSamples) returns the
%   (1 x order+1) coefficients of a degree-`order` Chebyshev series that
%   approximates the scalar operator
%
%       h_l(t) ~ bound * max(0, t),    t in [-1, 1]
%
%   so that evaluating chebVal(g/bound, coeffs) approximates max(0, g) (the
%   per-constraint hinge / ReLU). The fit uses Chebyshev nodes and a normal
%   equation, matching ChebyshevReLUCoeffs in GoVempc/solvers/cheb.go.
%
%   Pass nSamples <= 0 (or omit) to use the default 10*order (>= 200) nodes.
    if nargin < 3 || isempty(nSamples) || nSamples <= 0
        nSamples = max(200, 10 * order);
    end
    if order < 0
        order = 0;
    end

    k = (0:nSamples - 1).';
    t = cos(pi * (2 * k + 1) ./ (2 * nSamples));   % Chebyshev nodes in (-1,1)
    y = zeros(nSamples, 1);
    pos = t > 0;
    y(pos) = bound * t(pos);                        % target: bound * ReLU(t)

    % Chebyshev basis T_0..T_order evaluated at the nodes.
    T = zeros(nSamples, order + 1);
    T(:, 1) = 1;
    if order >= 1
        T(:, 2) = t;
    end
    for j = 3:order + 1
        T(:, j) = 2 .* t .* T(:, j - 1) - T(:, j - 2);
    end

    coeffs = (T.' * T) \ (T.' * y);
    coeffs = coeffs(:).';
end
