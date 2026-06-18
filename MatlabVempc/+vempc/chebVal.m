function y = chebVal(t, coeffs)
%CHEBVAL Evaluate a Chebyshev series via the Clenshaw recurrence.
%   y = vempc.chebVal(t, coeffs) evaluates
%       sum_{k=0}^{order} coeffs(k+1) * T_k(t)
%   where T_k are Chebyshev polynomials of the first kind. t may be a scalar,
%   vector, or matrix; the evaluation is elementwise and y has size(t).
%
%   This reproduces chebVal in GoVempc/core/variational.go bit-for-bit.
    coeffs = coeffs(:);
    nc = numel(coeffs);
    if nc == 0
        y = zeros(size(t));
        return;
    end
    if nc == 1
        y = coeffs(1) * ones(size(t));
        return;
    end
    b0 = zeros(size(t));
    b1 = zeros(size(t));
    for i = nc:-1:2
        b2 = b1;
        b1 = b0;
        b0 = 2 .* t .* b1 - b2 + coeffs(i);
    end
    y = t .* b0 - b1 + coeffs(1);
end
