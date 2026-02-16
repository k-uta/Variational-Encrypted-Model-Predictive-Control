import numpy as np
from . import qpMPC

def sample_tilted(variational, x0, K):
    """
    Draw samples from the tilted Gaussian using core parameters,
    matching the original sampling orientation for reproducibility.
    Returns array shape (K, Nm).
    """
    mU = variational.m_U(x0)
    L = variational.L_U
    xi = np.random.randn(variational.mpc.Nm, K)
    Us = mU[:, None] + L @ xi
    return Us.T


def sample_variational_control(
    x0,
    variational,
    penalty,
    K=2000,
    eps=1e-12,
    cheb_coeffs=None,
    cheb_bound=None,
    cheb_clip=True,
    cheb_weight_mode="product",
    cheb_eta=None,
):
    """
    Variational MPC using indicator weights.

    If cheb_coeffs is provided, uses a Chebyshev approximation of the
    indicator function for constraint residuals.

    Returns:
      u0_hat (applied input),
      U_hat  (estimated sequence),
      info   (acceptance, weight stats)
    """
    U_samples = sample_tilted(variational, x0, K)

    if cheb_coeffs is not None and cheb_bound is not None and penalty.has_constraints:
        residuals = penalty.constraint_residual(U_samples, x0)
        r = qpMPC.eval_indicator_poly(residuals, cheb_coeffs, cheb_bound, clip=cheb_clip)
        if cheb_weight_mode == "product":
            w = np.prod(r, axis=1)
        elif cheb_weight_mode == "exp":
            if cheb_eta is None:
                cheb_eta = 1.0
            w = np.exp(-cheb_eta * np.sum(1.0 - r, axis=1))
        else:
            raise ValueError("cheb_weight_mode must be 'product' or 'exp'")
        accept_rate = float(np.mean(w))
    else:
        feasible = penalty.is_feasible(U_samples, x0)
        w = feasible.astype(float)
        accept_rate = float(feasible.mean())

    w_sum = w.sum()
    if w_sum <= eps:
        U_hat = variational.m_U(x0).copy()
        u0_hat = U_hat[:variational.mpc.m]
        info = {"w_sum": float(w_sum), "fallback": True, "accept_rate": accept_rate}
        return u0_hat, U_hat, info

    U_hat = (U_samples * w[:, None]).sum(axis=0) / w_sum
    u0_hat = U_hat[:variational.mpc.m]

    info = {
        "w_sum": float(w_sum),
        "fallback": False,
        "accept_rate": accept_rate,
        "w_max": float(w.max()),
        "w_min": float(w.min()),
        "cheb_weight_mode": cheb_weight_mode if cheb_coeffs is not None else "hard",
    }
    return u0_hat, U_hat, info

