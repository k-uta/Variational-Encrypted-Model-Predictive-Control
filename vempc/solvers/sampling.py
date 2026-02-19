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
    xi = np.random.randn(K, variational.mpc.Nm)
    Us = mU[None, :] + xi @ L.T
    return Us

def sample_variational_control(
    x0,
    variational,
    penalty,
    K=2000,
    eps=1e-12,
    cheb_coeffs=None,
    cheb_bound=None,
    cheb_clip=True,
    cheb_eta=None,
):
    U_samples = sample_tilted(variational, x0, K)

    true_feasible = penalty.is_feasible(U_samples, x0)
    true_accept_num = int(np.sum(true_feasible))

    if cheb_coeffs is not None and cheb_bound is not None and penalty.has_constraints:
        residuals = penalty.constraint_residual(U_samples, x0)
        h_l = qpMPC.eval_relu_poly(residuals, cheb_coeffs, cheb_bound, clip=cheb_clip)
        s_l = np.sum(h_l, axis=1)
        eta = float(cheb_eta)
        w = np.exp(-eta * s_l)
        # accept_rate = float(np.mean(w)) # This is the surrogate weight average
        accept_rate = w.sum()
    else:
        w = true_feasible.astype(float)
        # accept_rate = float(true_feasible.mean())
        accept_rate = w.sum()

    # 2. Use the exact physical count instead of the surrogate weight threshold
    accept_num = true_accept_num

    w_sum = w.sum()

    U_hat = (U_samples * w[:, None]).sum(axis=0) / w_sum
    u0_hat = U_hat[:variational.mpc.m]

    info = {
        "w_sum": float(w_sum),
        "fallback": False,
        "accept_rate": accept_rate,
        "accept_num": accept_num,
        "w_max": float(w.max()),
        "w_min": float(w.min()),
    }
    return u0_hat, U_hat, info


# def sample_variational_control(
#     x0,
#     variational,
#     penalty,
#     K=2000,
#     eps=1e-12,
#     cheb_coeffs=None,
#     cheb_bound=None,
#     cheb_clip=True,
#     cheb_eta=None,
# ):
#     """
#     Variational MPC using polynomial surrogate weights.

#     If cheb_coeffs is provided, uses a Chebyshev approximation of ReLU
#     for constraint residuals and weights:
#         r_l = exp(-eta * sum_j h_l(g_j)).

#     Returns:
#       u0_hat (applied input),
#       U_hat  (estimated sequence),
#       info   (acceptance, weight stats)
#     """
#     U_samples = sample_tilted(variational, x0, K)

#     if cheb_coeffs is not None and cheb_bound is not None and penalty.has_constraints:
#         residuals = penalty.constraint_residual(U_samples, x0)
#         h_l = qpMPC.eval_relu_poly(residuals, cheb_coeffs, cheb_bound, clip=cheb_clip)
#         s_l = np.sum(h_l, axis=1)
#         eta = 1.0 if cheb_eta is None else float(cheb_eta)
#         w = np.exp(-eta * s_l)
#         accept_rate = float(np.mean(w))
#     else:
#         feasible = penalty.is_feasible(U_samples, x0)
#         w = feasible.astype(float)
#         accept_rate = float(feasible.mean())

#     accept_num = int(np.sum(w > 0.5))
#     w_sum = w.sum()
#     if w_sum <= eps:
#         U_hat = variational.m_U(x0).copy()
#         u0_hat = U_hat[:variational.mpc.m]
#         info = {"w_sum": float(w_sum), "fallback": True, "accept_rate": accept_rate,"accept_num": accept_num,}
#         return u0_hat, U_hat, info

#     U_hat = (U_samples * w[:, None]).sum(axis=0) / w_sum
#     u0_hat = U_hat[:variational.mpc.m]

#     info = {
#         "w_sum": float(w_sum),
#         "fallback": False,
#         "accept_rate": accept_rate,
#         "accept_num": accept_num,
#         "w_max": float(w.max()),
#         "w_min": float(w.min()),
#     }
#     return u0_hat, U_hat, info
