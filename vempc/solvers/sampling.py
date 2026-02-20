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
    # Draw from N(mU, Sigma_U) via mean + L * xi.
    xi = np.random.randn(K, variational.mpc.Nm)
    Us = mU[None, :] + xi @ L.T
    return Us

def sample_variational_control(
    x0,
    variational,
    penalty,
    K=2000,
    cheb_coeffs=None,
    cheb_bound=None,
    cheb_eta=None,
):
    U_samples = sample_tilted(variational, x0, K)

    true_feasible = penalty.is_feasible(U_samples, x0)
    true_accept_num = int(np.sum(true_feasible))  # exact feasibility count

    if cheb_coeffs is not None and cheb_bound is not None and penalty.has_constraints:
        residuals = penalty.constraint_residual(U_samples, x0)
        h_l = qpMPC.eval_relu_poly(residuals, cheb_coeffs, cheb_bound)
        threshold = qpMPC.eval_relu_poly(0.0, cheb_coeffs, cheb_bound)
        # h_thr = np.where(h_l < threshold, 0.0, h_l - threshold)
        h_thr = np.where(h_l < threshold, 0.0, h_l)
        s_l = np.sum(h_thr, axis=1)
        eta = float(cheb_eta)
        # Polynomial surrogate weights: r_l = exp(-eta * s_l).
        w = np.exp(-eta * s_l)
        # Sum of weights; used for monitoring (not a true feasibility rate).
        w_sum = w.sum()
    else:
        w = true_feasible.astype(float)
        # Sum of feasible indicators equals the feasible sample count.
        w_sum = w.sum()


    # Weighted Monte Carlo estimate of the optimal control sequence.
    U_hat = (U_samples * w[:, None]).sum(axis=0) / w_sum
    u0_hat = U_hat[:variational.mpc.m]

    return u0_hat, U_hat, float(w_sum), int(true_accept_num)

