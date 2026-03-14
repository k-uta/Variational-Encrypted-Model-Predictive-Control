import numpy as np
from . import qpMPC

def sample_variational_control(
    x0,
    variational,
    penalty,
    K=2000,
    cheb_coeffs=None,
    cheb_bound=None,
    cheb_eta=None,
):

    U_samples = variational.sample_kappa_tilde(x0, K)


    true_feasible = penalty.is_feasible(U_samples, x0)
    true_accept_num = int(np.sum(true_feasible))  # exact feasibility count

    if cheb_coeffs is not None and cheb_bound is not None and penalty.has_constraints:

        # Compute per-constraint polynomial surrogate h_l(g_j)
        residuals = penalty.constraint_residual(U_samples, x0)
        h_l = qpMPC.eval_relu_poly(residuals, cheb_coeffs, cheb_bound)

        # Aggregate violation score s_l = sum_j h_l(g_j)
        s_l = np.sum(h_l, axis=1)

        # Threshold at aggregate level per Corollary 1: tau_s = h_l(0)
        p = residuals.shape[1]  # number of constraints
        tau_s = p * float(qpMPC.eval_relu_poly(0.0, cheb_coeffs, cheb_bound))
        s_l_bar = np.maximum(s_l - tau_s, 0.0)

        # Polynomial surrogate desirability: r_bar_l = exp(-eta * s_bar_l)
        eta = float(cheb_eta)
        # Polynomial surrogate weights: r_l = exp(-eta * s_l_bar).
        w = np.exp(-eta * s_l_bar)
        # Sum of weights; used for monitoring (not a true feasibility rate).

    else:
        w = true_feasible.astype(float)
        # Sum of feasible indicators equals the feasible sample count.

    w_sum = w.sum()

    if w_sum == 0.0:
        raise RuntimeError(
            f"sample_variational_control: all weights are zero "
            f"(K={K}, true_accept_num={true_accept_num}). "
            f"Increase K, reduce eta, or widen Sigma_0."
        )


    # Weighted Monte Carlo estimate of the optimal control sequence.
    U_hat = (U_samples * w[:, None]).sum(axis=0) / w_sum
    u0_hat = U_hat[:variational.mpc.m]

    return u0_hat, U_hat, float(w_sum), int(true_accept_num)

