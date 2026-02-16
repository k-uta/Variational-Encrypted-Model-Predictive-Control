import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

# Constrained QP
def make_standard_qp_solver(H, G):
    U = cp.Variable(H.shape[0])
    q = cp.Parameter(H.shape[0])          # q = S' x0
    hpar = cp.Parameter(G.shape[0])       # h(x0)

    obj = 0.5 * cp.quad_form(U, H) + q @ U # constant term 무시
    cons = [G @ U <= hpar]
    prob = cp.Problem(cp.Minimize(obj), cons)
    return prob, U, q, hpar

def solve_standard_mpc(
    x0,
    *,
    S,
    h_of_x0,
    m_in,
    prob,
    U_var,
    q_par,
    h_par,
    warm_start_U=None,
    verbose=False,
    solver=cp.OSQP,
):
    q_par.value = np.asarray(S.T @ x0).reshape(-1)
    h_par.value = np.asarray(h_of_x0(x0)).reshape(-1)

    if warm_start_U is not None:
        U_var.value = warm_start_U

    prob.solve(solver=solver, warm_start=True, verbose=verbose)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"QP solve failed with status={prob.status}")

    Ustar = np.array(U_var.value).reshape(-1)
    u0 = Ustar[:m_in]
    return u0, Ustar

def chebyshev_indicator_coeffs(order, bound, n_samples=None):
    """
    Chebyshev approximation of the indicator 1_{z <= 0} on [-bound, bound].

    Returns coefficients for Chebyshev basis on t = z / bound in [-1, 1].
    """
    if bound <= 0:
        raise ValueError("bound must be positive")
    if n_samples is None:
        n_samples = max(200, 10 * order)

    k = np.arange(n_samples)
    t_nodes = np.cos(np.pi * (2 * k + 1) / (2 * n_samples))  # Chebyshev nodes in [-1, 1]
    z_nodes = bound * t_nodes
    y = (z_nodes <= 0).astype(float)

    coeffs = np.polynomial.chebyshev.chebfit(t_nodes, y, deg=order)
    return coeffs


def eval_indicator_poly(z, coeffs, bound, clip=True):
    """
    Evaluate Chebyshev indicator approximation at z.
    """
    t = np.asarray(z, dtype=float) / bound
    if clip:
        t = np.clip(t, -1.0, 1.0)
    y = np.polynomial.chebyshev.chebval(t, coeffs)
    if clip:
        y = np.clip(y, 0.0, 1.0)
    return y


# Unconstrained QP with augmented cost function (polynomial indicator)
def make_indicator_qp_solver(H, G, *, cheb_order=10, cheb_bound=1.0, eta=1.0, n_samples=None):
    coeffs = chebyshev_indicator_coeffs(cheb_order, cheb_bound, n_samples=n_samples)
    return {
        "H": H,
        "G": G,
        "coeffs": coeffs,
        "bound": cheb_bound,
        "eta": eta,
    }


def solve_indicator_mpc(
    x0,
    *,
    S,
    h_of_x0,
    m_in,
    solver,
    warm_start_U=None,
    verbose=False,
    maxiter=1000,
    ftol=1e-9,
    gtol=1e-6,
):
    H = solver["H"]
    G = solver["G"]
    coeffs = solver["coeffs"]
    bound = solver["bound"]
    eta = solver["eta"]

    q = np.asarray(S.T @ x0).reshape(-1)
    h = np.asarray(h_of_x0(x0)).reshape(-1)

    def objective(U):
        U = np.asarray(U)
        cost = 0.5 * U @ H @ U + q @ U
        g = G @ U - h
        r = eval_indicator_poly(g, coeffs, bound, clip=True)
        penalty = eta * np.sum(1.0 - r)
        return cost + penalty

    if warm_start_U is None:
        # Unconstrained quadratic minimizer as a stable starting point
        try:
            x_init = -np.linalg.solve(H, q)
        except np.linalg.LinAlgError:
            x_init = np.zeros(H.shape[0])
    else:
        x_init = np.asarray(warm_start_U).reshape(-1)

    res = minimize(
        objective,
        x_init,
        method="L-BFGS-B",
        options={
            "maxiter": maxiter,
            "ftol": ftol,
            "gtol": gtol,
            "disp": verbose,
        },
    )
    if not res.success:
        # Fall back to the best available starting point
        Ustar = np.asarray(x_init).reshape(-1)
    else:
        Ustar = np.asarray(res.x).reshape(-1)
    u0 = Ustar[:m_in]
    return u0, Ustar
