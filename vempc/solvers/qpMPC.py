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
        # Warm-start the QP with the previous solution to speed convergence.
        U_var.value = warm_start_U

    prob.solve(solver=solver, warm_start=True, verbose=verbose)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"QP solve failed with status={prob.status}")

    Ustar = np.array(U_var.value).reshape(-1)
    u0 = Ustar[:m_in]
    return u0, Ustar

def chebyshev_relu_coeffs(order, bound, n_samples=None):
    """
    Chebyshev approximation of ReLU(z) = max(0, z) on [-bound, bound].

    Returns coefficients for Chebyshev basis on t = z / bound in [-1, 1].
    """
    if bound <= 0:
        raise ValueError("bound must be positive")
    if n_samples is None:
        n_samples = max(200, 10 * order)

    k = np.arange(n_samples)
    # Chebyshev nodes on [-1, 1] for stable polynomial fitting.
    t_nodes = np.cos(np.pi * (2 * k + 1) / (2 * n_samples))
    y = bound * np.maximum(0.0, t_nodes)
    coeffs = np.polynomial.chebyshev.chebfit(t_nodes, y, deg=order)
    return coeffs


def eval_relu_poly(z, coeffs, bound):
    """
    Evaluate Chebyshev ReLU approximation at z.
    """
    # Map to the normalized domain used for the Chebyshev basis.
    t = np.asarray(z, dtype=float) / bound
    y = np.polynomial.chebyshev.chebval(t, coeffs)
    return y


# Unconstrained QP with polynomial surrogate penalty (ReLU approximation)
def make_surrogate_qp_solver(H, G, *, cheb_order=10, cheb_bound=1.0, eta=1.0, n_samples=None):
    coeffs = chebyshev_relu_coeffs(cheb_order, cheb_bound, n_samples=n_samples)
    return {
        "H": H,
        "G": G,
        "coeffs": coeffs,
        "bound": cheb_bound,
        "eta": eta,
    }


def solve_surrogate_mpc(
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
        if G is None:
            return cost
        g = G @ U - h
        s_l = eval_relu_poly(g, coeffs, bound)
        # Soft-penalty: larger violations increase the objective via s_l.
        penalty = eta * np.sum(s_l)
        return cost + penalty

    if warm_start_U is None:
        # Unconstrained minimizer gives a stable starting point for L-BFGS-B.
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
        Ustar = np.asarray(x_init).reshape(-1)
    else:
        Ustar = np.asarray(res.x).reshape(-1)
    u0 = Ustar[:m_in]
    return u0, Ustar
