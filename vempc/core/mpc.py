"""
MPC problem formulation with condensed representation.

This module handles the standard LQ-MPC formulation and converts it to a compact form suitable for variational methods.
"""

import time
import numpy as np


class MPCProblem:
    """
    Linear-Quadratic MPC problem in condensed form.
    
    Formulates the MPC problem:
        min_{U} x_0^T P x_0 + x_0^T S U + (1/2) U^T H U
        subject to G U <= h(x_0)
    
    where U = [u_0^T, u_1^T, ..., u_{N-1}^T]^T is the stacked control sequence.
    
    Parameters
    ----------
    A : ndarray, shape (n, n)
        State transition matrix
    B : ndarray, shape (n, m)
        Control input matrix
    Q : ndarray, shape (n, n)
        State cost matrix (positive semidefinite)
    R : ndarray, shape (m, m)
        Control cost matrix (positive definite)
    Qf : ndarray, shape (n, n)
        Terminal state cost matrix (positive semidefinite)
    N : int
        Prediction horizon
    """
    
    def __init__(self, A, B, Q, R, Qf, N):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.N = N
        
        self.n = A.shape[0]  # state dimension
        self.m = B.shape[1]  # control dimension
        self.Nm = N * self.m  # stacked control dimension
        
        # Compute condensed MPC matrices
        self.Lambda, self.Psi = self._compute_prediction_matrices()
        self.P, self.S, self.H = self._compute_cost_matrices()
        
    def _compute_prediction_matrices(self):
        """
        Compute prediction matrices Lambda and Psi such that:
            X = Lambda * x_0 + Psi * U
        
        Returns
        -------
        Lambda : ndarray, shape (N*n, n)
            State transition matrix for initial state
        Psi : ndarray, shape (N*n, N*m)
            Control influence matrix
        """
        n, m, N = self.n, self.m, self.N
        A, B = self.A, self.B

        # Precompute A^k to avoid repeated matrix_power calls
        A_powers = [np.eye(n)]
        for _ in range(N):
            A_powers.append(A_powers[-1] @ A)

        # Lambda = [A; A^2; ...; A^N]
        Lambda = np.vstack(A_powers[1:])

        Psi = np.zeros((N * n, N * m))
        for i in range(N):
            row = slice(i * n, (i + 1) * n)
            for j in range(i + 1):
                col = slice(j * m, (j + 1) * m)
                Psi[row, col] = A_powers[i - j] @ B

        return Lambda, Psi
    
    def _compute_cost_matrices(self):
        """
        Compute condensed cost matrices P, S, H such that:
            J_0(x_0, U) = x_0^T P x_0 + x_0^T S U + (1/2) U^T H U
        
        Returns
        -------
        P : ndarray, shape (n, n)
            Initial state cost matrix
        S : ndarray, shape (n, N*m)
            Cross term matrix
        H : ndarray, shape (N*m, N*m)
            Control cost matrix (Hessian)
        """
        n, m, N = self.n, self.m, self.N
        
        # Construct block diagonal Q_bar and R_bar
        # Q_bar = diag(Q, Q, ..., Q, Qf)
        Q_bar = np.kron(np.eye(N), self.Q)
        Q_bar[(N - 1) * n:N * n, (N - 1) * n:N * n] = self.Qf

        # R_bar = diag(R, R, ..., R)
        R_bar = np.kron(np.eye(N), self.R)
        
        # Compute condensed matrices
        # P = Q + Lambda^T Q_bar Lambda
        P = self.Q + self.Lambda.T @ Q_bar @ self.Lambda
        
        # S = 2 Lambda^T Q_bar Psi
        S = 2 * self.Lambda.T @ Q_bar @ self.Psi
        
        # H = 2(R_bar + Psi^T Q_bar Psi)
        H = 2 * (R_bar + self.Psi.T @ Q_bar @ self.Psi)
        
        return P, S, H
    
    def quadratic_cost(self, x0, U):
        """
        Evaluate quadratic cost J_0(x_0, U).
        
        Parameters
        ----------
        x0 : ndarray, shape (n,)
            Initial state
        U : ndarray, shape (N*m,)
            Stacked control sequence
        
        Returns
        -------
        cost : float
            Quadratic cost value
        """
        return (x0.T @ self.P @ x0 + x0.T @ self.S @ U + 0.5 * U.T @ self.H @ U)
    
    def get_control_sequence(self, U):
        """
        Reshape stacked control U into sequence form.
        
        Parameters
        ----------
        U : ndarray, shape (N*m,)
            Stacked control
        
        Returns
        -------
        u_seq : ndarray, shape (N, m)
            Control sequence
        """
        return U.reshape((self.N, self.m))
    
    def get_state_trajectory(self, x0, U):
        """
        Compute predicted state trajectory.
        
        Parameters
        ----------
        x0 : ndarray, shape (n,)
            Initial state
        U : ndarray, shape (N*m,)
            Stacked control
        
        Returns
        -------
        x_traj : ndarray, shape (N+1, n)
            State trajectory [x_0, x_1, ..., x_N]
        """
        X = self.Lambda @ x0 + self.Psi @ U
        X_reshaped = X.reshape((self.N, self.n))
        return np.vstack([x0.reshape(1, -1), X_reshaped])
    
    def build_constraint_matrices(self, Gx, hx, Gu, hu):
        """
        Build condensed inequality constraints:
        G U <= h(x0)
        from per-step constraints:
        Gx xk <= hx for k=1..N
        Gu uk <= hu for k=0..N-1
        """
        n = self.n
        m = self.m
        N = self.N
        Lambda = self.Lambda
        Psi = self.Psi

        px = Gx.shape[0]
        pu = Gu.shape[0]

        # Stack constraints across horizon
        Gx_bar = np.kron(np.eye(N), Gx)       # (N*px) x (N*n)
        hx_bar = np.tile(hx, N)              # (N*px,)

        Gu_bar = np.kron(np.eye(N), Gu)      # (N*pu) x (N*m)
        hu_bar = np.tile(hu, N)              # (N*pu,)

        # Condensed:
        #   Gx_bar (Lambda x0 + Psi U) <= hx_bar
        # => (Gx_bar Psi) U <= hx_bar - (Gx_bar Lambda) x0
        G_top = Gx_bar @ Psi
        L_top = Gx_bar @ Lambda

        # Input: Gu_bar U <= hu_bar
        G_bot = Gu_bar
        h_bot = hu_bar

        G = np.vstack([G_top, G_bot])

        def h_of_x0(x0):
            h_top = hx_bar - L_top @ x0
            return np.hstack([h_top, h_bot])

        return G, h_of_x0


def simulate(controller_name, x0, controller_fn, *, A, B, T_steps):
    n = A.shape[0]
    m = B.shape[1]
    xs = np.empty((T_steps + 1, n), dtype=float)
    us = np.empty((T_steps, m), dtype=float)
    info_log = []
    U_warm = None

    start = time.perf_counter()
    x = x0.copy()
    xs[0] = x
    for k in range(T_steps):
        out = controller_fn(x, U_warm)
        if isinstance(out, tuple) and len(out) == 3:
            u, Useq, info = out
        else:
            u, Useq = out
            info = {}

        # store warm start if Useq exists
        U_warm = Useq.copy() if Useq is not None else None

        # apply input
        u_arr = np.asarray(u, dtype=float).reshape(-1)
        us[k] = u_arr
        info_log.append(info)

        # propagate
        x = A @ x + B @ u_arr
        xs[k + 1] = x

    elapsed = time.perf_counter() - start
    return xs, us, info_log, elapsed


def trajectory_cost(xs, us, *, Q, R, Qf):
    steps = min(len(us), len(xs) - 1)
    if steps <= 0:
        xN = xs[0]
        return float(xN.T @ Qf @ xN)

    xk = xs[:steps]
    uk = us[:steps]
    state_cost = np.einsum("ij,jk,ik->", xk, Q, xk)
    input_cost = np.einsum("ij,jk,ik->", uk, R, uk)
    xN = xs[steps]
    cost = state_cost + input_cost + xN.T @ Qf @ xN
    return float(cost)


def max_constraint_violation(xs, us, *, Gx, hx, Gu, hu):
    vx = 0.0
    if xs.shape[0] > 1:
        res_x = xs[1:] @ Gx.T - hx
        vx = float(np.max(res_x))

    vu = 0.0
    if len(us) > 0:
        res_u = us @ Gu.T - hu
        vu = float(np.max(res_u))

    return max(0.0, vx), max(0.0, vu)


def avg_acceptance(info_log):
    vals = np.array([d.get("accept_rate", np.nan) for d in info_log], dtype=float)
    if np.all(np.isnan(vals)):
        return float("nan")
    return float(np.nanmean(vals))
