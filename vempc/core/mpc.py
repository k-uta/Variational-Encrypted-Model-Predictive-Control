"""
MPC problem formulation with condensed representation.

This module handles the standard LQ-MPC formulation and converts it to a compact form suitable for variational methods.
"""

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
        
        # Lambda: stacked powers of A
        # Lambda = [A; A^2; A^3; ...; A^N]
        Lambda = np.zeros((N * n, n))
        A_power = A.copy()
        for i in range(N):
            Lambda[i*n:(i+1)*n, :] = A_power
            A_power = A_power @ A
        
        Psi = np.zeros((N * n, N * m))
        for i in range(N):
            for j in range(i + 1):
                power = i - j
                if power == 0:
                    A_power = np.eye(n)
                else:
                    A_power = np.linalg.matrix_power(A, power)
                Psi[i*n:(i+1)*n, j*m:(j+1)*m] = A_power @ B
        
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
        Q_bar = np.zeros((N * n, N * n))
        for i in range(N - 1):
            Q_bar[i*n:(i+1)*n, i*n:(i+1)*n] = self.Q
        Q_bar[(N-1)*n:N*n, (N-1)*n:N*n] = self.Qf
        
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