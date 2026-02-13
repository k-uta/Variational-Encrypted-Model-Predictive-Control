"""
Variational reformulation for MPC.

Implements the variational principle with tilted Gaussian sampling.
"""

import numpy as np
from .mpc import MPCProblem
from .constraints import ConstraintPenalty


class VariationalMPC:
    """
    Variational MPC with importance sampling.
    
    Uses tilted distribution kappa_tilde = N(m_U(x_0), Sigma_U) for sampling
    and computes the estimator from equation (U_hat_mc_tilted).
    
    Parameters
    ----------
    mpc_problem : MPCProblem
        MPC problem instance
    penalty : ConstraintPenalty or None
        Constraint penalty (desirability function)
    lambda_param : float, default=1.0
        Temperature parameter lambda
    Sigma_0 : ndarray or None
        Prior covariance for kappa_0 = N(0, Sigma_0)
    """
    
    def __init__(self, mpc_problem, penalty=None, lambda_param=1.0, Sigma_0=None):
        self.mpc = mpc_problem
        self.penalty = penalty if penalty is not None else ConstraintPenalty(None, None)
        self.lambda_param = lambda_param
        
        # Prior kappa_0 = N(0, Sigma_0)
        if Sigma_0 is not None:
            self.Sigma_0 = Sigma_0
        else:
            self.Sigma_0 = np.eye(mpc_problem.Nm)
        
        # Tilted distribution parameters
        self.Sigma_U = self._compute_Sigma_U()
        self.L_U = np.linalg.cholesky(self.Sigma_U)
    
    def _compute_Sigma_U(self):
        """
        Sigma_U = (Sigma_0^{-1} + (1/lambda)H)^{-1}
        """
        Sigma_0_inv = np.linalg.inv(self.Sigma_0)
        H_scaled = self.mpc.H / self.lambda_param
        return np.linalg.inv(Sigma_0_inv + H_scaled)
    
    def m_U(self, x0):
        """
        m_U(x_0) = -(1/lambda) Sigma_U S^T x_0
        """
        return -(1.0 / self.lambda_param) * self.Sigma_U @ self.mpc.S.T @ x0
    
    def sample_kappa_tilde(self, x0, K, random_state=None):
        """
        Sample from kappa_tilde(\cdot | x_0) = N(m_U(x_0), Sigma_U).
        
        Parameters
        ----------
        x0 : ndarray, shape (n,)
            Initial state
        K : int
            Number of samples
        random_state : int or None
            Random seed
        
        Returns
        -------
        U_samples : ndarray, shape (K, N*m)
            Sampled control sequences
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        mean = self.m_U(x0)
        xi = np.random.randn(K, self.mpc.Nm)
        return mean + xi @ self.L_U.T
    
    def U_hat_K(self, x0, K, random_state=None):
        """
        Monte Carlo estimator from equation (U_hat_mc_tilted):
        
            U_hat_K(x_0) = sum_i U^(i) r(U^(i); x_0) / sum_i r(U^(i); x_0)
        
        Parameters
        ----------
        x0 : ndarray, shape (n,)
            Initial state
        K : int
            Number of samples
        random_state : int or None
            Random seed
        
        Returns
        -------
        U_hat : ndarray, shape (N*m,)
            Estimated optimal control
        """
        # Sample from kappa_tilde
        U_samples = self.sample_kappa_tilde(x0, K, random_state)
        
        # Compute desirability weights r(U^(i); x_0)
        r_weights = self.penalty.desirability_weight(U_samples, x0)
        
        # Equation (U_hat_mc_tilted)
        numerator = np.sum(r_weights[:, np.newaxis] * U_samples, axis=0)
        denominator = np.sum(r_weights)
        
        if denominator > 0:
            return numerator / denominator
        else:
            # No feasible samples: return mean of tilted distribution
            return self.m_U(x0)