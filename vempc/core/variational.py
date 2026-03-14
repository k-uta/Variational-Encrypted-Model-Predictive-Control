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
        
        self.L_0 = np.linalg.cholesky(self.Sigma_0)

        # Tilted distribution parameters
        self.Sigma_U = self._compute_Sigma_U()
        self.L_U = np.linalg.cholesky(self.Sigma_U)

    
    

    # Tilted distribution parameters
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


    # sampling with kappa_0
    def sample_kappa_0(self, K, random_state=None):
        """
        Sample from prior kappa_0 = N(0, Sigma_0).
        
        Parameters
        ----------
        K : int
            Number of samples
        random_state : int or None
            Random seed
        
        Returns
        -------
        U_samples : ndarray, shape (K, N*m)
            Sampled control sequences from kappa_0
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        xi = np.random.randn(K, self.mpc.Nm)
        return xi @ self.L_0.T
    
    # sampling with kappa_tilde
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
    
    def compute_weights(
        self,
        U_samples,
        x0,
        reference='tilde_kappa',
        cheb_coeffs=None,
        cheb_bound=None,
        cheb_clip=True,
        cheb_weight_mode="product",
        cheb_eta=None,
        eps=1e-12,
    ):
        """
        Compute importance weights for sampled trajectories.
        
        Parameters
        ----------
        U_samples : ndarray, shape (K, N*m)
            Sampled control sequences
        x0 : ndarray, shape (n,)
            Initial state
        reference : str, default='tilde_kappa'
            Reference distribution that samples came from:
            - 'tilde_kappa': samples from tilted distribution (cost already in distribution)
            - 'kappa_0': samples from prior (need to apply cost weighting)
        cheb_coeffs : ndarray or None
            Chebyshev coefficients for indicator approximation (optional).
        cheb_bound : float or None
            Bound used to scale residuals into [-1, 1] before evaluation.
        cheb_clip : bool, default=True
            Clip scaled residuals and polynomial outputs to [-1,1] / [0,1].
        cheb_weight_mode : str, default='product'
            'product' for product of per-constraint indicators,
            'exp' for exp(-eta * sum(1 - r_j)).
        cheb_eta : float or None
            Scale for exp weighting when cheb_weight_mode='exp'.
        eps : float, default=1e-12
            Numerical epsilon for log/normalization safety.
        
        Returns
        -------
        weights : ndarray, shape (K,)
            Normalized importance weights. Returns None if no feasible samples.
        """
        K = U_samples.shape[0]

        def _cheb_indicator(residuals):
            t = residuals / cheb_bound
            if cheb_clip:
                t = np.clip(t, -1.0, 1.0)
            y = np.polynomial.chebyshev.chebval(t, cheb_coeffs)
            if cheb_clip:
                y = np.clip(y, 0.0, 1.0)
            return y

        if reference == 'kappa_0':
            # w_i = exp(-J_0(U_i, x0)/lambda) * r(U_i; x0)

            
            costs = np.array([self.mpc.quadratic_cost(x0, U_samples[i]) for i in range(K)])
            
            if self.penalty.has_constraints:
                if cheb_coeffs is not None and cheb_bound is not None:
                    residuals = self.penalty.constraint_residual(U_samples, x0)
                    r = _cheb_indicator(residuals)
                    if cheb_weight_mode == "product":
                        r = np.prod(r, axis=1)
                    elif cheb_weight_mode == "exp":
                        if cheb_eta is None:
                            cheb_eta = 1.0
                        r = np.exp(-cheb_eta * np.sum(1.0 - r, axis=1))
                    else:
                        raise ValueError("cheb_weight_mode must be 'product' or 'exp'")

                    log_r = np.log(np.maximum(r, eps))
                    log_weights = -costs / self.lambda_param + log_r
                else:
                    desirability = self.penalty.desirability_weight(U_samples, x0)
                    feasible_mask = (desirability > 0)

                    # Initialize log weights to -inf (infeasible)
                    log_weights = np.full(K, -np.inf)

                    # Only feasible samples get finite weights
                    if np.any(feasible_mask):
                        log_weights[feasible_mask] = -costs[feasible_mask] / self.lambda_param
                    else:
                        # No feasible samples
                        return None
            else:
                log_weights = -costs / self.lambda_param
        
        elif reference == 'tilde_kappa':
            # Samples from tilde_kappa: cost factored into sampling already
            # Only apply feasibility weighting: w_i = r(U_i; x0)
            
            if self.penalty.has_constraints:
                if cheb_coeffs is not None and cheb_bound is not None:
                    residuals = self.penalty.constraint_residual(U_samples, x0)
                    r = _cheb_indicator(residuals)
                    if cheb_weight_mode == "product":
                        r = np.prod(r, axis=1)
                    elif cheb_weight_mode == "exp":
                        if cheb_eta is None:
                            cheb_eta = 1.0
                        r = np.exp(-cheb_eta * np.sum(1.0 - r, axis=1))
                    else:
                        raise ValueError("cheb_weight_mode must be 'product' or 'exp'")

                    log_weights = np.log(np.maximum(r, eps))
                else:
                    desirability = self.penalty.desirability_weight(U_samples, x0)
                    feasible_mask = (desirability > 0)

                    log_weights = np.full(K, -np.inf)

                    if np.any(feasible_mask):
                        log_weights[feasible_mask] = 0.0  # Uniform weights among feasible samples
                    else:
                        # No feasible samples
                        return None

            else:
                # Unconstrained: uniform weights
                log_weights = np.zeros(K)
        
        else:
            raise ValueError(f"Unknown reference distribution: {reference}. "
                           f"Use 'kappa_0' or 'tilde_kappa'.")
        
        # Convert to linear weights with numerical stability
        finite_mask = np.isfinite(log_weights)
        
        if not np.any(finite_mask):
            # Should not reach here due to early returns above, but safety check
            return None
        
        log_max = np.max(log_weights[finite_mask])
        weights = np.zeros(K)
        weights[finite_mask] = np.exp(log_weights[finite_mask] - log_max)
        
        # Normalize
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            return None
        
        return weights
    
    def U_hat_K(self, x0, K, random_state=None, reference='tilde_kappa'):
        """
        Monte Carlo estimator from equation (U_hat_mc_tilted):
        
            U_hat_K(x_0) = sum_i U^(i) w_i / sum_i w_i
        
        Parameters
        ----------
        x0 : ndarray, shape (n,)
            Initial state
        K : int
            Number of samples
        random_state : int or None
            Random seed
        reference : str, default='tilde_kappa'
            Reference distribution to sample from:
            - 'tilde_kappa': sample from tilted N(m_U(x_0), Sigma_U) [recommended]
            - 'kappa_0': sample from prior N(0, Sigma_0) [for comparison]
        
        Returns
        -------
        U_hat : ndarray, shape (N*m,)
            Estimated optimal control
        """
        # Sample from specified reference distribution
        if reference == 'tilde_kappa':
            U_samples = self.sample_kappa_tilde(x0, K, random_state)
        elif reference == 'kappa_0':
            U_samples = self.sample_kappa_0(K, random_state)
        else:
            raise ValueError(f"Unknown reference distribution: {reference}. "
                        f"Use 'kappa_0' or 'tilde_kappa'.")
        
        # Compute weights
        weights = self.compute_weights(U_samples, x0, reference=reference)
        
        # Check for failure (no feasible samples)
        if weights is None:
            raise RuntimeError(
                f"ERROR: No feasible samples from {reference} with K={K}. "
                f"Cannot compute U_hat."
            )
        
        # Weighted average
        U_hat = np.sum(weights[:, np.newaxis] * U_samples, axis=0)
        
        return U_hat
