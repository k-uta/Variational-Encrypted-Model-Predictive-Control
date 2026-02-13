"""
Constraint handling for variational MPC.

This module provides desirability weights for constraint satisfaction.
Initially uses exact indicator functions (hard constraints), with
polynomial approximation support to be added later for encryption.
"""

import numpy as np


class ConstraintPenalty:
    """
    Constraint penalty for feasibility handling.
    
    Computes desirability weights r(U; x_0) for control sequences based
    on constraint satisfaction. Initially implements exact indicator:
        r(U; x_0) = 1 if U is feasible, 0 otherwise
    
    Parameters
    ----------
    G : ndarray, shape (p, N*m) or None
        Constraint matrix (stacked)
    h_func : callable or None
        Function h(x_0) that returns constraint bound vector of shape (p,)
    mode : str, default='indicator'
        Constraint handling mode:
        - 'indicator': exact feasibility indicator r = 1_{feasible}
        - 'polynomial': polynomial approximation (to be implemented)
    """
    
    def __init__(self, G, h_func, mode='indicator'):
        self.G = G
        self.h_func = h_func
        self.mode = mode
        
        # Determine if constraints are present
        self.has_constraints = (G is not None and h_func is not None)
        
        if self.has_constraints:
            self.p = G.shape[0]  # number of constraints
        else:
            self.p = 0
    
    def constraint_residual(self, U, x0):
        """
        Compute constraint residual g(U; x_0) = G U - h(x_0).
        
        Parameters
        ----------
        U : ndarray, shape (N*m,) or (K, N*m)
            Control sequence(s)
        x0 : ndarray, shape (n,)
            Initial state
        
        Returns
        -------
        residual : ndarray, shape (p,) or (K, p)
            Constraint residual for each constraint
        """
        if not self.has_constraints:
            # No constraints: return empty array
            if U.ndim == 1:
                return np.zeros(0)
            else:
                return np.zeros((U.shape[0], 0))
        
        h_val = self.h_func(x0)
        
        if U.ndim == 1:
            # Single control sequence
            return self.G @ U - h_val
        else:
            # Batch computation: U is (K, N*m)
            return (U @ self.G.T) - h_val
    
    def is_feasible(self, U, x0, tolerance=1e-6):
        """
        Check if control sequence(s) satisfy constraints.
        
        Parameters
        ----------
        U : ndarray, shape (N*m,) or (K, N*m)
            Control sequence(s)
        x0 : ndarray, shape (n,)
            Initial state
        tolerance : float, default=1e-6
            Feasibility tolerance
        
        Returns
        -------
        feasible : bool or ndarray of bool, shape (K,)
            True if all constraints are satisfied
        """
        if not self.has_constraints:
            # No constraints: always feasible
            if U.ndim == 1:
                return True
            else:
                return np.ones(U.shape[0], dtype=bool)
        
        residuals = self.constraint_residual(U, x0)
        
        if U.ndim == 1:
            # Single sequence
            return np.all(residuals <= tolerance)
        else:
            # Batch: check each sample
            return np.all(residuals <= tolerance, axis=1)
    
    def desirability_weight(self, U, x0, tolerance=1e-6):
        """
        Compute desirability weight r(U; x_0).
        
        For 'indicator' mode:
            r(U; x_0) = 1 if feasible, 0 otherwise
        
        Parameters
        ----------
        U : ndarray, shape (N*m,) or (K, N*m)
            Control sequence(s)
        x0 : ndarray, shape (n,)
            Initial state
        tolerance : float, default=1e-6
            Feasibility tolerance
        
        Returns
        -------
        weights : float or ndarray, shape (K,)
            Desirability weights (0 or 1 for indicator mode)
        """
        if self.mode == 'indicator':
            # Exact indicator function
            feasible = self.is_feasible(U, x0, tolerance)
            if U.ndim == 1:
                return 1.0 if feasible else 0.0
            else:
                return feasible.astype(float)
        else:
            raise NotImplementedError(f"Mode '{self.mode}' not implemented yet")

    def feasibility_rate(self, U_samples, x0, tolerance=1e-6):
        """
        Compute fraction of feasible samples.
        
        Parameters
        ----------
        U_samples : ndarray, shape (K, N*m)
            Sampled control sequences
        x0 : ndarray, shape (n,)
            Initial state
        tolerance : float, default=1e-6
            Feasibility tolerance
        
        Returns
        -------
        rate : float
            Fraction of feasible samples in [0, 1]
        """
        if not self.has_constraints:
            return 1.0
        
        feasible = self.is_feasible(U_samples, x0, tolerance)
        return np.mean(feasible)
    
    def max_violation(self, U, x0):
        """
        Compute maximum constraint violation.
        
        Parameters
        ----------
        U : ndarray, shape (N*m,) or (K, N*m)
            Control sequence(s)
        x0 : ndarray, shape (n,)
            Initial state
        
        Returns
        -------
        violation : float or ndarray, shape (K,)
            Maximum constraint violation (0 if feasible)
        """
        if not self.has_constraints:
            if U.ndim == 1:
                return 0.0
            else:
                return np.zeros(U.shape[0])
        
        residuals = self.constraint_residual(U, x0)
        
        if U.ndim == 1:
            return np.maximum(0.0, np.max(residuals))
        else:
            return np.maximum(0.0, np.max(residuals, axis=1))