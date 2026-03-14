"""
Plaintext (mock) backend for VEMPC.

Implements BackendBase using numpy only -- no encryption.
Used for developing and validating the full protocol before integrating real homomorphic encryption.
"""

import numpy as np
from .base import BackendBase

class PlaintextBackend(BackendBase):
    """
    Plaintext backend: all operations are standard numpy.

    'Ciphertexts' are just numpy arrays. This backend is a place-holder for OpenFHEBackend during testing.
    """

    # ------------------------------------------------------------------
    # Encryption / decryption (identity maps, no real enc/dec)
    # ------------------------------------------------------------------

    def encrypt(self, vec: np.ndarray) -> np.ndarray:
        """
        'Encrypt': return a copy of the vector as-is.

        Parameters
        ----------
        vec : ndarray, shape (d,)

        Returns
        -------
        ndarray, shape (d,)
        """
        return np.array(vec, dtype=float)

    def decrypt(self, ct: np.ndarray, length: int) -> list:
        """
        'Decrypt': return the first `length` elements.

        Parameters
        ----------
        ct : ndarray
        length : int

        Returns
        -------
        list of float, length == length
        """
        return list(np.asarray(ct, dtype=float)[:length])

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def add(self, ct1: np.ndarray, ct2: np.ndarray) -> np.ndarray:
        """
        Element-wise addition.

        Parameters
        ----------
        ct1, ct2 : ndarray

        Returns
        -------
        ndarray
        """
        return np.asarray(ct1, dtype=float) + np.asarray(ct2, dtype=float)

    def eval_poly(self, ct: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """
        Evaluate polynomial slot-wise using Horner's method.

        p(x) = coeffs[0] + coeffs[1]*x + ... + coeffs[d]*x^d

        Parameters
        ----------
        ct : ndarray, shape (d,)
        coeffs : ndarray, shape (deg+1,)
            Index i corresponds to degree i (ascending order).

        Returns
        -------
        ndarray, shape (d,)
        """
        x = np.asarray(ct, dtype=float)
        # numpy.polyval expects descending order, so reverse coeffs
        result = np.polyval(coeffs[::-1], x)
        return result