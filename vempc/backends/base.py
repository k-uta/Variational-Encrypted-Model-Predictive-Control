"""
Abstract backend interface for VEMPC.

All backends (plaintext, OpenFHE, etc.) can implement this interface.
This allows the rest of the codebase to remain agnostic to whether computation is performed in plaintext or under encryption.
"""

from abc import ABC, abstractmethod
import numpy as np

class BackendBase(ABC):
    """
    Abstract base class for VEMPC compute backends.

    A backend is responsible for:
      - Encrypting / encoding vectors
      - Decrypting / decoding ciphertexts
      - Homomorphic (or plaintext) addition
      - Polynomial evaluation on (encrypted) data
    """

    # ------------------------------------------------------------------
    # Encryption / decryption
    # ------------------------------------------------------------------

    @abstractmethod
    def encrypt(self, vec: np.ndarray):
        """
        Encrypt (or encode) a real vector.

        Parameters
        ----------
        vec : ndarray, shape (d,)
            Plaintext vector to encrypt.

        Returns
        -------
        ct : backend-specific ciphertext or array
        """

    @abstractmethod
    def decrypt(self, ct, length: int) -> list:
        """
        Decrypt (or decode) a ciphertext to a real vector.

        Parameters
        ----------
        ct : backend-specific ciphertext or array
        length : int
            Number of slots to return.

        Returns
        -------
        values : list of float, length == length
        """

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    @abstractmethod
    def add(self, ct1, ct2):
        """
        Add two ciphertexts (or arrays).

        Parameters
        ----------
        ct1, ct2 : backend-specific ciphertexts or arrays

        Returns
        -------
        ct_sum : same type as inputs
        """

    @abstractmethod
    def eval_poly(self, ct, coeffs: np.ndarray):
        """
        Evaluate a polynomial on a ciphertext (or array) slot-wise.

        The polynomial is defined in the monomial basis:
            p(x) = coeffs[0] + coeffs[1]*x + ... + coeffs[d]*x^d

        Parameters
        ----------
        ct : backend-specific ciphertext or array
        coeffs : ndarray, shape (d+1,)
            Monomial coefficients, index i corresponds to degree i.

        Returns
        -------
        ct_result : same type as ct
        """