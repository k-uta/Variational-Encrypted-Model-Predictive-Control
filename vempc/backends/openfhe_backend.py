"""
OpenFHE backend for VEMPC.

Implements BackendBase using the OpenFHE Python bindings (CKKS scheme).
Wraps CryptoSetup and PolyEvaluator from vempc/crypto/.
"""

import numpy as np
from .base import BackendBase

try:
    from ..crypto.setup import CryptoSetup
    from ..crypto.poly_eval import PolyEvaluator
    import openfhe
    _OPENFHE_AVAILABLE = True
except ImportError:
    _OPENFHE_AVAILABLE = False


class OpenFHEBackend(BackendBase):
    """
    OpenFHE (CKKS) backend.

    Parameters
    ----------
    dim : int
        Dimension of control vector (N*m).
    k : int
        Number of samples K per online cycle.
    coeffs : array-like, shape (deg+1,)
        Monomial coefficients for polynomial surrogate (ascending order).
    ring_dim : int
        CKKS ring dimension (power of 2).
    mult_depth : int
        Multiplicative depth budget.
    scaling_mod : int
        Scaling modulus size in bits.
    """

    def __init__(
        self,
        dim: int,
        k: int,
        coeffs,
        ring_dim: int = 1 << 12,
        mult_depth: int = 7,
        scaling_mod: int = 50,
    ):
        if not _OPENFHE_AVAILABLE:
            raise RuntimeError(
                "OpenFHE or vempc.crypto is not available. "
                "Install OpenFHE Python bindings before using this backend."
            )
        self._coeffs = np.asarray(coeffs, dtype=float)

        # Initialize crypto context
        self.crypto = CryptoSetup(
            dim=dim,
            k=k,
            ring_dim=ring_dim,
            mult_depth=mult_depth,
            scaling_mod=scaling_mod,
        )
        self.crypto.setup()

        # Polynomial evaluator (uses ascending-order monomial coeffs)
        self.poly_eval = PolyEvaluator(self.crypto, list(self._coeffs))

    # ------------------------------------------------------------------
    # Encryption / decryption
    # ------------------------------------------------------------------

    def encrypt(self, vec: np.ndarray):
        """
        Encrypt a real vector under CKKS.

        Parameters
        ----------
        vec : ndarray, shape (d,)

        Returns
        -------
        ct : OpenFHE Ciphertext
        """
        cc = self.crypto.cc
        ptxt = cc.MakeCKKSPackedPlaintext(list(np.asarray(vec, dtype=float)))
        return cc.Encrypt(self.crypto.keys.publicKey, ptxt)

    def decrypt(self, ct, length: int) -> list:
        """
        Decrypt a CKKS ciphertext and return the first `length` slots.

        Parameters
        ----------
        ct : OpenFHE Ciphertext
        length : int

        Returns
        -------
        list of float, length == length
        """
        cc = self.crypto.cc
        ptxt = cc.Decrypt(ct, self.crypto.keys.secretKey)
        ptxt.SetLength(length)
        return [x.real for x in ptxt.GetCKKSPackedValue()]

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def add(self, ct1, ct2):
        """
        Homomorphic addition of two ciphertexts.

        Parameters
        ----------
        ct1, ct2 : OpenFHE Ciphertext

        Returns
        -------
        OpenFHE Ciphertext
        """
        return self.crypto.cc.EvalAdd(ct1, ct2)

    def eval_poly(self, ct, coeffs: np.ndarray = None):
        """
        Evaluate polynomial on a ciphertext using PolyEvaluator.

        Uses coefficients provided at construction unless overridden.

        Parameters
        ----------
        ct : OpenFHE Ciphertext
        coeffs : ndarray or None
            If provided, overrides the coefficients set at construction.

        Returns
        -------
        OpenFHE Ciphertext
        """
        if coeffs is not None:
            # Temporary evaluator with overridden coefficients
            tmp = PolyEvaluator(self.crypto, list(np.asarray(coeffs, dtype=float)))
            return tmp.evaluate_poly(ct)
        return self.poly_eval.evaluate_poly(ct)

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        return _OPENFHE_AVAILABLE