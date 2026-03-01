"""
Polynomial evaluator for the constraint surrogate h_ell under CKKS.

Slot layout convention (used by both offline.py and online.py):
  - Control samples U^(i) are packed with slot width `dim` (= N*m).
  - Constraint residuals g^(i) are packed with slot width `p` (= num constraints).
  - Sample k occupies slots [k*p, k*p + p) in the residual ciphertext.
  - After sum_within_slots, the aggregate score s^(i) sits at slot k*p.

This layout must be consistent with OfflinePreprocessor._pack_samples
when called for Gamma (which uses width p, not dim).
"""

import numpy as np
from .setup import CryptoSetup


class PolyEvaluator:
    """
    Evaluates the polynomial surrogate h_ell slot-wise and aggregates
    per-sample scores via intra-slot summation.

    Parameters
    ----------
    crypto : CryptoSetup
        Initialized crypto context.
    coeffs : list of float
        Monomial coefficients in ascending order:
        p(x) = coeffs[0] + coeffs[1]*x + ... + coeffs[d]*x^d
    p : int
        Number of constraints (slot width for residual ciphertext).
    """

    def __init__(self, crypto: CryptoSetup, coeffs: list, p: int):
        self.crypto = crypto
        self.coeffs = list(coeffs)
        self.p      = p          # constraint count = slot width for residuals

    def register_rotation_keys(self):
        """
        Register rotation keys needed for intra-slot summation.
        Call this after OfflinePreprocessor.generate_cache() to avoid
        duplicate key registration errors.
        """
        self.crypto.generate_rotation_keys(list(range(1, self.p)))

    # ------------------------------------------------------------------
    # Horner evaluation (slot-wise)
    # ------------------------------------------------------------------

    def horner_eval(self, ct):
        """
        Evaluate polynomial slot-wise on a ciphertext using Horner's method.

        For p(x) = c0 + c1*x + ... + cd*x^d, Horner's scheme is:
            p(x) = c0 + x*(c1 + x*(c2 + ... + x*cd)...)

        Parameters
        ----------
        ct : OpenFHE Ciphertext
            Ciphertext containing residual values g^(i)_j in packed slots.

        Returns
        -------
        OpenFHE Ciphertext
            Slot-wise polynomial evaluation h_ell(g^(i)_j).
        """
        cc         = self.crypto.cc
        batch_size = self.crypto.batch_size
        coeffs     = self.coeffs

        # Initialize accumulator with leading coefficient
        ct_accum = cc.EvalAdd(
            cc.EvalMult(ct, cc.MakeCKKSPackedPlaintext([0.0] * batch_size)),
            cc.MakeCKKSPackedPlaintext([coeffs[-1]] * batch_size)
        )

        # Horner steps from second-to-last down to constant term
        for c_i in reversed(coeffs[:-1]):
            ct_accum = cc.EvalAdd(
                cc.EvalMult(ct_accum, ct),
                cc.MakeCKKSPackedPlaintext([c_i] * batch_size)
            )

        return ct_accum

    # ------------------------------------------------------------------
    # Intra-slot summation (aggregate per-sample score)
    # ------------------------------------------------------------------

    def sum_within_slots(self, ct):
        """
        Sum p consecutive slots within each sample block.

        For sample k occupying slots [k*p, k*p+p), this computes
            s^(i) = sum_{j=0}^{p-1} h_ell(g^(i)_j)
        and places the result at slot k*p (first slot of each block).

        Uses rotation-and-add over r = 1, ..., p-1, then masks to
        retain only the first slot of each block.

        Parameters
        ----------
        ct : OpenFHE Ciphertext
            Slot-wise polynomial evaluations h_ell(g^(i)_j).

        Returns
        -------
        OpenFHE Ciphertext
            Aggregate scores s^(i) at slots k*p, others zeroed.
        """
        cc         = self.crypto.cc
        p          = self.p
        batch_size = self.crypto.batch_size

        # Sum rotations r = 1, ..., p-1 within each block
        ct_sum = ct
        for r in range(1, p):
            ct_sum = cc.EvalAdd(ct_sum, cc.EvalRotate(ct, r))

        # Mask: retain only slot 0 of each p-wide block
        mask = [0.0] * batch_size
        for k in range(batch_size // p):
            mask[k * p] = 1.0

        return cc.EvalMult(ct_sum, cc.MakeCKKSPackedPlaintext(mask))

    # ------------------------------------------------------------------
    # Combined evaluation
    # ------------------------------------------------------------------

    def evaluate_poly(self, ct):
        """
        Full pipeline: slot-wise Horner evaluation followed by
        intra-slot summation.

        Parameters
        ----------
        ct : OpenFHE Ciphertext
            Packed residual ciphertext g^(i)_j.

        Returns
        -------
        OpenFHE Ciphertext
            Aggregate surrogate scores s^(i) at slots k*p.
        """
        ct_h = self.horner_eval(ct)
        return self.sum_within_slots(ct_h)