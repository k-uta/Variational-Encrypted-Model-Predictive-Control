"""
CKKS crypto context setup and key management for VEMPC.

Handles parameter configuration, key generation, and serialization / deserialization of the crypto context.
"""

import os
import numpy as np

from openfhe import (
    CCParamsCKKSRNS,
    GenCryptoContext,
    SecurityLevel,
    PKESchemeFeature,
    SerializeToFile,
    DeserializeCryptoContext,
    DeserializePublicKey,
    DeserializePrivateKey,
    BINARY,
)

class CryptoSetup:
    """
    CKKS cryptographic context and key container.

    Parameters
    ----------
    dim : int
        Dimension of the control vector (N*m).
    k : int
        Number of samples K per online cycle.
    ring_dim : int
        CKKS ring dimension (power of 2). Determines slot count = ring_dim / 2.
    mult_depth : int
        Multiplicative depth budget. Must cover polynomial degree + packing ops.
    scaling_mod : int
        Scaling modulus size in bits (typically 50 or 59).
    """

    def __init__(
        self,
        dim: int,
        k: int,
        ring_dim: int = 1 << 12,
        mult_depth: int = 7,
        scaling_mod: int = 50,
    ):
        self.dim         = dim
        self.k           = k
        self.ring_dim    = ring_dim
        self.mult_depth  = mult_depth
        self.scaling_mod = scaling_mod

        self.cc         = None
        self.keys       = None
        self.batch_size = None  # number of usable slots = ring_dim / 2

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        """
        Initialize the CKKS crypto context and generate keys.
        Must be called before any encrypt / decrypt operations.
        """
        params = CCParamsCKKSRNS()
        params.SetSecurityLevel(SecurityLevel.HEStd_NotSet)
        params.SetRingDim(self.ring_dim)
        params.SetMultiplicativeDepth(self.mult_depth)
        params.SetScalingModSize(self.scaling_mod)

        self.cc = GenCryptoContext(params)
        self.cc.Enable(PKESchemeFeature.PKE)
        self.cc.Enable(PKESchemeFeature.KEYSWITCH)
        self.cc.Enable(PKESchemeFeature.LEVELEDSHE)

        self.keys       = self.cc.KeyGen()
        self.batch_size = self.cc.GetRingDimension() // 2

        # Multiplication key (required for poly eval)
        self.cc.EvalMultKeyGen(self.keys.secretKey)

    def generate_rotation_keys(self, rotation_amounts: list):
        """
        Register rotation keys for the given shift amounts.

        Must be called after setup(). Duplicate amounts are deduplicated.
        Zero shifts are skipped (no-op rotation).

        Parameters
        ----------
        rotation_amounts : list of int
        """
        amounts = list(set(r for r in rotation_amounts if r != 0))
        if amounts:
            self.cc.EvalRotateKeyGen(self.keys.secretKey, amounts)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, cache_dir: str):
        """
        Serialize crypto context and keys to disk.

        Parameters
        ----------
        cache_dir : str
            Directory to write serialized files into.

        Notes
        -----
        secret_key.bin must remain on the client and never be
        transmitted to the cloud.
        """
        os.makedirs(cache_dir, exist_ok=True)
        SerializeToFile(os.path.join(cache_dir, "context.bin"),    self.cc,             BINARY)
        SerializeToFile(os.path.join(cache_dir, "public_key.bin"), self.keys.publicKey, BINARY)
        SerializeToFile(os.path.join(cache_dir, "secret_key.bin"), self.keys.secretKey, BINARY)
        self.cc.SerializeEvalMultKey(        os.path.join(cache_dir, "eval_mult_key.bin"), BINARY)
        self.cc.SerializeEvalAutomorphismKey(os.path.join(cache_dir, "eval_rot_key.bin"),  BINARY)
        print(f"Crypto context and keys saved to '{cache_dir}'.")
        print("NOTE: secret_key.bin should be kept by the client only.")

    def load(self, cache_dir: str):
        """
        Deserialize crypto context and keys from disk.

        Parameters
        ----------
        cache_dir : str
            Directory containing serialized files from save().
        """
        self.cc, ok = DeserializeCryptoContext(os.path.join(cache_dir, "context.bin"), BINARY)
        if not ok:
            raise RuntimeError("Failed to deserialize CryptoContext.")

        self.cc.DeserializeEvalMultKey(        os.path.join(cache_dir, "eval_mult_key.bin"), BINARY)
        self.cc.DeserializeEvalAutomorphismKey(os.path.join(cache_dir, "eval_rot_key.bin"),  BINARY)

        pub_key, ok = DeserializePublicKey( os.path.join(cache_dir, "public_key.bin"), BINARY)
        if not ok:
            raise RuntimeError("Failed to deserialize public key.")

        sec_key, ok = DeserializePrivateKey(os.path.join(cache_dir, "secret_key.bin"), BINARY)
        if not ok:
            raise RuntimeError("Failed to deserialize secret key.")

        self.batch_size = self.cc.GetRingDimension() // 2
        self.keys = type("Keys", (), {"publicKey": pub_key, "secretKey": sec_key})()
        print(f"Crypto context and keys loaded from '{cache_dir}'.")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def slot_budget(self) -> dict:
        """
        Report slot usage for the current configuration.

        Returns
        -------
        dict with keys: batch_size, slots_per_sample, max_samples
        """
        if self.batch_size is None:
            raise RuntimeError("Call setup() or load() before slot_budget().")
        return {
            "batch_size":       self.batch_size,
            "slots_per_sample": self.dim,
            "max_samples":      self.batch_size // self.dim,
        }