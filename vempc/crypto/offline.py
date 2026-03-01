"""
Offline protocol for Variational Encrypted MPC (Algorithm 1 from the paper).

Supports parallel cache generation across N_WORKERS independent processes, each handling K_chunk = K // N_WORKERS samples with a shared key set.

Slot layout:
  - L_U perturbations: slot width `dim` (= N*m), sample i at [i*dim, i*dim+dim)
  - Gamma perturbations: slot width `p` (= num constraints), sample i at [i*p, i*p+p)

Diagonal encoding strategy:
  - L_U is lower triangular (dim x dim): requires exactly `dim` diagonals.
  - Gamma is rectangular (p x dim): padded to (p x p) square, requires `p` diagonals, with xi padded from length dim to length p with zeros.
  - When p <= dim, Gamma can be handled with dim diagonals; the p > dim case requires p diagonals to avoid row undersampling.
"""

import os
import time
import numpy as np
from multiprocessing import Pool

from openfhe import SerializeToFile, BINARY
from .setup import CryptoSetup


# ------------------------------------------------------------------
# Worker function
# ------------------------------------------------------------------

def _generate_worker_cache(args):
    """
    Worker process: generate and serialize encrypted perturbations for a single worker's cache directory.

    Parameters
    ----------
    args : tuple
        (worker_id, cache_dir, T, K_chunk, dim, p, L_U, Gamma, crypto_dir)
    """
    worker_id, cache_dir, T, K_chunk, dim, p, L_U, Gamma, crypto_dir = args

    os.makedirs(cache_dir, exist_ok=True)

    # Load shared crypto context from disk
    crypto = CryptoSetup(dim=dim, k=K_chunk)
    crypto.load(crypto_dir)

    cc  = crypto.cc
    pub = crypto.keys.publicKey
    bs  = crypto.batch_size

    # -- L_U: lower triangular (dim x dim), exactly dim diagonals --
    L_U_diags = []
    for k in range(dim):
        diag = [L_U[i, (i + k) % dim] for i in range(dim)]
        L_U_diags.append(cc.Encrypt(pub, cc.MakeCKKSPackedPlaintext(diag)))

    # -- Gamma: rectangular (p x dim), padded to (p x p), p diagonals --
    Gamma_sq = np.zeros((p, p))
    Gamma_sq[:p, :dim] = Gamma
    Gamma_diags = []
    for k in range(p):
        diag = [Gamma_sq[i, (i + k) % p] for i in range(p)]
        Gamma_diags.append(cc.Encrypt(pub, cc.MakeCKKSPackedPlaintext(diag)))

    # -- Unified encrypted matrix-vector product --
    def _enc_matvec(ct_diags, xi):
        """xi must have length == len(ct_diags)."""
        ct = None
        for k, ct_d in enumerate(ct_diags):
            rot  = list(xi[k:]) + list(xi[:k])
            ptxt = cc.MakeCKKSPackedPlaintext(rot)
            term = cc.EvalMult(ct_d, ptxt)
            ct   = term if ct is None else cc.EvalAdd(ct, term)
        return ct

    def _pack(ct_list, slot_width):
        packed = None
        for k, ct in enumerate(ct_list):
            r = (-k * slot_width) % bs
            if r != 0:
                mask = [1.0] * slot_width + [0.0] * (bs - slot_width)
                ct = cc.EvalMult(ct, cc.MakeCKKSPackedPlaintext(mask))
                ct = cc.EvalRotate(ct, r)
            packed = ct if packed is None else cc.EvalAdd(packed, ct)
        return packed

    for t in range(T):
        # xi has length dim; pad to p for Gamma multiplication
        xi_batch     = [np.random.randn(dim) for _ in range(K_chunk)]
        xi_batch_pad = [np.concatenate([xi, np.zeros(p - dim)]) for xi in xi_batch]

        ct_Lu_list    = [_enc_matvec(L_U_diags,   xi)     for xi in xi_batch]
        ct_Gamma_list = [_enc_matvec(Gamma_diags, xi_pad) for xi_pad in xi_batch_pad]

        ct_Lu_packed    = _pack(ct_Lu_list,    slot_width=dim)
        ct_Gamma_packed = _pack(ct_Gamma_list, slot_width=p)

        SerializeToFile(os.path.join(cache_dir, f"ct_Lu_{t:04d}.bin"),    ct_Lu_packed,    BINARY)
        SerializeToFile(os.path.join(cache_dir, f"ct_Gamma_{t:04d}.bin"), ct_Gamma_packed, BINARY)

    print(f"    Worker {worker_id} done -> {cache_dir}")


# ------------------------------------------------------------------
# OfflinePreprocessor
# ------------------------------------------------------------------

class OfflinePreprocessor:
    """
    Implements the offline protocol (Algorithm 1) with parallel cache generation across N_WORKERS processes.

    Parameters
    ----------
    crypto : CryptoSetup
        Initialized crypto context (setup() already called).
    L_U : ndarray, shape (Nm, Nm)
        Cholesky factor of Sigma_U = L_U L_U^T.
    G : ndarray, shape (p, Nm)
        Condensed constraint matrix.
    n_workers : int, default=4
        Number of parallel worker processes for cache generation.
    """

    def __init__(
        self,
        crypto: CryptoSetup,
        L_U: np.ndarray,
        G: np.ndarray,
        n_workers: int = 4,
    ):
        self.crypto    = crypto
        self.L_U       = np.asarray(L_U, dtype=float)
        self.G         = np.asarray(G,   dtype=float)
        self.n_workers = n_workers

        self.dim   = L_U.shape[0]
        self.p     = G.shape[0]
        self.Gamma = G @ L_U  # shape (p, dim); state-independent

        self.ct_L_U_diags   = []
        self.ct_Gamma_diags = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rotate_vector(vec, k):
        vec = list(vec)
        k   = k % len(vec)
        return vec[k:] + vec[:k]

    def _enc_matvec(self, ct_diags, xi):
        """Encrypted matrix-vector product. xi must have length == len(ct_diags)."""
        cc = self.crypto.cc
        ct = None
        for k, ct_d in enumerate(ct_diags):
            rot  = self._rotate_vector(xi, k)
            ptxt = cc.MakeCKKSPackedPlaintext(rot)
            term = cc.EvalMult(ct_d, ptxt)
            ct   = term if ct is None else cc.EvalAdd(ct, term)
        return ct

    def _pack_samples(self, ct_list, slot_width):
        cc = self.crypto.cc
        bs = self.crypto.batch_size
        packed = None
        for k, ct in enumerate(ct_list):
            r = (-k * slot_width) % bs
            if r != 0:
                mask = [1.0] * slot_width + [0.0] * (bs - slot_width)
                ct   = cc.EvalMult(ct, cc.MakeCKKSPackedPlaintext(mask))
                ct   = cc.EvalRotate(ct, r)
            packed = ct if packed is None else cc.EvalAdd(packed, ct)
        return packed

    def _register_rotation_keys(self, K_chunk):
        bs      = self.crypto.batch_size
        amounts = list(set(
            [(-k * self.dim) % bs for k in range(1, K_chunk)] +
            [(-k * self.p)   % bs for k in range(1, K_chunk)] +
            list(range(1, self.p))   # rotations for sum_within_slots in poly_eval
        ))
        self.crypto.generate_rotation_keys(amounts)

    # ------------------------------------------------------------------
    # Client side: encrypt L_U and Gamma
    # ------------------------------------------------------------------

    def encrypt_factors(self, K_chunk: int):
        """
        Client step: encrypt diagonals of L_U and Gamma = G L_U.

        L_U is lower triangular (dim x dim): dim diagonals.
        Gamma is rectangular (p x dim): padded to (p x p), p diagonals.
        xi is padded from dim to p with zeros for Gamma multiplication.
        """
        cc, pub = self.crypto.cc, self.crypto.keys.publicKey
        self._register_rotation_keys(K_chunk)

        # L_U: lower triangular, dim diagonals
        self.ct_L_U_diags = []
        for k in range(self.dim):
            diag = [self.L_U[i, (i + k) % self.dim] for i in range(self.dim)]
            self.ct_L_U_diags.append(cc.Encrypt(pub, cc.MakeCKKSPackedPlaintext(diag)))

        # Gamma: pad to (p x p), p diagonals
        Gamma_sq = np.zeros((self.p, self.p))
        Gamma_sq[:self.p, :self.dim] = self.Gamma
        self.ct_Gamma_diags = []
        for k in range(self.p):
            diag = [Gamma_sq[i, (i + k) % self.p] for i in range(self.p)]
            self.ct_Gamma_diags.append(cc.Encrypt(pub, cc.MakeCKKSPackedPlaintext(diag)))

    # ------------------------------------------------------------------
    # Cloud side: parallel cache generation
    # ------------------------------------------------------------------

    def generate_cache(
        self,
        T: int,
        cache_base_dir: str,
        K: int,
        crypto_dir: str,
    ):
        """
        Cloud step: generate encrypted perturbation cache in parallel.

        Spawns N_WORKERS processes, each generating (K_chunk = K // N_WORKERS) samples per timestep into its own subdirectory.

        Parameters
        ----------
        T : int
            Number of timesteps.
        cache_base_dir : str
            Base directory; worker w writes to cache_base_dir/worker_w/.
        K : int
            Total sample count. Must be divisible by n_workers.
        crypto_dir : str
            Directory containing serialized crypto context and keys.
        """
        assert K % self.n_workers == 0, "K must be divisible by n_workers."
        K_chunk = K // self.n_workers

        worker_args = [
            (
                w,
                os.path.join(cache_base_dir, f"worker_{w}"),
                T,
                K_chunk,
                self.dim,
                self.p,
                self.L_U,    # pass actual L_U to workers
                self.Gamma,
                crypto_dir,
            )
            for w in range(self.n_workers)
        ]

        print(f"Generating cache: T={T}, K={K}, n_workers={self.n_workers}, K_chunk={K_chunk}")
        t_start = time.time()

        with Pool(processes=self.n_workers) as pool:
            pool.map(_generate_worker_cache, worker_args)

        elapsed = time.time() - t_start
        print(f"Done: {elapsed:.2f} s total.")
        self._report_cache_size(cache_base_dir, T)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _report_cache_size(self, cache_base_dir, T):
        path_lu = os.path.join(cache_base_dir, "worker_0", "ct_Lu_0000.bin")
        path_g  = os.path.join(cache_base_dir, "worker_0", "ct_Gamma_0000.bin")
        if os.path.exists(path_lu) and os.path.exists(path_g):
            sz_lu = os.path.getsize(path_lu)
            sz_g  = os.path.getsize(path_g)
            total = (sz_lu + sz_g) * T * self.n_workers
            print(f"    ct_Lu (per worker):   {sz_lu/1024:.1f} KB")
            print(f"    ct_Gamma (per worker):{sz_g/1024:.1f} KB")
            print(f"    Total cache:          {total/(1024**2):.1f} MB")