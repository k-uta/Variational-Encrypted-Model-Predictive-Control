"""
Online protocol for Variational Encrypted MPC (Algorithm 2).

Supports parallel execution across N_WORKERS processes, each handling K_chunk = K // N_WORKERS samples from its own cache subdirectory.

Slot layout (must match offline.py and poly_eval.py):
  - ct_U: slot width `dim`, sample i at [i*dim, i*dim+dim)
  - ct_s: slot width `p`,   score i at slot i*p
"""

import os
import time
import numpy as np
from multiprocessing import Pool

from openfhe import DeserializeCiphertext, BINARY
from .setup import CryptoSetup
from .poly_eval import PolyEvaluator


# ------------------------------------------------------------------
# Worker function
# ------------------------------------------------------------------

# Per-process global state (initialized once per worker process)
_worker_state = {}

def _worker_init(crypto_dir, dim, k_chunk, ring_dim, cheb_mono_coeffs, p):
    """Initializer: load crypto context once per worker process."""
    global _worker_state
    crypto = CryptoSetup(dim=dim, k=k_chunk, ring_dim=ring_dim)
    crypto.load(crypto_dir)
    poly_eval = PolyEvaluator(crypto, cheb_mono_coeffs, p=p)
    _worker_state = {
        "crypto":    crypto,
        "poly_eval": poly_eval,
        "dim":       dim,
        "p":         p,
        "k_chunk":   k_chunk,
    }

def _worker_noop(_):
    """No-op task used to synchronize pool initialization."""
    return True

def _worker_run_cycle(args):
    """
    Worker process: one online cycle for a single worker's cache chunk.

    Parameters
    ----------
    args : tuple
        (t, cache_dir, m_U_arr, b_arr)
    """
    t, cache_dir, m_U_arr, b_arr = args

    s      = _worker_state
    cc     = s["crypto"].cc
    sk     = s["crypto"].keys.secretKey
    pub    = s["crypto"].keys.publicKey
    bs     = s["crypto"].batch_size
    dim    = s["dim"]
    p      = s["p"]
    K      = s["k_chunk"]

    # -- Load cached perturbations --
    def _load(path):
        ct, ok = DeserializeCiphertext(path, BINARY)
        if not ok:
            raise RuntimeError(f"Failed to deserialize: {path}")
        return ct

    ct_Lu    = _load(os.path.join(cache_dir, f"ct_Lu_{t:04d}.bin"))
    ct_Gamma = _load(os.path.join(cache_dir, f"ct_Gamma_{t:04d}.bin"))

    # -- Tile and encrypt state-dependent quantities --
    def _tile_encrypt(vec, slot_width):
        tiled  = np.tile(vec[:slot_width], K).tolist()
        padded = tiled + [0.0] * (bs - len(tiled))
        return cc.Encrypt(pub, cc.MakeCKKSPackedPlaintext(padded))

    ct_mU = _tile_encrypt(m_U_arr, dim)
    ct_b  = _tile_encrypt(b_arr,   p)

    # -- Cloud compute (Algorithm 2, lines 8-10) --
    ct_U = cc.EvalAdd(ct_mU, ct_Lu)
    ct_g = cc.EvalAdd(ct_b,  ct_Gamma)
    ct_s = s["poly_eval"].evaluate_poly(ct_g)

    # -- Decrypt --
    ptxt_U = cc.Decrypt(ct_U, sk)
    ptxt_U.SetLength(K * dim)
    flat_U    = [x.real for x in ptxt_U.GetCKKSPackedValue()]
    U_samples = np.array(flat_U).reshape(K, dim)

    ptxt_s = cc.Decrypt(ct_s, sk)
    ptxt_s.SetLength(K * p)
    flat_s    = [x.real for x in ptxt_s.GetCKKSPackedValue()]
    s_samples = np.array([flat_s[i * p] for i in range(K)])

    return U_samples, s_samples


# ------------------------------------------------------------------
# OnlineController
# ------------------------------------------------------------------

class OnlineController:
    """
    Implements the online protocol (Algorithm 2) with parallel execution
    across N_WORKERS processes.

    Parameters
    ----------
    crypto : CryptoSetup
        Initialized crypto context (shared key set).
    poly_eval : PolyEvaluator
        Polynomial evaluator.
    cache_base_dir : str
        Base directory containing worker_0/, worker_1/, ... subdirectories.
    G : ndarray, shape (p, Nm)
        Condensed constraint matrix.
    h_func : callable
        h_func(x0) -> ndarray shape (p,).
    Sigma_U : ndarray, shape (Nm, Nm)
        Tilted distribution covariance.
    S : ndarray, shape (n, Nm)
        Condensed MPC cross-term matrix.
    m_in : int
        Control input dimension m.
    lambda_param : float
        Temperature parameter lambda.
    eta : float
        Penalty sharpness.
    tau_s : float
        Threshold for post-processed score (Corollary 1).
    n_workers : int, default=4
        Number of parallel worker processes.
    ring_dim : int
        CKKS ring dimension (must match offline setup).
    cheb_mono_coeffs : list
        Monomial coefficients for polynomial surrogate (ascending order).
    crypto_dir : str
        Directory containing serialized crypto context and keys.
    """

    def __init__(
        self,
        crypto: CryptoSetup,
        poly_eval: PolyEvaluator,
        cache_base_dir: str,
        G: np.ndarray,
        h_func,
        Sigma_U: np.ndarray,
        S: np.ndarray,
        m_in: int,
        lambda_param: float,
        eta: float,
        tau_s: float,
        n_workers: int = 4,
        ring_dim: int = 1 << 13,
        cheb_mono_coeffs: list = None,
        crypto_dir: str = None,
    ):
        self.crypto          = crypto
        self.poly_eval       = poly_eval
        self.cache_base_dir  = cache_base_dir
        self.G               = np.asarray(G, dtype=float)
        self.h_func          = h_func
        self.Sigma_U         = np.asarray(Sigma_U, dtype=float)
        self.S               = np.asarray(S, dtype=float)
        self.m_in            = int(m_in)
        self.lambda_param    = float(lambda_param)
        self.eta             = float(eta)
        self.tau_s           = float(tau_s)
        self.n_workers       = n_workers
        self.ring_dim        = ring_dim
        self.cheb_mono_coeffs = cheb_mono_coeffs or []
        self.crypto_dir      = crypto_dir

        self.n_workers   = int(n_workers)
        self.k_per_worker = crypto.k // n_workers  # store for use in decrypt

        self.dim = G.shape[1]
        self.p   = G.shape[0]

        assert self.p == poly_eval.p, (
            f"G.shape[0]={self.p} does not match poly_eval.p={poly_eval.p}."
        )

        # Pool initialized lazily on first run_cycle call
        self._pool = None

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def _ensure_pool(self, K: int):
        """Initialize worker pool and block until all workers are ready."""
        if self._pool is not None:
            return
        assert K % self.n_workers == 0, "K must be divisible by n_workers."
        k_chunk = K // self.n_workers
        self._pool = Pool(
            processes=self.n_workers,
            initializer=_worker_init,
            initargs=(
                self.crypto_dir,
                self.dim,
                k_chunk,
                self.ring_dim,
                self.cheb_mono_coeffs,
                self.p,
            ),
        )
        self._k_chunk = k_chunk

        # Block until all worker initializers have finished.
        # map() is synchronous: it will not return until every worker
        # has completed _worker_noop, which can only run after
        # _worker_init finishes in that process.
        self._pool.map(_worker_noop, range(self.n_workers))

    def close(self):
        """Terminate worker pool. Call when done with all cycles."""
        if self._pool is not None:
            self._pool.terminate()
            self._pool = None

    # ------------------------------------------------------------------
    # Client side: state-dependent quantities
    # ------------------------------------------------------------------

    def compute_mean(self, x0: np.ndarray) -> np.ndarray:
        """m_U(x_t) = -(1/lambda) Sigma_U S^T x_t, shape (Nm,)."""
        return -(1.0 / self.lambda_param) * self.Sigma_U @ self.S.T @ x0

    def compute_offset(self, x0: np.ndarray, m_U: np.ndarray) -> np.ndarray:
        """b(x_t) = G m_U(x_t) - h(x_t), shape (p,)."""
        return self.G @ m_U - self.h_func(x0)

    # ------------------------------------------------------------------
    # Estimate from decrypted samples
    # ------------------------------------------------------------------

    def compute_estimate(
        self,
        U_samples: np.ndarray,
        s_samples: np.ndarray,
    ):
        """
        Algorithm 2, lines 13-16: threshold, weight, weighted average.

        Parameters
        ----------
        U_samples : ndarray, shape (K, Nm)
            Decrypted control samples aggregated across all workers.
        s_samples : ndarray, shape (K,)
            Decrypted scores aggregated across all workers.

        Returns
        -------
        U_hat : ndarray, shape (Nm,)
        weights : ndarray, shape (K,)
        """
        s_bar = np.maximum(s_samples - self.tau_s, 0.0)
        log_w = -self.eta * s_bar
        log_w -= log_w.max()
        w     = np.exp(log_w)
        w_sum = w.sum()

        if w_sum == 0.0:
            raise RuntimeError(
                "All importance weights are zero. "
                "Increase K, reduce eta, or widen Sigma_U."
            )
        w /= w_sum
        U_hat = (U_samples * w[:, None]).sum(axis=0)
        return U_hat, w

    # ------------------------------------------------------------------
    # Full cycle
    # ------------------------------------------------------------------

    def run_cycle(self, t: int, x0: np.ndarray, K: int):
        """
        One complete online MPC cycle (Algorithm 2) with parallel workers.

        Parameters
        ----------
        t : int
            Current timestep index (selects cache file).
        x0 : ndarray, shape (n,)
            Current state.
        K : int
            Total number of samples (must equal n_workers * k_chunk).

        Returns
        -------
        u0 : ndarray, shape (m,)
            First control input to apply.
        U_hat : ndarray, shape (Nm,)
            Full estimated optimal control sequence.
        info : dict
            Timing and diagnostics.
        """
        t_start = time.perf_counter()

        # Ensure pool is running
        self._ensure_pool(K)

        # Client: compute state-dependent quantities
        m_U = self.compute_mean(x0)
        b   = self.compute_offset(x0, m_U)

        # Dispatch parallel worker cycles
        worker_args = [
            (
                t,
                os.path.join(self.cache_base_dir, f"worker_{w}"),
                m_U,
                b,
            )
            for w in range(self.n_workers)
        ]

        results = self._pool.map(_worker_run_cycle, worker_args)

        # Aggregate results from all workers
        U_all = np.vstack([r[0] for r in results])  # (K, dim)
        s_all = np.concatenate([r[1] for r in results])  # (K,)

        U_hat, weights = self.compute_estimate(U_all, s_all)
        u0             = U_hat[:self.m_in]

        elapsed = time.perf_counter() - t_start
        info = {
            "cycle_time_ms":  elapsed * 1000,
            "weight_entropy": float(-np.sum(weights * np.log(weights + 1e-12))),
            "s_mean":         float(s_all.mean()),
            "s_max":          float(s_all.max()),
        }
        return u0, U_hat, info