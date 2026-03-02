#!/usr/bin/env python
# coding: utf-8

"""
Encrypted End-to-End Simulation -- Cart-Pole (Inverted Pendulum).

System:  linearized cart-pole about the upright equilibrium.
State:   x = [cart_pos, cart_vel, theta, theta_dot]
Input:   u = horizontal cart force

Slot budget at N=6, K=500, N_WORKERS=4:
    p = (8+2)*6 = 60 constraints
    K_chunk = 500/4 = 125
    K_chunk * p = 7500 < BATCH_SIZE=8192  (RING_DIM=1<<14)
    K_chunk * Nm = 125 * 6 = 750 < 8192
    => RING_DIM = 1<<14 is the minimum required.

Set REGENERATE_CACHE = True to redo the offline phase.
Set REGENERATE_CACHE = False to reuse an existing cache.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "../..")

# -- set OMP threads before importing openfhe --
os.environ["OMP_NUM_THREADS"] = "1"  # n_workers * omp_threads <= num_cores

import vempc.examples.pendulum as pend
from vempc.examples.pendulum import A, B, Q, R, Qf, Gx, hx, Gu, hu
from vempc.core.mpc import MPCProblem, simulate, trajectory_cost, max_constraint_violation
from vempc.core.constraints import ConstraintPenalty
from vempc.core.variational import VariationalMPC
from vempc.solvers.qpMPC import (
    make_standard_qp_solver, solve_standard_mpc,
    chebyshev_relu_coeffs, eval_relu_poly,
)
from vempc.crypto.setup import CryptoSetup
from vempc.crypto.offline import OfflinePreprocessor
from vempc.crypto.online import OnlineController
from vempc.crypto.poly_eval import PolyEvaluator

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
REGENERATE_CACHE = True
CACHE_BASE_DIR   = "../../cache_pend"
CRYPTO_DIR       = "../../cache_pend/crypto"

N         = 6    # MPC horizon; longer horizon improves stabilization of
                 # the unstable cart-pole system
K         = 500  # total samples
N_WORKERS = 4    # K_chunk = 125
T_steps   = 30

# Small initial deviation from upright equilibrium.
# Angle of 0.15 rad (~8.6 deg) is well within the relaxed theta_max=0.5.
x0 = np.array([0.0, 0.0, 0.15, 0.0])   # [pos, vel, theta, omega]

# RING_DIM=1<<14 required: K_chunk*p = 125*60 = 7500 < BATCH_SIZE=8192.
RING_DIM  = 1 << 14

# ------------------------------------------------------------------
# Controller functions (module-level for pickling across worker processes)
# ------------------------------------------------------------------
def qp_controller(x, U_warm):
    u0, Useq = solve_standard_mpc(
        x, S=_mpc.S, h_of_x0=_h_of_x0, m_in=_mpc.m,
        prob=_prob, U_var=_U_var, q_par=_q_par, h_par=_h_par,
        warm_start_U=U_warm,
    )
    return u0, Useq, {}

_enc_cycle = [0]

def encrypted_controller(x, U_warm):
    t              = _enc_cycle[0]
    _enc_cycle[0] += 1
    u0, U_hat, info = _online.run_cycle(t=t, x0=x, K=K)
    return u0, U_hat, info


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == '__main__':

    mpc        = MPCProblem(A, B, Q, R, Qf, N)
    G, h_of_x0 = mpc.build_constraint_matrices(Gx, hx, Gu, hu)
    p          = G.shape[0]
    K_chunk    = K // N_WORKERS

    # Expose globals used by module-level controller functions
    _mpc      = mpc
    _h_of_x0  = h_of_x0

    # ------------------------------------------------------------------
    # Variational parameters
    # ------------------------------------------------------------------
    # sigma_0=3.0 gives broader exploration; necessary for the encrypted
    # controller to find enough feasible samples with K=500.
    # lambda_param=0.5 sharpens the importance weighting.
    lambda_param = 0.5
    sigma_0      = 3.0
    Sigma_0      = sigma_0**2 * np.eye(mpc.Nm)

    cheb_order       = 3
    cheb_bound       = 2.0
    eta              = 1.0
    cheb_coeffs      = chebyshev_relu_coeffs(cheb_order, cheb_bound)
    tau_s            = p * float(eval_relu_poly(0.0, cheb_coeffs, cheb_bound))
    cheb_mono_coeffs = list(np.polynomial.chebyshev.cheb2poly(cheb_coeffs))

    # ------------------------------------------------------------------
    # Slot budget verification
    # ------------------------------------------------------------------
    BATCH_SIZE = RING_DIM // 2
    print(f"Nm={mpc.Nm}, p={p}, K={K}, K_chunk={K_chunk}, N_WORKERS={N_WORKERS}")
    print(f"K_chunk*Nm={K_chunk*mpc.Nm}, K_chunk*p={K_chunk*p}, batch_size={BATCH_SIZE}")
    print(f"tau_s={tau_s:.6f}")
    assert K_chunk * mpc.Nm <= BATCH_SIZE, "K_chunk*Nm exceeds batch_size."
    assert K_chunk * p      <= BATCH_SIZE, "K_chunk*p exceeds batch_size."
    assert K % N_WORKERS    == 0,          "K must be divisible by N_WORKERS."

    # ------------------------------------------------------------------
    # Derive Sigma_U and L_U from variational formulation
    # ------------------------------------------------------------------
    penalty     = ConstraintPenalty(G, h_of_x0, mode='indicator')
    variational = VariationalMPC(mpc, penalty, lambda_param=lambda_param, Sigma_0=Sigma_0)
    Sigma_U     = variational.Sigma_U
    L_U         = variational.L_U

    # ------------------------------------------------------------------
    # 1. Crypto setup
    # ------------------------------------------------------------------
    crypto = CryptoSetup(
        dim=mpc.Nm,
        k=K_chunk,
        ring_dim=RING_DIM,
        mult_depth=7,
        scaling_mod=50,
    )

    if REGENERATE_CACHE:
        print("\n[1/4] Setting up crypto context ...")
        crypto.setup()
        print(f"      Slot budget: {crypto.slot_budget()}")
    else:
        print("\n[1/4] Loading crypto context ...")
        crypto.load(CRYPTO_DIR)
        print(f"      Slot budget: {crypto.slot_budget()}")

    # ------------------------------------------------------------------
    # 2. Offline protocol (Algorithm 1)
    # ------------------------------------------------------------------
    poly_eval = PolyEvaluator(crypto, cheb_mono_coeffs, p=p)
    offline   = OfflinePreprocessor(crypto, L_U, G, n_workers=N_WORKERS)

    if REGENERATE_CACHE:
        print("\n[2/4] Encrypting L_U and Gamma ...")
        offline.encrypt_factors(K_chunk=K_chunk)

        print("      Registering rotation keys ...")
        poly_eval.register_rotation_keys()

        print("      Saving crypto context and keys ...")
        crypto.save(CRYPTO_DIR)

        print("\n[3/4] Generating parallel cache ...")
        offline.generate_cache(
            T=T_steps,
            cache_base_dir=CACHE_BASE_DIR,
            K=K,
            crypto_dir=CRYPTO_DIR,
        )
    else:
        print("\n[2/4] Skipping offline phase (reusing cache).")
        print("[3/4] Registering rotation keys ...")
        poly_eval.register_rotation_keys()

    # ------------------------------------------------------------------
    # 3. Online controller setup (Algorithm 2)
    # ------------------------------------------------------------------
    print("\n[4/4] Setting up parallel online controller ...")
    _online = OnlineController(
        crypto=crypto,
        poly_eval=poly_eval,
        cache_base_dir=CACHE_BASE_DIR,
        G=G,
        h_func=h_of_x0,
        Sigma_U=Sigma_U,
        S=mpc.S,
        m_in=mpc.m,
        lambda_param=lambda_param,
        eta=eta,
        tau_s=tau_s,
        n_workers=N_WORKERS,
        ring_dim=RING_DIM,
        cheb_mono_coeffs=cheb_mono_coeffs,
        crypto_dir=CRYPTO_DIR,
    )

    # ------------------------------------------------------------------
    # 4. Pool initialization (warm-up)
    # ------------------------------------------------------------------
    print("\nInitializing worker pool (warm-up) ...")
    t_warmup_start = time.perf_counter()
    _online._ensure_pool(K)
    t_warmup = time.perf_counter() - t_warmup_start
    print(f"      Worker pool ready in {t_warmup*1000:.1f} ms")

    # ------------------------------------------------------------------
    # 5. QP baseline
    # ------------------------------------------------------------------
    _prob, _U_var, _q_par, _h_par = make_standard_qp_solver(mpc.H, G)

    # ------------------------------------------------------------------
    # 6. Run simulations
    # ------------------------------------------------------------------
    print("\nRunning QP baseline ...")
    xs_qp, us_qp, _, t_qp = simulate(
        "QP", x0, qp_controller, A=A, B=B, T_steps=T_steps)

    _enc_cycle[0] = 0

    print("\n\n\n ******** Running encrypted VEMPC ********\n\n\n")
    t_enc_start = time.perf_counter()
    xs_enc, us_enc, info_enc, _ = simulate(
        "Encrypted", x0, encrypted_controller, A=A, B=B, T_steps=T_steps)
    t_enc = time.perf_counter() - t_enc_start

    _online.close()

    # ------------------------------------------------------------------
    # 7. Results summary
    # ------------------------------------------------------------------
    cost_qp  = trajectory_cost(xs_qp,  us_qp,  Q=Q, R=R, Qf=Qf)
    cost_enc = trajectory_cost(xs_enc, us_enc, Q=Q, R=R, Qf=Qf)
    vx_qp,  vu_qp  = max_constraint_violation(xs_qp,  us_qp,  Gx=Gx, hx=hx, Gu=Gu, hu=hu)
    vx_enc, vu_enc = max_constraint_violation(xs_enc, us_enc, Gx=Gx, hx=hx, Gu=Gu, hu=hu)
    cycle_times    = [d["cycle_time_ms"] for d in info_enc]

    print(f"\n{'':20s} {'QP':>10s} {'Encrypted':>12s}")
    print(f"{'Trajectory cost':20s} {cost_qp:>10.3f} {cost_enc:>12.3f}")
    print(f"{'Max state viol.':20s} {vx_qp:>10.4f} {vx_enc:>12.4f}")
    print(f"{'Max input viol.':20s} {vu_qp:>10.4f} {vu_enc:>12.4f}")
    print(f"{'Solve time (s)':20s} {t_qp:>10.3f} {t_enc:>12.3f}")
    print(f"{'Pool warm-up (ms)':20s} {'':>10s} {t_warmup*1000:>12.1f}")
    print(f"{'Avg. time / cycle (ms)':20s} {'':>10s} {np.mean(cycle_times):>12.2f}")
    print("\nPer-cycle times:\n")
    for i, info in enumerate(info_enc):
        print(f"    t={i}: {info['cycle_time_ms']:.1f} ms")

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------
    t_axis  = np.arange(T_steps + 1)
    tu_axis = np.arange(T_steps)
    labels  = ["Standard MPC", "Encrypted VEMPC"]
    xs_list = [xs_qp, xs_enc]
    us_list = [us_qp, us_enc]
    colors  = ["tab:blue", "tab:orange"]

    state_labels = ["cart pos (m)", "cart vel (m/s)", "theta (rad)", "omega (rad/s)"]
    state_bounds = [pend.x_max, pend.v_max, pend.theta_max, pend.omega_max]

    # -- Full trajectory plots with constraint bounds --
    fig, axes = plt.subplots(5, 1, figsize=(9, 12), sharex=True)

    for si in range(4):
        for xs, lbl, col in zip(xs_list, labels, colors):
            axes[si].plot(t_axis, xs[:, si], label=lbl, color=col, linewidth=2.0)
        axes[si].axhline( state_bounds[si], color="gray", linestyle="--", linewidth=1.5)
        axes[si].axhline(-state_bounds[si], color="gray", linestyle="--", linewidth=1.5)
        axes[si].set_ylabel(state_labels[si])
        axes[si].grid(True, alpha=0.3)

    axes[0].legend(fontsize=10)

    for us, lbl, col in zip(us_list, labels, colors):
        axes[4].step(tu_axis, us[:, 0], where="post", label=lbl, color=col, linewidth=2.0)
    axes[4].axhline( pend.u_max, color="gray", linestyle="--", linewidth=1.5)
    axes[4].axhline(-pend.u_max, color="gray", linestyle="--", linewidth=1.5)
    axes[4].set_ylabel("input u (N)")
    axes[4].set_xlabel("time step")
    axes[4].grid(True, alpha=0.3)

    fig.suptitle(f"Cart-Pole: Encrypted VEMPC ({N_WORKERS} workers, N={N}) vs QP")
    plt.tight_layout()
    plt.savefig("encrypted_simulation_pendulum.png", dpi=150)
    plt.show()

    # -- Zoomed comparison: Standard vs Encrypted (auto-scaled, no constraint lines) --
    fig2, axes2 = plt.subplots(5, 1, figsize=(9, 12), sharex=True)

    for si in range(4):
        for xs, lbl, col in zip(xs_list, labels, colors):
            axes2[si].plot(t_axis, xs[:, si], label=lbl, color=col, linewidth=2.0)
        axes2[si].set_ylabel(state_labels[si])
        axes2[si].grid(True, alpha=0.3)

    axes2[0].legend(fontsize=10)

    for us, lbl, col in zip(us_list, labels, colors):
        axes2[4].step(tu_axis, us[:, 0], where="post", label=lbl, color=col, linewidth=2.0)
    axes2[4].set_ylabel("input u (N)")
    axes2[4].set_xlabel("time step")
    axes2[4].grid(True, alpha=0.3)

    fig2.suptitle(f"Cart-Pole: Zoomed Comparison (Standard vs Encrypted, N={N})")
    plt.tight_layout()
    plt.savefig("encrypted_simulation_pendulum_zoomed.png", dpi=150)
    plt.show()

    # -- Cycle time plot --
    plt.figure(figsize=(8, 3))
    plt.plot(np.arange(T_steps), cycle_times, linewidth=2.0)
    plt.ylabel("cycle time (ms)")
    plt.xlabel("time step")
    plt.title(f"Cart-Pole Encrypted VEMPC: Time per Cycle ({N_WORKERS} workers, K={K})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
