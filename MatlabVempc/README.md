# MATLAB Implementation (MatlabVempc)

A MATLAB implementation of the paper *Variational Encrypted Model Predictive
Control* (Suh, Jang, Kim, Tanaka — IEEE L-CSS 2026), with **two entry points**:

| Entry point | What it runs | Needs Go? |
|-------------|--------------|-----------|
| `run_vempc` | **Plaintext** variational MPC (Section IV) — pure MATLAB | No |
| `run_vempc_encrypted` | **Encrypted** VEMPC (Algorithms 1–2) — MATLAB orchestrates the real CKKS engine | Yes |

`run_vempc` implements the variational reformulation (tilted Gaussian sampling +
Chebyshev feasibility surrogate) entirely in MATLAB and reproduces the
closed-loop behavior of Fig. 1.

`run_vempc_encrypted` runs the **genuine 128-bit-secure CKKS protocol**: the
client-side preparation and orchestration are in MATLAB, while the homomorphic
cloud computation is delegated to the project's Lattigo (Go) engine
(`GoVempc/cmd/ckks_offline` + `ckks_online`). CKKS is not available natively in
MATLAB (no multiprecision/RLWE library comparable to OpenFHE/Lattigo), so the
encryption core is reused rather than reimplemented; MATLAB drives the full
pipeline and overlays the encrypted trajectory against the plaintext one.

No toolboxes are required for `run_vempc`. If the Optimization Toolbox is
installed, a standard QP-MPC reference (`quadprog`) is also overlaid.

## Layout

```
MatlabVempc/
├── run_vempc.m                 % entry point: plaintext variational MPC
├── run_vempc_encrypted.m       % entry point: encrypted VEMPC (drives Go CKKS)
├── config/
│   └── vempc_config.json       % parameters (same schema as the Go config)
├── +vempc/                     % MATLAB package (namespace vempc.*)
│   ├── invertedPendulum.m      % continuous linearized pendulum model
│   ├── discretizeZOH.m         % ZOH discretization via expm
│   ├── MPCProblem.m            % condensed LQ-MPC matrices (Lambda,Psi,P,S,H)
│   ├── ConstraintPenalty.m     % constraint residual + feasibility
│   ├── VariationalMPC.m        % tilted sampling + surrogate weighting
│   ├── chebyshevReLUCoeffs.m   % Chebyshev fit of ReLU (feasibility surrogate)
│   ├── chebVal.m               % Clenshaw evaluation of a Chebyshev series
│   ├── sampleVariationalControl.m % one online MPC step (plaintext)
│   ├── simulate.m              % closed-loop rollout
│   ├── trajectoryCost.m        % realized LQ cost
│   ├── maxConstraintViolation.m   % max state/input violation
│   ├── loadConfig.m            % JSON loader with paper defaults
│   ├── findGo.m                % locate the Go toolchain (encrypted path)
│   ├── writeGoConfig.m         % MATLAB cfg -> GoVempc ckks_config.json
│   └── runGoEncrypted.m        % run Go offline+online, read encrypted outputs
└── output/                     % generated CSVs + figures (git-ignored)
```

## Quickstart

From MATLAB:

```matlab
cd MatlabVempc
run_vempc                       % uses config/vempc_config.json
% or with a custom config:
run_vempc('path/to/config.json')
% capture trajectories/metrics:
results = run_vempc;
```

Or from a shell (headless):

```bash
cd MatlabVempc
matlab -batch "run_vempc"
```

Outputs (`xs_variational.csv`, `us_variational.csv`, `variational_mpc.png`, and
the standard-MPC CSVs when `quadprog` is available) are written to
`MatlabVempc/output/`.

### Encrypted path (real CKKS via Go)

```matlab
cd MatlabVempc
run_vempc_encrypted             % runs the genuine CKKS protocol, then overlays
```

This requires the **Go toolchain** (https://go.dev/dl). On first run, fetch the
Go dependencies once:

```bash
cd GoVempc
go mod download
```

`run_vempc_encrypted` then:
1. writes the shared config to `GoVempc/output/ckks_config.json`,
2. runs `cmd/ckks_offline` (Algorithm 1) → encrypted cache,
3. runs `cmd/ckks_online` (Algorithm 2) → encrypted closed-loop trajectory,
4. reads the results back, computes cost/violation and the CKKS approximation
   error vs the plaintext run, and saves `encrypted_vs_plaintext.png` +
   `xs_ckks.csv` / `us_ckks.csv` to `MatlabVempc/output/`.

If Go is not found, the function prints setup instructions and returns without
error (the plaintext `run_vempc` always works without Go).

## Compatibility with the Go / Python ports

This port deliberately keeps the **same algorithm and the same parameters** as
the Go and Python implementations — only the execution environment differs:

- **Identical model & math.** Same 2-state inverted pendulum, same condensed
  LQ-MPC matrices (Λ, Ψ, P, S, H), same tilted Gaussian sampling (Theorem 1),
  and the same Chebyshev ReLU feasibility surrogate with threshold correction.
  The numerics were cross-checked against `GoVempc` (Clenshaw evaluation and the
  least-squares Chebyshev fit are ported line-for-line).
- **Identical config schema.** `config/vempc_config.json` is the same schema as
  `GoVempc/output/ckks_config.json`. You can point MATLAB at a Go config file
  directly and it just works:

  ```matlab
  run_vempc('../GoVempc/output/ckks_config.json')
  ```

  The CKKS-only fields (`logN`, `logQ`, `logP`, `logDefaultScale`, `nWorkers`)
  are accepted for parity but ignored by the plaintext port. `TSteps` takes
  precedence over `T` (same rule as Go), and the RNG is seeded with `0` to match
  the Go reference's `rand.Seed(0)`.
- **What is *not* portable.** Go, Python (NumPy), and MATLAB use different
  pseudo-random generators, so individual sample realizations — and therefore
  bit-exact trajectories — cannot match across languages. The *statistics* do:
  the variational solution tracks the standard QP-MPC solution (cost ≈ 34.4,
  zero constraint violation) in every implementation.

## Configuration

`config/vempc_config.json` uses the same schema as
`GoVempc/output/ckks_config.json`; the CKKS-only fields (`logN`, `logQ`, …) are
accepted but ignored by the plaintext port. The defaults reproduce the
numerical example of the paper (Section V):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `m, l, g` | 0.2, 0.5, 9.81 | pendulum mass / length / gravity |
| `dt, N`   | 0.05, 10 | sampling period, prediction horizon |
| `QDiag, RDiag, QfScale` | [50, 5], [0.1], 2.0 | LQ weights (`Qf = 2Q`) |
| `x0` | [0.3, 0.1] | initial `[theta, theta_dot]` |
| `thetaMax, omegaMax, uMax` | 0.5, 0.8, 1.0 | box constraints (→ `p = 60`) |
| `sigma0, lambda, K` | 0.25, 0.1, 240 | prior std, temperature, # samples |
| `chebOrder, chebBound, chebEta` | 3, 5.0, 500 | feasibility-surrogate params |
| `T` (or `TSteps`) | 40 | simulation steps (`TSteps` overrides `T`) |
| `logN, logQ, logP, logDefaultScale, nWorkers` | — | CKKS-only; ignored here |

## Notes

- Each online step draws `K` samples from the tilted Gaussian
  `N(m_U(x0), Sigma_U)` and weights them with the Chebyshev surrogate of the
  feasibility indicator — exactly the plaintext math of the Go/Python cores.
- The RNG is fixed to seed `0` (matching the Go reference) for reproducibility.
  Add an optional `"seed"` field to the JSON to draw a different realization.
- Verified against the Go reference: the variational trajectory overlays the
  standard QP-MPC solution and satisfies all constraints.
