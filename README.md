# Variational Encrypted Model Predictive Control (VEMPC)

Implementation accompanying the paper:

> **Variational Encrypted Model Predictive Control**  
> Jihoon Suh, Yeongjun Jang, Junsoo Kim, and Takashi Tanaka  
> *Submitted to IEEE L-CSS / CDC 2026 (under review)*  
> Preprint: [arXiv:2603.19450](https://arxiv.org/abs/2603.19450)

---

## Overview

VEMPC is a privacy-preserving MPC protocol that outsources control computation to an untrusted cloud server. The key idea is to reformulate the constrained MPC problem as a sampling-based estimator, where the quadratic cost is absorbed into a tilted Gaussian sampling distribution. This eliminates expensive encrypted matrix-vector multiplications from the online phase, leaving only low-degree polynomial evaluations.

**Main features:**
- Online execution requires only encrypted polynomial operations (no intermediate decryption)
- Single round of client-cloud communication per MPC step
- Two-level parallelism: plaintext sample-level and ciphertext SIMD-level
- Average online time of ~28 ms at 128-bit security (Apple M5, 4 performance cores)

Three implementations are provided:
- **Python** (`vempc/`): encrypted, uses [OpenFHE-Python](https://github.com/openfheorg/openfhe-python)
- **Go** (`GoVempc/`): encrypted, uses [Lattigo](https://github.com/tuneinsight/lattigo)
- **MATLAB** (`MatlabVempc/`): plaintext variational MPC (the unencrypted baseline that VEMPC reproduces; no CKKS), requires no toolboxes

---

## Repository Structure

```
.
├── vempc/                  # Python implementation
│   ├── core/               # MPC and variational logic
│   ├── crypto/             # CKKS offline/online protocols
│   ├── solvers/            # QP solver and sampling
│   ├── backends/           # OpenFHE and plaintext backends
│   ├── examples/           # Pendulum example
│   └── notebooks/          # Jupyter notebooks for testing
├── GoVempc/                # Go implementation
│   ├── cmd/
│   │   ├── ckks_offline/   # Offline protocol binary
│   │   ├── ckks_online/    # Online protocol binary
│   │   ├── unencrypted/    # Plaintext baseline binary
│   │   └── test/           # main.ipynb and run_ablation.py
│   ├── core/               # MPC and variational logic
│   ├── solvers/            # QP and sampling
│   └── internal/           # CKKS config
├── MatlabVempc/            # MATLAB implementation (plaintext)
│   ├── run_vempc.m         # Entry point
│   ├── +vempc/             # MATLAB package (MPC + variational logic)
│   └── config/             # JSON config (shared schema with Go)
└── results/                # Figures
```

---

## Requirements

### Python

```bash
conda env create -f environment.yml
conda activate cdc26
```

OpenFHE-Python must be installed separately. Follow the instructions at https://github.com/openfheorg/openfhe-python.

### Go

Go 1.21 or later. Dependencies are managed via `go.mod`:

```bash
cd GoVempc
go mod download
```

### MATLAB

MATLAB R2019b or later. No toolboxes are required to run the variational MPC
(the Optimization Toolbox is used only for the optional `quadprog` reference).

```matlab
cd MatlabVempc
run_vempc            % or:  matlab -batch "run_vempc"  from a shell
```

This runs the plaintext variational MPC and writes trajectories + a figure to
`MatlabVempc/output/`. See [`MatlabVempc/README.md`](MatlabVempc/README.md) for
details.

---

## Quickstart (Go Implementation)

The Go implementation is the primary benchmark reference.

### 1. Set configuration

Edit `GoVempc/output/ckks_config.json` or use `GoVempc/cmd/test/main.ipynb`. The default configuration used in the paper:

```json
{
  "m": 0.2, "l": 0.5, "g": 9.81,
  "dt": 0.05, "N": 10,
  "QDiag": [50.0, 5.0], "RDiag": [0.1], "QfScale": 2.0,
  "x0": [0.3, 0.1],
  "thetaMax": 0.5, "omegaMax": 0.8, "uMax": 1.0,
  "sigma0": 0.25, "lambda": 0.1, "K": 240,
  "logN": 13, "logQ": [33, 30, 30, 30], "logP": [35],
  "logDefaultScale": 30, "chebOrder": 3,
  "chebBound": 5.0, "chebEta": 500.0,
  "T": 40, "nWorkers": 4
}
```

> **Tip:** The offline phase scales linearly with `T`. To verify the pipeline quickly, set `"T": 2` or `"T": 5` before running with the full `"T": 40`.

### 2. Run offline protocol (Algorithm 1)

```bash
cd GoVempc
go build ./cmd/ckks_offline/ && ./ckks_offline
```

### 3. Run online protocol (Algorithm 2)

```bash
go build ./cmd/ckks_online/ && ./ckks_online
```

### 4. Run unencrypted baseline

```bash
go build ./cmd/unencrypted/ && ./unencrypted
```

Outputs (state/input trajectories, timing) are saved to `GoVempc/output/`.

`GoVempc/cmd/test/main.ipynb` orchestrates all phases with visualization.

---

## Hardware

All benchmarks were run on an Apple MacBook Pro (M5, 4 performance cores, 24 GB unified memory). Running on battery power can cause significant CPU throttling and slower timing.

---

## Citation

If you find this work helpful, please cite:

```bibtex
@misc{suh2026variationalencryptedmodelpredictive,
  title         = {Variational Encrypted Model Predictive Control},
  author        = {Suh, Jihoon and Jang, Yeongjun and Kim, Junsoo and Tanaka, Takashi},
  year          = {2026},
  eprint        = {2603.19450},
  archivePrefix = {arXiv},
  primaryClass  = {eess.SY},
  url           = {https://arxiv.org/abs/2603.19450}
}
```

---

## License

MIT License. See `LICENSE` for details.