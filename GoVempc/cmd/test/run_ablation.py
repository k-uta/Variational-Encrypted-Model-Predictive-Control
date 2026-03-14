"""
run_ablation.py
===============

    rows:    ell (chebOrder) in {3, 4, 5}
    columns: (logN=13, K1), (logN=13, K2), (logN=14, K1), (logN=14, K2)

Run:
    caffeinate -i python run_ablation.py

Dry-run (print grid only):
    python run_ablation.py --dry-run

Resume after interruption: re-run; completed tags are skipped.
"""

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

def find_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "go.mod").exists() and (p / "cmd" / "ckks_offline").exists():
            return p
        if (p / "GoVempc" / "go.mod").exists():
            return p / "GoVempc"
    return start

ROOT = find_root(Path.cwd())
os.chdir(str(ROOT))
os.environ["GOMODCACHE"] = str(ROOT / ".gomodcache")
os.environ["GOCACHE"]    = str(ROOT / ".gocache")

# ---------------------------------------------------------------------------
# Fixed sweep parameters
# ---------------------------------------------------------------------------

N_WORKERS   = 4
CHEB_ORDERS = [3, 4, 5]

# logQ chain length must be >= chebOrder + 1
CHEB_ORDER_TO_LOGQ = {
    3: [30, 30, 30, 30],
    4: [30, 30, 30, 30, 30],
    5: [30, 30, 30, 30, 30, 30],
}

# ---------------------------------------------------------------------------
# Slot budget helpers
# ---------------------------------------------------------------------------

def slot_width(cfg: dict) -> int:
    """p = 2 * (nx*N_hor + m*N_hor), the slot footprint per sample."""
    N_hor = cfg["N"]
    nx    = len(cfg["x0"])
    m     = len(cfg["RDiag"])
    return 2 * (nx * N_hor + m * N_hor)

def k_pair(logN: int, cfg: dict) -> tuple:
    """
    Return (K1, K2):
    """
    slots     = 1 << (logN - 1)
    p         = slot_width(cfg)
    max_chunk = slots // p          # maximum K_chunk per worker
    K2        = max_chunk * N_WORKERS
    K1        = max(1, max_chunk // 2) * N_WORKERS
    return K1, K2

# ---------------------------------------------------------------------------
# Grid builder: 2 logN x 2 K x 3 chebOrder = 12 runs
# ---------------------------------------------------------------------------
def build_grid(base_cfg: dict) -> list:
    grid    = []
    skipped = []

    for logN in [13, 14]:
        K1, K2    = k_pair(logN, base_cfg)
        max_chunk = (1 << (logN - 1)) // slot_width(base_cfg)

        for K in (K1, K2):
            K_chunk = K // N_WORKERS
            if K_chunk > max_chunk:
                skipped.append(
                    f"  SKIP logN={logN} K={K}: K_chunk={K_chunk} > max={max_chunk}")
                continue

            for chebO in CHEB_ORDERS:
                cfg = copy.deepcopy(base_cfg)
                cfg["logN"]      = logN
                cfg["logQ"]      = CHEB_ORDER_TO_LOGQ[chebO]
                cfg["K"]         = K
                cfg["nWorkers"]  = N_WORKERS
                cfg["chebOrder"] = chebO
                tag = f"logN{logN}_K{K}_w{N_WORKERS}_c{chebO}"
                grid.append((tag, cfg))

    if skipped:
        print("Infeasible combinations skipped:")
        for s in skipped:
            print(s)

    return grid

# ---------------------------------------------------------------------------
# Single-run executor
# ---------------------------------------------------------------------------

def run_case(tag: str, cfg: dict) -> dict:
    out_dir  = ROOT / "output"
    dest_dir = out_dir / "ablation" / tag
    dest_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "tag":            tag,
        "logN":           cfg["logN"],
        "K":              cfg["K"],
        "nWorkers":       cfg["nWorkers"],
        "chebOrder":      cfg["chebOrder"],
        "online_mean_ms": float("nan"),
        "online_std_ms":  float("nan"),
        "status":         "pending",
    }

    # Overwrite shared config so Go binaries pick up this run's parameters
    cfg_path = out_dir / "ckks_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    def _go(cmd, label, parse_mean_std=False):
        """Run a Go subprocess, stream stdout, return (elapsed_s, mean_ms, std_ms)."""
        print(f"\n  [{tag}] {label}")
        t0      = time.time()
        mean_ms = float("nan")
        std_ms  = float("nan")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            cwd=str(ROOT),
        )
        if proc.stdout:
            for line in proc.stdout:
                print(f"    {line}", end="")
                if parse_mean_std and "Control iteration time mean/std:" in line:
                    # Expected format: "Control iteration time mean/std: 28.662 ms / 1.604 ms"
                    try:
                        parts   = line.split(":")[-1].strip().split("/")
                        mean_ms = float(parts[0].strip().split()[0])
                        std_ms  = float(parts[1].strip().split()[0])
                    except (IndexError, ValueError):
                        pass
        ret     = proc.wait()
        elapsed = time.time() - t0
        if ret != 0:
            raise RuntimeError(f"'{' '.join(cmd)}' exited with code {ret}")
        return elapsed, mean_ms, std_ms

    try:
        _go(["go", "run", "./cmd/ckks_offline"], "offline")
        _, row["online_mean_ms"], row["online_std_ms"] = _go(
            ["go", "run", "./cmd/ckks_online"], "online", parse_mean_std=True)

        # Archive outputs into the tag directory for later inspection
        for fname in ["xs_ckks.csv", "us_ckks.csv",
                      "xs_variational.csv", "us_variational.csv"]:
            src = out_dir / fname
            if src.exists():
                shutil.copy2(str(src), str(dest_dir / fname))
        shutil.copy2(str(cfg_path), str(dest_dir / "ckks_config.json"))
        row["status"] = "ok"

    except RuntimeError as exc:
        print(f"\n  ERROR [{tag}]: {exc}")
        row["status"] = "error"

    return row

# ---------------------------------------------------------------------------
# Summary CSV  (checkpoint after every run)
# ---------------------------------------------------------------------------

def write_summary_csv(rows: list, path: Path):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(str(path), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

def load_cached_result(tag: str, summary_path: Path) -> dict:
    """Recover mean/std from an existing summary.csv for a cached tag."""
    mean_ms = float("nan")
    std_ms  = float("nan")
    if summary_path.exists():
        with open(str(summary_path)) as f:
            hdr = f.readline().strip().split(",")
            for line in f:
                vals = dict(zip(hdr, line.strip().split(",")))
                if vals.get("tag") == tag:
                    try:
                        mean_ms = float(vals["online_mean_ms"])
                        std_ms  = float(vals["online_std_ms"])
                    except (KeyError, ValueError):
                        pass
    return mean_ms, std_ms

# ---------------------------------------------------------------------------
# LaTeX table printer
# ---------------------------------------------------------------------------

def print_latex_table(summary: list, base_cfg: dict):
    """Print the fully populated LaTeX table to stdout."""

    # Build lookup: (logN, K, chebOrder) -> (mean_ms, std_ms)
    data = {}
    for r in summary:
        if r["status"] not in ("ok", "cached"):
            continue
        try:
            mean = float(r["online_mean_ms"])
            std  = float(r["online_std_ms"])
            data[(int(r["logN"]), int(r["K"]), int(r["chebOrder"]))] = (mean, std)
        except (ValueError, TypeError):
            continue

    K1_13, K2_13 = k_pair(13, base_cfg)
    K1_14, K2_14 = k_pair(14, base_cfg)

    def cell(logN, K, chebO):
        v = data.get((logN, K, chebO))
        if v is None:
            return r"$\text{--}$"
        return f"${v[0]:.2f} \\pm {v[1]:.2f}$"

    lines = []
    lines.append(f"% K pairs:  logN=13 -> (K1={K1_13}, K2={K2_13})"
                 f"   logN=14 -> (K1={K1_14}, K2={K2_14})")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{%")
    lines.append(r"    Online time per MPC step (mean $\pm$ std, ms),")
    lines.append(r"    $N_{\rm workers} = 4$.")
    lines.append(r"    Columns show two representative $K$ values within the slot budget.")
    lines.append(r"    Near-constant values across $K$ columns confirm SIMD batching")
    lines.append(r"    efficiency; increasing $\ell$ or $\log N_{\rm ckks}$ increases time.%")
    lines.append(r"  }")
    lines.append(r"  \label{tab:ablation}")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(r"  \renewcommand{\arraystretch}{1.0}")
    lines.append(r"  \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}")
    lines.append(r"      c                  % ell")
    lines.append(r"      r r                % logN=13: K1, K2")
    lines.append(r"      @{\hspace{12pt}}   % gap between the two logN groups")
    lines.append(r"      r r                % logN=14: K1, K2")
    lines.append(r"    @{}}")
    lines.append(r"    \toprule")
    lines.append(r"    & \multicolumn{2}{c}{$\log N_{\rm ckks} = 13$}")
    lines.append(r"    & \multicolumn{2}{c}{$\log N_{\rm ckks} = 14$} \\")
    lines.append(r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    lines.append(
        f"    $\\ell$ & $K={K1_13}$ & $K={K2_13}$"
        f" & $K={K1_14}$ & $K={K2_14}$ \\\\")
    lines.append(r"    \midrule")

    for chebO in CHEB_ORDERS:
        lines.append(
            f"    {chebO} & {cell(13, K1_13, chebO)} & {cell(13, K2_13, chebO)}"
            f"\n      & {cell(14, K1_14, chebO)} & {cell(14, K2_14, chebO)} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular*}")
    lines.append(r"\end{table}")

    print("\n% ---- populated LaTeX table ----")
    print("\n".join(lines))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print grid and exit without running anything.")
    args = parser.parse_args()

    # Base config must already exist (written by main.ipynb config cell)
    cfg_path = ROOT / "output" / "ckks_config.json"
    if not cfg_path.exists():
        sys.exit("ERROR: output/ckks_config.json not found. "
                 "Run the config cell in main.ipynb first.")

    base_cfg = json.loads(cfg_path.read_text())
    grid     = build_grid(base_cfg)

    print(f"\nAblation grid: {len(grid)} runs")
    print(f"  {'tag':<35} logN     K  nW  cheb")
    for tag, cfg in grid:
        print(f"  {tag:<35}  {cfg['logN']}  {cfg['K']:4d}"
              f"   {cfg['nWorkers']}     {cfg['chebOrder']}")

    if args.dry_run:
        print("\n[dry-run] Exiting.")
        sys.exit(0)

    ablation_dir = ROOT / "output" / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    summary_path = ablation_dir / "summary.csv"
    summary      = []

    for i, (tag, cfg) in enumerate(grid):
        dest = ablation_dir / tag
        # Resume: skip tags that already produced output
        if dest.exists() and (dest / "xs_ckks.csv").exists():
            saved   = json.loads((dest / "ckks_config.json").read_text())
            mean_ms, std_ms = load_cached_result(tag, summary_path)
            summary.append({
                "tag":            tag,
                "logN":           saved["logN"],
                "K":              saved["K"],
                "nWorkers":       saved["nWorkers"],
                "chebOrder":      saved["chebOrder"],
                "online_mean_ms": mean_ms,
                "online_std_ms":  std_ms,
                "status":         "cached",
            })
            print(f"[{i+1}/{len(grid)}] Skipping {tag} (cached)")
            continue

        print(f"\n[{i+1}/{len(grid)}] Starting {tag}")
        row = run_case(tag, cfg)
        summary.append(row)
        write_summary_csv(summary, summary_path)   # checkpoint after every run

    write_summary_csv(summary, summary_path)
    ok  = sum(1 for r in summary if r["status"] in ("ok", "cached"))
    err = sum(1 for r in summary if r["status"] == "error")
    print(f"\nDone. {ok} ok, {err} errors. Summary at {summary_path}")

    print_latex_table(summary, base_cfg)