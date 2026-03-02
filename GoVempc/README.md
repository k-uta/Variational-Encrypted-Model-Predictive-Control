# GoVempc

Go translation of the `vempc` Python code plus a Go implementation of `test.ipynb`.

## Layout
- `core`: condensed MPC, constraints, variational MPC
- `solvers`: QP surrogate + sampling utilities
- `examples`: pendulum/cartpole model
- `cmd/test`: Go version of `test.ipynb`

## Running the notebook port
From `GoVempc`:

```bash
go run ./cmd/test
```

Outputs are written to `GoVempc/output` as CSV files.

## Notes on the standard QP solver
The Go version uses a smooth penalty formulation to approximate the
inequality-constrained QP. This keeps the implementation self-contained
while preserving the structure of the original experiment.
