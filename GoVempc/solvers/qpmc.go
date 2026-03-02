package solvers

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
)

type SolverSettings struct {
	MaxIter int
	Ftol    float64
	Gtol    float64
}

type StandardQPSolver struct {
	// H is the quadratic term, G is the stacked constraint matrix.
	// This implementation uses a soft-penalty (not a hard QP solve).
	H        *mat.Dense
	G        *mat.Dense
	Rho      float64
	Settings SolverSettings
}

func NewStandardQPSolver(H, G *mat.Dense) *StandardQPSolver {
	return &StandardQPSolver{
		H:   H,
		G:   G,
		Rho: 1e4,
		Settings: SolverSettings{
			MaxIter: 500,
			Ftol:    1e-9,
			Gtol:    1e-6,
		},
	}
}

func SolveStandardMPC(
	x0 []float64,
	S *mat.Dense,
	hOfX0 func([]float64) []float64,
	mIn int,
	solver *StandardQPSolver,
	warmStart []float64,
) ([]float64, []float64, error) {
	// q = S^T x0, h = h(x0).
	q := matVecMul(S.T(), x0)
	h := hOfX0(x0)

	objective := func(U []float64) float64 {
		// Penalized objective: 0.5 U^T H U + q^T U + rho * ||[G U - h]_+||^2.
		cost := 0.5*dot(U, matVecMul(solver.H, U)) + dot(q, U)
		if solver.G == nil {
			return cost
		}
		g := matVecMul(solver.G, U)
		var penalty float64
		for i := range g {
			g[i] -= h[i]
			if g[i] > 0 {
				penalty += g[i] * g[i]
			}
		}
		return cost + solver.Rho*penalty
	}

	gradient := func(grad, U []float64) {
		// Gradient of the penalized objective.
		HU := matVecMul(solver.H, U)
		for i := range grad {
			grad[i] = HU[i] + q[i]
		}
		if solver.G == nil {
			return
		}
		g := matVecMul(solver.G, U)
		v := make([]float64, len(g))
		for i := range g {
			g[i] -= h[i]
			if g[i] > 0 {
				v[i] = 2.0 * g[i]
			}
		}
		gtv := matVecMul(solver.G.T(), v)
		for i := range grad {
			grad[i] += solver.Rho * gtv[i]
		}
	}

	var xInit []float64
	if warmStart != nil {
		// Use the previous U sequence for warm-starting.
		xInit = append([]float64(nil), warmStart...)
	} else {
		// Unconstrained minimizer provides a stable starting point.
		xInit = make([]float64, len(q))
		if err := solveSymmetric(solver.H, q, xInit); err != nil {
			for i := range xInit {
				xInit[i] = 0
			}
		}
		for i := range xInit {
			xInit[i] = -xInit[i]
		}
	}

	Ustar, err := minimizeLBFGS(xInit, objective, gradient, solver.Settings)
	if err != nil {
		// Fall back to the initial guess if optimization fails.
		Ustar = append([]float64(nil), xInit...)
	}
	u0 := append([]float64(nil), Ustar[:mIn]...)
	return u0, Ustar, nil
}

func minimizeLBFGS(x0 []float64, f func([]float64) float64, grad func([]float64, []float64), settings SolverSettings) ([]float64, error) {
	// Thin wrapper around gonum/optimize L-BFGS.
	problem := optimize.Problem{
		Func: f,
		Grad: grad,
	}
	set := optimize.Settings{
		GradientThreshold: settings.Gtol,
		FuncEvaluations:   settings.MaxIter,
		MajorIterations:   settings.MaxIter,
	}
	res, _ := optimize.Minimize(problem, x0, &set, &optimize.LBFGS{})
	return res.X, nil
}

func matVecMul(a mat.Matrix, x []float64) []float64 {
	// Local helper to avoid importing core/linalg.
	var y mat.VecDense
	y.MulVec(a, mat.NewVecDense(len(x), x))
	out := make([]float64, y.Len())
	copy(out, y.RawVector().Data)
	return out
}

func dot(a, b []float64) float64 {
	// Local dot product to keep solver self-contained.
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func solveSymmetric(H *mat.Dense, b []float64, dst []float64) error {
	// Solve H x = b for symmetric positive definite H via Cholesky.
	n, _ := H.Dims()
	sym := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			sym.SetSym(i, j, H.At(i, j))
		}
	}
	var chol mat.Cholesky
	_ = chol.Factorize(sym)
	bVec := mat.NewVecDense(len(b), b)
	var x mat.VecDense
	_ = chol.SolveVecTo(&x, bVec)
	copy(dst, x.RawVector().Data)
	return nil
}
