package core

import (
	"math"
	"time"

	"gonum.org/v1/gonum/mat"
)

type MPCProblem struct {
	// Discrete-time linear dynamics x_{k+1} = A x_k + B u_k.
	A, B *mat.Dense
	// Quadratic costs for state and input.
	Q, R *mat.Dense
	// Terminal state cost.
	Qf *mat.Dense
	// Prediction horizon length.
	N int

	n  int
	m  int
	Nm int

	// Condensed prediction and cost matrices.
	Lambda *mat.Dense
	Psi    *mat.Dense
	P      *mat.Dense
	S      *mat.Dense
	H      *mat.Dense
}

func NewMPCProblem(A, B, Q, R, Qf *mat.Dense, N int) *MPCProblem {
	// Build condensed MPC representation for a linear-quadratic setup.
	mpc := &MPCProblem{
		A:  A,
		B:  B,
		Q:  Q,
		R:  R,
		Qf: Qf,
		N:  N,
	}
	n, _ := A.Dims()
	_, m := B.Dims()
	mpc.n = n
	mpc.m = m
	mpc.Nm = N * m
	mpc.Lambda, mpc.Psi = mpc.computePredictionMatrices()
	mpc.P, mpc.S, mpc.H = mpc.computeCostMatrices()
	return mpc
}

func (m *MPCProblem) StateDim() int {
	return m.n
}

func (m *MPCProblem) InputDim() int {
	return m.m
}

func (m *MPCProblem) StackedInputDim() int {
	return m.Nm
}

func (m *MPCProblem) computePredictionMatrices() (*mat.Dense, *mat.Dense) {
	n, mIn, N := m.n, m.m, m.N
	A := m.A
	B := m.B

	// Precompute A^k to build Lambda and Psi efficiently.
	A_powers := make([]*mat.Dense, N+1)
	A_powers[0] = Identity(n)
	for k := 1; k <= N; k++ {
		ap := mat.NewDense(n, n, nil)
		ap.Mul(A_powers[k-1], A)
		A_powers[k] = ap
	}

	Lambda := mat.NewDense(N*n, n, nil)
	for i := 0; i < N; i++ {
		copyBlock(Lambda, i*n, 0, A_powers[i+1])
	}

	// Psi maps stacked inputs U to stacked states X.
	Psi := mat.NewDense(N*n, N*mIn, nil)
	for i := 0; i < N; i++ {
		row := i * n
		for j := 0; j <= i; j++ {
			col := j * mIn
			block := mat.NewDense(n, mIn, nil)
			block.Mul(A_powers[i-j], B)
			copyBlock(Psi, row, col, block)
		}
	}

	return Lambda, Psi
}

func (m *MPCProblem) computeCostMatrices() (*mat.Dense, *mat.Dense, *mat.Dense) {
	n, N := m.n, m.N

	// Q_bar = diag(Q,...,Q,Qf), R_bar = diag(R,...,R).
	Q_bar := BlockDiagRepeat(m.Q, N)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			Q_bar.Set((N-1)*n+i, (N-1)*n+j, m.Qf.At(i, j))
		}
	}
	R_bar := BlockDiagRepeat(m.R, N)

	var tmp mat.Dense
	tmp.Mul(Q_bar, m.Lambda)
	var P mat.Dense
	P.Mul(m.Lambda.T(), &tmp)
	P.Add(&P, m.Q)

	var tmp2 mat.Dense
	tmp2.Mul(Q_bar, m.Psi)
	var S mat.Dense
	// S = 2 * Lambda^T * Q_bar * Psi.
	S.Mul(m.Lambda.T(), &tmp2)
	S.Scale(2.0, &S)

	var tmp3 mat.Dense
	tmp3.Mul(m.Psi.T(), &tmp2)
	var H mat.Dense
	// H = 2 * (R_bar + Psi^T * Q_bar * Psi).
	H.Add(&tmp3, R_bar)
	H.Scale(2.0, &H)

	return &P, &S, &H
}

func (m *MPCProblem) QuadraticCost(x0 []float64, U []float64) float64 {
	// J_0(x0, U) = x0^T P x0 + x0^T S U + 0.5 U^T H U.
	Px0 := MatVecMul(m.P, x0)
	x0Px0 := Dot(x0, Px0)

	SU := MatVecMul(m.S, U)
	x0SU := Dot(x0, SU)

	HU := MatVecMul(m.H, U)
	UTHU := Dot(U, HU)

	return x0Px0 + x0SU + 0.5*UTHU
}

func (m *MPCProblem) BuildConstraintMatrices(Gx *mat.Dense, hx []float64, Gu *mat.Dense, hu []float64) (*mat.Dense, func([]float64) []float64) {
	N := m.N

	// Stack per-step state and input constraints across the horizon.
	GxBar := BlockDiagRepeat(Gx, N)
	hxBar := TileVector(hx, N)

	GuBar := BlockDiagRepeat(Gu, N)
	huBar := TileVector(hu, N)

	var GTop mat.Dense
	GTop.Mul(GxBar, m.Psi)

	var LTop mat.Dense
	LTop.Mul(GxBar, m.Lambda)

	GBot := GuBar
	HBot := huBar

	G := VStack(&GTop, GBot)

	hFunc := func(x0 []float64) []float64 {
		// h(x0) = [hx_bar - (Gx_bar * Lambda) x0; hu_bar].
		Lx := MatVecMul(&LTop, x0)
		hTop := VecSub(hxBar, Lx)
		out := make([]float64, len(hTop)+len(HBot))
		copy(out, hTop)
		copy(out[len(hTop):], HBot)
		return out
	}

	return G, hFunc
}

func copyBlock(dst *mat.Dense, rowOffset, colOffset int, src mat.Matrix) {
	r, c := src.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			dst.Set(rowOffset+i, colOffset+j, src.At(i, j))
		}
	}
}

type ControllerFunc func(x []float64, warm []float64) (u []float64, Useq []float64, info map[string]float64)

func Simulate(x0 []float64, controller ControllerFunc, A, B *mat.Dense, steps int) (*mat.Dense, *mat.Dense, []map[string]float64, time.Duration) {
	// Roll out the closed-loop system with a provided controller.
	n, _ := A.Dims()
	_, mIn := B.Dims()
	xs := mat.NewDense(steps+1, n, nil)
	us := mat.NewDense(steps, mIn, nil)
	infoLog := make([]map[string]float64, 0, steps)
	var warm []float64

	start := time.Now()
	x := make([]float64, n)
	copy(x, x0)
	for j := 0; j < n; j++ {
		xs.Set(0, j, x[j])
	}

	for k := 0; k < steps; k++ {
		// Controller can optionally return a warm-start sequence and info map.
		u, Useq, info := controller(x, warm)
		if len(u) < mIn {
			fallback := make([]float64, mIn)
			copy(fallback, u)
			u = fallback
		}
		if Useq != nil {
			// Save warm start for the next step.
			warm = make([]float64, len(Useq))
			copy(warm, Useq)
		}
		if info == nil {
			info = map[string]float64{}
		}
		infoLog = append(infoLog, info)

		for j := 0; j < mIn; j++ {
			us.Set(k, j, u[j])
		}

		// State propagation.
		Ax := MatVecMul(A, x)
		Bu := MatVecMul(B, u)
		for i := 0; i < n; i++ {
			x[i] = Ax[i] + Bu[i]
			xs.Set(k+1, i, x[i])
		}
	}

	elapsed := time.Since(start)
	return xs, us, infoLog, elapsed
}

func TrajectoryCost(xs, us *mat.Dense, Q, R, Qf *mat.Dense) float64 {
	// Sum of stage costs plus terminal cost for a realized trajectory.
	rus, _ := us.Dims()
	steps := rus

	cost := 0.0
	for k := 0; k < steps; k++ {
		xk := rowSlice(xs, k)
		uk := rowSlice(us, k)
		cost += Dot(xk, MatVecMul(Q, xk))
		cost += Dot(uk, MatVecMul(R, uk))
	}
	xN := rowSlice(xs, steps)
	cost += Dot(xN, MatVecMul(Qf, xN))
	return cost
}

func MaxConstraintViolation(xs, us *mat.Dense, Gx *mat.Dense, hx []float64, Gu *mat.Dense, hu []float64) (float64, float64) {
	// Max positive violations for state and input constraints.
	vx := 0.0
	rxs, _ := xs.Dims()
	if rxs > 1 {
		for k := 1; k < rxs; k++ {
			xk := rowSlice(xs, k)
			res := MatVecMul(Gx, xk)
			for i := range res {
				res[i] -= hx[i]
				if res[i] > vx {
					vx = res[i]
				}
			}
		}
	}

	vu := 0.0
	rus, _ := us.Dims()
	if rus > 0 {
		for k := 0; k < rus; k++ {
			uk := rowSlice(us, k)
			res := MatVecMul(Gu, uk)
			for i := range res {
				res[i] -= hu[i]
				if res[i] > vu {
					vu = res[i]
				}
			}
		}
	}

	if vx < 0.0 {
		vx = 0.0
	}
	if vu < 0.0 {
		vu = 0.0
	}
	return vx, vu
}

func AvgAcceptance(info []map[string]float64) float64 {
	// Average acceptance rate from controller info (if provided).
	sum := 0.0
	count := 0
	for _, d := range info {
		if v, ok := d["accept_rate"]; ok {
			if !math.IsNaN(v) {
				sum += v
				count++
			}
		}
	}
	if count == 0 {
		return math.NaN()
	}
	return sum / float64(count)
}

func rowSlice(m *mat.Dense, row int) []float64 {
	_, c := m.Dims()
	out := make([]float64, c)
	for j := 0; j < c; j++ {
		out[j] = m.At(row, j)
	}
	return out
}
