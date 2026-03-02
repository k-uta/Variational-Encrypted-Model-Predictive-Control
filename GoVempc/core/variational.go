package core

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type VariationalMPC struct {
	// Condensed MPC problem (prediction and cost matrices).
	MPC *MPCProblem
	// Constraint handler used for feasibility weighting.
	Penalty *ConstraintPenalty
	// Temperature parameter lambda.
	LambdaParam float64
	// Prior covariance used to build the tilted covariance Sigma_U.
	Sigma0 *mat.Dense
	// Tilted covariance Sigma_U and its Cholesky factor.
	SigmaU *mat.Dense
	LU     *mat.Dense
}

func NewVariationalMPC(mpc *MPCProblem, penalty *ConstraintPenalty, lambdaParam float64, sigma0 *mat.Dense) *VariationalMPC {
	SigmaU := computeSigmaU(sigma0, mpc.H, lambdaParam)
	LU := choleskyLower(SigmaU)

	return &VariationalMPC{
		MPC:         mpc,
		Penalty:     penalty,
		LambdaParam: lambdaParam,
		Sigma0:      sigma0,
		SigmaU:      SigmaU,
		LU:          LU,
	}
}

func computeSigmaU(Sigma0, H *mat.Dense, lambda float64) *mat.Dense {
	// Sigma_U = (Sigma_0^{-1} + (1/lambda) H)^{-1}.
	var Sigma0Inv mat.Dense
	_ = Sigma0Inv.Inverse(Sigma0)
	var scaledH mat.Dense
	scaledH.Scale(1.0/lambda, H)
	var tmp mat.Dense
	tmp.Add(&Sigma0Inv, &scaledH)
	var SigmaU mat.Dense
	_ = SigmaU.Inverse(&tmp)
	return &SigmaU
}

func choleskyLower(A *mat.Dense) *mat.Dense {
	// Compute the lower-triangular Cholesky factor of A.
	n, _ := A.Dims()
	sym := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			sym.SetSym(i, j, A.At(i, j))
		}
	}
	var chol mat.Cholesky
	_ = chol.Factorize(sym)
	var L mat.TriDense
	chol.LTo(&L)
	rows, cols := L.Dims()
	out := mat.NewDense(rows, cols, nil)
	out.Copy(&L)
	return out
}

func (v *VariationalMPC) mU(x0 []float64) []float64 {
	// m_U(x0) = -(1/lambda) * Sigma_U * S^T * x0.
	Stx := MatVecMul(v.MPC.S.T(), x0)
	SigmaStx := MatVecMul(v.SigmaU, Stx)
	return VecScale(SigmaStx, -(1.0 / v.LambdaParam))
}

func (v *VariationalMPC) SampleKappaTilde(x0 []float64, K int, seed int64) *mat.Dense {
	// Sample from kappa_tilde = N(m_U(x0), Sigma_U).
	rng := rand.New(rand.NewSource(seed))
	Nm := v.MPC.Nm
	mean := v.mU(x0)
	// Each row represents U^{(i)}
	xi := mat.NewDense(K, Nm, nil)
	for i := 0; i < K; i++ {
		for j := 0; j < Nm; j++ {
			xi.Set(i, j, rng.NormFloat64())
		}
	}
	var noise mat.Dense
	noise.Mul(xi, v.LU.T())
	samples := mat.NewDense(K, Nm, nil)
	for i := 0; i < K; i++ {
		for j := 0; j < Nm; j++ {
			samples.Set(i, j, mean[j]+noise.At(i, j))
		}
	}
	return samples
}

type WeightOptions struct {
	ChebCoeffs []float64
	ChebBound  float64
	ChebClip   bool
	ChebEta    float64
	Eps        float64
}

func (v *VariationalMPC) ComputeWeights(U *mat.Dense, x0 []float64, opts WeightOptions) ([]float64, float64) {
	K, _ := U.Dims()
	if opts.Eps == 0 {
		opts.Eps = 1e-12
	}

	// We work in log-space for numerical stability, initializing as -Inf
	// That is, we are essentially computing
	//                               log(r_s(U;x_0)) = -eta \sum h_l(g_j(U;x_0))
	// so infeasible samples naturally stay at weight 0 after exponentiation.
	logWeights := make([]float64, K)
	for i := range logWeights {
		logWeights[i] = math.Inf(-1)
	}

	// Samples drawn from the tilted distribution already include the cost term,
	// so we only apply feasibility weighting.

	// Compute g(U;x_0) for all samples at once
	// Each column of [residuals] is g(U^{(i)};x_0)
	residuals := v.Penalty.ConstraintResidualMat(U, x0)

	h := chebReLU(residuals, opts.ChebCoeffs, opts.ChebBound, opts.ChebClip)

	// threshold = h_l(0). Use h_l(gamma) if h_l(gamma) > threshold, else 0.
	threshold := chebVal(0.0, opts.ChebCoeffs)

	eta := opts.ChebEta
	if eta == 0 {
		eta = 1.0
	}
	for i := 0; i < K; i++ {
		// Sum only entries above the threshold.
		var s float64
		_, p := h.Dims()
		for j := 0; j < p; j++ {
			val := h.At(i, j)
			if val > threshold {
				s += val
			}
		}
		logWeights[i] = -eta * s
	}

	// Normalize in log-space: subtract the maximum log-weight to prevent overflow.
	maxLog := math.Inf(-1)
	for _, v := range logWeights {
		if math.IsInf(v, -1) {
			continue
		}
		if v > maxLog {
			maxLog = v
		}
	}
	if math.IsInf(maxLog, -1) {
		maxLog = 0.0
	}

	weights := make([]float64, K)
	var sum float64
	for i, v := range logWeights {
		if math.IsInf(v, -1) {
			weights[i] = 0
			continue
		}
		w := math.Exp(v - maxLog)
		weights[i] = w
		sum += w
	}
	if sum > 0 {
		for i := range weights {
			weights[i] /= sum
		}
	}
	return weights, sum
}

// chebReLU evaluates the Chebyshev approximation of ReLU on each residual:
// h_l(gamma) ≈ max(0, gamma) using coeffs fitted on t in [-1,1] for B*[t]_+.
func chebReLU(residuals *mat.Dense, coeffs []float64, bound float64, clip bool) *mat.Dense {
	// Map residuals into [-1,1], evaluate Chebyshev series, optionally clip negatives.
	r, c := residuals.Dims()
	out := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			t := residuals.At(i, j) / bound
			if clip {
				if t < -1.0 {
					t = -1.0
				}
				if t > 1.0 {
					t = 1.0
				}
			}
			y := chebVal(t, coeffs)
			if clip && y < 0.0 {
				y = 0.0
			}
			out.Set(i, j, y)
		}
	}
	return out
}

func sumRow(m *mat.Dense, row int) float64 {
	_, c := m.Dims()
	sum := 0.0
	for j := 0; j < c; j++ {
		sum += m.At(row, j)
	}
	return sum
}

func chebVal(t float64, coeffs []float64) float64 {
	n := len(coeffs)
	if n == 0 {
		return 0
	}
	if n == 1 {
		return coeffs[0]
	}
	b0 := 0.0
	b1 := 0.0
	b2 := 0.0
	for i := n - 1; i >= 1; i-- {
		b2 = b1
		b1 = b0
		b0 = 2*t*b1 - b2 + coeffs[i]
	}
	return t*b0 - b1 + coeffs[0]
}
