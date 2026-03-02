package solvers

import (
	"gonum.org/v1/gonum/mat"

	"govempc/core"
)

// Is this necessary?
func SampleTilted(variational *core.VariationalMPC, x0 []float64, K int, seed int64) *mat.Dense {
	// Draw K samples from the tilted Gaussian kappa_tilde.
	return variational.SampleKappaTilde(x0, K, seed)
}

func SampleVariationalControl(
	x0 []float64,
	variational *core.VariationalMPC,
	penalty *core.ConstraintPenalty,
	K int,
	chebCoeffs []float64,
	chebBound float64,
	chebEta float64,
	chebClip bool,
	seed int64,
) ([]float64, []float64, float64, int) {

	// Monte Carlo estimator with polynomial surrogate feasibility weights.
	// Each row represents U^{(i)}
	U := SampleTilted(variational, x0, K, seed)

	// Exact feasibility count (used only for diagnostics).
	feasible := penalty.IsFeasibleMat(U, x0, 1e-6)
	trueAccept := 0
	for _, f := range feasible {
		if f {
			trueAccept++
		}
	}

	weights, wSum := variational.ComputeWeights(U, x0, core.WeightOptions{
		ChebCoeffs: chebCoeffs,
		ChebBound:  chebBound,
		ChebEta:    chebEta,
		ChebClip:   chebClip,
		Eps:        1e-12,
	})

	_, Nm := U.Dims()
	UHat := make([]float64, Nm)
	if wSum == 0 {
		// No effective weight: return zeros to avoid NaNs.
		mIn := variational.MPC.InputDim()
		u0 := make([]float64, mIn)
		return u0, UHat, 0.0, trueAccept
	}
	for i := 0; i < K; i++ {
		for j := 0; j < Nm; j++ {
			UHat[j] += weights[i] * U.At(i, j)
		}
	}

	mIn := variational.MPC.InputDim()
	u0 := append([]float64(nil), UHat[:mIn]...)
	return u0, UHat, wSum, trueAccept
}
