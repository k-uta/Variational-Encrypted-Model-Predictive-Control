package solvers

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// ChebyshevReLUCoeffs fits a Chebyshev series approximation of ReLU on [-bound, bound].
func ChebyshevReLUCoeffs(order int, bound float64, nSamples int) []float64 {
	if nSamples <= 0 {
		nSamples = int(math.Max(200, float64(10*order)))
	}
	if order < 0 {
		order = 0
	}

	tNodes := make([]float64, nSamples)
	y := make([]float64, nSamples)
	for k := 0; k < nSamples; k++ {
		t := math.Cos(math.Pi * (2.0*float64(k) + 1.0) / (2.0 * float64(nSamples)))
		tNodes[k] = t
		if t > 0 {
			y[k] = bound * t
		}
	}

	T := mat.NewDense(nSamples, order+1, nil)
	for i := 0; i < nSamples; i++ {
		vals := chebVals(tNodes[i], order)
		for j := 0; j <= order; j++ {
			T.Set(i, j, vals[j])
		}
	}

	var TT mat.Dense
	TT.Mul(T.T(), T)
	rhs := mat.NewVecDense(nSamples, y)
	var TTy mat.VecDense
	TTy.MulVec(T.T(), rhs)

	coeff := mat.NewVecDense(order+1, nil)
	_ = coeff.SolveVec(&TT, &TTy)
	out := make([]float64, order+1)
	copy(out, coeff.RawVector().Data)
	return out
}

func chebVals(t float64, order int) []float64 {
	vals := make([]float64, order+1)
	vals[0] = 1.0
	if order == 0 {
		return vals
	}
	vals[1] = t
	for k := 2; k <= order; k++ {
		vals[k] = 2.0*t*vals[k-1] - vals[k-2]
	}
	return vals
}
