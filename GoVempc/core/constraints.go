package core

import "gonum.org/v1/gonum/mat"

type ConstraintPenalty struct {
	// G U <= h(x0) defines stacked linear inequality constraints.
	G              *mat.Dense
	// HFunc computes h(x0) for the current state.
	HFunc          func([]float64) []float64
	// Mode controls how feasibility is turned into weights (currently indicator only).
	Mode           string
	HasConstraints bool
	// P is the number of scalar constraints.
	P              int
}

func NewConstraintPenalty(G *mat.Dense, hFunc func([]float64) []float64, mode string) *ConstraintPenalty {
	p := 0
	r, _ := G.Dims()
	p = r

	return &ConstraintPenalty{
		G:              G,
		HFunc:          hFunc,
		Mode:           mode,
		HasConstraints: true,
		P:              p,
	}
}

func (c *ConstraintPenalty) ConstraintResidualMat(U *mat.Dense, x0 []float64) *mat.Dense {
	// Batch form: residuals for each row U_i.
	hVal := c.HFunc(x0)
	var res mat.Dense
	res.Mul(U, c.G.T())
	r, p := res.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < p; j++ {
			res.Set(i, j, res.At(i, j)-hVal[j])
		}
	}
	return &res
}

func (c *ConstraintPenalty) IsFeasibleMat(U *mat.Dense, x0 []float64, tol float64) []bool {
	// Row-wise feasibility check for batch samples.
	res := c.ConstraintResidualMat(U, x0)
	r, p := res.Dims()
	out := make([]bool, r)
	for i := 0; i < r; i++ {
		feasible := true
		for j := 0; j < p; j++ {
			if res.At(i, j) > tol {
				feasible = false
				break
			}
		}
		out[i] = feasible
	}
	return out
}
