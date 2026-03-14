package examples

import "gonum.org/v1/gonum/mat"

func LinearizedInvertedPendulumContinuous(m, l, g float64) (*mat.Dense, *mat.Dense) {
	// Linearized inverted pendulum around upright equilibrium (small angle).
	// State: [theta, theta_dot], input: torque at the pivot.
	Ac := mat.NewDense(2, 2, nil)
	Bc := mat.NewDense(2, 1, nil)

	// theta_dot = omega
	Ac.Set(0, 1, 1.0)
	// omega_dot = (g/l) * theta + (1/(m*l^2)) * u
	Ac.Set(1, 0, g/l)
	Bc.Set(1, 0, 1.0/(m*l*l))

	return Ac, Bc
}

func Discretize(Ac, Bc *mat.Dense, dt float64) (*mat.Dense, *mat.Dense) {
	// Zero-order hold discretization using matrix exponential of the
	// augmented continuous-time system.
	n, _ := Ac.Dims()
	_, m := Bc.Dims()

	aug := mat.NewDense(n+m, n+m, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			aug.Set(i, j, Ac.At(i, j))
		}
		for j := 0; j < m; j++ {
			aug.Set(i, n+j, Bc.At(i, j))
		}
	}

	var augScaled mat.Dense
	augScaled.Scale(dt, aug)
	var exp mat.Dense
	exp.Exp(&augScaled)

	Ad := mat.NewDense(n, n, nil)
	Bd := mat.NewDense(n, m, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			Ad.Set(i, j, exp.At(i, j))
		}
		for j := 0; j < m; j++ {
			Bd.Set(i, j, exp.At(i, n+j))
		}
	}
	return Ad, Bd
}
