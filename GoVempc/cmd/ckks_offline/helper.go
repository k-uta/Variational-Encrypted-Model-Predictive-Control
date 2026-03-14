package main

import (
	"sort"

	"gonum.org/v1/gonum/mat"
)

// collectRotations returns the rotation keys needed for offline packing.
func collectRotations(kChunk, dim, p, slots int) []int {
	if kChunk <= 1 {
		return nil
	}
	rotSet := make(map[int]struct{})
	for k := 1; k < kChunk; k++ {
		r1 := mod(-k*dim, slots)
		if r1 != 0 {
			rotSet[r1] = struct{}{}
		}
		r2 := mod(-k*p, slots)
		if r2 != 0 {
			rotSet[r2] = struct{}{}
		}
	}
	rots := make([]int, 0, len(rotSet))
	for r := range rotSet {
		rots = append(rots, r)
	}
	sort.Ints(rots)
	return rots
}

// mod returns a positive modulo result.
func mod(a, b int) int {
	r := a % b
	if r < 0 {
		r += b
	}
	return r
}

// maxInt returns the larger of two ints.
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// diagDense builds a diagonal dense matrix from a slice.
func diagDense(vals []float64) *mat.Dense {
	n := len(vals)
	out := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		out.Set(i, i, vals[i])
	}
	return out
}

// scaledIdentity builds value times the identity matrix.
func scaledIdentity(n int, value float64) *mat.Dense {
	out := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		out.Set(i, i, value)
	}
	return out
}

// stackIdentity returns the stacked box-constraint matrix [I; -I].
func stackIdentity(n int) *mat.Dense {
	out := mat.NewDense(2*n, n, nil)
	for i := 0; i < n; i++ {
		out.Set(i, i, 1.0)
		out.Set(i+n, i, -1.0)
	}
	return out
}
