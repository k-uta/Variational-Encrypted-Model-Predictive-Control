package core

import "gonum.org/v1/gonum/mat"

func Identity(n int) *mat.Dense {
	// Dense identity matrix I_n.
	data := make([]float64, n*n)
	for i := 0; i < n; i++ {
		data[i*n+i] = 1.0
	}
	return mat.NewDense(n, n, data)
}

func BlockDiagRepeat(block *mat.Dense, repeats int) *mat.Dense {
	// Build a block-diagonal matrix with the same block repeated on the diagonal.
	r, c := block.Dims()
	out := mat.NewDense(repeats*r, repeats*c, nil)
	for k := 0; k < repeats; k++ {
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				out.Set(k*r+i, k*c+j, block.At(i, j))
			}
		}
	}
	return out
}

func VStack(a, b *mat.Dense) *mat.Dense {
	// Stack two matrices vertically (same column count required).
	ra, ca := a.Dims()
	rb, cb := b.Dims()
	out := mat.NewDense(ra+rb, ca, nil)
	for i := 0; i < ra; i++ {
		for j := 0; j < ca; j++ {
			out.Set(i, j, a.At(i, j))
		}
	}
	for i := 0; i < rb; i++ {
		for j := 0; j < cb; j++ {
			out.Set(ra+i, j, b.At(i, j))
		}
	}
	return out
}

func TileVector(v []float64, repeats int) []float64 {
	// Repeat a vector v, repeats times, into a single long vector.
	out := make([]float64, len(v)*repeats)
	for k := 0; k < repeats; k++ {
		copy(out[k*len(v):(k+1)*len(v)], v)
	}
	return out
}

func MatVecMul(a mat.Matrix, x []float64) []float64 {
	// Compute a * x and return a standalone slice.
	var y mat.VecDense
	y.MulVec(a, mat.NewVecDense(len(x), x))
	out := make([]float64, y.Len())
	copy(out, y.RawVector().Data)
	return out
}

func VecAdd(a, b []float64) []float64 {
	// Elementwise addition.
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

func VecSub(a, b []float64) []float64 {
	// Elementwise subtraction.
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] - b[i]
	}
	return out
}

func VecScale(a []float64, s float64) []float64 {
	// Scalar-vector scaling.
	out := make([]float64, len(a))
	for i := range a {
		out[i] = s * a[i]
	}
	return out
}

func Dot(a, b []float64) float64 {
	// Dot product a^T b.
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
