package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"gonum.org/v1/gonum/mat"

	"govempc/core"
)

// computeMU returns the tilted Gaussian mean m_U(x).
func computeMU(v *core.VariationalMPC, x0 []float64) []float64 {
	Stx := core.MatVecMul(v.MPC.S.T(), x0)
	SigmaStx := core.MatVecMul(v.SigmaU, Stx)
	return core.VecScale(SigmaStx, -(1.0 / v.LambdaParam))
}

// computeB returns the affine term b(x) = G m_U(x) - h(x).
func computeB(p *core.ConstraintPenalty, mU []float64, x0 []float64) []float64 {
	h := p.HFunc(x0)
	GmU := core.MatVecMul(p.G, mU)
	return core.VecSub(GmU, h)
}

// collectOnlineRotations returns the full rotation-key set needed online.
func collectOnlineRotations(kChunk, dim, p, slots int) []int {
	rotSet := make(map[int]struct{})
	for _, r := range collectPackRotations(kChunk, dim, p, slots) {
		rotSet[r] = struct{}{}
	}
	for r := 1; r < p; r++ {
		rotSet[r] = struct{}{}
	}
	rots := make([]int, 0, len(rotSet))
	for r := range rotSet {
		rots = append(rots, r)
	}
	return rots
}

// collectPackRotations returns the rotations used for packed sample placement.
func collectPackRotations(kChunk, dim, p, slots int) []int {
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
	return rots
}

// maxInt returns the larger of two ints.
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// mod returns a positive modulo result.
func mod(a, b int) int {
	r := a % b
	if r < 0 {
		r += b
	}
	return r
}

// meanStd computes the mean and standard deviation of a slice.
func meanStd(vals []float64) (float64, float64) {
	if len(vals) == 0 {
		return 0, 0
	}
	var sum float64
	var sumSq float64
	for _, v := range vals {
		sum += v
		sumSq += v * v
	}
	mean := sum / float64(len(vals))
	variance := (sumSq / float64(len(vals))) - mean*mean
	if variance < 0 {
		variance = 0
	}
	return mean, math.Sqrt(variance)
}

// chebToPower converts Chebyshev coefficients to the power basis.
func chebToPower(cheb []float64) []float64 {
	n := len(cheb)
	if n == 0 {
		return nil
	}
	poly := make([]float64, n)

	tkm2 := []float64{1.0}
	addPolyScaled(poly, tkm2, cheb[0])
	if n == 1 {
		return poly
	}

	tkm1 := []float64{0.0, 1.0}
	addPolyScaled(poly, tkm1, cheb[1])

	for k := 2; k < n; k++ {
		tk := make([]float64, k+1)
		for i := 0; i < len(tkm1); i++ {
			tk[i+1] += 2.0 * tkm1[i]
		}
		for i := 0; i < len(tkm2); i++ {
			tk[i] -= tkm2[i]
		}
		addPolyScaled(poly, tk, cheb[k])
		tkm2 = tkm1
		tkm1 = tk
	}
	return poly
}

// addPolyScaled adds scale times src into dst.
func addPolyScaled(dst, src []float64, scale float64) {
	for i := 0; i < len(src); i++ {
		dst[i] += scale * src[i]
	}
}

// scalePoly rescales a polynomial for z-domain evaluation.
func scalePoly(poly []float64, bound float64) []float64 {
	out := make([]float64, len(poly))
	pow := 1.0
	for i := 0; i < len(poly); i++ {
		out[i] = poly[i] / pow
		pow *= bound
	}
	return out
}

// evalPolyReal evaluates a real polynomial with Horner's rule.
func evalPolyReal(coeffs []float64, x float64) float64 {
	y := 0.0
	for k := len(coeffs) - 1; k >= 0; k-- {
		y = y*x + coeffs[k]
	}
	return y
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

// writeMatCSV writes a dense matrix to CSV.
func writeMatCSV(path string, m *mat.Dense) {
	f, _ := os.Create(path)
	defer f.Close()

	w := csv.NewWriter(f)
	r, c := m.Dims()
	row := make([]string, c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			row[j] = fmt.Sprintf("%.10f", m.At(i, j))
		}
		_ = w.Write(row)
	}
	w.Flush()
	_ = w.Error()
}

// writeSeriesCSV writes a vector to a single-column CSV.
func writeSeriesCSV(path string, series []float64) {
	f, _ := os.Create(path)
	defer f.Close()

	w := csv.NewWriter(f)
	for _, v := range series {
		_ = w.Write([]string{fmt.Sprintf("%.10f", v)})
	}
	w.Flush()
	_ = w.Error()
}

// progressBar renders a simple tqdm-style terminal progress indicator.
type progressBar struct {
	label string
	total int
	width int
	start time.Time
	last  time.Time
	out   *os.File
}

// newProgressBar allocates a progress bar for a fixed number of steps.
func newProgressBar(label string, total int) *progressBar {
	return &progressBar{
		label: label,
		total: total,
		width: 28,
		start: time.Now(),
		out:   os.Stderr,
	}
}

// Update refreshes the bar with the current completed step count.
func (p *progressBar) Update(done int) {
	if p.total <= 0 {
		return
	}
	if done < 0 {
		done = 0
	}
	if done > p.total {
		done = p.total
	}

	now := time.Now()
	if done != p.total && !p.last.IsZero() && now.Sub(p.last) < 120*time.Millisecond {
		return
	}
	p.last = now

	pct := float64(done) / float64(p.total)
	filled := int(math.Round(pct * float64(p.width)))
	if filled < 0 {
		filled = 0
	}
	if filled > p.width {
		filled = p.width
	}

	bar := strings.Repeat("=", filled) + strings.Repeat("-", p.width-filled)
	eta := "--"
	elapsed := now.Sub(p.start).Seconds()
	if done > 0 && elapsed > 0 {
		rate := float64(done) / elapsed
		remaining := float64(p.total - done)
		eta = formatETASeconds(remaining / rate)
	}

	fmt.Fprintf(p.out, "\r%s [%s] %3.0f%% %d/%d eta %s", p.label, bar, pct*100, done, p.total, eta)
	if done == p.total {
		fmt.Fprint(p.out, "\n")
	}
}

// formatETASeconds formats a rough ETA for the progress bar.
func formatETASeconds(sec float64) string {
	if math.IsInf(sec, 0) || math.IsNaN(sec) {
		return "--"
	}
	d := time.Duration(sec * float64(time.Second))
	if d < 0 {
		d = 0
	}
	h := d / time.Hour
	d -= h * time.Hour
	m := d / time.Minute
	d -= m * time.Minute
	s := d / time.Second
	if h > 0 {
		return fmt.Sprintf("%02d:%02d:%02d", h, m, s)
	}
	return fmt.Sprintf("%02d:%02d", m, s)
}
