package main

import (
	"encoding/csv"
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"

	"govempc/core"
	"govempc/examples"
	"govempc/solvers"
)

func main() {
	rand.Seed(0)

	// System Define
	M, m, l, g := 1.0, 0.1, 0.5, 9.81
	dt := 0.01

	Ac, Bc := examples.LinearizedCartpoleContinuous(M, m, l, g)
	A, B := examples.Discretize(Ac, Bc, dt)

	n, _ := A.Dims()
	_, mIn := B.Dims()

	N := 20
	TSteps := 2000

	Q := diagDense([]float64{10.0, 1.0, 200.0, 5.0})
	R := diagDense([]float64{0.2})
	Qf := mat.NewDense(n, n, nil)
	Qf.Scale(5.0, Q)

	// Constraints
	xMax := 2.0
	vMax := 1.0
	thetaMax := 0.35
	omegaMax := 3.0
	uMax := 1.0

	sigma0 := 0.25
	KSamples := 1000
	lambdaPlot := 0.1
	chebOrder := 3
	chebBound := 20.0
	chebEta := 200.0
	chebClip := true

	xBound := []float64{xMax, vMax, thetaMax, omegaMax}

	Gx := stackIdentity(n) // +- identities
	hx := append(append([]float64{}, xBound...), xBound...)

	Gu := stackIdentity(mIn) // +- identities
	hu := []float64{uMax, uMax}

	// Define Lambda, Psi, P, S, H for compact MPC representation
	mpc := core.NewMPCProblem(A, B, Q, R, Qf, N)
	S, H := mpc.S, mpc.H

	// G is a matrix
	// hOfX0 is  a function that takes x_0 as input
	G, hOfX0 := mpc.BuildConstraintMatrices(Gx, hx, Gu, hu)
	penalty := core.NewConstraintPenalty(G, hOfX0, "indicator")

	// Solve using L-BFGS. Penalty method
	stdSolver := solvers.NewStandardQPSolver(H, G)

	solveStandard := func(x []float64, warm []float64) ([]float64, []float64, map[string]float64) {
		u, Useq, _ := solvers.SolveStandardMPC(x, S, hOfX0, mIn, stdSolver, warm)
		return u, Useq, nil
	}

	x0 := []float64{0.3, 0.0, 0.20, 0.0}

	xsStd, usStd, _, tStd := core.Simulate(x0, solveStandard, A, B, TSteps)

	stdCost := core.TrajectoryCost(xsStd, usStd, Q, R, Qf)
	stdVx, stdVu := core.MaxConstraintViolation(xsStd, usStd, Gx, hx, Gu, hu)

	fmt.Println("Standard MPC")
	fmt.Printf("  runtime: %s\n", tStd.Round(time.Millisecond))
	fmt.Printf("  cost:    %.6f\n", stdCost)
	fmt.Printf("  viol:    (x=%.6f, u=%.6f)\n", stdVx, stdVu)

	Sigma0 := scaledIdentity(mpc.StackedInputDim(), sigma0*sigma0)
	// Computing SigmaU and LU
	variational := core.NewVariationalMPC(mpc, penalty, lambdaPlot, Sigma0)
	// Pre-compute Chebyshev polynomial coefficients
	chebCoeffs := solvers.ChebyshevReLUCoeffs(chebOrder, chebBound, 0)

	ctrlVariational := func(x []float64, warm []float64) ([]float64, []float64, map[string]float64) {
		seed := rand.Int63()
		u, Useq, wSum, acceptNum := solvers.SampleVariationalControl(
			x,
			variational,
			penalty,
			KSamples,
			chebCoeffs,
			chebBound,
			chebEta,
			chebClip,
			seed,
		)
		info := map[string]float64{
			"w_sum":      wSum,
			"accept_num": float64(acceptNum), // Number of actual feasible samples
		}
		return u, Useq, info
	}

	xsVar, usVar, infoVar, tVar := core.Simulate(x0, ctrlVariational, A, B, TSteps)
	varCost := core.TrajectoryCost(xsVar, usVar, Q, R, Qf)
	varVx, varVu := core.MaxConstraintViolation(xsVar, usVar, Gx, hx, Gu, hu)

	wSumSeries, accSeries := extractSeries(infoVar)
	wSumMean := mean(wSumSeries)
	accMean := mean(accSeries)

	fmt.Println("Variational MPC")
	fmt.Printf("  runtime: %s\n", tVar.Round(time.Millisecond))
	fmt.Printf("  cost:    %.6f\n", varCost)
	fmt.Printf("  viol:    (x=%.6f, u=%.6f)\n", varVx, varVu)
	fmt.Printf("  w_sum mean: %.6f\n", wSumMean)
	fmt.Printf("  accept mean: %.6f\n", accMean)

	outDir := filepath.Join("output")
	_ = os.MkdirAll(outDir, 0o755)
	writeMatCSV(filepath.Join(outDir, "xs_standard.csv"), xsStd)
	writeMatCSV(filepath.Join(outDir, "us_standard.csv"), usStd)
	writeMatCSV(filepath.Join(outDir, "xs_variational.csv"), xsVar)
	writeMatCSV(filepath.Join(outDir, "us_variational.csv"), usVar)
	writeSeriesCSV(filepath.Join(outDir, "w_sum_series.csv"), wSumSeries)
	writeSeriesCSV(filepath.Join(outDir, "accept_series.csv"), accSeries)

	plotDir := filepath.Join(outDir, "plots")
	_ = os.MkdirAll(plotDir, 0o755)
	_ = plotMPC(xsStd, usStd, dt, xMax, vMax, thetaMax, omegaMax, uMax, "Standard MPC",
		filepath.Join(plotDir, "standard_mpc.png"))
	_ = plotMPC(xsVar, usVar, dt, xMax, vMax, thetaMax, omegaMax, uMax, "Variational MPC",
		filepath.Join(plotDir, "variational_mpc.png"))
	_ = plotVariationalStats(wSumSeries, accSeries, dt, TSteps, KSamples,
		filepath.Join(plotDir, "variational_stats.png"))

	fmt.Printf("\nSaved CSV outputs to %s\n", outDir)
	fmt.Printf("Saved plots to %s\n", plotDir)
}

func diagDense(vals []float64) *mat.Dense {
	n := len(vals)
	out := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		out.Set(i, i, vals[i])
	}
	return out
}

func stackIdentity(n int) *mat.Dense {
	out := mat.NewDense(2*n, n, nil)
	for i := 0; i < n; i++ {
		out.Set(i, i, 1.0)
		out.Set(n+i, i, -1.0)
	}
	return out
}

func scaledIdentity(n int, scale float64) *mat.Dense {
	out := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		out.Set(i, i, scale)
	}
	return out
}

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

func extractSeries(info []map[string]float64) ([]float64, []float64) {
	w := make([]float64, len(info))
	acc := make([]float64, len(info))
	for i, d := range info {
		w[i] = d["w_sum"]
		acc[i] = d["accept_num"]
	}
	return w, acc
}

func mean(v []float64) float64 {
	if len(v) == 0 {
		return math.NaN()
	}
	var sum float64
	for _, x := range v {
		sum += x
	}
	return sum / float64(len(v))
}

func plotMPC(xs, us *mat.Dense, dt float64, xMax, vMax, thetaMax, omegaMax, uMax float64, title, outPath string) error {
	t := make([]float64, xs.RawMatrix().Rows)
	for i := range t {
		t[i] = float64(i) * dt
	}
	tu := make([]float64, us.RawMatrix().Rows)
	for i := range tu {
		tu[i] = float64(i) * dt
	}

	p0, err := linePlot(t, colSlice(xs, 0), "time [s]", "p [m]")
	if err != nil {
		return err
	}
	p0.Title.Text = title
	addBounds(p0, t[0], t[len(t)-1], xMax)

	p1, err := linePlot(t, colSlice(xs, 1), "time [s]", "p_dot [m/s]")
	if err != nil {
		return err
	}
	addBounds(p1, t[0], t[len(t)-1], vMax)

	p2, err := linePlot(t, colSlice(xs, 2), "time [s]", "theta [rad]")
	if err != nil {
		return err
	}
	addBounds(p2, t[0], t[len(t)-1], thetaMax)

	p3, err := linePlot(t, colSlice(xs, 3), "time [s]", "theta_dot [rad/s]")
	if err != nil {
		return err
	}
	addBounds(p3, t[0], t[len(t)-1], omegaMax)

	p4, err := stepPlot(tu, colSlice(us, 0), "time [s]", "u [N]")
	if err != nil {
		return err
	}
	addBounds(p4, tu[0], tu[len(tu)-1], uMax)

	grid := [][]*plot.Plot{
		{p0, p1, p2},
		{p3, p4, nil},
	}
	return saveGrid(grid, vg.Points(900), vg.Points(520), outPath)
}

func plotVariationalStats(wSum, acc []float64, dt float64, steps int, kSamples int, outPath string) error {
	t := make([]float64, len(wSum))
	for i := range t {
		t[i] = float64(i) * dt
	}

	p0, err := linePlot(t, acc, "time [s]", "# feasible")
	if err != nil {
		return err
	}
	p0.Title.Text = "Variational MPC Stats"
	p0.Y.Min = 0
	p0.Y.Max = float64(kSamples) + 20
	p0.X.Min = 0
	p0.X.Max = float64(steps) * dt

	p1, err := linePlot(t, wSum, "time [s]", "sum r_l")
	if err != nil {
		return err
	}
	p1.Y.Min = 0
	p1.Y.Max = float64(kSamples) + 20
	p1.X.Min = 0
	p1.X.Max = float64(steps) * dt

	grid := [][]*plot.Plot{
		{p0},
		{p1},
	}
	return saveGrid(grid, vg.Points(600), vg.Points(500), outPath)
}

func linePlot(x, y []float64, xLabel, yLabel string) (*plot.Plot, error) {
	p := plot.New()
	p.X.Label.Text = xLabel
	p.Y.Label.Text = yLabel
	xy := make(plotter.XYs, len(x))
	for i := range x {
		xy[i].X = x[i]
		xy[i].Y = y[i]
	}
	line, err := plotter.NewLine(xy)
	if err != nil {
		return nil, err
	}
	line.LineStyle.Width = vg.Points(2)
	p.Add(line)
	grid := plotter.NewGrid()
	grid.Horizontal.Color = color.RGBA{R: 200, G: 200, B: 200, A: 80}
	grid.Vertical.Color = color.RGBA{R: 200, G: 200, B: 200, A: 80}
	p.Add(grid)
	return p, nil
}

func stepPlot(x, y []float64, xLabel, yLabel string) (*plot.Plot, error) {
	p := plot.New()
	p.X.Label.Text = xLabel
	p.Y.Label.Text = yLabel
	if len(x) == 0 {
		return p, nil
	}
	xy := make(plotter.XYs, 0, 2*len(x))
	for i := 0; i < len(x)-1; i++ {
		xy = append(xy, plotter.XY{X: x[i], Y: y[i]})
		xy = append(xy, plotter.XY{X: x[i+1], Y: y[i]})
	}
	xy = append(xy, plotter.XY{X: x[len(x)-1], Y: y[len(y)-1]})
	line, err := plotter.NewLine(xy)
	if err != nil {
		return nil, err
	}
	line.LineStyle.Width = vg.Points(2)
	p.Add(line)
	grid := plotter.NewGrid()
	grid.Horizontal.Color = color.RGBA{R: 200, G: 200, B: 200, A: 80}
	grid.Vertical.Color = color.RGBA{R: 200, G: 200, B: 200, A: 80}
	p.Add(grid)
	return p, nil
}

func addBounds(p *plot.Plot, xMin, xMax, bound float64) {
	zero := hLine(xMin, xMax, 0.0, color.RGBA{R: 200, G: 0, B: 0, A: 255}, vg.Points(1), []vg.Length{vg.Points(4), vg.Points(4)})
	upper := hLine(xMin, xMax, bound, color.RGBA{B: 0, G: 0, R: 0, A: 255}, vg.Points(1), []vg.Length{vg.Points(3), vg.Points(3)})
	lower := hLine(xMin, xMax, -bound, color.RGBA{B: 0, G: 0, R: 0, A: 255}, vg.Points(1), []vg.Length{vg.Points(3), vg.Points(3)})
	p.Add(zero, upper, lower)
}

func hLine(xMin, xMax, y float64, c color.Color, width vg.Length, dashes []vg.Length) *plotter.Line {
	xy := plotter.XYs{{X: xMin, Y: y}, {X: xMax, Y: y}}
	line, _ := plotter.NewLine(xy)
	line.Color = c
	line.LineStyle.Width = width
	if len(dashes) > 0 {
		line.LineStyle.Dashes = dashes
	}
	return line
}

func saveGrid(plots [][]*plot.Plot, width, height vg.Length, path string) error {
	img := vgimg.New(width, height)
	dc := draw.New(img)
	rows := len(plots)
	cols := 0
	for _, row := range plots {
		if len(row) > cols {
			cols = len(row)
		}
	}
	tiles := draw.Tiles{Rows: rows, Cols: cols}
	canvases := plot.Align(plots, tiles, dc)
	for i := range plots {
		for j := range plots[i] {
			if plots[i][j] == nil {
				continue
			}
			plots[i][j].Draw(canvases[i][j])
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	png := vgimg.PngCanvas{Canvas: img}
	_, err = png.WriteTo(f)
	return err
}

func colSlice(m *mat.Dense, col int) []float64 {
	r, _ := m.Dims()
	out := make([]float64, r)
	for i := 0; i < r; i++ {
		out[i] = m.At(i, col)
	}
	return out
}
