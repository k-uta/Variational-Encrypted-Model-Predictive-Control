package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"gonum.org/v1/gonum/mat"

	"govempc/core"
	"govempc/examples"
	"govempc/solvers"
)

func main() {
	// End-to-end CKKS MPC demo:
	// - client offline preprocessing and encryption
	// - cloud offline caching of Enc(LU*xi), Enc(Gamma*xi)
	// - online encrypted evaluation and client-side aggregation

	// Load config (shared by offline/online).
	cfgPath := filepath.Join("output", "ckks_config.json")
	cfg, ok := loadConfig(cfgPath)
	if !ok {
		panic("missing output/ckks_config.json; run the config cell in cmd/test/main.ipynb")
	}

	// Match the MPC setup used elsewhere for consistency.
	// This section defines the linearized cart-pole model and MPC weights.
	if cfg.MCart <= 0 || cfg.Mass <= 0 || cfg.Length <= 0 || cfg.Gravity <= 0 || cfg.DT <= 0 || cfg.N <= 0 {
		panic("invalid model parameters in output/ckks_config.json; set M, m, l, g, dt, N")
	}
	M, m, l, g := cfg.MCart, cfg.Mass, cfg.Length, cfg.Gravity
	dt := cfg.DT
	N := cfg.N
	Ac, Bc := examples.LinearizedCartpoleContinuous(M, m, l, g)
	A, B := examples.Discretize(Ac, Bc, dt)

	// Dimensions and horizon length.
	n, _ := A.Dims()
	_, mIn := B.Dims()
	if len(cfg.QDiag) != n {
		panic("QDiag length must match state dimension n in output/ckks_config.json")
	}
	if len(cfg.RDiag) != mIn {
		panic("RDiag length must match input dimension m in output/ckks_config.json")
	}
	if cfg.QfScale <= 0 {
		panic("invalid QfScale in output/ckks_config.json; set a positive value")
	}

	// Quadratic cost weights.
	Q := diagDense(cfg.QDiag)
	R := diagDense(cfg.RDiag)
	Qf := mat.NewDense(n, n, nil)
	Qf.Scale(cfg.QfScale, Q)

	// Box constraints on state and input.
	if cfg.XMax <= 0 || cfg.VMax <= 0 || cfg.ThetaMax <= 0 || cfg.OmegaMax <= 0 || cfg.UMax <= 0 {
		panic("invalid constraint bounds in output/ckks_config.json; set xMax/vMax/thetaMax/omegaMax/uMax")
	}
	xMax := cfg.XMax
	vMax := cfg.VMax
	thetaMax := cfg.ThetaMax
	omegaMax := cfg.OmegaMax
	uMax := cfg.UMax

	// Build Gx x <= hx and Gu u <= hu (stacked as [I; -I]).
	xBound := []float64{xMax, vMax, thetaMax, omegaMax}
	Gx := stackIdentity(n)
	hx := append(append([]float64{}, xBound...), xBound...)
	Gu := stackIdentity(mIn)
	hu := []float64{uMax, uMax}

	// Build condensed MPC matrices (Lambda, Psi, S, H).
	mpc := core.NewMPCProblem(A, B, Q, R, Qf, N)
	G, hOfX0 := mpc.BuildConstraintMatrices(Gx, hx, Gu, hu)
	// ConstraintPenalty wraps G and h(x0) to compute constraint residuals.
	penalty := core.NewConstraintPenalty(G, hOfX0, "indicator")

	// Chebyshev polynomial parameters for ReLU surrogate.
	// These control the polynomial approximation h_l used for encrypted constraint penalties.
	chebOrder := cfg.ChebOrder
	chebBound := cfg.ChebBound
	chebEta := cfg.ChebEta
	if chebOrder <= 0 || chebBound <= 0 || chebEta <= 0 {
		panic("invalid cheb parameters in output/ckks_config.json; set chebOrder/chebBound/chebEta in main.ipynb")
	}

	// Variational MPC parameters.
	if cfg.Sigma0 <= 0 || cfg.LambdaParam <= 0 {
		panic("invalid sigma0 or lambda in output/ckks_config.json; set positive values")
	}
	sigma0 := cfg.Sigma0
	Sigma0 := scaledIdentity(mpc.StackedInputDim(), sigma0*sigma0)
	lambdaParam := cfg.LambdaParam
	K := cfg.K
	T := cfg.T
	if K <= 0 || T <= 0 {
		panic("invalid K or T in output/ckks_config.json; set positive values in main.ipynb")
	}

	// ---- Client offline preprocessing (Algorithm 1: steps 1-4) ----
	variational := core.NewVariationalMPC(mpc, penalty, lambdaParam, Sigma0)
	LU := variational.LU
	var Gamma mat.Dense
	Gamma.Mul(penalty.G, LU)

	// Load CKKS parameters and keys from offline cache.
	cacheDir := filepath.Join("output", "ckks_cache")
	cryptoDir := filepath.Join(cacheDir, "crypto")
	params, sk, pk, rlk := loadCrypto(cryptoDir)

	// Nm = N * m (stacked control dimension).
	Nm := mpc.StackedInputDim()
	dim := Nm
	rLU, _ := LU.Dims()
	rG, _ := Gamma.Dims()
	p := rG

	slots := params.MaxSlots()
	if dim > slots || p > slots {
		panic("dim or p exceeds CKKS slots")
	}

	slotWidth := maxInt(dim, p)
	maxPacked := slots / slotWidth
	if maxPacked < 1 {
		panic("slotWidth exceeds CKKS slots; cannot pack any samples")
	}
	KChunk := K
	if KChunk > maxPacked {
		fmt.Printf("Reducing K from %d to %d to fit in one ciphertext (slots=%d, width=%d)\n", KChunk, maxPacked, slots, slotWidth)
		KChunk = maxPacked
	}

	// Rotation keys for summing constraint slots within each p-block.
	var gks []*rlwe.GaloisKey
	if p > 1 {
		kgen := ckks.NewKeyGenerator(params)
		for r := 1; r < p; r++ {
			gks = append(gks, kgen.GenGaloisKeyNew(params.GaloisElementForRotation(r), sk))
		}
	}

	encoder := ckks.NewEncoder(params)
	encryptor := ckks.NewEncryptor(params, pk)
	decryptor := ckks.NewDecryptor(params, sk)
	evalKeys := rlwe.NewMemEvaluationKeySet(rlk, gks...)
	evaluator := ckks.NewEvaluator(params, evalKeys)

	// Reuse decode buffer to reduce allocations.
	decBuf := make([]complex128, slots)
	// Mask to keep the first slot of each p-block after summing constraints.
	maskP := make([]complex128, slots)
	for i := 0; i < KChunk; i++ {
		idx := i * p
		if idx < slots {
			maskP[idx] = complex(1.0, 0)
		}
	}

	// Precompute polynomial coefficients in power basis (z-domain).
	// We evaluate h_l on encrypted g values using Horner's rule.
	chebCoeffs := solvers.ChebyshevReLUCoeffs(chebOrder, chebBound, 0)
	polyT := chebToPower(chebCoeffs)
	polyZ := scalePoly(polyT, chebBound)
	threshold := polyZ[0]

	// ---- Online protocol (Algorithm 2) ----
	// We simulate T time steps to produce a trajectory.
	x0 := []float64{0.3, 0.0, 0.20, 0.0}
	x := make([]float64, len(x0))
	copy(x, x0)

	xs := mat.NewDense(T+1, n, nil)
	us := mat.NewDense(T, mIn, nil)
	for j := 0; j < n; j++ {
		xs.Set(0, j, x[j])
	}

	cloudMsSeries := make([]float64, T)
	Uhat := make([]float64, Nm)
	uHatScratch := make([]float64, Nm)

	for t := 0; t < T; t++ {
		// Client: compute and encrypt m_U(x_t), b(x_t).
		mU := computeMU(variational, x)
		b := computeB(penalty, mU, x)

		ctLu := loadCacheCiphertext(cacheDir, "ct_Lu", t)
		ctGamma := loadCacheCiphertext(cacheDir, "ct_Gamma", t)

		ctMU := tileEncryptVector(mU, dim, KChunk, slots, params, encoder, encryptor, ctLu.Level())
		ctB := tileEncryptVector(b, p, KChunk, slots, params, encoder, encryptor, ctGamma.Level())

		// Cloud: compute Enc(U) and Enc(s_l) on packed slots.
		cloudStart := time.Now()
		ctU, _ := evaluator.AddNew(ctMU, ctLu)
		ctG, _ := evaluator.AddNew(ctB, ctGamma)
		ctS := evalPoly(ctG, polyZ, evaluator)
		// Sum constraint penalties within each p-block and keep only the first slot.
		ctS = sumWithinBlocks(ctS, p, KChunk, slots, maskP, params, encoder, evaluator)
		cloudMs := time.Since(cloudStart).Seconds() * 1000.0
		cloudMsSeries[t] = cloudMs
		fmt.Printf("step %d cloud_ms %.3f\n", t, cloudMs)

		// Client: decrypt packed U and s, then aggregate.
		ptU := decryptor.DecryptNew(ctU)
		_ = encoder.Decode(ptU, decBuf)

		ptS := decryptor.DecryptNew(ctS)
		decS := make([]complex128, slots)
		_ = encoder.Decode(ptS, decS)

		// Reset accumulator.
		for j := 0; j < Nm; j++ {
			Uhat[j] = 0.0
		}

		// Compute weights with log-sum-exp stabilization.
		logW := make([]float64, KChunk)
		logWMax := math.Inf(-1)
		for i := 0; i < KChunk; i++ {
			s := real(decS[i*p])
			if s < threshold {
				s = 0.0
			}
			logW[i] = -chebEta * s
			if logW[i] > logWMax {
				logWMax = logW[i]
			}
		}
		var wSum float64
		for i := 0; i < KChunk; i++ {
			w := math.Exp(logW[i] - logWMax)
			wSum += w
			off := i * dim
			for j := 0; j < Nm; j++ {
				if off+j < slots {
					uHatScratch[j] = real(decBuf[off+j])
				} else {
					uHatScratch[j] = 0.0
				}
			}
			for j := 0; j < Nm; j++ {
				Uhat[j] += w * uHatScratch[j]
			}
		}
		if wSum > 0 {
			for j := 0; j < Nm; j++ {
				Uhat[j] /= wSum
			}
		}

		// Apply only the first control input of the stacked sequence.
		for j := 0; j < mIn; j++ {
			us.Set(t, j, Uhat[j])
		}

		// State propagation with discrete-time linear dynamics.
		Ax := core.MatVecMul(A, x)
		Bu := core.MatVecMul(B, Uhat[:mIn])
		for i := 0; i < n; i++ {
			x[i] = Ax[i] + Bu[i]
			xs.Set(t+1, i, x[i])
		}
	}

	cloudMean, cloudStd := meanStd(cloudMsSeries)

	// Report only summary info (no raw values).
	fmt.Printf("L_U dims: %dx%d\n", rLU, dim)
	fmt.Printf("Gamma dims: %dx%d\n", rG, dim)
	fmt.Printf("CKKS slots: %d\n", slots)
	fmt.Printf("Packed samples per ct: %d (slotWidth=%d)\n", KChunk, slotWidth)
	fmt.Printf("Cache dir: %s\n", cacheDir)
	fmt.Printf("Online cloud time mean/std: %.3f ms / %.3f ms\n", cloudMean, cloudStd)

	// Persist trajectories and cloud timing.
	outDir := filepath.Join("output")
	_ = os.MkdirAll(outDir, 0o755)
	writeMatCSV(filepath.Join(outDir, "xs_ckks.csv"), xs)
	writeMatCSV(filepath.Join(outDir, "us_ckks.csv"), us)
	writeSeriesCSV(filepath.Join(outDir, "cloud_ms_ckks.csv"), cloudMsSeries)
	fmt.Printf("Saved CKKS MPC outputs to %s\n", outDir)
}

func computeMU(v *core.VariationalMPC, x0 []float64) []float64 {
	// m_U(x) = -(1/lambda) * Sigma_U * S^T * x
	Stx := core.MatVecMul(v.MPC.S.T(), x0)
	SigmaStx := core.MatVecMul(v.SigmaU, Stx)
	return core.VecScale(SigmaStx, -(1.0 / v.LambdaParam))
}

func computeB(p *core.ConstraintPenalty, mU []float64, x0 []float64) []float64 {
	// b(x) = G * m_U(x) - h(x), used in g = b + Gamma * xi.
	h := p.HFunc(x0)
	GmU := core.MatVecMul(p.G, mU)
	return core.VecSub(GmU, h)
}

type ckksConfig struct {
	MCart           float64 `json:"M"`
	Mass            float64 `json:"m"`
	Length          float64 `json:"l"`
	Gravity         float64 `json:"g"`
	DT              float64 `json:"dt"`
	N               int     `json:"N"`
	QDiag           []float64 `json:"QDiag"`
	RDiag           []float64 `json:"RDiag"`
	QfScale         float64 `json:"QfScale"`
	XMax            float64 `json:"xMax"`
	VMax            float64 `json:"vMax"`
	ThetaMax        float64 `json:"thetaMax"`
	OmegaMax        float64 `json:"omegaMax"`
	UMax            float64 `json:"uMax"`
	Sigma0          float64 `json:"sigma0"`
	LambdaParam     float64 `json:"lambda"`
	LogN            int     `json:"logN"`
	LogQ            []int   `json:"logQ"`
	LogP            []int   `json:"logP"`
	LogDefaultScale int     `json:"logDefaultScale"`
	K               int     `json:"K"`
	T               int     `json:"T"`
	ChebOrder       int     `json:"chebOrder"`
	ChebBound       float64 `json:"chebBound"`
	ChebEta         float64 `json:"chebEta"`
}

func loadConfig(path string) (ckksConfig, bool) {
	var cfg ckksConfig
	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, false
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		panic(err)
	}
	return cfg, true
}

func loadCrypto(dir string) (ckks.Parameters, *rlwe.SecretKey, *rlwe.PublicKey, *rlwe.RelinearizationKey) {
	paramsData, err := os.ReadFile(filepath.Join(dir, "params.bin"))
	if err != nil {
		panic(err)
	}
	var params ckks.Parameters
	if err := params.UnmarshalBinary(paramsData); err != nil {
		panic(err)
	}

	skData, err := os.ReadFile(filepath.Join(dir, "sk.bin"))
	if err != nil {
		panic(err)
	}
	pkData, err := os.ReadFile(filepath.Join(dir, "pk.bin"))
	if err != nil {
		panic(err)
	}
	rlkData, err := os.ReadFile(filepath.Join(dir, "rlk.bin"))
	if err != nil {
		panic(err)
	}

	sk := new(rlwe.SecretKey)
	pk := new(rlwe.PublicKey)
	rlk := new(rlwe.RelinearizationKey)
	if err := sk.UnmarshalBinary(skData); err != nil {
		panic(err)
	}
	if err := pk.UnmarshalBinary(pkData); err != nil {
		panic(err)
	}
	if err := rlk.UnmarshalBinary(rlkData); err != nil {
		panic(err)
	}
	return params, sk, pk, rlk
}

func loadCacheCiphertext(cacheDir, prefix string, t int) *rlwe.Ciphertext {
	path := filepath.Join(cacheDir, fmt.Sprintf("%s_%04d.bin", prefix, t))
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) && t != 0 {
			path = filepath.Join(cacheDir, fmt.Sprintf("%s_%04d.bin", prefix, 0))
		} else if err != nil {
			panic(err)
		}
	}
	data, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	ct := new(rlwe.Ciphertext)
	if err := ct.UnmarshalBinary(data); err != nil {
		panic(err)
	}
	return ct
}

func tileEncryptVector(
	vec []float64,
	slotWidth int,
	kChunk int,
	slots int,
	params ckks.Parameters,
	encoder *ckks.Encoder,
	encryptor *rlwe.Encryptor,
	level int,
) *rlwe.Ciphertext {
	vals := make([]complex128, slots)
	for i := 0; i < kChunk; i++ {
		off := i * slotWidth
		for j := 0; j < slotWidth && off+j < slots; j++ {
			if j < len(vec) {
				vals[off+j] = complex(vec[j], 0)
			}
		}
	}
	pt := ckks.NewPlaintext(params, level)
	pt.Scale = params.DefaultScale()
	_ = encoder.Encode(vals, pt)
	ct, _ := encryptor.EncryptNew(pt)
	return ct
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func sumWithinBlocks(
	ctIn *rlwe.Ciphertext,
	blockSize int,
	kChunk int,
	slots int,
	maskVals []complex128,
	params ckks.Parameters,
	encoder *ckks.Encoder,
	eval *ckks.Evaluator,
) *rlwe.Ciphertext {
	if blockSize <= 1 {
		return ctIn
	}
	ctSum := ctIn.CopyNew()
	for r := 1; r < blockSize; r++ {
		ctRot, err := eval.RotateNew(ctIn, r)
		if err != nil {
			panic(err)
		}
		eval.Add(ctSum, ctRot, ctSum)
	}
	pt := ckks.NewPlaintext(params, ctSum.Level())
	pt.Scale = params.DefaultScale()
	_ = encoder.Encode(maskVals, pt)
	ctMasked, err := eval.MulNew(ctSum, pt)
	if err != nil {
		panic(err)
	}
	_ = eval.Rescale(ctMasked, ctMasked)
	return ctMasked
}

func evalPoly(ctX *rlwe.Ciphertext, coeffs []float64, eval *ckks.Evaluator) *rlwe.Ciphertext {
	// Evaluate polynomial in power basis using Horner's rule.
	if len(coeffs) == 0 {
		return ctX.CopyNew()
	}
	// Initialize y = c_d (as ciphertext constant).
	ctY, _ := eval.MulNew(ctX, 0.0)
	_ = eval.Add(ctY, coeffs[len(coeffs)-1], ctY)

	for k := len(coeffs) - 2; k >= 0; k-- {
		ctXk := ctX
		if ctXk.Level() > ctY.Level() {
			ctXk = eval.DropLevelNew(ctX, ctX.Level()-ctY.Level())
		}
		ctY, _ = eval.MulRelinNew(ctY, ctXk)
		_ = eval.Rescale(ctY, ctY)
		_ = eval.Add(ctY, coeffs[k], ctY)
	}
	return ctY
}

func decryptMaxAbsError(ct *rlwe.Ciphertext, ref *mat.Dense, row int, xi []float64, nm int, buf []complex128, encoder *ckks.Encoder, decryptor *rlwe.Decryptor) float64 {
	// Compare decrypted inner product in slot 0 vs. plaintext reference.
	want := 0.0
	for k := 0; k < nm; k++ {
		want += ref.At(row, k) * xi[k]
	}
	ptDec := decryptor.DecryptNew(ct)
	_ = encoder.Decode(ptDec, buf)
	got := real(buf[0])
	return math.Abs(got - want)
}

func decryptScalar(ct *rlwe.Ciphertext, buf []complex128, encoder *ckks.Encoder, decryptor *rlwe.Decryptor) float64 {
	// Decrypt and read slot 0.
	ptDec := decryptor.DecryptNew(ct)
	_ = encoder.Decode(ptDec, buf)
	return real(buf[0])
}

type progressBar struct {
	label string
	total int
	width int
	start time.Time
	last  time.Time
	out   *os.File
}

func newProgressBar(label string, total int) *progressBar {
	return &progressBar{
		label: label,
		total: total,
		width: 28,
		start: time.Now(),
		out:   os.Stderr,
	}
}

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
	if done != p.total && !p.last.IsZero() && now.Sub(p.last) < 200*time.Millisecond {
		return
	}
	p.last = now
	pct := float64(done) / float64(p.total)
	filled := int(math.Round(pct * float64(p.width)))
	if filled < 0 {
		filled = 0
	} else if filled > p.width {
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

func encodeRealVector(encoder *ckks.Encoder, v []float64, pt *rlwe.Plaintext) {
	// Encode a real vector into the CKKS plaintext slots.
	vals := make([]complex128, len(v))
	for i := range v {
		vals[i] = complex(v[i], 0)
	}
	_ = encoder.Encode(vals, pt)
}

func chebToPower(cheb []float64) []float64 {
	// Convert Chebyshev coefficients (T basis) to power basis.
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

func addPolyScaled(dst, src []float64, scale float64) {
	// dst += scale * src
	for i := 0; i < len(src); i++ {
		dst[i] += scale * src[i]
	}
}

func scalePoly(poly []float64, bound float64) []float64 {
	// Scale coefficients for the z-domain (x = z / bound).
	out := make([]float64, len(poly))
	pow := 1.0
	for i := 0; i < len(poly); i++ {
		out[i] = poly[i] / pow
		pow *= bound
	}
	return out
}

func nextPow2(n int) int {
	// Small helper for CKKS slot alignment.
	if n <= 1 {
		return 1
	}
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

func writeMatCSV(path string, m *mat.Dense) {
	// Write dense matrix to CSV with fixed precision.
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
	// Write a vector to a single-column CSV.
	f, _ := os.Create(path)
	defer f.Close()

	w := csv.NewWriter(f)
	for _, v := range series {
		_ = w.Write([]string{fmt.Sprintf("%.10f", v)})
	}
	w.Flush()
	_ = w.Error()
}

func diagDense(vals []float64) *mat.Dense {
	// Create a diagonal matrix from the provided vector.
	n := len(vals)
	out := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		out.Set(i, i, vals[i])
	}
	return out
}

func scaledIdentity(n int, value float64) *mat.Dense {
	// Build value * I_n.
	out := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		out.Set(i, i, value)
	}
	return out
}

func stackIdentity(n int) *mat.Dense {
	// Stack [I; -I] to build box constraints.
	out := mat.NewDense(2*n, n, nil)
	for i := 0; i < n; i++ {
		out.Set(i, i, 1.0)
		out.Set(i+n, i, -1.0)
	}
	return out
}
