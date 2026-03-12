package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"gonum.org/v1/gonum/mat"

	"govempc/core"
	"govempc/examples"
	"govempc/internal/ckksconfig"
	"govempc/solvers"
)

func main() {
	// End-to-end CKKS MPC demo:
	// - client offline preprocessing and encryption
	// - cloud offline caching of Enc(LU*xi), Enc(Gamma*xi)
	// - online encrypted evaluation and client-side aggregation

	// Load config (shared by offline/online).
	cfgPath := filepath.Join("output", "ckks_config.json")
	cfg, ok := ckksconfig.Load(cfgPath)
	if !ok {
		panic("missing output/ckks_config.json; run the config cell in cmd/test/main.ipynb")
	}

	// Match the MPC setup used elsewhere for consistency.
	// This section defines the linearized inverted pendulum model and MPC weights.
	if cfg.Mass <= 0 || cfg.Length <= 0 || cfg.Gravity <= 0 || cfg.DT <= 0 || cfg.N <= 0 {
		panic("invalid model parameters in output/ckks_config.json; set m, l, g, dt, N")
	}
	m, l, g := cfg.Mass, cfg.Length, cfg.Gravity
	dt := cfg.DT
	N := cfg.N
	Ac, Bc := examples.LinearizedInvertedPendulumContinuous(m, l, g)
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
	if len(cfg.X0) == 0 {
		panic("missing x0 in output/ckks_config.json; set x0 in the config cell in cmd/test/main.ipynb")
	}
	if len(cfg.X0) != n {
		panic("x0 length must match state dimension n in output/ckks_config.json")
	}

	// Quadratic cost weights.
	Q := diagDense(cfg.QDiag)
	R := diagDense(cfg.RDiag)
	Qf := mat.NewDense(n, n, nil)
	Qf.Scale(cfg.QfScale, Q)

	// Box constraints on state and input.
	if cfg.ThetaMax <= 0 || cfg.OmegaMax <= 0 || cfg.UMax <= 0 {
		panic("invalid constraint bounds in output/ckks_config.json; set thetaMax/omegaMax/uMax")
	}
	thetaMax := cfg.ThetaMax
	omegaMax := cfg.OmegaMax
	uMax := cfg.UMax

	// Build Gx x <= hx and Gu u <= hu (stacked as [I; -I]).
	xBound := []float64{thetaMax, omegaMax}
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
	sigma0 := cfg.Sigma0
	Sigma0 := scaledIdentity(mpc.StackedInputDim(), sigma0*sigma0)
	lambdaParam := cfg.LambdaParam
	K := cfg.K
	T := cfg.T
	nWorkers := cfg.NWorkers
	if nWorkers <= 0 {
		nWorkers = 1
	}
	if K%nWorkers != 0 {
		panic("K must be divisible by nWorkers in output/ckks_config.json")
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
	KChunk := K / nWorkers
	if KChunk < 1 {
		panic("K_chunk must be positive")
	}
	if slots < slotWidth*KChunk {
		panic("not enough slots to process K_chunk samples; increase logN or reduce K or nWorkers")
	}

	logW := make([]float64, K)
	kgen := ckks.NewKeyGenerator(params)
	rotations := collectOnlineRotations(KChunk, dim, p, slots)
	gks := make([]*rlwe.GaloisKey, 0, len(rotations))
	for _, rot := range rotations {
		gks = append(gks, kgen.GenGaloisKeyNew(params.GaloisElementForRotation(rot), sk))
	}
	evalKeys := rlwe.NewMemEvaluationKeySet(rlk, gks...)

	// Precompute polynomial coefficients in power basis (z-domain).
	// We evaluate h_l on encrypted g values using Horner's rule.
	chebCoeffs := solvers.ChebyshevReLUCoeffs(chebOrder, chebBound, 0)
	polyT := chebToPower(chebCoeffs)
	polyZ := scalePoly(polyT, chebBound)
	// Match the plain variational controller threshold using delta_l = h_l(0).
	deltaL := evalPolyReal(polyZ, 0.0)
	tauS := float64(p) * deltaL

	workers := make([]*onlineWorker, nWorkers)
	for w := 0; w < nWorkers; w++ {
		workers[w] = newOnlineWorker(
			workerCacheDir(cacheDir, w, nWorkers),
			KChunk,
			dim,
			p,
			slots,
			params,
			pk,
			sk,
			evalKeys,
			polyZ,
			tauS,
		)
	}

	// ---- Online protocol (Algorithm 2) ----
	// We simulate T time steps to produce a trajectory.
	x0 := cfg.X0
	x := make([]float64, len(x0))
	copy(x, x0)

	xs := mat.NewDense(T+1, n, nil)
	us := mat.NewDense(T, mIn, nil)
	for j := 0; j < n; j++ {
		xs.Set(0, j, x[j])
	}

	controlMsSeries := make([]float64, T)
	Uhat := make([]float64, Nm)

	for t := 0; t < T; t++ {
		// Time check start
		iterStart := time.Now()

		// Client: compute and encrypt m_U(x_t), b(x_t).
		mU := computeMU(variational, x)
		b := computeB(penalty, mU, x)

		var wg sync.WaitGroup
		errCh := make(chan error, nWorkers)
		for _, worker := range workers {
			worker := worker
			wg.Add(1)
			go func() {
				defer wg.Done()
				if err := worker.runCycle(t, mU, b); err != nil {
					errCh <- err
				}
			}()
		}
		wg.Wait()
		close(errCh)
		for err := range errCh {
			if err != nil {
				panic(err)
			}
		}

		// Client: aggregate all worker chunks with global log-sum-exp stabilization.
		for j := 0; j < Nm; j++ {
			Uhat[j] = 0.0
		}

		logWMax := math.Inf(-1)
		for w, worker := range workers {
			base := w * KChunk
			for i := 0; i < KChunk; i++ {
				logW[base+i] = -chebEta * worker.sVals[i]
				if logW[base+i] > logWMax {
					logWMax = logW[base+i]
				}
			}
		}
		var wSum float64
		for w, worker := range workers {
			base := w * KChunk
			for i := 0; i < KChunk; i++ {
				wi := math.Exp(logW[base+i] - logWMax)
				wSum += wi
				off := i * dim
				for j := 0; j < Nm; j++ {
					Uhat[j] += wi * worker.uFlat[off+j]
				}
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
		// Time check end
		controlMs := time.Since(iterStart).Seconds() * 1000.0
		controlMsSeries[t] = controlMs
		fmt.Printf("step %d control_iter_ms %.3f\n", t, controlMs)
	}

	controlMean, controlStd := meanStd(controlMsSeries)

	// Report only summary info (no raw values).
	fmt.Printf("L_U dims: %dx%d\n", rLU, dim)
	fmt.Printf("Gamma dims: %dx%d\n", rG, dim)
	fmt.Printf("CKKS slots: %d\n", slots)
	fmt.Printf("K: %d, nWorkers: %d, K_chunk: %d\n", K, nWorkers, KChunk)
	fmt.Printf("Packed samples per ct: %d (slotWidth=%d)\n", KChunk, slotWidth)
	fmt.Printf("Cache dir: %s\n", cacheDir)
	fmt.Printf("Control iteration time mean/std: %.3f ms / %.3f ms\n", controlMean, controlStd)

	// Persist trajectories and cloud timing.
	outDir := filepath.Join("output")
	_ = os.MkdirAll(outDir, 0o755)
	writeMatCSV(filepath.Join(outDir, "xs_ckks.csv"), xs)
	writeMatCSV(filepath.Join(outDir, "us_ckks.csv"), us)
	writeSeriesCSV(filepath.Join(outDir, "control_iter_ms_ckks.csv"), controlMsSeries)
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

type onlineWorker struct {
	cache        *cipherCache
	encoder      *ckks.Encoder
	encryptor    *rlwe.Encryptor
	decryptor    *rlwe.Decryptor
	evaluator    *ckks.Evaluator
	ptMU         *rlwe.Plaintext
	ptB          *rlwe.Plaintext
	valsMU       []complex128
	valsB        []complex128
	decU         []complex128
	decS         []complex128
	uFlat        []float64
	sVals        []float64
	polyZ        []float64
	scoreMask    *rlwe.Plaintext
	scoreMaskBuf []complex128
	tauS         float64
	dim          int
	p            int
	kChunk       int
	slots        int
	defaultScale rlwe.Scale
	cloudMs      float64
	params       ckks.Parameters
}

func newOnlineWorker(
	cacheDir string,
	kChunk int,
	dim int,
	p int,
	slots int,
	params ckks.Parameters,
	pk *rlwe.PublicKey,
	sk *rlwe.SecretKey,
	evalKeys *rlwe.MemEvaluationKeySet,
	polyZ []float64,
	tauS float64,
) *onlineWorker {
	return &onlineWorker{
		cache:        newCipherCache(cacheDir),
		encoder:      ckks.NewEncoder(params),
		encryptor:    ckks.NewEncryptor(params, pk),
		decryptor:    ckks.NewDecryptor(params, sk),
		evaluator:    ckks.NewEvaluator(params, evalKeys),
		valsMU:       make([]complex128, slots),
		valsB:        make([]complex128, slots),
		decU:         make([]complex128, slots),
		decS:         make([]complex128, slots),
		uFlat:        make([]float64, kChunk*dim),
		sVals:        make([]float64, kChunk),
		polyZ:        polyZ,
		scoreMaskBuf: make([]complex128, slots),
		tauS:         tauS,
		dim:          dim,
		p:            p,
		kChunk:       kChunk,
		slots:        slots,
		defaultScale: params.DefaultScale(),
		params:       params,
	}
}

func (w *onlineWorker) runCycle(t int, mU []float64, b []float64) error {
	ctLu := w.cache.Load("ct_Lu", t)
	ctGamma := w.cache.Load("ct_Gamma", t)

	if w.ptMU == nil || w.ptMU.Level() != ctLu.Level() {
		w.ptMU = ckks.NewPlaintext(w.params, ctLu.Level())
	}
	if w.ptB == nil || w.ptB.Level() != ctGamma.Level() {
		w.ptB = ckks.NewPlaintext(w.params, ctGamma.Level())
	}

	ctMU := tileEncryptVector(mU, w.dim, w.kChunk, w.slots, w.encoder, w.encryptor, w.ptMU, w.valsMU, w.defaultScale)
	ctB := tileEncryptVector(b, w.p, w.kChunk, w.slots, w.encoder, w.encryptor, w.ptB, w.valsB, w.defaultScale)

	cloudStart := time.Now()
	ctU, err := w.evaluator.AddNew(ctMU, ctLu)
	if err != nil {
		return err
	}
	ctG, err := w.evaluator.AddNew(ctB, ctGamma)
	if err != nil {
		return err
	}
	ctPoly := evalPoly(ctG, w.polyZ, w.evaluator)
	if ctPoly == nil {
		return fmt.Errorf("nil ciphertext after polynomial evaluation")
	}
	if w.scoreMask == nil || w.scoreMask.Level() != ctPoly.Level() {
		w.scoreMask = makeFirstSlotMaskPlaintext(w.p, w.kChunk, w.slots, w.params, ctPoly.Level(), w.encoder, w.scoreMaskBuf)
	}
	ctS := sumWithinBlocks(ctPoly, w.p, w.scoreMask, w.evaluator)
	w.cloudMs = time.Since(cloudStart).Seconds() * 1000.0

	ptU := w.decryptor.DecryptNew(ctU)
	if err := w.encoder.Decode(ptU, w.decU); err != nil {
		return err
	}
	ptS := w.decryptor.DecryptNew(ctS)
	if err := w.encoder.Decode(ptS, w.decS); err != nil {
		return err
	}

	for i := 0; i < w.kChunk; i++ {
		uOff := i * w.dim
		for j := 0; j < w.dim; j++ {
			w.uFlat[uOff+j] = real(w.decU[uOff+j])
		}
		score := real(w.decS[i*w.p])
		score -= w.tauS
		if score < 0 {
			score = 0
		}
		w.sVals[i] = score
	}
	return nil
}

type cipherCache struct {
	dir   string
	ctMap map[string]*rlwe.Ciphertext
}

func newCipherCache(dir string) *cipherCache {
	return &cipherCache{dir: dir, ctMap: make(map[string]*rlwe.Ciphertext)}
}

func (c *cipherCache) Load(prefix string, t int) *rlwe.Ciphertext {
	origKey := fmt.Sprintf("%s_%04d", prefix, t)
	if ct, ok := c.ctMap[origKey]; ok {
		return ct
	}
	key := origKey
	path := filepath.Join(c.dir, key+".bin")
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) && t != 0 {
			key0 := fmt.Sprintf("%s_%04d", prefix, 0)
			if ct, ok := c.ctMap[key0]; ok {
				c.ctMap[origKey] = ct
				return ct
			}
			key = key0
			path = filepath.Join(c.dir, key+".bin")
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
	c.ctMap[key] = ct
	if key != origKey {
		c.ctMap[origKey] = ct
	}
	return ct
}

func workerCacheDir(baseDir string, workerID int, nWorkers int) string {
	if nWorkers <= 1 {
		return baseDir
	}
	return filepath.Join(baseDir, fmt.Sprintf("worker_%d", workerID))
}

func tileEncryptVector(
	vec []float64,
	slotWidth int,
	kChunk int,
	slots int,
	encoder *ckks.Encoder,
	encryptor *rlwe.Encryptor,
	pt *rlwe.Plaintext,
	vals []complex128,
	defaultScale rlwe.Scale,
) *rlwe.Ciphertext {
	for i := range vals {
		vals[i] = 0
	}
	vecLen := len(vec)
	for i := 0; i < kChunk; i++ {
		off := i * slotWidth
		if off >= slots {
			break
		}
		max := slotWidth
		if off+max > slots {
			max = slots - off
		}
		if vecLen < max {
			max = vecLen
		}
		for j := 0; j < max; j++ {
			vals[off+j] = complex(vec[j], 0)
		}
	}
	pt.Scale = defaultScale
	_ = encoder.Encode(vals, pt)
	ct, _ := encryptor.EncryptNew(pt)
	return ct
}

func makeFirstSlotMaskPlaintext(
	blockSize int,
	kChunk int,
	slots int,
	params ckks.Parameters,
	level int,
	encoder *ckks.Encoder,
	buf []complex128,
) *rlwe.Plaintext {
	for i := range buf {
		buf[i] = 0
	}
	if blockSize <= 0 {
		blockSize = 1
	}
	for i := 0; i < kChunk; i++ {
		off := i * blockSize
		if off >= slots {
			break
		}
		buf[off] = complex(1.0, 0)
	}
	pt := ckks.NewPlaintext(params, level)
	pt.Scale = params.DefaultScale()
	_ = encoder.Encode(buf, pt)
	return pt
}

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

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func mod(a, b int) int {
	r := a % b
	if r < 0 {
		r += b
	}
	return r
}

func sumWithinBlocks(
	ctIn *rlwe.Ciphertext,
	blockSize int,
	mask *rlwe.Plaintext,
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
	ctMasked, err := eval.MulNew(ctSum, mask)
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

func evalPolyReal(coeffs []float64, x float64) float64 {
	// Evaluate a real polynomial in power basis at x (Horner's rule).
	y := 0.0
	for k := len(coeffs) - 1; k >= 0; k-- {
		y = y*x + coeffs[k]
	}
	return y
}

func approxReLUError(polyZ []float64, bound float64, samples int) float64 {
	// Approximate delta_l = sup_{z in [-bound,bound]} |[z]_+ - h_l(z)|.
	if bound <= 0 || samples < 2 {
		return 0.0
	}
	step := 2.0 * bound / float64(samples-1)
	maxErr := 0.0
	for i := 0; i < samples; i++ {
		z := -bound + step*float64(i)
		relu := math.Max(z, 0.0)
		approx := evalPolyReal(polyZ, z)
		err := math.Abs(relu - approx)
		if err > maxErr {
			maxErr = err
		}
	}
	return maxErr
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
