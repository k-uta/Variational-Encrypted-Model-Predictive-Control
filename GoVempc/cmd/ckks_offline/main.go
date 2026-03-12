package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"gonum.org/v1/gonum/mat"

	"govempc/core"
	"govempc/examples"
	"govempc/internal/ckksconfig"
)

func main() {
	// Load config (shared by offline/online).
	cfgPath := filepath.Join("output", "ckks_config.json")
	cfg, ok := ckksconfig.Load(cfgPath)
	if !ok {
		panic("missing output/ckks_config.json; run the config cell in cmd/test/main.ipynb")
	}

	m, l, g := cfg.Mass, cfg.Length, cfg.Gravity
	dt := cfg.DT
	N := cfg.N
	Ac, Bc := examples.LinearizedInvertedPendulumContinuous(m, l, g)
	A, B := examples.Discretize(Ac, Bc, dt)
	fmt.Printf("A =\n%v\n", mat.Formatted(A, mat.Prefix("  "), mat.Squeeze()))
	fmt.Printf("B =\n%v\n", mat.Formatted(B, mat.Prefix("  "), mat.Squeeze()))

	// Dimensions and horizon length.
	n, _ := A.Dims()
	_, mIn := B.Dims()

	// Quadratic cost weights.
	Q := diagDense(cfg.QDiag)
	R := diagDense(cfg.RDiag)
	Qf := mat.NewDense(n, n, nil)
	Qf.Scale(cfg.QfScale, Q)

	thetaMax := cfg.ThetaMax
	omegaMax := cfg.OmegaMax
	uMax := cfg.UMax

	// Build Gx x <= hx and Gu u <= hu (stacked as [I; -I]).
	xBound := []float64{thetaMax, omegaMax}
	Gx := stackIdentity(n)
	hx := append(append([]float64{}, xBound...), xBound...)
	Gu := stackIdentity(mIn)
	hu := []float64{uMax, uMax}

	// Build condensed MPC matrices and constraint penalty.
	mpc := core.NewMPCProblem(A, B, Q, R, Qf, N)
	G, hOfX0 := mpc.BuildConstraintMatrices(Gx, hx, Gu, hu)
	penalty := core.NewConstraintPenalty(G, hOfX0, "indicator")

	// Variational MPC parameters
	sigma0 := cfg.Sigma0
	Sigma0 := scaledIdentity(mpc.StackedInputDim(), sigma0*sigma0)
	lambdaParam := cfg.LambdaParam
	clientStart := time.Now()
	variational := core.NewVariationalMPC(mpc, penalty, lambdaParam, Sigma0)

	// Precompute matrices needed in the encrypted protocol.
	LU := variational.LU
	var Gamma mat.Dense
	Gamma.Mul(penalty.G, LU)

	// Dimensions.
	rLU, cLU := LU.Dims()
	rG, cG := Gamma.Dims()
	dim := rLU
	p := rG

	// Cache sizes (samples per packed ciphertext).
	K := cfg.K
	T := cfg.T
	nWorkers := cfg.NWorkers
	if nWorkers <= 0 {
		nWorkers = 1
	}
	if K%nWorkers != 0 {
		panic("K must be divisible by nWorkers in output/ckks_config.json")
	}
	KChunk := K / nWorkers
	slotWidth := maxInt(dim, p)
	slotsNeeded := slotWidth * KChunk
	fmt.Printf("Slot width: %d, slots needed per worker: %d\n", slotWidth, slotsNeeded)

	// CKKS parameters (relaxed security for speed).
	logN := cfg.LogN
	if (1 << (logN - 1)) < slotsNeeded {
		panic("logN too small for required slots; increase logN or reduce K")
	}
	logQ := cfg.LogQ
	logP := cfg.LogP
	logDefaultScale := cfg.LogDefaultScale

	paramsLit := ckks.ParametersLiteral{
		LogN:            logN,
		LogQ:            logQ,
		LogP:            logP,
		LogDefaultScale: logDefaultScale,
	}
	params, _ := ckks.NewParametersFromLiteral(paramsLit)
	slots := params.MaxSlots()
	fmt.Printf("CKKS slots: %d\n", slots)
	if dim > slots || p > slots {
		panic("dim or p exceeds CKKS slots; increase LogN or reduce N")
	}
	if slots < slotWidth*KChunk {
		panic("not enough slots to pack K_chunk samples; increase logN or reduce K or nWorkers")
	}
	cacheDir := filepath.Join("output", "ckks_cache")
	cryptoDir := filepath.Join(cacheDir, "crypto")
	_ = os.MkdirAll(cryptoDir, 0o755)

	// Key generation and evaluator setup.
	kgen := ckks.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	rlk := kgen.GenRelinearizationKeyNew(sk)

	if data, err := params.MarshalBinary(); err == nil {
		_ = writeBinary(filepath.Join(cryptoDir, "params.bin"), data)
	}
	if data, err := sk.MarshalBinary(); err == nil {
		_ = writeBinary(filepath.Join(cryptoDir, "sk.bin"), data)
	}
	if data, err := pk.MarshalBinary(); err == nil {
		_ = writeBinary(filepath.Join(cryptoDir, "pk.bin"), data)
	}
	if data, err := rlk.MarshalBinary(); err == nil {
		_ = writeBinary(filepath.Join(cryptoDir, "rlk.bin"), data)
	}

	rotations := collectRotations(KChunk, dim, p, slots)
	gks := make([]*rlwe.GaloisKey, 0, len(rotations))
	for _, rot := range rotations {
		gks = append(gks, kgen.GenGaloisKeyNew(params.GaloisElementForRotation(rot), sk))
	}
	evalKeys := rlwe.NewMemEvaluationKeySet(rlk, gks...)

	encoder := ckks.NewEncoder(params)
	encryptor := ckks.NewEncryptor(params, pk)

	// Client offline: encrypt cyclic diagonals of L_U and Gamma.
	encLU := encryptCyclicDiagonalsSquare(LU, dim, slots, encoder, encryptor, params)
	encGamma := encryptCyclicDiagonalsGamma(&Gamma, p, dim, slots, encoder, encryptor, params)
	clientMs := time.Since(clientStart).Seconds() * 1000.0

	fmt.Printf("L_U dims: %dx%d\n", rLU, cLU)
	fmt.Printf("Gamma dims: %dx%d\n", rG, cG)
	fmt.Printf("CKKS slots: %d, K: %d, nWorkers: %d, K_chunk: %d, T: %d\n", slots, K, nWorkers, KChunk, T)

	// Cloud offline: each worker samples KChunk xi values, computes Enc(LU * xi)
	// and Enc(Gamma * xi), packs them, and writes its own cache files.
	cloudStart := time.Now()
	var wg sync.WaitGroup
	errCh := make(chan error, nWorkers)
	for workerID := 0; workerID < nWorkers; workerID++ {
		workerID := workerID
		wg.Add(1)
		go func() {
			defer wg.Done()
			workerDir := cacheDir
			if nWorkers > 1 {
				workerDir = filepath.Join(cacheDir, fmt.Sprintf("worker_%d", workerID))
			}
			if err := os.MkdirAll(workerDir, 0o755); err != nil {
				errCh <- err
				return
			}

			workerEncoder := ckks.NewEncoder(params)
			workerEvaluator := ckks.NewEvaluator(params, evalKeys)
			maskLU := makeMaskPlaintext(dim, slots, params, workerEncoder)
			maskG := makeMaskPlaintext(p, slots, params, workerEncoder)
			scratchLU := newEncMatVecScratch(dim, slots, params)
			scratchG := newEncMatVecScratch(p, slots, params)
			rng := rand.New(rand.NewSource(int64(workerID + 1)))

			ctLUList := make([]*rlwe.Ciphertext, KChunk)
			ctGList := make([]*rlwe.Ciphertext, KChunk)
			xi := make([]float64, dim)
			xiPad := make([]float64, p)

			for t := 0; t < T; t++ {
				for s := 0; s < KChunk; s++ {
					for i := 0; i < dim; i++ {
						xi[i] = rng.NormFloat64()
					}
					copy(xiPad, xi)

					ctLUList[s] = encMatVec(encLU, xi, dim, workerEncoder, workerEvaluator, scratchLU)
					ctGList[s] = encMatVec(encGamma, xiPad, p, workerEncoder, workerEvaluator, scratchG)
				}

				cacheLU := packCiphertexts(ctLUList, dim, slots, maskLU, workerEvaluator)
				cacheG := packCiphertexts(ctGList, p, slots, maskG, workerEvaluator)

				if err := writeCiphertext(workerCachePath(workerDir, "ct_Lu", t), cacheLU); err != nil {
					errCh <- err
					return
				}
				if err := writeCiphertext(workerCachePath(workerDir, "ct_Gamma", t), cacheG); err != nil {
					errCh <- err
					return
				}
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
	cloudMs := time.Since(cloudStart).Seconds() * 1000.0

	fmt.Printf("Cache Enc(L_U * xi): T=%d, workers=%d, packed=%d samples per ct\n", T, nWorkers, KChunk)
	fmt.Printf("Cache Enc(Gamma * xi): T=%d, workers=%d, packed=%d samples per ct\n", T, nWorkers, KChunk)

	fmt.Printf("Client offline time: %.3f ms\n", clientMs)
	fmt.Printf("Cloud offline time: %.3f ms\n", cloudMs)
}

func encryptCyclicDiagonalsSquare(
	matIn *mat.Dense,
	dim int,
	slots int,
	encoder *ckks.Encoder,
	encryptor *rlwe.Encryptor,
	params ckks.Parameters,
) []*rlwe.Ciphertext {
	out := make([]*rlwe.Ciphertext, dim)
	vals := make([]complex128, slots)
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	pt.Scale = params.DefaultScale()
	for k := 0; k < dim; k++ {
		for i := 0; i < dim; i++ {
			vals[i] = complex(matIn.At(i, (i+k)%dim), 0)
		}
		_ = encoder.Encode(vals, pt)
		ct, _ := encryptor.EncryptNew(pt)
		out[k] = ct
	}
	return out
}

func encryptCyclicDiagonalsGamma(
	gamma *mat.Dense,
	p int,
	dim int,
	slots int,
	encoder *ckks.Encoder,
	encryptor *rlwe.Encryptor,
	params ckks.Parameters,
) []*rlwe.Ciphertext {
	out := make([]*rlwe.Ciphertext, p)
	vals := make([]complex128, slots)
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	pt.Scale = params.DefaultScale()
	for k := 0; k < p; k++ {
		for i := 0; i < p; i++ {
			vals[i] = 0
		}
		for i := 0; i < p; i++ {
			j := (i + k) % p
			if j < dim {
				vals[i] = complex(gamma.At(i, j), 0)
			}
		}
		_ = encoder.Encode(vals, pt)
		ct, _ := encryptor.EncryptNew(pt)
		out[k] = ct
	}
	return out
}

func encMatVec(
	ctDiags []*rlwe.Ciphertext,
	xi []float64,
	vecLen int,
	encoder *ckks.Encoder,
	evaluator *ckks.Evaluator,
	scratch *encMatVecScratch,
) *rlwe.Ciphertext {
	var acc *rlwe.Ciphertext
	// Rotate vector and multiply with each encryption of cyclic diagonal
	// Then, add them.
	for k, ctDiag := range ctDiags {
		rotateVectorInto(scratch.rot, xi, k)
		for i := 0; i < vecLen && i < len(scratch.rot); i++ {
			scratch.vals[i] = complex(scratch.rot[i], 0)
		}
		_ = encoder.Encode(scratch.vals, scratch.pt)
		term, _ := evaluator.MulNew(ctDiag, scratch.pt)
		_ = evaluator.Rescale(term, term)
		if acc == nil {
			acc = term
		} else {
			evaluator.Add(acc, term, acc)
		}
	}
	return acc
}

type encMatVecScratch struct {
	vals []complex128
	rot  []float64
	pt   *rlwe.Plaintext
}

func newEncMatVecScratch(vecLen, slots int, params ckks.Parameters) *encMatVecScratch {
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	pt.Scale = params.DefaultScale()
	return &encMatVecScratch{
		vals: make([]complex128, slots),
		rot:  make([]float64, vecLen),
		pt:   pt,
	}
}

func rotateVectorInto(dst, vec []float64, k int) {
	n := len(vec)
	if n == 0 {
		return
	}
	k = mod(k, n)
	copy(dst, vec[k:])
	copy(dst[n-k:], vec[:k])
}

func makeMaskPlaintext(slotWidth, slots int, params ckks.Parameters, encoder *ckks.Encoder) *rlwe.Plaintext {
	vals := make([]complex128, slots)
	for i := 0; i < slotWidth && i < slots; i++ {
		vals[i] = complex(1.0, 0)
	}
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	_ = encoder.Encode(vals, pt)
	return pt
}

func packCiphertexts(
	ctList []*rlwe.Ciphertext,
	slotWidth int,
	slots int,
	mask *rlwe.Plaintext,
	evaluator *ckks.Evaluator,
) *rlwe.Ciphertext {
	var packed *rlwe.Ciphertext
	for k, ct := range ctList {
		ctMasked, _ := evaluator.MulNew(ct, mask)
		_ = evaluator.Rescale(ctMasked, ctMasked)

		// Rotate and add. Each sample in their respective slots
		r := mod(-k*slotWidth, slots)
		if r != 0 {
			ctRot, _ := evaluator.RotateNew(ctMasked, r)
			ctMasked = ctRot
		}

		if packed == nil {
			packed = ctMasked
		} else {
			evaluator.Add(packed, ctMasked, packed)
		}
	}
	return packed
}

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

func decryptMaxAbsErrorVec(
	ct *rlwe.Ciphertext,
	ref *mat.Dense,
	xi []float64,
	vecLen int,
	decBuf []complex128,
	encoder *ckks.Encoder,
	decryptor *rlwe.Decryptor,
) float64 {
	ptDec := decryptor.DecryptNew(ct)
	_ = encoder.Decode(ptDec, decBuf)

	maxErr := 0.0
	r, c := ref.Dims()
	if vecLen > r {
		vecLen = r
	}
	if len(xi) < c {
		c = len(xi)
	}
	for i := 0; i < vecLen; i++ {
		want := 0.0
		for j := 0; j < c; j++ {
			want += ref.At(i, j) * xi[j]
		}
		got := real(decBuf[i])
		err := math.Abs(got - want)
		if err > maxErr {
			maxErr = err
		}
	}
	return maxErr
}

func mod(a, b int) int {
	r := a % b
	if r < 0 {
		r += b
	}
	return r
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func writeCiphertext(path string, ct *rlwe.Ciphertext) error {
	if ct == nil {
		return fmt.Errorf("nil ciphertext for %s", path)
	}
	data, err := ct.MarshalBinary()
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func writeBinary(path string, data []byte) error {
	return os.WriteFile(path, data, 0o644)
}

func workerCachePath(baseDir, prefix string, t int) string {
	return filepath.Join(baseDir, fmt.Sprintf("%s_%04d.bin", prefix, t))
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
