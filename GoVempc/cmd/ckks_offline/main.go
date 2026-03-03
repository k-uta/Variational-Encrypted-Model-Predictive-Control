package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"gonum.org/v1/gonum/mat"

	"govempc/internal/ckksconfig"
	"govempc/core"
	"govempc/examples"
)

func main() {
	// Demonstrate the CKKS "offline" phase only:
	// 1) client builds MPC matrices and encrypts row-packed matrices, then
	// 2) cloud multiplies them by random xi samples to populate a cache.
	// This is a lightweight sanity check for ciphertext-plaintext products.

	// Load config (shared by offline/online).
	cfgPath := filepath.Join("output", "ckks_config.json")
	cfg, ok := ckksconfig.Load(cfgPath)
	if !ok {
		panic("missing output/ckks_config.json; run the config cell in cmd/test/main.ipynb")
	}

	// Match the MPC setup used in cmd/test for consistency.
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

	// Build condensed MPC matrices and constraint penalty.
	mpc := core.NewMPCProblem(A, B, Q, R, Qf, N)
	G, hOfX0 := mpc.BuildConstraintMatrices(Gx, hx, Gu, hu)
	penalty := core.NewConstraintPenalty(G, hOfX0, "indicator")

	// Variational MPC parameters.
	if cfg.Sigma0 <= 0 || cfg.LambdaParam <= 0 {
		panic("invalid sigma0 or lambda in output/ckks_config.json; set positive values")
	}
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
	if K <= 0 || T <= 0 {
		panic("invalid K or T in output/ckks_config.json; set positive values in main.ipynb")
	}
	slotWidth := maxInt(dim, p)
	slotsNeeded := slotWidth * K
	fmt.Printf("Slot width: %d, slots needed: %d\n", slotWidth, slotsNeeded)

	// CKKS parameters (relaxed security for speed).
	logN := cfg.LogN
	if logN <= 0 {
		panic("invalid logN in output/ckks_config.json; set a positive value in main.ipynb")
	}
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
	if slots < slotWidth*K {
		panic("not enough slots to pack K samples; increase logN or reduce K")
	}
	// Change this if implementing parallel workers
	KChunk := K
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
	decryptor := ckks.NewDecryptor(params, sk)
	evaluator := ckks.NewEvaluator(params, evalKeys)

	// Client offline: encrypt cyclic diagonals of L_U and Gamma.
	encLU := encryptCyclicDiagonalsSquare(LU, dim, slots, encoder, encryptor, params)
	encGamma := encryptCyclicDiagonalsGamma(&Gamma, p, dim, slots, encoder, encryptor, params)
	clientMs := time.Since(clientStart).Seconds() * 1000.0

	fmt.Printf("L_U dims: %dx%d\n", rLU, cLU)
	fmt.Printf("Gamma dims: %dx%d\n", rG, cG)
	fmt.Printf("CKKS slots: %d, K_chunk: %d, T: %d\n", slots, KChunk, T)

	// Cloud offline: sample xi, compute Enc(LU * xi) and Enc(Gamma * xi), then pack.
	cloudStart := time.Now()
	maxErrLU := 0.0
	maxErrGamma := 0.0

	rng := rand.New(rand.NewSource(0))
	decBuf := make([]complex128, slots)

	maskLU := makeMaskPlaintext(dim, slots, params, encoder)
	maskG := makeMaskPlaintext(p, slots, params, encoder)

	cacheLU := make([]*rlwe.Ciphertext, T)
	cacheG := make([]*rlwe.Ciphertext, T)

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

			// Cyclic diagonal encryption
			ctLU := encMatVec(encLU, xi, dim, slots, params, encoder, evaluator)
			ctG := encMatVec(encGamma, xiPad, p, slots, params, encoder, evaluator)

			errLU := decryptMaxAbsErrorVec(ctLU, LU, xi, dim, decBuf, encoder, decryptor)
			if errLU > maxErrLU {
				maxErrLU = errLU
			}
			errG := decryptMaxAbsErrorVec(ctG, &Gamma, xi, p, decBuf, encoder, decryptor)
			if errG > maxErrGamma {
				maxErrGamma = errG
			}

			ctLUList[s] = ctLU
			ctGList[s] = ctG
		}

		cacheLU[t] = packCiphertexts(ctLUList, dim, slots, maskLU, evaluator)
		cacheG[t] = packCiphertexts(ctGList, p, slots, maskG, evaluator)
	}
	cloudMs := time.Since(cloudStart).Seconds() * 1000.0
	_ = os.MkdirAll(cacheDir, 0o755)
	for t := 0; t < T; t++ {
		if err := writeCiphertext(filepath.Join(cacheDir, fmt.Sprintf("ct_Lu_%04d.bin", t)), cacheLU[t]); err != nil {
			panic(err)
		}
		if err := writeCiphertext(filepath.Join(cacheDir, fmt.Sprintf("ct_Gamma_%04d.bin", t)), cacheG[t]); err != nil {
			panic(err)
		}
	}

	fmt.Printf("Cache Enc(L_U * xi): T=%d, packed=%d samples per ct\n", T, KChunk)
	fmt.Printf("Cache Enc(Gamma * xi): T=%d, packed=%d samples per ct\n", T, KChunk)
	fmt.Printf("Saved cache to %s\n", cacheDir)
	fmt.Printf("Saved params/keys to %s\n", cryptoDir)

	fmt.Printf("Max abs error (L_U * xi): %.3e\n", maxErrLU)
	fmt.Printf("Max abs error (Gamma * xi): %.3e\n", maxErrGamma)
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
	slots int,
	params ckks.Parameters,
	encoder *ckks.Encoder,
	evaluator *ckks.Evaluator,
) *rlwe.Ciphertext {
	var acc *rlwe.Ciphertext
	vals := make([]complex128, slots)
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	pt.Scale = params.DefaultScale()
	rot := make([]float64, vecLen)
	// Rotate vector and multiply with each encryption of cyclic diagonal
	// Then, add them.
	for k, ctDiag := range ctDiags {
		rotateVectorInto(rot, xi, k)
		for i := 0; i < vecLen && i < len(rot); i++ {
			vals[i] = complex(rot[i], 0)
		}
		_ = encoder.Encode(vals, pt)
		term, _ := evaluator.MulNew(ctDiag, pt)
		_ = evaluator.Rescale(term, term)
		if acc == nil {
			acc = term
		} else {
			evaluator.Add(acc, term, acc)
		}
	}
	return acc
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
