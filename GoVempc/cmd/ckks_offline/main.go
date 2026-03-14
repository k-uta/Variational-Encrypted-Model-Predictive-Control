package main

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
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
	// Load config
	cfgPath := filepath.Join("output", "ckks_config.json")
	cfg, _ := ckksconfig.Load(cfgPath)

	m, l, g := cfg.Mass, cfg.Length, cfg.Gravity
	dt := cfg.DT
	N := cfg.N
	Ac, Bc := examples.LinearizedInvertedPendulumContinuous(m, l, g)
	A, B := examples.Discretize(Ac, Bc, dt)

	// Build the condensed MPC problem.
	n, _ := A.Dims()
	_, mIn := B.Dims()
	Q := diagDense(cfg.QDiag)
	R := diagDense(cfg.RDiag)
	Qf := mat.NewDense(n, n, nil)
	Qf.Scale(cfg.QfScale, Q)

	// Constraints
	xBound := []float64{cfg.ThetaMax, cfg.OmegaMax}
	Gx := stackIdentity(n)
	hx := append(append([]float64{}, xBound...), xBound...)
	Gu := stackIdentity(mIn)
	hu := []float64{cfg.UMax, cfg.UMax}

	// Compact QP representation
	mpc := core.NewMPCProblem(A, B, Q, R, Qf, N)
	G, hOfX0 := mpc.BuildConstraintMatrices(Gx, hx, Gu, hu)
	penalty := core.NewConstraintPenalty(G, hOfX0, "indicator")

	// Variational parameters
	sigma0 := cfg.Sigma0
	Sigma0 := scaledIdentity(mpc.StackedInputDim(), sigma0*sigma0)
	variational := core.NewVariationalMPC(mpc, penalty, cfg.LambdaParam, Sigma0)

	// Precompute the plain matrices used by the encrypted protocol.
	LU := variational.LU
	var Gamma mat.Dense
	Gamma.Mul(penalty.G, LU)
	dim, _ := LU.Dims()
	p, _ := Gamma.Dims()

	// Cache sizes.
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

	// Configure CKKS
	if (1 << (cfg.LogN - 1)) < slotsNeeded {
		panic("logN too small for required slots; increase logN or reduce K")
	}
	paramsLit := ckks.ParametersLiteral{
		LogN:            cfg.LogN,
		LogQ:            cfg.LogQ,
		LogP:            cfg.LogP,
		LogDefaultScale: cfg.LogDefaultScale,
	}
	params, _ := ckks.NewParametersFromLiteral(paramsLit)
	slots := params.MaxSlots()
	if dim > slots || p > slots {
		panic("dim or p exceeds CKKS slots; increase logN or reduce N")
	}
	if slots < slotWidth*KChunk {
		panic("not enough slots to pack K_chunk samples; increase logN or reduce K or nWorkers")
	}

	cacheDir := filepath.Join("output", "ckks_cache")
	cryptoDir := filepath.Join(cacheDir, "crypto")
	_ = os.MkdirAll(cryptoDir, 0o755)

	// Generate keys and save the material the online step will reload.
	kgen := ckks.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	rlk := kgen.GenRelinearizationKeyNew(sk)
	saveCryptoMaterial(cryptoDir, params, sk, pk, rlk)

	rotations := collectRotations(KChunk, dim, p, slots)
	gks := make([]*rlwe.GaloisKey, 0, len(rotations))
	for _, rot := range rotations {
		gks = append(gks, kgen.GenGaloisKeyNew(params.GaloisElementForRotation(rot), sk))
	}
	evalKeys := rlwe.NewMemEvaluationKeySet(rlk, gks...)

	encoder := ckks.NewEncoder(params)
	encryptor := ckks.NewEncryptor(params, pk)

	// Client offline preprocessing.
	clientStart := time.Now()
	encLU := encryptCyclicDiagonalsSquare(LU, dim, slots, encoder, encryptor, params)
	encGamma := encryptCyclicDiagonalsGamma(&Gamma, p, dim, slots, encoder, encryptor, params)
	clientMs := time.Since(clientStart).Seconds() * 1000.0

	fmt.Printf("L_U dims: %dx%d\n", dim, dim)
	fmt.Printf("Gamma dims: %dx%d\n", p, dim)
	fmt.Printf("CKKS slots: %d, K: %d, nWorkers: %d, K_chunk: %d, T: %d\n", slots, K, nWorkers, KChunk, T)

	// Cloud offline cache generation.
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
	fmt.Printf("Client offline time: %.3f ms\n", clientMs)
	fmt.Printf("Cloud offline time: %.3f ms\n", cloudMs)
}
