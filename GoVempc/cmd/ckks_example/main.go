package main

import (
	"fmt"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func main() {
	paramsLit := ckks.ParametersLiteral{
		// Speed-first demo parameters (security not targeted)
		LogN: 6,
		// Use 30-bit primes and scale 2^30 for faster runtime.
		LogQ: []int{30, 30, 30, 30},
		// Optional P primes for relinearization keys.
		LogP:            []int{30},
		LogDefaultScale: 30,
	}

	params, _ := ckks.NewParametersFromLiteral(paramsLit)

	logN := params.LogN()
	ringN := 1 << logN
	logQP := params.LogQP()
	fmt.Printf("Parameters: %+v\n", paramsLit)
	fmt.Printf("LogN: %d, Ring dimension N: %d\n", logN, ringN)
	fmt.Printf("LogQP (bits): %.0f\n", logQP)
	fmt.Printf("Max multiplicative depth (MaxLevel): %d\n", params.MaxLevel())
	fmt.Printf("Default scale (log2): %d\n", params.LogDefaultScale())
	fmt.Printf("Security note: parameters are tuned for accuracy, not 128-bit security.\n")

	// Key generation:
	// - sk/pk for encryption/decryption
	// - rlk for relinearization after multiplication
	// - galois key for slot rotation
	kgen := ckks.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()

	rlk := kgen.GenRelinearizationKeyNew(sk)
	rot1 := params.GaloisElementForRotation(1)
	gk := kgen.GenGaloisKeyNew(rot1, sk)
	EVK := rlwe.NewMemEvaluationKeySet(rlk, gk)

	encoder := ckks.NewEncoder(params)
	encryptor := ckks.NewEncryptor(params, pk)
	decryptor := ckks.NewDecryptor(params, sk)
	evaluator := ckks.NewEvaluator(params, EVK)

	// CKKS packs complex values into slots (N/2 slots for Standard ring).
	valuesA := []complex128{1 + 0i, 2 + 0i, 3 + 0i, 4 + 0i}
	valuesB := []complex128{10 + 0i, 20 + 0i, 30 + 0i, 40 + 0i}

	ptA := ckks.NewPlaintext(params, params.MaxLevel())
	ptB := ckks.NewPlaintext(params, params.MaxLevel())

	// Encoding maps slot values into a plaintext polynomial at the given scale.
	_ = encoder.Encode(valuesA, ptA)
	_ = encoder.Encode(valuesB, ptB)

	// Decryption + decoding returns approximate values (CKKS is approximate).
	decode := func(ct *rlwe.Ciphertext, n int) []complex128 {
		pt := decryptor.DecryptNew(ct)
		out := make([]complex128, n)
		_ = encoder.Decode(pt, out)
		return out
	}

	// Timing over multiple runs (20 iterations).
	runs := 20
	var encDur, addDur, mulDur, rotDur, decDur time.Duration
	var lastAdd, lastMul, lastRot *rlwe.Ciphertext

	for i := 0; i < runs; i++ {
		start := time.Now()
		ctA, _ := encryptor.EncryptNew(ptA)
		encDur += time.Since(start)

		start = time.Now()
		ctB, _ := encryptor.EncryptNew(ptB)
		encDur += time.Since(start)

		// Homomorphic addition is scale-preserving.
		start = time.Now()
		ctAdd, _ := evaluator.AddNew(ctA, ctB)
		addDur += time.Since(start)

		// Multiplication requires relinearization and rescaling to control scale growth.
		start = time.Now()
		ctMul, _ := evaluator.MulRelinNew(ctA, ctB)
		_ = evaluator.Rescale(ctMul, ctMul)
		mulDur += time.Since(start)

		// Rotation shifts slots; requires a Galois key for the rotation index.
		start = time.Now()
		ctRot, _ := evaluator.RotateNew(ctA, 1)
		rotDur += time.Since(start)

		start = time.Now()
		_ = decode(ctAdd, len(valuesA))
		_ = decode(ctMul, len(valuesA))
		_ = decode(ctRot, len(valuesA))
		decDur += time.Since(start)

		lastAdd, lastMul, lastRot = ctAdd, ctMul, ctRot
	}

	fmt.Printf("Mean encrypt (per ct): %s\n", encDur/time.Duration(2*runs))
	fmt.Printf("Mean add: %s\n", addDur/time.Duration(runs))
	fmt.Printf("Mean mul+relin+rescale: %s\n", mulDur/time.Duration(runs))
	fmt.Printf("Mean rotate: %s\n", rotDur/time.Duration(runs))
	fmt.Printf("Mean decrypt+decode (per ct): %s\n", decDur/time.Duration(3*runs))

	fmt.Println("Add:", decode(lastAdd, len(valuesA)))
	fmt.Println("Mul:", decode(lastMul, len(valuesA)))
	fmt.Println("Rot:", decode(lastRot, len(valuesA)))
}
