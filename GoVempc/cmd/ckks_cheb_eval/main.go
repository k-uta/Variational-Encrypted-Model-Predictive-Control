package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type Input struct {
	Bound  float64   `json:"bound"`
	Coeffs []float64 `json:"coeffs"` // power basis coefficients for z
	Z      []float64 `json:"z"`
}

type Output struct {
	Z        []float64 `json:"z"`
	Y        []float64 `json:"y"`
	TimeEnc  float64   `json:"time_enc_ms"`
	TimeEval float64   `json:"time_eval_ms"`
	TimeDec  float64   `json:"time_dec_ms"`
}

func main() {
	inputPath := filepath.Join("output", "cheb_ckks_input.json")
	outputPath := filepath.Join("output", "cheb_ckks_output.json")

	data, _ := os.ReadFile(inputPath)
	var in Input
	_ = json.Unmarshal(data, &in)

	// CKKS parameters (very relaxed security for speed)
	paramsLit := ckks.ParametersLiteral{
		LogN: 6,
		// Use 30-bit primes and scale 2^30 for faster runtime.
		LogQ: []int{30, 30, 30, 30},
		LogP: []int{20},
		// LogDefaultScale must be <= 64 for PREC64
		LogDefaultScale: 30,
	}
	params, _ := ckks.NewParametersFromLiteral(paramsLit)

	slots := params.MaxSlots()
	if len(in.Z) > slots {
		in.Z = in.Z[:slots]
	}

	// Prepare plaintext values: treat each z as a real slot.
	vals := make([]complex128, len(in.Z))
	for i, z := range in.Z {
		vals[i] = complex(z, 0)
	}

	kgen := ckks.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()

	rlk := kgen.GenRelinearizationKeyNew(sk)
	evalKeys := rlwe.NewMemEvaluationKeySet(rlk)

	encoder := ckks.NewEncoder(params)
	encryptor := ckks.NewEncryptor(params, pk)
	decryptor := ckks.NewDecryptor(params, sk)
	evaluator := ckks.NewEvaluator(params, evalKeys)

	pt := ckks.NewPlaintext(params, params.MaxLevel())
	// Encoding maps slot values to a plaintext polynomial at scale 2^LogDefaultScale.
	_ = encoder.Encode(vals, pt)

	startEnc := time.Now()
	ctX, _ := encryptor.EncryptNew(pt)
	encMs := time.Since(startEnc).Seconds() * 1000.0

	// Horner evaluation: y = (((c_d)x + c_{d-1})x + ... + c_0)
	// We keep ctY and ctX at compatible levels/scales via DropLevel + Rescale.
	startEval := time.Now()

	// Initialize y with constant coeffs[d] as an encrypted constant ciphertext.
	constVals := make([]complex128, len(in.Z))
	for i := range constVals {
		constVals[i] = complex(in.Coeffs[len(in.Coeffs)-1], 0)
	}
	ptConst := ckks.NewPlaintext(params, ctX.Level())
	_ = encoder.Encode(constVals, ptConst)
	ctY, _ := encryptor.EncryptNew(ptConst)

	for k := len(in.Coeffs) - 2; k >= 0; k-- {
		ctXk := ctX
		// Align levels: ctY may be at a lower level after rescale, so drop ctX as needed.
		if ctXk.Level() > ctY.Level() {
			ctXk = evaluator.DropLevelNew(ctX, ctX.Level()-ctY.Level())
		}

		// Multiply + relinearize + rescale keeps scale roughly constant.
		ctY, _ = evaluator.MulRelinNew(ctY, ctXk)
		_ = evaluator.Rescale(ctY, ctY)

		// Add next coefficient as a plaintext scalar (scale preserved by evaluator.Add).
		_ = evaluator.Add(ctY, in.Coeffs[k], ctY)
	}

	evalMs := time.Since(startEval).Seconds() * 1000.0

	startDec := time.Now()
	ptOut := decryptor.DecryptNew(ctY)
	outVals := make([]complex128, len(in.Z))
	_ = encoder.Decode(ptOut, outVals)
	decMs := time.Since(startDec).Seconds() * 1000.0

	out := Output{Z: in.Z, Y: make([]float64, len(in.Z)), TimeEnc: encMs, TimeEval: evalMs, TimeDec: decMs}
	for i := range outVals {
		out.Y[i] = real(outVals[i])
	}

	outData, _ := json.MarshalIndent(out, "", "  ")
	_ = os.WriteFile(outputPath, outData, 0o644)

	fmt.Printf("Wrote %s\n", outputPath)
	fmt.Printf("Enc: %.3f ms, Eval: %.3f ms, Dec: %.3f ms\n", encMs, evalMs, decMs)
}
