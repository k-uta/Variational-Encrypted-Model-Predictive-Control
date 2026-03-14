package main

import (
	"os"
	"path/filepath"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// inputEncryptor reuses buffers for tiled client-side input encryption.
type inputEncryptor struct {
	encoder      *ckks.Encoder
	encryptor    *rlwe.Encryptor
	pt           *rlwe.Plaintext
	vals         []complex128
	slots        int
	params       ckks.Parameters
	defaultScale rlwe.Scale
}

// newInputEncryptor allocates the reusable encoder state for one input stream.
func newInputEncryptor(params ckks.Parameters, pk *rlwe.PublicKey, slots int) *inputEncryptor {
	return &inputEncryptor{
		encoder:      ckks.NewEncoder(params),
		encryptor:    ckks.NewEncryptor(params, pk),
		vals:         make([]complex128, slots),
		slots:        slots,
		params:       params,
		defaultScale: params.DefaultScale(),
	}
}

// EncryptTiled repeats one vector into per-sample slot blocks and encrypts it.
func (e *inputEncryptor) EncryptTiled(vec []float64, slotWidth int, kChunk int, level int) *rlwe.Ciphertext {
	if e.pt == nil || e.pt.Level() != level {
		e.pt = ckks.NewPlaintext(e.params, level)
	}
	return tileEncryptVector(vec, slotWidth, kChunk, e.slots, e.encoder, e.encryptor, e.pt, e.vals, e.defaultScale)
}

// loadCrypto reloads the CKKS parameters and keys produced by ckks_offline.
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

// tileEncryptVector repeats one vector into per-sample slot blocks and encrypts it.
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

// makeFirstSlotMaskPlaintext keeps the first slot of each packed block.
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

// sumWithinBlocks collapses each block to its first slot.
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

// evalPoly evaluates a power-basis polynomial with Horner's rule.
func evalPoly(ctX *rlwe.Ciphertext, coeffs []float64, eval *ckks.Evaluator) *rlwe.Ciphertext {
	if len(coeffs) == 0 {
		return ctX.CopyNew()
	}
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
