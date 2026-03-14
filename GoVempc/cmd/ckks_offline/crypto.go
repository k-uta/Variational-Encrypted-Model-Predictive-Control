package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"gonum.org/v1/gonum/mat"
)

// saveCryptoMaterial writes the CKKS parameters and keys used by the online step.
func saveCryptoMaterial(dir string, params ckks.Parameters, sk *rlwe.SecretKey, pk *rlwe.PublicKey, rlk *rlwe.RelinearizationKey) {
	if data, err := params.MarshalBinary(); err == nil {
		_ = writeBinary(filepath.Join(dir, "params.bin"), data)
	}
	if data, err := sk.MarshalBinary(); err == nil {
		_ = writeBinary(filepath.Join(dir, "sk.bin"), data)
	}
	if data, err := pk.MarshalBinary(); err == nil {
		_ = writeBinary(filepath.Join(dir, "pk.bin"), data)
	}
	if data, err := rlk.MarshalBinary(); err == nil {
		_ = writeBinary(filepath.Join(dir, "rlk.bin"), data)
	}
}

// writeCiphertext marshals a ciphertext and writes it to disk.
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

// writeBinary writes raw bytes to disk.
func writeBinary(path string, data []byte) error {
	return os.WriteFile(path, data, 0o644)
}

// workerCachePath returns the per-worker cache filename for one timestep.
func workerCachePath(baseDir, prefix string, t int) string {
	return filepath.Join(baseDir, fmt.Sprintf("%s_%04d.bin", prefix, t))
}

// encryptCyclicDiagonalsSquare encrypts the cyclic diagonals of a square matrix.
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

// encryptCyclicDiagonalsGamma encrypts Gamma in padded cyclic-diagonal form.
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

// encMatVec applies an encrypted diagonal matrix to one plaintext vector sample.
func encMatVec(
	ctDiags []*rlwe.Ciphertext,
	xi []float64,
	vecLen int,
	encoder *ckks.Encoder,
	evaluator *ckks.Evaluator,
	scratch *encMatVecScratch,
) *rlwe.Ciphertext {
	var acc *rlwe.Ciphertext
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

// encMatVecScratch holds reusable buffers for encrypted matrix-vector products.
type encMatVecScratch struct {
	vals []complex128
	rot  []float64
	pt   *rlwe.Plaintext
}

// newEncMatVecScratch allocates the buffers reused by encMatVec.
func newEncMatVecScratch(vecLen, slots int, params ckks.Parameters) *encMatVecScratch {
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	pt.Scale = params.DefaultScale()
	return &encMatVecScratch{
		vals: make([]complex128, slots),
		rot:  make([]float64, vecLen),
		pt:   pt,
	}
}

// rotateVectorInto writes a cyclic rotation of vec into dst.
func rotateVectorInto(dst, vec []float64, k int) {
	n := len(vec)
	if n == 0 {
		return
	}
	k = mod(k, n)
	copy(dst, vec[k:])
	copy(dst[n-k:], vec[:k])
}

// makeMaskPlaintext builds a mask that keeps the active slot range.
func makeMaskPlaintext(slotWidth, slots int, params ckks.Parameters, encoder *ckks.Encoder) *rlwe.Plaintext {
	vals := make([]complex128, slots)
	for i := 0; i < slotWidth && i < slots; i++ {
		vals[i] = complex(1.0, 0)
	}
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	_ = encoder.Encode(vals, pt)
	return pt
}

// packCiphertexts packs one ciphertext per sample into disjoint slot blocks.
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
