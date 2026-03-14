package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// onlineWorker owns one chunk of the encrypted online computation.
type onlineWorker struct {
	cache        *cipherCache
	encoder      *ckks.Encoder
	decryptor    *rlwe.Decryptor
	evaluator    *ckks.Evaluator
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
	cloudMs      float64
	params       ckks.Parameters
}

// newOnlineWorker allocates one worker and its reusable buffers.
func newOnlineWorker(
	cacheDir string,
	kChunk int,
	dim int,
	p int,
	slots int,
	params ckks.Parameters,
	sk *rlwe.SecretKey,
	evalKeys *rlwe.MemEvaluationKeySet,
	polyZ []float64,
	tauS float64,
) *onlineWorker {
	return &onlineWorker{
		cache:        newCipherCache(cacheDir),
		encoder:      ckks.NewEncoder(params),
		decryptor:    ckks.NewDecryptor(params, sk),
		evaluator:    ckks.NewEvaluator(params, evalKeys),
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
		params:       params,
	}
}

// runCycle executes one online control iteration for a worker chunk.
func (w *onlineWorker) runCycle(t int, ctMU *rlwe.Ciphertext, ctB *rlwe.Ciphertext) error {
	ctLu := w.cache.Load("ct_Lu", t)
	ctGamma := w.cache.Load("ct_Gamma", t)

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

// cipherCache memoizes offline cache ciphertexts on first load.
type cipherCache struct {
	dir   string
	ctMap map[string]*rlwe.Ciphertext
}

// newCipherCache constructs the cache wrapper for one directory.
func newCipherCache(dir string) *cipherCache {
	return &cipherCache{dir: dir, ctMap: make(map[string]*rlwe.Ciphertext)}
}

// Load returns one cached ciphertext, reusing timestep 0 if later files are missing.
func (c *cipherCache) Load(prefix string, t int) *rlwe.Ciphertext {
	origKey := fmt.Sprintf("%s_%04d", prefix, t)
	if ct, ok := c.ctMap[origKey]; ok {
		return ct
	}
	key := origKey
	data, err := os.ReadFile(filepath.Join(c.dir, key+".bin"))
	if err != nil {
		if !os.IsNotExist(err) || t == 0 {
			panic(err)
		}
		key0 := fmt.Sprintf("%s_%04d", prefix, 0)
		if ct, ok := c.ctMap[key0]; ok {
			c.ctMap[origKey] = ct
			return ct
		}
		key = key0
		data, err = os.ReadFile(filepath.Join(c.dir, key+".bin"))
		if err != nil {
			panic(err)
		}
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

// workerCacheDir returns the cache directory used by one worker.
func workerCacheDir(baseDir string, workerID int, nWorkers int) string {
	if nWorkers <= 1 {
		return baseDir
	}
	return filepath.Join(baseDir, fmt.Sprintf("worker_%d", workerID))
}
