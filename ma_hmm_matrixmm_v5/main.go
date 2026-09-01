package main

import (
	"flag"
	"fmt"
	"math"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
)

type config struct {
	d, k, b       int
	workers       int
	warmups       int
	trials        int
	mode          string
	optimizedOnly bool
}

var defaultMode = "safe"

func main() {
	cfg := config{}
	flag.IntVar(&cfg.d, "d", 64, "square matrix dimension")
	flag.IntVar(&cfg.k, "k", 2, "Ma encoding-space parameter")
	flag.IntVar(&cfg.b, "b", 1, "replication grouping parameter")
	flag.IntVar(&cfg.workers, "workers", min(runtime.NumCPU(), 8), "parallel AlgoC blocks")
	flag.IntVar(&cfg.warmups, "warmups", 1, "paired warmup runs")
	flag.IntVar(&cfg.trials, "trials", 3, "paired timed runs")
	flag.StringVar(&cfg.mode, "mode", defaultMode, "evaluator ownership: safe or unsafe")
	flag.BoolVar(&cfg.optimizedOnly, "optimized-only", false, "skip Ma and measure only PCR/PBP")
	flag.Parse()

	if err := run(cfg); err != nil {
		panic(err)
	}
}

func run(cfg config) error {
	if cfg.d <= 0 || cfg.k <= 0 || cfg.d%cfg.k != 0 || cfg.b != 1 {
		return fmt.Errorf("require d>0, k>0, k divides d, and b=1")
	}
	if cfg.workers < 1 || cfg.trials < 1 || cfg.warmups < 0 {
		return fmt.Errorf("workers/trials must be positive and warmups non-negative")
	}
	if cfg.mode != "safe" && cfg.mode != "unsafe" {
		return fmt.Errorf("mode must be safe or unsafe")
	}

	params, err := hmmParams(cfg.d, cfg.k)
	if err != nil {
		return err
	}
	need := cfg.d * cfg.d * cfg.k
	if need > params.MaxSlots() {
		return fmt.Errorf("need %d slots, context has %d", need, params.MaxSlots())
	}
	m := params.MaxSlots() / need
	packed := need * m

	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	encoder := hefloat.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	level := params.MaxLevel() - 1

	a, b := matrixInputs(cfg.d, packed)
	ctA, err := encryptVector(params, encoder, encryptor, a, level)
	if err != nil {
		return err
	}
	ctB, err := encryptVector(params, encoder, encryptor, b, level)
	if err != nil {
		return err
	}

	keys := rotationKeys(cfg.d, cfg.k, cfg.b)
	rtks := kgen.GenGaloisKeysNew(params.GaloisElements(keys), sk)
	evlKey := rlwe.NewMemEvaluationKeySet(rlk, rtks...)
	baseEval := hefloat.NewEvaluator(params, evlKey)

	maxBlocks := cfg.d / cfg.k
	workers := min(cfg.workers, maxBlocks)
	evals := make([]*hefloat.Evaluator, 3+2*workers)
	if cfg.mode == "safe" {
		evals[0] = baseEval
		for i := 1; i < len(evals); i++ {
			evals[i] = baseEval.ShallowCopy()
		}
	} else {
		poolSize := max(2, min(workers, 8))
		pool := make([]*hefloat.Evaluator, poolSize)
		pool[0] = baseEval
		for i := 1; i < poolSize; i++ {
			pool[i] = baseEval.WithKey(evlKey)
		}
		evals[0], evals[1] = pool[0], pool[1]
		for worker := 0; worker < workers; worker++ {
			evals[2+2*worker] = pool[worker%poolSize]
			evals[3+2*worker] = pool[(worker+1)%poolSize]
		}
		evals[2+2*workers] = pool[0]
	}

	maPT, err := encodeMasks(params, encoder, maskA(cfg.d, cfg.k, m, cfg.b), level)
	if err != nil {
		return err
	}
	mbPT, err := encodeMasks(params, encoder, maskB(cfg.d, cfg.k, m, cfg.b), level)
	if err != nil {
		return err
	}

	runMa := func() (*rlwe.Ciphertext, error) {
		aa, err := algoA(ctA, cfg.d, cfg.k, evals[0])
		if err != nil {
			return nil, err
		}
		bb, err := algoB(ctB, cfg.d, cfg.k, evals[0])
		if err != nil {
			return nil, err
		}
		return algoCSequential(aa, bb, cfg.d, cfg.k, cfg.b, maPT, mbPT, evals[0])
	}
	runOpt := func() (*rlwe.Ciphertext, error) {
		var aa, bb *rlwe.Ciphertext
		var errA, errB error
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			aa, errA = algoA(ctA, cfg.d, cfg.k, evals[0])
		}()
		go func() {
			defer wg.Done()
			bb, errB = algoB(ctB, cfg.d, cfg.k, evals[1])
		}()
		wg.Wait()
		if errA != nil {
			return nil, errA
		}
		if errB != nil {
			return nil, errB
		}
		return algoCParallel(aa, bb, cfg.d, cfg.k, cfg.b, maPT, mbPT, evals[2:], workers)
	}

	for i := 0; i < cfg.warmups; i++ {
		if !cfg.optimizedOnly {
			if _, err = runMa(); err != nil {
				return fmt.Errorf("Ma warmup: %w", err)
			}
		}
		if _, err = runOpt(); err != nil {
			return fmt.Errorf("optimized warmup: %w", err)
		}
	}

	maTimes := make([]float64, cfg.trials)
	optTimes := make([]float64, cfg.trials)
	var maOut, optOut *rlwe.Ciphertext
	for i := 0; i < cfg.trials; i++ {
		if !cfg.optimizedOnly {
			t0 := time.Now()
			if maOut, err = runMa(); err != nil {
				return fmt.Errorf("Ma trial %d: %w", i, err)
			}
			maTimes[i] = milliseconds(t0)
		}

		t0 := time.Now()
		if optOut, err = runOpt(); err != nil {
			return fmt.Errorf("optimized trial %d: %w", i, err)
		}
		optTimes[i] = milliseconds(t0)
	}

	expected := plainProduct(a[:cfg.d*cfg.d], b[:cfg.d*cfg.d], cfg.d)
	optValues, err := decryptVector(encoder, decryptor, optOut, packed)
	if err != nil {
		return err
	}
	maErr := math.NaN()
	if maOut != nil {
		maValues, e := decryptVector(encoder, decryptor, maOut, packed)
		if e != nil {
			return e
		}
		maErr = maxDiff(maValues[:cfg.d*cfg.d], expected)
	}
	optErr := maxDiff(optValues[:cfg.d*cfg.d], expected)
	pairErr := math.NaN()
	if maOut != nil {
		maValues, e := decryptVector(encoder, decryptor, maOut, packed)
		if e != nil {
			return e
		}
		pairErr = maxDiff(maValues, optValues)
	}
	expectedMax := maxAbs(expected)
	tolerance := math.Max(math.Pow(2, -14), expectedMax*math.Pow(2, -15))
	correct := optErr < tolerance
	if !cfg.optimizedOnly {
		correct = maErr < tolerance && optErr < tolerance && pairErr < tolerance
	}

	maMedian, maStd := medianStd(maTimes)
	optMedian, optStd := medianStd(optTimes)
	if cfg.optimizedOnly {
		fmt.Printf("HMM_V5 mode=%s optimized-only d=%d k=%d b=%d m=%d workers=%d PCR_PBP=%.1f±%.1fms OptErr=%.3e Tol=%.3e correct=%t\n",
			cfg.mode, cfg.d, cfg.k, cfg.b, m, workers, optMedian, optStd, optErr, tolerance, correct)
	} else {
		fmt.Printf("HMM_V5 mode=%s d=%d k=%d b=%d m=%d workers=%d MaSeq=%.1f±%.1fms PCR_PBP=%.1f±%.1fms speedup=%.2fx MaErr=%.3e OptErr=%.3e PairErr=%.3e Tol=%.3e correct=%t\n",
			cfg.mode, cfg.d, cfg.k, cfg.b, m, workers, maMedian, maStd, optMedian, optStd,
			maMedian/optMedian, maErr, optErr, pairErr, tolerance, correct)
	}
	if !correct {
		if cfg.mode == "safe" {
			return fmt.Errorf("decrypted result failed correctness gate")
		}
		fmt.Println("WARNING: unsafe mode failed correctness; timing retained because unsafe execution was explicitly requested")
	}
	return nil
}

func hmmParams(d, k int) (hefloat.Parameters, error) {
	need := d * d * k
	if need <= 32768 {
		return hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
			LogN: 15, LogQ: []int{51, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40},
			LogP: []int{50, 50, 50}, LogDefaultScale: 40, RingType: ring.ConjugateInvariant,
		})
	}
	if need <= 65536 {
		return hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
			LogN: 16, LogQ: []int{55, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45},
			LogP: []int{55, 55, 55}, LogDefaultScale: 45, RingType: ring.ConjugateInvariant,
		})
	}
	return hefloat.Parameters{}, fmt.Errorf("d=%d k=%d requires %d slots", d, k, need)
}

func matrixInputs(d, packed int) ([]float64, []float64) {
	a, b := make([]float64, packed), make([]float64, packed)
	for i := 0; i < d*d; i++ {
		a[i] = float64((17*i)%101)/128 + 0.03125
		b[i] = float64((29*i)%89)/128 + 0.0625
	}
	return a, b
}

func encryptVector(params hefloat.Parameters, encoder *hefloat.Encoder, encryptor *rlwe.Encryptor, values []float64, level int) (*rlwe.Ciphertext, error) {
	pt := hefloat.NewPlaintext(params, level)
	if err := encoder.Encode(values, pt); err != nil {
		return nil, err
	}
	return encryptor.EncryptNew(pt)
}

func decryptVector(encoder *hefloat.Encoder, decryptor *rlwe.Decryptor, ct *rlwe.Ciphertext, n int) ([]float64, error) {
	values := make([]float64, n)
	if err := encoder.Decode(decryptor.DecryptNew(ct), values); err != nil {
		return nil, err
	}
	return values, nil
}

func transpose(ct *rlwe.Ciphertext, step, k int, eval *hefloat.Evaluator) error {
	var tail *rlwe.Ciphertext
	for k > 1 {
		if k&1 == 1 {
			r, err := eval.RotateNew(ct, (k-1)*step)
			if err != nil {
				return err
			}
			if tail == nil {
				tail = r
			} else if err = eval.Add(r, tail, tail); err != nil {
				return err
			}
		}
		r, err := eval.RotateNew(ct, step)
		if err != nil {
			return err
		}
		if err = eval.Add(r, ct, ct); err != nil {
			return err
		}
		step, k = step<<1, k>>1
	}
	if tail != nil {
		err := eval.Add(tail, ct, ct)
		return err
	}
	return nil
}

func algoA(ct *rlwe.Ciphertext, d, k int, eval *hefloat.Evaluator) (*rlwe.Ciphertext, error) {
	out := ct.CopyNew()
	return out, transpose(out, 1-d*d, k, eval)
}

func algoB(ct *rlwe.Ciphertext, d, k int, eval *hefloat.Evaluator) (*rlwe.Ciphertext, error) {
	out := ct.CopyNew()
	return out, transpose(out, d-d*d, k, eval)
}

func replicate(inputs map[int]*rlwe.Ciphertext, kk, d, u, group int, masks []*rlwe.Plaintext, eval *hefloat.Evaluator) (*rlwe.Ciphertext, error) {
	var out *rlwe.Ciphertext
	for i, j, t := 0, u*(kk&(group-1)), kk-(kk&(group-1)); i < group; i++ {
		input, ok := inputs[j]
		if !ok || masks[t+i] == nil {
			return nil, fmt.Errorf("missing replication input/mask for block %d", kk)
		}
		if i == 0 {
			var err error
			if out, err = eval.MulNew(input, masks[t+i]); err != nil {
				return nil, err
			}
		} else if err := eval.MulThenAdd(input, masks[t+i], out); err != nil {
			return nil, err
		}
		j -= u
	}
	if err := eval.Rescale(out, out); err != nil {
		return nil, err
	}
	d, kk, u = d/group, (kk<<1)/group, u*group
	for d > 1 {
		r, err := eval.RotateNew(out, u*((kk&2)-1))
		if err != nil {
			return nil, err
		}
		if err = eval.Add(r, out, out); err != nil {
			return nil, err
		}
		u, kk, d = u<<1, kk>>1, d>>1
	}
	return out, nil
}

func productBlock(a, b *rlwe.Ciphertext, block, d, group int, ma, mb []*rlwe.Plaintext, evalA, evalB *hefloat.Evaluator) (*rlwe.Ciphertext, error) {
	aInputs := map[int]*rlwe.Ciphertext{0: a}
	bInputs := map[int]*rlwe.Ciphertext{0: b}
	var ar, br *rlwe.Ciphertext
	var errA, errB error
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		ar, errA = replicate(aInputs, block, d, 1, group, ma, evalA)
	}()
	go func() {
		defer wg.Done()
		br, errB = replicate(bInputs, block, d, d, group, mb, evalB)
	}()
	wg.Wait()
	if errA != nil {
		return nil, errA
	}
	if errB != nil {
		return nil, errB
	}
	return evalA.MulNew(ar, br)
}

func algoCSequential(a, b *rlwe.Ciphertext, d, k, group int, ma, mb []*rlwe.Plaintext, eval *hefloat.Evaluator) (*rlwe.Ciphertext, error) {
	aInputs := map[int]*rlwe.Ciphertext{0: a}
	bInputs := map[int]*rlwe.Ciphertext{0: b}
	var result *rlwe.Ciphertext
	for block := 0; block < d; block += k {
		ar, err := replicate(aInputs, block, d, 1, group, ma, eval)
		if err != nil {
			return nil, err
		}
		br, err := replicate(bInputs, block, d, d, group, mb, eval)
		if err != nil {
			return nil, err
		}
		if result == nil {
			if result, err = eval.MulNew(ar, br); err != nil {
				return nil, err
			}
		} else if err = eval.MulThenAdd(ar, br, result); err != nil {
			return nil, err
		}
	}
	return finishProduct(result, d, k, eval)
}

func algoCParallel(a, b *rlwe.Ciphertext, d, k, group int, ma, mb []*rlwe.Plaintext, evals []*hefloat.Evaluator, workers int) (*rlwe.Ciphertext, error) {
	blocks := make(chan int, d/k)
	for block := 0; block < d; block += k {
		blocks <- block
	}
	close(blocks)
	type blockResult struct {
		ct  *rlwe.Ciphertext
		err error
	}
	results := make(chan blockResult, d/k)
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		evalA, evalB := evals[2*worker], evals[2*worker+1]
		wg.Add(1)
		go func() {
			defer wg.Done()
			for block := range blocks {
				ct, err := productBlock(a, b, block, d, group, ma, mb, evalA, evalB)
				results <- blockResult{ct: ct, err: err}
				if err != nil {
					return
				}
			}
		}()
	}
	go func() {
		wg.Wait()
		close(results)
	}()

	accEval := evals[2*workers]
	var result *rlwe.Ciphertext
	for item := range results {
		if item.err != nil {
			return nil, item.err
		}
		if result == nil {
			result = item.ct
		} else if err := accEval.Add(result, item.ct, result); err != nil {
			return nil, err
		}
	}
	return finishProduct(result, d, k, accEval)
}

func finishProduct(result *rlwe.Ciphertext, d, k int, eval *hefloat.Evaluator) (*rlwe.Ciphertext, error) {
	if result == nil {
		return nil, fmt.Errorf("empty product")
	}
	if err := eval.Relinearize(result, result); err != nil {
		return nil, err
	}
	if err := eval.Rescale(result, result); err != nil {
		return nil, err
	}
	for step, end := d*d, k*d*d; step < end; step <<= 1 {
		r, err := eval.RotateNew(result, step)
		if err != nil {
			return nil, err
		}
		if err = eval.Add(result, r, result); err != nil {
			return nil, err
		}
	}
	return result, nil
}

func rotationKeys(d, k, group int) []int {
	set := map[int]bool{}
	for width := k; width > 1; width >>= 1 {
		if width&1 == 1 {
			set[(width-1)*(1-d*d)] = true
			set[(width-1)*(d-d*d)] = true
			width--
		}
		set[(width/2)*(1-d*d)] = true
		set[(width/2)*(d-d*d)] = true
	}
	for step := group; step < d; step <<= 1 {
		set[step], set[-step] = true, true
		set[step*d], set[-step*d] = true, true
	}
	for step := d * d; step < k*d*d; step <<= 1 {
		set[step] = true
	}
	keys := make([]int, 0, len(set))
	for key := range set {
		if key != 0 {
			keys = append(keys, key)
		}
	}
	sort.Ints(keys)
	return keys
}

func maskA(d, k, m, group int) [][]float64 {
	masks := make([][]float64, d)
	for i := 0; i < d/k; i++ {
		for j := 0; j < group && j < k; j++ {
			masks[i*k+j] = make([]float64, d*d*k*m)
			for l := 0; l < d*k*m; l++ {
				masks[i*k+j][l*d+i*k+j] = 1
			}
		}
	}
	return masks
}

func maskB(d, k, m, group int) [][]float64 {
	masks := make([][]float64, d)
	for i := 0; i < d/k; i++ {
		for j := 0; j < group && j < k; j++ {
			masks[i*k+j] = make([]float64, d*d*k*m)
			for l := 0; l < k*m; l++ {
				for col := 0; col < d; col++ {
					masks[i*k+j][l*d*d+(i*k+j)*d+col] = 1
				}
			}
		}
	}
	return masks
}

func encodeMasks(params hefloat.Parameters, encoder *hefloat.Encoder, masks [][]float64, level int) ([]*rlwe.Plaintext, error) {
	out := make([]*rlwe.Plaintext, len(masks))
	for i, mask := range masks {
		if mask == nil {
			continue
		}
		out[i] = hefloat.NewPlaintext(params, level)
		if err := encoder.Encode(mask, out[i]); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func plainProduct(a, b []float64, d int) []float64 {
	out := make([]float64, d*d)
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			for k := 0; k < d; k++ {
				out[i*d+j] += a[i*d+k] * b[k*d+j]
			}
		}
	}
	return out
}

func maxDiff(a, b []float64) float64 {
	err := 0.0
	for i := range a {
		err = math.Max(err, math.Abs(a[i]-b[i]))
	}
	return err
}

func maxAbs(values []float64) float64 {
	max := 0.0
	for _, value := range values {
		max = math.Max(max, math.Abs(value))
	}
	return max
}

func milliseconds(start time.Time) float64 {
	return float64(time.Since(start).Microseconds()) / 1000
}

func medianStd(values []float64) (median, std float64) {
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	if len(sorted)&1 == 1 {
		median = sorted[len(sorted)/2]
	} else {
		median = (sorted[len(sorted)/2-1] + sorted[len(sorted)/2]) / 2
	}
	mean := 0.0
	for _, value := range values {
		mean += value
	}
	mean /= float64(len(values))
	for _, value := range values {
		std += (value - mean) * (value - mean)
	}
	std = math.Sqrt(std / float64(len(values)))
	return
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
