package main

import (
	"flag"
	"fmt"
	"math"
	"runtime"
	"sort"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
)

type config struct {
	d, ell          int
	calls           int
	maxFuse         int
	n1, workers     int
	warmups, trials int
	sweep           bool
}

type measurement struct {
	times    []float64
	median   float64
	std      float64
	maxError float64
	endLevel int
}

func parameters() (hefloat.Parameters, error) {
	return hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{55, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{55, 55, 55},
		LogDefaultScale: 46,
		RingType:        ring.ConjugateInvariant,
	})
}

func main() {
	cfg := config{}
	flag.IntVar(&cfg.d, "d", 64, "square matrix dimension (power of two, 16..256)")
	flag.IntVar(&cfg.ell, "ell", 4, "decomposition depth used by both methods")
	flag.BoolVar(&cfg.sweep, "sweep", false, "benchmark every schedule in {1,2,3}^calls")
	flag.IntVar(&cfg.calls, "calls", 2, "right-factor HLT calls in schedule sweep")
	flag.IntVar(&cfg.maxFuse, "max-fuse", 3, "maximum consecutive right factors per fused HLT")
	flag.IntVar(&cfg.n1, "n1", 0, "BSGS baby-step size (0 selects 8 for d<=128, 16 for d=256)")
	flag.IntVar(&cfg.workers, "workers", minInt(runtime.NumCPU(), 8), "private QP GadgetProduct evaluators")
	flag.IntVar(&cfg.warmups, "warmups", 1, "paired untimed warmups")
	flag.IntVar(&cfg.trials, "trials", 3, "paired timed trials")
	flag.Parse()
	if err := run(cfg); err != nil {
		panic(err)
	}
}

func validate(cfg *config) error {
	if cfg.d < 16 || cfg.d > 256 || cfg.d&(cfg.d-1) != 0 {
		return fmt.Errorf("d must be a power of two in [16,256]")
	}
	if cfg.ell < 1 || cfg.maxFuse < 1 || cfg.workers < 1 || cfg.trials < 1 || cfg.warmups < 0 {
		return fmt.Errorf("ell, max-fuse, workers and trials must be positive; warmups must be non-negative")
	}
	if cfg.calls < 1 || cfg.calls > 4 {
		return fmt.Errorf("calls must be in [1,4]")
	}
	if cfg.n1 == 0 {
		cfg.n1 = 8
		if cfg.d == 256 {
			cfg.n1 = 16
		}
	}
	if cfg.n1 < 1 {
		return fmt.Errorf("n1 must be positive")
	}
	return nil
}

func run(cfg config) error {
	if err := validate(&cfg); err != nil {
		return err
	}
	if cfg.sweep {
		return runScheduleSweep(cfg)
	}
	depth := fullDepth(cfg.d)
	if cfg.ell > depth {
		return fmt.Errorf("ell=%d exceeds full HMT depth %d for d=%d", cfg.ell, depth, cfg.d)
	}

	params, err := parameters()
	if err != nil {
		return err
	}
	encoder := hefloat.NewEncoder(params)
	maPlan, err := buildPlan(params, encoder, cfg.d, cfg.ell, cfg.n1, false, 1, cfg.ell)
	if err != nil {
		return fmt.Errorf("build Ma plan: %w", err)
	}
	optPlan, err := buildPlan(params, encoder, cfg.d, cfg.ell, cfg.n1, true, cfg.maxFuse, cfg.ell)
	if err != nil {
		return fmt.Errorf("build optimized plan: %w", err)
	}

	rotations := rotationUnion(maPlan, optPlan)
	kgen := rlwe.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	rtks := kgen.GenGaloisKeysNew(func() []uint64 {
		galEls := make([]uint64, len(rotations))
		for i, rotation := range rotations {
			galEls[i] = params.GaloisElement(rotation)
		}
		return galEls
	}(), sk)
	eval := hefloat.NewEvaluator(params, rlwe.NewMemEvaluationKeySet(nil, rtks...))
	encryptor := hefloat.NewEncryptor(params, pk)
	decryptor := hefloat.NewDecryptor(params, sk)

	n := cfg.d * cfg.d
	x := make([]float64, n)
	for i := range x {
		x[i] = float64((i*17)%101)/128 + 0.03125
	}
	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	logCols := 0
	for 1<<logCols < n {
		logCols++
	}
	pt.LogDimensions = ring.Dimensions{Rows: 0, Cols: logCols}
	if err := encoder.Encode(x, pt); err != nil {
		return err
	}
	ct0, err := encryptor.EncryptNew(pt)
	if err != nil {
		return err
	}
	expected := transposePlain(x, cfg.d)

	maEval := eval.ShallowCopy()
	optEval := eval.ShallowCopy()
	pool := makeEvaluatorPool(optEval, optPlan, cfg.workers)

	fmt.Printf("HMT_TRANSPOSE_V5 d=%d logN=%d active_slots=%d n1=%d workers=%d keys=%d\n", cfg.d, params.LogN(), n, cfg.n1, len(pool), len(rotations))
	fmt.Printf("  Ma:  ell=%d schedule=%v right_HLT=%d residual_diags=%d branches=%d\n", maPlan.ell, maPlan.schedule, maPlan.rightCalls, maPlan.stages[len(maPlan.stages)-1].diags, len(maPlan.branches))
	fmt.Printf("  Opt: ell=%d schedule=%v right_HLT=%d residual_diags=%d branches=%d QP=parallel/global-ModDown\n", optPlan.ell, optPlan.schedule, optPlan.rightCalls, optPlan.stages[len(optPlan.stages)-1].diags, len(optPlan.branches))

	for i := 0; i < cfg.warmups; i++ {
		if _, err := evalChainStock(params, maEval, ct0, maPlan); err != nil {
			return fmt.Errorf("Ma warmup: %w", err)
		}
		if _, err := evalChainQP(params, optEval, pool, ct0, optPlan); err != nil {
			return fmt.Errorf("optimized warmup: %w", err)
		}
	}

	maTimes := make([]float64, cfg.trials)
	optTimes := make([]float64, cfg.trials)
	var maOut, optOut *rlwe.Ciphertext
	for trial := 0; trial < cfg.trials; trial++ {
		if trial%2 == 0 {
			maOut, maTimes[trial], err = timeStock(params, maEval, ct0, maPlan)
			if err == nil {
				optOut, optTimes[trial], err = timeQP(params, optEval, pool, ct0, optPlan)
			}
		} else {
			optOut, optTimes[trial], err = timeQP(params, optEval, pool, ct0, optPlan)
			if err == nil {
				maOut, maTimes[trial], err = timeStock(params, maEval, ct0, maPlan)
			}
		}
		if err != nil {
			return fmt.Errorf("trial %d: %w", trial, err)
		}
	}

	ma := summarize(maTimes, maxDiff(decode(encoder, decryptor, maOut, n), expected), maOut.Level())
	opt := summarize(optTimes, maxDiff(decode(encoder, decryptor, optOut, n), expected), optOut.Level())
	optStockOut, err := evalChainStock(params, eval.ShallowCopy(), ct0, optPlan)
	if err != nil {
		return fmt.Errorf("fused stock correctness audit: %w", err)
	}
	optStockErr := maxDiff(decode(encoder, decryptor, optStockOut, n), expected)
	tolerance := math.Pow(2, -20)
	correct := ma.maxError < tolerance && opt.maxError < tolerance && optStockErr < tolerance
	fmt.Printf("  RESULT Ma=%.1f±%.1fms Opt=%.1f±%.1fms speedup=%.2fx MaErr=%.3e OptErr=%.3e FusedStockErr=%.3e levels=%d/%d correct=%t\n", ma.median, ma.std, opt.median, opt.std, ma.median/opt.median, ma.maxError, opt.maxError, optStockErr, ma.endLevel, opt.endLevel, correct)
	if !correct {
		return fmt.Errorf("correctness gate failed: require both errors < 2^-20 (%.3e)", tolerance)
	}
	return nil
}

func makeEvaluatorPool(eval *hefloat.Evaluator, plan transformPlan, requested int) []*hefloat.Evaluator {
	maxGroups := 1
	for _, st := range append(append([]stage{}, plan.stages...), plan.branches...) {
		if len(st.groups) > maxGroups {
			maxGroups = len(st.groups)
		}
	}
	count := minInt(requested, maxGroups)
	pool := make([]*hefloat.Evaluator, count)
	for i := range pool {
		pool[i] = eval.ShallowCopy()
	}
	return pool
}

func evalChainStock(params hefloat.Parameters, eval *hefloat.Evaluator, ct0 *rlwe.Ciphertext, plan transformPlan) (*rlwe.Ciphertext, error) {
	ct := ct0.CopyNew()
	for i, st := range plan.stages {
		preRotated, err := prepareHLT(eval, ct, st.lt)
		if err != nil {
			return nil, err
		}
		ct, err = evalStock(eval, ct, st.lt, preRotated)
		if err != nil {
			return nil, err
		}
		if i+1 < len(plan.stages) {
			if err := eval.RescaleTo(ct, params.DefaultScale(), ct); err != nil {
				return nil, err
			}
		}
	}
	for _, branch := range plan.branches {
		preRotated, err := prepareHLT(eval, ct0, branch.lt)
		if err != nil {
			return nil, err
		}
		part, err := evalStock(eval, ct0, branch.lt, preRotated)
		if err != nil {
			return nil, err
		}
		if part.Level() > ct.Level() {
			eval.DropLevel(part, part.Level()-ct.Level())
		}
		ct, err = eval.AddNew(ct, part)
		if err != nil {
			return nil, err
		}
		ct.Scale = part.Scale
	}
	if err := eval.RescaleTo(ct, params.DefaultScale(), ct); err != nil {
		return nil, err
	}
	return ct, nil
}

func evalChainQP(params hefloat.Parameters, eval *hefloat.Evaluator, pool []*hefloat.Evaluator, ct0 *rlwe.Ciphertext, plan transformPlan) (*rlwe.Ciphertext, error) {
	ct := ct0.CopyNew()
	for i, st := range plan.stages {
		preRotated, err := prepareHLT(eval, ct, st.lt)
		if err != nil {
			return nil, err
		}
		ct, err = evalGroupsQP(eval, ct, st.groups, preRotated, pool)
		if err != nil {
			return nil, err
		}
		if i+1 < len(plan.stages) {
			if err := eval.RescaleTo(ct, params.DefaultScale(), ct); err != nil {
				return nil, err
			}
		}
	}
	for _, branch := range plan.branches {
		preRotated, err := prepareHLT(eval, ct0, branch.lt)
		if err != nil {
			return nil, err
		}
		part, err := evalGroupsQP(eval, ct0, branch.groups, preRotated, pool)
		if err != nil {
			return nil, err
		}
		if part.Level() > ct.Level() {
			eval.DropLevel(part, part.Level()-ct.Level())
		}
		ct, err = eval.AddNew(ct, part)
		if err != nil {
			return nil, err
		}
		ct.Scale = part.Scale
	}
	if err := eval.RescaleTo(ct, params.DefaultScale(), ct); err != nil {
		return nil, err
	}
	return ct, nil
}

func timeStock(params hefloat.Parameters, eval *hefloat.Evaluator, ct0 *rlwe.Ciphertext, plan transformPlan) (*rlwe.Ciphertext, float64, error) {
	start := time.Now()
	out, err := evalChainStock(params, eval, ct0, plan)
	return out, float64(time.Since(start).Microseconds()) / 1000, err
}

func timeQP(params hefloat.Parameters, eval *hefloat.Evaluator, pool []*hefloat.Evaluator, ct0 *rlwe.Ciphertext, plan transformPlan) (*rlwe.Ciphertext, float64, error) {
	start := time.Now()
	out, err := evalChainQP(params, eval, pool, ct0, plan)
	return out, float64(time.Since(start).Microseconds()) / 1000, err
}

func transposePlain(x []float64, d int) []float64 {
	y := make([]float64, len(x))
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			y[j*d+i] = x[i*d+j]
		}
	}
	return y
}

func decode(encoder *hefloat.Encoder, decryptor *rlwe.Decryptor, ct *rlwe.Ciphertext, n int) []float64 {
	values := make([]float64, n)
	if err := encoder.Decode(decryptor.DecryptNew(ct), values); err != nil {
		panic(err)
	}
	return values
}

func maxDiff(a, b []float64) float64 {
	max := 0.0
	for i := range a {
		max = math.Max(max, math.Abs(a[i]-b[i]))
	}
	return max
}

func summarize(times []float64, maxError float64, level int) measurement {
	ordered := append([]float64(nil), times...)
	sort.Float64s(ordered)
	median := ordered[len(ordered)/2]
	variance := 0.0
	for _, value := range ordered {
		variance += (value - median) * (value - median)
	}
	return measurement{times: times, median: median, std: math.Sqrt(variance / float64(len(ordered))), maxError: maxError, endLevel: level}
}
