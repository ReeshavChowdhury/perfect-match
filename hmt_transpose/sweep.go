package main

import (
	"fmt"
	"math"
	"runtime"
	"sort"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
)

type sweepResult struct {
	schedule      []int
	effective     []int
	ell           int
	residualDiags int
	measurement   measurement
	fusedStockErr float64
	speedup       float64
}

func enumerateSchedules(calls int) [][]int {
	var schedules [][]int
	current := make([]int, calls)
	var visit func(int)
	visit = func(index int) {
		if index == calls {
			schedules = append(schedules, append([]int(nil), current...))
			return
		}
		for width := 1; width <= 3; width++ {
			current[index] = width
			visit(index + 1)
		}
	}
	visit(0)
	return schedules
}

func runScheduleSweep(cfg config) error {
	depth := fullDepth(cfg.d)
	if cfg.calls > depth {
		return fmt.Errorf("d=%d has full depth %d, fewer than B=%d calls", cfg.d, depth, cfg.calls)
	}
	params, err := parameters()
	if err != nil {
		return err
	}
	encoder := hefloat.NewEncoder(params)
	maPlan, err := buildPlan(params, encoder, cfg.d, cfg.calls, cfg.n1, false, 1, cfg.calls)
	if err != nil {
		return fmt.Errorf("build Ma [1,...,1] plan: %w", err)
	}
	schedules := enumerateSchedules(cfg.calls)
	rotationSet := map[int]bool{}
	addRotations := func(plan transformPlan) {
		for _, rotation := range rotationUnion(plan) {
			rotationSet[rotation] = true
		}
	}
	addRotations(maPlan)
	for _, schedule := range schedules {
		plan, buildErr := buildScheduledPlan(params, encoder, cfg.d, cfg.n1, schedule)
		if buildErr != nil {
			return fmt.Errorf("build schedule %v for rotation union: %w", schedule, buildErr)
		}
		addRotations(plan)
		plan = transformPlan{}
		runtime.GC()
	}
	rotations := make([]int, 0, len(rotationSet))
	for rotation := range rotationSet {
		rotations = append(rotations, rotation)
	}
	sort.Ints(rotations)

	kgen := rlwe.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	rtks := kgen.GenGaloisKeysNew(func() []uint64 {
		galEls := make([]uint64, len(rotations))
		for i, rotation := range rotations {
			galEls[i] = params.GaloisElement(rotation)
		}
		return galEls
	}(), sk)
	baseEval := hefloat.NewEvaluator(params, rlwe.NewMemEvaluationKeySet(nil, rtks...))
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
	tolerance := math.Pow(2, -20)
	expectedEndLevel := params.MaxLevel() - (cfg.calls + 1)

	fmt.Printf("HMT_SCHEDULE_SWEEP_V5 d=%d B=%d candidates=%d logN=%d active_slots=%d n1=%d workers=%d keys=%d warmups=%d trials=%d\n",
		cfg.d, cfg.calls, len(schedules), params.LogN(), n, cfg.n1, cfg.workers, len(rotations), cfg.warmups, cfg.trials)

	maEval := baseEval.ShallowCopy()
	for i := 0; i < cfg.warmups; i++ {
		if _, err := evalChainStock(params, maEval, ct0, maPlan); err != nil {
			return fmt.Errorf("Ma warmup: %w", err)
		}
	}
	maTimes := make([]float64, cfg.trials)
	var maOut *rlwe.Ciphertext
	for trial := 0; trial < cfg.trials; trial++ {
		maOut, maTimes[trial], err = timeStock(params, maEval, ct0, maPlan)
		if err != nil {
			return fmt.Errorf("Ma trial %d: %w", trial, err)
		}
	}
	ma := summarize(maTimes, maxDiff(decode(encoder, decryptor, maOut, n), expected), maOut.Level())
	if ma.maxError >= tolerance || ma.endLevel != expectedEndLevel {
		return fmt.Errorf("Ma correctness failed: error=%.3e level=%d wantLevel=%d", ma.maxError, ma.endLevel, expectedEndLevel)
	}
	fmt.Printf("  MA schedule=%v median=%.1fms std=%.1fms error=%.3e level=%d correct=true\n",
		maPlan.schedule, ma.median, ma.std, ma.maxError, ma.endLevel)

	results := make([]sweepResult, 0, len(schedules))
	for i, schedule := range schedules {
		plan, buildErr := buildScheduledPlan(params, encoder, cfg.d, cfg.n1, schedule)
		if buildErr != nil {
			return fmt.Errorf("rebuild schedule %v: %w", schedule, buildErr)
		}
		eval := baseEval.ShallowCopy()
		pool := makeEvaluatorPool(eval, plan, cfg.workers)
		for warmup := 0; warmup < cfg.warmups; warmup++ {
			if _, err := evalChainQP(params, eval, pool, ct0, plan); err != nil {
				return fmt.Errorf("schedule %v warmup: %w", plan.schedule, err)
			}
		}
		times := make([]float64, cfg.trials)
		var out *rlwe.Ciphertext
		for trial := 0; trial < cfg.trials; trial++ {
			out, times[trial], err = timeQP(params, eval, pool, ct0, plan)
			if err != nil {
				return fmt.Errorf("schedule %v trial %d: %w", plan.schedule, trial, err)
			}
		}
		measured := summarize(times, maxDiff(decode(encoder, decryptor, out, n), expected), out.Level())
		stockOut, err := evalChainStock(params, baseEval.ShallowCopy(), ct0, plan)
		if err != nil {
			return fmt.Errorf("schedule %v fused-stock audit: %w", plan.schedule, err)
		}
		stockErr := maxDiff(decode(encoder, decryptor, stockOut, n), expected)
		correct := measured.maxError < tolerance && stockErr < tolerance && measured.endLevel == expectedEndLevel
		result := sweepResult{
			schedule:      append([]int(nil), plan.schedule...),
			effective:     append([]int(nil), plan.effectiveSchedule...),
			ell:           plan.ell,
			residualDiags: plan.stages[len(plan.stages)-1].diags,
			measurement:   measured,
			fusedStockErr: stockErr,
			speedup:       ma.median / measured.median,
		}
		results = append(results, result)
		fmt.Printf("  CANDIDATE %d/%d schedule=%v effective=%v ell=%d right_HLT=%d residual_diags=%d branches=%d median=%.1fms std=%.1fms speedup=%.2fx error=%.3e stock_error=%.3e level=%d correct=%t\n",
			i+1, len(schedules), plan.schedule, plan.effectiveSchedule, plan.ell, plan.rightCalls, plan.stages[len(plan.stages)-1].diags, len(plan.branches), measured.median, measured.std, result.speedup, measured.maxError, stockErr, measured.endLevel, correct)
		if !correct {
			return fmt.Errorf("schedule %v failed correctness", plan.schedule)
		}
		plan = transformPlan{}
		pool = nil
		eval = nil
		out = nil
		stockOut = nil
		runtime.GC()
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].measurement.median < results[j].measurement.median
	})
	for rank, result := range results {
		fmt.Printf("  RANK %d schedule=%v effective=%v median=%.1fms std=%.1fms speedup=%.2fx ell=%d residual_diags=%d\n",
			rank+1, result.schedule, result.effective, result.measurement.median, result.measurement.std, result.speedup, result.ell, result.residualDiags)
	}
	winner := results[0]
	fmt.Printf("  FASTEST schedule=%v effective=%v median=%.1fms Ma=%.1fms speedup=%.2fx correct=true\n",
		winner.schedule, winner.effective, winner.measurement.median, ma.median, winner.speedup)
	return nil
}
