package main

import (
	"fmt"
	"math"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	mtrxmult "project1-fhe_extension_v1.0/matrix_mult"
)

type sweepResult struct {
	groups      []int
	modelCost   float64
	median, std float64
	speedup     float64
	maxError    float64
	correct     bool
}

func measureMaFull(p hefloat.Parameters, enc *hefloat.Encoder, dec *rlwe.Decryptor, eval *hefloat.Evaluator, ct0 *rlwe.Ciphertext, ma plan, sigExpected, tauExpected []float64, n int) (median, std, maxErr float64, correct bool) {
	sigEval, tauEval := eval.ShallowCopy(), eval.ShallowCopy()
	for i := 0; i < benchmarkWarmups; i++ {
		_, _ = evalChainStock(p, sigEval, ct0, ma.sig, ma.sigBranches)
		_, _ = evalChainStock(p, tauEval, ct0, ma.tau, ma.tauBranches)
	}
	times := make([]float64, benchmarkTrials)
	var sig, tau *rlwe.Ciphertext
	for i := range times {
		start := time.Now()
		sig, _ = evalChainStock(p, sigEval, ct0, ma.sig, ma.sigBranches)
		tau, _ = evalChainStock(p, tauEval, ct0, ma.tau, ma.tauBranches)
		times[i] = float64(time.Since(start).Microseconds()) / 1000
	}
	median, std = medianStd(times)
	maxErr = math.Max(maxDiff(decode(enc, dec, sig, n), sigExpected), maxDiff(decode(enc, dec, tau, n), tauExpected))
	return median, std, maxErr, maxErr < math.Pow(2, -20)
}

func measureScheduleFull(p hefloat.Parameters, enc *hefloat.Encoder, dec *rlwe.Decryptor, eval *hefloat.Evaluator, ct0 *rlwe.Ciphertext, q plan, sigExpected, tauExpected []float64, n int) (median, std, maxErr float64, correct bool) {
	sigEval, tauEval := eval.ShallowCopy(), eval.ShallowCopy()
	sigPool := qpWorkerPool(sigEval, q.sig, q.sigBranches)
	tauPool := qpWorkerPool(tauEval, q.tau, q.tauBranches)
	run := func() (sig, tau *rlwe.Ciphertext, err error) {
		var se, te error
		var wg sync.WaitGroup
		wg.Add(2)
		go func() { defer wg.Done(); sig, se = evalChainQP(p, enc, sigEval, sigPool, ct0, q.sig, q.sigBranches) }()
		go func() { defer wg.Done(); tau, te = evalChainQP(p, enc, tauEval, tauPool, ct0, q.tau, q.tauBranches) }()
		wg.Wait()
		if se != nil || te != nil {
			return nil, nil, fmt.Errorf("Sigma=%v Tau=%v", se, te)
		}
		return sig, tau, nil
	}
	for i := 0; i < benchmarkWarmups; i++ {
		if _, _, err := run(); err != nil {
			panic(err)
		}
	}
	times := make([]float64, benchmarkTrials)
	var sig, tau *rlwe.Ciphertext
	for i := range times {
		start := time.Now()
		var err error
		sig, tau, err = run()
		if err != nil {
			panic(err)
		}
		times[i] = float64(time.Since(start).Microseconds()) / 1000
	}
	median, std = medianStd(times)
	maxErr = math.Max(maxDiff(decode(enc, dec, sig, n), sigExpected), maxDiff(decode(enc, dec, tau, n), tauExpected))
	return median, std, maxErr, maxErr < math.Pow(2, -20)
}

func runScheduleSweep(p hefloat.Parameters, enc *hefloat.Encoder, dec *rlwe.Decryptor, eval *hefloat.Evaluator, ct0 *rlwe.Ciphertext, ma plan, candidates []plan, x []float64, d int) {
	n := d * d
	sigU, _ := mtrxmult.Gen_sigma_diagonalVecotrs(d)
	tauU, _ := mtrxmult.Gen_tao_diagonalVectors(d)
	tauU = mtrxmult.CentralizeKeys(tauU, d, n)
	sigExpected, tauExpected := applyDiag(x, sigU, n), applyDiag(x, tauU, n)
	maMed, maStd, maErr, maOK := measureMaFull(p, enc, dec, eval, ct0, ma, sigExpected, tauExpected, n)
	fmt.Printf("SWEEP_BASELINE d=%d Ma=%.1f±%.1fms error=%.3e correct=%t\n", d, maMed, maStd, maErr, maOK)
	results := make([]sweepResult, 0, 9)
	measureOne := func(q plan) {
		med, std, err, ok := measureScheduleFull(p, enc, dec, eval, ct0, q, sigExpected, tauExpected, n)
		sp := 0.0
		if maOK && ok {
			sp = maMed / med
		}
		r := sweepResult{groups: append([]int(nil), q.groups...), modelCost: q.cost, median: med, std: std, speedup: sp, maxError: err, correct: maOK && ok}
		results = append(results, r)
		fmt.Printf("SWEEP_ROW groups=%v model=%.1f median=%.1f±%.1fms speedup=%.2fx error=%.3e correct=%t\n", r.groups, r.modelCost, r.median, r.std, r.speedup, r.maxError, r.correct)
		runtime.GC()
	}
	if len(candidates) != 0 {
		for i := range candidates {
			q := candidates[i]
			measureOne(q)
			candidates[i] = plan{}
			runtime.GC()
		}
	} else {
		for a := 1; a <= 3; a++ {
			for b := 1; b <= 3; b++ {
				q, err := makePlan(p, enc, d, []int{a, b})
				if err != nil {
					panic(err)
				}
				measureOne(q)
				q = plan{}
				runtime.GC()
			}
		}
	}
	sort.Slice(results, func(i, j int) bool { return results[i].median < results[j].median })
	fmt.Printf("SWEEP_RANKING d=%d\n", d)
	for i, r := range results {
		fmt.Printf("  rank=%d groups=%v median=%.1fms speedup=%.2fx model=%.1f correct=%t\n", i+1, r.groups, r.median, r.speedup, r.modelCost, r.correct)
	}
}
