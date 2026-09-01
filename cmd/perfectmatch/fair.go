package main

import (
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	mtrxmult "project1-fhe_extension_v1.0/matrix_mult"
)

func evalChainStock(p hefloat.Parameters, eval *hefloat.Evaluator, ct0 *rlwe.Ciphertext, stages, branches []stage) (*rlwe.Ciphertext, error) {
	ct := ct0.CopyNew()
	for si, s := range stages {
		_, pre, err := prepare(eval, ct, s.lt)
		if err != nil {
			return nil, err
		}
		ct, err = stockEval(eval, ct, s.lt, pre)
		if err != nil {
			return nil, err
		}
		if si+1 < len(stages) {
			if err = eval.RescaleTo(ct, p.DefaultScale(), ct); err != nil {
				return nil, err
			}
		}
	}
	for _, b := range branches {
		_, pre, err := prepare(eval, ct0, b.lt)
		if err != nil {
			return nil, err
		}
		part, err := stockEval(eval, ct0, b.lt, pre)
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
	if err := eval.RescaleTo(ct, p.DefaultScale(), ct); err != nil {
		return nil, err
	}
	return ct, nil
}

func runFairPipeline(p hefloat.Parameters, enc *hefloat.Encoder, dec *rlwe.Decryptor, eval *hefloat.Evaluator, ct0 *rlwe.Ciphertext, ma, opt plan, x []float64, d, trials int) {
	n := d * d
	sig, _ := mtrxmult.Gen_sigma_diagonalVecotrs(d)
	tau, _ := mtrxmult.Gen_tao_diagonalVectors(d)
	tau = mtrxmult.CentralizeKeys(tau, d, n)
	sigExpected, tauExpected := applyDiag(x, sig, n), applyDiag(x, tau, n)
	maSigEval, maTauEval := eval.ShallowCopy(), eval.ShallowCopy()
	optSigEval, optTauEval := eval.ShallowCopy(), eval.ShallowCopy()
	optSigPool := qpWorkerPool(optSigEval, opt.sig, opt.sigBranches)
	optTauPool := qpWorkerPool(optTauEval, opt.tau, opt.tauBranches)
	for i := 0; i < benchmarkWarmups; i++ { // same complete boundaries as timed runs
		_, _ = evalChainStock(p, maSigEval, ct0, ma.sig, ma.sigBranches)
		_, _ = evalChainStock(p, maTauEval, ct0, ma.tau, ma.tauBranches)
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			_, _ = evalChainQP(p, enc, optSigEval, optSigPool, ct0, opt.sig, opt.sigBranches)
		}()
		go func() {
			defer wg.Done()
			_, _ = evalChainQP(p, enc, optTauEval, optTauPool, ct0, opt.tau, opt.tauBranches)
		}()
		wg.Wait()
	}
	maTimes, optTimes := make([]float64, trials), make([]float64, trials)
	var maSig, maTau, opSig, opTau *rlwe.Ciphertext
	for t := 0; t < trials; t++ {
		start := time.Now()
		maSig, _ = evalChainStock(p, maSigEval, ct0, ma.sig, ma.sigBranches)
		maTau, _ = evalChainStock(p, maTauEval, ct0, ma.tau, ma.tauBranches)
		maTimes[t] = float64(time.Since(start).Microseconds()) / 1000
		start = time.Now()
		var wg sync.WaitGroup
		wg.Add(2)
		var se, te error
		go func() {
			defer wg.Done()
			opSig, se = evalChainQP(p, enc, optSigEval, optSigPool, ct0, opt.sig, opt.sigBranches)
		}()
		go func() {
			defer wg.Done()
			opTau, te = evalChainQP(p, enc, optTauEval, optTauPool, ct0, opt.tau, opt.tauBranches)
		}()
		wg.Wait()
		if se != nil || te != nil {
			panic(fmt.Errorf("optimized pipeline failed: sigma=%v tau=%v", se, te))
		}
		optTimes[t] = float64(time.Since(start).Microseconds()) / 1000
	}
	maMed, maStd := medianStd(maTimes)
	opMed, opStd := medianStd(optTimes)
	sigErr := maxDiff(decode(enc, dec, opSig, n), sigExpected)
	tauErr := maxDiff(decode(enc, dec, opTau, n), tauExpected)
	maErr := math.Max(maxDiff(decode(enc, dec, maSig, n), sigExpected), maxDiff(decode(enc, dec, maTau, n), tauExpected))
	ok := sigErr < math.Pow(2, -20) && tauErr < math.Pow(2, -20) && maErr < math.Pow(2, -20)
	fmt.Printf("  FAIR d=%d groups=%v Ma=%.1f±%.1fms Opt=%.1f±%.1fms speedup=%.2fx MaErr=%.3e OptSigmaErr=%.3e OptTauErr=%.3e correct=%t\n", d, opt.groups, maMed, maStd, opMed, opStd, maMed/opMed, maErr, sigErr, tauErr, ok)
}
