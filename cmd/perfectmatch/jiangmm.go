package main

import (
	"fmt"
	"math"
	"os"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
	mtrxmult "project1-fhe_extension_v1.0/matrix_mult"
)

type jiangOuterMasks struct {
	colK, colWrap []*rlwe.Plaintext
	rowK, rowWrap []*rlwe.Plaintext
}

func encodeJiangOuterMasks(p hefloat.Parameters, enc *hefloat.Encoder, d, level int, dims ring.Dimensions) (m jiangOuterMasks, err error) {
	n := d * d
	m.colK = make([]*rlwe.Plaintext, d)
	m.colWrap = make([]*rlwe.Plaintext, d)
	m.rowK = make([]*rlwe.Plaintext, d)
	m.rowWrap = make([]*rlwe.Plaintext, d)
	for k := 1; k < d; k++ {
		U, e := mtrxmult.Gen_colShift_diagonalVectors(d, k)
		if e != nil {
			return m, e
		}
		m.colK[k] = hefloat.NewPlaintext(p, level)
		m.colK[k].LogDimensions = dims
		if e = enc.Encode(U[k][:n], m.colK[k]); e != nil {
			return m, e
		}
		m.colWrap[k] = hefloat.NewPlaintext(p, level)
		m.colWrap[k].LogDimensions = dims
		if e = enc.Encode(U[k-d][:n], m.colWrap[k]); e != nil {
			return m, e
		}

		top, bottom := make([]float64, n), make([]float64, n)
		for i := 0; i < d; i++ {
			for j := 0; j < d; j++ {
				if i < d-k {
					top[i*d+j] = 1
				} else {
					bottom[i*d+j] = 1
				}
			}
		}
		m.rowK[k] = hefloat.NewPlaintext(p, level)
		m.rowK[k].LogDimensions = dims
		if e = enc.Encode(top, m.rowK[k]); e != nil {
			return m, e
		}
		m.rowWrap[k] = hefloat.NewPlaintext(p, level)
		m.rowWrap[k].LogDimensions = dims
		if e = enc.Encode(bottom, m.rowWrap[k]); e != nil {
			return m, e
		}
	}
	return m, nil
}

func jiangOuterHoistedV5(eval *hefloat.Evaluator, sig, tau *rlwe.Ciphertext, d int, masks jiangOuterMasks) (*rlwe.Ciphertext, error) {
	rotsA := make([]int, 0, 2*(d-1))
	rotsB := make([]int, 0, d-1)
	for k := 1; k < d; k++ {
		rotsA = append(rotsA, k, k-d)
		rotsB = append(rotsB, k*d, (k-d)*d)
	}
	ha, err := eval.RotateHoistedNew(sig, rotsA)
	if err != nil {
		return nil, err
	}
	hb, err := eval.RotateHoistedNew(tau, rotsB)
	if err != nil {
		return nil, err
	}
	sum, err := eval.MulNew(sig, tau)
	if err != nil {
		return nil, err
	}
	for k := 1; k < d; k++ {
		col, e := eval.MulNew(ha[k], masks.colK[k])
		if e != nil {
			return nil, e
		}
		if e = eval.MulThenAdd(ha[k-d], masks.colWrap[k], col); e != nil {
			return nil, e
		}
		if e = eval.RescaleTo(col, sig.Scale, col); e != nil {
			return nil, e
		}
		row, e := eval.MulNew(hb[k*d], masks.rowK[k])
		if e != nil {
			return nil, e
		}
		if e = eval.MulThenAdd(hb[(k-d)*d], masks.rowWrap[k], row); e != nil {
			return nil, e
		}
		if e = eval.RescaleTo(row, tau.Scale, row); e != nil {
			return nil, e
		}
		if sum.Level() > col.Level() {
			eval.DropLevel(sum, sum.Level()-col.Level())
		}
		if e = eval.MulThenAdd(col, row, sum); e != nil {
			return nil, e
		}
	}
	if os.Getenv("PERFECTMATCH_JIANGMM_DEBUG") == "1" {
		fmt.Printf("    outer debug: sigLevel=%d tauLevel=%d sumLevel=%d degree=%d scale=%v\n", sig.Level(), tau.Level(), sum.Level(), sum.Degree(), sum.Scale)
	}
	if err = eval.Relinearize(sum, sum); err != nil {
		return nil, err
	}
	if err = eval.Rescale(sum, sum); err != nil {
		return nil, err
	}
	return sum, nil
}

func plainMatrixProduct(a, b []float64, d int) []float64 {
	c := make([]float64, d*d)
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			for k := 0; k < d; k++ {
				c[i*d+j] += a[i*d+k] * b[k*d+j]
			}
		}
	}
	return c
}

func runFairJiangMatrixMultiplication(p hefloat.Parameters, enc *hefloat.Encoder, dec *rlwe.Decryptor, eval *hefloat.Evaluator, ctA, ctB *rlwe.Ciphertext, ma, opt plan, a, b []float64, d, trials int) {
	n := d * d
	maSigEval, maTauEval := eval.ShallowCopy(), eval.ShallowCopy()
	optSigEval, optTauEval := eval.ShallowCopy(), eval.ShallowCopy()
	maOuterEval, optOuterEval := eval.ShallowCopy(), eval.ShallowCopy()
	optSigPool := qpWorkerPool(optSigEval, opt.sig, opt.sigBranches)
	optTauPool := qpWorkerPool(optTauEval, opt.tau, opt.tauBranches)

	probe, err := evalChainStock(p, maSigEval, ctA, ma.sig, ma.sigBranches)
	if err != nil {
		panic(err)
	}
	masks, err := encodeJiangOuterMasks(p, enc, d, probe.Level(), probe.LogDimensions)
	if err != nil {
		panic(err)
	}

	runMa := func() (*rlwe.Ciphertext, error) {
		sig, e := evalChainStock(p, maSigEval, ctA, ma.sig, ma.sigBranches)
		if e != nil {
			return nil, e
		}
		tau, e := evalChainStock(p, maTauEval, ctB, ma.tau, ma.tauBranches)
		if e != nil {
			return nil, e
		}
		return jiangOuterHoistedV5(maOuterEval, sig, tau, d, masks)
	}
	runOpt := func() (*rlwe.Ciphertext, error) {
		var sig, tau *rlwe.Ciphertext
		var sigErr, tauErr error
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			sig, sigErr = evalChainQP(p, enc, optSigEval, optSigPool, ctA, opt.sig, opt.sigBranches)
		}()
		go func() {
			defer wg.Done()
			tau, tauErr = evalChainQP(p, enc, optTauEval, optTauPool, ctB, opt.tau, opt.tauBranches)
		}()
		wg.Wait()
		if sigErr != nil {
			return nil, sigErr
		}
		if tauErr != nil {
			return nil, tauErr
		}
		return jiangOuterHoistedV5(optOuterEval, sig, tau, d, masks)
	}

	for i := 0; i < benchmarkWarmups; i++ {
		if _, err = runMa(); err != nil {
			panic(err)
		}
		if _, err = runOpt(); err != nil {
			panic(err)
		}
	}

	expected := plainMatrixProduct(a[:n], b[:n], d)
	expectedMax := 0.0
	for _, v := range expected {
		expectedMax = math.Max(expectedMax, math.Abs(v))
	}
	tolerance := math.Max(math.Pow(2, -16), expectedMax*math.Pow(2, -17))
	maTimes := make([]float64, trials)
	optTimes := make([]float64, trials)
	maMaxErr, optMaxErr := 0.0, 0.0
	for trial := 0; trial < trials; trial++ {
		start := time.Now()
		maOut, e := runMa()
		if e != nil {
			panic(e)
		}
		maTimes[trial] = float64(time.Since(start).Microseconds()) / 1000
		maMaxErr = math.Max(maMaxErr, maxDiff(decode(enc, dec, maOut, n), expected))

		start = time.Now()
		optOut, e := runOpt()
		if e != nil {
			panic(e)
		}
		optTimes[trial] = float64(time.Since(start).Microseconds()) / 1000
		optMaxErr = math.Max(optMaxErr, maxDiff(decode(enc, dec, optOut, n), expected))
	}
	maMedian, maStd := medianStd(maTimes)
	optMedian, optStd := medianStd(optTimes)
	correct := maMaxErr < tolerance && optMaxErr < tolerance
	fmt.Printf("  JIANG_MM d=%d groups=%v Ma=%.1f±%.1fms FusedParallel=%.1f±%.1fms speedup=%.2fx MaErr=%.3e OptErr=%.3e Tol=%.3e correct=%t\n",
		d, opt.groups, maMedian, maStd, optMedian, optStd, maMedian/optMedian, maMaxErr, optMaxErr, tolerance, correct)
}
