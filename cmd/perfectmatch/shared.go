package main

import (
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring/ringqp"
	mtrxmult "project1-fhe_extension_v1.0/matrix_mult"
)

type qpJob struct {
	ct    *rlwe.Ciphertext
	group he.LinearTransformation
	pre   map[int]*rlwe.Element[ringqp.Poly]
	out   *rlwe.Element[ringqp.Poly]
	done  chan error
}

type globalQPExecutor struct {
	jobs chan qpJob
}

func newGlobalQPExecutor(eval *hefloat.Evaluator, workers int) *globalQPExecutor {
	if workers < 1 {
		workers = 1
	}
	x := &globalQPExecutor{jobs: make(chan qpJob, 64)}
	for i := 0; i < workers; i++ {
		worker := eval.ShallowCopy()
		go func() {
			for job := range x.jobs {
				ltEval := hefloat.NewLinearTransformationEvaluator(worker)
				err := he.MultiplyByDiagMatrixBSGSNoModDown(ltEval.EvaluatorForLinearTransformation, job.ct, job.group, job.pre, job.out)
				job.done <- err
			}
		}()
	}
	return x
}

func (x *globalQPExecutor) evaluate(eval *hefloat.Evaluator, ct *rlwe.Ciphertext, groups []he.LinearTransformation, pre map[int]*rlwe.Element[ringqp.Poly]) (*rlwe.Ciphertext, error) {
	params := eval.GetRLWEParameters()
	levelQ := min(ct.Level(), groups[0].Level)
	levelP := params.MaxLevelP()
	outs := make([]*rlwe.Element[ringqp.Poly], len(groups))
	done := make(chan error, len(groups))
	for i := range groups {
		outs[i] = rlwe.NewElementExtended(params, 1, levelQ, levelP)
		x.jobs <- qpJob{ct: ct, group: groups[i], pre: pre, out: outs[i], done: done}
	}
	for range groups {
		if err := <-done; err != nil {
			return nil, err
		}
	}
	rq := params.RingQP().AtLevel(levelQ, levelP)
	sum := outs[0]
	for i := 1; i < len(outs); i++ {
		rq.Add(sum.Value[0], outs[i].Value[0], sum.Value[0])
		rq.Add(sum.Value[1], outs[i].Value[1], sum.Value[1])
	}
	out := rlwe.NewCiphertext(params, 1, levelQ)
	*out.MetaData = *ct.MetaData
	out.Scale = ct.Scale.Mul(groups[0].Scale)
	eval.ModDownQPtoQNTT(levelQ, levelP, sum.Value[0].Q, sum.Value[0].P, out.Value[0])
	eval.ModDownQPtoQNTT(levelQ, levelP, sum.Value[1].Q, sum.Value[1].P, out.Value[1])
	return out, nil
}

func prepareShared(eval *hefloat.Evaluator, ct *rlwe.Ciphertext, a, b he.LinearTransformation) ([]he.LinearTransformation, []he.LinearTransformation, map[int]*rlwe.Element[ringqp.Poly], error) {
	level := min(ct.Level(), min(a.Level, b.Level))
	buff := eval.GetBuffDecompQP()
	eval.DecomposeNTT(level, eval.GetRLWEParameters().MaxLevelP(), eval.GetRLWEParameters().PCount(), ct.Value[1], ct.IsNTT, buff)
	_, _, ar := a.BSGSIndex()
	_, _, br := b.BSGSIndex()
	seen := map[int]bool{}
	rots := make([]int, 0, len(ar)+len(br))
	for _, r := range append(ar, br...) {
		if !seen[r] {
			seen[r] = true
			rots = append(rots, r)
		}
	}
	pre := map[int]*rlwe.Element[ringqp.Poly]{}
	if err := he.GetPreRotatedCiphertextForDiagonalMatrixMultiplication(level, eval, ct, buff, rots, pre); err != nil {
		return nil, nil, nil, err
	}
	return splitGroups(a), splitGroups(b), pre, nil
}

func evalGlobalTail(p hefloat.Parameters, eval *hefloat.Evaluator, exec *globalQPExecutor, ct, original *rlwe.Ciphertext, stages, branches []stage) (*rlwe.Ciphertext, error) {
	for si, s := range stages {
		groups, pre, err := prepare(eval, ct, s.lt)
		if err != nil {
			return nil, err
		}
		ct, err = exec.evaluate(eval, ct, groups, pre)
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
		groups, pre, err := prepare(eval, original, b.lt)
		if err != nil {
			return nil, err
		}
		part, err := exec.evaluate(eval, original, groups, pre)
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

func evalSharedPipeline(p hefloat.Parameters, prepEval, sigEval, tauEval *hefloat.Evaluator, exec *globalQPExecutor, ct0 *rlwe.Ciphertext, q plan) (*rlwe.Ciphertext, *rlwe.Ciphertext, error) {
	if len(q.sig) == 0 || len(q.tau) == 0 {
		return nil, nil, fmt.Errorf("empty Sigma/Tau chain")
	}
	sg, tg, pre, err := prepareShared(prepEval, ct0, q.sig[0].lt, q.tau[0].lt)
	if err != nil {
		return nil, nil, err
	}
	var sig, tau *rlwe.Ciphertext
	var se, te error
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); sig, se = exec.evaluate(sigEval, ct0, sg, pre) }()
	go func() { defer wg.Done(); tau, te = exec.evaluate(tauEval, ct0, tg, pre) }()
	wg.Wait()
	if se != nil || te != nil {
		return nil, nil, fmt.Errorf("shared first stage: sigma=%v tau=%v", se, te)
	}
	if len(q.sig) > 1 {
		if err = sigEval.RescaleTo(sig, p.DefaultScale(), sig); err != nil {
			return nil, nil, err
		}
	}
	if len(q.tau) > 1 {
		if err = tauEval.RescaleTo(tau, p.DefaultScale(), tau); err != nil {
			return nil, nil, err
		}
	}
	wg.Add(2)
	go func() {
		defer wg.Done()
		sig, se = evalGlobalTail(p, sigEval, exec, sig, ct0, q.sig[1:], q.sigBranches)
	}()
	go func() {
		defer wg.Done()
		tau, te = evalGlobalTail(p, tauEval, exec, tau, ct0, q.tau[1:], q.tauBranches)
	}()
	wg.Wait()
	if se != nil || te != nil {
		return nil, nil, fmt.Errorf("tail: sigma=%v tau=%v", se, te)
	}
	return sig, tau, nil
}

func runSharedHoist(p hefloat.Parameters, enc *hefloat.Encoder, dec *rlwe.Decryptor, eval *hefloat.Evaluator, ct0 *rlwe.Ciphertext, ma, opt plan, x []float64, d, workers, trials int) {
	n := d * d
	sigU, _ := mtrxmult.Gen_sigma_diagonalVecotrs(d)
	tauU, _ := mtrxmult.Gen_tao_diagonalVectors(d)
	tauU = mtrxmult.CentralizeKeys(tauU, d, n)
	sigExpected, tauExpected := applyDiag(x, sigU, n), applyDiag(x, tauU, n)
	maSigEval, maTauEval := eval.ShallowCopy(), eval.ShallowCopy()
	prepEval, sigEval, tauEval := eval.ShallowCopy(), eval.ShallowCopy(), eval.ShallowCopy()
	exec := newGlobalQPExecutor(eval, workers)
	for i := 0; i < benchmarkWarmups; i++ {
		_, _ = evalChainStock(p, maSigEval, ct0, ma.sig, ma.sigBranches)
		_, _ = evalChainStock(p, maTauEval, ct0, ma.tau, ma.tauBranches)
		_, _, _ = evalSharedPipeline(p, prepEval, sigEval, tauEval, exec, ct0, opt)
	}
	maTimes, opTimes := make([]float64, trials), make([]float64, trials)
	var maS, maT, opS, opT *rlwe.Ciphertext
	for i := 0; i < trials; i++ {
		start := time.Now()
		maS, _ = evalChainStock(p, maSigEval, ct0, ma.sig, ma.sigBranches)
		maT, _ = evalChainStock(p, maTauEval, ct0, ma.tau, ma.tauBranches)
		maTimes[i] = float64(time.Since(start).Microseconds()) / 1000
		start = time.Now()
		var err error
		opS, opT, err = evalSharedPipeline(p, prepEval, sigEval, tauEval, exec, ct0, opt)
		if err != nil {
			panic(err)
		}
		opTimes[i] = float64(time.Since(start).Microseconds()) / 1000
	}
	maMed, maStd := medianStd(maTimes)
	opMed, opStd := medianStd(opTimes)
	maErr := math.Max(maxDiff(decode(enc, dec, maS, n), sigExpected), maxDiff(decode(enc, dec, maT, n), tauExpected))
	sigErr := maxDiff(decode(enc, dec, opS, n), sigExpected)
	tauErr := maxDiff(decode(enc, dec, opT, n), tauExpected)
	ok := maErr < math.Pow(2, -20) && sigErr < math.Pow(2, -20) && tauErr < math.Pow(2, -20)
	fmt.Printf("  SHARED d=%d groups=%v workers=%d Ma=%.1f±%.1fms Opt=%.1f±%.1fms speedup=%.2fx MaErr=%.3e SigmaErr=%.3e TauErr=%.3e correct=%t\n", d, opt.groups, workers, maMed, maStd, opMed, opStd, maMed/opMed, maErr, sigErr, tauErr, ok)
}
