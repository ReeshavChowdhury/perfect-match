package main

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"sync"
	"time"

	mtrxmult "project1-fhe_extension_v1.0/matrix_mult"
	"project1-fhe_extension_v1.0/packopt"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/ring/ringqp"
	"github.com/tuneinsight/lattigo/v5/utils"
)

var qpTokens chan struct{}
var benchmarkWarmups = 2
var benchmarkTrials = 3

func parameters() (hefloat.Parameters, error) {
	return hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN: 16,
		LogQ: []int{55, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45},
		LogP: []int{55, 55, 55}, LogDefaultScale: 45,
		RingType: ring.ConjugateInvariant,
	})
}

func splitGroups(lt he.LinearTransformation) []he.LinearTransformation {
	index, _, _ := lt.BSGSIndex()
	js := make([]int, 0, len(index))
	for j := range index {
		js = append(js, j)
	}
	sort.Ints(js)
	out := make([]he.LinearTransformation, 0, len(js))
	for _, j := range js {
		s := lt
		s.Vec = make(map[int]ringqp.Poly, len(index[j]))
		for _, i := range index[j] {
			s.Vec[i+j] = lt.Vec[i+j]
		}
		out = append(out, s)
	}
	return out
}

func prepare(eval *hefloat.Evaluator, ct *rlwe.Ciphertext, lt he.LinearTransformation) ([]he.LinearTransformation, map[int]*rlwe.Element[ringqp.Poly], error) {
	level := ct.Level()
	if lt.Level < level {
		level = lt.Level
	}
	buff := eval.GetBuffDecompQP()
	eval.DecomposeNTT(level, eval.GetRLWEParameters().MaxLevelP(), eval.GetRLWEParameters().PCount(), ct.Value[1], ct.IsNTT, buff)
	_, _, rots := lt.BSGSIndex()
	pre := map[int]*rlwe.Element[ringqp.Poly]{}
	if err := he.GetPreRotatedCiphertextForDiagonalMatrixMultiplication(level, eval, ct, buff, rots, pre); err != nil {
		return nil, nil, err
	}
	return splitGroups(lt), pre, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func evalGroups(eval *hefloat.Evaluator, ct *rlwe.Ciphertext, groups []he.LinearTransformation, pre map[int]*rlwe.Element[ringqp.Poly], workers []*hefloat.Evaluator, parallel bool) (*rlwe.Ciphertext, error) {
	params := eval.GetRLWEParameters()
	levelQ := min(ct.Level(), groups[0].Level)
	levelP := params.MaxLevelP()
	outs := make([]*rlwe.Element[ringqp.Poly], len(groups))
	for i := range outs {
		outs[i] = rlwe.NewElementExtended(params, 1, levelQ, levelP)
	}
	if parallel {
		var wg sync.WaitGroup
		errs := make([]error, len(groups))
		for i := range groups {
			wg.Add(1)
			go func(i int) {
				if qpTokens != nil {
					qpTokens <- struct{}{}
					defer func() { <-qpTokens }()
				}
				de := workers[i]
				deLT := hefloat.NewLinearTransformationEvaluator(de)
				errs[i] = he.MultiplyByDiagMatrixBSGSNoModDown(deLT.EvaluatorForLinearTransformation, ct, groups[i], pre, outs[i])
				wg.Done()
			}(i)
		}
		wg.Wait()
		for _, err := range errs {
			if err != nil {
				return nil, err
			}
		}
	} else {
		deLT := hefloat.NewLinearTransformationEvaluator(eval)
		for i := range groups {
			if err := he.MultiplyByDiagMatrixBSGSNoModDown(deLT.EvaluatorForLinearTransformation, ct, groups[i], pre, outs[i]); err != nil {
				return nil, err
			}
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

func makeLT(p hefloat.Parameters, enc *hefloat.Encoder, U map[int][]float64, n1 int) (he.LinearTransformation, error) {
	indices := packopt.NonzeroKeys(U)
	if len(indices) == 0 || len(U[indices[0]]) == 0 {
		return he.LinearTransformation{}, fmt.Errorf("empty diagonal map")
	}
	packedSlots := len(U[indices[0]])
	logCols := 0
	for (1 << logCols) < packedSlots {
		logCols++
	}
	diags := hefloat.Diagonals[float64]{}
	for _, k := range indices {
		diags[k] = append([]float64(nil), U[k]...)
	}
	lt := hefloat.NewLinearTransformation(p, hefloat.LinearTransformationParameters{
		DiagonalsIndexList: indices, Level: p.MaxLevel(), Scale: p.DefaultScale(),
		LogDimensions: ring.Dimensions{Rows: 0, Cols: logCols}, LogBabyStepGianStepRatio: -1,
	})
	h := he.LinearTransformation(lt)
	h.N1 = n1
	if err := hefloat.EncodeLinearTransformation(enc, diags, hefloat.LinearTransformation(h)); err != nil {
		return he.LinearTransformation{}, err
	}
	return h, nil
}

type stage struct {
	lt     he.LinearTransformation
	groups []he.LinearTransformation
	plain  map[int][]float64
}

func packByGroups(factors []map[int][]float64, groups []int, n int) []map[int][]float64 {
	out := make([]map[int][]float64, 0, len(groups))
	i := 0
	for _, g := range groups {
		if i >= len(factors) {
			break
		}
		if i+g > len(factors) {
			g = len(factors) - i
		}
		prod := factors[i]
		for j := 1; j < g; j++ {
			prod = packopt.ComposeDiag(factors[i+j], prod, n)
		}
		out = append(out, prod)
		i += g
	}
	return out
}

func zeroMap(z []float64) map[int][]float64 {
	return map[int][]float64{0: append([]float64(nil), z...)}
}

func buildStages(p hefloat.Parameters, enc *hefloat.Encoder, U map[int][]float64, d0, n int, schedule []int) ([]stage, []stage, error) {
	ell := 0
	for _, g := range schedule {
		ell += g
	}
	de, zeros, err := mtrxmult.CheckConflictInDecompV5(U, d0, n, ell)
	if err != nil {
		return nil, nil, err
	}
	factors := packopt.RightFactors(de)
	packed := packByGroups(factors, schedule, n)
	packed = append(packed, de[0])
	stages := make([]stage, len(packed))
	for i, m := range packed {
		lt, err := makeLT(p, enc, m, 16)
		if err != nil {
			return nil, nil, err
		}
		stages[i] = stage{lt: lt, groups: splitGroups(lt), plain: m}
	}
	branches := make([]stage, 0, len(zeros))
	for j := 0; j < len(zeros); j++ {
		var bm map[int][]float64
		if j == len(zeros)-1 {
			bm = zeroMap(zeros[j])
		} else {
			i := j + 2
			suffix := de[len(de)-1]
			for k := len(de) - 2; k >= i; k-- {
				suffix = packopt.ComposeDiag(de[k], suffix, n)
			}
			bm = packopt.ComposeDiag(zeroMap(zeros[j]), suffix, n)
		}
		lt, err := makeLT(p, enc, bm, 16)
		if err != nil {
			return nil, nil, err
		}
		branches = append(branches, stage{lt: lt, groups: splitGroups(lt), plain: bm})
	}
	return stages, branches, nil
}

func rotations(stages []stage) []int {
	seen := map[int]bool{}
	for _, s := range stages {
		_, a, b := s.lt.BSGSIndex()
		for _, r := range utils.GetDistincts(append(a, b...)) {
			if r != 0 {
				seen[r] = true
			}
		}
	}
	out := make([]int, 0, len(seen))
	for r := range seen {
		out = append(out, r)
	}
	sort.Ints(out)
	return out
}

func decode(enc *hefloat.Encoder, dec *rlwe.Decryptor, ct *rlwe.Ciphertext, n int) []float64 {
	x := make([]float64, n)
	if err := enc.Decode(dec.DecryptNew(ct), x); err != nil {
		panic(err)
	}
	return x
}

func maxDiff(a, b []float64) float64 {
	e := 0.0
	for i := range a {
		e = math.Max(e, math.Abs(a[i]-b[i]))
	}
	return e
}

func permutationOutput(x []float64, p []int) []float64 {
	y := make([]float64, len(x))
	for i, j := range p {
		if i < len(x) && j >= 0 && j < len(y) {
			y[j] = x[i]
		}
	}
	return y
}

func stockEval(eval *hefloat.Evaluator, ct *rlwe.Ciphertext, lt he.LinearTransformation, pre map[int]*rlwe.Element[ringqp.Poly]) (*rlwe.Ciphertext, error) {
	out := rlwe.NewCiphertext(eval.GetRLWEParameters(), 1, min(ct.Level(), lt.Level))
	de := hefloat.NewLinearTransformationEvaluator(eval)
	return out, he.MultiplyByDiagMatrixBSGS(de.EvaluatorForLinearTransformation, ct, lt, pre, out)
}

func medianStd(x []float64) (float64, float64) {
	y := append([]float64(nil), x...)
	sort.Float64s(y)
	m := y[len(y)/2]
	var s float64
	for _, v := range y {
		s += (v - m) * (v - m)
	}
	return m, math.Sqrt(s / float64(len(y)))
}

type plan struct {
	groups                   []int
	sig, tau                 []stage
	sigBranches, tauBranches []stage
	cost                     float64
	description              string
}

func planCost(sig, tau []stage) float64 {
	m := packopt.DefaultCost()
	cost := 0.0
	for i, s := range append(append([]stage{}, sig...), tau...) {
		w := 1.0 - 0.02*float64(i%len(sig))
		if w < 0.8 {
			w = 0.8
		}
		cost += m.HLT(len(s.lt.Vec)) * w
	}
	return cost
}

func makePlan(p hefloat.Parameters, enc *hefloat.Encoder, d int, groups []int) (plan, error) {
	n := d * d
	sig, err := mtrxmult.Gen_sigma_diagonalVecotrs(d)
	if err != nil {
		return plan{}, err
	}
	tau, err := mtrxmult.Gen_tao_diagonalVectors(d)
	if err != nil {
		return plan{}, err
	}
	tau = mtrxmult.CentralizeKeys(tau, d, n)
	sigStages, sigBranches, err := buildStages(p, enc, sig, 1, n, groups)
	if err != nil {
		return plan{}, err
	}
	tauStages, tauBranches, err := buildStages(p, enc, tau, d, n, groups)
	if err != nil {
		return plan{}, err
	}
	return plan{groups: append([]int(nil), groups...), sig: sigStages, tau: tauStages, sigBranches: sigBranches, tauBranches: tauBranches, cost: planCost(sigStages, tauStages)}, nil
}

func chooseDP(p hefloat.Parameters, enc *hefloat.Encoder, d int) (plan, []plan, error) {
	var best plan
	best.cost = math.Inf(1)
	all := make([]plan, 0, 9)
	for a := 1; a <= 3; a++ {
		for b := 1; b <= 3; b++ {
			q, err := makePlan(p, enc, d, []int{a, b})
			if err != nil {
				fmt.Printf("DP candidate [%d,%d] rejected: %v\n", a, b, err)
				continue
			}
			all = append(all, q)
			if q.cost < best.cost {
				best = q
			}
		}
	}
	if len(all) == 0 {
		return plan{}, nil, fmt.Errorf("no feasible DP schedule for d=%d", d)
	}
	return best, all, nil
}

func applyDiag(x []float64, U map[int][]float64, n int) []float64 {
	y := make([]float64, len(x))
	for k, d := range U {
		for i := 0; i < n && i < len(d); i++ {
			if d[i] == 0 {
				continue
			}
			j := ((i+k)%n + n) % n
			y[i] += d[i] * x[j]
		}
	}
	return y
}
func applyDecomp(x []float64, U []map[int][]float64, n int) []float64 {
	y := append([]float64(nil), x...)
	for i := len(U) - 1; i >= 0; i-- {
		y = applyDiag(y, U[i], n)
	}
	return y
}

func sigmaPlain(x []float64, d int) []float64 {
	y := make([]float64, len(x))
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			y[i*d+j] = x[i*d+((i+j)%d)]
		}
	}
	return y
}

func tauPlain(x []float64, d int) []float64 {
	y := make([]float64, len(x))
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			y[i*d+j] = x[((i+j)%d)*d+j]
		}
	}
	return y
}

func runPlan(p hefloat.Parameters, enc *hefloat.Encoder, dec *rlwe.Decryptor, eval *hefloat.Evaluator, ct0 *rlwe.Ciphertext, q plan, x []float64, d int, trials int) (float64, float64, bool) {
	n := d * d
	maxErr := 0.0
	totalStock, totalPar := 0.0, 0.0
	sigU, err := mtrxmult.Gen_sigma_diagonalVecotrs(d)
	if err != nil {
		panic(err)
	}
	tauU, err := mtrxmult.Gen_tao_diagonalVectors(d)
	if err != nil {
		panic(err)
	}
	tauU = mtrxmult.CentralizeKeys(tauU, d, n)
	sigExpected := applyDiag(x, sigU, n)
	tauExpected := applyDiag(x, tauU, n)
	for _, item := range []struct {
		name     string
		stages   []stage
		branches []stage
		expected []float64
	}{{"Sigma", q.sig, q.sigBranches, sigExpected}, {"Tau", q.tau, q.tauBranches, tauExpected}} {
		stockCt, parCt := ct0.CopyNew(), ct0.CopyNew()
		for si, s := range item.stages {
			groups, pre, err := prepare(eval, stockCt, s.lt)
			if err != nil {
				panic(err)
			}
			workers := make([]*hefloat.Evaluator, len(groups))
			for i := range workers {
				workers[i] = eval.ShallowCopy()
			}
			var sm, pm float64
			for t := 0; t < trials; t++ {
				t0 := time.Now()
				so, e := stockEval(eval, stockCt, s.lt, pre)
				if e != nil {
					panic(e)
				}
				sm += float64(time.Since(t0).Microseconds()) / 1000
				t0 = time.Now()
				po, e := evalGroups(eval, parCt, groups, pre, workers, true)
				if e != nil {
					panic(e)
				}
				pm += float64(time.Since(t0).Microseconds()) / 1000
				if t == trials-1 {
					maxErr = math.Max(maxErr, maxDiff(decode(enc, dec, so, n), decode(enc, dec, po, n)))
				}
			}
			totalStock += sm / float64(trials)
			totalPar += pm / float64(trials)
			var e error
			stockCt, e = stockEval(eval, stockCt, s.lt, pre)
			if e != nil {
				panic(e)
			}
			parCt, e = evalGroups(eval, parCt, groups, pre, workers, true)
			if e != nil {
				panic(e)
			}
			if si+1 < len(item.stages) {
				if e := eval.RescaleTo(stockCt, p.DefaultScale(), stockCt); e != nil {
					panic(e)
				}
				if e := eval.RescaleTo(parCt, p.DefaultScale(), parCt); e != nil {
					panic(e)
				}
			}
		}
		for _, b := range item.branches {
			bg, bp, e := prepare(eval, ct0, b.lt)
			if e != nil {
				panic(e)
			}
			bw := make([]*hefloat.Evaluator, len(bg))
			for i := range bw {
				bw[i] = eval.ShallowCopy()
			}
			bs, e := stockEval(eval, ct0, b.lt, bp)
			if e != nil {
				panic(e)
			}
			bq, e := evalGroups(eval, ct0, bg, bp, bw, true)
			if e != nil {
				panic(e)
			}
			if bs.Level() > stockCt.Level() {
				eval.DropLevel(bs, bs.Level()-stockCt.Level())
			}
			if bq.Level() > parCt.Level() {
				eval.DropLevel(bq, bq.Level()-parCt.Level())
			}
			stockCt, e = eval.AddNew(stockCt, bs)
			if e != nil {
				panic(e)
			}
			stockCt.Scale = bs.Scale
			parCt, e = eval.AddNew(parCt, bq)
			if e != nil {
				panic(e)
			}
			parCt.Scale = bq.Scale
		}
		if e := eval.RescaleTo(stockCt, p.DefaultScale(), stockCt); e != nil {
			panic(e)
		}
		if e := eval.RescaleTo(parCt, p.DefaultScale(), parCt); e != nil {
			panic(e)
		}
		stockErr := maxDiff(decode(enc, dec, stockCt, n), item.expected)
		parErr := maxDiff(decode(enc, dec, parCt, n), item.expected)
		maxErr = math.Max(maxErr, stockErr)
		maxErr = math.Max(maxErr, parErr)
		if stockErr >= math.Pow(2, -20) || parErr >= math.Pow(2, -20) {
			fmt.Printf("    %s plaintext_error stock=%.3e qp=%.3e\n", item.name, stockErr, parErr)
		}
	}
	return totalStock, totalPar, maxErr < math.Pow(2, -20)
}

func qpWorkerPool(eval *hefloat.Evaluator, stages, branches []stage) []*hefloat.Evaluator {
	maxGroups := 1
	for _, s := range append(append([]stage{}, stages...), branches...) {
		if g := len(splitGroups(s.lt)); g > maxGroups {
			maxGroups = g
		}
	}
	pool := make([]*hefloat.Evaluator, maxGroups)
	for i := range pool {
		pool[i] = eval.ShallowCopy()
	}
	return pool
}

func evalChainQP(p hefloat.Parameters, enc *hefloat.Encoder, eval *hefloat.Evaluator, pool []*hefloat.Evaluator, ct0 *rlwe.Ciphertext, stages, branches []stage) (*rlwe.Ciphertext, error) {
	ct := ct0.CopyNew()
	for si, s := range stages {
		groups, pre, err := prepare(eval, ct, s.lt)
		if err != nil {
			return nil, err
		}
		workers := pool[:len(groups)]
		ct, err = evalGroups(eval, ct, groups, pre, workers, true)
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
		groups, pre, err := prepare(eval, ct0, b.lt)
		if err != nil {
			return nil, err
		}
		workers := pool[:len(groups)]
		part, err := evalGroups(eval, ct0, groups, pre, workers, true)
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

func runCombinedQP(p hefloat.Parameters, enc *hefloat.Encoder, dec *rlwe.Decryptor, eval *hefloat.Evaluator, ct0 *rlwe.Ciphertext, q plan, x []float64, d, trials int) (float64, float64, bool) {
	n := d * d
	sigU, _ := mtrxmult.Gen_sigma_diagonalVecotrs(d)
	tauU, _ := mtrxmult.Gen_tao_diagonalVectors(d)
	tauU = mtrxmult.CentralizeKeys(tauU, d, n)
	sigExpected := applyDiag(x, sigU, n)
	tauExpected := applyDiag(x, tauU, n)
	times := make([]float64, 0, trials)
	var sigOut, tauOut *rlwe.Ciphertext
	for t := 0; t < trials; t++ {
		var wg sync.WaitGroup
		var se, te error
		seval, teval := eval.ShallowCopy(), eval.ShallowCopy()
		start := time.Now()
		wg.Add(2)
		go func() {
			defer wg.Done()
			sigOut, se = evalChainQP(p, enc, seval, qpWorkerPool(seval, q.sig, q.sigBranches), ct0, q.sig, q.sigBranches)
		}()
		go func() {
			defer wg.Done()
			tauOut, te = evalChainQP(p, enc, teval, qpWorkerPool(teval, q.tau, q.tauBranches), ct0, q.tau, q.tauBranches)
		}()
		wg.Wait()
		if se != nil {
			panic(se)
		}
		if te != nil {
			panic(te)
		}
		times = append(times, float64(time.Since(start).Microseconds())/1000)
	}
	med, _ := medianStd(times)
	sigErr := maxDiff(decode(enc, dec, sigOut, n), sigExpected)
	tauErr := maxDiff(decode(enc, dec, tauOut, n), tauExpected)
	ok := sigErr < math.Pow(2, -20) && tauErr < math.Pow(2, -20)
	fmt.Printf("    combined SigmaErr=%.3e TauErr=%.3e precision=%t\n", sigErr, tauErr, ok)
	return med, math.Max(sigErr, tauErr), ok
}

func main() {
	p, err := parameters()
	if err != nil {
		panic(err)
	}
	if v := os.Getenv("QP_MAX_ACTIVE"); v != "" {
		var n int
		if _, e := fmt.Sscanf(v, "%d", &n); e == nil && n > 0 {
			qpTokens = make(chan struct{}, n)
		}
	}
	if v := os.Getenv("PERFECTMATCH_WARMUPS"); v != "" {
		_, _ = fmt.Sscanf(v, "%d", &benchmarkWarmups)
	}
	if v := os.Getenv("PERFECTMATCH_TRIALS"); v != "" {
		_, _ = fmt.Sscanf(v, "%d", &benchmarkTrials)
	}
	enc := hefloat.NewEncoder(p)
	kgen := rlwe.NewKeyGenerator(p)
	sk, pk := kgen.GenKeyPairNew()
	var rlk *rlwe.RelinearizationKey
	if os.Getenv("PERFECTMATCH_JIANGMM") == "1" {
		rlk = kgen.GenRelinearizationKeyNew(sk)
	}
	pub := hefloat.NewEncryptor(p, pk)
	dec := hefloat.NewDecryptor(p, sk)
	maxD := 256
	if v := os.Getenv("ADAPTIVE_MAXD"); v == "128" {
		maxD = 128
	}
	dims := []int{128, 256}
	switch os.Getenv("ADAPTIVE_D") {
	case "8":
		dims = []int{8}
	case "64":
		dims = []int{64}
	case "128":
		dims = []int{128}
	case "256":
		dims = []int{256}
	}
	if os.Getenv("ADAPTIVE_COMBINED") == "1" && os.Getenv("ADAPTIVE_D") == "" {
		dims = []int{64, 128}
	}
	if os.Getenv("ADAPTIVE_FAIR") == "1" && os.Getenv("ADAPTIVE_D") == "" {
		dims = []int{64, 128}
	}
	if os.Getenv("ADAPTIVE_SHARED") == "1" && os.Getenv("ADAPTIVE_D") == "" {
		dims = []int{64, 128}
	}
	if os.Getenv("PERFECTMATCH_BEST") == "1" && os.Getenv("ADAPTIVE_D") == "" {
		dims = []int{64, 128, 256}
	}
	if os.Getenv("PERFECTMATCH_SWEEP") == "1" && os.Getenv("ADAPTIVE_D") == "" {
		dims = []int{128}
	}
	for _, d := range dims {
		if d > maxD {
			continue
		}
		best, all, err := chooseDP(p, enc, d)
		if err != nil {
			panic(err)
		}
		if gs := os.Getenv("ADAPTIVE_GROUPS"); gs != "" {
			var a, b int
			if _, e := fmt.Sscanf(gs, "%d,%d", &a, &b); e == nil && a >= 1 && a <= 3 && b >= 1 && b <= 3 {
				if forced, e := makePlan(p, enc, d, []int{a, b}); e == nil {
					best = forced
				}
			}
		}
		if os.Getenv("PERFECTMATCH_BEST") == "1" {
			groups := []int{1, 2}
			if d == 64 {
				groups = []int{1, 1}
			} else if d == 128 {
				groups = []int{2, 1}
			}
			if forced, e := makePlan(p, enc, d, groups); e == nil {
				best = forced
			}
		}
		ma, _ := makePlan(p, enc, d, []int{1, 1})
		pairs, _ := makePlan(p, enc, d, []int{2, 2})
		plans := []plan{ma, pairs, best}
		preUnion := map[int]bool{}
		streamSweep := d == 256 && os.Getenv("PERFECTMATCH_SWEEP") == "1"
		if streamSweep {
			for _, q := range all {
				stages := append(append([]stage{}, q.sig...), q.tau...)
				stages = append(stages, q.sigBranches...)
				stages = append(stages, q.tauBranches...)
				for _, s := range stages {
					_, a, b := s.lt.BSGSIndex()
					for _, r := range append(a, b...) {
						if r != 0 {
							preUnion[r] = true
						}
					}
				}
			}
			all = nil
			plans = []plan{ma}
			runtime.GC()
		}
		if os.Getenv("PERFECTMATCH_BEST") == "1" {
			plans = []plan{ma, best}
			all = nil
			runtime.GC()
		}
		if d == 256 && !streamSweep {
			plans = []plan{ma, best}
			all = nil
			runtime.GC()
		}
		if os.Getenv("ADAPTIVE_ONLY") == "dp" {
			plans = []plan{best}
		}
		if os.Getenv("ADAPTIVE_ONLY") == "ma" {
			plans = []plan{ma}
		}
		union := preUnion
		for _, q := range append(all, plans...) {
			allStages := append(append([]stage{}, q.sig...), q.tau...)
			allStages = append(allStages, q.sigBranches...)
			allStages = append(allStages, q.tauBranches...)
			for _, s := range allStages {
				_, a, b := s.lt.BSGSIndex()
				for _, r := range append(a, b...) {
					if r != 0 {
						union[r] = true
					}
				}
			}
		}
		if os.Getenv("PERFECTMATCH_JIANGMM") == "1" {
			for k := 1; k < d; k++ {
				union[k] = true
				union[k-d] = true
				union[k*d] = true
				union[(k-d)*d] = true
			}
		}
		rtks := kgen.GenGaloisKeysNew(func() []uint64 {
			z := make([]uint64, 0, len(union))
			for r := range union {
				z = append(z, p.GaloisElement(r))
			}
			return z
		}(), sk)
		eval := hefloat.NewEvaluator(p, rlwe.NewMemEvaluationKeySet(rlk, rtks...))
		n := d * d
		x := make([]float64, p.MaxSlots())
		for i := 0; i < n; i++ {
			if os.Getenv("PERFECTMATCH_JIANGMM") == "1" {
				x[i] = float64((i*17)%101)/128 + 0.03125
			} else {
				x[i] = float64((i*17)%101) + 0.125
			}
		}
		pt := hefloat.NewPlaintext(p, p.MaxLevel())
		logCols := 0
		for (1 << logCols) < n {
			logCols++
		}
		pt.LogDimensions = ring.Dimensions{Rows: 0, Cols: logCols}
		if err := enc.Encode(x[:n], pt); err != nil {
			panic(err)
		}
		ct0, err := pub.EncryptNew(pt)
		if err != nil {
			panic(err)
		}
		fmt.Printf("ADAPTIVE_FUSED_V5 d=%d logN=%d slots=%d shared_keys=%d DP=%v cost=%.1f candidates=%d\n", d, p.LogN(), p.MaxSlots(), len(union), best.groups, best.cost, len(all))
		if os.Getenv("PERFECTMATCH_JIANGMM") == "1" {
			b := make([]float64, p.MaxSlots())
			for i := 0; i < n; i++ {
				b[i] = float64((i*29)%89)/128 + 0.0625
			}
			ptB := hefloat.NewPlaintext(p, p.MaxLevel())
			ptB.LogDimensions = ring.Dimensions{Rows: 0, Cols: logCols}
			if err := enc.Encode(b[:n], ptB); err != nil {
				panic(err)
			}
			ctB, err := pub.EncryptNew(ptB)
			if err != nil {
				panic(err)
			}
			runFairJiangMatrixMultiplication(p, enc, dec, eval, ct0, ctB, ma, best, x, b, d, benchmarkTrials)
			continue
		}
		if os.Getenv("PERFECTMATCH_SWEEP") == "1" {
			runScheduleSweep(p, enc, dec, eval, ct0, ma, all, x, d)
			continue
		}
		if os.Getenv("PERFECTMATCH_BEST") == "1" {
			runFairPipeline(p, enc, dec, eval, ct0, ma, best, x, d, benchmarkTrials)
			continue
		}
		if os.Getenv("ADAPTIVE_SHARED") == "1" {
			workers := 4
			if v := os.Getenv("SHARED_WORKERS"); v != "" {
				_, _ = fmt.Sscanf(v, "%d", &workers)
			}
			runSharedHoist(p, enc, dec, eval, ct0, ma, best, x, d, workers, 3)
			continue
		}
		if os.Getenv("ADAPTIVE_FAIR") == "1" {
			runFairPipeline(p, enc, dec, eval, ct0, ma, best, x, d, 3)
			continue
		}
		if os.Getenv("ADAPTIVE_COMBINED") == "1" {
			maStock, _, maOK := runPlan(p, enc, dec, eval, ct0, ma, x, d, 3)
			combined, _, combinedOK := runCombinedQP(p, enc, dec, eval, ct0, best, x, d, 3)
			fmt.Printf("  COMBINED groups=%v Ma_stock=%.1fms SigmaTau_QP_parallel=%.1fms speedup=%.2fx Ma_correct=%t combined_correct=%t\n", best.groups, maStock, combined, maStock/combined, maOK, combinedOK)
			continue
		}
		meas := make([][2]float64, len(plans))
		oks := make([]bool, len(plans))
		for i, q := range plans {
			meas[i][0], meas[i][1], oks[i] = runPlan(p, enc, dec, eval, ct0, q, x, d, 3)
		}
		base := meas[0][0]
		for i, q := range plans {
			stock, par := meas[i][0], meas[i][1]
			fmt.Printf("  groups=%v cost=%.1f stock=%.1fms qp_parallel=%.1fms Ma/stock=%.2fx Ma/qp=%.2fx stock/qp=%.2fx correct=%t\n", q.groups, q.cost, stock, par, base/stock, base/par, stock/par, oks[i])
		}
	}
}
