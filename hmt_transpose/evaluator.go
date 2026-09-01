package main

import (
	"fmt"
	"sort"
	"sync"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/ring/ringqp"
	"github.com/tuneinsight/lattigo/v5/utils"

	"project1-fhe_extension_v1.0/packopt"
)

func splitGiantGroups(lt he.LinearTransformation) []he.LinearTransformation {
	index, _, _ := lt.BSGSIndex()
	js := make([]int, 0, len(index))
	for j := range index {
		js = append(js, j)
	}
	sort.Ints(js)
	groups := make([]he.LinearTransformation, 0, len(js))
	for _, j := range js {
		group := lt
		group.Vec = make(map[int]ringqp.Poly, len(index[j]))
		for _, i := range index[j] {
			group.Vec[i+j] = lt.Vec[i+j]
		}
		groups = append(groups, group)
	}
	return groups
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func prepareHLT(eval *hefloat.Evaluator, ct *rlwe.Ciphertext, lt he.LinearTransformation) (map[int]*rlwe.Element[ringqp.Poly], error) {
	level := minInt(ct.Level(), lt.Level)
	buff := eval.GetBuffDecompQP()
	eval.DecomposeNTT(level, eval.GetRLWEParameters().MaxLevelP(), eval.GetRLWEParameters().PCount(), ct.Value[1], ct.IsNTT, buff)
	_, _, rotations := lt.BSGSIndex()
	preRotated := map[int]*rlwe.Element[ringqp.Poly]{}
	if err := he.GetPreRotatedCiphertextForDiagonalMatrixMultiplication(level, eval, ct, buff, rotations, preRotated); err != nil {
		return nil, err
	}
	return preRotated, nil
}

func evalGroupsQP(eval *hefloat.Evaluator, ct *rlwe.Ciphertext, groups []he.LinearTransformation, preRotated map[int]*rlwe.Element[ringqp.Poly], pool []*hefloat.Evaluator) (*rlwe.Ciphertext, error) {
	if len(groups) == 0 {
		return nil, fmt.Errorf("linear transformation has no BSGS giant groups")
	}
	params := eval.GetRLWEParameters()
	levelQ := minInt(ct.Level(), groups[0].Level)
	levelP := params.MaxLevelP()
	partials := make([]*rlwe.Element[ringqp.Poly], len(groups))
	for i := range partials {
		partials[i] = rlwe.NewElementExtended(params, 1, levelQ, levelP)
	}

	nWorkers := minInt(len(pool), len(groups))
	if nWorkers < 1 {
		return nil, fmt.Errorf("QP evaluator pool is empty")
	}
	jobs := make(chan int)
	errs := make([]error, len(groups))
	var wg sync.WaitGroup
	wg.Add(nWorkers)
	for workerID := 0; workerID < nWorkers; workerID++ {
		go func(workerID int) {
			de := hefloat.NewLinearTransformationEvaluator(pool[workerID])
			for groupID := range jobs {
				errs[groupID] = he.MultiplyByDiagMatrixBSGSNoModDown(
					de.EvaluatorForLinearTransformation,
					ct,
					groups[groupID],
					preRotated,
					partials[groupID],
				)
			}
			wg.Done()
		}(workerID)
	}
	for i := range groups {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}

	ringQP := params.RingQP().AtLevel(levelQ, levelP)
	sum := partials[0]
	for i := 1; i < len(partials); i++ {
		ringQP.Add(sum.Value[0], partials[i].Value[0], sum.Value[0])
		ringQP.Add(sum.Value[1], partials[i].Value[1], sum.Value[1])
	}
	out := rlwe.NewCiphertext(params, 1, levelQ)
	*out.MetaData = *ct.MetaData
	out.Scale = ct.Scale.Mul(groups[0].Scale)
	eval.ModDownQPtoQNTT(levelQ, levelP, sum.Value[0].Q, sum.Value[0].P, out.Value[0])
	eval.ModDownQPtoQNTT(levelQ, levelP, sum.Value[1].Q, sum.Value[1].P, out.Value[1])
	return out, nil
}

func evalStock(eval *hefloat.Evaluator, ct *rlwe.Ciphertext, lt he.LinearTransformation, preRotated map[int]*rlwe.Element[ringqp.Poly]) (*rlwe.Ciphertext, error) {
	out := rlwe.NewCiphertext(eval.GetRLWEParameters(), 1, minInt(ct.Level(), lt.Level))
	de := hefloat.NewLinearTransformationEvaluator(eval)
	return out, he.MultiplyByDiagMatrixBSGS(de.EvaluatorForLinearTransformation, ct, lt, preRotated, out)
}

func makeLinearTransformation(params hefloat.Parameters, encoder *hefloat.Encoder, diagonals map[int][]float64, n1 int) (he.LinearTransformation, error) {
	indices := packopt.NonzeroKeys(diagonals)
	if len(indices) == 0 || len(diagonals[indices[0]]) == 0 {
		return he.LinearTransformation{}, fmt.Errorf("empty diagonal map")
	}
	packedSlots := len(diagonals[indices[0]])
	logCols := 0
	for 1<<logCols < packedSlots {
		logCols++
	}
	encodedDiagonals := hefloat.Diagonals[float64]{}
	for _, k := range indices {
		encodedDiagonals[k] = append([]float64(nil), diagonals[k]...)
	}
	lt := hefloat.NewLinearTransformation(params, hefloat.LinearTransformationParameters{
		DiagonalsIndexList:       indices,
		Level:                    params.MaxLevel(),
		Scale:                    params.DefaultScale(),
		LogDimensions:            ring.Dimensions{Rows: 0, Cols: logCols},
		LogBabyStepGianStepRatio: -1,
	})
	h := he.LinearTransformation(lt)
	h.N1 = n1
	if err := hefloat.EncodeLinearTransformation(encoder, encodedDiagonals, hefloat.LinearTransformation(h)); err != nil {
		return he.LinearTransformation{}, err
	}
	return h, nil
}

func rotationUnion(plans ...transformPlan) []int {
	seen := map[int]bool{}
	for _, plan := range plans {
		for _, st := range append(append([]stage{}, plan.stages...), plan.branches...) {
			_, baby, giant := st.lt.BSGSIndex()
			for _, rotation := range utils.GetDistincts(append(baby, giant...)) {
				if rotation != 0 {
					seen[rotation] = true
				}
			}
		}
	}
	rotations := make([]int, 0, len(seen))
	for rotation := range seen {
		rotations = append(rotations, rotation)
	}
	sort.Ints(rotations)
	return rotations
}
