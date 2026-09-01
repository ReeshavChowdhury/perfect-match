package main

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v5/he"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"

	mtrxmult "project1-fhe_extension_v1.0/matrix_mult"
	"project1-fhe_extension_v1.0/packopt"
)

type stage struct {
	lt     he.LinearTransformation
	groups []he.LinearTransformation
	diags  int
}

type transformPlan struct {
	name              string
	ell               int
	schedule          []int
	effectiveSchedule []int
	stages            []stage
	branches          []stage
	rightCalls        int
	modelCost         float64
}

func fullDepth(d int) int {
	depth := 0
	for x := d - 1; x > 1; x >>= 1 {
		depth++
	}
	return depth
}

func zeroDiagonal(diagonal []float64) map[int][]float64 {
	return map[int][]float64{0: append([]float64(nil), diagonal...)}
}

func hasNonzero(U map[int][]float64) bool {
	return packopt.NDiags(U) != 0
}

func bestFusion(factors []map[int][]float64, n, maxFuse, maxSegments int) (packed []map[int][]float64, schedule []int, cost float64, err error) {
	if len(factors) == 0 {
		return nil, nil, 0, nil
	}
	if maxFuse < 1 {
		maxFuse = 1
	}
	if maxSegments < 1 {
		maxSegments = len(factors)
	}
	model := packopt.DefaultCost()
	bestCost := math.Inf(1)
	var visit func(int, []map[int][]float64, []int, float64)
	visit = func(index int, maps []map[int][]float64, spans []int, running float64) {
		if index == len(factors) {
			if running < bestCost {
				bestCost = running
				packed = append([]map[int][]float64(nil), maps...)
				schedule = append([]int(nil), spans...)
			}
			return
		}
		if len(spans) == maxSegments {
			return
		}
		remainingSlots := maxSegments - len(spans)
		if len(factors)-index > remainingSlots*maxFuse {
			return
		}
		product := factors[index]
		for width := 1; width <= maxFuse && index+width <= len(factors); width++ {
			if width > 1 {
				product = packopt.ComposeDiag(factors[index+width-1], product, n)
			}
			segmentCost := model.HLT(packopt.NDiags(product))
			visit(index+width, append(maps, product), append(spans, width), running+segmentCost)
		}
	}
	visit(0, nil, nil, 0)
	if math.IsInf(bestCost, 1) {
		return nil, nil, 0, fmt.Errorf("cannot cover %d factors with %d segments of width <=%d", len(factors), maxSegments, maxFuse)
	}
	return packed, schedule, bestCost, nil
}

func forcedFusion(factors []map[int][]float64, n int, schedule []int) (packed []map[int][]float64, cost float64, err error) {
	covered := 0
	model := packopt.DefaultCost()
	for _, width := range schedule {
		if width < 1 || width > 3 {
			return nil, 0, fmt.Errorf("schedule width %d is outside {1,2,3}", width)
		}
		if covered+width > len(factors) {
			return nil, 0, fmt.Errorf("schedule %v covers more than %d factors", schedule, len(factors))
		}
		product := factors[covered]
		for offset := 1; offset < width; offset++ {
			product = packopt.ComposeDiag(factors[covered+offset], product, n)
		}
		packed = append(packed, product)
		cost += model.HLT(packopt.NDiags(product))
		covered += width
	}
	if covered != len(factors) {
		return nil, 0, fmt.Errorf("schedule %v covers %d of %d factors", schedule, covered, len(factors))
	}
	return packed, cost, nil
}

func buildPlan(params hefloat.Parameters, encoder *hefloat.Encoder, d, ell, n1 int, optimized bool, maxFuse, maxSegments int) (transformPlan, error) {
	return buildPlanInternal(params, encoder, d, ell, n1, optimized, maxFuse, maxSegments, nil)
}

func buildScheduledPlan(params hefloat.Parameters, encoder *hefloat.Encoder, d, n1 int, schedule []int) (transformPlan, error) {
	remaining := fullDepth(d)
	effective := make([]int, len(schedule))
	ell := 0
	for i, width := range schedule {
		if width < 1 || width > 3 {
			return transformPlan{}, fmt.Errorf("schedule width %d is outside {1,2,3}", width)
		}
		if width > remaining {
			width = remaining
		}
		if width < 1 {
			return transformPlan{}, fmt.Errorf("schedule %v has more calls than the HMT depth %d", schedule, fullDepth(d))
		}
		effective[i] = width
		ell += width
		remaining -= width
	}
	plan, err := buildPlanInternal(params, encoder, d, ell, n1, true, 3, len(effective), effective)
	if err != nil {
		return transformPlan{}, err
	}
	plan.schedule = append([]int(nil), schedule...)
	plan.effectiveSchedule = effective
	return plan, nil
}

func buildPlanInternal(params hefloat.Parameters, encoder *hefloat.Encoder, d, ell, n1 int, optimized bool, maxFuse, maxSegments int, forcedSchedule []int) (transformPlan, error) {
	n := d * d
	U, err := mtrxmult.Gen_transpose_diagonalVectors(d)
	if err != nil {
		return transformPlan{}, err
	}
	decomp, zeroDiags, err := mtrxmult.CheckConflictInDecompV5(U, d-1, n, ell)
	if err != nil {
		return transformPlan{}, err
	}
	factors := packopt.RightFactors(decomp)
	var packed []map[int][]float64
	var schedule []int
	var modelCost float64
	if forcedSchedule != nil {
		schedule = append([]int(nil), forcedSchedule...)
		packed, modelCost, err = forcedFusion(factors, n, schedule)
		if err != nil {
			return transformPlan{}, err
		}
	} else if optimized {
		packed, schedule, modelCost, err = bestFusion(factors, n, maxFuse, maxSegments)
		if err != nil {
			return transformPlan{}, err
		}
	} else {
		packed = append([]map[int][]float64(nil), factors...)
		schedule = make([]int, len(factors))
		model := packopt.DefaultCost()
		for i, factor := range factors {
			schedule[i] = 1
			modelCost += model.HLT(packopt.NDiags(factor))
		}
	}
	packed = append(packed, decomp[0])
	stages := make([]stage, 0, len(packed))
	for _, Ustage := range packed {
		lt, err := makeLinearTransformation(params, encoder, Ustage, n1)
		if err != nil {
			return transformPlan{}, err
		}
		stages = append(stages, stage{lt: lt, groups: splitGiantGroups(lt), diags: packopt.NDiags(Ustage)})
	}

	branches := make([]stage, 0, len(zeroDiags))
	for j := 0; j < len(zeroDiags); j++ {
		var branchMap map[int][]float64
		if j == len(zeroDiags)-1 {
			branchMap = zeroDiagonal(zeroDiags[j])
		} else {
			i := j + 2
			suffix := decomp[len(decomp)-1]
			for k := len(decomp) - 2; k >= i; k-- {
				suffix = packopt.ComposeDiag(decomp[k], suffix, n)
			}
			branchMap = packopt.ComposeDiag(zeroDiagonal(zeroDiags[j]), suffix, n)
		}
		if !hasNonzero(branchMap) {
			continue
		}
		lt, err := makeLinearTransformation(params, encoder, branchMap, n1)
		if err != nil {
			return transformPlan{}, err
		}
		branches = append(branches, stage{lt: lt, groups: splitGiantGroups(lt), diags: packopt.NDiags(branchMap)})
	}
	name := "Ma separate"
	if optimized {
		name = "fused + QP parallel"
	}
	return transformPlan{
		name:              name,
		ell:               ell,
		schedule:          schedule,
		effectiveSchedule: append([]int(nil), schedule...),
		stages:            stages,
		branches:          branches,
		rightCalls:        len(packed) - 1,
		modelCost:         modelCost,
	}, nil
}
