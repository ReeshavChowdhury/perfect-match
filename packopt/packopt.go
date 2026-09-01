package packopt

import (
	"fmt"
	"sort"
)

type CostModel struct {
	Decomp  float64 // fixed per HLT call
	PerDiag float64 // per nonzero diagonal
}

func DefaultCost() CostModel {
	return CostModel{Decomp: 100, PerDiag: 25}
}

func (m CostModel) HLT(nDiags int) float64 {
	if nDiags < 0 {
		nDiags = 0
	}
	return m.Decomp + m.PerDiag*float64(nDiags)
}

func ComposeDiag(A, B map[int][]float64, n int) map[int][]float64 {
	C := make(map[int][]float64)
	for kA, diagA := range A {
		for kB, diagB := range B {
			for l := 0; l < n; l++ {
				if l >= len(diagA) || diagA[l] == 0 {
					continue
				}
				j := ((l+kA)%n + n) % n
				if j >= len(diagB) || diagB[j] == 0 {
					continue
				}
				kC := kA + kB
				if C[kC] == nil {
					C[kC] = make([]float64, n)
				}
				C[kC][l] += diagA[l] * diagB[j]
			}
		}
	}
	return C
}

func NDiags(U map[int][]float64) int {
	c := 0
	for _, d := range U {
		for _, v := range d {
			if v != 0 {
				c++
				break
			}
		}
	}
	return c
}

func NonzeroKeys(U map[int][]float64) []int {
	var k []int
	for key, d := range U {
		for _, v := range d {
			if v != 0 {
				k = append(k, key)
				break
			}
		}
	}
	sort.Ints(k)
	return k
}

func RightFactors(Udecomp []map[int][]float64) []map[int][]float64 {
	ell := len(Udecomp) - 1
	if ell <= 0 {
		return nil
	}
	out := make([]map[int][]float64, 0, ell)
	for i := ell; i >= 1; i-- {
		out = append(out, Udecomp[i])
	}
	return out
}

func GreedyPack(factors []map[int][]float64, n, maxPack int) []map[int][]float64 {
	if maxPack < 1 {
		maxPack = 1
	}
	var packed []map[int][]float64
	i := 0
	for i < len(factors) {
		take := maxPack
		if i+take > len(factors) {
			take = len(factors) - i
		}
		prod := factors[i]
		for t := 1; t < take; t++ {
			prod = ComposeDiag(factors[i+t], prod, n)
		}
		packed = append(packed, prod)
		i += take
	}
	return packed
}

func segmentProduct(factors []map[int][]float64, L, R, n int) map[int][]float64 {
	prod := factors[L]
	for t := L + 1; t <= R; t++ {
		prod = ComposeDiag(factors[t], prod, n)
	}
	return prod
}

func DPPack(factors []map[int][]float64, n int, model CostModel, maxSeg, maxSegCount int) (packed []map[int][]float64, cost float64, report string) {
	m := len(factors)
	if m == 0 {
		return nil, 0, "empty"
	}
	if maxSeg < 1 {
		maxSeg = m
	}
	if maxSeg > m {
		maxSeg = m
	}

	type seg struct {
		U  map[int][]float64
		c  float64
		nd int
	}
	segs := make([][]seg, m)
	for i := 0; i < m; i++ {
		segs[i] = make([]seg, m)
		for j := i; j < m && j-i+1 <= maxSeg; j++ {
			U := segmentProduct(factors, i, j, n)
			nd := NDiags(U)
			segs[i][j] = seg{U, model.HLT(nd), nd}
		}
	}

	const inf = 1e100
	Kmax := m
	if maxSegCount > 0 && maxSegCount < Kmax {
		Kmax = maxSegCount
	}
	dp := make([][]float64, Kmax+1)
	prev := make([][]int, Kmax+1)
	for k := 0; k <= Kmax; k++ {
		dp[k] = make([]float64, m)
		prev[k] = make([]int, m)
		for j := 0; j < m; j++ {
			dp[k][j] = inf
			prev[k][j] = -1
		}
	}
	for j := 0; j < m && j+1 <= maxSeg; j++ {
		dp[1][j] = segs[0][j].c
		prev[1][j] = 0
	}
	for k := 2; k <= Kmax; k++ {
		for j := 0; j < m; j++ {
			for s := j; s >= 0 && j-s+1 <= maxSeg; s-- {
				if s == 0 {
					continue
				}
				if dp[k-1][s-1] >= inf/2 {
					continue
				}
				cand := dp[k-1][s-1] + segs[s][j].c
				if cand < dp[k][j] {
					dp[k][j] = cand
					prev[k][j] = s
				}
			}
		}
	}

	bestK, bestC := 1, dp[1][m-1]
	for k := 2; k <= Kmax; k++ {
		if dp[k][m-1] < bestC {
			bestC = dp[k][m-1]
			bestK = k
		}
	}
	if bestC >= inf/2 {
		return GreedyPack(factors, n, 2), model.HLT(7) * float64((m+1)/2), "DP failed; greedy pair\n"
	}

	type span struct{ L, R int }
	var spans []span
	j := m - 1
	for k := bestK; k >= 1; k-- {
		s := prev[k][j]
		spans = append(spans, span{s, j})
		j = s - 1
	}
	for i, h := 0, len(spans)-1; i < h; i, h = i+1, h-1 {
		spans[i], spans[h] = spans[h], spans[i]
	}
	for _, sp := range spans {
		U := segs[sp.L][sp.R].U
		packed = append(packed, U)
		report += fmt.Sprintf("DP seg factors[%d..%d] → %d diags cost=%.1f\n",
			sp.L, sp.R, segs[sp.L][sp.R].nd, segs[sp.L][sp.R].c)
	}
	report += fmt.Sprintf("DP total cost=%.1f segments=%d (sep would be %.1f)\n",
		bestC, len(packed), model.HLT(3)*float64(m)) // rough
	return packed, bestC, report
}

func LastMileFuse(packed []map[int][]float64, residual map[int][]float64, n, maxDiags int) (out []map[int][]float64, absorbed bool, report string) {
	if len(packed) == 0 || residual == nil {
		return packed, false, "no fuse"
	}
	last := packed[len(packed)-1]
	prod := ComposeDiag(residual, last, n)
	nd := NDiags(prod)
	ndLast := NDiags(last)
	ndRes := NDiags(residual)
	model := DefaultCost()
	cFuse := model.HLT(nd)
	cSep := model.HLT(ndLast) + model.HLT(ndRes)
	if nd <= maxDiags && cFuse < cSep*0.95 {
		out = append([]map[int][]float64{}, packed[:len(packed)-1]...)
		out = append(out, prod)
		report = fmt.Sprintf("last-mile fuse: %d+%d diags → %d diags (cost %.1f < %.1f)\n",
			ndLast, ndRes, nd, cFuse, cSep)
		return out, true, report
	}
	report = fmt.Sprintf("last-mile skip: product %d diags cost %.1f vs sep %.1f\n", nd, cFuse, cSep)
	return packed, false, report
}

func FillArithSeq(U map[int][]float64, cd, n int) map[int][]float64 {
	if cd <= 0 {
		return U
	}
	minK, maxK := 0, 0
	first := true
	for k, diag := range U {
		nz := false
		for _, v := range diag {
			if v != 0 {
				nz = true
				break
			}
		}
		if !nz {
			continue
		}
		if first || k < minK {
			minK = k
		}
		if first || k > maxK {
			maxK = k
		}
		first = false
	}
	out := make(map[int][]float64)
	for k, v := range U {
		out[k] = v
	}
	for k := 0; k >= minK; k -= cd {
		if out[k] == nil {
			out[k] = make([]float64, n)
		}
	}
	for k := cd; k <= maxK; k += cd {
		if out[k] == nil {
			out[k] = make([]float64, n)
		}
	}
	if out[0] == nil {
		out[0] = make([]float64, n)
	}
	return out
}

func ResidualPolicy(nDiag, cd, defaultN1 int) (n1, commonDiff int) {
	if nDiag <= 8 {
		return 0, 0
	}
	n1 = defaultN1
	if n1 > nDiag {
		n1 = 1
		for n1*n1 < nDiag {
			n1++
		}
	}
	return n1, cd
}
