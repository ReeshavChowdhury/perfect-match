package decomp

import (
	"project1-fhe_extension_v1.0/perfectmatch_decomp_v5/matching"
)

type PermInfo struct {
	N       int
	Spacing int // a
	Span    int // r = max |p[i]-i|
}

type Decomposition1 struct {
	PL        []int // left / residual factor (applied last): intermediate slots
	PR        []int // right factor (applied first)
	Rc, T     int
	MatchSize int
	Deficient []int // unmatched left rows (hybrid)
	Perfect   bool
}

func CanonicalCutoff(a, r int) int {
	if a <= 0 {
		a = 1
	}
	half := (r + 2*a - 1) / (2 * a) // ceil(r/(2a))
	return a * half
}

func ComputePermInfo(perm []int) PermInfo {
	n := len(perm)
	span, spacing := 0, 0
	for i, pi := range perm {
		ki := pi - i
		if ki < 0 {
			ki = -ki
		}
		if ki == 0 {
			continue
		}
		if ki > span {
			span = ki
		}
		if spacing == 0 {
			spacing = ki
		} else {
			spacing = gcd(spacing, ki)
		}
	}
	if spacing == 0 {
		spacing = n
	}
	return PermInfo{N: n, Spacing: spacing, Span: span}
}

func BuildConstraintGraph(perm []int, rc, t int) [][]int {
	n := len(perm)
	adj := make([][]int, n)
	deltas := [3]int{0, rc, -rc}
	for i, pi := range perm {
		ki := pi - i
		adj[i] = make([]int, 0, 3)
		for _, delta := range deltas {
			diff := ki - delta
			if diff < 0 {
				diff = -diff
			}
			if diff > t {
				continue
			}
			v := pi - delta // integer, NOT mod n
			if v >= 0 && v < n {
				adj[i] = append(adj[i], v)
			}
		}
		if rc == 0 && len(adj[i]) > 1 {
			adj[i] = uniqueInts(adj[i])
		}
	}
	return adj
}

func BuildConstraintGraphFixedSpacing(perm []int, a0, r int) (adj [][]int, rc, t int) {
	rc = CanonicalCutoff(a0, r)
	t = r - rc
	if t < 0 {
		t = 0
	}
	adj = BuildConstraintGraph(perm, rc, t)
	return
}

func Depth1Match(perm []int, info PermInfo) Decomposition1 {
	return Depth1MatchWithSpacing(perm, info.Spacing, info.Span)
}

func Depth1MatchWithSpacing(perm []int, a, r int) Decomposition1 {
	n := len(perm)
	rc := CanonicalCutoff(a, r)
	t := r - rc
	if t < 0 {
		t = 0
	}
	adj := BuildConstraintGraph(perm, rc, t)
	matchL, matchR, size := matching.HopcroftKarpSquare(n, adj)
	out := Decomposition1{Rc: rc, T: t, MatchSize: size, Perfect: size == n}
	if size == n {
		out.PL, out.PR = extractFactors(perm, matchL, n)
	} else {
		for i := 0; i < n; i++ {
			if matchL[i] == -1 {
				out.Deficient = append(out.Deficient, i)
			}
		}
		out.PL = make([]int, n)
		copy(out.PL, matchL)
		_ = matchR
	}
	return out
}

func MaxMatchDecomp(perm []int, a0, r int) Decomposition1 {
	n := len(perm)
	rc := CanonicalCutoff(a0, r)
	t := r - rc
	if t < 0 {
		t = 0
	}
	adj := BuildConstraintGraph(perm, rc, t)
	matchL, matchR, size := matching.HopcroftKarpSquare(n, adj)

	freeR := make([]int, 0, n-size)
	for v := 0; v < n; v++ {
		if matchR[v] == -1 {
			freeR = append(freeR, v)
		}
	}
	fi := 0
	var deficient []int
	for u := 0; u < n; u++ {
		if matchL[u] == -1 {
			deficient = append(deficient, u)
			if fi >= len(freeR) {
				panic("matching invariant violated")
			}
			matchL[u] = freeR[fi]
			matchR[freeR[fi]] = u
			fi++
		}
	}

	pL, pR := extractFactors(perm, matchL, n)
	return Decomposition1{
		PL: pL, PR: pR, Rc: rc, T: t,
		MatchSize: size, Deficient: deficient, Perfect: size == n,
	}
}

func FullDepthMatch(perm []int) (chain [][]int, residual []int, cutoffs []int) {
	cur := append([]int(nil), perm...)
	for {
		info := ComputePermInfo(cur)
		if info.Span <= info.Spacing {
			break
		}
		d1 := Depth1Match(cur, info)
		if !d1.Perfect {
			break
		}
		chain = append(chain, d1.PR)
		cutoffs = append(cutoffs, d1.Rc)
		cur = d1.PL
	}
	return chain, cur, cutoffs
}

func FullDepthMatchFixedSpacing(perm []int, a0 int, maxRounds int) (chain [][]int, residual []int, cutoffs []int) {
	cur := append([]int(nil), perm...)
	for round := 0; round < maxRounds; round++ {
		r := maxAbsDiag(cur)
		if r <= a0 {
			break
		}
		d1 := Depth1MatchWithSpacing(cur, a0, r)
		if !d1.Perfect {
			break
		}
		chain = append(chain, d1.PR)
		cutoffs = append(cutoffs, d1.Rc)
		cur = d1.PL
	}
	return chain, cur, cutoffs
}

func extractFactors(perm []int, matchL []int, n int) (pL, pR []int) {
	pL = make([]int, n)
	copy(pL, matchL)
	invPL := make([]int, n)
	for i, v := range pL {
		invPL[v] = i
	}
	pR = make([]int, n)
	for j := 0; j < n; j++ {
		pR[j] = perm[invPL[j]]
	}
	return
}

func CountNonzeroDiags(perm []int) int {
	seen := map[int]struct{}{}
	for i, pi := range perm {
		seen[pi-i] = struct{}{}
	}
	return len(seen)
}

func VerifyComposition(p, pL, pR []int) bool {
	if len(p) != len(pL) || len(p) != len(pR) {
		return false
	}
	for i := range p {
		mid := pL[i]
		if mid < 0 || mid >= len(pR) || pR[mid] != p[i] {
			return false
		}
	}
	return true
}

func ApplyPerm(p, x []int) []int {
	q := make([]int, len(x))
	for i := range x {
		q[i] = p[x[i]]
	}
	return q
}

func ComposeChain(residual []int, chain [][]int) []int {
	cur := append([]int(nil), residual...)
	for _, pr := range chain {
		next := make([]int, len(cur))
		for i := range cur {
			next[i] = pr[cur[i]]
		}
		cur = next
	}
	return cur
}

func maxAbsDiag(perm []int) int {
	m := 0
	for i, pi := range perm {
		d := pi - i
		if d < 0 {
			d = -d
		}
		if d > m {
			m = d
		}
	}
	return m
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	if a < 0 {
		return -a
	}
	return a
}

func uniqueInts(a []int) []int {
	seen := make(map[int]struct{}, len(a))
	out := make([]int, 0, len(a))
	for _, v := range a {
		if _, ok := seen[v]; !ok {
			seen[v] = struct{}{}
			out = append(out, v)
		}
	}
	return out
}
