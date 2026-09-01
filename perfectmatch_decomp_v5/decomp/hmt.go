package decomp

import "math/rand"

func GenHMTPerm(d int) []int {
	n := d * d
	p := make([]int, n)
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			p[i*d+j] = j*d + i
		}
	}
	return p
}

func FullDepth(d int) int {
	L := 0
	for x := d - 1; x > 1; x >>= 1 {
		L++
	}
	return L
}

func TheoreticalCost(d, ell int) float64 {
	L := FullDepth(d)
	if ell < 0 {
		ell = 0
	}
	if ell > L {
		ell = L
	}
	dRes := d >> uint(ell)
	if dRes < 1 {
		dRes = 1
	}
	numDiags := 2*dRes - 1
	if numDiags < 1 {
		numDiags = 1
	}
	return 2*float64(ell) + 2*sqrtF(float64(numDiags))
}

func OptimalTruncation(d int) (ellStar int, cost float64) {
	L := FullDepth(d)
	cost = 1e18
	for ell := 0; ell <= L; ell++ {
		c := TheoreticalCost(d, ell)
		if c < cost {
			cost = c
			ellStar = ell
		}
	}
	return
}

func sqrtF(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 20; i++ {
		z = 0.5 * (z + x/z)
	}
	return z
}

func PerturbHMT(d, k int, seed int64) []int {
	p := GenHMTPerm(d)
	n := len(p)
	rng := rand.New(rand.NewSource(seed))
	for t := 0; t < k; t++ {
		i := rng.Intn(n)
		j := rng.Intn(n)
		if i == j {
			j = (j + 1) % n
		}
		p[i], p[j] = p[j], p[i]
	}
	return p
}

type HybridResult struct {
	CorrPR     []int   // residual-correction right factor (≤ 3+|U| diags)
	IdealChain [][]int // subsequent ideal right factors on the left residual
	Residual   []int   // final left residual (BSGS)
	Cutoffs    []int
	Deficient  int // |U|
	Rc         int
	OK         bool // composition check pR[pL[i]]==p[i]
}

func HybridNearHMT(p []int, d int, maxIdealRounds int) HybridResult {
	a0 := d - 1
	r := maxAbsDiag(p)
	if r < a0 {
		r = a0
	}
	d1 := MaxMatchDecomp(p, a0, r)
	chain, residual, cuts := FullDepthMatchFixedSpacing(d1.PL, a0, maxIdealRounds)
	ok := VerifyComposition(p, d1.PL, d1.PR)
	return HybridResult{
		CorrPR:     d1.PR,
		IdealChain: chain,
		Residual:   residual,
		Cutoffs:    cuts,
		Deficient:  len(d1.Deficient),
		Rc:         d1.Rc,
		OK:         ok,
	}
}

func DiagMap(perm []int) map[int][]float64 {
	n := len(perm)
	U := make(map[int][]float64)
	for i, pi := range perm {
		k := pi - i
		km := k % n
		if km > n/2 {
			km -= n
		}
		if km <= -n/2 {
			km += n
		}
		if U[km] == nil {
			U[km] = make([]float64, n)
		}
		U[km][i] = 1
	}
	return U
}

func DiagMapIdealDmp(perm []int) map[int][]float64 {
	n := len(perm)
	inv := make([]int, n)
	for i, pi := range perm {
		inv[pi] = i
	}
	U := make(map[int][]float64)
	for j := 0; j < n; j++ {
		k := inv[j] - j
		km := ((k % n) + n) % n
		if km > n/2 {
			km -= n
		}
		if U[km] == nil {
			U[km] = make([]float64, n)
		}
		U[km][j] = 1
	}
	return U
}
