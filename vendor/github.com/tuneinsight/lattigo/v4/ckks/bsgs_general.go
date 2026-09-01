package ckks

import "github.com/tuneinsight/lattigo/v4/rlwe/ringqp"

// BsgsIndex4ArithmeticSeqGeneral is a generalization of BsgsIndex4ArithmeticSeq
// that supports arbitrary N1 (not restricted to powers of two), using proper
// floor-modulo arithmetic instead of the bitmask trick (rot & (N1-1)), which
// only computes a correct modulo when N1 is a power of two. Verified to
// produce IDENTICAL output to the original at every power-of-two N1 (see
// Step 3's self-check before trusting any timing number from this phase).
func BsgsIndex4ArithmeticSeqGeneral(el interface{}, slots, N1 int, commonDiff int) (index map[int][]int, rotN1, rotN2 []int) {
	index = make(map[int][]int)
	rotN1Map := make(map[int]bool)
	rotN2Map := make(map[int]bool)
	var nonZeroDiags []int
	switch element := el.(type) {
	case map[int][]complex128:
		nonZeroDiags = make([]int, len(element))
		var i int
		for key := range element {
			nonZeroDiags[i] = key
			i++
		}
	case map[int][]float64:
		nonZeroDiags = make([]int, len(element))
		var i int
		for key := range element {
			nonZeroDiags[i] = key
			i++
		}
	case map[int]bool:
		nonZeroDiags = make([]int, len(element))
		var i int
		for key := range element {
			nonZeroDiags[i] = key
			i++
		}
	case map[int]ringqp.Poly:
		nonZeroDiags = make([]int, len(element))
		var i int
		for key := range element {
			nonZeroDiags[i] = key
			i++
		}
	case []int:
		nonZeroDiags = element
	}

	pmod := func(a, m int) int {
		r := a % m
		if r < 0 {
			r += m
		}
		return r
	}

	for _, rot := range nonZeroDiags {
		var rotDivByCD int
		if rot%commonDiff != 0 {
			rotDivByCD = (rot - slots)
			rotDivByCD = rotDivByCD / commonDiff
		} else {
			rotDivByCD = rot / commonDiff
		}
		rotDivByCD &= (slots - 1)

		i := pmod(rotDivByCD, N1)
		j := (rotDivByCD - i) / N1

		idxN1 := (j * N1 * commonDiff) & (slots - 1)
		idxN2 := (i * commonDiff) & (slots - 1)

		if index[idxN1] == nil {
			index[idxN1] = []int{idxN2}
		} else {
			index[idxN1] = append(index[idxN1], idxN2)
		}
		rotN1Map[idxN1] = true
		rotN2Map[idxN2] = true
	}

	rotN1 = []int{}
	for i := range rotN1Map {
		rotN1 = append(rotN1, i)
	}
	rotN2 = []int{}
	for i := range rotN2Map {
		rotN2 = append(rotN2, i)
	}
	return
}
