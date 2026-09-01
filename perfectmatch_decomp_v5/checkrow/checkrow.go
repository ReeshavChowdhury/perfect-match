package checkrow

type Result struct {
	M      map[int]map[int]int
	Solved bool
	Nodes  int64
	PL, PR []int
}

func Depth1Search(p []int, rc int) Result {
	n := len(p)
	M := map[int]map[int]int{
		rc:  {},
		0:   {},
		-rc: {},
	}
	res := Result{M: M}

	for i := 0; i < n; i++ {
		k := p[i] - i
		if k == 0 {
			a := convertR(k, i, 0, n)
			if _, ok := M[0][a]; ok {
				res.Solved = false
				return res
			}
			M[0][a] = k
			continue
		}
		if k > 0 {
			a := convertR(k, i, rc, n)
			if _, ok := M[rc][a]; ok {
				res.Solved = false
				return res
			}
			M[rc][a] = k
		} else {
			a := convertR(k, i, -rc, n)
			if _, ok := M[-rc][a]; ok {
				res.Solved = false
				return res
			}
			M[-rc][a] = k
		}
	}

	type pair struct{ a, bPlus, bMinus int }
	var conflicts []int
	for a, b1 := range M[rc] {
		if b2, ok := M[-rc][a]; ok {
			conflicts = append(conflicts, a)
			_ = b1
			_ = b2
		}
	}
	sortInts(conflicts)

	for _, a1 := range conflicts {
		b1, ok1 := M[rc][a1]
		b2, ok2 := M[-rc][a1]
		if !ok1 || !ok2 {
			continue // already resolved
		}
		if abs(b1) > rc && abs(b2) > rc {
			res.Solved = false
			return res
		}
		solved := false
		if abs(b1) <= rc {
			a0 := mod(a1+rc, n)
			delete(M[rc], a1)
			M[0][a0] = b1
			hDiag := []int{rc}
			hKey := []int{a1}
			hVal := []int{b1}
			hNew := []int{a0}
			ok, nodes := checkZero2RC(M, a0, b1, n, rc, &hDiag, &hKey, &hVal, &hNew)
			res.Nodes += nodes
			if ok {
				solved = true
			} else {
				delete(M[0], a0)
				M[rc][a1] = b1
			}
		}
		if !solved && abs(b2) <= rc {
			a0 := mod(a1-rc, n)
			delete(M[-rc], a1)
			M[0][a0] = b2
			hDiag := []int{-rc}
			hKey := []int{a1}
			hVal := []int{b2}
			hNew := []int{a0}
			ok, nodes := checkZero2RC(M, a0, b2, n, rc, &hDiag, &hKey, &hVal, &hNew)
			res.Nodes += nodes
			if ok {
				solved = true
			} else {
				delete(M[0], a0)
				M[-rc][a1] = b2
			}
		}
		if !solved {
			res.Solved = false
			return res
		}
	}

	rowOcc := make(map[int]int)
	for diag, rows := range M {
		for a := range rows {
			if prev, ok := rowOcc[a]; ok && prev != diag {
				res.Solved = false
				return res
			}
			rowOcc[a] = diag
		}
	}

	res.PR, res.PL = extractFactorsFromM(p, M, n, rc)
	if res.PR == nil {
		res.Solved = false
		return res
	}
	res.Solved = true
	return res
}

func checkZero2RC(
	M map[int]map[int]int,
	a0, b0, n, rc int,
	hDiag, hKey, hVal, hNew *[]int,
) (solved bool, nodes int64) {
	nodes = 1
	solved = true

	if b1, ok := M[rc][a0]; ok {
		if abs(b1) > rc {
			return false, nodes
		}
		aNew := mod(a0+rc, n)
		M[0][aNew] = b1
		delete(M[rc], a0)
		*hDiag = append(*hDiag, rc)
		*hKey = append(*hKey, a0)
		*hVal = append(*hVal, b1)
		*hNew = append(*hNew, aNew)
		ok, n2 := checkZero2RC(M, aNew, b1, n, rc, hDiag, hKey, hVal, hNew)
		nodes += n2
		if !ok {
			undoFrom(M, hDiag, hKey, hVal, hNew, len(*hDiag)-1)
			return false, nodes
		}
	}

	if b2, ok := M[-rc][a0]; ok && solved {
		if abs(b2) > rc {
			return false, nodes
		}
		aNew := mod(a0-rc, n)
		M[0][aNew] = b2
		delete(M[-rc], a0)
		*hDiag = append(*hDiag, -rc)
		*hKey = append(*hKey, a0)
		*hVal = append(*hVal, b2)
		*hNew = append(*hNew, aNew)
		ok, n2 := checkZero2RC(M, aNew, b2, n, rc, hDiag, hKey, hVal, hNew)
		nodes += n2
		if !ok {
			undoFrom(M, hDiag, hKey, hVal, hNew, len(*hDiag)-1)
			return false, nodes
		}
	}
	return true, nodes
}

func undoFrom(M map[int]map[int]int, hDiag, hKey, hVal, hNew *[]int, last int) {
	if last < 0 || last >= len(*hDiag) {
		return
	}
	d := (*hDiag)[last]
	a := (*hKey)[last]
	b := (*hVal)[last]
	an := (*hNew)[last]
	delete(M[0], an)
	M[d][a] = b
	*hDiag = (*hDiag)[:last]
	*hKey = (*hKey)[:last]
	*hVal = (*hVal)[:last]
	*hNew = (*hNew)[:last]
}

func extractFactorsFromM(p []int, M map[int]map[int]int, n, rc int) (pR, pL []int) {
	pL = make([]int, n)
	pR = make([]int, n)
	for i := range pL {
		pL[i] = -1
		pR[i] = -1
	}
	for d, rows := range M {
		for a, k := range rows {
			l := mod(a-k+d, n)
			if p[l]-l != k {
				continue
			}
			if pL[l] != -1 {
				return nil, nil
			}
			pL[l] = a
			pR[a] = p[l]
		}
	}
	usedL := make([]bool, n)
	usedR := make([]bool, n)
	for i := 0; i < n; i++ {
		if pL[i] >= 0 {
			usedL[i] = true
			if pL[i] < n {
				usedR[pL[i]] = true
			}
		}
	}
	for i := 0; i < n; i++ {
		if pL[i] < 0 {
			return nil, nil
		}
	}
	for i := 0; i < n; i++ {
		if pR[pL[i]] != p[i] {
			return nil, nil
		}
	}
	return pR, pL
}

func convertR(k, l, r, n int) int {
	return mod(k+l-r, n)
}

func mod(x, n int) int {
	x %= n
	if x < 0 {
		x += n
	}
	return x
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func sortInts(a []int) {
	for i := 1; i < len(a); i++ {
		v := a[i]
		j := i
		for j > 0 && a[j-1] > v {
			a[j] = a[j-1]
			j--
		}
		a[j] = v
	}
}

func Depth1SearchWithTimeout(p []int, rc int, maxNodes int64) Result {
	nodeLimit = maxNodes
	nodeCount = 0
	res := depth1SearchLimited(p, rc)
	nodeLimit = 0
	return res
}

var nodeLimit int64
var nodeCount int64

func depth1SearchLimited(p []int, rc int) Result {
	n := len(p)
	M := map[int]map[int]int{rc: {}, 0: {}, -rc: {}}
	res := Result{M: M}

	for i := 0; i < n; i++ {
		k := p[i] - i
		if k == 0 {
			a := convertR(k, i, 0, n)
			if _, ok := M[0][a]; ok {
				return res
			}
			M[0][a] = k
			continue
		}
		if k > 0 {
			a := convertR(k, i, rc, n)
			if _, ok := M[rc][a]; ok {
				return res
			}
			M[rc][a] = k
		} else {
			a := convertR(k, i, -rc, n)
			if _, ok := M[-rc][a]; ok {
				return res
			}
			M[-rc][a] = k
		}
	}

	var conflicts []int
	for a := range M[rc] {
		if _, ok := M[-rc][a]; ok {
			conflicts = append(conflicts, a)
		}
	}
	sortInts(conflicts)

	for _, a1 := range conflicts {
		b1, ok1 := M[rc][a1]
		b2, ok2 := M[-rc][a1]
		if !ok1 || !ok2 {
			continue
		}
		if abs(b1) > rc && abs(b2) > rc {
			return res
		}
		solved := false
		if abs(b1) <= rc {
			a0 := mod(a1+rc, n)
			delete(M[rc], a1)
			M[0][a0] = b1
			hDiag := []int{rc}
			hKey := []int{a1}
			hVal := []int{b1}
			hNew := []int{a0}
			ok := checkZero2RCLimited(M, a0, n, rc, &hDiag, &hKey, &hVal, &hNew)
			if ok {
				solved = true
			} else {
				delete(M[0], a0)
				M[rc][a1] = b1
				if nodeLimit > 0 && nodeCount >= nodeLimit {
					res.Nodes = nodeCount
					return res
				}
			}
		}
		if !solved && abs(b2) <= rc {
			a0 := mod(a1-rc, n)
			delete(M[-rc], a1)
			M[0][a0] = b2
			hDiag := []int{-rc}
			hKey := []int{a1}
			hVal := []int{b2}
			hNew := []int{a0}
			ok := checkZero2RCLimited(M, a0, n, rc, &hDiag, &hKey, &hVal, &hNew)
			if ok {
				solved = true
			} else {
				delete(M[0], a0)
				M[-rc][a1] = b2
			}
		}
		if !solved {
			res.Nodes = nodeCount
			return res
		}
	}
	res.Nodes = nodeCount
	res.PR, res.PL = extractFactorsFromM(p, M, n, rc)
	if res.PR == nil {
		return res
	}
	res.Solved = true
	return res
}

func checkZero2RCLimited(M map[int]map[int]int, a0, n, rc int, hDiag, hKey, hVal, hNew *[]int) bool {
	nodeCount++
	if nodeLimit > 0 && nodeCount >= nodeLimit {
		return false
	}
	if b1, ok := M[rc][a0]; ok {
		if abs(b1) > rc {
			return false
		}
		aNew := mod(a0+rc, n)
		M[0][aNew] = b1
		delete(M[rc], a0)
		*hDiag = append(*hDiag, rc)
		*hKey = append(*hKey, a0)
		*hVal = append(*hVal, b1)
		*hNew = append(*hNew, aNew)
		if !checkZero2RCLimited(M, aNew, n, rc, hDiag, hKey, hVal, hNew) {
			last := len(*hDiag) - 1
			delete(M[0], (*hNew)[last])
			M[(*hDiag)[last]][(*hKey)[last]] = (*hVal)[last]
			*hDiag = (*hDiag)[:last]
			*hKey = (*hKey)[:last]
			*hVal = (*hVal)[:last]
			*hNew = (*hNew)[:last]
			return false
		}
	}
	if b2, ok := M[-rc][a0]; ok {
		if abs(b2) > rc {
			return false
		}
		aNew := mod(a0-rc, n)
		M[0][aNew] = b2
		delete(M[-rc], a0)
		*hDiag = append(*hDiag, -rc)
		*hKey = append(*hKey, a0)
		*hVal = append(*hVal, b2)
		*hNew = append(*hNew, aNew)
		if !checkZero2RCLimited(M, aNew, n, rc, hDiag, hKey, hVal, hNew) {
			last := len(*hDiag) - 1
			delete(M[0], (*hNew)[last])
			M[(*hDiag)[last]][(*hKey)[last]] = (*hVal)[last]
			*hDiag = (*hDiag)[:last]
			*hKey = (*hKey)[:last]
			*hVal = (*hVal)[:last]
			*hNew = (*hNew)[:last]
			return false
		}
	}
	return true
}
