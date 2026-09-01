package matching

func HopcroftKarp(nL, nR int, adj [][]int) (matchL, matchR []int, size int) {
	const inf = 1<<62 - 1
	matchL = make([]int, nL)
	matchR = make([]int, nR)
	for i := range matchL {
		matchL[i] = -1
	}
	for i := range matchR {
		matchR[i] = -1
	}
	dist := make([]int, nL)
	queue := make([]int, nL)

	bfs := func() bool {
		head, tail := 0, 0
		for u := 0; u < nL; u++ {
			if matchL[u] == -1 {
				dist[u] = 0
				queue[tail] = u
				tail++
			} else {
				dist[u] = inf
			}
		}
		found := false
		for head < tail {
			u := queue[head]
			head++
			for _, v := range adj[u] {
				w := matchR[v]
				if w == -1 {
					found = true
				} else if dist[w] == inf {
					dist[w] = dist[u] + 1
					queue[tail] = w
					tail++
				}
			}
		}
		return found
	}

	var dfs func(u int) bool
	dfs = func(u int) bool {
		for _, v := range adj[u] {
			w := matchR[v]
			if w == -1 || (dist[w] == dist[u]+1 && dfs(w)) {
				matchL[u] = v
				matchR[v] = u
				return true
			}
		}
		dist[u] = inf
		return false
	}

	for bfs() {
		for u := 0; u < nL; u++ {
			if matchL[u] == -1 {
				if dfs(u) {
					size++
				}
			}
		}
	}
	return
}

func HopcroftKarpSquare(n int, adj [][]int) (matchL, matchR []int, size int) {
	return HopcroftKarp(n, n, adj)
}

func Degree3Match(n int, adj [][]int) (matchL, matchR []int, size int) {
	matchL = make([]int, n)
	matchR = make([]int, n)
	seen := make([]int, n)
	for i := range matchL {
		matchL[i] = -1
		matchR[i] = -1
	}
	stamp := 0
	var aug func(int) bool
	aug = func(u int) bool {
		for _, v := range adj[u] {
			if seen[v] == stamp {
				continue
			}
			seen[v] = stamp
			if matchR[v] == -1 || aug(matchR[v]) {
				matchL[u], matchR[v] = v, u
				return true
			}
		}
		return false
	}
	for u := 0; u < n; u++ {
		stamp++
		if aug(u) {
			size++
		}
	}
	return
}
