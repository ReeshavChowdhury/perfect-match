package main

import "testing"

func TestFullDepth(t *testing.T) {
	for d, want := range map[int]int{8: 2, 16: 3, 64: 5, 128: 6, 256: 7} {
		if got := fullDepth(d); got != want {
			t.Fatalf("fullDepth(%d)=%d, want %d", d, got, want)
		}
	}
}

func TestTransposePlain(t *testing.T) {
	x := []float64{0, 1, 2, 3, 4, 5, 6, 7, 8}
	want := []float64{0, 3, 6, 1, 4, 7, 2, 5, 8}
	got := transposePlain(x, 3)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("transpose[%d]=%v, want %v", i, got[i], want[i])
		}
	}
}

func TestBestFusionRespectsBudget(t *testing.T) {
	const n = 8
	factors := make([]map[int][]float64, 5)
	for i := range factors {
		diag := make([]float64, n)
		for j := range diag {
			diag[j] = 1
		}
		factors[i] = map[int][]float64{0: diag}
	}
	packed, schedule, _, err := bestFusion(factors, n, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(packed) > 2 || len(schedule) != len(packed) {
		t.Fatalf("packed=%d schedule=%v, want at most two segments", len(packed), schedule)
	}
	covered := 0
	for _, width := range schedule {
		if width < 1 || width > 3 {
			t.Fatalf("invalid fused width %d", width)
		}
		covered += width
	}
	if covered != len(factors) {
		t.Fatalf("schedule %v covers %d factors, want %d", schedule, covered, len(factors))
	}
}

func TestEnumerateSchedulesB2(t *testing.T) {
	schedules := enumerateSchedules(2)
	if len(schedules) != 9 {
		t.Fatalf("got %d schedules, want 9", len(schedules))
	}
	seen := map[[2]int]bool{}
	for _, schedule := range schedules {
		key := [2]int{schedule[0], schedule[1]}
		seen[key] = true
	}
	for a := 1; a <= 3; a++ {
		for b := 1; b <= 3; b++ {
			if !seen[[2]int{a, b}] {
				t.Fatalf("missing schedule [%d,%d]", a, b)
			}
		}
	}
}
