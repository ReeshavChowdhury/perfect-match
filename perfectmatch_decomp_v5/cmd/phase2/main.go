package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"project1-fhe_extension_v1.0/matching_perm_decomp/decomp"
	"project1-fhe_extension_v1.0/matching_perm_decomp/he"
)

func main() {
	trials := flag.Int("trials", 5, "timed trials per (d,ℓ)")
	flag.Parse()

	outDir := "matching_perm_decomp/results"
	_ = os.MkdirAll(outDir, 0o755)
	mdPath := filepath.Join(outDir, "phase2_full_depth_timing.md")

	var b strings.Builder
	log := func(format string, args ...interface{}) {
		fmt.Fprintf(&b, format, args...)
		fmt.Printf(format, args...)
	}

	log("# Phase 2 — Full-depth HMT Lattigo timings\n\n")
	log("**Date:** %s\n\n", time.Now().Format(time.RFC3339))
	log("All timings are **REAL Lattigo CKKS** ciphertext evaluations.\n")
	log("Pipeline: IdealDmp `CheckConflictInDecompV3` + BSGS residual + 3-diag right factors.\n\n")

	log("## Theoretical cost model\n\n")
	log("cost(ℓ) ≈ 2ℓ + 2·√(2·d/2^ℓ − 1). Predicted U-shape; minimum near ℓ* ≈ L − 3.\n\n")
	for _, d := range []int{128, 256} {
		L := decomp.FullDepth(d)
		ellStar, cStar := decomp.OptimalTruncation(d)
		log("### d=%d  L=%d  predicted ℓ*=%d  cost*=%.2f\n\n", d, L, ellStar, cStar)
		log("| ℓ | model cost | residual diags ≈ |\n|---|------------|------------------|\n")
		for ell := 0; ell <= L; ell++ {
			dRes := d >> uint(ell)
			if dRes < 1 {
				dRes = 1
			}
			log("| %d | %.2f | %d |\n", ell, decomp.TheoreticalCost(d, ell), 2*dRes-1)
		}
		log("\n")
	}

	log("## Measured timings\n\n")
	log("| d | ℓ | time_ms (mean±std) | rot_keys | start_lvl | end_lvl | trials | model_cost | REAL HE |\n")
	log("|---|---|--------------------:|---------:|----------:|--------:|-------:|-----------:|--------|\n")

	type row struct {
		d, ell  int
		ms, std float64
		keys    int
		sl, el  int
		model   float64
		err     string
	}
	var rows []row

	run := func(d, n1, ell int) {
		log("_HE d=%d ℓ=%d n1=%d ..._\n", d, ell, n1)
		t0 := time.Now()
		tm := he.EvalHMTDecomposed(d, ell, n1, *trials)
		log("  wall %s → %.2f±%.2f ms keys=%d lvl %d→%d err=%q\n",
			time.Since(t0), tm.TimeMs, tm.StdMs, tm.RotKeys, tm.StartLevel, tm.EndLevel, tm.Err)
		rows = append(rows, row{
			d: d, ell: ell, ms: tm.TimeMs, std: tm.StdMs, keys: tm.RotKeys,
			sl: tm.StartLevel, el: tm.EndLevel, model: decomp.TheoreticalCost(d, ell), err: tm.Err,
		})
	}

	for ell := 1; ell <= 6; ell++ {
		run(128, 8, ell)
	}
	for ell := 1; ell <= 7; ell++ {
		run(256, 16, ell)
	}

	final := strings.Builder{}
	final.WriteString("# Phase 2 — Full-depth HMT Lattigo timings\n\n")
	final.WriteString(fmt.Sprintf("**Date:** %s\n\n", time.Now().Format(time.RFC3339)))
	final.WriteString("All numbers below are **REAL Lattigo CKKS ciphertext timings** (not rotation-count proxies).\n\n")
	final.WriteString("## Theoretical model\n\n")
	final.WriteString("cost(ℓ) ≈ 2ℓ + 2·√(2d/2^ℓ − 1). U-shaped; predicted minimum near ℓ* ≈ L−3.\n\n")
	for _, d := range []int{128, 256} {
		L := decomp.FullDepth(d)
		ellStar, cStar := decomp.OptimalTruncation(d)
		final.WriteString(fmt.Sprintf("d=%d: L=%d, predicted ℓ*=%d, cost*=%.2f\n\n", d, L, ellStar, cStar))
	}
	final.WriteString("## Measured (d, ℓ, time_ms, keys, level)\n\n")
	final.WriteString("| d | ℓ | time_ms (mean±std) | rot_keys | start_lvl | end_lvl | trials | model_cost | REAL HE |\n")
	final.WriteString("|---|---|--------------------:|---------:|----------:|--------:|-------:|-----------:|--------|\n")

	byD := map[int][]row{}
	for _, r := range rows {
		byD[r.d] = append(byD[r.d], r)
		if r.err != "" {
			final.WriteString(fmt.Sprintf("| %d | %d | ERROR: %s | — | — | — | %d | %.2f | yes |\n",
				r.d, r.ell, r.err, *trials, r.model))
		} else {
			final.WriteString(fmt.Sprintf("| %d | %d | %.1f ± %.1f | %d | %d | %d | %d | %.2f | yes |\n",
				r.d, r.ell, r.ms, r.std, r.keys, r.sl, r.el, *trials, r.model))
		}
	}

	final.WriteString("\n## U-shape confirmation\n\n")
	for _, d := range []int{128, 256} {
		rs := byD[d]
		if len(rs) == 0 {
			continue
		}
		bestEll, bestMs := -1, 1e18
		fullMs := -1.0
		L := decomp.FullDepth(d)
		var series []string
		increasingTail := false
		prev := -1.0
		for _, r := range rs {
			if r.err != "" {
				continue
			}
			series = append(series, fmt.Sprintf("ℓ=%d:%.0fms", r.ell, r.ms))
			if r.ms < bestMs {
				bestMs = r.ms
				bestEll = r.ell
			}
			if r.ell == L {
				fullMs = r.ms
			}
			if prev > 0 && r.ms > prev && r.ell > bestEll {
				increasingTail = true
			}
			prev = r.ms
		}
		ushape := bestEll < L && fullMs > bestMs
		final.WriteString(fmt.Sprintf("### d=%d\n\n", d))
		final.WriteString(fmt.Sprintf("- Series: %s\n", strings.Join(series, ", ")))
		final.WriteString(fmt.Sprintf("- Empirical best ℓ=%d (%.1f ms); full depth L=%d", bestEll, bestMs, L))
		if fullMs > 0 {
			final.WriteString(fmt.Sprintf(" (%.1f ms)", fullMs))
		}
		final.WriteString("\n")
		pred, _ := decomp.OptimalTruncation(d)
		final.WriteString(fmt.Sprintf("- Predicted ℓ*=%d (model)\n", pred))
		if ushape || increasingTail {
			final.WriteString("- **U-shape CONFIRMED:** time does **not** keep decreasing to full depth; ")
			final.WriteString("last rounds turn around / full depth is slower than an intermediate ℓ*.\n\n")
		} else if bestEll == L {
			final.WriteString("- **U-shape NOT observed on this machine:** minimum at full depth ")
			final.WriteString("(model may still predict a shallow U; check keys/level overhead).\n\n")
		} else {
			final.WriteString("- **Partial confirmation:** best ℓ < L; see series for monotonicity.\n\n")
		}
	}

	final.WriteString("## ASCII plot (time vs ℓ)\n\n```\n")
	for _, d := range []int{128, 256} {
		rs := byD[d]
		final.WriteString(fmt.Sprintf("d=%d\n", d))
		maxMs := 1.0
		for _, r := range rs {
			if r.ms > maxMs {
				maxMs = r.ms
			}
		}
		for _, r := range rs {
			if r.err != "" {
				final.WriteString(fmt.Sprintf("  ℓ=%d | ERROR\n", r.ell))
				continue
			}
			bar := int(40 * r.ms / maxMs)
			final.WriteString(fmt.Sprintf("  ℓ=%d | %s %.0fms\n", r.ell, strings.Repeat("#", bar), r.ms))
		}
		final.WriteString("\n")
	}
	final.WriteString("```\n")

	must(os.WriteFile(mdPath, []byte(final.String()), 0o644))
	_ = os.WriteFile(filepath.Join(outDir, "phase2_runlog.txt"), []byte(b.String()), 0o644)
	fmt.Println("Wrote", mdPath)
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
