# PerfectMatch

Lattigo v5 code for PerfectMatch permutation fusion, **Rizomiliotis–Triakosia HMM**, **Jiang HMM**, and HMT transpose.

## Requirements

- Go 1.22 or newer
- Linux on amd64
- Run all commands from the repository root
- Builds use `-mod=vendor` (a `vendor/` tree is included)

Set `GOMAXPROCS` to the number of cores you want used. On a 20-thread i9-10900, `GOMAXPROCS=20` is appropriate.

---

## Rizomiliotis–Triakosia HMM

This is the packed square-matrix product from Rizomiliotis and Triakosia, CCSW 2022 (`AlgoA` / `AlgoB` / `AlgoC`). The binary in `ma_hmm_matrixmm_v5/` compares:

- **Ma seq:** sequential AlgoA, AlgoB, then AlgoC
- **Ours:** concurrent AlgoA∥AlgoB, then PCR + PBP (parallel AlgoC blocks)

`k` is the encoding-space parameter (`k` copies of the `d×d` matrix; `k` must divide `d`). `b` must be `1`. Slot need is `d²k`. Packing factor `m = ⌊S / (d²k)⌋` is computed automatically.

### CKKS parameters

| Condition | LogN | LogQ | LogP | scale | ring | slots `S` |
|---|---|---|---|---|---|---|
| `d²k ≤ 32768` | 15 | 51 + 17×40 | 3×50 | 2⁴⁰ | conjugate-invariant | 32768 |
| `32768 < d²k ≤ 65536` | 16 | 55 + 17×45 | 3×55 | 2⁴⁵ | conjugate-invariant | 65536 |

### Build

```bash
make -C ma_hmm_matrixmm_v5 all
```

Binaries: `ma_hmm_matrixmm_v5/bin/hmm-v5-safe` and `ma_hmm_matrixmm_v5/bin/hmm-v5-unsafe`.

Use **safe** for correct concurrent evaluators. **unsafe** shares evaluator buffers (faster to allocate, not thread-safe).

### Run one size

```bash
GOMAXPROCS=20 ./ma_hmm_matrixmm_v5/bin/hmm-v5-safe \
  -d 64 -k 4 -workers 20 -warmups 1 -trials 10
```

| Flag | Default | Meaning |
|---|---|---|
| `-d` | 64 | square matrix size |
| `-k` | 2 | encoding-space parameter; must divide `d` |
| `-b` | 1 | replication grouping; **must stay 1** |
| `-workers` | `min(CPU, 8)` | parallel AlgoC blocks; internally capped at `d/k` |
| `-warmups` | 1 | discarded paired runs |
| `-trials` | 3 | timed paired runs (printed as median/std in this tree) |
| `-mode` | `safe` | `safe` or `unsafe` (the `hmm-v5-*` binaries bake this in) |
| `-optimized-only` | false | skip sequential Ma and time only PCR/PBP |

A successful line looks like:

```text
HMM_V5 mode=safe d=64 k=4 b=1 m=2 workers=16 MaSeq=...ms PCR_PBP=...ms speedup=...x ... correct=true
```

### Paper-table sizes

These match the large-`d` R&T / Ma-seq vs Ours table (`m` is for `S = 32768`):

```bash
BIN=./ma_hmm_matrixmm_v5/bin/hmm-v5-safe
export GOMAXPROCS=20
$BIN -d 32  -k 32 -workers 20 -warmups 1 -trials 10   # m=1, workers→1
$BIN -d 32  -k 8  -workers 20 -warmups 1 -trials 10   # m=4, workers→4
$BIN -d 64  -k 8  -workers 20 -warmups 1 -trials 10   # m=1, workers→8
$BIN -d 64  -k 4  -workers 20 -warmups 1 -trials 10   # m=2, workers→16
$BIN -d 128 -k 2  -workers 20 -warmups 1 -trials 10   # m=1, workers→20
```

Makefile shortcuts:

```bash
make -C ma_hmm_matrixmm_v5 run-safe      # d=64 k=2, 8 workers, 3 trials
make -C ma_hmm_matrixmm_v5 suite-safe    # full (d,k) sweep, 1 trial each
make -C ma_vs_perfectmatch_v5 run64      # d=64 k=4, 20 workers
make -C ma_vs_perfectmatch_v5 run128     # d=128 k=2, 20 workers
```

`d=128` is slow and memory-heavy (Galois keys + per-worker evaluator buffers). Prefer `hmm-v5-safe`. LogN=16 is selected only when `d²k > 32768` (for example `d=128 k=4`).

---

## Jiang HMM

Jiang, Kim, Lauter et al. (CCS 2018) packed matrix product: Σ/Τ linear transforms, then a hoisted outer product. The comparison is **Ma sequential Σ/Τ + Jiang outer** vs **PerfectMatch fused/parallel Σ/Τ + the same Jiang outer**.

This is **not** the R&T binary. It lives in `cmd/perfectmatch` and is enabled with `PERFECTMATCH_JIANGMM=1`.

CKKS parameters are always LogN=16, `LogQ = {55}+17×{45}`, `LogP = 3×{55}`, scale 2⁴⁵, conjugate-invariant.

### Build

```bash
make build
```

Executable: `bin/perfectmatch-best`.

### Run (required flags)

You must set **both**:

- `PERFECTMATCH_JIANGMM=1` — Jiang C=A·B path (also generates the relinearization key)
- `PERFECTMATCH_BEST=1` — skip PerfectMatch-only permutation benchmarks
- `ADAPTIVE_D` — **one** matrix size. If you omit it, `PERFECTMATCH_BEST=1` would otherwise try `d = 64, 128, 256`

```bash
make jiangmm                 # d=8, 3 trials (smoke)
make jiangmm-d64             # d=64
make jiangmm-d128            # d=128; many rotation keys, several GB RAM
```

Equivalent explicit command:

```bash
GOMAXPROCS=20 \
PERFECTMATCH_JIANGMM=1 \
PERFECTMATCH_BEST=1 \
ADAPTIVE_D=64 \
PERFECTMATCH_WARMUPS=1 \
PERFECTMATCH_TRIALS=10 \
./bin/perfectmatch-best
```

| Variable | Role |
|---|---|
| `PERFECTMATCH_JIANGMM=1` | enable Jiang matrix product |
| `PERFECTMATCH_BEST=1` | only the Ma-vs-fused comparison |
| `ADAPTIVE_D` | `8`, `64`, `128`, or `256` |
| `PERFECTMATCH_WARMUPS` | discarded paired runs (default 2 if unset) |
| `PERFECTMATCH_TRIALS` | timed runs (default 3 if unset) |
| `PERFECTMATCH_JIANGMM_DEBUG=1` | print outer-product ciphertext levels |
| `ADAPTIVE_GROUPS` | force fusion groups, e.g. `2,1` |
| `QP_MAX_ACTIVE` | cap concurrent QP workers |

A successful line looks like:

```text
ADAPTIVE_FUSED_V5 d=64 logN=16 slots=65536 ...
  JIANG_MM d=64 groups=[...] Ma=...ms FusedParallel=...ms speedup=...x ... correct=true
```

`Ma` here is sequential Σ/Τ (not R&T AlgoC). `FusedParallel` is PerfectMatch on Σ/Τ, then the same Jiang outer product.

Do not run Jiang at `d=256` unless you have a large RAM budget: every shift `k, k-d, k·d, (k-d)·d` for `k = 1…d-1` is materialized as a Galois key.

---

## PerfectMatch (permutations only)

No matrix product; Σ/Τ fusion only.

```bash
make build
make run      # d=64,128,256  (PERFECTMATCH_BEST=1)
make quick    # 1 trial
make paper    # 5 warmups, 15 trials
make sweep    # schedule sweep at d=128
```

```bash
GOMAXPROCS=10 PERFECTMATCH_BEST=1 ADAPTIVE_D=128 ./bin/perfectmatch-best
```

---

## HMT transpose

```bash
make -C hmt_transpose build
./hmt_transpose/bin/hmt-transpose-v5
```

---

## Tests

```bash
go test ./...
```
