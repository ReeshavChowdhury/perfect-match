# Rizomiliotis–Triakosia HMM (Lattigo v5)

Packed square-matrix multiplication from Rizomiliotis and Triakosia (CCSW 2022): `AlgoA`, `AlgoB`, `AlgoC`. This binary compares sequential Ma vs concurrent AlgoA∥AlgoB + PCR + PBP.

`k` must divide `d`. `-b` must be `1`. Slot need is `d²k`. `m = ⌊S/(d²k)⌋` is computed from the CKKS slot count.

## Build

```bash
make all
```

- `bin/hmm-v5-safe` — one evaluator copy per worker (correct under concurrency)
- `bin/hmm-v5-unsafe` — shared evaluator buffers

## Run

From this directory:

```bash
GOMAXPROCS=20 ./bin/hmm-v5-safe -d 64 -k 4 -workers 20 -warmups 1 -trials 10
```

From the repository root:

```bash
GOMAXPROCS=20 ./ma_hmm_matrixmm_v5/bin/hmm-v5-safe -d 64 -k 4 -workers 20 -warmups 1 -trials 10
```

| Flag | Default | Constraint |
|---|---|---|
| `-d` | 64 | square size |
| `-k` | 2 | encoding space; must divide `d` |
| `-b` | 1 | must stay 1 |
| `-workers` | `min(CPU, 8)` | capped at `d/k` |
| `-warmups` | 1 | |
| `-trials` | 3 | |
| `-mode` | baked into the binary | `safe` or `unsafe` |
| `-optimized-only` | false | skip sequential Ma |

CKKS: LogN=15 when `d²k ≤ 32768`, else LogN=16 (need at most 65536). Conjugate-invariant ring.

Paper-table sizes (`S = 32768`): `(d,k) = (32,32), (32,8), (64,8), (64,4), (128,2)`.

```bash
make run-safe
make suite-safe
```

This is not Jiang HMM. Jiang is `make jiangmm-d64` from the repository root (`PERFECTMATCH_JIANGMM=1`).
