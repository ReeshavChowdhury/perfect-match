# PerfectMatch

This repository contains the Lattigo v5 PerfectMatch implementation, the Ma
HMM comparison, and the fused QP-parallel HMT transpose comparison. Vendored
dependencies are included, so the documented builds do not require network
access.

No generated binaries, build directories, benchmark logs, or result files are
included. Build commands create binaries locally under the relevant `bin/`
directory.

## Requirements

- Go 1.22 or newer
- Linux on amd64
- Eight or more CPU threads recommended for benchmark runs
- Sufficient RAM for the selected encrypted dimension; d=256 HMT sweeps are
  memory intensive

## Verify the source tree

From the repository root:

```bash
cd /home/user/Desktop/perfect_match_final_github
go test -mod=vendor ./...
```

## PerfectMatch Jiang implementation

Build:

```bash
make build
```

Generated executable:

```text
bin/perfectmatch-best
```

Run a quick correctness check:

```bash
make quick
```

Run the standard benchmark:

```bash
make run
```

Run the longer publication protocol:

```bash
make paper
```

Run the schedule sweep or complete encrypted Jiang multiplication:

```bash
make sweep
make jiangmm
```

## Ma HMM comparison

Build the safe and unsafe compatibility executables:

```bash
make -C ma_hmm_matrixmm_v5 all
```

Generated executables:

```text
ma_hmm_matrixmm_v5/bin/hmm-v5-safe
ma_hmm_matrixmm_v5/bin/hmm-v5-unsafe
```

Run the safe smoke test and standard safe comparison:

```bash
make -C ma_hmm_matrixmm_v5 smoke-safe
make -C ma_hmm_matrixmm_v5 run-safe
```

Run the complete safe configuration suite:

```bash
make -C ma_hmm_matrixmm_v5 suite-safe
```

The unsafe targets reproduce shared-buffer compatibility behavior and are not
correctness-qualified:

```bash
make -C ma_hmm_matrixmm_v5 smoke-unsafe
make -C ma_hmm_matrixmm_v5 run-unsafe
```

## Fused HMT transpose comparison

Build:

```bash
make -C hmt_transpose build
```

Generated executable:

```text
hmt_transpose/bin/hmt-transpose-v5
```

Run the encrypted smoke test:

```bash
make -C hmt_transpose smoke
```

Run the standard equal-depth d=64 comparison:

```bash
make -C hmt_transpose run
```

Run equal-depth d=64 and d=128 comparisons:

```bash
make -C hmt_transpose bench
```

Run all nine Algorithm-3 `B=2` schedules at one dimension:

```bash
make -C hmt_transpose sweep-b2-d64
make -C hmt_transpose sweep-b2
make -C hmt_transpose sweep-b2-d256
```

Run the streamed d=64, d=128, and d=256 sweep sequence:

```bash
make -C hmt_transpose sweep-b2-all
```

Direct invocation supports custom dimensions, decomposition depth, worker
count, warmups, and trials:

```bash
GOMAXPROCS=8 ./hmt_transpose/bin/hmt-transpose-v5 \
  -d 128 -ell 4 -max-fuse 3 -workers 8 -warmups 1 -trials 3

GOMAXPROCS=8 ./hmt_transpose/bin/hmt-transpose-v5 \
  -d 128 -sweep -calls 2 -workers 8 -warmups 0 -trials 3
```

## Cleaning generated binaries

```bash
make clean
make -C ma_hmm_matrixmm_v5 clean
make -C hmt_transpose clean
```
