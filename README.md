# PerfectMatch

Encrypted matrix computation implementations and comparisons built with
Lattigo v5.

## Requirements

- Go 1.22 or newer
- Linux on amd64

## Build

Run the following commands from the repository root.

Build PerfectMatch:

```bash
make build
```

The executable is created at `bin/perfectmatch-best`.

Build the Ma HMM comparison:

```bash
make -C ma_hmm_matrixmm_v5 all
```

The executables are created under `ma_hmm_matrixmm_v5/bin/`.

Build the HMT transpose comparison:

```bash
make -C hmt_transpose build
```

The executable is created at `hmt_transpose/bin/hmt-transpose-v5`.
