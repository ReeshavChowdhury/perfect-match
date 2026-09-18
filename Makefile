.PHONY: build test run quick paper sweep jiangmm jiangmm-d8 jiangmm-d64 jiangmm-d128 clean

BIN := bin/perfectmatch-best

build:
	mkdir -p bin
	go build -mod=vendor -o $(BIN) ./cmd/perfectmatch

test:
	go test ./...

run: build
	GOMAXPROCS=10 PERFECTMATCH_BEST=1 ./$(BIN)

quick: build
	GOMAXPROCS=10 PERFECTMATCH_BEST=1 PERFECTMATCH_WARMUPS=1 PERFECTMATCH_TRIALS=1 ./$(BIN)

paper: build
	GOMAXPROCS=10 PERFECTMATCH_BEST=1 PERFECTMATCH_WARMUPS=5 PERFECTMATCH_TRIALS=15 ./$(BIN)

sweep: build
	GOMAXPROCS=10 PERFECTMATCH_SWEEP=1 PERFECTMATCH_WARMUPS=1 PERFECTMATCH_TRIALS=3 ./$(BIN)

# Jiang HMM (Jiang et al. CCS 2018): sequential Ma Σ/Τ vs fused/parallel Σ/Τ,
# then the same hoisted outer product. Always set ADAPTIVE_D to a single size.
jiangmm: jiangmm-d8

jiangmm-d8: build
	GOMAXPROCS=10 PERFECTMATCH_JIANGMM=1 PERFECTMATCH_BEST=1 ADAPTIVE_D=8 PERFECTMATCH_WARMUPS=0 PERFECTMATCH_TRIALS=3 ./$(BIN)

jiangmm-d64: build
	GOMAXPROCS=20 PERFECTMATCH_JIANGMM=1 PERFECTMATCH_BEST=1 ADAPTIVE_D=64 PERFECTMATCH_WARMUPS=1 PERFECTMATCH_TRIALS=3 ./$(BIN)

jiangmm-d128: build
	GOMAXPROCS=20 PERFECTMATCH_JIANGMM=1 PERFECTMATCH_BEST=1 ADAPTIVE_D=128 PERFECTMATCH_WARMUPS=1 PERFECTMATCH_TRIALS=3 ./$(BIN)

clean:
	go clean
