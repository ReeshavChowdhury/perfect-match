.PHONY: build test run quick paper sweep jiangmm clean

BIN := bin/perfectmatch-best

build:
	mkdir -p bin
	go build -o $(BIN) ./cmd/perfectmatch

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

# Complete Jiang C=A*B comparison. Ma's Sigma/Tau chains remain sequential;
# only the fused implementation uses QP workers and Sigma/Tau concurrency.
jiangmm: build
	GOMAXPROCS=10 PERFECTMATCH_JIANGMM=1 PERFECTMATCH_BEST=1 ADAPTIVE_D=8 PERFECTMATCH_WARMUPS=0 PERFECTMATCH_TRIALS=3 ./$(BIN)

clean:
	go clean
