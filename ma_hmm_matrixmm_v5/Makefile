.PHONY: all build-safe build-unsafe smoke-safe smoke-unsafe run-safe run-unsafe suite-safe suite-unsafe clean

SAFE_BIN := bin/hmm-v5-safe
UNSAFE_BIN := bin/hmm-v5-unsafe

all: build-safe build-unsafe

build-safe:
	mkdir -p bin
	go build -mod=vendor -ldflags '-X main.defaultMode=safe' -o $(SAFE_BIN) .

build-unsafe:
	mkdir -p bin
	go build -mod=vendor -ldflags '-X main.defaultMode=unsafe' -o $(UNSAFE_BIN) .

smoke-safe: build-safe
	GOMAXPROCS=4 ./$(SAFE_BIN) -d 8 -k 2 -workers 2 -warmups 0 -trials 1

smoke-unsafe: build-unsafe
	GOMAXPROCS=4 ./$(UNSAFE_BIN) -d 8 -k 2 -workers 2 -warmups 0 -trials 1

run-safe: build-safe
	GOMAXPROCS=8 ./$(SAFE_BIN) -d 64 -k 2 -workers 8 -warmups 0 -trials 3

run-unsafe: build-unsafe
	GOMAXPROCS=8 ./$(UNSAFE_BIN) -d 64 -k 2 -workers 8 -warmups 0 -trials 3

# Phase-13 configurations. These are intentionally long-running.
suite-safe: build-safe
	GOMAXPROCS=8 ./$(SAFE_BIN) -d 32 -k 32 -workers 8 -warmups 0 -trials 1
	GOMAXPROCS=8 ./$(SAFE_BIN) -d 32 -k 8  -workers 8 -warmups 0 -trials 1
	GOMAXPROCS=8 ./$(SAFE_BIN) -d 32 -k 4  -workers 8 -warmups 0 -trials 1
	GOMAXPROCS=8 ./$(SAFE_BIN) -d 64 -k 8  -workers 8 -warmups 0 -trials 1
	GOMAXPROCS=8 ./$(SAFE_BIN) -d 64 -k 4  -workers 8 -warmups 0 -trials 1
	GOMAXPROCS=8 ./$(SAFE_BIN) -d 64 -k 2  -workers 8 -warmups 0 -trials 1
	GOMAXPROCS=8 ./$(SAFE_BIN) -d 128 -k 2 -workers 8 -warmups 0 -trials 1
	GOMAXPROCS=8 ./$(SAFE_BIN) -d 128 -k 1 -workers 8 -warmups 0 -trials 1

suite-unsafe: build-unsafe
	GOMAXPROCS=8 ./$(UNSAFE_BIN) -d 32 -k 32 -workers 8 -warmups 0 -trials 1
	GOMAXPROCS=8 ./$(UNSAFE_BIN) -d 32 -k 8  -workers 8 -warmups 0 -trials 1
	GOMAXPROCS=8 ./$(UNSAFE_BIN) -d 32 -k 4  -workers 8 -warmups 0 -trials 1
	GOMAXPROCS=8 ./$(UNSAFE_BIN) -d 64 -k 8  -workers 8 -warmups 0 -trials 1
	GOMAXPROCS=8 ./$(UNSAFE_BIN) -d 64 -k 4  -workers 8 -warmups 0 -trials 1
	GOMAXPROCS=8 ./$(UNSAFE_BIN) -d 64 -k 2  -workers 8 -warmups 0 -trials 1
	GOMAXPROCS=8 ./$(UNSAFE_BIN) -d 128 -k 2 -workers 8 -warmups 0 -trials 1
	GOMAXPROCS=8 ./$(UNSAFE_BIN) -d 128 -k 1 -workers 8 -warmups 0 -trials 1

clean:
	rm -f $(SAFE_BIN) $(UNSAFE_BIN)
