.PHONY: build smoke run bench bench-d64 bench-d128 sweep-b2 sweep-b2-all sweep-b2-d64 sweep-b2-d256 clean

BIN := bin/hmt-transpose-v5

build:
	mkdir -p bin
	go build -o $(BIN) .

smoke: build
	GOMAXPROCS=4 ./$(BIN) -d 16 -ell 3 -max-fuse 2 -workers 4 -warmups 0 -trials 1

run: build
	GOMAXPROCS=8 ./$(BIN) -d 64 -ell 4 -max-fuse 3 -workers 8 -warmups 1 -trials 3

bench: bench-d64 bench-d128

bench-d64: build
	GOMAXPROCS=8 ./$(BIN) -d 64 -ell 4 -max-fuse 3 -workers 8 -warmups 1 -trials 3

bench-d128: build
	GOMAXPROCS=8 ./$(BIN) -d 128 -ell 4 -max-fuse 3 -workers 8 -warmups 1 -trials 3

sweep-b2: build
	GOMAXPROCS=8 ./$(BIN) -d 128 -sweep -calls 2 -workers 8 -warmups 0 -trials 3

sweep-b2-d64: build
	GOMAXPROCS=8 ./$(BIN) -d 64 -sweep -calls 2 -workers 8 -warmups 0 -trials 3

sweep-b2-d256: build
	GOMAXPROCS=8 ./$(BIN) -d 256 -sweep -calls 2 -n1 16 -workers 8 -warmups 0 -trials 3

sweep-b2-all: sweep-b2-d64 sweep-b2 sweep-b2-d256

clean:
	go clean
