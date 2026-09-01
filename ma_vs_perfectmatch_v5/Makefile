.PHONY: build run64 run128

build:
	$(MAKE) -C ../ma_hmm_matrixmm_v5 all

run64: build
	GOMAXPROCS=20 ../ma_hmm_matrixmm_v5/bin/hmm-v5-safe -d 64 -k 4 -workers 20 -warmups 0 -trials 3

run128: build
	GOMAXPROCS=20 ../ma_hmm_matrixmm_v5/bin/hmm-v5-safe -d 128 -k 2 -workers 20 -warmups 0 -trials 3
