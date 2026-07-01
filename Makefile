
NVCC  ?= nvcc
ARCH  ?= sm_89
NVCCFLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Iinclude -Isrc
LIBS  = -lcublas

BIN = bin/benchmark
SRC = src/benchmark.cu
HDRS = include/bench.cuh $(wildcard src/*.cuh)

$(BIN): $(SRC) $(HDRS)
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(BIN) $(LIBS)

build: $(BIN)

# full benchmark
run: $(BIN)
	bash scripts/benchmark.sh

# just one track: make square / make shape / make tensor
# square = Cuda Core Square GEMM (4092^3)
# shape = Cuda Core Tall / Skinny GEMM
# tensor = Tensor Core Square GEMM
square shape tensor: $(BIN)
	./$(BIN) $@

clean:
	rm -rf bin

.PHONY: build run square shape tensor clean
