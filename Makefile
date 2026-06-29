
NVCC ?= nvcc
ARCH ?= sm_89   # RTX 4090 (Ada)
NVCCFLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo
LIBS = -lcublas

SRC = main.cu
HDRS = bench.cuh kernels/k0_naive.cuh kernels/k1_coalesce.cuh \
       kernels/k2_tiling.cuh kernels/k3_microtile.cuh kernels/k4_vectorize.cuh \
       kernels/k5_warptile.cuh kernels/k6_cpasync.cuh kernels/k7_regdb.cuh \
       kernels/k8_splitk.cuh

matmul: $(SRC) $(HDRS)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o matmul $(LIBS)

run: matmul
	./matmul

# skinny (M=8) / tall (N=8) GEMM
shape: main_shape.cu $(HDRS)
	$(NVCC) $(NVCCFLAGS) main_shape.cu -o shape $(LIBS)
	./shape

# per-kernel register count and spill bytes. handy for comparing k6_cpasync vs k7_regdb
regs:
	$(NVCC) $(NVCCFLAGS) -Xptxas -v -c $(SRC) -o /dev/null

clean:
	rm -f matmul shape

.PHONY: run clean regs shape
