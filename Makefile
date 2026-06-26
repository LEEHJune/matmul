
NVCC ?= nvcc
ARCH ?= native
NVCCFLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo
LIBS = -lcublas

SRC = main.cu
HDRS = bench.cuh kernels/k0_naive.cuh kernels/k1_coalesce.cuh \
       kernels/k2_tiling.cuh kernels/k3_microtile.cuh kernels/k4_vectorize.cuh \
       kernels/k5_warptile.cuh kernels/k6_cpasync.cuh kernels/k7_regdb.cuh

matmul: $(SRC) $(HDRS)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o matmul $(LIBS)

run: matmul
	./matmul

# 커널별 레지스터 수 + spill 바이트 확인 (k6_cpasync vs k7_regdb 비교용)
regs:
	$(NVCC) $(NVCCFLAGS) -Xptxas -v -c $(SRC) -o /dev/null

clean:
	rm -f matmul

.PHONY: run clean regs
