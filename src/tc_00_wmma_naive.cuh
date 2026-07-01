#pragma once
// T0: naive WMMA TF32 gemm. one warp computes one 16x16 output tile, loading A/B straight from global.
#include <mma.h>
#include <cuda_runtime.h>
using namespace nvcuda;

// TF32 WMMA tile. note K=8 for tf32 (fp16 path would be 16).
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

// assumes M,N multiples of 16 and K multiple of 8 (4096 is fine), so no bounds checks.
__global__ void t0_wmma_naive(const float *A, const float *B, float *C,
                              int M, int N, int K) {
    // map this warp to a (warpM, warpN) output tile of C
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int aRow = warpM * WMMA_M;
    int bCol = warpN * WMMA_N;

    for (int k = 0; k < K; k += WMMA_K) {
        // A tile: 16 rows x 8 cols at (aRow, k), row stride = K
        wmma::load_matrix_sync(a_frag, A + aRow * K + k, K);
        // B tile: 8 rows x 16 cols at (k, bCol), row stride = N
        wmma::load_matrix_sync(b_frag, B + k * N + bCol, N);

        // round each loaded element down to tf32, else mma takes the slow full-fp32 path
        for (int t = 0; t < a_frag.num_elements; t++)
            a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
        for (int t = 0; t < b_frag.num_elements; t++)
            b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + (warpM * WMMA_M) * N + warpN * WMMA_N,
                            c_frag, N, wmma::mem_row_major);
}

inline void launch_t0(const float *A, const float *B, float *C, int M, int N, int K) {
    // blockDim.x=128 -> 4 warps along M; blockDim.y=4 warps along N. each block = 64x64 of C.
    dim3 block(128, 4);
    dim3 grid((M / WMMA_M + 3) / 4, (N / WMMA_N + 3) / 4);
    t0_wmma_naive<<<grid, block>>>(A, B, C, M, N, K);
}
