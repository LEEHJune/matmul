#pragma once
// T2: warp tiling on top of T1. block (16 warps = 4x4) computes a 128x128 tile of C,
// each warp now owns a 32x32 = 2x2 grid of WMMA fragments (T1 had 1). loading a_frag/b_frag
// once and reusing across the 2x2 drops the smem-load:mma ratio from 2:1 (T1) to 1:1,
// so the tensor cores don't stall waiting on ldmatrix.
#include <mma.h>
#include <cuda_runtime.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

// 128x128 block tile. 4x4 warps, each warp = 32x32 -> 2x2 fragments.
#define T2_BM 128
#define T2_BN 128
#define T2_BK 16          // k-slab, multiple of WMMA_K -> 2 mma steps per slab
#define WARP_M 32
#define WARP_N 32
#define WMITER (WARP_M / WMMA_M)   // 2
#define WNITER (WARP_N / WMMA_N)   // 2

// 66 reg면 SM당 block 1개(occupancy 33%)에 갇힌다. <=64 reg로 강제해서 2 block(67%) 확보.
__global__ void __launch_bounds__(512, 2)
t2_wmma_warptile(const float *A, const float *B, float *C,
                                 int M, int N, int K) {
    int blockRow = blockIdx.x * T2_BM;
    int blockCol = blockIdx.y * T2_BN;

    // this warp's 32x32 slot inside the 128x128 block tile
    int warpRow = (threadIdx.x / warpSize) * WARP_M;   // 0,32,64,96
    int warpCol = (threadIdx.y) * WARP_N;              // 0,32,64,96

    int tid = threadIdx.y * blockDim.x + threadIdx.x;  // 0..511
    int nThreads = blockDim.x * blockDim.y;            // 512

    __shared__ float As[T2_BM][T2_BK];   // 128x16
    __shared__ float Bs[T2_BK][T2_BN];   // 16x128

    // 2x2 grid of accumulators stays live across the whole K loop
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WMITER][WNITER];
    for (int i = 0; i < WMITER; i++)
        for (int j = 0; j < WNITER; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += T2_BK) {
        // stage A/B block tiles into smem, coalesced + tf32-rounded once
        for (int i = tid; i < T2_BM * T2_BK; i += nThreads) {
            int r = i / T2_BK, c = i % T2_BK;
            As[r][c] = wmma::__float_to_tf32(A[(blockRow + r) * K + (k0 + c)]);
        }
        for (int i = tid; i < T2_BK * T2_BN; i += nThreads) {
            int r = i / T2_BN, c = i % T2_BN;
            Bs[r][c] = wmma::__float_to_tf32(B[(k0 + r) * N + (blockCol + c)]);
        }
        __syncthreads();

        for (int kk = 0; kk < T2_BK; kk += WMMA_K) {
            // load the warp's row of A frags and col of B frags once...
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag[WMITER];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag[WNITER];
            for (int i = 0; i < WMITER; i++)
                wmma::load_matrix_sync(a_frag[i], &As[warpRow + i * WMMA_M][kk], T2_BK);
            for (int j = 0; j < WNITER; j++)
                wmma::load_matrix_sync(b_frag[j], &Bs[kk][warpCol + j * WMMA_N], T2_BN);

            // ...then reuse them across the 2x2 outer product
            for (int i = 0; i < WMITER; i++)
                for (int j = 0; j < WNITER; j++)
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }
        __syncthreads();
    }

    for (int i = 0; i < WMITER; i++)
        for (int j = 0; j < WNITER; j++)
            wmma::store_matrix_sync(
                C + (blockRow + warpRow + i * WMMA_M) * N + (blockCol + warpCol + j * WMMA_N),
                c_frag[i][j], N, wmma::mem_row_major);
}

inline void launch_t2(const float *A, const float *B, float *C, int M, int N, int K) {
    dim3 block(128, 4);                       // 16 warps
    dim3 grid(M / T2_BM, N / T2_BN);          // one block per 128x128 of C
    t2_wmma_warptile<<<grid, block>>>(A, B, C, M, N, K);
}
