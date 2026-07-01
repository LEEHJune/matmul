#pragma once
// T3: cp.async double buffering on top of T2's warp tiling. two smem buffers ping-pong:
// while the tensor cores chew on slab s (buf[s&1]), cp.async streams slab s+1 into the
// other buffer in the background. kills the hard load->sync->compute serialization T2 had,
// so the gmem latency hides behind compute instead of behind occupancy.
#include <mma.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>   // __pipeline_memcpy_async / commit / wait_prior
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

#define T3_BM 128
#define T3_BN 128
#define T3_BK 16
#define WARP_M 32
#define WARP_N 32
#define WMITER (WARP_M / WMMA_M)   // 2
#define WNITER (WARP_N / WMMA_N)   // 2

// issue async copies for one k-slab (gmem -> the given smem buffer). single-float copies
// for now; float4 is the next rung. caller does commit()/wait().
__device__ __forceinline__ void t3_load_slab(
        float As[T3_BM][T3_BK], float Bs[T3_BK][T3_BN],
        const float *A, const float *B,
        int blockRow, int blockCol, int k0, int K, int N,
        int tid, int nThreads) {
    for (int i = tid; i < T3_BM * T3_BK; i += nThreads) {
        int r = i / T3_BK, c = i % T3_BK;
        __pipeline_memcpy_async(&As[r][c], &A[(blockRow + r) * K + (k0 + c)], sizeof(float));
    }
    for (int i = tid; i < T3_BK * T3_BN; i += nThreads) {
        int r = i / T3_BN, c = i % T3_BN;
        __pipeline_memcpy_async(&Bs[r][c], &B[(k0 + r) * N + (blockCol + c)], sizeof(float));
    }
}

__global__ void t3_wmma_doublebuf(const float *A, const float *B, float *C,
                                  int M, int N, int K) {
    int blockRow = blockIdx.x * T3_BM;
    int blockCol = blockIdx.y * T3_BN;

    int warpRow = (threadIdx.x / warpSize) * WARP_M;   // 0,32,64,96
    int warpCol = (threadIdx.y) * WARP_N;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;  // 0..511
    int nThreads = blockDim.x * blockDim.y;            // 512

    // double-buffered smem: 2 * (128x16 + 16x128) * 4B = 32KB
    __shared__ float As[2][T3_BM][T3_BK];
    __shared__ float Bs[2][T3_BK][T3_BN];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WMITER][WNITER];
    for (int i = 0; i < WMITER; i++)
        for (int j = 0; j < WNITER; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    int nSlabs = K / T3_BK;

    // prologue: kick off slab 0 into buf 0
    t3_load_slab(As[0], Bs[0], A, B, blockRow, blockCol, 0, K, N, tid, nThreads);
    __pipeline_commit();

    for (int s = 0; s < nSlabs; s++) {
        if (s + 1 < nSlabs) {
            // prefetch next slab into the other buffer; it streams while we compute below
            int nxt = (s + 1) & 1;
            t3_load_slab(As[nxt], Bs[nxt], A, B, blockRow, blockCol,
                         (s + 1) * T3_BK, K, N, tid, nThreads);
            __pipeline_commit();
            __pipeline_wait_prior(1);   // keep only the just-issued next group in flight
        } else {
            __pipeline_wait_prior(0);   // last slab: drain everything
        }
        __syncthreads();

        int cur = s & 1;
        for (int kk = 0; kk < T3_BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag[WMITER];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag[WNITER];
            for (int i = 0; i < WMITER; i++) {
                wmma::load_matrix_sync(a_frag[i], &As[cur][warpRow + i * WMMA_M][kk], T3_BK);
                // cp.async copied raw fp32 -> must round to tf32 here (couldn't round during the copy)
                for (int t = 0; t < a_frag[i].num_elements; t++)
                    a_frag[i].x[t] = wmma::__float_to_tf32(a_frag[i].x[t]);
            }
            for (int j = 0; j < WNITER; j++) {
                wmma::load_matrix_sync(b_frag[j], &Bs[cur][kk][warpCol + j * WMMA_N], T3_BN);
                for (int t = 0; t < b_frag[j].num_elements; t++)
                    b_frag[j].x[t] = wmma::__float_to_tf32(b_frag[j].x[t]);
            }
            for (int i = 0; i < WMITER; i++)
                for (int j = 0; j < WNITER; j++)
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }
        __syncthreads();   // done reading buf[cur] before a later iter overwrites it
    }

    for (int i = 0; i < WMITER; i++)
        for (int j = 0; j < WNITER; j++)
            wmma::store_matrix_sync(
                C + (blockRow + warpRow + i * WMMA_M) * N + (blockCol + warpCol + j * WMMA_N),
                c_frag[i][j], N, wmma::mem_row_major);
}

inline void launch_t3(const float *A, const float *B, float *C, int M, int N, int K) {
    dim3 block(128, 4);
    dim3 grid(M / T3_BM, N / T3_BN);
    t3_wmma_doublebuf<<<grid, block>>>(A, B, C, M, N, K);
}
