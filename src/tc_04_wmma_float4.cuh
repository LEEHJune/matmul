#pragma once
// T4: T3 + float4 vectorized cp.async. only the gmem->smem copy changes: 4-byte copies
// become 16-byte (float4), so cp.async issues 1/4 as many instructions and gmem moves in
// 128-bit transactions. everything else (warp tiling, double buffer, tf32 rounding) == T3.
// hypothesis: ncu showed DRAM only ~14% busy, so this should barely move -- the real
// bottleneck is smem bank conflicts, not gmem. this kernel is the control experiment.
#include <mma.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

#define T4_BM 128
#define T4_BN 128
#define T4_BK 16
#define WARP_M 32
#define WARP_N 32
#define WMITER (WARP_M / WMMA_M)
#define WNITER (WARP_N / WMMA_N)

// async copy one k-slab, now in float4 chunks (16B each). aligned because K,N,blockRow/Col,
// k0 are all multiples of 4 and smem rows are 16B-aligned.
__device__ __forceinline__ void t4_load_slab(
        float As[T4_BM][T4_BK], float Bs[T4_BK][T4_BN],
        const float *A, const float *B,
        int blockRow, int blockCol, int k0, int K, int N,
        int tid, int nThreads) {
    for (int i = tid; i < (T4_BM * T4_BK) / 4; i += nThreads) {
        int e = i * 4;
        int r = e / T4_BK, c = e % T4_BK;
        __pipeline_memcpy_async(&As[r][c], &A[(blockRow + r) * K + (k0 + c)], sizeof(float4));
    }
    for (int i = tid; i < (T4_BK * T4_BN) / 4; i += nThreads) {
        int e = i * 4;
        int r = e / T4_BN, c = e % T4_BN;
        __pipeline_memcpy_async(&Bs[r][c], &B[(k0 + r) * N + (blockCol + c)], sizeof(float4));
    }
}

__global__ void t4_wmma_float4(const float *A, const float *B, float *C,
                               int M, int N, int K) {
    int blockRow = blockIdx.x * T4_BM;
    int blockCol = blockIdx.y * T4_BN;

    int warpRow = (threadIdx.x / warpSize) * WARP_M;
    int warpCol = (threadIdx.y) * WARP_N;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nThreads = blockDim.x * blockDim.y;

    __shared__ float As[2][T4_BM][T4_BK];
    __shared__ float Bs[2][T4_BK][T4_BN];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WMITER][WNITER];
    for (int i = 0; i < WMITER; i++)
        for (int j = 0; j < WNITER; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    int nSlabs = K / T4_BK;

    t4_load_slab(As[0], Bs[0], A, B, blockRow, blockCol, 0, K, N, tid, nThreads);
    __pipeline_commit();

    for (int s = 0; s < nSlabs; s++) {
        if (s + 1 < nSlabs) {
            int nxt = (s + 1) & 1;
            t4_load_slab(As[nxt], Bs[nxt], A, B, blockRow, blockCol,
                         (s + 1) * T4_BK, K, N, tid, nThreads);
            __pipeline_commit();
            __pipeline_wait_prior(1);
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();

        int cur = s & 1;
        for (int kk = 0; kk < T4_BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag[WMITER];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag[WNITER];
            for (int i = 0; i < WMITER; i++) {
                wmma::load_matrix_sync(a_frag[i], &As[cur][warpRow + i * WMMA_M][kk], T4_BK);
                for (int t = 0; t < a_frag[i].num_elements; t++)
                    a_frag[i].x[t] = wmma::__float_to_tf32(a_frag[i].x[t]);
            }
            for (int j = 0; j < WNITER; j++) {
                wmma::load_matrix_sync(b_frag[j], &Bs[cur][kk][warpCol + j * WMMA_N], T4_BN);
                for (int t = 0; t < b_frag[j].num_elements; t++)
                    b_frag[j].x[t] = wmma::__float_to_tf32(b_frag[j].x[t]);
            }
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

inline void launch_t4(const float *A, const float *B, float *C, int M, int N, int K) {
    dim3 block(128, 4);
    dim3 grid(M / T4_BM, N / T4_BN);
    t4_wmma_float4<<<grid, block>>>(A, B, C, M, N, K);
}
