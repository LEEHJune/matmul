#pragma once
// T1: block-tiled WMMA TF32. one block (16 warps = 4x4) computes a 64x64 tile of C.
// each k-step the block stages A/B block tiles into smem once, then all 16 warps read
// their fragments from smem -> kills the redundant global loads T0 had (same A row /
// B col was pulled from global by 4 warps each).
#include <mma.h>
#include <cuda_runtime.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

// block tile of C. 64x64 = 4x4 warp tiles, matches block(128,4) = 16 warps.
#define BM 64
#define BN 64
#define BK 16   // smem k-slab width, multiple of WMMA_K(8) -> 2 mma steps per slab

__global__ void t1_wmma_smem(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    // which 64x64 tile of C this block owns
    int blockRow = blockIdx.x * BM;
    int blockCol = blockIdx.y * BN;

    // this warp's 16x16 slot inside the 64x64 block tile
    int warpRow = (threadIdx.x / warpSize) * WMMA_M;   // 0,16,32,48
    int warpCol = (threadIdx.y) * WMMA_N;              // 0,16,32,48

    // linear thread id for the cooperative smem load
    int tid = threadIdx.y * blockDim.x + threadIdx.x;  // 0..511
    int nThreads = blockDim.x * blockDim.y;            // 512

    __shared__ float As[BM][BK];   // 64x16
    __shared__ float Bs[BK][BN];   // 16x64

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // stage A block (BMxBK) and B block (BKxBN) into smem, coalesced + tf32-rounded once.
        // rounding here (not on the fragment) means each element is rounded a single time
        // instead of once per warp that reads it.
        for (int i = tid; i < BM * BK; i += nThreads) {
            int r = i / BK, c = i % BK;
            As[r][c] = wmma::__float_to_tf32(A[(blockRow + r) * K + (k0 + c)]);
        }
        for (int i = tid; i < BK * BN; i += nThreads) {
            int r = i / BN, c = i % BN;
            Bs[r][c] = wmma::__float_to_tf32(B[(k0 + r) * N + (blockCol + c)]);
        }
        __syncthreads();

        // each warp does its 16x16 tile, walking the BK slab in WMMA_K steps from smem
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::load_matrix_sync(a_frag, &As[warpRow][kk], BK);
            wmma::load_matrix_sync(b_frag, &Bs[kk][warpCol], BN);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();   // don't let next slab's load clobber smem before everyone's done
    }

    wmma::store_matrix_sync(C + (blockRow + warpRow) * N + (blockCol + warpCol),
                            c_frag, N, wmma::mem_row_major);
}

inline void launch_t1(const float *A, const float *B, float *C, int M, int N, int K) {
    dim3 block(128, 4);                       // 16 warps
    dim3 grid(M / BM, N / BN);                // one block per 64x64 of C
    t1_wmma_smem<<<grid, block>>>(A, B, C, M, N, K);
}
