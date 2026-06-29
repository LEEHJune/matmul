#pragma once
#include <cuda_pipeline.h>
// K6's cp.async multi-stage plus register-level double buffering.
// in the smem->register step A is pulled out as float4 in one go,
// and B's next-k values are pre-loaded into two regN buffers to overlap with compute.
// pointers are taken as __restrict__ so the compiler can schedule loads more freely.
template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN,
          int NUM_THREADS, int STAGES>
__global__ void __launch_bounds__(NUM_THREADS)
k7_regdb(const float *__restrict__ A, const float *__restrict__ B,
         float *__restrict__ C, int M, int N, int K) {
    constexpr int WMITER = (WM * WN) / (32 * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    auto swzK = [](int m, int k) { return k ^ (4 * ((m / 8) & 3)); };
    constexpr int rowStrideA = NUM_THREADS / (BK / 4);
    constexpr int rowStrideB = NUM_THREADS / (BN / 4);

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    const int warpIdx = threadIdx.x / 32;
    const int warpRow = warpIdx / (BN / WN);
    const int warpCol = warpIdx % (BN / WN);
    const int laneId = threadIdx.x % 32;
    const int threadRowInWarp = laneId / (WSUBN / TN);
    const int threadColInWarp = laneId % (WSUBN / TN);

    constexpr int sAs = BM * BK, sBs = BK * BN;
    extern __shared__ __align__(16) float smem[];
    float *As = smem;
    float *Bs = smem + STAGES * sAs;

    A += cRow * BM * K;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);

    float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
    float4 Af[WMITER * TM]; // 4 k of the c-block. zero per-kk LDS
    float regN[2][WNITER * TN]; // layer-2 DB. B for the next kk ahead of time

    const int numTiles = K / BK;

    // bring one tile from gmem into smem[buf]. A is stored with swizzle
    auto load = [&](int buf, int t) {
#pragma unroll
        for (int off = 0; off + rowStrideA <= BM; off += rowStrideA) {
            int m = innerRowA + off;
            __pipeline_memcpy_async(&As[buf * sAs + m * BK + swzK(m, innerColA * 4)],
                                    &A[m * K + t * BK + innerColA * 4], 16);
        }
#pragma unroll
        for (int off = 0; off + rowStrideB <= BK; off += rowStrideB)
            __pipeline_memcpy_async(
                &Bs[buf * sBs + (innerRowB + off) * BN + innerColB * 4],
                &B[(t * BK + innerRowB + off) * N + innerColB * 4], 16);
        __pipeline_commit();
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) load(s, s);

    for (int tile = 0; tile < numTiles; ++tile) {
        __pipeline_wait_prior(STAGES - 2);
        __syncthreads();
        const int cur = tile % STAGES;

        // bring one row of B into regN[nbuf]
        auto loadB = [&](int nbuf, int k) {
#pragma unroll
            for (int wsc = 0; wsc < WNITER; ++wsc)
#pragma unroll
                for (int jj = 0; jj < TN; jj += 4) {
                    float4 bv = *reinterpret_cast<const float4 *>(
                        &Bs[cur * sBs + k * BN + warpCol * WN + wsc * WSUBN +
                            threadColInWarp * TN + jj]);
                    regN[nbuf][wsc * TN + jj + 0] = bv.x;
                    regN[nbuf][wsc * TN + jj + 1] = bv.y;
                    regN[nbuf][wsc * TN + jj + 2] = bv.z;
                    regN[nbuf][wsc * TN + jj + 3] = bv.w;
                }
        };

#pragma unroll
        for (int c = 0; c < BK / 4; ++c) {
            // A: 4 k of the c-block as float4 in one go
#pragma unroll
            for (int wsr = 0; wsr < WMITER; ++wsr)
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    int _m = warpRow * WM + wsr * WSUBM + threadRowInWarp * TM + i;
                    Af[wsr * TM + i] = *reinterpret_cast<const float4 *>(
                        &As[cur * sAs + _m * BK + swzK(_m, 4 * c)]);
                }

            // pre-load the next kk's B before the current FMA. toggle regN 0/1.
            // Af's components are literals so the 4 kk are unrolled.
            loadB(0, 4 * c + 0);
            loadB(1, 4 * c + 1);
#pragma unroll
            for (int wsr = 0; wsr < WMITER; ++wsr) // kk=0: Af.x * regN[0]
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    float a = Af[wsr * TM + i].x;
#pragma unroll
                    for (int wsc = 0; wsc < WNITER; ++wsc)
#pragma unroll
                        for (int j = 0; j < TN; ++j)
                            threadResults[(wsr * TM + i) * (WNITER * TN) + wsc * TN + j] += a * regN[0][wsc * TN + j];
                }
            loadB(0, 4 * c + 2);
#pragma unroll
            for (int wsr = 0; wsr < WMITER; ++wsr) // kk=1: Af.y * regN[1]
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    float a = Af[wsr * TM + i].y;
#pragma unroll
                    for (int wsc = 0; wsc < WNITER; ++wsc)
#pragma unroll
                        for (int j = 0; j < TN; ++j)
                            threadResults[(wsr * TM + i) * (WNITER * TN) + wsc * TN + j] += a * regN[1][wsc * TN + j];
                }
            loadB(1, 4 * c + 3);
#pragma unroll
            for (int wsr = 0; wsr < WMITER; ++wsr) // kk=2: Af.z * regN[0]
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    float a = Af[wsr * TM + i].z;
#pragma unroll
                    for (int wsc = 0; wsc < WNITER; ++wsc)
#pragma unroll
                        for (int j = 0; j < TN; ++j)
                            threadResults[(wsr * TM + i) * (WNITER * TN) + wsc * TN + j] += a * regN[0][wsc * TN + j];
                }
#pragma unroll
            for (int wsr = 0; wsr < WMITER; ++wsr) // kk=3: Af.w * regN[1]
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    float a = Af[wsr * TM + i].w;
#pragma unroll
                    for (int wsc = 0; wsc < WNITER; ++wsc)
#pragma unroll
                        for (int j = 0; j < TN; ++j)
                            threadResults[(wsr * TM + i) * (WNITER * TN) + wsc * TN + j] += a * regN[1][wsc * TN + j];
                }
        }

        const int loadTile = tile + STAGES - 1;
        if (loadTile < numTiles) load(loadTile % STAGES, loadTile);
    }

#pragma unroll
    for (int wsr = 0; wsr < WMITER; ++wsr)
#pragma unroll
        for (int wsc = 0; wsc < WNITER; ++wsc) {
            float *Csub = C + (wsr * WSUBM) * N + wsc * WSUBN;
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; j += 4) {
                    int idx = (wsr * TM + i) * (WNITER * TN) + wsc * TN + j;
                    float4 t;
                    t.x = threadResults[idx + 0];
                    t.y = threadResults[idx + 1];
                    t.z = threadResults[idx + 2];
                    t.w = threadResults[idx + 3];
                    reinterpret_cast<float4 *>(
                        &Csub[(threadRowInWarp * TM + i) * N +
                              threadColInWarp * TN + j])[0] = t;
                }
        }
}

template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN,
          int NUM_THREADS, int STAGES>
inline void launch_k7_cfg(const float *A, const float *B, float *C,
                          int M, int N, int K) {
    constexpr int smemBytes = STAGES * (BM * BK + BK * BN) * (int)sizeof(float);
    auto kern = k7_regdb<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, STAGES>;
    if (smemBytes > 48 * 1024)
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);
    dim3 block(NUM_THREADS);
    dim3 grid(N / BN, M / BM);
    kern<<<grid, block, smemBytes>>>(A, B, C, M, N, K);
}

inline void launch_k7(const float *A, const float *B, float *C,
                      int M, int N, int K) {
    // 128x128 tile, 3-stage config. small tile leaves headroom for register double buffering.
    launch_k7_cfg<128, 128, 16, 64, 64, 2, 8, 8, 128, 3>(A, B, C, M, N, K);
}
