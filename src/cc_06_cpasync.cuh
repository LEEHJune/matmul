#pragma once
#include <cuda_pipeline.h> // cp.async family. sm_80+
// cp.async based multi-stage pipeline. the gmem->smem copy is fired asynchronously without
// going through registers, so several tiles can be prefetched and overlapped with compute.
// A is loaded in natural layout without transpose. the SWZ flag adds an XOR swizzle that cuts
// bank conflicts, and the VEC4 flag reads A as float4 to bundle the LDS loads.
template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN,
          int NUM_THREADS, int STAGES, bool SWZ, bool VEC4>
__global__ void __launch_bounds__(NUM_THREADS)
k6_cpasync(const float *__restrict__ A, const float *__restrict__ B,
           float *__restrict__ C, int M, int N, int K) {
    constexpr int WMITER = (WM * WN) / (32 * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    // swizzle offset. the XOR is a multiple of 4 so the 16B chunk stays intact
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

    // dynamic smem is needed to go past the 48KB limit and allow stage 4 or more.
    // layout is STAGES copies of As up front, then STAGES copies of Bs.
    constexpr int sAs = BM * BK, sBs = BK * BN;
    extern __shared__ __align__(16) float smem[];
    float *As = smem;
    float *Bs = smem + STAGES * sAs;

    A += cRow * BM * K; // block origin. each tile is offset by +t*BK
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);

    float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
    float regM[WMITER * TM];
    float regN[WNITER * TN];

    const int numTiles = K / BK;

    // bring one tile from gmem into smem[buf]. float4 cp.async, natural layout so no transpose
    auto load = [&](int buf, int t) {
#pragma unroll
        for (int off = 0; off + rowStrideA <= BM; off += rowStrideA) {
            int m = innerRowA + off;
            int ac = SWZ ? swzK(m, innerColA * 4) : innerColA * 4;
            __pipeline_memcpy_async(&As[buf * sAs + m * BK + ac],
                                    &A[m * K + t * BK + innerColA * 4], 16);
        }
#pragma unroll
        for (int off = 0; off + rowStrideB <= BK; off += rowStrideB)
            __pipeline_memcpy_async(
                &Bs[buf * sBs + (innerRowB + off) * BN + innerColB * 4],
                &B[(t * BK + innerRowB + off) * N + innerColB * 4], 16);
        __pipeline_commit();
    };

    // prologue: fire the first STAGES-1 tiles to fill the pipe
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) load(s, s);

    for (int tile = 0; tile < numTiles; ++tile) {
        // tail fix: once prefetch runs out, shrink the wait window so the last STAGES-2 tiles
        // still get their cp.async fully waited. else they read a buffer whose async copy is
        // still in flight (__syncthreads doesn't order cp.async) -> stale-buffer RAW race.
        const int ahead = numTiles - 1 - tile;
        __pipeline_wait_prior(ahead < STAGES - 2 ? ahead : STAGES - 2);
        __syncthreads();
        const int cur = tile % STAGES;

        if constexpr (!VEC4) {
#pragma unroll
            for (int dot = 0; dot < BK; ++dot) {
                // A: natural-layout column so it's a stride-BK scalar. with SWZ, swizzle dot the same way
#pragma unroll
                for (int wsr = 0; wsr < WMITER; ++wsr)
#pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        int _m = warpRow * WM + wsr * WSUBM + threadRowInWarp * TM + i;
                        regM[wsr * TM + i] = As[cur * sAs + _m * BK + (SWZ ? swzK(_m, dot) : dot)];
                    }
#pragma unroll
                for (int wsc = 0; wsc < WNITER; ++wsc)
#pragma unroll
                    for (int i = 0; i < TN; i += 4) {
                        float4 v = *reinterpret_cast<const float4 *>(
                            &Bs[cur * sBs + dot * BN + warpCol * WN + wsc * WSUBN +
                                threadColInWarp * TN + i]);
                        regN[wsc * TN + i + 0] = v.x;
                        regN[wsc * TN + i + 1] = v.y;
                        regN[wsc * TN + i + 2] = v.z;
                        regN[wsc * TN + i + 3] = v.w;
                    }
#pragma unroll
                for (int wsr = 0; wsr < WMITER; ++wsr)
#pragma unroll
                    for (int wsc = 0; wsc < WNITER; ++wsc)
#pragma unroll
                        for (int i = 0; i < TM; ++i)
#pragma unroll
                            for (int j = 0; j < TN; ++j)
                                threadResults[(wsr * TM + i) * (WNITER * TN) + wsc * TN + j] +=
                                    regM[wsr * TM + i] * regN[wsc * TN + j];
            }
        } else {
            // VEC4: read A as float4 once per c-block, then unroll kk 0~3 for the FMA
#pragma unroll
            for (int c = 0; c < BK / 4; ++c) {
                float4 Af[WMITER * TM]; // 4 contiguous k per row
#pragma unroll
                for (int wsr = 0; wsr < WMITER; ++wsr)
#pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        int _m = warpRow * WM + wsr * WSUBM + threadRowInWarp * TM + i;
                        Af[wsr * TM + i] = *reinterpret_cast<const float4 *>(
                            &As[cur * sAs + _m * BK + swzK(_m, 4 * c)]);
                    }

                // bring one row of B into regN. float4
                auto loadB = [&](int k) {
#pragma unroll
                    for (int wsc = 0; wsc < WNITER; ++wsc)
#pragma unroll
                        for (int jj = 0; jj < TN; jj += 4) {
                            float4 bv = *reinterpret_cast<const float4 *>(
                                &Bs[cur * sBs + k * BN + warpCol * WN + wsc * WSUBN +
                                    threadColInWarp * TN + jj]);
                            regN[wsc * TN + jj + 0] = bv.x;
                            regN[wsc * TN + jj + 1] = bv.y;
                            regN[wsc * TN + jj + 2] = bv.z;
                            regN[wsc * TN + jj + 3] = bv.w;
                        }
                };

                // Af's components are .x/.y/.z/.w literals so just unroll the 4 kk
                loadB(4 * c + 0);
#pragma unroll
                for (int wsr = 0; wsr < WMITER; ++wsr)
#pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        float a = Af[wsr * TM + i].x;
#pragma unroll
                        for (int wsc = 0; wsc < WNITER; ++wsc)
#pragma unroll
                            for (int j = 0; j < TN; ++j)
                                threadResults[(wsr * TM + i) * (WNITER * TN) + wsc * TN + j] += a * regN[wsc * TN + j];
                    }
                loadB(4 * c + 1);
#pragma unroll
                for (int wsr = 0; wsr < WMITER; ++wsr)
#pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        float a = Af[wsr * TM + i].y;
#pragma unroll
                        for (int wsc = 0; wsc < WNITER; ++wsc)
#pragma unroll
                            for (int j = 0; j < TN; ++j)
                                threadResults[(wsr * TM + i) * (WNITER * TN) + wsc * TN + j] += a * regN[wsc * TN + j];
                    }
                loadB(4 * c + 2);
#pragma unroll
                for (int wsr = 0; wsr < WMITER; ++wsr)
#pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        float a = Af[wsr * TM + i].z;
#pragma unroll
                        for (int wsc = 0; wsc < WNITER; ++wsc)
#pragma unroll
                            for (int j = 0; j < TN; ++j)
                                threadResults[(wsr * TM + i) * (WNITER * TN) + wsc * TN + j] += a * regN[wsc * TN + j];
                    }
                loadB(4 * c + 3);
#pragma unroll
                for (int wsr = 0; wsr < WMITER; ++wsr)
#pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        float a = Af[wsr * TM + i].w;
#pragma unroll
                        for (int wsc = 0; wsc < WNITER; ++wsc)
#pragma unroll
                            for (int j = 0; j < TN; ++j)
                                threadResults[(wsr * TM + i) * (WNITER * TN) + wsc * TN + j] += a * regN[wsc * TN + j];
                    }
            }
        }

        // fire the next tile. overwrite the buffer that compute just freed
        const int loadTile = tile + STAGES - 1;
        if (loadTile < numTiles) load(loadTile % STAGES, loadTile);
    }

    // store result. same as K5
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
          int NUM_THREADS, int STAGES, bool SWZ, bool VEC4>
inline void launch_k6_cfg(const float *A, const float *B, float *C,
                           int M, int N, int K) {
    constexpr int smemBytes = STAGES * (BM * BK + BK * BN) * (int)sizeof(float);
    auto kern = k6_cpasync<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, STAGES, SWZ, VEC4>;
    // past 48KB the launch fails without the opt-in
    if (smemBytes > 48 * 1024)
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);
    dim3 block(NUM_THREADS);
    dim3 grid(N / BN, M / BM);
    kern<<<grid, block, smemBytes>>>(A, B, C, M, N, K);
}

inline void launch_k6(const float *A, const float *B, float *C,
                       int M, int N, int K) {
    // 256x128 tile, 3-stage, swizzle + float4 A config
    launch_k6_cfg<256, 128, 16, 64, 64, 2, 8, 8, 256, 3, true, true>(A, B, C, M, N, K);
}
