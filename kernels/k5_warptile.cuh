#pragma once
// warp tiling. adds a warp-tile layer between the block tile and the thread micro-tile.
// the hierarchy is Block BM x BN > Warp WM x WN > the sub-tile a thread owns.
// inside a warp the lane mapping is kept narrow so the 32 lanes hit different banks,
// and each thread owns several sub-tiles to reuse registers more.
template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
k5_warptile(const float *A, const float *B, float *C, int M, int N, int K) {
    constexpr int WMITER = (WM * WN) / (32 * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER; // sub-tile height
    constexpr int WSUBN = WN / WNITER; // width
    constexpr int rowStrideA = (NUM_THREADS * 4) / BK;
    constexpr int rowStrideB = NUM_THREADS / (BN / 4);

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    // where this warp sits inside the block tile
    const int warpIdx = threadIdx.x / 32;
    const int warpRow = warpIdx / (BN / WN);
    const int warpCol = warpIdx % (BN / WN);

    // this thread's position inside the warp. 32 lanes laid out as a grid
    const int laneId = threadIdx.x % 32;
    const int threadRowInWarp = laneId / (WSUBN / TN);
    const int threadColInWarp = laneId % (WSUBN / TN);

    // the transposed As store has stride BM, so several threads in one row hit the same bank.
    // padding the leading dim by 4 breaks the stride and halves 4-way down to 2-way.
    // fully removing it needs swizzle. here padding only gets halfway.
    constexpr int LDAs = BM + 4;
    __shared__ float As[BK * LDAs]; // stored transposed. As[dot*LDAs + row]
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    // move C to the top-left of this warp's output tile
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    // float4 load indices
    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);

    float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
    float regM[WMITER * TM];
    float regN[WNITER * TN];

    for (int bk = 0; bk < K; bk += BK) {
        // load As transposed. read float4, transpose, scatter into SMEM
        for (int off = 0; off + rowStrideA <= BM; off += rowStrideA) {
            float4 t = reinterpret_cast<const float4 *>(
                &A[(innerRowA + off) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * LDAs + innerRowA + off] = t.x;
            As[(innerColA * 4 + 1) * LDAs + innerRowA + off] = t.y;
            As[(innerColA * 4 + 2) * LDAs + innerRowA + off] = t.z;
            As[(innerColA * 4 + 3) * LDAs + innerRowA + off] = t.w;
        }
        // load Bs. float4 as-is
        for (int off = 0; off + rowStrideB <= BK; off += rowStrideB) {
            reinterpret_cast<float4 *>(
                &Bs[(innerRowB + off) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4 *>(
                    &B[(innerRowB + off) * N + innerColB * 4])[0];
        }
        __syncthreads();

        for (int dot = 0; dot < BK; ++dot) {
            // pull regM/regN as float4. As is transposed so rows are contiguous, Bs has contiguous cols.
            // take it into a float4 temp and unpack to scalars so it stays in registers.
            // if the address leaks it demotes to local.
            for (int wsr = 0; wsr < WMITER; ++wsr)
                for (int i = 0; i < TM; i += 4) {
                    float4 v = reinterpret_cast<const float4 *>(
                        &As[dot * LDAs + warpRow * WM + wsr * WSUBM +
                            threadRowInWarp * TM + i])[0];
                    regM[wsr * TM + i + 0] = v.x;
                    regM[wsr * TM + i + 1] = v.y;
                    regM[wsr * TM + i + 2] = v.z;
                    regM[wsr * TM + i + 3] = v.w;
                }
            for (int wsc = 0; wsc < WNITER; ++wsc)
                for (int i = 0; i < TN; i += 4) {
                    float4 v = reinterpret_cast<const float4 *>(
                        &Bs[dot * BN + warpCol * WN + wsc * WSUBN +
                            threadColInWarp * TN + i])[0];
                    regN[wsc * TN + i + 0] = v.x;
                    regN[wsc * TN + i + 1] = v.y;
                    regN[wsc * TN + i + 2] = v.z;
                    regN[wsc * TN + i + 3] = v.w;
                }

            // accumulate the outer product over the sub-tile grid
            for (int wsr = 0; wsr < WMITER; ++wsr)
                for (int wsc = 0; wsc < WNITER; ++wsc)
                    for (int i = 0; i < TM; ++i)
                        for (int j = 0; j < TN; ++j)
                            threadResults[(wsr * TM + i) * (WNITER * TN) +
                                          wsc * TN + j] +=
                                regM[wsr * TM + i] * regN[wsc * TN + j];
        }
        A += BK;
        B += BK * N;
        __syncthreads();
    }

    // store result. float4 per sub-tile
    for (int wsr = 0; wsr < WMITER; ++wsr)
        for (int wsc = 0; wsc < WNITER; ++wsc) {
            float *Csub = C + (wsr * WSUBM) * N + wsc * WSUBN;
            for (int i = 0; i < TM; ++i)
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

template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN, int NUM_THREADS>
inline void launch_k5_cfg(const float *A, const float *B, float *C,
                          int M, int N, int K) {
    dim3 block(NUM_THREADS);
    dim3 grid(N / BN, M / BM);
    k5_warptile<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<grid, block>>>(A, B, C, M, N, K);
}

inline void launch_k5(const float *A, const float *B, float *C,
                      int M, int N, int K) {
    // 128 threads, warp 2x2, thread 8x8 config
    launch_k5_cfg<128, 128, 16, 64, 64, 2, 8, 8, 128>(A, B, C, M, N, K);
}
