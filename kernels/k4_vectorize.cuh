#pragma once
// micro-tiling plus memory-access tuning.
// As is stored transposed in SMEM, so reading it into registers hits contiguous
// addresses and comes in as one LDS.128.
// gmem<->smem transfers and the C output are grouped as float4 to cut the load/store
// instruction count to a quarter.
// assumes BK, BN, TN are multiples of 4.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void k4_vectorize(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    __shared__ float As[BM * BK]; // stored transposed. As[dot*BM + row]
    __shared__ float Bs[BK * BN];

    const int totalThreads = (BM * BN) / (TM * TN);

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // float4 load indices. columns counted in groups of 4
    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    const int strideA = totalThreads / (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);
    const int strideB = totalThreads / (BN / 4);

    float threadResults[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (int bk = 0; bk < K; bk += BK) {
        // A: read as float4, transpose, store into SMEM
        for (int off = 0; off < BM; off += strideA) {
            float4 t = reinterpret_cast<const float4 *>(
                &A[(innerRowA + off) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + off] = t.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + off] = t.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + off] = t.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + off] = t.w;
        }
        // B: store into SMEM as float4 directly
        for (int off = 0; off < BK; off += strideB) {
            reinterpret_cast<float4 *>(
                &Bs[(innerRowB + off) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4 *>(
                    &B[(innerRowB + off) * N + innerColB * 4])[0];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dot = 0; dot < BK; ++dot) {
            // As is transposed so regM is contiguous -> LDS.128
            for (int i = 0; i < TM; ++i)
                regM[i] = As[dot * BM + threadRow * TM + i];
            for (int j = 0; j < TN; ++j)
                regN[j] = Bs[dot * BN + threadCol * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    threadResults[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

    // store the result as float4 too
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; j += 4) {
            float4 t;
            t.x = threadResults[i * TN + j + 0];
            t.y = threadResults[i * TN + j + 1];
            t.z = threadResults[i * TN + j + 2];
            t.w = threadResults[i * TN + j + 3];
            reinterpret_cast<float4 *>(
                &C[(threadRow * TM + i) * N + threadCol * TN + j])[0] = t;
        }
}

template <int BM, int BN, int BK, int TM, int TN>
inline void launch_k4_cfg(const float *A, const float *B, float *C,
                          int M, int N, int K) {
    dim3 block((BM * BN) / (TM * TN));
    dim3 grid(N / BN, M / BM);
    k4_vectorize<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, M, N, K);
}

inline void launch_k4(const float *A, const float *B, float *C,
                      int M, int N, int K) {
    // BK 32, 8x8 micro-tile config
    launch_k4_cfg<128, 128, 32, 8, 8>(A, B, C, M, N, K);
}
