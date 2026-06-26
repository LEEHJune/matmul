#pragma once
// register tiling. one thread owns a TM x TN micro-tile of C.
// the block stages a BM x BN tile into SMEM and walks K in steps of BK,
// pulling only TM+TN values into registers each step and doing TM*TN multiplies.
// more work per thread raises the compute-to-memory ratio.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void k3_microtile(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int totalThreads = (BM * BN) / (TM * TN);

    // where this thread's micro-tile sits
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // move pointers to the start of this block's tile
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // indices and strides for the threads to split the As, Bs loads
    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int strideA = totalThreads / BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;
    const int strideB = totalThreads / BN;

    float threadResults[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (int bk = 0; bk < K; bk += BK) {
        // threads split the A tile and B tile into SMEM
        for (int off = 0; off < BM; off += strideA)
            As[(innerRowA + off) * BK + innerColA] =
                A[(innerRowA + off) * K + innerColA];
        for (int off = 0; off < BK; off += strideB)
            Bs[(innerRowB + off) * BN + innerColB] =
                B[(innerRowB + off) * N + innerColB];
        __syncthreads();

        A += BK; // one step along K
        B += BK * N;

        // pull one row at a time into registers and accumulate the outer product
        for (int dot = 0; dot < BK; ++dot) {
            for (int i = 0; i < TM; ++i)
                regM[i] = As[(threadRow * TM + i) * BK + dot];
            for (int j = 0; j < TN; ++j)
                regN[j] = Bs[dot * BN + threadCol * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    threadResults[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            C[(threadRow * TM + i) * N + threadCol * TN + j] =
                threadResults[i * TN + j];
}

// version that takes the tile config as template params
template <int BM, int BN, int BK, int TM, int TN>
inline void launch_k3_cfg(const float *A, const float *B, float *C,
                          int M, int N, int K) {
    dim3 block((BM * BN) / (TM * TN));
    dim3 grid(N / BN, M / BM);
    k3_microtile<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, M, N, K);
}

inline void launch_k3(const float *A, const float *B, float *C,
                      int M, int N, int K) {
    launch_k3_cfg<128, 128, 8, 8, 8>(A, B, C, M, N, K);
}
