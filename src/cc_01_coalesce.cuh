#pragma once
// same math as naive, threads just laid out along the column instead.
// thread.x % BS is the column so a warp reads a contiguous run of columns,
// which makes B and C accesses coalesced.
template <int BS>
__global__ void k1_coalesce(const float *A, const float *B, float *C,
                            int M, int N, int K) {
    const int row = blockIdx.y * BS + (threadIdx.x / BS); // row is fixed across the warp -> A is a broadcast
    const int col = blockIdx.x * BS + (threadIdx.x % BS);
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

inline void launch_k1(const float *A, const float *B, float *C,
                      int M, int N, int K) {
    constexpr int BS = 32;
    dim3 block(BS * BS);
    dim3 grid((N + BS - 1) / BS, (M + BS - 1) / BS);
    k1_coalesce<BS><<<grid, block>>>(A, B, C, M, N, K);
}
