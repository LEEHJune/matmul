#pragma once
// the most basic matmul. one thread owns a single element of C and dots its row with its column.
// no shared memory, every value is read straight from global.
__global__ void k0_naive(const float *A, const float *B, float *C,
                         int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    // within a warp thread.x is the row, so neighboring threads read different rows.
    // A and C accesses jump by K and N, so memory coalescing breaks.
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

inline void launch_k0(const float *A, const float *B, float *C,
                      int M, int N, int K) {
    dim3 block(32, 32);
    dim3 grid((M + 31) / 32, (N + 31) / 32);
    k0_naive<<<grid, block>>>(A, B, C, M, N, K);
}
