#pragma once
// shared memory tiling. stage A and B into SMEM one TILE x TILE block at a time and multiply there,
// so the same value isn't re-read from global over and over.
// one thread still computes just a single element of C.
template <int TILE>
__global__ void k2_tiling(const float *A, const float *B, float *C,
                          int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x % TILE;
    const int ty = threadIdx.x / TILE;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    for (int t = 0; t < K; t += TILE) {
        As[ty][tx] = A[row * K + (t + tx)]; // tx is contiguous so the load is coalesced too
        Bs[ty][tx] = B[(t + ty) * N + col];
        __syncthreads();                    // wait until the whole tile is loaded

        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();                    // finish the math before the next tile overwrites it
    }
    C[row * N + col] = acc;
}

inline void launch_k2(const float *A, const float *B, float *C,
                      int M, int N, int K) {
    constexpr int TILE = 32;
    dim3 block(TILE * TILE);
    dim3 grid(N / TILE, M / TILE);
    k2_tiling<TILE><<<grid, block>>>(A, B, C, M, N, K);
}
