#pragma once
// shared bench helpers. make the inputs, check against cuBLAS, time with CUDA events, print the table.
// problem: C = A * B, row-major FP32. here M=N=K=4096.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(_e));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t _s = (call);                                             \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS error %s:%d: status %d\n", __FILE__,        \
                    __LINE__, (int)_s);                                         \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

inline void fill_random(float *a, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; ++i)
        a[i] = (float)rand() / (float)RAND_MAX - 0.5f; // [-0.5, 0.5]
}

// error between my result and the reference. copy to host and compare.
// a per-element relative error blows up where the reference is near zero, so it's a bad fit for GEMM.
// normalize over the whole matrix instead.
inline float max_rel_error(const float *d_x, const float *d_ref, int n) {
    std::vector<float> x(n), r(n);
    CUDA_CHECK(cudaMemcpy(x.data(), d_x, (size_t)n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.data(), d_ref, (size_t)n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float maxabs = 0.0f, maxref = 0.0f;
    for (int i = 0; i < n; ++i) {
        float a = fabsf(x[i] - r[i]);
        if (a > maxabs) maxabs = a;
        float m = fabsf(r[i]);
        if (m > maxref) maxref = m;
    }
    return maxabs / (maxref + 1e-12f);
}

// launch = a lambda that fires the kernel once. run warmups, then time each rep and return the min ms.
// min instead of mean since the rep with the least jitter is closest to real peak, so it's more repeatable.
template <typename F>
inline float time_ms(F launch, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float best = 1e30f;
    for (int i = 0; i < reps; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (ms < best) best = ms;
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return best;
}

inline double gflops(int M, int N, int K, float ms) {
    return (2.0 * (double)M * (double)N * (double)K) / (ms * 1e-3) / 1e9; // 2*M*N*K FLOP
}

struct Result {
    std::string name;
    float ms;
    double gflops;
    float max_rel;
    bool pass;
};

inline void print_header() {
    printf("\n%-22s %8s %9s %8s\n", "kernel", "ms", "GFLOP/s", "vs cuBLAS");
}

inline void print_row(const Result &r, double baseline_gflops) {
    double pct = baseline_gflops > 0 ? 100.0 * r.gflops / baseline_gflops : 0.0;
    // only tack on FAIL when the check fails
    printf("%-22s %8.2f %9.0f %7.0f%%%s\n", r.name.c_str(), r.ms, r.gflops, pct,
           r.pass ? "" : "   FAIL");
}
