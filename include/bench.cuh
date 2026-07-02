#pragma once
// shared bench helpers for both tracks (CUDA-core SGEMM + Tensor-core TF32).
// make the inputs, check against cuBLAS, time with CUDA events, print the table.
// problem: C = A * B, row-major, float buffers. square case is M=N=K=4096.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#endif

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

// launch = a lambda that fires the kernel once. run warmups, then time each rep and return the average ms.
template <typename F>
inline float time_ms(F launch, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float total = 0;
    for (int i = 0; i < reps; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total += ms / reps;
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total;
}

inline double gflops(int M, int N, int K, float ms) {
    return (2.0 * (double)M * (double)N * (double)K) / (ms * 1e-3) / 1e9; // 2*M*N*K FLOP
}

// effective operand bandwidth: A+B+C touched once. the meaningful metric in the memory-bound
// (skinny/tall) regime; for the square case it just sits far below peak.
inline double operand_gbps(int M, int N, int K, float ms) {
    double bytes = ((double)M * K + (double)K * N + (double)M * N) * sizeof(float);
    return bytes / (ms * 1e-3) / 1e9;
}

struct Result {
    std::string name;
    float ms;
    double gflops;
    float max_rel;
    bool pass;
};

inline void print_device_info() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    int mem_clk_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clk_khz, cudaDevAttrMemoryClockRate, dev));
    double mem_bw = 2.0 * (double)mem_clk_khz * 1e3 *
                    ((double)p.memoryBusWidth / 8.0) / 1e9;
    printf("Device: %s  (sm_%d%d, %d SMs, %.0f GB/s peak BW)\n",
           p.name, p.major, p.minor, p.multiProcessorCount, mem_bw);
}

inline void print_header() {
    printf("\n%-72s %8s %9s %8s\n", "kernel (config)", "ms", "GFLOP/s", "vs cuBLAS");
}

inline void print_row(const Result &r, double baseline_gflops) {
    double pct = baseline_gflops > 0 ? 100.0 * r.gflops / baseline_gflops : 0.0;
    // only tack on FAIL when the check fails
    printf("%-72s %8.2f %9.0f %7.0f%%%s\n", r.name.c_str(), r.ms, r.gflops, pct,
           r.pass ? "" : "   FAIL");
}

// make a directory if it isn't there (so results/ exists before we write into it)
inline void ensure_dir(const char *d) {
#if defined(_WIN32)
    _mkdir(d);
#else
    mkdir(d, 0777);
#endif
}

// dump the table to a CSV so runs can be saved / diffed / plotted. one schema for every track.
// append=false writes header+rows (fresh file); append=true adds rows only.
inline void write_csv(const char *path, const char *track, const char *shape,
                      int M, int N, int K, const std::vector<Result> &results,
                      double baseline, bool append = false) {
    FILE *f = fopen(path, append ? "a" : "w");
    if (!f) { fprintf(stderr, "warn: cannot write %s\n", path); return; }
    if (!append)
        fprintf(f, "track,shape,kernel,ms,gflops,pct_cublas,gbps,max_rel,pass\n");
    for (const auto &r : results) {
        double pct = baseline > 0 ? 100.0 * r.gflops / baseline : 0.0;
        fprintf(f, "%s,\"%s\",\"%s\",%.4f,%.1f,%.2f,%.1f,%.3e,%d\n", track, shape,
                r.name.c_str(), r.ms, r.gflops, pct, operand_gbps(M, N, K, r.ms),
                r.max_rel, r.pass ? 1 : 0);
    }
    fclose(f);
    printf("[saved] %s\n", path);
}
