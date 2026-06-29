// skinny / tall GEMM sweep, DRAM-bound regime (K=8192): the 256MB operand spills the 4090's
// 72MB L2, so this is genuinely DRAM-BW-bound (the realistic memory-bound case for these shapes).
// memory-bound -> GB/s (effective operand bandwidth) is the meaningful metric, kept with ms/GFLOP/s.
#include "bench.cuh"
#include "kernels/k7_regdb.cuh"

// effective bandwidth from the minimal operand traffic (A + B + C read/written once).
static inline double gbps(int M, int N, int K, double ms) {
    double bytes = ((double)M * K + (double)K * N + (double)M * N) * sizeof(float);
    return bytes / (ms * 1e-3) / 1e9;
}

int main(int argc, char **argv) {
    int reps = (argc > 1) ? atoi(argv[1]) : 50;
    const int warmup = 5;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;

    // run one problem shape: alloc, build cuBLAS reference, bench a set of configs, print table.
    auto run_shape = [&](const char *tag, int M, int N, int K, auto configs) {
        printf("\n=== %s : C[%dx%d] = A[%dx%d]*B[%dx%d], reps=%d ===\n", tag, M, N, M, K, K, N, reps);
        const size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;
        std::vector<float> hA(szA), hB(szB);
        fill_random(hA.data(), (int)szA, 1);
        fill_random(hB.data(), (int)szB, 2);
        float *dA, *dB, *dC, *dRef;
        CUDA_CHECK(cudaMalloc(&dA, szA * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, szB * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, szC * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dRef, szC * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), szA * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), szB * sizeof(float), cudaMemcpyHostToDevice));

        // cuBLAS reference (col-major trick: compute C^T = B^T*A^T)
        auto run_cublas = [&](float *out) {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                     &alpha, dB, N, dA, K, &beta, out, N));
        };
        run_cublas(dRef);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<Result> results;
        {
            float ms = time_ms([&] { run_cublas(dC); }, warmup, reps);
            results.push_back({"cuBLAS (FP32)", ms, gflops(M, N, K, ms), 0.0f, true});
        }
        double baseline = results[0].gflops;

        auto bench = [&](const char *name, auto launch) {
            CUDA_CHECK(cudaMemset(dC, 0, szC * sizeof(float)));
            launch();
            if (cudaGetLastError() != cudaSuccess) {
                results.push_back({std::string(name) + " [launch fail]", 0, 0, 0, false});
                return;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            float rel = max_rel_error(dC, dRef, (int)szC);
            float ms = time_ms(launch, warmup, reps);
            results.push_back({name, ms, gflops(M, N, K, ms), rel, rel < 1e-2f});
        };
        configs(dA, dB, dC, M, N, K, bench);

        printf("%-46s %8s %9s %8s %8s\n", "config (BM BN BK WM WN WIT TM TN NT S)",
               "ms", "GFLOP/s", "vs cuB", "GB/s");
        for (auto &r : results) {
            double pct = baseline > 0 ? 100.0 * r.gflops / baseline : 0.0;
            printf("%-46s %8.3f %9.0f %7.0f%% %8.0f%s\n", r.name.c_str(), r.ms, r.gflops,
                   pct, gbps(M, N, K, r.ms), r.pass ? "" : "  FAIL");
        }

        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
    };

    // SKINNY (M tiny): BM=8, WNITER=1, TN=4. B(K x N) is read fully coalesced -> near DRAM peak.
    auto skinny_cfgs = [](float *A, float *B, float *C, int M, int N, int K, auto bench) {
        bench("BM8 BN128 BK16 WM8 WN128 WIT1 TM8 TN4 NT32 S3",
              [&] { launch_k7_cfg<8, 128, 16, 8, 128, 1, 8, 4, 32, 3>(A, B, C, M, N, K); });
    };

    // TALL (N tiny): BN=8, WN=8, WNITER=1, TN=8. big operand A(M x K) is read across K-rows, so
    // BK must be large enough that a warp covers a row contiguously (coalesced). BK64 saturates DRAM BW; 
    // split-K was the real lever here.
    auto tall_cfgs = [](float *A, float *B, float *C, int M, int N, int K, auto bench) {
        bench("BM32 BN8 BK64 WM32 WN8 WIT1 TM1 TN8 NT32 S3",
              [&] { launch_k7_cfg<32, 8, 64, 32, 8, 1, 1, 8, 32, 3>(A, B, C, M, N, K); });
    };

    run_shape("SKINNY M=8 N=8192 K=8192", 8, 8192, 8192, skinny_cfgs);
    run_shape("TALL   M=8192 N=8 K=8192", 8192, 8, 8192, tall_cfgs);

    cublasDestroy(handle);
    return 0;
}
