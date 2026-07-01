// unified GEMM benchmark driver. one binary, three tracks, cuBLAS as the reference:
//   square  : CUDA-core SGEMM ladder K0..K8   (FP32,  vs cuBLAS SGEMM)
//   shape   : skinny / tall sweep with K7      (FP32,  DRAM-bound regime, +GB/s)
//   tensor  : tensor-core TF32 ladder T0..T5    (TF32,  vs cuBLAS FAST_TF32)
// usage: ./benchmark [all|square|shape|tensor] [N] [reps]      (default: all 4096 20)
#include "bench.cuh"
// CUDA-core ladder
#include "cc_00_naive.cuh"
#include "cc_01_coalesce.cuh"
#include "cc_02_tiling.cuh"
#include "cc_03_microtile.cuh"
#include "cc_04_vectorize.cuh"
#include "cc_05_warptile.cuh"
#include "cc_06_cpasync.cuh"
#include "cc_07_regdb.cuh"
#include "cc_08_splitk.cuh"
// tensor-core ladder
#include "tc_00_wmma_naive.cuh"
#include "tc_01_wmma_smem.cuh"
#include "tc_02_wmma_warptile.cuh"
#include "tc_03_wmma_doublebuf.cuh"
#include "tc_04_wmma_float4.cuh"
#include "tc_05_wmma_pad.cuh"

// ---- CUDA-core SGEMM ladder, square (M=N=K=N), vs cuBLAS FP32 ----
static void run_square(int N, int reps) {
    const int warmup = 3;
    int M = N, K = N;
    printf("\n===== CUDA-core SGEMM (square) =====\n");
    printf("Problem: C[%d x %d] = A[%d x %d] * B[%d x %d], FP32, "
           "reps=%d (warmup=%d)\n", M, N, M, K, K, N, reps, warmup);

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

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;

    // cuBLAS is column-major so it can't do row-major C=A*B directly.
    // swap the B,A order and m/n so it computes C^T = B^T*A^T instead. the result is still C.
    auto run_cublas = [&](float *out) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                 &alpha, dB, N, dA, K, &beta, out, N));
    };

    run_cublas(dRef); // build the reference once and validate against it
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
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        float rel = max_rel_error(dC, dRef, (int)szC);
        float ms = time_ms(launch, warmup, reps);
        results.push_back({name, ms, gflops(M, N, K, ms), rel, rel < 1e-2f});
    };

    bench("K0 naive  blk32x32",                                          [&] { launch_k0(dA, dB, dC, M, N, K); });
    bench("K1 coalesce  BS32",                                           [&] { launch_k1(dA, dB, dC, M, N, K); });
    bench("K2 tiling  TILE32",                                           [&] { launch_k2(dA, dB, dC, M, N, K); });
    bench("K3 micro  BM128 BN128 BK8 TM8 TN8",                           [&] { launch_k3(dA, dB, dC, M, N, K); });
    bench("K4 vec  BM128 BN128 BK32 TM8 TN8",                            [&] { launch_k4(dA, dB, dC, M, N, K); });
    bench("K5 warptile  BM128 BN128 BK16 WM64 WN64 WIT2 TM8 TN8 NT128",  [&] { launch_k5(dA, dB, dC, M, N, K); });
    bench("K6 cp.async  BM256 BN128 BK16 WM64 WN64 WIT2 TM8 TN8 NT256 S3 swz vec4", [&] { launch_k6(dA, dB, dC, M, N, K); });
    bench("K7 reg-DB  BM128 BN256 BK16 WM64 WN64 WIT2 TM8 TN8 NT256 S3",            [&] { launch_k7(dA, dB, dC, M, N, K); });
    bench("K8 split-K  BM128 BN256 BK16 WM64 WN64 WIT2 TM8 TN8 NT256 S3 SK1",       [&] { launch_k8(dA, dB, dC, M, N, K); });

    print_header();
    for (auto &r : results) print_row(r, baseline);
    printf("\n");

    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
}

// ---- tensor-core TF32 ladder, square, vs cuBLAS FAST_TF32 ----
static void run_tensor(int N, int reps) {
    const int warmup = 3;
    int M = N, K = N;
    printf("\n===== Tensor-core GEMM (TF32, square) =====\n");
    printf("Problem: C[%d x %d] = A[%d x %d] * B[%d x %d], TF32 baseline, "
           "reps=%d (warmup=%d)\n", M, N, M, K, K, N, reps, warmup);

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

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;

    // cuBLAS is column-major; swap B,A and m/n so it computes C^T = B^T*A^T (= our row-major C).
    // COMPUTE_32F_FAST_TF32 forces the TF32 tensor-core path = the one baseline for everything.
    auto run_cublas = [&](float *out) {
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                  &alpha, dB, CUDA_R_32F, N, dA, CUDA_R_32F, K,
                                  &beta, out, CUDA_R_32F, N,
                                  CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT));
    };

    run_cublas(dRef);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<Result> results;
    {
        float ms = time_ms([&] { run_cublas(dC); }, warmup, reps);
        results.push_back({"cuBLAS (TF32)", ms, gflops(M, N, K, ms), 0.0f, true});
    }
    double baseline = results[0].gflops;

    auto bench = [&](const char *name, auto launch) {
        CUDA_CHECK(cudaMemset(dC, 0, szC * sizeof(float)));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        float rel = max_rel_error(dC, dRef, (int)szC);
        float ms = time_ms(launch, warmup, reps);
        // TF32 has ~10 mantissa bits, so accept a looser bound than SGEMM.
        results.push_back({name, ms, gflops(M, N, K, ms), rel, rel < 1e-2f});
    };

    bench("T0 wmma naive  WMMA16x16x8 blk64x64 16warps",                        [&] { launch_t0(dA, dB, dC, M, N, K); });
    bench("T1 wmma smem  WMMA16x16x8 BM64 BN64 BK16 16warps",                    [&] { launch_t1(dA, dB, dC, M, N, K); });
    bench("T2 warptile  WMMA16x16x8 BM128 BN128 BK16 WARP32x32(2x2) 16warps",    [&] { launch_t2(dA, dB, dC, M, N, K); });
    bench("T3 doublebuf  WMMA16x16x8 BM128 BN128 BK16 WARP32x32(2x2) 2stage",    [&] { launch_t3(dA, dB, dC, M, N, K); });
    bench("T4 float4  WMMA16x16x8 BM128 BN128 BK16 WARP32x32(2x2) 2stage f4",    [&] { launch_t4(dA, dB, dC, M, N, K); });
    bench("T5 pad+4  WMMA16x16x8 BM128 BN128 BK16 WARP32x32(2x2) 2stage pad4",   [&] { launch_t5(dA, dB, dC, M, N, K); });

    print_header();
    for (auto &r : results) print_row(r, baseline);
    printf("\n");

    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
}

// ---- skinny / tall shape sweep (K=8192), DRAM-bound regime. GB/s is the meaningful metric. ----
static void run_shape(int reps) {
    const int warmup = 5;
    printf("\n===== CUDA-core GEMM (skinny / tall shapes) =====\n");

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;

    // run one problem shape: alloc, build cuBLAS reference, bench a set of configs, print table.
    auto run_shape_one = [&](const char *tag, int M, int N, int K, auto configs) {
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
                   pct, operand_gbps(M, N, K, r.ms), r.pass ? "" : "  FAIL");
        }

        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
    };

    // SKINNY (M tiny): BM=8, WNITER=1, TN=4. B(K x N) is read fully coalesced -> near DRAM peak.
    auto skinny_cfgs = [](float *A, float *B, float *C, int M, int N, int K, auto bench) {
        bench("BM8 BN128 BK16 WM8 WN128 WIT1 TM8 TN4 NT32 S3",
              [&] { launch_k7_cfg<8, 128, 16, 8, 128, 1, 8, 4, 32, 3>(A, B, C, M, N, K); });
    };

    // TALL (N tiny): BN=8, WN=8, WNITER=1, TN=8. big operand A(M x K) is read across K-rows, so
    // BK must be large enough that a warp covers a row contiguously (coalesced). BK64 saturates DRAM BW.
    auto tall_cfgs = [](float *A, float *B, float *C, int M, int N, int K, auto bench) {
        bench("BM32 BN8 BK64 WM32 WN8 WIT1 TM1 TN8 NT32 S3",
              [&] { launch_k7_cfg<32, 8, 64, 32, 8, 1, 1, 8, 32, 3>(A, B, C, M, N, K); });
    };

    run_shape_one("SKINNY M=8 N=8192 K=8192", 8, 8192, 8192, skinny_cfgs);
    run_shape_one("TALL   M=8192 N=8 K=8192", 8192, 8, 8192, tall_cfgs);

    cublasDestroy(handle);
}

int main(int argc, char **argv) {
    std::string mode = (argc > 1) ? argv[1] : "all";
    int N = (argc > 2) ? atoi(argv[2]) : 4096;
    int reps = (argc > 3) ? atoi(argv[3]) : 20;

    print_device_info();

    bool all = (mode == "all");
    if (all || mode == "square") run_square(N, reps);
    if (all || mode == "shape")  run_shape(50);
    if (all || mode == "tensor") run_tensor(N, reps);
    if (!all && mode != "square" && mode != "shape" && mode != "tensor") {
        fprintf(stderr, "unknown mode '%s' (use: all | square | shape | tensor)\n",
                mode.c_str());
        return 1;
    }
    return 0;
}
