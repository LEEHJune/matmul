// matmul bench driver. uses cuBLAS as the reference and runs K0~K7 into a comparison table.
// K0 naive -> K1 coalesce -> K2 smem tile -> K3 microtile -> K4 float4+transpose
//         -> K5 warptile -> K6 cp.async multistage+swizzle+float4 -> K7 +reg-DB
#include "bench.cuh"
#include "kernels/k0_naive.cuh"
#include "kernels/k1_coalesce.cuh"
#include "kernels/k2_tiling.cuh"
#include "kernels/k3_microtile.cuh"
#include "kernels/k4_vectorize.cuh"
#include "kernels/k5_warptile.cuh"
#include "kernels/k6_cpasync.cuh"
#include "kernels/k7_regdb.cuh"
#include "kernels/k8_splitk.cuh"

static void print_device_info() {
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

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 4096;
    int reps = (argc > 2) ? atoi(argv[2]) : 20;
    const int warmup = 3;
    int M = N, K = N; // keep it square

    print_device_info();
    printf("\nProblem: C[%d x %d] = A[%d x %d] * B[%d x %d], FP32, "
           "reps=%d (warmup=%d)\n",
           M, N, M, K, K, N, reps, warmup);

    const size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;

    std::vector<float> hA(szA), hB(szB);
    fill_random(hA.data(), (int)szA, 1);
    fill_random(hB.data(), (int)szB, 2);

    float *dA, *dB, *dC, *dRef;
    CUDA_CHECK(cudaMalloc(&dA, szA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, szB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, szC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dRef, szC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), szA * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), szB * sizeof(float),
                          cudaMemcpyHostToDevice));

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

    // run one kernel, validate and time it, then add it to the table
    auto bench = [&](const char *name, auto launch) {
        CUDA_CHECK(cudaMemset(dC, 0, szC * sizeof(float)));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        float rel = max_rel_error(dC, dRef, (int)szC);
        float ms = time_ms(launch, warmup, reps);
        results.push_back({name, ms, gflops(M, N, K, ms), rel, rel < 1e-2f});
    };

    // one per kernel. each calls its own launch_kX wrapper from its header.
    // the tuning sweep is done and stripped out. the chosen config is in the wrapper
    bench("K0 naive  blk32x32",                                          [&] { launch_k0(dA, dB, dC, M, N, K); });
    bench("K1 coalesce  BS32",                                           [&] { launch_k1(dA, dB, dC, M, N, K); });
    bench("K2 tiling  TILE32",                                           [&] { launch_k2(dA, dB, dC, M, N, K); });
    bench("K3 micro  BM128 BN128 BK8 TM8 TN8",                           [&] { launch_k3(dA, dB, dC, M, N, K); });
    bench("K4 vec  BM128 BN128 BK32 TM8 TN8",                            [&] { launch_k4(dA, dB, dC, M, N, K); });
    bench("K5 warptile  BM128 BN128 BK16 WM64 WN64 WIT2 TM8 TN8 NT128",  [&] { launch_k5(dA, dB, dC, M, N, K); });

    // cp.async breakthrough track
    bench("K6 cp.async  BM256 BN128 BK16 WM64 WN64 WIT2 TM8 TN8 NT256 S3 swz vec4", [&] { launch_k6(dA, dB, dC, M, N, K); });
    bench("K7 reg-DB  BM128 BN256 BK16 WM64 WN64 WIT2 TM8 TN8 NT256 S3",            [&] { launch_k7(dA, dB, dC, M, N, K); });
    bench("K8 split-K  BM128 BN256 BK16 WM64 WN64 WIT2 TM8 TN8 NT256 S3 SK1",       [&] { launch_k8(dA, dB, dC, M, N, K); });

    print_header();
    for (auto &r : results) print_row(r, baseline);
    printf("\n");

    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
    return 0;
}
