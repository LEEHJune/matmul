# GEMM Optimization

## Layout

```
matmul/
├── README.md
├── Makefile
├── include/
│   └── bench.cuh
├── src/
│   ├── cc_00_naive.cuh … cc_08_splitk.cuh          # CUDA-core
│   ├── tc_00_wmma_naive.cuh … tc_05_wmma_pad.cuh   # tensor-core
│   └── benchmark.cu
├── scripts/
│   └── benchmark.sh
└── results/                                        # saved runs (*.csv)
```

- `benchmark.cu`: single driver — builds inputs, runs cuBLAS as the reference
- `bench.cuh`: helper function — input gen, cuBLAS validation, CUDA-event timing, table printing, CSV output
- `cc_0N_*` / `tc_0N_*`: one kernel per file, numbered in optimization order (`cc` = CUDA core, `tc` = tensor core). Each kernel's chosen config lives in its `launch` wrapper
- `Makefile`: build and run the benchmark
- `scripts/benchmark.sh`: build + run helper (invoked by `make run`)
- `results/*.csv`: every run is saved here — `square.csv`, `shape.csv`, `tensor.csv` (one row per kernel)

## Build & run


```bash
make            # build
make run        # runs all kernels, prints the tables
```

Every run also saves the table to `results/` as CSV (`square.csv` / `shape.csv` / `tensor.csv`).

Run one track

```bash
make square             # CUDA-core K0..K8
make shape              # skinny / tall sweep
make tensor             # tensor-core T0..T5
```
