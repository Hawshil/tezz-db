# Release Notes — GPUDB v1.0.0

## 🚀 What's New

### Cross-Vendor GPU Database Engine
A high-performance GPU-accelerated columnar database with SQL support, targeting **NVIDIA**, **AMD**, and **Intel** GPUs.

### Core Features
- **Columnar Storage**: TypedColumn<T> with null bitmaps, zero-copy access
- **SQL Parser**: Hand-written lexer + recursive-descent parser (SELECT/WHERE/GROUP BY/JOIN)
- **CPU Baseline**: Full operator suite (Scan, Filter, Aggregate, HashJoin)
- **CUDA Backend**: Filter, SUM, GROUP BY, Hash Join with Thrust + warp shuffle
- **HIP/ROCm Backend**: Wavefront-aware (64-thread) reductions + rocPRIM
- **SYCL/oneAPI Backend**: `sycl::reduction()`, `sycl::atomic_ref`, oneDPL
- **Apache Arrow**: IPC I/O + 3 GPU transfer strategies (copy/pinned/UVM)
- **Compressed Execution**: Dictionary + RLE — kernels operate on compressed data
- **Late Materialisation**: CPU predicate → selective GPU transfer (100x PCIe savings)
- **Out-of-Core**: ChunkedExecutor auto-sizes from VRAM, handles 50GB+ on 8GB GPUs
- **CUDA/HIP Graphs**: Graph capture for repeated queries
- **Interactive CLI**: SQL REPL with LOAD/SELECT/BENCH/EXPLAIN
- **Backend Abstraction**: `GpuBackend` interface — swap CUDA↔HIP↔SYCL at compile time

### Benchmark Highlights

| Operation | 10M rows | 100M rows | GPU Speedup |
|---|---|---|---|
| Filter | 2.1 ms | 12.3 ms | **85-120x** |
| SUM Reduction | 0.9 ms | 5.4 ms | **110-150x** |
| GROUP BY + SUM | 4.8 ms | 28.1 ms | **45-60x** |
| Hash Join (1M×10M) | — | 18.7 ms | **35-50x** |

*Results from NVIDIA CUDA. HIP/SYCL benchmarks available via platform-specific builds.*

## 📦 Included in This Release

| File | Description |
|---|---|
| `gpudb_bench.exe` | CPU benchmark executable |
| `gpudb_cli.exe` | Interactive SQL REPL (requires CUDA GPU) |
| `gpudb_demo.exe` | CPU-only demo (no GPU needed) |
| `benchmark_results.csv` | Performance measurements |
| `BENCHMARK_REPORT.md` | Full performance analysis report |
| `README.md` | Documentation |
| `LICENSE` | MIT License |

## 🔧 Build From Source

```bash
# NVIDIA CUDA
cmake -S . -B build -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# AMD ROCm
cmake -S . -B build-hip -C CMakeLists.hip.cmake
cmake --build build-hip

# Intel oneAPI
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl -C CMakeLists.sycl.cmake -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl
```

## 📋 System Requirements
- **OS**: Windows 10/11, Ubuntu 22.04+
- **Compiler**: MSVC 2022 / GCC 11+ / Clang 14+
- **GPU**: NVIDIA (CUDA 11.8+), AMD (ROCm 5.0+), or Intel (oneAPI 2024+)
- **RAM**: 8 GB minimum, 32 GB recommended for large benchmarks

## 🏗️ Technologies
C++17 · CUDA · HIP/ROCm · SYCL/DPC++ · Apache Arrow · Thrust · rocPRIM · oneDPL · CMake · Google Test
