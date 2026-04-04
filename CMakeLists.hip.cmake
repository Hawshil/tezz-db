# ──────────────────────────────────────────────────────────────────────────────
# CMakeLists.hip.cmake — AMD ROCm / HIP build configuration
#
# Usage:
#   cmake -S . -B build-hip -C CMakeLists.hip.cmake
#   cmake --build build-hip --config Release
#   ./build-hip/gpudb_hip_bench
#
# Prerequisites:
#   - ROCm 5.0+ installed (https://rocm.docs.amd.com)
#   - rocPRIM package: sudo apt install rocprim-dev
#   - Set CMAKE_PREFIX_PATH=/opt/rocm if not auto-detected
# ──────────────────────────────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.21)
project(gpudb_hip LANGUAGES CXX HIP)

# ── Standards ────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# Target AMD GPU architectures (adjust to your hardware):
#   gfx906  = MI50/MI60          gfx908  = MI100
#   gfx90a  = MI200 series       gfx942  = MI300
#   gfx1030 = RX 6800/6900       gfx1100 = RX 7900
set(CMAKE_HIP_ARCHITECTURES gfx906 gfx908 gfx90a gfx1030 gfx1100)

list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
find_package(rocprim REQUIRED CONFIG)

# ── Core library (CPU, plain C++) ────────────────────────────────────────────
add_library(gpudb_core STATIC
    src/core/table.cpp
    src/core/schema.cpp
)
target_include_directories(gpudb_core PUBLIC src)

# ── HIP GPU library ─────────────────────────────────────────────────────────
add_library(gpudb_hip_lib STATIC
    src/hip/hip_filter.hip
    src/hip/hip_aggregate.hip
    src/hip/hip_groupby.hip
    src/hip/hip_join.hip
    src/hip/hip_pipeline.hip
)
target_include_directories(gpudb_hip_lib PUBLIC src)
target_link_libraries(gpudb_hip_lib PUBLIC gpudb_core roc::rocprim_hip)

# ── HIP benchmark executable ────────────────────────────────────────────────
add_executable(gpudb_hip_bench src/hip_benchmark.hip)
target_include_directories(gpudb_hip_bench PRIVATE src)
target_link_libraries(gpudb_hip_bench PRIVATE gpudb_hip_lib gpudb_core)
