# ──────────────────────────────────────────────────────────────────────────────
# CMakeLists.sycl.cmake — Intel oneAPI / SYCL (DPC++) build configuration
#
# Usage:
#   # On Linux with oneAPI installed:
#   source /opt/intel/oneapi/setvars.sh
#   cmake -S . -B build-sycl -C CMakeLists.sycl.cmake \
#         -DCMAKE_CXX_COMPILER=icpx
#   cmake --build build-sycl --config Release
#   ./build-sycl/gpudb_sycl_bench
#
#   # On Windows with oneAPI installed:
#   "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
#   cmake -S . -B build-sycl -C CMakeLists.sycl.cmake -G Ninja ^
#         -DCMAKE_CXX_COMPILER=icx
#   cmake --build build-sycl
#
# Prerequisites:
#   - Intel oneAPI Base Toolkit (includes DPC++ compiler + oneDPL + VTune)
#   - Intel GPU driver (for Arc/Xe GPUs), or CPU fallback
# ──────────────────────────────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.20)
project(gpudb_sycl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Core library (CPU, plain C++) ────────────────────────────────────────────
add_library(gpudb_core STATIC
    src/core/table.cpp
    src/core/schema.cpp
)
target_include_directories(gpudb_core PUBLIC src)

# ── SYCL GPU library ────────────────────────────────────────────────────────
add_library(gpudb_sycl_lib STATIC
    src/sycl/sycl_filter.cpp
    src/sycl/sycl_aggregate.cpp
    src/sycl/sycl_groupby.cpp
    src/sycl/sycl_join.cpp
)
target_include_directories(gpudb_sycl_lib PUBLIC src)
target_link_libraries(gpudb_sycl_lib PUBLIC gpudb_core)

# -fsycl enables SYCL compilation; -fsycl-targets selects Intel GPU backend.
# For Arc/Xe GPUs use spir64_gen; for CPU fallback use spir64.
target_compile_options(gpudb_sycl_lib PUBLIC -fsycl)
target_link_options(gpudb_sycl_lib PUBLIC -fsycl)

# Uncomment for specific Intel GPU targeting:
# target_compile_options(gpudb_sycl_lib PUBLIC -fsycl -fsycl-targets=intel_gpu_pvc)

# ── SYCL benchmark executable ───────────────────────────────────────────────
add_executable(gpudb_sycl_bench src/sycl_benchmark.cpp)
target_include_directories(gpudb_sycl_bench PRIVATE src)
target_link_libraries(gpudb_sycl_bench PRIVATE gpudb_sycl_lib gpudb_core)
target_compile_options(gpudb_sycl_bench PRIVATE -fsycl -O2)
target_link_options(gpudb_sycl_bench PRIVATE -fsycl)
