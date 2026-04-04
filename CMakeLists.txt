cmake_minimum_required(VERSION 3.18)
project(gpudb LANGUAGES CXX)

# ── Options (declare before use) ─────────────────────────────────────────────
option(USE_CUDA "Enable CUDA backend" ON)
option(BUILD_TESTS "Build Google Test suite" OFF)

# ── Standards ────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89)
endif()

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Optional: Apache Arrow ───────────────────────────────────────────────────
find_package(Arrow QUIET)
if(Arrow_FOUND)
    message(STATUS "Apache Arrow found — Arrow bridge enabled")
    set(HAVE_ARROW TRUE)
else()
    message(STATUS "Apache Arrow not found — Arrow bridge disabled")
    set(HAVE_ARROW FALSE)
endif()

# ── Core library ─────────────────────────────────────────────────────────────
add_library(gpudb_core STATIC
    src/core/table.cpp
    src/core/schema.cpp
)
target_include_directories(gpudb_core PUBLIC src)

# ── I/O library ──────────────────────────────────────────────────────────────
set(IO_SOURCES src/io/csv_reader.cpp)
if(HAVE_ARROW)
    list(APPEND IO_SOURCES src/io/arrow_bridge.cpp)
endif()
add_library(gpudb_io STATIC ${IO_SOURCES})
target_include_directories(gpudb_io PUBLIC src)
target_link_libraries(gpudb_io PUBLIC gpudb_core)
if(HAVE_ARROW)
    target_link_libraries(gpudb_io PUBLIC Arrow::arrow_shared)
    target_compile_definitions(gpudb_io PUBLIC GPUDB_HAS_ARROW=1)
endif()

# ── SQL library ──────────────────────────────────────────────────────────────
add_library(gpudb_sql STATIC src/sql/lexer.cpp src/sql/parser.cpp)
target_include_directories(gpudb_sql PUBLIC src)
target_link_libraries(gpudb_sql PUBLIC gpudb_core)

# ── Query engine library ─────────────────────────────────────────────────────
add_library(gpudb_query STATIC src/query/operator_node.cpp src/query/planner.cpp)
target_include_directories(gpudb_query PUBLIC src)
target_link_libraries(gpudb_query PUBLIC gpudb_core gpudb_sql)

# ── GPU library (CUDA kernels) ───────────────────────────────────────────────
if(USE_CUDA)
    add_library(gpudb_gpu STATIC
        src/gpu/gpu_filter.cu
        src/gpu/gpu_aggregate.cu
        src/gpu/gpu_groupby.cu
        src/gpu/gpu_join.cu
        src/gpu/gpu_pipeline.cu
        src/gpu/gpu_compressed.cu
        src/gpu/gpu_gather.cu
        src/gpu/gpu_graph.cu
        src/gpu/arrow_gpu_transfer.cu
        src/backend/cuda_backend.cu
    )
    target_include_directories(gpudb_gpu PUBLIC src)
    target_link_libraries(gpudb_gpu PUBLIC gpudb_core)
    target_compile_definitions(gpudb_gpu PUBLIC USE_CUDA=1)
endif()

# ── Demo executable (CPU SQL engine) ─────────────────────────────────────────
add_executable(gpudb_demo src/main.cpp)
target_link_libraries(gpudb_demo PRIVATE gpudb_core gpudb_io gpudb_sql gpudb_query)

# ── Interactive CLI REPL (requires CUDA) ─────────────────────────────────────
if(USE_CUDA)
    add_executable(gpudb_cli src/cli/repl.cpp)
    target_include_directories(gpudb_cli PRIVATE src)
    target_link_libraries(gpudb_cli PRIVATE gpudb_core gpudb_io gpudb_sql gpudb_query gpudb_gpu)
    target_compile_definitions(gpudb_cli PRIVATE USE_CUDA=1)
endif()

# ── CPU benchmark ────────────────────────────────────────────────────────────
add_executable(gpudb_bench src/benchmark.cpp)
target_link_libraries(gpudb_bench PRIVATE gpudb_core gpudb_query)

if(MSVC)
    target_compile_options(gpudb_bench PRIVATE /O2)
else()
    target_compile_options(gpudb_bench PRIVATE -O2)
endif()

# ── GPU benchmark ────────────────────────────────────────────────────────────
if(USE_CUDA)
    add_executable(gpudb_gpu_bench src/gpu_benchmark.cu)
    target_include_directories(gpudb_gpu_bench PRIVATE src)
    target_link_libraries(gpudb_gpu_bench PRIVATE gpudb_gpu gpudb_core)

    # ── Performance report generator ─────────────────────────────────────────
    add_executable(gpudb_report src/benchmark/perf_report.cpp)
    target_include_directories(gpudb_report PRIVATE src)
    target_link_libraries(gpudb_report PRIVATE gpudb_gpu gpudb_core)
    target_compile_definitions(gpudb_report PRIVATE USE_CUDA=1)

    if(MSVC)
        target_compile_options(gpudb_gpu_bench PRIVATE /O2)
    else()
        target_compile_options(gpudb_gpu_bench PRIVATE -O2)
    endif()
endif()

# ── Google Test suite ────────────────────────────────────────────────────────
if(BUILD_TESTS)
    find_package(GTest REQUIRED)
    add_executable(gpudb_tests tests/test_correctness.cpp)
    target_include_directories(gpudb_tests PRIVATE src)
    if(USE_CUDA)
        target_link_libraries(gpudb_tests PRIVATE
            GTest::gtest GTest::gtest_main
            gpudb_core gpudb_io gpudb_query gpudb_gpu)
        target_compile_definitions(gpudb_tests PRIVATE USE_CUDA=1)
    else()
        target_link_libraries(gpudb_tests PRIVATE
            GTest::gtest GTest::gtest_main
            gpudb_core gpudb_io gpudb_query)
    endif()
    if(HAVE_ARROW)
        target_compile_definitions(gpudb_tests PRIVATE GPUDB_HAS_ARROW=1)
    endif()
    enable_testing()
    add_test(NAME correctness COMMAND gpudb_tests)
endif()
