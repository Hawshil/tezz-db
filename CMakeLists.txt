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
    # CUDA 13.0 only officially supports VS 2019-2022; allow VS 2026 (v18)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# if(MSVC)
#     add_compile_options(/W4 /permissive-)
# else()
#     add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

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
        src/gpu/gpu_pool_benchmark.cu
        src/gpu/arrow_gpu_transfer.cu
        src/gpu/gpu_window.cu
        src/gpu/gpu_asof.cu
        src/backend/cuda_backend.cu
    )
    target_include_directories(gpudb_gpu PUBLIC src)
    target_link_libraries(gpudb_gpu PUBLIC gpudb_core)
    target_compile_definitions(gpudb_gpu PUBLIC USE_CUDA=1)
endif()

# ── Demo executable (CPU SQL engine) ─────────────────────────────────────────
add_executable(gpudb_demo src/main.cpp)
target_link_libraries(gpudb_demo PRIVATE gpudb_core gpudb_io gpudb_sql gpudb_query)

# ── Ring-buffer smoke test ───────────────────────────────────────────────────
add_executable(ring_test src/ring_test.cpp)
target_include_directories(ring_test PRIVATE src)
target_link_libraries(ring_test PRIVATE gpudb_core)

# ── Live tick-feed demo (CPU-only) ───────────────────────────────────────────
add_executable(gpudb_tick_demo src/tick_feed_demo.cpp)
target_include_directories(gpudb_tick_demo PRIVATE src)
target_link_libraries(gpudb_tick_demo PRIVATE gpudb_core gpudb_sql gpudb_query)

# ── ASOF join smoke test (CPU-only) ──────────────────────────────────────────
add_executable(asof_join_test src/asof_join_test.cpp)
target_include_directories(asof_join_test PRIVATE src)
target_link_libraries(asof_join_test PRIVATE gpudb_core gpudb_sql gpudb_query)

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
        target_compile_options(gpudb_gpu_bench PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/O2>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/O2>)
    else()
        target_compile_options(gpudb_gpu_bench PRIVATE -O2)
    endif()
    # ── Window function benchmark ────────────────────────────────────────────
    add_executable(gpudb_window_bench src/gpu_window_benchmark.cu)
    target_include_directories(gpudb_window_bench PRIVATE src)
    target_link_libraries(gpudb_window_bench PRIVATE gpudb_gpu gpudb_core)
    if(MSVC)
        target_compile_options(gpudb_window_bench PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/O2>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/O2>)
    else()
        target_compile_options(gpudb_window_bench PRIVATE -O2)
    endif()
    # ── Finance benchmark (CPU vs GPU) ───────────────────────────────────────
    add_executable(gpudb_finance_bench src/gpu_finance_bench.cu)
    target_include_directories(gpudb_finance_bench PRIVATE src)
    target_link_libraries(gpudb_finance_bench PRIVATE
        gpudb_gpu gpudb_core gpudb_sql gpudb_query)
    if(MSVC)
        target_compile_options(gpudb_finance_bench PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/O2>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/O2>)
    else()
        target_compile_options(gpudb_finance_bench PRIVATE -O2)
    endif()
endif()

# ── Google Test suite (modular per-component tests) ──────────────────────────
if(BUILD_TESTS)
    find_package(GTest REQUIRED)
    enable_testing()

    # Parser tests (CPU-only, no GPU needed)
    add_executable(test_parser tests/parser_test.cpp)
    target_include_directories(test_parser PRIVATE src)
    target_link_libraries(test_parser PRIVATE
        GTest::gtest GTest::gtest_main gpudb_core gpudb_sql)
    add_test(NAME parser COMMAND test_parser)

    # Arrow bridge tests (CPU-only, requires Arrow)
    add_executable(test_arrow tests/arrow_bridge_test.cpp)
    target_include_directories(test_arrow PRIVATE src)
    target_link_libraries(test_arrow PRIVATE
        GTest::gtest GTest::gtest_main gpudb_core gpudb_io)
    if(HAVE_ARROW)
        target_compile_definitions(test_arrow PRIVATE GPUDB_HAS_ARROW=1)
    endif()
    add_test(NAME arrow_bridge COMMAND test_arrow)

    # GPU kernel tests (requires CUDA)
    if(USE_CUDA)
        add_executable(test_gpu tests/gpu_filter_test.cu)
        target_include_directories(test_gpu PRIVATE src)
        target_link_libraries(test_gpu PRIVATE
            GTest::gtest gpudb_gpu gpudb_core)
        target_compile_definitions(test_gpu PRIVATE USE_CUDA=1)
        add_test(NAME gpu_kernels COMMAND test_gpu)
    endif()

    # Full integration tests
    add_executable(test_integration tests/test_correctness.cpp)
    target_include_directories(test_integration PRIVATE src)
    if(USE_CUDA)
        target_link_libraries(test_integration PRIVATE
            GTest::gtest GTest::gtest_main
            gpudb_core gpudb_io gpudb_query gpudb_gpu)
        target_compile_definitions(test_integration PRIVATE USE_CUDA=1)
    else()
        target_link_libraries(test_integration PRIVATE
            GTest::gtest GTest::gtest_main
            gpudb_core gpudb_io gpudb_query)
    endif()
    if(HAVE_ARROW)
        target_compile_definitions(test_integration PRIVATE GPUDB_HAS_ARROW=1)
    endif()
    add_test(NAME integration COMMAND test_integration)

    # Ring-buffer concurrency tests (CPU-only)
    add_executable(test_ring_buffer tests/ring_buffer_test.cpp)
    target_include_directories(test_ring_buffer PRIVATE src)
    target_link_libraries(test_ring_buffer PRIVATE
        GTest::gtest GTest::gtest_main gpudb_core)
    add_test(NAME ring_buffer COMMAND test_ring_buffer)

    # Finance integration tests (CPU-only)
    add_executable(test_finance tests/finance_test.cpp)
    target_include_directories(test_finance PRIVATE src)
    target_link_libraries(test_finance PRIVATE
        GTest::gtest GTest::gtest_main gpudb_core gpudb_sql gpudb_query)
    add_test(NAME finance COMMAND test_finance)
endif()

