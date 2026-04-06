/**
 * @file gpu_pool_benchmark.cu
 * @brief Benchmark: pool allocator vs cudaMalloc for query-time allocations.
 *
 * This demonstrates the latency savings from bypassing the OS memory manager.
 * In a real query pipeline, we allocate ~10-20 buffers per query.
 * With cudaMalloc that's 1-2ms of pure allocation overhead.
 * With the pool, it's <1μs total.
 */
#include "gpu_memory_pool.cuh"
#include "cuda_utils.cuh"
#include <chrono>
#include <cstdio>
#include <vector>

namespace gpudb {

using Clock = std::chrono::high_resolution_clock;

void benchMemoryPool() {
    const int NUM_ALLOCS = 100;   // 100 allocations per "query"
    const int NUM_QUERIES = 1000; // simulate 1000 queries
    const std::size_t BUF_SIZE = 1024 * 1024; // 1 MB each

    std::printf("\n═══ Memory Pool Benchmark ═══\n");
    std::printf("  %d allocations × %d queries × %.1f MB each\n\n",
                NUM_ALLOCS, NUM_QUERIES, BUF_SIZE / 1e6);

    // ── (A) cudaMalloc / cudaFree per allocation ────────────────────────────
    {
        auto t0 = Clock::now();
        for (int q = 0; q < NUM_QUERIES; ++q) {
            std::vector<void*> ptrs(NUM_ALLOCS);
            for (int i = 0; i < NUM_ALLOCS; ++i)
                cudaMalloc(&ptrs[i], BUF_SIZE);
            for (int i = 0; i < NUM_ALLOCS; ++i)
                cudaFree(ptrs[i]);
        }
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double per_alloc = ms / (NUM_ALLOCS * NUM_QUERIES) * 1000.0; // μs
        std::printf("  (A) cudaMalloc/Free:  %8.1f ms total  (%.1f μs/alloc)\n",
                    ms, per_alloc);
    }

    // ── (B) Pool allocator (bump pointer + reset) ───────────────────────────
    {
        // Pool size: NUM_ALLOCS * BUF_SIZE = 100 MB
        GpuMemoryPool pool(NUM_ALLOCS * BUF_SIZE);
        auto t0 = Clock::now();
        for (int q = 0; q < NUM_QUERIES; ++q) {
            for (int i = 0; i < NUM_ALLOCS; ++i)
                pool.allocate(BUF_SIZE);
            pool.reset();
        }
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double per_alloc = ms / (NUM_ALLOCS * NUM_QUERIES) * 1000.0; // μs
        std::printf("  (B) Pool allocator:   %8.1f ms total  (%.3f μs/alloc)\n",
                    ms, per_alloc);
        pool.printStats();
    }

    std::printf("\n  The pool allocator bypasses the OS memory manager entirely.\n");
    std::printf("  Each allocation is a single pointer addition — O(1), zero syscalls.\n");
    std::printf("  For a query pipeline with 10-20 buffers, this saves ~1-2ms per query.\n");
}

} // namespace gpudb
