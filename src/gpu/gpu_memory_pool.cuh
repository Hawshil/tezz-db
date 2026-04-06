/**
 * @file gpu_memory_pool.cuh
 * @brief Pre-allocated GPU memory pool — bypasses the OS memory manager.
 *
 * Why this matters:
 *   cudaMalloc calls the OS/driver to allocate GPU memory, which takes ~100μs
 *   per call. In a query engine that allocates dozens of buffers per query,
 *   this adds up to milliseconds of pure overhead.
 *
 *   This pool pre-allocates a large contiguous block (e.g. 2 GB) at startup.
 *   During queries, allocations are O(1) bump-pointer advances — zero OS calls.
 *   This is the same technique used by production GPU databases (SQream, Kinetica)
 *   and RAPIDS RMM (Resource Memory Manager).
 *
 * Design:
 *   ┌──────────────────────────────────────────────────────────┐
 *   │  Pre-allocated GPU Block (2 GB)                         │
 *   │  ┌─────┬───────┬────┬─────────────────────────────────┐ │
 *   │  │used │ used  │free│          free space              │ │
 *   │  └─────┴───────┴────┴─────────────────────────────────┘ │
 *   │                  ↑ offset_                               │
 *   └──────────────────────────────────────────────────────────┘
 *
 *   allocate(n_bytes) → advances offset, returns pointer
 *   reset()           → resets offset to 0 (reuse all memory)
 *   No individual free — reset between queries (arena-style)
 */
#pragma once
#include "cuda_utils.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace gpudb {

class GpuMemoryPool {
public:
    /**
     * @param pool_bytes Total pool size in bytes (default 2 GB).
     * @param alignment  Alignment for each allocation (default 256 bytes —
     *                   matches CUDA texture alignment requirements).
     */
    explicit GpuMemoryPool(std::size_t pool_bytes = 2ULL * 1024 * 1024 * 1024,
                           std::size_t alignment = 256)
        : total_(pool_bytes), alignment_(alignment), offset_(0),
          peak_(0), alloc_count_(0) {
        CUDA_CHECK(cudaMalloc(&base_, total_));
        std::printf("[MemPool] Pre-allocated %.1f MB GPU memory at %p\n",
                    total_ / 1e6, base_);
    }

    ~GpuMemoryPool() {
        if (base_) {
            cudaFree(base_);
            std::printf("[MemPool] Released %.1f MB (peak usage: %.1f MB, "
                        "%zu allocations served)\n",
                        total_ / 1e6, peak_ / 1e6, alloc_count_);
        }
    }

    // Non-copyable, movable
    GpuMemoryPool(const GpuMemoryPool&) = delete;
    GpuMemoryPool& operator=(const GpuMemoryPool&) = delete;
    GpuMemoryPool(GpuMemoryPool&& o) noexcept
        : base_(o.base_), total_(o.total_), alignment_(o.alignment_),
          offset_(o.offset_), peak_(o.peak_), alloc_count_(o.alloc_count_) {
        o.base_ = nullptr;
    }

    /**
     * Allocate n_bytes from the pool. O(1) bump-pointer — no OS call.
     * Returns nullptr if pool is exhausted.
     */
    void* allocate(std::size_t n_bytes) {
        // Align up
        std::size_t aligned = (n_bytes + alignment_ - 1) & ~(alignment_ - 1);
        if (offset_ + aligned > total_) {
            std::fprintf(stderr, "[MemPool] OOM! Requested %zu bytes, "
                         "only %zu available (pool=%zu)\n",
                         n_bytes, total_ - offset_, total_);
            return nullptr;
        }
        void* ptr = static_cast<std::uint8_t*>(base_) + offset_;
        offset_ += aligned;
        ++alloc_count_;
        if (offset_ > peak_) peak_ = offset_;
        return ptr;
    }

    /**
     * Typed allocate: returns T* for n elements.
     */
    template<typename T>
    T* allocate(std::size_t n_elements) {
        return static_cast<T*>(allocate(n_elements * sizeof(T)));
    }

    /**
     * Reset the pool — all previous allocations become invalid.
     * Call this between queries to reuse memory without OS calls.
     */
    void reset() {
        offset_ = 0;
    }

    // ── Diagnostics ─────────────────────────────────────────────────────────
    std::size_t used()      const { return offset_; }
    std::size_t available() const { return total_ - offset_; }
    std::size_t total()     const { return total_; }
    std::size_t peak()      const { return peak_; }
    std::size_t allocCount() const { return alloc_count_; }

    void printStats() const {
        std::printf("[MemPool] Used: %.1f MB / %.1f MB (%.1f%%)  "
                    "Peak: %.1f MB  Allocs: %zu\n",
                    offset_ / 1e6, total_ / 1e6,
                    100.0 * offset_ / total_,
                    peak_ / 1e6, alloc_count_);
    }

private:
    void*       base_  = nullptr;
    std::size_t total_;
    std::size_t alignment_;
    std::size_t offset_;
    std::size_t peak_;
    std::size_t alloc_count_;
};

/**
 * PoolBuffer<T> — RAII wrapper around pool-allocated GPU memory.
 * Unlike GpuBuffer<T>, this does NOT call cudaMalloc/cudaFree.
 * The pool owns the underlying memory.
 */
template<typename T>
class PoolBuffer {
public:
    PoolBuffer() = default;
    PoolBuffer(GpuMemoryPool& pool, std::size_t count)
        : ptr_(pool.allocate<T>(count)), size_(count) {
        if (!ptr_ && count > 0)
            throw std::runtime_error("GpuMemoryPool allocation failed");
    }

    T*          data()  const { return ptr_; }
    std::size_t size()  const { return size_; }
    std::size_t bytes() const { return size_ * sizeof(T); }

    void copyFrom(const T* h_src, std::size_t count) {
        CUDA_CHECK(cudaMemcpy(ptr_, h_src, count * sizeof(T),
                               cudaMemcpyHostToDevice));
    }
    void copyTo(T* h_dst, std::size_t count) const {
        CUDA_CHECK(cudaMemcpy(h_dst, ptr_, count * sizeof(T),
                               cudaMemcpyDeviceToHost));
    }
    void memsetZero() {
        CUDA_CHECK(cudaMemset(ptr_, 0, size_ * sizeof(T)));
    }

private:
    T*          ptr_  = nullptr;
    std::size_t size_ = 0;
};

} // namespace gpudb
