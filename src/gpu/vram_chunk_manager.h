/**
 * @file vram_chunk_manager.h
 * @brief Manages GPU VRAM allocations for RingTable chunks — lazy upload,
 *        caching, and LRU-style eviction.
 *
 * Only compiled when USE_CUDA is defined. Provides a getChunk() API that
 * uploads a double-column chunk to the GPU on first access, and returns the
 * cached device pointer on subsequent calls.
 */
#pragma once

#ifdef USE_CUDA

#include "cuda_utils.cuh"
#include "../core/ring_table.h"

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace gpudb {

class VramChunkManager {
public:
    /**
     * @param max_vram_bytes  Maximum bytes to keep in GPU VRAM for chunks.
     *                        When exceeded, the oldest chunks are evicted.
     */
    explicit VramChunkManager(std::size_t max_vram_bytes =
                                  2ULL * 1024 * 1024 * 1024)
        : max_vram_bytes_(max_vram_bytes) {}

    /**
     * @brief Upload chunk chunk_idx of double-column col_idx from rt to VRAM
     *        if not already present. Returns device pointer.
     *
     * Thread-safe (uses internal mutex).
     */
    const double* getChunk(const RingTable& rt,
                           std::size_t col_idx,
                           std::size_t chunk_idx) {
        std::lock_guard<std::mutex> lock(mtx_);

        auto k = key(col_idx, chunk_idx);
        auto it = cache_.find(k);
        if (it != cache_.end()) {
            // Cache hit — return existing device pointer.
            return it->second.data();
        }

        // Cache miss — upload the chunk.
        const auto& ring_var = rt.getCol(col_idx);
        const auto& ring_col = std::get<RingColumn<double>>(ring_var);
        const auto& chunks = ring_col.chunks();

        if (chunk_idx >= chunks.size())
            throw std::out_of_range(
                "VramChunkManager::getChunk — chunk_idx out of range");

        const auto& chunk = chunks[chunk_idx];
        std::size_t n = chunk.data.size();
        std::size_t bytes = n * sizeof(double);

        // Evict if needed.
        evictLRUIfNeeded(bytes);

        // Allocate and upload.
        GpuBuffer<double> buf(n);
        CUDA_CHECK(cudaMemcpy(buf.data(), chunk.data.data(),
                              bytes, cudaMemcpyHostToDevice));

        used_bytes_ += bytes;
        auto* ptr = buf.data();
        cache_.emplace(k, std::move(buf));
        // Track access order for LRU.
        access_order_.push_back(k);

        return ptr;
    }

    /**
     * @brief Remove a chunk from VRAM (e.g. when ring evicts it).
     */
    void evict(std::size_t col_idx, std::size_t chunk_idx) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto k = key(col_idx, chunk_idx);
        auto it = cache_.find(k);
        if (it != cache_.end()) {
            used_bytes_ -= it->second.size() * sizeof(double);
            cache_.erase(it);
            // Remove from access order.
            access_order_.erase(
                std::remove(access_order_.begin(), access_order_.end(), k),
                access_order_.end());
        }
    }

    /// Total bytes currently in VRAM across all managed chunks.
    std::size_t usedBytes() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return used_bytes_;
    }

    void printStats() const {
        std::lock_guard<std::mutex> lock(mtx_);
        std::printf("  VramChunkManager: %zu chunks cached, %.1f MB used / %.1f MB max\n",
                    cache_.size(),
                    used_bytes_ / 1e6,
                    max_vram_bytes_ / 1e6);
    }

private:
    std::size_t max_vram_bytes_;
    std::size_t used_bytes_ = 0;
    mutable std::mutex mtx_;

    /// key: (col_idx << 32 | chunk_idx) → GpuBuffer<double>
    std::unordered_map<std::uint64_t, GpuBuffer<double>> cache_;

    /// Access order for LRU eviction (front = oldest).
    std::vector<std::uint64_t> access_order_;

    /// Evict oldest cached chunks until enough space is available.
    void evictLRUIfNeeded(std::size_t needed_bytes) {
        while (used_bytes_ + needed_bytes > max_vram_bytes_ &&
               !access_order_.empty()) {
            auto oldest_key = access_order_.front();
            access_order_.erase(access_order_.begin());
            auto it = cache_.find(oldest_key);
            if (it != cache_.end()) {
                used_bytes_ -= it->second.size() * sizeof(double);
                cache_.erase(it);
            }
        }
    }

    std::uint64_t key(std::size_t col, std::size_t chunk) const {
        return (static_cast<std::uint64_t>(col) << 32) |
               static_cast<std::uint64_t>(chunk);
    }
};

}  // namespace gpudb

#endif  // USE_CUDA
