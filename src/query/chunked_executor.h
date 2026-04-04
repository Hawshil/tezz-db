/**
 * @file chunked_executor.h
 * @brief Out-of-core GPU processing for datasets larger than VRAM.
 *
 * Chunk-size trade-offs:
 *   • Large chunks: fewer PCIe round-trips, better kernel efficiency,
 *     but may not fit in VRAM alongside hash tables and temporaries.
 *   • Small chunks: more kernel launches (each has ~5μs overhead),
 *     more PCIe round-trips, but fits in any GPU.
 *   • Sweet spot: ~50-60% of free VRAM per chunk, leaving headroom for
 *     intermediate buffers. Query with cudaMemGetInfo to auto-tune.
 */
#pragma once
#include "core/table.h"
#include "gpu/cuda_utils.cuh"
#include "gpu/gpu_ops.cuh"
#include <chrono>
#include <cstdio>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace gpudb {

class ChunkedExecutor {
public:
    // Auto-detect chunk size from free VRAM
    static std::size_t autoChunkRows(std::size_t bytes_per_row,
                                      double vram_fraction = 0.5) {
        std::size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        std::size_t usable = (std::size_t)(free_bytes * vram_fraction);
        return usable / bytes_per_row;
    }

    // ── Chunked GROUP BY + SUM ──────────────────────────────────────────────
    static std::unordered_map<int, double>
    chunkedGroupBySum(const int* h_keys, const double* h_vals,
                      std::size_t total_rows, std::size_t chunk_rows, int ht_cap) {
        using Clock = std::chrono::high_resolution_clock;
        auto t_start = Clock::now();

        std::unordered_map<int, double> merged;
        std::size_t num_chunks = (total_rows + chunk_rows - 1) / chunk_rows;
        std::size_t total_transferred = 0;

        std::vector<int>    chunk_keys(ht_cap);
        std::vector<double> chunk_sums(ht_cap);

        for (std::size_t c = 0; c < num_chunks; ++c) {
            std::size_t off = c * chunk_rows;
            std::size_t n = std::min(chunk_rows, total_rows - off);

            // Transfer chunk to GPU
            GpuBuffer<int>    dk(n);
            GpuBuffer<double> dv(n);
            CUDA_CHECK(cudaMemcpy(dk.data(), h_keys + off, n * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dv.data(), h_vals + off, n * 8, cudaMemcpyHostToDevice));
            total_transferred += n * 12;

            // Run GPU kernel
            int groups = gpuGroupBySum(dk.data(), dv.data(), (int)n,
                                       chunk_keys.data(), chunk_sums.data(), ht_cap);

            // Merge partial results on CPU
            for (int i = 0; i < groups; ++i) {
                merged[chunk_keys[i]] += chunk_sums[i];
            }
        }

        auto t_end = Clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double xfer_gb = total_transferred / 1e9;

        std::printf("  ┌── Chunked Executor Report ────────────────────┐\n");
        std::printf("  │  Total rows:     %12zu                  \n", total_rows);
        std::printf("  │  Chunk size:     %12zu rows             \n", chunk_rows);
        std::printf("  │  Chunks:         %12zu                  \n", num_chunks);
        std::printf("  │  Data transferred: %8.2f GB              \n", xfer_gb);
        std::printf("  │  Total time:     %10.1f ms              \n", elapsed);
        std::printf("  │  Throughput:     %10.1f M rows/sec      \n",
                    total_rows / (elapsed / 1000.0) / 1e6);
        std::printf("  │  Groups found:   %12zu                  \n", merged.size());
        std::printf("  └────────────────────────────────────────────────┘\n");

        return merged;
    }

    // ── Chunked Filter ──────────────────────────────────────────────────────
    static std::vector<double>
    chunkedFilter(const double* h_data, std::size_t total_rows,
                  std::size_t chunk_rows, double threshold) {
        std::vector<double> result;
        result.reserve(total_rows / 2);

        for (std::size_t off = 0; off < total_rows; off += chunk_rows) {
            std::size_t n = std::min(chunk_rows, total_rows - off);

            GpuBuffer<double> d_in(n), d_out(n);
            CUDA_CHECK(cudaMemcpy(d_in.data(), h_data + off, n * 8, cudaMemcpyHostToDevice));

            int cnt = gpuFilter(d_in.data(), (int)n, threshold,
                                CompareOp::GT, d_out.data());

            std::vector<double> chunk_out(cnt);
            CUDA_CHECK(cudaMemcpy(chunk_out.data(), d_out.data(), cnt * 8,
                                   cudaMemcpyDeviceToHost));
            result.insert(result.end(), chunk_out.begin(), chunk_out.end());
        }
        return result;
    }
};

} // namespace gpudb
