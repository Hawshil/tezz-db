/**
 * @file gpu_pipeline.cu
 * @brief Double-buffered async query pipeline using CUDA Streams.
 *
 * Architecture
 * ────────────
 * Two streams alternate between PCIe transfer and kernel execution:
 *   Stream 0: [copy chunk 0] [kernel chunk 0]           [copy chunk 2] ...
 *   Stream 1:                [copy chunk 1] [kernel chunk 1]           ...
 *
 * This overlaps H2D transfer of chunk i+1 with kernel execution on chunk i,
 * hiding PCIe latency when the kernel is compute-bound.
 *
 * Bottleneck Analysis
 * ───────────────────
 * • If kernel_time > transfer_time: pipeline is compute-bound.
 *   Transfer is fully hidden; adding more SMs would help.
 * • If transfer_time > kernel_time: pipeline is PCIe-bound.
 *   Kernel finishes before next chunk arrives; wider bus / NVLink helps.
 * • Optimal: kernel_time ≈ transfer_time (both fully utilised).
 */
#include "cuda_utils.cuh"
#include "gpu_ops.cuh"
#include <algorithm>
#include <cstdio>
#include <vector>

namespace gpudb {

// Reuse the groupby kernel from gpu_groupby.cu (declaration only needed here)
extern __global__ void gpu_groupby_sum_kernel(
    const int* keys, const double* vals, int n,
    int* ht_keys, double* ht_vals, int ht_cap);

void runAsyncPipeline(const double* h_vals, const int* h_keys,
                      int total_rows, int chunk_size, int num_groups) {
    int num_chunks = (total_rows + chunk_size - 1) / chunk_size;
    int ht_cap = num_groups * 4;

    // Create two streams
    cudaStream_t streams[2];
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));

    // Double device buffers
    GpuBuffer<double> d_vals[2] = { GpuBuffer<double>(chunk_size),
                                     GpuBuffer<double>(chunk_size) };
    GpuBuffer<int>    d_keys[2] = { GpuBuffer<int>(chunk_size),
                                     GpuBuffer<int>(chunk_size) };

    // Global hash table (accumulates across all chunks)
    GpuBuffer<int>    d_ht_keys(ht_cap);
    GpuBuffer<double> d_ht_vals(ht_cap);
    d_ht_keys.fill(0xFF);
    d_ht_vals.memsetZero();

    // Events for timing
    cudaEvent_t ev_start, ev_stop, ev_copy[2], ev_kern[2];
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaEventCreate(&ev_copy[s]));
        CUDA_CHECK(cudaEventCreate(&ev_kern[s]));
    }

    float total_copy_ms = 0, total_kern_ms = 0;

    CUDA_CHECK(cudaEventRecord(ev_start));

    for (int c = 0; c < num_chunks; ++c) {
        int s = c % 2;
        int off = c * chunk_size;
        int n = std::min(chunk_size, total_rows - off);
        int blk = 256, grd = std::min((n + blk - 1) / blk, 2048);

        // ── Async H2D copy on stream[s] ────────────────────────────────────
        cudaEvent_t ev_c0, ev_c1;
        CUDA_CHECK(cudaEventCreate(&ev_c0));
        CUDA_CHECK(cudaEventCreate(&ev_c1));
        CUDA_CHECK(cudaEventRecord(ev_c0, streams[s]));

        CUDA_CHECK(cudaMemcpyAsync(d_vals[s].data(), h_vals + off,
                                    n * sizeof(double), cudaMemcpyHostToDevice, streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(d_keys[s].data(), h_keys + off,
                                    n * sizeof(int), cudaMemcpyHostToDevice, streams[s]));

        CUDA_CHECK(cudaEventRecord(ev_c1, streams[s]));

        // ── Launch kernel on stream[s] ─────────────────────────────────────
        cudaEvent_t ev_k0, ev_k1;
        CUDA_CHECK(cudaEventCreate(&ev_k0));
        CUDA_CHECK(cudaEventCreate(&ev_k1));
        CUDA_CHECK(cudaEventRecord(ev_k0, streams[s]));

        gpu_groupby_sum_kernel<<<grd, blk, 0, streams[s]>>>(
            d_keys[s].data(), d_vals[s].data(), n,
            d_ht_keys.data(), d_ht_vals.data(), ht_cap);

        CUDA_CHECK(cudaEventRecord(ev_k1, streams[s]));

        // Accumulate per-chunk timings after sync
        CUDA_CHECK(cudaEventSynchronize(ev_c1));
        CUDA_CHECK(cudaEventSynchronize(ev_k1));
        float copy_ms, kern_ms;
        CUDA_CHECK(cudaEventElapsedTime(&copy_ms, ev_c0, ev_c1));
        CUDA_CHECK(cudaEventElapsedTime(&kern_ms, ev_k0, ev_k1));
        total_copy_ms += copy_ms;
        total_kern_ms += kern_ms;

        cudaEventDestroy(ev_c0); cudaEventDestroy(ev_c1);
        cudaEventDestroy(ev_k0); cudaEventDestroy(ev_k1);
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float wall_ms;
    CUDA_CHECK(cudaEventElapsedTime(&wall_ms, ev_start, ev_stop));

    // ── Read results ────────────────────────────────────────────────────────
    std::vector<int>    h_ht_k(ht_cap);
    std::vector<double> h_ht_v(ht_cap);
    CUDA_CHECK(cudaMemcpy(h_ht_k.data(), d_ht_keys.data(), ht_cap*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ht_v.data(), d_ht_vals.data(), ht_cap*8, cudaMemcpyDeviceToHost));

    int groups_found = 0;
    for (int i = 0; i < ht_cap; ++i) if (h_ht_k[i] != -1) ++groups_found;

    // ── Report ──────────────────────────────────────────────────────────────
    double data_gb = (double)total_rows * (sizeof(double) + sizeof(int)) / 1e9;
    std::printf("  ┌── Async Pipeline Report ──────────────────────┐\n");
    std::printf("  │  Chunks:  %d × %dM rows                      \n",
                num_chunks, chunk_size / 1000000);
    std::printf("  │  Σ PCIe copy time : %8.1f ms                \n", total_copy_ms);
    std::printf("  │  Σ Kernel time    : %8.1f ms                \n", total_kern_ms);
    std::printf("  │  Wall-clock time  : %8.1f ms                \n", wall_ms);
    std::printf("  │  PCIe utilisation : %5.1f%%                   \n",
                total_copy_ms / wall_ms * 100.0);
    std::printf("  │  Effective BW     : %5.1f GB/s               \n",
                data_gb / (wall_ms / 1000.0));
    std::printf("  │  Groups found     : %d                       \n", groups_found);
    std::printf("  └───────────────────────────────────────────────┘\n");
    if (total_kern_ms > total_copy_ms)
        std::printf("  → Pipeline is COMPUTE-BOUND (kernel > transfer).\n");
    else
        std::printf("  → Pipeline is PCIe-BOUND (transfer > kernel).\n");

    // Cleanup
    for (int s = 0; s < 2; ++s) CUDA_CHECK(cudaStreamDestroy(streams[s]));
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    for (int s = 0; s < 2; ++s) { cudaEventDestroy(ev_copy[s]); cudaEventDestroy(ev_kern[s]); }
}

} // namespace gpudb
