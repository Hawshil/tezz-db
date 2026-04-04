/**
 * @file gpu_gather.cu
 * @brief GPU gather kernel for late materialization + selective GPU transfer.
 *
 * Instead of transferring 100M rows, transfer only the 1M rows that passed
 * the CPU predicate. This kernel gathers scattered source rows into a
 * dense output buffer on GPU.
 */
#include "cuda_utils.cuh"
#include "gpu_ops.cuh"
#include <cstdio>
#include <chrono>
#include <vector>

namespace gpudb {

__global__ void gpu_gather_kernel(const double* src, const int* indices,
                                  int sel_count, double* dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sel_count) dst[i] = src[indices[i]];
}

void gpuGather(const double* d_src, const int* d_indices, int sel_count,
               double* d_out) {
    int blk = 256, grd = (sel_count + blk - 1) / blk;
    gpu_gather_kernel<<<grd, blk>>>(d_src, d_indices, sel_count, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ── RLE aggregation wrapper (prefix-sum based) ──────────────────────────────

extern __global__ void gpu_rle_sum_kernel(const int*, const int*, int,
                                          double*, const double*, int*);

void gpuRleAggSum(const int* d_rle_vals, const int* d_rle_lens, int num_runs,
                  const double* d_vals, double* h_group_sums, int num_groups) {
    // Compute prefix sum of run lengths on CPU (small data)
    std::vector<int> h_lens(num_runs), h_prefix(num_runs);
    CUDA_CHECK(cudaMemcpy(h_lens.data(), d_rle_lens, num_runs * sizeof(int),
                           cudaMemcpyDeviceToHost));
    h_prefix[0] = 0;
    for (int i = 1; i < num_runs; ++i) h_prefix[i] = h_prefix[i-1] + h_lens[i-1];

    GpuBuffer<int> d_prefix(num_runs);
    CUDA_CHECK(cudaMemcpy(d_prefix.data(), h_prefix.data(), num_runs * sizeof(int),
                           cudaMemcpyHostToDevice));

    GpuBuffer<double> d_sums(num_groups);
    d_sums.memsetZero();

    int blk = 256, grd = (num_runs + blk - 1) / blk;
    gpu_rle_sum_kernel<<<grd, blk>>>(d_rle_vals, d_rle_lens, num_runs,
                                      d_sums.data(), d_vals, d_prefix.data());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_group_sums, d_sums.data(), num_groups * sizeof(double),
                           cudaMemcpyDeviceToHost));
}

// ═══════════════════════════════════════════════════════════════════════════
// Late materialisation GPU benchmark:
//   Compare full-column GPU transfer vs selective gather
// ═══════════════════════════════════════════════════════════════════════════

void benchLateMaterialization(const double* h_src, const int* h_sel,
                               int total_rows, int sel_count) {
    using Clock = std::chrono::high_resolution_clock;
    std::printf("\n═══ Late Materialisation GPU Benchmark ═══\n");
    std::printf("  Total rows: %d  Selected: %d  Selectivity: %.2f%%\n",
                total_rows, sel_count, 100.0 * sel_count / total_rows);

    // (A) Full column transfer → GPU filter
    {
        auto t0 = Clock::now();
        GpuBuffer<double> d_all(total_rows);
        CUDA_CHECK(cudaMemcpy(d_all.data(), h_src, (size_t)total_rows * 8,
                               cudaMemcpyHostToDevice));
        // Filter on GPU
        GpuBuffer<double> d_out(total_rows);
        int cnt = gpuFilter(d_all.data(), total_rows, 0.5, CompareOp::GT, d_out.data());
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::size_t bytes = (size_t)total_rows * 8;
        std::printf("  (A) Full transfer:   %7.1f ms  PCIe: %zu MB  result: %d rows\n",
                    ms, bytes / (1024*1024), cnt);
    }

    // (B) CPU predicate → transfer only selected → GPU gather
    {
        auto t0 = Clock::now();
        // Transfer indices (small: sel_count * 4 bytes)
        GpuBuffer<int> d_sel(sel_count);
        CUDA_CHECK(cudaMemcpy(d_sel.data(), h_sel, (size_t)sel_count * 4,
                               cudaMemcpyHostToDevice));
        // Gather selected values (sel_count * 8 bytes instead of total * 8)
        std::vector<double> h_selected(sel_count);
        for (int i = 0; i < sel_count; ++i) h_selected[i] = h_src[h_sel[i]];
        GpuBuffer<double> d_gathered(sel_count);
        CUDA_CHECK(cudaMemcpy(d_gathered.data(), h_selected.data(),
                               (size_t)sel_count * 8, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::size_t bytes = (size_t)sel_count * (8 + 4);
        std::printf("  (B) Late material:   %7.1f ms  PCIe: %zu MB  result: %d rows\n",
                    ms, bytes / (1024*1024), sel_count);
    }

    std::printf("  PCIe savings: %.0fx less data transferred\n",
                (double)total_rows / sel_count);
}

} // namespace gpudb
