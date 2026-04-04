/**
 * @file gpu_compressed.cu
 * @brief GPU kernels operating directly on compressed data (dict + RLE).
 */
#include "cuda_utils.cuh"
#include "gpu_ops.cuh"
#include <cstdio>
#include <vector>

namespace gpudb {

// ── Dict-encoded GROUP BY SUM ───────────────────────────────────────────────
// Groups by dictionary codes (int32) instead of strings — 10x+ faster.

__global__ void gpu_dict_groupby_sum_kernel(const int* codes, const double* vals,
                                            int n, double* sums, int num_groups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        atomicAdd(&sums[codes[i]], vals[i]);
    }
}

void gpuDictGroupBySum(const int* d_codes, const double* d_vals, int n,
                       double* h_sums, int num_groups) {
    GpuBuffer<double> d_sums(num_groups);
    d_sums.memsetZero();

    int blk = 256, grd = std::min((n + blk - 1) / blk, 4096);
    gpu_dict_groupby_sum_kernel<<<grd, blk>>>(d_codes, d_vals, n,
                                               d_sums.data(), num_groups);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_sums, d_sums.data(), num_groups * sizeof(double),
                           cudaMemcpyDeviceToHost));
}

// ── RLE aggregation kernel ──────────────────────────────────────────────────
// Operates directly on (value, run_length) pairs — no decompression.

__global__ void gpu_rle_sum_kernel(const int* rle_vals, const int* rle_lens,
                                   int num_runs, double* group_sums,
                                   const double* per_row_vals, int* prefix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_runs) return;

    int val = rle_vals[idx];
    int len = rle_lens[idx];
    int start = prefix[idx];  // prefix sum of run lengths

    double sum = 0;
    for (int i = start; i < start + len; ++i)
        sum += per_row_vals[i];

    atomicAdd(&group_sums[val], sum);
}

} // namespace gpudb
