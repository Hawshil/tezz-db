/**
 * @file hip_filter.hip
 * @brief HIP port of gpu_filter.cu — parallel filter with stream compaction.
 *
 * Changes from CUDA:
 *   - thrust::exclusive_scan → rocprim::exclusive_scan (temp-storage pattern)
 *   - cudaMemcpy → hipMemcpy
 *   - <<<>>> syntax works identically in HIP
 */
#include "hip_utils.h"
#include "hip_ops.h"
#include <rocprim/rocprim.hpp>

namespace gpudb {

__global__ void hip_filter_kernel(const double* col, int n, double threshold,
                                  int op, int* mask) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double v = col[i];
    int pass = 0;
    switch (op) {
        case 0: pass = (v >  threshold); break;
        case 1: pass = (v >= threshold); break;
        case 2: pass = (v <  threshold); break;
        case 3: pass = (v <= threshold); break;
        case 4: pass = (v == threshold); break;
        case 5: pass = (v != threshold); break;
    }
    mask[i] = pass;
}

__global__ void hip_compact_kernel(const double* in, const int* mask,
                                   const int* pos, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && mask[i]) out[pos[i]] = in[i];
}

int hipFilter(const double* d_in, int n, double threshold,
              HipCompareOp op, double* d_out) {
    const int BLK = 256, GRD = (n + BLK - 1) / BLK;
    HipBuffer<int> d_mask(n), d_pos(n);

    // 1. Mark
    hip_filter_kernel<<<GRD, BLK>>>(d_in, n, threshold, (int)op, d_mask.data());

    // 2. rocPRIM exclusive scan (temp-storage two-pass pattern)
    size_t temp_bytes = 0;
    rocprim::exclusive_scan(nullptr, temp_bytes, d_mask.data(), d_pos.data(), 0, n);
    HipBuffer<char> d_temp(temp_bytes);
    rocprim::exclusive_scan(d_temp.data(), temp_bytes,
                            d_mask.data(), d_pos.data(), 0, n);
    HIP_CHECK(hipDeviceSynchronize());

    // 3. Total count
    int last_mask = 0, last_pos = 0;
    HIP_CHECK(hipMemcpy(&last_mask, d_mask.data()+n-1, 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&last_pos,  d_pos.data()+n-1,  4, hipMemcpyDeviceToHost));
    int total = last_pos + last_mask;

    // 4. Compact
    hip_compact_kernel<<<GRD, BLK>>>(d_in, d_mask.data(), d_pos.data(), d_out, n);
    HIP_CHECK(hipDeviceSynchronize());
    return total;
}

} // namespace gpudb
