/**
 * @file gpu_filter.cu
 * @brief GPU parallel column filter with stream compaction.
 *
 * Pipeline: filter_kernel → thrust::exclusive_scan → compact_kernel.
 */
#include "cuda_utils.cuh"
#include "gpu_ops.cuh"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace gpudb {

__global__ void gpu_filter_kernel(const double* col, int n, double threshold,
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

__global__ void gpu_compact_kernel(const double* in, const int* mask,
                                   const int* pos, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && mask[i]) out[pos[i]] = in[i];
}

int gpuFilter(const double* d_in, int n, double threshold,
              CompareOp op, double* d_out) {
    const int BLK = 256;
    const int GRD = (n + BLK - 1) / BLK;

    GpuBuffer<int> d_mask(n), d_pos(n);

    // 1. Mark matching elements
    gpu_filter_kernel<<<GRD, BLK>>>(d_in, n, threshold, (int)op, d_mask.data());

    // 2. Prefix sum → output positions
    thrust::device_ptr<int> mp(d_mask.data()), pp(d_pos.data());
    thrust::exclusive_scan(mp, mp + n, pp);

    // 3. Get total output count
    int last_mask = 0, last_pos = 0;
    CUDA_CHECK(cudaMemcpy(&last_mask, d_mask.data()+n-1, 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_pos,  d_pos.data()+n-1,  4, cudaMemcpyDeviceToHost));
    int total = last_pos + last_mask;

    // 4. Compact
    gpu_compact_kernel<<<GRD, BLK>>>(d_in, d_mask.data(), d_pos.data(), d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    return total;
}

} // namespace gpudb
