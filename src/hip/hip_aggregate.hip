/**
 * @file hip_aggregate.hip
 * @brief HIP port of gpu_aggregate.cu — SUM and COUNT with wavefront shuffle.
 *
 * Wavefront-Width Adjustments (64 vs 32)
 * ───────────────────────────────────────
 * 1. Shared-memory tree reduction now stops at warpSize (64 on CDNA, 32 on RDNA)
 *    instead of the CUDA hardcoded 32.
 * 2. The wavefront shuffle loop runs 6 iterations (64→1) instead of 5 (32→1)
 *    on CDNA hardware.
 * 3. __shfl_down_sync(mask, val, off) → __shfl_down(val, off, warpSize)
 *    HIP's shuffle has no explicit mask — all lanes in the wavefront participate.
 * 4. We use the built-in `warpSize` for runtime portability across CDNA/RDNA.
 */
#include "hip_utils.h"
#include "hip_ops.h"

namespace gpudb {

__global__ void hip_sum_kernel(const double* input, int n, double* result) {
    extern __shared__ double sdata[];
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x * 2 + tid;

    double val = (idx < (unsigned)n) ? input[idx] : 0.0;
    if (idx + blockDim.x < (unsigned)n) val += input[idx + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    // Tree reduction in shared memory — stop at wavefront size (not 32).
    for (unsigned s = blockDim.x / 2; s > (unsigned)warpSize; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Wavefront-level reduction using __shfl_down (no mask in HIP).
    if (tid < (unsigned)warpSize) {
        double wval = sdata[tid];
        if (blockDim.x >= 2u * warpSize) wval += sdata[tid + warpSize];
        // 6 steps on CDNA (64→1), 5 steps on RDNA (32→1)
        for (int off = warpSize / 2; off > 0; off >>= 1)
            wval += __shfl_down(wval, off, warpSize);

        if (tid == 0) atomicAdd(result, wval);
    }
}

double hipSum(const double* d_in, int n) {
    const int BLK = 256;
    int grid = (n + BLK * 2 - 1) / (BLK * 2);
    HipBuffer<double> d_res(1); d_res.memsetZero();

    hip_sum_kernel<<<grid, BLK, BLK * sizeof(double)>>>(d_in, n, d_res.data());
    HIP_CHECK(hipDeviceSynchronize());

    double h;
    HIP_CHECK(hipMemcpy(&h, d_res.data(), sizeof(double), hipMemcpyDeviceToHost));
    return h;
}

// ── COUNT ───────────────────────────────────────────────────────────────────
__global__ void hip_count_kernel(const int* mask, int n, long long* result) {
    extern __shared__ long long sidata[];
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x * 2 + tid;

    long long val = (idx < (unsigned)n) ? mask[idx] : 0;
    if (idx + blockDim.x < (unsigned)n) val += mask[idx + blockDim.x];
    sidata[tid] = val;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > (unsigned)warpSize; s >>= 1) {
        if (tid < s) sidata[tid] += sidata[tid + s];
        __syncthreads();
    }
    if (tid < (unsigned)warpSize) {
        long long wval = sidata[tid];
        if (blockDim.x >= 2u * warpSize) wval += sidata[tid + warpSize];
        for (int off = warpSize / 2; off > 0; off >>= 1)
            wval += __shfl_down(wval, off, warpSize);
        if (tid == 0) atomicAdd(result, wval);
    }
}

std::int64_t hipCount(const int* d_in, int n) {
    const int BLK = 256;
    int grid = (n + BLK * 2 - 1) / (BLK * 2);
    HipBuffer<long long> d_res(1); d_res.memsetZero();

    hip_count_kernel<<<grid, BLK, BLK * sizeof(long long)>>>(d_in, n, d_res.data());
    HIP_CHECK(hipDeviceSynchronize());

    long long h;
    HIP_CHECK(hipMemcpy(&h, d_res.data(), sizeof(long long), hipMemcpyDeviceToHost));
    return static_cast<std::int64_t>(h);
}

} // namespace gpudb
