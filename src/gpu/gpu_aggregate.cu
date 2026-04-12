/**
 * @file gpu_aggregate.cu
 * @brief GPU parallel SUM and COUNT with tree reduction + warp shuffle.
 *
 * Warp Shuffle Optimisation
 * ─────────────────────────
 * The last 5 reduction steps (32→16→8→4→2→1) happen within a single warp.
 * Instead of shared-memory loads + __syncthreads(), we use __shfl_down_sync()
 * which reads a register from another lane in the same warp in a single cycle.
 * This eliminates shared-memory bank conflicts and sync overhead entirely.
 */
#include "cuda_utils.cuh"
#include "gpu_ops.cuh"

namespace gpudb {

// ── SUM kernel ──────────────────────────────────────────────────────────────

__global__ void gpu_sum_kernel(const double* input, int n, double* result) {
    extern __shared__ double sdata[];
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x * 2 + tid;

    // Each thread loads two elements (halves the number of blocks needed).
    double val = (idx < (unsigned)n) ? input[idx] : 0.0;
    if (idx + blockDim.x < (unsigned)n) val += input[idx + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    // Tree reduction in shared memory down to warp size.
    for (unsigned s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Warp-level reduction using __shfl_down_sync (no sync needed).
    if (tid < 32) {
        double wval = sdata[tid];
        if (blockDim.x >= 64) wval += sdata[tid + 32];
        for (int off = 16; off > 0; off >>= 1)
            wval += __shfl_down_sync(0xFFFFFFFF, wval, off);

        // Thread 0 of each block atomically accumulates into global result.
        if (tid == 0) atomicAdd(result, wval);
    }
}

double gpuSum(const double* d_in, int n) {
    const int BLK = 256;
    int grid = (n + BLK * 2 - 1) / (BLK * 2);

    GpuBuffer<double> d_result(1);
    d_result.memsetZero();

    gpu_sum_kernel<<<grid, BLK, BLK * sizeof(double)>>>(d_in, n, d_result.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    double h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result.data(), sizeof(double),
                           cudaMemcpyDeviceToHost));
    return h_result;
}

// ── COUNT kernel (counts 1s in an int mask array) ───────────────────────────

__global__ void gpu_count_kernel(const int* mask, int n, unsigned long long* result) {
    extern __shared__ unsigned long long sidata[];
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x * 2 + tid;

    unsigned long long val = (idx < (unsigned)n) ? mask[idx] : 0;
    if (idx + blockDim.x < (unsigned)n) val += mask[idx + blockDim.x];
    sidata[tid] = val;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sidata[tid] += sidata[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        unsigned long long wval = sidata[tid];
        if (blockDim.x >= 64) wval += sidata[tid + 32];
        for (int off = 16; off > 0; off >>= 1)
            wval += __shfl_down_sync(0xFFFFFFFF, wval, off);
        if (tid == 0) atomicAdd(result, wval);
    }
}

std::int64_t gpuCount(const int* d_in, int n) {
    const int BLK = 256;
    int grid = (n + BLK * 2 - 1) / (BLK * 2);

    GpuBuffer<unsigned long long> d_res(1);
    d_res.memsetZero();

    gpu_count_kernel<<<grid, BLK, BLK * sizeof(unsigned long long)>>>(d_in, n, d_res.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long h;
    CUDA_CHECK(cudaMemcpy(&h, d_res.data(), sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    return static_cast<std::int64_t>(h);
}

} // namespace gpudb
