/**
 * @file gpu_groupby.cu
 * @brief GPU GROUP BY + SUM using a lock-free open-addressing hash table.
 *
 * Uses atomicCAS for key insertion and atomicAdd for value accumulation.
 * Shared-memory block-level partial aggregation reduces global atomic contention.
 */
#include "cuda_utils.cuh"
#include "gpu_ops.cuh"

namespace gpudb {

#define SM_HT_SIZE 256  // shared-memory hash table slots per block

__device__ int ht_insert(int* keys, int key, int cap) {
    unsigned slot = (unsigned)(key & 0x7FFFFFFF) % cap;
    while (true) {
        int old = atomicCAS(&keys[slot], -1, key);
        if (old == -1 || old == key) return (int)slot;
        slot = (slot + 1) % cap;
    }
}

// ── Kernel with shared-memory partial aggregation ───────────────────────────
__global__ void gpu_groupby_sum_kernel(const int* keys, const double* vals, int n,
                                       int* ht_keys, double* ht_vals, int ht_cap) {
    __shared__ int   sm_keys[SM_HT_SIZE];
    __shared__ double sm_vals[SM_HT_SIZE];

    // Init shared HT to empty
    for (int i = threadIdx.x; i < SM_HT_SIZE; i += blockDim.x) {
        sm_keys[i] = -1; sm_vals[i] = 0.0;
    }
    __syncthreads();

    // Phase 1: accumulate into shared-memory HT
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        int key = keys[i]; double val = vals[i];
        unsigned slot = (unsigned)(key & 0x7FFFFFFF) % SM_HT_SIZE;
        while (true) {
            int old = atomicCAS(&sm_keys[slot], -1, key);
            if (old == -1 || old == key) {
                atomicAdd(&sm_vals[slot], val);
                break;
            }
            slot = (slot + 1) % SM_HT_SIZE;
        }
    }
    __syncthreads();

    // Phase 2: flush shared HT → global HT
    for (int i = threadIdx.x; i < SM_HT_SIZE; i += blockDim.x) {
        if (sm_keys[i] != -1) {
            int slot = ht_insert(ht_keys, sm_keys[i], ht_cap);
            atomicAdd(&ht_vals[slot], sm_vals[i]);
        }
    }
}

int gpuGroupBySum(const int* d_keys, const double* d_vals, int n,
                  int* h_out_keys, double* h_out_sums, int ht_cap) {
    GpuBuffer<int>    d_ht_keys(ht_cap);
    GpuBuffer<double> d_ht_vals(ht_cap);
    d_ht_keys.fill(0xFF);   // -1 for empty
    d_ht_vals.memsetZero();

    int block = 256, grid = min((n + block - 1) / block, 2048);
    gpu_groupby_sum_kernel<<<grid, block>>>(d_keys, d_vals, n,
                                            d_ht_keys.data(), d_ht_vals.data(), ht_cap);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy HT back to host and extract non-empty slots
    std::vector<int>    h_keys(ht_cap);
    std::vector<double> h_vals(ht_cap);
    CUDA_CHECK(cudaMemcpy(h_keys.data(), d_ht_keys.data(), ht_cap*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vals.data(), d_ht_vals.data(), ht_cap*8, cudaMemcpyDeviceToHost));

    int cnt = 0;
    for (int i = 0; i < ht_cap; ++i) {
        if (h_keys[i] != -1) {
            h_out_keys[cnt] = h_keys[i];
            h_out_sums[cnt] = h_vals[i];
            ++cnt;
        }
    }
    return cnt;
}

} // namespace gpudb
