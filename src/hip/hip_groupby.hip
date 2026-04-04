/**
 * @file hip_groupby.hip
 * @brief HIP port of gpu_groupby.cu — GROUP BY + SUM with GPU hash table.
 *
 * atomicCAS and atomicAdd work identically in HIP on GFX9+ (MI50 and later).
 * The shared-memory partial aggregation pattern is unchanged.
 */
#include "hip_utils.h"
#include "hip_ops.h"
#include <vector>

namespace gpudb {

#define SM_HT_SIZE 256

__device__ int hip_ht_insert(int* keys, int key, int cap) {
    unsigned slot = (unsigned)(key & 0x7FFFFFFF) % cap;
    while (true) {
        int old = atomicCAS(&keys[slot], -1, key);
        if (old == -1 || old == key) return (int)slot;
        slot = (slot + 1) % cap;
    }
}

__global__ void hip_groupby_sum_kernel(const int* keys, const double* vals, int n,
                                       int* ht_keys, double* ht_vals, int ht_cap) {
    __shared__ int   sm_keys[SM_HT_SIZE];
    __shared__ double sm_vals[SM_HT_SIZE];

    for (int i = threadIdx.x; i < SM_HT_SIZE; i += blockDim.x) {
        sm_keys[i] = -1; sm_vals[i] = 0.0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        int key = keys[i]; double val = vals[i];
        unsigned slot = (unsigned)(key & 0x7FFFFFFF) % SM_HT_SIZE;
        while (true) {
            int old = atomicCAS(&sm_keys[slot], -1, key);
            if (old == -1 || old == key) { atomicAdd(&sm_vals[slot], val); break; }
            slot = (slot + 1) % SM_HT_SIZE;
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < SM_HT_SIZE; i += blockDim.x) {
        if (sm_keys[i] != -1) {
            int slot = hip_ht_insert(ht_keys, sm_keys[i], ht_cap);
            atomicAdd(&ht_vals[slot], sm_vals[i]);
        }
    }
}

int hipGroupBySum(const int* d_keys, const double* d_vals, int n,
                  int* h_out_keys, double* h_out_sums, int ht_cap) {
    HipBuffer<int>    d_ht_keys(ht_cap);
    HipBuffer<double> d_ht_vals(ht_cap);
    d_ht_keys.fill(0xFF); d_ht_vals.memsetZero();

    int blk = 256, grd = min((n + blk - 1) / blk, 2048);
    hip_groupby_sum_kernel<<<grd, blk>>>(d_keys, d_vals, n,
                                          d_ht_keys.data(), d_ht_vals.data(), ht_cap);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<int> hk(ht_cap); std::vector<double> hv(ht_cap);
    HIP_CHECK(hipMemcpy(hk.data(), d_ht_keys.data(), ht_cap*4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hv.data(), d_ht_vals.data(), ht_cap*8, hipMemcpyDeviceToHost));
    int cnt = 0;
    for (int i = 0; i < ht_cap; ++i)
        if (hk[i] != -1) { h_out_keys[cnt] = hk[i]; h_out_sums[cnt] = hv[i]; ++cnt; }
    return cnt;
}

} // namespace gpudb
