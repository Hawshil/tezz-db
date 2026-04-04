/**
 * @file gpu_join.cu
 * @brief GPU inner hash join with linear probing + Thrust prefix-sum compaction.
 *
 * Build:  each thread inserts (key, row_idx) into a hash table via atomicCAS.
 * Probe:  two-pass — count matches, prefix-sum for positions, write matches.
 */
#include "cuda_utils.cuh"
#include "gpu_ops.cuh"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace gpudb {

// ── Build kernel ────────────────────────────────────────────────────────────
__global__ void gpu_join_build(const int* keys, int n,
                               int* ht_keys, int* ht_vals, int cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int key = keys[i];
    unsigned slot = (unsigned)(key & 0x7FFFFFFF) % cap;
    while (true) {
        int old = atomicCAS(&ht_keys[slot], -1, key);
        if (old == -1) { ht_vals[slot] = i; return; }
        slot = (slot + 1) % cap;          // linear probing
    }
}

// ── Probe pass 1: count matches per probe row ──────────────────────────────
__global__ void gpu_join_count(const int* probe, int n,
                               const int* ht_keys, int cap, int* counts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int key = probe[i]; int c = 0;
    unsigned slot = (unsigned)(key & 0x7FFFFFFF) % cap;
    while (ht_keys[slot] != -1) {
        if (ht_keys[slot] == key) ++c;
        slot = (slot + 1) % cap;
    }
    counts[i] = c;
}

// ── Probe pass 2: write matched pairs at computed positions ─────────────────
__global__ void gpu_join_write(const int* probe, int n,
                               const int* ht_keys, const int* ht_vals, int cap,
                               const int* offsets, int* out_build, int* out_probe) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int key = probe[i], pos = offsets[i];
    unsigned slot = (unsigned)(key & 0x7FFFFFFF) % cap;
    while (ht_keys[slot] != -1) {
        if (ht_keys[slot] == key) {
            out_build[pos] = ht_vals[slot];
            out_probe[pos] = i;
            ++pos;
        }
        slot = (slot + 1) % cap;
    }
}

int gpuHashJoin(const int* d_build, int n_build,
                const int* d_probe, int n_probe,
                int* d_out_build, int* d_out_probe, int ht_cap) {
    const int BLK = 256;

    // 1. Build hash table
    GpuBuffer<int> d_ht_keys(ht_cap), d_ht_vals(ht_cap);
    d_ht_keys.fill(0xFF);  // -1

    int g1 = (n_build + BLK - 1) / BLK;
    gpu_join_build<<<g1, BLK>>>(d_build, n_build, d_ht_keys.data(), d_ht_vals.data(), ht_cap);

    // 2. Count matches per probe row
    GpuBuffer<int> d_counts(n_probe), d_offsets(n_probe);
    int g2 = (n_probe + BLK - 1) / BLK;
    gpu_join_count<<<g2, BLK>>>(d_probe, n_probe, d_ht_keys.data(), ht_cap, d_counts.data());

    // 3. Prefix sum → output positions
    thrust::device_ptr<int> cp(d_counts.data()), op(d_offsets.data());
    thrust::exclusive_scan(cp, cp + n_probe, op);

    // 4. Total matches
    int last_cnt = 0, last_off = 0;
    CUDA_CHECK(cudaMemcpy(&last_cnt, d_counts.data()+n_probe-1, 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_off, d_offsets.data()+n_probe-1, 4, cudaMemcpyDeviceToHost));
    int total = last_off + last_cnt;

    // 5. Write matched pairs
    gpu_join_write<<<g2, BLK>>>(d_probe, n_probe, d_ht_keys.data(), d_ht_vals.data(),
                                ht_cap, d_offsets.data(), d_out_build, d_out_probe);
    CUDA_CHECK(cudaDeviceSynchronize());
    return total;
}

} // namespace gpudb
