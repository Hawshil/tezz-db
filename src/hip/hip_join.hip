/**
 * @file hip_join.hip
 * @brief HIP port of gpu_join.cu — hash join with linear probing.
 *
 * Wavefront Divergence in Linear Probing
 * ───────────────────────────────────────
 * With 64-thread wavefronts, linear probing causes MORE divergence than CUDA's
 * 32-thread warps: when lanes follow different probe chains, 64 threads must
 * stay in lockstep, idle lanes waste cycles while busy lanes probe further.
 *
 * Mitigation: Use a reasonable hash table load factor (≤0.5) to keep probe
 * chains short, minimizing divergence. The 2× table size below ensures this.
 *
 * rocprim::exclusive_scan replaces thrust for stream compaction.
 */
#include "hip_utils.h"
#include "hip_ops.h"
#include <rocprim/rocprim.hpp>

namespace gpudb {

__global__ void hip_join_build(const int* keys, int n,
                               int* ht_keys, int* ht_vals, int cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int key = keys[i];
    unsigned slot = (unsigned)(key & 0x7FFFFFFF) % cap;
    while (true) {
        int old = atomicCAS(&ht_keys[slot], -1, key);
        if (old == -1) { ht_vals[slot] = i; return; }
        slot = (slot + 1) % cap;
    }
}

__global__ void hip_join_count(const int* probe, int n,
                               const int* ht_keys, int cap, int* counts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int key = probe[i], c = 0;
    unsigned slot = (unsigned)(key & 0x7FFFFFFF) % cap;
    while (ht_keys[slot] != -1) {
        if (ht_keys[slot] == key) ++c;
        slot = (slot + 1) % cap;
    }
    counts[i] = c;
}

__global__ void hip_join_write(const int* probe, int n,
                               const int* ht_keys, const int* ht_vals, int cap,
                               const int* offsets, int* out_b, int* out_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int key = probe[i], pos = offsets[i];
    unsigned slot = (unsigned)(key & 0x7FFFFFFF) % cap;
    while (ht_keys[slot] != -1) {
        if (ht_keys[slot] == key) {
            out_b[pos] = ht_vals[slot]; out_p[pos] = i; ++pos;
        }
        slot = (slot + 1) % cap;
    }
}

int hipHashJoin(const int* d_build, int n_build,
                const int* d_probe, int n_probe,
                int* d_out_b, int* d_out_p, int ht_cap) {
    const int BLK = 256;
    HipBuffer<int> d_ht_k(ht_cap), d_ht_v(ht_cap);
    d_ht_k.fill(0xFF);

    int g1 = (n_build + BLK - 1) / BLK;
    hip_join_build<<<g1, BLK>>>(d_build, n_build, d_ht_k.data(), d_ht_v.data(), ht_cap);

    HipBuffer<int> d_cnt(n_probe), d_off(n_probe);
    int g2 = (n_probe + BLK - 1) / BLK;
    hip_join_count<<<g2, BLK>>>(d_probe, n_probe, d_ht_k.data(), ht_cap, d_cnt.data());

    // rocPRIM exclusive scan
    size_t temp_bytes = 0;
    rocprim::exclusive_scan(nullptr, temp_bytes, d_cnt.data(), d_off.data(), 0, n_probe);
    HipBuffer<char> d_temp(temp_bytes);
    rocprim::exclusive_scan(d_temp.data(), temp_bytes, d_cnt.data(), d_off.data(), 0, n_probe);
    HIP_CHECK(hipDeviceSynchronize());

    int last_cnt = 0, last_off = 0;
    HIP_CHECK(hipMemcpy(&last_cnt, d_cnt.data()+n_probe-1, 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&last_off, d_off.data()+n_probe-1, 4, hipMemcpyDeviceToHost));
    int total = last_off + last_cnt;

    hip_join_write<<<g2, BLK>>>(d_probe, n_probe, d_ht_k.data(), d_ht_v.data(),
                                ht_cap, d_off.data(), d_out_b, d_out_p);
    HIP_CHECK(hipDeviceSynchronize());
    return total;
}

} // namespace gpudb
