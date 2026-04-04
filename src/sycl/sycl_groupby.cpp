/**
 * @file sycl_groupby.cpp
 * @brief SYCL port of GPU GROUP BY + SUM using sycl::atomic_ref.
 *
 * CUDA atomicCAS(addr, expected, desired) maps to:
 *   sycl::atomic_ref<int, relaxed, device, global_space> atom(*addr);
 *   atom.compare_exchange_strong(expected, desired);
 *
 * CUDA atomicAdd(addr, val) maps to:
 *   sycl::atomic_ref<double, relaxed, device, global_space> atom(*addr);
 *   atom.fetch_add(val);
 */
#include "sycl_utils.h"
#include "sycl_ops.h"
#include <vector>

namespace gpudb {

int syclGroupBySum(sycl::queue& q, const int* d_keys, const double* d_vals, int n,
                   int* h_out_keys, double* h_out_sums, int ht_cap) {
    auto* d_ht_keys = sycl::malloc_device<int>(ht_cap, q);
    auto* d_ht_vals = sycl::malloc_device<double>(ht_cap, q);
    q.memset(d_ht_keys, 0xFF, ht_cap * sizeof(int)).wait();   // -1
    q.memset(d_ht_vals, 0,    ht_cap * sizeof(double)).wait();

    constexpr int WG = 256;
    int global = ((n + WG - 1) / WG) * WG;

    q.parallel_for(sycl::nd_range<1>(global, WG),
        [=](sycl::nd_item<1> item) {
            int i = item.get_global_id(0);
            if (i >= n) return;

            int key = d_keys[i];
            double val = d_vals[i];
            unsigned slot = (unsigned)(key & 0x7FFFFFFF) % ht_cap;

            // Open-addressing insert with atomic CAS
            while (true) {
                sycl::atomic_ref<int,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>
                        atom_key(d_ht_keys[slot]);

                int expected = -1;
                if (atom_key.compare_exchange_strong(expected, key) ||
                    expected == key) {
                    // Slot acquired — accumulate value
                    sycl::atomic_ref<double,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                            atom_val(d_ht_vals[slot]);
                    atom_val.fetch_add(val);
                    break;
                }
                slot = (slot + 1) % ht_cap;
            }
        }).wait();

    // Copy HT back and extract
    std::vector<int>    hk(ht_cap);
    std::vector<double> hv(ht_cap);
    q.memcpy(hk.data(), d_ht_keys, ht_cap * sizeof(int)).wait();
    q.memcpy(hv.data(), d_ht_vals, ht_cap * sizeof(double)).wait();

    int cnt = 0;
    for (int i = 0; i < ht_cap; ++i)
        if (hk[i] != -1) { h_out_keys[cnt] = hk[i]; h_out_sums[cnt] = hv[i]; ++cnt; }

    sycl::free(d_ht_keys, q); sycl::free(d_ht_vals, q);
    return cnt;
}

} // namespace gpudb
