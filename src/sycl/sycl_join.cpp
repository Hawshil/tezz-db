/**
 * @file sycl_join.cpp
 * @brief SYCL port of GPU hash join — build/count/write with atomic_ref CAS.
 *
 * Uses oneDPL exclusive_scan for output compaction, and sycl::queue with
 * in_order=false (out-of-order) for overlapping transfer + compute.
 */
#include "sycl_utils.h"
#include "sycl_ops.h"
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

namespace gpudb {

int syclHashJoin(sycl::queue& q, const int* d_build, int n_build,
                 const int* d_probe, int n_probe,
                 int* d_match_build, int* d_match_probe, int ht_cap) {
    auto* d_ht_keys = sycl::malloc_device<int>(ht_cap, q);
    auto* d_ht_vals = sycl::malloc_device<int>(ht_cap, q);
    q.memset(d_ht_keys, 0xFF, ht_cap * sizeof(int)).wait(); // -1

    // 1. Build — insert (key, row_idx) with linear probing
    q.parallel_for(sycl::range<1>(n_build), [=](sycl::id<1> idx) {
        int i = idx[0], key = d_build[i];
        unsigned slot = (unsigned)(key & 0x7FFFFFFF) % ht_cap;
        while (true) {
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space> atom(d_ht_keys[slot]);
            int expected = -1;
            if (atom.compare_exchange_strong(expected, key)) {
                d_ht_vals[slot] = i; return;
            }
            slot = (slot + 1) % ht_cap;
        }
    }).wait();

    // 2. Count matches per probe row
    auto* d_cnt = sycl::malloc_device<int>(n_probe, q);
    auto* d_off = sycl::malloc_device<int>(n_probe, q);

    q.parallel_for(sycl::range<1>(n_probe), [=](sycl::id<1> idx) {
        int i = idx[0], key = d_probe[i], c = 0;
        unsigned slot = (unsigned)(key & 0x7FFFFFFF) % ht_cap;
        while (d_ht_keys[slot] != -1) {
            if (d_ht_keys[slot] == key) ++c;
            slot = (slot + 1) % ht_cap;
        }
        d_cnt[i] = c;
    }).wait();

    // 3. oneDPL exclusive scan
    auto policy = oneapi::dpl::execution::make_device_policy(q);
    std::exclusive_scan(policy, d_cnt, d_cnt + n_probe, d_off, 0);

    // 4. Total matches
    int last_cnt = 0, last_off = 0;
    q.memcpy(&last_cnt, d_cnt + n_probe - 1, sizeof(int)).wait();
    q.memcpy(&last_off, d_off + n_probe - 1, sizeof(int)).wait();
    int total = last_off + last_cnt;

    // 5. Write matches
    q.parallel_for(sycl::range<1>(n_probe), [=](sycl::id<1> idx) {
        int i = idx[0], key = d_probe[i], pos = d_off[i];
        unsigned slot = (unsigned)(key & 0x7FFFFFFF) % ht_cap;
        while (d_ht_keys[slot] != -1) {
            if (d_ht_keys[slot] == key) {
                d_match_build[pos] = d_ht_vals[slot];
                d_match_probe[pos] = i;
                ++pos;
            }
            slot = (slot + 1) % ht_cap;
        }
    }).wait();

    sycl::free(d_ht_keys, q); sycl::free(d_ht_vals, q);
    sycl::free(d_cnt, q); sycl::free(d_off, q);
    return total;
}

} // namespace gpudb
