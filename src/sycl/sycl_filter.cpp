/**
 * @file sycl_filter.cpp
 * @brief SYCL port of GPU filter — parallel predicate + oneDPL scan + compact.
 *
 * Uses oneDPL (oneAPI Data Parallel Library) for exclusive_scan,
 * which is the SYCL equivalent of Thrust/rocPRIM.
 */
#include "sycl_utils.h"
#include "sycl_ops.h"
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>

namespace gpudb {

int syclFilter(sycl::queue& q, const double* d_in, int n,
               double threshold, SyclCompareOp op, double* d_out) {
    auto* d_mask = sycl::malloc_device<int>(n, q);
    auto* d_pos  = sycl::malloc_device<int>(n, q);

    // 1. Mark matching elements
    int iop = (int)op;
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
        int i = idx[0]; double v = d_in[i]; int pass = 0;
        switch (iop) {
            case 0: pass = (v >  threshold); break;
            case 1: pass = (v >= threshold); break;
            case 2: pass = (v <  threshold); break;
            case 3: pass = (v <= threshold); break;
            case 4: pass = (v == threshold); break;
            case 5: pass = (v != threshold); break;
        }
        d_mask[i] = pass;
    }).wait();

    // 2. oneDPL exclusive scan for output positions
    auto policy = oneapi::dpl::execution::make_device_policy(q);
    std::exclusive_scan(policy, d_mask, d_mask + n, d_pos, 0);

    // 3. Get total count
    int last_mask = 0, last_pos = 0;
    q.memcpy(&last_mask, d_mask + n - 1, sizeof(int)).wait();
    q.memcpy(&last_pos,  d_pos  + n - 1, sizeof(int)).wait();
    int total = last_pos + last_mask;

    // 4. Compact
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
        int i = idx[0];
        if (d_mask[i]) d_out[d_pos[i]] = d_in[i];
    }).wait();

    sycl::free(d_mask, q); sycl::free(d_pos, q);
    return total;
}

} // namespace gpudb
