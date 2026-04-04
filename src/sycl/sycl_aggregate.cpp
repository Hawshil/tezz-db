/**
 * @file sycl_aggregate.cpp
 * @brief SYCL port of GPU aggregation — SUM and COUNT.
 *
 * Two implementations:
 * 1. Idiomatic SYCL: sycl::reduction() in parallel_for — the runtime handles
 *    all work-group and sub-group reductions automatically.
 * 2. Manual: explicit work-group reduction using sycl::local_accessor +
 *    sycl::group_barrier + sycl::shift_group_left for sub-group shuffles.
 *
 * Sub-group shuffle equivalent:
 *   CUDA:  __shfl_down_sync(0xFFFF, val, offset)
 *   HIP:   __shfl_down(val, offset, warpSize)
 *   SYCL:  sycl::shift_group_left(sub_group, val, offset)
 */
#include "sycl_utils.h"
#include "sycl_ops.h"

namespace gpudb {

// ── Idiomatic SYCL reduction ────────────────────────────────────────────────
double syclSum(sycl::queue& q, const double* d_in, int n) {
    double* d_result = sycl::malloc_shared<double>(1, q);
    *d_result = 0.0;

    q.submit([&](sycl::handler& h) {
        auto sum_reduction = sycl::reduction(d_result, sycl::plus<double>());
        h.parallel_for(sycl::range<1>(n), sum_reduction,
            [=](sycl::id<1> i, auto& sum) {
                sum += d_in[i];
            });
    }).wait();

    double result = *d_result;
    sycl::free(d_result, q);
    return result;
}

// ── Manual work-group reduction (for learning) ──────────────────────────────
double syclSumManual(sycl::queue& q, const double* d_in, int n) {
    constexpr int WG = 256;
    int global = ((n + WG*2 - 1) / (WG*2)) * WG;

    double* d_result = sycl::malloc_shared<double>(1, q);
    *d_result = 0.0;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> sdata(sycl::range<1>(WG), h);

        h.parallel_for(sycl::nd_range<1>(global, WG),
            [=](sycl::nd_item<1> item) {
                auto wg = item.get_group();
                auto sg = item.get_sub_group();
                int lid = item.get_local_id(0);
                int gid = item.get_global_id(0);
                int gid2 = gid + item.get_global_range(0);

                // Each work-item loads two elements
                double val = (gid < n) ? d_in[gid] : 0.0;
                if (gid2 < n) val += d_in[gid2];
                sdata[lid] = val;
                sycl::group_barrier(wg);

                // Tree reduction in local memory
                int sg_size = sg.get_local_linear_range();
                for (int s = WG / 2; s > sg_size; s >>= 1) {
                    if (lid < s) sdata[lid] += sdata[lid + s];
                    sycl::group_barrier(wg);
                }

                // Sub-group shuffle reduction (equivalent to __shfl_down)
                if (lid < sg_size) {
                    double wval = sdata[lid];
                    if (WG >= 2 * sg_size) wval += sdata[lid + sg_size];
                    for (int off = sg_size / 2; off > 0; off >>= 1)
                        wval = wval + sycl::shift_group_left(sg, wval, off);

                    if (lid == 0) {
                        sycl::atomic_ref<double,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                                atom(d_result[0]);
                        atom.fetch_add(wval);
                    }
                }
            });
    }).wait();

    double result = *d_result;
    sycl::free(d_result, q);
    return result;
}

// ── COUNT ───────────────────────────────────────────────────────────────────
std::int64_t syclCount(sycl::queue& q, const int* d_in, int n) {
    std::int64_t* d_result = sycl::malloc_shared<std::int64_t>(1, q);
    *d_result = 0;

    q.submit([&](sycl::handler& h) {
        auto cnt_reduction = sycl::reduction(d_result, sycl::plus<std::int64_t>());
        h.parallel_for(sycl::range<1>(n), cnt_reduction,
            [=](sycl::id<1> i, auto& cnt) {
                cnt += (std::int64_t)d_in[i];
            });
    }).wait();

    std::int64_t result = *d_result;
    sycl::free(d_result, q);
    return result;
}

} // namespace gpudb
