/**
 * @file sycl_usm_compare.cpp
 * @brief SYCL USM memory model comparison: device vs host vs shared.
 *
 * Trade-off:
 *   • malloc_device + explicit q.memcpy(): predictable, highest bandwidth
 *   • malloc_shared (USM): simpler code, but implicit page faults
 *   • buffer/accessor: runtime manages everything, best for productivity
 *
 * For GPU databases, malloc_device wins because we need predictable latency.
 * malloc_shared is useful for prototyping and mixed CPU/GPU algorithms.
 */
#include "sycl/sycl_utils.h"
#include "sycl/sycl_ops.h"
#include <chrono>
#include <cstdio>
#include <vector>

namespace gpudb {

using Clock = std::chrono::high_resolution_clock;

void benchSyclUsmComparison(int n) {
    sycl::queue q = makeGpuQueue();
    std::printf("\n═══ SYCL USM Comparison — %dM elements ═══\n", n / 1000000);

    std::srand(42);
    std::vector<double> h(n);
    for (int i = 0; i < n; ++i) h[i] = (double)std::rand() / RAND_MAX;

    // ── (a) malloc_device + explicit memcpy ─────────────────────────────────
    {
        auto* d = sycl::malloc_device<double>(n, q);
        auto t0 = Clock::now();
        q.memcpy(d, h.data(), n * sizeof(double)).wait();
        double sum = syclSum(q, d, n);
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  (a) malloc_device + memcpy: %7.1f ms  sum=%.4f\n", ms, sum);
        sycl::free(d, q);
    }

    // ── (b) malloc_shared (USM) ─────────────────────────────────────────────
    {
        auto* s = sycl::malloc_shared<double>(n, q);
        std::memcpy(s, h.data(), n * sizeof(double));
        auto t0 = Clock::now();
        q.prefetch(s, n * sizeof(double)).wait();
        double sum = syclSum(q, s, n);
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  (b) malloc_shared + prefetch: %5.1f ms  sum=%.4f\n", ms, sum);
        sycl::free(s, q);
    }

    // ── (c) malloc_shared, no prefetch ──────────────────────────────────────
    {
        auto* s = sycl::malloc_shared<double>(n, q);
        std::memcpy(s, h.data(), n * sizeof(double));
        auto t0 = Clock::now();
        // No prefetch — on-demand migration
        double sum = syclSum(q, s, n);
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  (c) malloc_shared, no pref:   %5.1f ms  sum=%.4f\n", ms, sum);
        sycl::free(s, q);
    }

    // ── (d) buffer/accessor model ───────────────────────────────────────────
    {
        auto t0 = Clock::now();
        double result = 0;
        {
            sycl::buffer<double, 1> buf(h.data(), sycl::range<1>(n));
            sycl::buffer<double, 1> res_buf(&result, sycl::range<1>(1));
            q.submit([&](sycl::handler& cgh) {
                auto acc = buf.get_access<sycl::access::mode::read>(cgh);
                auto red = sycl::reduction(res_buf, cgh, sycl::plus<double>());
                cgh.parallel_for(sycl::range<1>(n), red,
                    [=](sycl::id<1> i, auto& sum) { sum += acc[i]; });
            }).wait();
        }
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  (d) buffer/accessor:          %5.1f ms  sum=%.4f\n", ms, result);
    }

    std::printf("\n  Recommendation: use malloc_device for production,\n");
    std::printf("  malloc_shared for prototyping, buffer/accessor for portability.\n");
}

} // namespace gpudb
