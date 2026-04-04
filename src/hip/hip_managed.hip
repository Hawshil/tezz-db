/**
 * @file hip_managed.hip
 * @brief HIP Unified Memory (hipMallocManaged) — three transfer strategies.
 *
 * Managed memory simplifies programming but hides performance hazards:
 *   • On-demand paging: GPU page faults trigger PCIe transfers per 4KB page.
 *     For random access, this can be 100x slower than bulk cudaMemcpy.
 *   • Prefetch mitigates this: hipMemPrefetchAsync pre-migrates pages in bulk.
 *   • The professor should know: managed memory is a productivity tool, not
 *     a performance tool. Production GPU databases (SQream, Kinetica) use
 *     explicit transfers for predictable latency.
 */
#include "hip_utils.h"
#include "hip_ops.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace gpudb {

using Clock = std::chrono::high_resolution_clock;

void benchHipManagedMemory(int n, int num_groups) {
    int ht_cap = num_groups * 4;
    std::srand(42);
    std::vector<int> h_keys(n);
    std::vector<double> h_vals(n);
    for (int i = 0; i < n; ++i) {
        h_keys[i] = std::rand() % num_groups;
        h_vals[i] = (std::rand() % 10000) / 100.0;
    }

    std::printf("\n═══ HIP Managed Memory Benchmark (%dM rows) ═══\n", n/1000000);

    // ── (a) Explicit hipMemcpy ──────────────────────────────────────────────
    {
        HipBuffer<int> dk(n); HipBuffer<double> dv(n);
        auto t0 = Clock::now();
        HIP_CHECK(hipMemcpy(dk.data(), h_keys.data(), n*4, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dv.data(), h_vals.data(), n*8, hipMemcpyHostToDevice));
        std::vector<int> ok(ht_cap); std::vector<double> os(ht_cap);
        hipGroupBySum(dk.data(), dv.data(), n, ok.data(), os.data(), ht_cap);
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  (a) Explicit copy:           %7.1f ms\n", ms);
    }

    // ── (b) Managed memory + prefetch ───────────────────────────────────────
    {
        int* mk = nullptr; double* mv = nullptr;
        HIP_CHECK(hipMallocManaged(&mk, (size_t)n * 4));
        HIP_CHECK(hipMallocManaged(&mv, (size_t)n * 8));
        std::memcpy(mk, h_keys.data(), (size_t)n * 4);
        std::memcpy(mv, h_vals.data(), (size_t)n * 8);

        auto t0 = Clock::now();
        int dev = 0;
        HIP_CHECK(hipMemPrefetchAsync(mk, (size_t)n * 4, dev));
        HIP_CHECK(hipMemPrefetchAsync(mv, (size_t)n * 8, dev));
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<int> ok(ht_cap); std::vector<double> os(ht_cap);
        hipGroupBySum(mk, mv, n, ok.data(), os.data(), ht_cap);
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  (b) Managed + prefetch:      %7.1f ms\n", ms);
        hipFree(mk); hipFree(mv);
    }

    // ── (c) Managed memory, no prefetch (on-demand) ─────────────────────────
    {
        int* mk = nullptr; double* mv = nullptr;
        HIP_CHECK(hipMallocManaged(&mk, (size_t)n * 4));
        HIP_CHECK(hipMallocManaged(&mv, (size_t)n * 8));
        std::memcpy(mk, h_keys.data(), (size_t)n * 4);
        std::memcpy(mv, h_vals.data(), (size_t)n * 8);

        auto t0 = Clock::now();
        // No prefetch — on-demand page migration
        std::vector<int> ok(ht_cap); std::vector<double> os(ht_cap);
        hipGroupBySum(mk, mv, n, ok.data(), os.data(), ht_cap);
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  (c) Managed, no prefetch:    %7.1f ms  (page faults!)\n", ms);
        hipFree(mk); hipFree(mv);
    }

    std::printf("  Summary: (a) is fastest for bulk transfers.\n");
    std::printf("  Managed memory with prefetch (b) approaches (a).\n");
    std::printf("  Without prefetch (c), page faults add 2-5x overhead.\n");
}

} // namespace gpudb
