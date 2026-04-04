/**
 * @file hip_graph.hip
 * @brief HIP Graph capture for repeated query execution (AMD port of gpu_graph.cu).
 *
 * Same benefit as CUDA Graphs: eliminates per-launch CPU overhead for repeated
 * queries. HIP API maps 1:1:
 *   cudaStreamBeginCapture → hipStreamBeginCapture
 *   cudaGraphInstantiate   → hipGraphInstantiate
 *   cudaGraphLaunch        → hipGraphLaunch
 */
#include "hip_utils.h"
#include "hip_ops.h"
#include <chrono>
#include <cstdio>
#include <vector>

namespace gpudb {

using Clock = std::chrono::high_resolution_clock;

extern __global__ void hip_groupby_sum_kernel(
    const int*, const double*, int, int*, double*, int);

void benchHipGraphExecution(const int* h_keys, const double* h_vals, int n,
                            int num_groups, int num_queries) {
    int ht_cap = num_groups * 4;
    HipBuffer<int>    d_keys(n);
    HipBuffer<double> d_vals(n);
    HipBuffer<int>    d_ht_keys(ht_cap);
    HipBuffer<double> d_ht_vals(ht_cap);

    HIP_CHECK(hipMemcpy(d_keys.data(), h_keys, n * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vals.data(), h_vals, n * sizeof(double), hipMemcpyHostToDevice));

    // ── Baseline ────────────────────────────────────────────────────────────
    {
        auto t0 = Clock::now();
        int blk = 256, grd = std::min((n + blk - 1) / blk, 2048);
        for (int q = 0; q < num_queries; ++q) {
            d_ht_keys.fill(0xFF); d_ht_vals.memsetZero();
            hip_groupby_sum_kernel<<<grd, blk>>>(
                d_keys.data(), d_vals.data(), n,
                d_ht_keys.data(), d_ht_vals.data(), ht_cap);
            HIP_CHECK(hipDeviceSynchronize());
        }
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  Individual launches: %8.1f ms for %d queries\n", ms, num_queries);
    }

    // ── HIP Graph ───────────────────────────────────────────────────────────
    {
        hipStream_t stream;
        HIP_CHECK(hipStreamCreate(&stream));

        HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

        int blk = 256, grd = std::min((n + blk - 1) / blk, 2048);
        hip_groupby_sum_kernel<<<grd, blk, 0, stream>>>(
            d_keys.data(), d_vals.data(), n,
            d_ht_keys.data(), d_ht_vals.data(), ht_cap);

        hipGraph_t graph;
        HIP_CHECK(hipStreamEndCapture(stream, &graph));

        hipGraphExec_t graphExec;
        HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

        auto t0 = Clock::now();
        for (int q = 0; q < num_queries; ++q) {
            HIP_CHECK(hipGraphLaunch(graphExec, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
        }
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  HIP Graph launch:   %8.1f ms for %d queries\n", ms, num_queries);

        HIP_CHECK(hipGraphExecDestroy(graphExec));
        HIP_CHECK(hipGraphDestroy(graph));
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

} // namespace gpudb
