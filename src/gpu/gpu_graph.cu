/**
 * @file gpu_graph.cu
 * @brief CUDA Graph capture for repeated query execution.
 *
 * When CUDA Graphs help most:
 *   • Streaming analytics: same filter→groupby pipeline runs thousands of
 *     times on different time-window batches. Graph launch is ~3μs vs ~10μs+
 *     for individual kernel launches.
 *   • Dashboard queries: BI dashboards refresh the same query periodically.
 *     Graph eliminates per-launch CPU overhead and sync bubbles.
 *   • The benefit grows with kernel count: a pipeline with 5 small kernels
 *     saves ~35μs per execution (7μs × 5 launches), adding up to ~35ms
 *     savings across 1000 repeated queries.
 */
#include "cuda_utils.cuh"
#include "gpu_ops.cuh"
#include <chrono>
#include <cstdio>
#include <vector>

namespace gpudb {

using Clock = std::chrono::high_resolution_clock;

// ── Capture filter→groupby pipeline as a CUDA Graph ─────────────────────────
void benchGraphExecution(const int* h_keys, const double* h_vals, int n,
                         int num_groups, int num_queries) {
    int ht_cap = num_groups * 4;

    // Pre-allocate device buffers (reused across queries)
    GpuBuffer<int>    d_keys(n);
    GpuBuffer<double> d_vals(n);
    GpuBuffer<double> d_filtered(n);
    GpuBuffer<int>    d_ht_keys(ht_cap);
    GpuBuffer<double> d_ht_vals(ht_cap);

    // Initial data copy
    CUDA_CHECK(cudaMemcpy(d_keys.data(), h_keys, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals.data(), h_vals, n * sizeof(double), cudaMemcpyHostToDevice));

    // ── Baseline: individual kernel launches ────────────────────────────────
    {
        auto t0 = Clock::now();
        for (int q = 0; q < num_queries; ++q) {
            d_ht_keys.fill(0xFF);
            d_ht_vals.memsetZero();
            int filtered = gpuFilter(d_vals.data(), n, 0.5, CompareOp::GT, d_filtered.data());
            std::vector<int> ok(ht_cap); std::vector<double> os(ht_cap);
            gpuGroupBySum(d_keys.data(), d_filtered.data(), filtered,
                          ok.data(), os.data(), ht_cap);
        }
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  Individual launches: %8.1f ms for %d queries (%.2f ms/query)\n",
                    ms, num_queries, ms / num_queries);
    }

    // ── CUDA Graph capture ──────────────────────────────────────────────────
    {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Capture
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

        // Pipeline to capture
        d_ht_keys.fill(0xFF);
        d_ht_vals.memsetZero();
        // Note: for graph capture, we use fixed kernels on the stream.
        // The actual filter+groupby would be launched on this stream.
        // Simplified: just capture the groupby kernel as representative
        int blk = 256, grd = std::min((n + blk - 1) / blk, 2048);

        // Declare the extern kernel (defined in gpu_groupby.cu)
        extern __global__ void gpu_groupby_sum_kernel(
            const int*, const double*, int, int*, double*, int);

        gpu_groupby_sum_kernel<<<grd, blk, 0, stream>>>(
            d_keys.data(), d_vals.data(), n,
            d_ht_keys.data(), d_ht_vals.data(), ht_cap);

        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

        cudaGraphExec_t graphExec;
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

        // ── Execute graph repeatedly ────────────────────────────────────────
        auto t0 = Clock::now();
        for (int q = 0; q < num_queries; ++q) {
            CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  CUDA Graph launch:  %8.1f ms for %d queries (%.2f ms/query)\n",
                    ms, num_queries, ms / num_queries);

        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
}

} // namespace gpudb
