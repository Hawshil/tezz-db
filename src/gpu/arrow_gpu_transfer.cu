/**
 * @file arrow_gpu_transfer.cu
 * @brief Benchmark: explicit copy vs zero-copy vs UVM on a 1 GB Arrow column.
 */
#include "arrow_gpu_transfer.cuh"
#include "gpu_ops.cuh"
#include <chrono>
#include <cstdio>
#include <cstring>

namespace gpudb {

using Clock = std::chrono::high_resolution_clock;
static double msec(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

void benchArrowTransferStrategies(const double* h_data, std::size_t n) {
    double data_gb = n * sizeof(double) / 1e9;
    std::printf("\n═══ Arrow→GPU Transfer Strategies (%.1f GB) ═══\n", data_gb);

    // ── (a) Explicit cudaMemcpy ─────────────────────────────────────────────
    {
        auto t0 = Clock::now();
        GpuBuffer<double> d = arrowToDevice_Copy(h_data, n);
        auto t1 = Clock::now();
        double sum = gpuSum(d.data(), (int)n);
        auto t2 = Clock::now();

        double copy_ms = msec(t0, t1), kern_ms = msec(t1, t2);
        std::printf("  (a) Explicit copy:    copy=%6.1f ms  kernel=%6.1f ms  "
                    "total=%6.1f ms  BW=%.1f GB/s  sum=%.4f\n",
                    copy_ms, kern_ms, copy_ms + kern_ms,
                    data_gb / (copy_ms / 1000.0), sum);
    }

    // ── (b) Zero-copy (pinned host) ─────────────────────────────────────────
    {
        // Need non-const for cudaHostRegister
        auto* h_mut = const_cast<double*>(h_data);
        auto t0 = Clock::now();
        double* d_ptr = arrowToDevice_ZeroCopy(h_mut, n);
        auto t1 = Clock::now();
        double sum = gpuSum(d_ptr, (int)n);
        auto t2 = Clock::now();
        arrowUnregister(h_mut);

        double reg_ms = msec(t0, t1), kern_ms = msec(t1, t2);
        std::printf("  (b) Zero-copy:        reg=%7.1f ms  kernel=%6.1f ms  "
                    "total=%6.1f ms  BW=%.1f GB/s  sum=%.4f\n",
                    reg_ms, kern_ms, reg_ms + kern_ms,
                    data_gb / (kern_ms / 1000.0), sum);
    }

    // ── (c) UVM with prefetch ───────────────────────────────────────────────
    {
        auto t0 = Clock::now();
        double* uvm = arrowToDevice_UVM(h_data, n);
        auto t1 = Clock::now();
        double sum = gpuSum(uvm, (int)n);
        auto t2 = Clock::now();
        arrowFreeUVM(uvm);

        double pre_ms = msec(t0, t1), kern_ms = msec(t1, t2);
        std::printf("  (c) UVM + prefetch:   prep=%6.1f ms  kernel=%6.1f ms  "
                    "total=%6.1f ms  BW=%.1f GB/s  sum=%.4f\n",
                    pre_ms, kern_ms, pre_ms + kern_ms,
                    data_gb / ((pre_ms + kern_ms) / 1000.0), sum);
    }

    std::printf("\n  Summary:\n");
    std::printf("  • (a) wins for compute-heavy kernels (amortised copy cost)\n");
    std::printf("  • (b) wins for single-pass scans or data > VRAM\n");
    std::printf("  • (c) wins for unpredictable access or mixed CPU/GPU\n");
}

} // namespace gpudb
