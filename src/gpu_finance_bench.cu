/**
 * @file gpu_finance_bench.cu
 * @brief CPU vs GPU benchmark for finance operations:
 *   - Window functions: SMA, EMA, RollingStd
 *   - ASOF join on sorted timestamps
 *
 * Generates synthetic tick data and reports timing, speedup, and CSV output.
 */
#include "gpu/cuda_utils.cuh"
#include "gpu/gpu_ops.cuh"
#include "gpu/gpu_asof.cuh"
#include "core/table.h"
#include "core/column.h"
#include "query/operator_node.h"
#include "benchmark/benchmark_runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference implementations
// ─────────────────────────────────────────────────────────────────────────────

static void cpuSMA(const double* in, int n, int w, double* out) {
    double running = 0.0;
    for (int i = 0; i < n; ++i) {
        running += in[i];
        if (i >= w) running -= in[i - w];
        int cnt = (i < w) ? (i + 1) : w;
        out[i] = running / cnt;
    }
}

static void cpuEMA(const double* in, int n, int w, double* out) {
    if (n == 0) return;
    double alpha = 2.0 / (w + 1);
    out[0] = in[0];
    for (int i = 1; i < n; ++i)
        out[i] = alpha * in[i] + (1.0 - alpha) * out[i - 1];
}

static void cpuRollingStd(const double* in, int n, int w, double* out) {
    double sumX = 0.0, sumX2 = 0.0;
    for (int i = 0; i < n; ++i) {
        sumX  += in[i];
        sumX2 += in[i] * in[i];
        if (i >= w) {
            sumX  -= in[i - w];
            sumX2 -= in[i - w] * in[i - w];
        }
        int cnt = (i < w) ? (i + 1) : w;
        if (cnt < 2) {
            out[i] = 0.0;
        } else {
            double mean = sumX / cnt;
            double var  = sumX2 / cnt - mean * mean;
            out[i] = (var > 0.0) ? std::sqrt(var) : 0.0;
        }
    }
}

static void cpuAsofJoin(const std::int64_t* left_ts, int n_left,
                        const std::int64_t* right_ts, int n_right,
                        int* out_idx) {
    for (int i = 0; i < n_left; ++i) {
        std::int64_t lt = left_ts[i];
        int lo = 0, hi = n_right - 1, best = -1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (right_ts[mid] <= lt) {
                best = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        out_idx[i] = best;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CSV result collector
// ─────────────────────────────────────────────────────────────────────────────

struct BenchRow {
    std::string operation;
    int n;
    int param;  // window size or n_right
    double cpu_ms;
    double gpu_ms;
    double speedup;
};

static std::vector<BenchRow> g_results;

static void recordResult(const char* op, int n, int param,
                          double cpu, double gpu) {
    double speedup = (gpu > 0.0) ? cpu / gpu : 0.0;
    g_results.push_back({op, n, param, cpu, gpu, speedup});
    std::printf("  %-14s  n=%10d  param=%4d  CPU: %8.1f ms  "
                "GPU: %8.1f ms  Speedup: %6.1fx\n",
                op, n, param, cpu, gpu, speedup);
}

// ─────────────────────────────────────────────────────────────────────────────
// Generate synthetic data
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<double> genPrices(int n) {
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i)
        v[i] = 100.0 + std::sin(i / 100.0) * 10.0;
    return v;
}

static std::vector<std::int64_t> genTimestamps(int n, int step = 1) {
    std::vector<std::int64_t> v(n);
    for (int i = 0; i < n; ++i)
        v[i] = static_cast<std::int64_t>(i) * step;
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Window function benchmarks
// ─────────────────────────────────────────────────────────────────────────────

static void benchWindow(const char* name, int n, int w,
    void (*cpuFn)(const double*, int, int, double*),
    void (*gpuFn)(const double*, int, int, double*))
{
    auto h_in = genPrices(n);
    std::vector<double> h_cpu_out(n);

    gpudb::GpuBuffer<double> d_in(n), d_out(n);
    CUDA_CHECK(cudaMemcpy(d_in.data(), h_in.data(), n * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Warm-up
    for (int i = 0; i < 3; ++i) gpuFn(d_in.data(), n, w, d_out.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure GPU (5 runs, median)
    std::vector<double> gpu_times;
    for (int i = 0; i < 5; ++i) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = Clock::now();
        gpuFn(d_in.data(), n, w, d_out.data());
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = Clock::now();
        gpu_times.push_back(ms(t0, t1));
    }
    std::sort(gpu_times.begin(), gpu_times.end());
    double gpu_median = gpu_times[2];

    // Measure CPU (1 run)
    auto tc0 = Clock::now();
    cpuFn(h_in.data(), n, w, h_cpu_out.data());
    auto tc1 = Clock::now();
    double cpu_ms_val = ms(tc0, tc1);

    // Verify first 5 values match
    std::vector<double> h_gpu_out(n);
    CUDA_CHECK(cudaMemcpy(h_gpu_out.data(), d_out.data(),
                          n * sizeof(double), cudaMemcpyDeviceToHost));
    std::printf("    First 5 values (CPU|GPU): ");
    for (int i = 0; i < 5 && i < n; ++i)
        std::printf("%.3f|%.3f  ", h_cpu_out[i], h_gpu_out[i]);
    std::printf("\n");

    recordResult(name, n, w, cpu_ms_val, gpu_median);
}

// ─────────────────────────────────────────────────────────────────────────────
// ASOF join benchmark
// ─────────────────────────────────────────────────────────────────────────────

static void benchAsofJoin(int n_left, int n_right) {
    auto h_left_ts  = genTimestamps(n_left, 1);
    auto h_right_ts = genTimestamps(n_right, n_left / n_right);

    std::vector<int> h_cpu_out(n_left);
    std::vector<int> h_gpu_out(n_left);

    // Measure CPU
    auto tc0 = Clock::now();
    cpuAsofJoin(h_left_ts.data(), n_left,
                h_right_ts.data(), n_right,
                h_cpu_out.data());
    auto tc1 = Clock::now();
    double cpu_ms_val = ms(tc0, tc1);

    // Count CPU matches
    int cpu_matches = 0;
    for (int i = 0; i < n_left; ++i)
        if (h_cpu_out[i] >= 0) ++cpu_matches;

    // GPU
    gpudb::GpuBuffer<std::int64_t> d_left_ts(n_left), d_right_ts(n_right);
    gpudb::GpuBuffer<std::int32_t> d_left_key(n_left), d_right_key(n_right);
    gpudb::GpuBuffer<int> d_out(n_left);

    CUDA_CHECK(cudaMemcpy(d_left_ts.data(), h_left_ts.data(),
                          n_left * sizeof(std::int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_right_ts.data(), h_right_ts.data(),
                          n_right * sizeof(std::int64_t), cudaMemcpyHostToDevice));
    // No keys → zero-initialize
    CUDA_CHECK(cudaMemset(d_left_key.data(), 0, n_left * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMemset(d_right_key.data(), 0, n_right * sizeof(std::int32_t)));

    // Warm-up
    for (int i = 0; i < 3; ++i)
        gpudb::gpuAsofJoin(d_left_ts.data(), n_left,
                           d_right_ts.data(), n_right,
                           d_left_key.data(), d_right_key.data(),
                           false, 0, d_out.data());

    // Measure GPU (5 runs, median)
    std::vector<double> gpu_times;
    for (int i = 0; i < 5; ++i) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = Clock::now();
        gpudb::gpuAsofJoin(d_left_ts.data(), n_left,
                           d_right_ts.data(), n_right,
                           d_left_key.data(), d_right_key.data(),
                           false, 0, d_out.data());
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = Clock::now();
        gpu_times.push_back(ms(t0, t1));
    }
    std::sort(gpu_times.begin(), gpu_times.end());
    double gpu_median = gpu_times[2];

    // Copy back and count GPU matches
    CUDA_CHECK(cudaMemcpy(h_gpu_out.data(), d_out.data(),
                          n_left * sizeof(int), cudaMemcpyDeviceToHost));
    int gpu_matches = 0;
    for (int i = 0; i < n_left; ++i)
        if (h_gpu_out[i] >= 0) ++gpu_matches;

    std::printf("    CPU matches: %d   GPU matches: %d   %s\n",
                cpu_matches, gpu_matches,
                (cpu_matches == gpu_matches) ? "MATCH ✓" : "MISMATCH ✗");

    recordResult("ASOF_JOIN", n_left, n_right, cpu_ms_val, gpu_median);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::printf("\n  ╔═══════════════════════════════════════════════════╗\n");
    std::printf("  ║  GPUDB — Finance Operations Benchmark            ║\n");
    std::printf("  ╚═══════════════════════════════════════════════════╝\n");

    gpudb::GpuContext ctx(0);
    ctx.printDeviceInfo();

    const int N = 10'000'000;

    // ── Window function benchmarks ──
    int window_pairs[][2] = {{200, 50}, {50, 20}};
    for (auto& wp : window_pairs) {
        int w_sma = wp[0], w_ema = wp[1];
        std::printf("\n═══ Window Functions — n=%dM ═══\n", N / 1'000'000);

        benchWindow("SMA", N, w_sma, cpuSMA, gpudb::gpuSMA);
        benchWindow("EMA", N, w_ema, cpuEMA, gpudb::gpuEMA);
        benchWindow("ROLLING_STD", N, w_sma, cpuRollingStd, gpudb::gpuRollingStd);
    }

    // ── ASOF join benchmark ──
    std::printf("\n═══ ASOF Join — left=%dM, right=%dM ═══\n",
                N / 1'000'000, 1);
    benchAsofJoin(N, 1'000'000);

    // ── Summary table ──
    std::printf("\n┌────────────────┬────────────┬────────┬──────────┬──────────┬──────────┐\n");
    std::printf("│ Operation      │ N          │ Param  │ CPU ms   │ GPU ms   │ Speedup  │\n");
    std::printf("├────────────────┼────────────┼────────┼──────────┼──────────┼──────────┤\n");
    for (auto& r : g_results) {
        std::printf("│ %-14s │ %10d │ %6d │ %8.1f │ %8.1f │ %6.1fx  │\n",
                    r.operation.c_str(), r.n, r.param,
                    r.cpu_ms, r.gpu_ms, r.speedup);
    }
    std::printf("└────────────────┴────────────┴────────┴──────────┴──────────┴──────────┘\n");

    // ── Write CSV ──
    const char* csv_path = "finance_benchmark_results.csv";
    std::ofstream csv(csv_path);
    csv << "operation,n,param,cpu_ms,gpu_ms,speedup\n";
    for (auto& r : g_results) {
        csv << r.operation << "," << r.n << "," << r.param << ","
            << r.cpu_ms << "," << r.gpu_ms << "," << r.speedup << "\n";
    }
    csv.close();
    std::printf("\n  Results written to %s (%zu rows)\n",
                csv_path, g_results.size());

    std::printf("\n✓ All finance benchmarks completed.\n");
    return 0;
}
