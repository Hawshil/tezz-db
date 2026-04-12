/**
 * @file gpu_window_benchmark.cu
 * @brief GPU vs CPU benchmark for window functions: SMA, EMA, RollingStd.
 *
 * Tests at multiple row counts and window sizes, reports median GPU timing
 * and compares against CPU execution.
 */
#include "gpu/cuda_utils.cuh"
#include "gpu/gpu_ops.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static void hdr(const char* title) {
    std::printf("\n═══ %s ═══\n", title);
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic benchmark driver
// ─────────────────────────────────────────────────────────────────────────────

static void benchWindowFunc(const char* name, int n, int w,
                             std::function<void()> cpu_fn,
                             std::function<void()> gpu_fn) {
    // Warm-up: 3 GPU runs
    for (int i = 0; i < 3; ++i) gpu_fn();

    // Measured: 5 GPU runs, collect timings
    std::vector<double> gpu_times;
    for (int i = 0; i < 5; ++i) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = Clock::now();
        gpu_fn();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = Clock::now();
        gpu_times.push_back(ms(t0, t1));
    }
    std::sort(gpu_times.begin(), gpu_times.end());
    double gpu_median = gpu_times[2]; // median of 5

    // CPU: 1 run
    auto tc0 = Clock::now();
    cpu_fn();
    auto tc1 = Clock::now();
    double cpu_ms_val = ms(tc0, tc1);

    double speedup = (gpu_median > 0.0) ? cpu_ms_val / gpu_median : 0.0;
    std::printf("  %-14s  n=%10d  w=%4d  CPU: %8.1f ms  GPU: %8.1f ms  Speedup: %.1fx\n",
                name, n, w, cpu_ms_val, gpu_median, speedup);
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference implementations (identical to WindowNode CPU path)
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

// ─────────────────────────────────────────────────────────────────────────────
// Per-function benchmarks
// ─────────────────────────────────────────────────────────────────────────────

static void benchSMA(int n, int w) {
    std::vector<double> h_in(n), h_cpu_out(n);
    std::srand(42);
    for (int i = 0; i < n; ++i) h_in[i] = static_cast<double>(std::rand()) / RAND_MAX;

    gpudb::GpuBuffer<double> d_in(n), d_out(n);
    CUDA_CHECK(cudaMemcpy(d_in.data(), h_in.data(), n * sizeof(double),
                          cudaMemcpyHostToDevice));

    benchWindowFunc("SMA", n, w,
        [&]() { cpuSMA(h_in.data(), n, w, h_cpu_out.data()); },
        [&]() { gpudb::gpuSMA(d_in.data(), n, w, d_out.data()); });
}

static void benchEMA(int n, int w) {
    std::vector<double> h_in(n), h_cpu_out(n);
    std::srand(42);
    for (int i = 0; i < n; ++i) h_in[i] = static_cast<double>(std::rand()) / RAND_MAX;

    gpudb::GpuBuffer<double> d_in(n), d_out(n);
    CUDA_CHECK(cudaMemcpy(d_in.data(), h_in.data(), n * sizeof(double),
                          cudaMemcpyHostToDevice));

    benchWindowFunc("EMA", n, w,
        [&]() { cpuEMA(h_in.data(), n, w, h_cpu_out.data()); },
        [&]() { gpudb::gpuEMA(d_in.data(), n, w, d_out.data()); });
}

static void benchRollingStd(int n, int w) {
    std::vector<double> h_in(n), h_cpu_out(n);
    std::srand(42);
    for (int i = 0; i < n; ++i) h_in[i] = static_cast<double>(std::rand()) / RAND_MAX;

    gpudb::GpuBuffer<double> d_in(n), d_out(n);
    CUDA_CHECK(cudaMemcpy(d_in.data(), h_in.data(), n * sizeof(double),
                          cudaMemcpyHostToDevice));

    benchWindowFunc("RollingStd", n, w,
        [&]() { cpuRollingStd(h_in.data(), n, w, h_cpu_out.data()); },
        [&]() { gpudb::gpuRollingStd(d_in.data(), n, w, d_out.data()); });
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::printf("\n  ╔═══════════════════════════════════════════════════╗\n");
    std::printf("  ║  GPUDB — Window Functions Benchmark              ║\n");
    std::printf("  ╚═══════════════════════════════════════════════════╝\n");

    gpudb::GpuContext ctx(0);
    ctx.printDeviceInfo();

    int sizes[] = {1'000'000, 10'000'000, 50'000'000};
    int windows[] = {20, 200};

    for (int n : sizes) {
        for (int w : windows) {
            char title[128];
            std::snprintf(title, sizeof(title),
                          "Window Bench — n=%dM, w=%d", n / 1'000'000, w);
            hdr(title);
            benchSMA(n, w);
            benchEMA(n, w);
            benchRollingStd(n, w);
        }
    }

    std::printf("\n✓ All window benchmarks completed.\n");
    return 0;
}
