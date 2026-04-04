/**
 * @file benchmark_runner.h
 * @brief Formal benchmark harness — warm-up, measured iterations, stats, CSV.
 */
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

namespace gpudb {

struct BenchResult {
    std::string op;
    int rows;
    double mean_ms, median_ms, p95_ms;
    double cpu_ms, gpu_speedup, bw_gbps;
    std::string cuda_tag{"N/A"}, hip_tag{"N/A"}, sycl_tag{"N/A"};
};

class BenchmarkRunner {
public:
    void run(const std::string& op, int rows, std::size_t data_bytes,
             std::function<double()> cpu_fn, std::function<double()> gpu_fn,
             const std::string& platform = "CUDA", int warmup = 5, int iters = 10) {
        // Warm-up
        for (int i = 0; i < warmup; ++i) { gpu_fn(); }

        // Measure
        std::vector<double> times(iters);
        for (int i = 0; i < iters; ++i) times[i] = gpu_fn();
        std::sort(times.begin(), times.end());

        double cpu = cpu_fn();
        double mean = std::accumulate(times.begin(), times.end(), 0.0) / iters;
        double med = times[iters / 2];
        double p95 = times[(int)(iters * 0.95)];
        double bw = data_bytes / (med / 1000.0) / 1e9;

        BenchResult r{op, rows, mean, med, p95, cpu, cpu / med, bw};
        if (platform == "CUDA") r.cuda_tag = fmtMs(med);
        if (platform == "HIP")  r.hip_tag  = fmtMs(med);
        if (platform == "SYCL") r.sycl_tag = fmtMs(med);
        results_.push_back(r);
    }

    void printTable() const {
        std::printf("\n┌──────────────┬───────────┬─────────┬─────────┬─────────┬─────────┬──────────┬──────────┬──────────┬──────────┐\n");
        std::printf("│ Operator     │ Rows      │ Mean ms │ Med  ms │ P95  ms │ CPU  ms │ Speedup  │ CUDA     │ HIP      │ SYCL     │\n");
        std::printf("├──────────────┼───────────┼─────────┼─────────┼─────────┼─────────┼──────────┼──────────┼──────────┼──────────┤\n");
        for (auto& r : results_) {
            std::printf("│ %-12s │ %9d │ %7.1f │ %7.1f │ %7.1f │ %7.1f │ %6.1fx  │ %-8s │ %-8s │ %-8s │\n",
                r.op.c_str(), r.rows, r.mean_ms, r.median_ms, r.p95_ms,
                r.cpu_ms, r.gpu_speedup, r.cuda_tag.c_str(),
                r.hip_tag.c_str(), r.sycl_tag.c_str());
        }
        std::printf("└──────────────┴───────────┴─────────┴─────────┴─────────┴─────────┴──────────┴──────────┴──────────┴──────────┘\n");
    }

    void writeCSV(const std::string& path) const {
        std::ofstream f(path);
        f << "operator,rows,mean_ms,median_ms,p95_ms,cpu_ms,speedup,bw_gbps,cuda,hip,sycl\n";
        for (auto& r : results_)
            f << r.op << "," << r.rows << "," << r.mean_ms << "," << r.median_ms
              << "," << r.p95_ms << "," << r.cpu_ms << "," << r.gpu_speedup
              << "," << r.bw_gbps << "," << r.cuda_tag << ","
              << r.hip_tag << "," << r.sycl_tag << "\n";
        std::printf("  Results written to %s\n", path.c_str());
    }

private:
    std::vector<BenchResult> results_;
    static std::string fmtMs(double v) {
        char buf[16]; std::snprintf(buf, sizeof(buf), "%.1f ms", v);
        return buf;
    }
};

} // namespace gpudb
