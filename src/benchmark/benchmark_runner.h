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
#include <ctime>

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

        // Live-stream this result as a JSON line to the .jsonl file
        if (!jsonl_path_.empty()) {
            streamJsonLine(r);
        }
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

    /// Set the path for the live JSON Lines stream file (.jsonl)
    void enableJsonStream(const std::string& jsonl_path) {
        jsonl_path_ = jsonl_path;
        // Truncate the file at the start of a new run
        std::ofstream f(jsonl_path_, std::ios::trunc);
    }

    /// Write a complete JSON array of all results
    void writeJSON(const std::string& path) const {
        std::ofstream f(path);
        f << "[\n";
        for (std::size_t i = 0; i < results_.size(); ++i) {
            auto& r = results_[i];
            char buf[1024];
            std::snprintf(buf, sizeof(buf),
                "  {\"type\":\"benchmark\",\"operation\":\"%s\",\"rows\":%d,"
                "\"cpu_ms\":%.2f,\"gpu_ms\":%.2f,\"speedup\":%.2f,"
                "\"bandwidth_gbps\":%.2f,\"mean_ms\":%.2f,"
                "\"median_ms\":%.2f,\"p95_ms\":%.2f}",
                r.op.c_str(), r.rows, r.cpu_ms, r.median_ms,
                r.gpu_speedup, r.bw_gbps, r.mean_ms,
                r.median_ms, r.p95_ms);
            f << buf;
            if (i + 1 < results_.size()) f << ",";
            f << "\n";
        }
        f << "]\n";
        std::printf("  JSON results written to %s\n", path.c_str());
    }

private:
    std::vector<BenchResult> results_;
    std::string jsonl_path_;

    static std::string fmtMs(double v) {
        char buf[16]; std::snprintf(buf, sizeof(buf), "%.1f ms", v);
        return buf;
    }

    /// Append a single JSON line for live streaming
    void streamJsonLine(const BenchResult& r) {
        std::ofstream f(jsonl_path_, std::ios::app);
        auto now = std::chrono::system_clock::now();
        auto epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        char buf[1024];
        std::snprintf(buf, sizeof(buf),
            "{\"type\":\"benchmark\",\"operation\":\"%s\",\"rows\":%d,"
            "\"cpu_ms\":%.2f,\"gpu_ms\":%.2f,\"speedup\":%.2f,"
            "\"bandwidth_gbps\":%.2f,\"mean_ms\":%.2f,"
            "\"median_ms\":%.2f,\"p95_ms\":%.2f,"
            "\"timestamp\":%lld}",
            r.op.c_str(), r.rows, r.cpu_ms, r.median_ms,
            r.gpu_speedup, r.bw_gbps, r.mean_ms,
            r.median_ms, r.p95_ms, (long long)epoch_ms);
        f << buf << "\n";
        f.flush();
    }
};

} // namespace gpudb
