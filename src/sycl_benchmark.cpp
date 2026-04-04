/**
 * @file sycl_benchmark.cpp
 * @brief Full Intel GPU benchmark — mirrors CUDA/HIP benchmarks.
 *
 * Profiling with Intel VTune:
 *   vtune -collect gpu-hotspots -- ./gpudb_sycl_bench
 *     → Reports EU utilization, L3 cache bandwidth, PCIe transfer time
 *
 *   vtune -collect gpu-offload -- ./gpudb_sycl_bench
 *     → Reports kernel execution times and data transfer overhead
 */
#include "sycl/sycl_utils.h"
#include "sycl/sycl_ops.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <unordered_map>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
static double tms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double,std::milli>(b-a).count();
}

static void benchFilter(sycl::queue& q) {
    std::printf("\n═══ SYCL Filter — 100M doubles ═══\n");
    const int N = 100'000'000;
    std::vector<double> h(N);
    std::srand(42);
    for (int i = 0; i < N; ++i) h[i] = (double)std::rand() / RAND_MAX;

    auto t0 = Clock::now();
    int cpu_cnt = 0;
    for (int i = 0; i < N; ++i) if (h[i] > 0.5) ++cpu_cnt;
    auto t1 = Clock::now();

    gpudb::SyclBuffer<double> d_in(N, q), d_out(N, q);
    d_in.copyFrom(h.data(), N);
    auto t2 = Clock::now();
    int gpu_cnt = gpudb::syclFilter(q, d_in.data(), N, 0.5,
                                     gpudb::SyclCompareOp::GT, d_out.data());
    auto t3 = Clock::now();

    std::printf("  CPU: %8.1f ms  (%d pass)\n", tms(t0,t1), cpu_cnt);
    std::printf("  GPU: %8.1f ms  (%d pass)\n", tms(t2,t3), gpu_cnt);
    std::printf("  Speedup: %.1fx\n", tms(t0,t1)/tms(t2,t3));
}

static void benchSum(sycl::queue& q) {
    std::printf("\n═══ SYCL Sum — 500M doubles ═══\n");
    const int N = 500'000'000;
    std::vector<double> h(N);
    for (int i = 0; i < N; ++i) h[i] = 1.0 / (i + 1);

    auto t0 = Clock::now();
    double cs = std::accumulate(h.begin(), h.end(), 0.0);
    auto t1 = Clock::now();

    gpudb::SyclBuffer<double> d(N, q);
    d.copyFrom(h.data(), N);

    auto t2 = Clock::now();
    double gs_auto = gpudb::syclSum(q, d.data(), N);
    auto t3 = Clock::now();

    double gs_manual = gpudb::syclSumManual(q, d.data(), N);
    auto t4 = Clock::now();

    std::printf("  CPU:             %8.1f ms  sum=%.6f\n", tms(t0,t1), cs);
    std::printf("  GPU (reduction): %8.1f ms  sum=%.6f\n", tms(t2,t3), gs_auto);
    std::printf("  GPU (manual):    %8.1f ms  sum=%.6f\n", tms(t3,t4), gs_manual);
    std::printf("  Speedup: %.1fx (idiomatic)\n", tms(t0,t1)/tms(t2,t3));
}

static void benchGroupBy(sycl::queue& q) {
    std::printf("\n═══ SYCL GroupBy — 100M rows, 100 groups ═══\n");
    const int N = 100'000'000, G = 100;
    std::vector<int> hk(N); std::vector<double> hv(N);
    std::srand(77);
    for (int i = 0; i < N; ++i) { hk[i] = std::rand()%G; hv[i] = (std::rand()%10000)/100.0; }

    auto t0 = Clock::now();
    std::unordered_map<int,double> cm; cm.reserve(G*2);
    for (int i = 0; i < N; ++i) cm[hk[i]] += hv[i];
    auto t1 = Clock::now();

    gpudb::SyclBuffer<int> dk(N, q); gpudb::SyclBuffer<double> dv(N, q);
    dk.copyFrom(hk.data(), N); dv.copyFrom(hv.data(), N);
    std::vector<int> ok(G*4); std::vector<double> os(G*4);
    auto t2 = Clock::now();
    int gc = gpudb::syclGroupBySum(q, dk.data(), dv.data(), N,
                                    ok.data(), os.data(), G*4);
    auto t3 = Clock::now();

    std::printf("  CPU: %8.1f ms  (%d groups)\n", tms(t0,t1), (int)cm.size());
    std::printf("  GPU: %8.1f ms  (%d groups)\n", tms(t2,t3), gc);
    std::printf("  Speedup: %.1fx\n", tms(t0,t1)/tms(t2,t3));
}

static void benchJoin(sycl::queue& q) {
    std::printf("\n═══ SYCL Join — 1M × 10M ═══\n");
    const int NB = 1'000'000, NP = 10'000'000;
    std::vector<int> hb(NB), hp(NP);
    for (int i = 0; i < NB; ++i) hb[i] = i;
    std::srand(99);
    for (int i = 0; i < NP; ++i) hp[i] = std::rand() % NB;

    auto t0 = Clock::now();
    std::unordered_multimap<int,int> cpum; cpum.reserve(NB);
    for (int i = 0; i < NB; ++i) cpum.emplace(hb[i], i);
    int cmatch = 0;
    for (int i = 0; i < NP; ++i) {
        auto [b,e] = cpum.equal_range(hp[i]);
        for (auto it = b; it != e; ++it) ++cmatch;
    }
    auto t1 = Clock::now();

    int ht_cap = NB * 2;
    gpudb::SyclBuffer<int> db(NB, q), dp(NP, q), ob(NP, q), op(NP, q);
    db.copyFrom(hb.data(), NB); dp.copyFrom(hp.data(), NP);
    auto t2 = Clock::now();
    int gmatch = gpudb::syclHashJoin(q, db.data(), NB, dp.data(), NP,
                                      ob.data(), op.data(), ht_cap);
    auto t3 = Clock::now();

    std::printf("  CPU: %8.1f ms  (%d matches)\n", tms(t0,t1), cmatch);
    std::printf("  GPU: %8.1f ms  (%d matches)\n", tms(t2,t3), gmatch);
    std::printf("  Speedup: %.1fx\n", tms(t0,t1)/tms(t2,t3));
}

int main() {
    std::printf("\n  ╔═══════════════════════════════════════════════════╗\n");
    std::printf("  ║  GPUDB — Intel oneAPI/SYCL Benchmark Suite        ║\n");
    std::printf("  ╚═══════════════════════════════════════════════════╝\n");

    std::printf("\n  Available devices:\n");
    gpudb::listAllDevices();

    sycl::queue q = gpudb::makeGpuQueue();
    gpudb::printDeviceInfo(q);
    gpudb::verifySetup(q);

    benchFilter(q);
    benchSum(q);
    benchGroupBy(q);
    benchJoin(q);

    std::printf("\n✓ All SYCL benchmarks completed.\n");
    std::printf("\nTo profile with Intel VTune:\n");
    std::printf("  vtune -collect gpu-hotspots -- ./gpudb_sycl_bench\n");
    return 0;
}
