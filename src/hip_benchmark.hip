/**
 * @file hip_benchmark.hip
 * @brief Full AMD GPU benchmark — mirrors gpu_benchmark.cu for GPU vs CPU comparison.
 *
 * Profiling with rocprof
 * ──────────────────────
 *   rocprof --stats ./gpudb_hip_bench
 *     → Prints per-kernel: duration, grid size, wavefront occupancy
 *
 *   rocprof -i counters.txt ./gpudb_hip_bench
 *     counters.txt contains: SQ_WAVES, SQ_INSTS_VALU, TCP_TCC_READ_REQ_sum, etc.
 *     → Reports wavefront occupancy, L1/L2 cache hit rate, memory bandwidth
 */
#include "hip/hip_utils.h"
#include "hip/hip_ops.h"
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

static void benchFilter() {
    std::printf("\n═══ HIP Filter — 100M doubles ═══\n");
    const int N = 100'000'000;
    std::vector<double> h(N);
    std::srand(42);
    for (int i = 0; i < N; ++i) h[i] = (double)std::rand() / RAND_MAX;

    auto t0 = Clock::now();
    int cpu_cnt = 0;
    for (int i = 0; i < N; ++i) if (h[i] > 0.5) ++cpu_cnt;
    auto t1 = Clock::now();

    gpudb::HipBuffer<double> d_in(N), d_out(N);
    HIP_CHECK(hipMemcpy(d_in.data(), h.data(), N*8, hipMemcpyHostToDevice));
    auto t2 = Clock::now();
    int gpu_cnt = gpudb::hipFilter(d_in.data(), N, 0.5, gpudb::HipCompareOp::GT, d_out.data());
    auto t3 = Clock::now();

    std::printf("  CPU: %8.1f ms  (%d pass)\n", tms(t0,t1), cpu_cnt);
    std::printf("  GPU: %8.1f ms  (%d pass)\n", tms(t2,t3), gpu_cnt);
    std::printf("  Speedup: %.1fx\n", tms(t0,t1)/tms(t2,t3));
}

static void benchSum() {
    std::printf("\n═══ HIP Sum — 500M doubles ═══\n");
    const int N = 500'000'000;
    std::vector<double> h(N);
    for (int i = 0; i < N; ++i) h[i] = 1.0 / (i + 1);

    auto t0 = Clock::now();
    double cs = std::accumulate(h.begin(), h.end(), 0.0);
    auto t1 = Clock::now();

    gpudb::HipBuffer<double> d(N);
    HIP_CHECK(hipMemcpy(d.data(), h.data(), (size_t)N*8, hipMemcpyHostToDevice));
    auto t2 = Clock::now();
    double gs = gpudb::hipSum(d.data(), N);
    auto t3 = Clock::now();

    std::printf("  CPU: %8.1f ms  sum=%.6f\n", tms(t0,t1), cs);
    std::printf("  GPU: %8.1f ms  sum=%.6f\n", tms(t2,t3), gs);
    std::printf("  Speedup: %.1fx\n", tms(t0,t1)/tms(t2,t3));
}

static void benchGroupBy() {
    std::printf("\n═══ HIP GroupBy — 100M rows, 100 groups ═══\n");
    const int N = 100'000'000, G = 100;
    std::vector<int> hk(N); std::vector<double> hv(N);
    std::srand(77);
    for (int i = 0; i < N; ++i) { hk[i] = std::rand()%G; hv[i] = (std::rand()%10000)/100.0; }

    auto t0 = Clock::now();
    std::unordered_map<int,double> cm; cm.reserve(G*2);
    for (int i = 0; i < N; ++i) cm[hk[i]] += hv[i];
    auto t1 = Clock::now();

    gpudb::HipBuffer<int> dk(N); gpudb::HipBuffer<double> dv(N);
    HIP_CHECK(hipMemcpy(dk.data(), hk.data(), N*4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dv.data(), hv.data(), N*8, hipMemcpyHostToDevice));
    std::vector<int> ok(G*4); std::vector<double> os(G*4);
    auto t2 = Clock::now();
    int gc = gpudb::hipGroupBySum(dk.data(), dv.data(), N, ok.data(), os.data(), G*4);
    auto t3 = Clock::now();

    std::printf("  CPU: %8.1f ms  (%d groups)\n", tms(t0,t1), (int)cm.size());
    std::printf("  GPU: %8.1f ms  (%d groups)\n", tms(t2,t3), gc);
    std::printf("  Speedup: %.1fx\n", tms(t0,t1)/tms(t2,t3));
}

static void benchJoin() {
    std::printf("\n═══ HIP Join — 1M × 10M ═══\n");
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
    gpudb::HipBuffer<int> db(NB), dp(NP), ob(NP), op(NP);
    HIP_CHECK(hipMemcpy(db.data(), hb.data(), NB*4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dp.data(), hp.data(), NP*4, hipMemcpyHostToDevice));
    auto t2 = Clock::now();
    int gmatch = gpudb::hipHashJoin(db.data(), NB, dp.data(), NP,
                                     ob.data(), op.data(), ht_cap);
    auto t3 = Clock::now();

    std::printf("  CPU: %8.1f ms  (%d matches)\n", tms(t0,t1), cmatch);
    std::printf("  GPU: %8.1f ms  (%d matches)\n", tms(t2,t3), gmatch);
    std::printf("  Speedup: %.1fx\n", tms(t0,t1)/tms(t2,t3));
}

static void benchPipeline() {
    std::printf("\n═══ HIP Async Pipeline — 500M rows ═══\n");
    const int N = 500'000'000, CHUNK = 50'000'000, G = 100;
    gpudb::HipPinnedBuffer<double> hv(N);
    gpudb::HipPinnedBuffer<int>    hk(N);
    std::srand(55);
    for (int i = 0; i < N; ++i) {
        hv.data()[i] = (std::rand()%10000)/100.0;
        hk.data()[i] = std::rand() % G;
    }
    gpudb::hipRunAsyncPipeline(hv.data(), hk.data(), N, CHUNK, G);
}

int main() {
    std::printf("\n  ╔═══════════════════════════════════════════════════╗\n");
    std::printf("  ║  GPUDB — AMD ROCm/HIP Benchmark Suite            ║\n");
    std::printf("  ╚═══════════════════════════════════════════════════╝\n");

    gpudb::HipContext ctx(0);
    ctx.printDeviceInfo();

    benchFilter();
    benchSum();
    benchGroupBy();
    benchJoin();
    benchPipeline();

    std::printf("\n✓ All HIP benchmarks completed.\n");
    std::printf("\nTo profile with rocprof:\n");
    std::printf("  rocprof --stats ./gpudb_hip_bench\n");
    std::printf("  rocprof -i counters.txt ./gpudb_hip_bench\n");
    return 0;
}
