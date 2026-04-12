/**
 * @file gpu_benchmark.cu
 * @brief Full GPU vs CPU benchmark: Filter, Sum, GroupBy, Join, Async Pipeline.
 */
#include "gpu/cuda_utils.cuh"
#include "gpu/gpu_ops.cuh"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <unordered_map>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double,std::milli>(b-a).count();
}
static void hdr(const char* t) { std::printf("\n═══ %s ═══\n", t); }

// ─────────────────────────────────────────────────────────────────────────────
// 1. Filter  (100 M elements, val > 0.5)
// ─────────────────────────────────────────────────────────────────────────────
static void benchFilter() {
    hdr("GPU Filter — 50M doubles, val > 0.5");
    const int N = 50'000'000;
    std::vector<double> h(N);
    std::srand(42);
    for (int i = 0; i < N; ++i) h[i] = (double)std::rand() / RAND_MAX;

    // CPU
    auto t0 = Clock::now();
    std::vector<double> cpu_out; cpu_out.reserve(N/2);
    for (int i = 0; i < N; ++i) if (h[i] > 0.5) cpu_out.push_back(h[i]);
    auto t1 = Clock::now();
    double cpu_ms = ms(t0, t1);

    // GPU
    gpudb::GpuBuffer<double> d_in(N), d_out(N);
    CUDA_CHECK(cudaMemcpy(d_in.data(), h.data(), N*8, cudaMemcpyHostToDevice));
    auto t2 = Clock::now();
    int gpu_cnt = gpudb::gpuFilter(d_in.data(), N, 0.5, gpudb::CompareOp::GT, d_out.data());
    auto t3 = Clock::now();
    double gpu_ms = ms(t2, t3);

    std::printf("  CPU: %8.1f ms  (%d selected)\n", cpu_ms, (int)cpu_out.size());
    std::printf("  GPU: %8.1f ms  (%d selected)\n", gpu_ms, gpu_cnt);
    std::printf("  Speedup: %.1fx\n", cpu_ms / gpu_ms);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. SUM  (500 M doubles)
// ─────────────────────────────────────────────────────────────────────────────
static void benchSum() {
    hdr("GPU Sum — 200M doubles");
    const int N = 200'000'000;
    std::vector<double> h(N);
    for (int i = 0; i < N; ++i) h[i] = 1.0 / (i + 1);

    // CPU
    auto t0 = Clock::now();
    double cpu_sum = std::accumulate(h.begin(), h.end(), 0.0);
    auto t1 = Clock::now();

    // GPU
    gpudb::GpuBuffer<double> d(N);
    CUDA_CHECK(cudaMemcpy(d.data(), h.data(), (size_t)N*8, cudaMemcpyHostToDevice));
    auto t2 = Clock::now();
    double gpu_sum = gpudb::gpuSum(d.data(), N);
    auto t3 = Clock::now();

    std::printf("  CPU: %8.1f ms  sum=%.6f\n", ms(t0,t1), cpu_sum);
    std::printf("  GPU: %8.1f ms  sum=%.6f\n", ms(t2,t3), gpu_sum);
    std::printf("  Speedup: %.1fx\n", ms(t0,t1) / ms(t2,t3));
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. GROUP BY + SUM  (100 M rows, 100 groups)
// ─────────────────────────────────────────────────────────────────────────────
static void benchGroupBy() {
    hdr("GPU GroupBy — 50M rows, 100 groups");
    const int N = 50'000'000, G = 100;
    std::vector<int>    h_k(N);
    std::vector<double> h_v(N);
    std::srand(77);
    for (int i = 0; i < N; ++i) { h_k[i] = std::rand() % G; h_v[i] = (std::rand()%10000)/100.0; }

    // CPU
    auto t0 = Clock::now();
    std::unordered_map<int, double> cpu_map; cpu_map.reserve(G*2);
    for (int i = 0; i < N; ++i) cpu_map[h_k[i]] += h_v[i];
    auto t1 = Clock::now();

    // GPU
    gpudb::GpuBuffer<int> dk(N); gpudb::GpuBuffer<double> dv(N);
    CUDA_CHECK(cudaMemcpy(dk.data(), h_k.data(), N*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv.data(), h_v.data(), N*8, cudaMemcpyHostToDevice));
    std::vector<int> ok(G*4); std::vector<double> os(G*4);
    auto t2 = Clock::now();
    int gc = gpudb::gpuGroupBySum(dk.data(), dv.data(), N, ok.data(), os.data(), G*4);
    auto t3 = Clock::now();

    std::printf("  CPU: %8.1f ms  (%d groups)\n", ms(t0,t1), (int)cpu_map.size());
    std::printf("  GPU: %8.1f ms  (%d groups)\n", ms(t2,t3), gc);
    std::printf("  Speedup: %.1fx\n", ms(t0,t1) / ms(t2,t3));
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Hash Join  (1 M build × 10 M probe)
// ─────────────────────────────────────────────────────────────────────────────
static void benchJoin() {
    hdr("GPU Join — 1M build × 10M probe");
    const int NB = 1'000'000, NP = 10'000'000;
    std::vector<int> hb(NB), hp(NP);
    for (int i = 0; i < NB; ++i) hb[i] = i;
    std::srand(99);
    for (int i = 0; i < NP; ++i) hp[i] = std::rand() % NB;

    // CPU
    auto t0 = Clock::now();
    std::unordered_multimap<int,int> cpum; cpum.reserve(NB);
    for (int i = 0; i < NB; ++i) cpum.emplace(hb[i], i);
    int cpu_matches = 0;
    for (int i = 0; i < NP; ++i) {
        auto [b,e] = cpum.equal_range(hp[i]);
        for (auto it = b; it != e; ++it) ++cpu_matches;
    }
    auto t1 = Clock::now();

    // GPU
    int ht_cap = NB * 2;
    gpudb::GpuBuffer<int> db(NB), dp(NP), ob(NP), op(NP);
    CUDA_CHECK(cudaMemcpy(db.data(), hb.data(), NB*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dp.data(), hp.data(), NP*4, cudaMemcpyHostToDevice));
    auto t2 = Clock::now();
    int gpu_matches = gpudb::gpuHashJoin(db.data(), NB, dp.data(), NP,
                                          ob.data(), op.data(), ht_cap);
    auto t3 = Clock::now();

    std::printf("  CPU: %8.1f ms  (%d matches)\n", ms(t0,t1), cpu_matches);
    std::printf("  GPU: %8.1f ms  (%d matches)\n", ms(t2,t3), gpu_matches);
    std::printf("  Speedup: %.1fx\n", ms(t0,t1) / ms(t2,t3));
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Async Pipeline  (500 M rows in 50 M chunks)
// ─────────────────────────────────────────────────────────────────────────────
static void benchPipeline() {
    hdr("Async Pipeline — 200M rows in 25M-row chunks");
    const int N = 200'000'000, CHUNK = 25'000'000, G = 100;

    // Pinned host memory for async DMA
    gpudb::GpuPinnedBuffer<double> h_vals(N);
    gpudb::GpuPinnedBuffer<int>    h_keys(N);
    std::srand(55);
    for (int i = 0; i < N; ++i) {
        h_vals.data()[i] = (std::rand() % 10000) / 100.0;
        h_keys.data()[i] = std::rand() % G;
    }

    gpudb::runAsyncPipeline(h_vals.data(), h_keys.data(), N, CHUNK, G);
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::printf("\n  ╔═══════════════════════════════════════════════════╗\n");
    std::printf("  ║  GPUDB — GPU vs CPU Benchmark Suite              ║\n");
    std::printf("  ╚═══════════════════════════════════════════════════╝\n");

    gpudb::GpuContext ctx(0);
    ctx.printDeviceInfo();

    benchFilter();
    benchSum();
    benchGroupBy();
    benchJoin();
    benchPipeline();

    std::printf("\n✓ All GPU benchmarks completed.\n");
    return 0;
}
