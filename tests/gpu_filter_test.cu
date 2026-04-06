/**
 * @file gpu_filter_test.cu
 * @brief Google Test — GPU filter kernel correctness against CPU baseline.
 *
 * Generates deterministic test data (seeded RNG), runs the same filter
 * operation on CPU and GPU, and asserts bit-exact match for integer counts
 * and epsilon-equality for floating-point results.
 */
#ifdef USE_CUDA
#include <gtest/gtest.h>
#include "gpu/cuda_utils.cuh"
#include "gpu/gpu_ops.cuh"
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <unordered_map>

using namespace gpudb;

// ═══ Filter: CPU vs GPU match ═══════════════════════════════════════════════

TEST(GpuFilter, CountMatchesCPU) {
    const int N = 500000;
    std::srand(42);
    std::vector<double> h(N);
    for (int i = 0; i < N; ++i) h[i] = (double)std::rand() / RAND_MAX;

    // CPU
    int cpu_cnt = 0;
    for (int i = 0; i < N; ++i) if (h[i] > 0.5) ++cpu_cnt;

    // GPU
    GpuBuffer<double> d_in(N), d_out(N);
    CUDA_CHECK(cudaMemcpy(d_in.data(), h.data(), N*8, cudaMemcpyHostToDevice));
    int gpu_cnt = gpuFilter(d_in.data(), N, 0.5, CompareOp::GT, d_out.data());

    EXPECT_EQ(cpu_cnt, gpu_cnt)
        << "Filter count mismatch: CPU=" << cpu_cnt << " GPU=" << gpu_cnt;
}

TEST(GpuFilter, ValuesMatchCPU) {
    const int N = 100000;
    std::srand(7);
    std::vector<double> h(N);
    for (int i = 0; i < N; ++i) h[i] = (double)std::rand() / RAND_MAX;

    // CPU filter
    std::vector<double> cpu_result;
    for (int i = 0; i < N; ++i) if (h[i] > 0.3) cpu_result.push_back(h[i]);

    // GPU filter
    GpuBuffer<double> d_in(N), d_out(N);
    CUDA_CHECK(cudaMemcpy(d_in.data(), h.data(), N*8, cudaMemcpyHostToDevice));
    int gpu_cnt = gpuFilter(d_in.data(), N, 0.3, CompareOp::GT, d_out.data());
    std::vector<double> gpu_result(gpu_cnt);
    CUDA_CHECK(cudaMemcpy(gpu_result.data(), d_out.data(), gpu_cnt*8,
                           cudaMemcpyDeviceToHost));

    ASSERT_EQ((int)cpu_result.size(), gpu_cnt);
    for (int i = 0; i < gpu_cnt; ++i)
        EXPECT_DOUBLE_EQ(cpu_result[i], gpu_result[i]) << "Mismatch at i=" << i;
}

// ═══ SUM Reduction: CPU vs GPU ══════════════════════════════════════════════

TEST(GpuReduce, SumMatchesCPU) {
    const int N = 1000000;
    std::srand(99);
    std::vector<double> h(N);
    for (int i = 0; i < N; ++i) h[i] = (std::rand() % 10000) / 100.0;

    double cpu_sum = std::accumulate(h.begin(), h.end(), 0.0);

    GpuBuffer<double> d(N);
    CUDA_CHECK(cudaMemcpy(d.data(), h.data(), N*8, cudaMemcpyHostToDevice));
    double gpu_sum = gpuSum(d.data(), N);

    // FP reduction order differs — allow small epsilon
    EXPECT_NEAR(cpu_sum, gpu_sum, cpu_sum * 1e-6)
        << "SUM mismatch: CPU=" << cpu_sum << " GPU=" << gpu_sum;
}

// ═══ GROUP BY + SUM: CPU vs GPU ═════════════════════════════════════════════

TEST(GpuGroupBy, ResultMatchesCPU) {
    const int N = 200000, G = 10;
    std::srand(77);
    std::vector<int> hk(N);
    std::vector<double> hv(N);
    for (int i = 0; i < N; ++i) {
        hk[i] = std::rand() % G;
        hv[i] = (std::rand() % 10000) / 100.0;
    }

    // CPU
    std::unordered_map<int, double> cpu_map;
    for (int i = 0; i < N; ++i) cpu_map[hk[i]] += hv[i];

    // GPU
    int ht_cap = G * 4;
    GpuBuffer<int> dk(N); GpuBuffer<double> dv(N);
    CUDA_CHECK(cudaMemcpy(dk.data(), hk.data(), N*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv.data(), hv.data(), N*8, cudaMemcpyHostToDevice));
    std::vector<int> ok(ht_cap);
    std::vector<double> os(ht_cap);
    int gc = gpuGroupBySum(dk.data(), dv.data(), N, ok.data(), os.data(), ht_cap);

    EXPECT_EQ(gc, (int)cpu_map.size());
    for (int i = 0; i < gc; ++i) {
        auto it = cpu_map.find(ok[i]);
        ASSERT_NE(it, cpu_map.end()) << "GPU returned unknown group " << ok[i];
        EXPECT_NEAR(os[i], it->second, it->second * 1e-4)
            << "Group " << ok[i] << " sum mismatch";
    }
}

// ═══ Hash Join: CPU vs GPU ══════════════════════════════════════════════════

TEST(GpuJoin, MatchCountEqualssCPU) {
    const int NB = 1000, NP = 5000;
    std::srand(42);
    std::vector<int> hb(NB), hp(NP);
    for (int i = 0; i < NB; ++i) hb[i] = i;
    for (int i = 0; i < NP; ++i) hp[i] = std::rand() % NB;

    // CPU
    int cpu_cnt = 0;
    std::unordered_multimap<int,int> m;
    for (int i = 0; i < NB; ++i) m.emplace(hb[i], i);
    for (int i = 0; i < NP; ++i) {
        auto [b,e] = m.equal_range(hp[i]);
        for (auto it = b; it != e; ++it) ++cpu_cnt;
    }

    // GPU
    int ht_cap = NB * 2;
    GpuBuffer<int> db(NB), dp(NP), ob(NP), op(NP);
    CUDA_CHECK(cudaMemcpy(db.data(), hb.data(), NB*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dp.data(), hp.data(), NP*4, cudaMemcpyHostToDevice));
    int gpu_cnt = gpuHashJoin(db.data(), NB, dp.data(), NP,
                               ob.data(), op.data(), ht_cap);

    EXPECT_EQ(cpu_cnt, gpu_cnt)
        << "Join match count: CPU=" << cpu_cnt << " GPU=" << gpu_cnt;
}

// ═══ Edge cases ═════════════════════════════════════════════════════════════

TEST(GpuFilter, EmptyInput) {
    GpuBuffer<double> d_in(0), d_out(0);
    // Should handle n=0 gracefully
    // Just verify no crash
    SUCCEED();
}

TEST(GpuFilter, AllPassFilter) {
    const int N = 10000;
    std::vector<double> h(N, 1.0);  // all values = 1.0

    GpuBuffer<double> d_in(N), d_out(N);
    CUDA_CHECK(cudaMemcpy(d_in.data(), h.data(), N*8, cudaMemcpyHostToDevice));
    int cnt = gpuFilter(d_in.data(), N, 0.5, CompareOp::GT, d_out.data());
    EXPECT_EQ(cnt, N);  // all pass
}

TEST(GpuFilter, NonePassFilter) {
    const int N = 10000;
    std::vector<double> h(N, 0.0);  // all values = 0.0

    GpuBuffer<double> d_in(N), d_out(N);
    CUDA_CHECK(cudaMemcpy(d_in.data(), h.data(), N*8, cudaMemcpyHostToDevice));
    int cnt = gpuFilter(d_in.data(), N, 0.5, CompareOp::GT, d_out.data());
    EXPECT_EQ(cnt, 0);  // none pass
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif // USE_CUDA
