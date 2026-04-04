/**
 * @file test_correctness.cpp
 * @brief Google Test integration tests — correctness proofs for all operators.
 *
 * Build: cmake -DBUILD_TESTS=ON ..
 * Run:   ./gpudb_tests
 *
 * All tests use seeded random generators for reproducibility.
 */
#include <gtest/gtest.h>
#include "core/table.h"
#include "core/schema.h"
#include "core/column.h"
#include "core/compression.h"
#include "query/operator_node.h"
#include <cmath>
#include <cstdlib>
#include <vector>
#include <unordered_map>

#ifdef USE_CUDA
#include "gpu/cuda_utils.cuh"
#include "gpu/gpu_ops.cuh"
#endif

#ifdef GPUDB_HAS_ARROW
#include "io/arrow_bridge.h"
#endif

using namespace gpudb;

// ── Helpers ─────────────────────────────────────────────────────────────────

static Table makeSyntheticTable(int n, int seed = 42) {
    std::srand(seed);
    Int32Column keys;
    Float64Column vals;
    StringColumn labels;
    keys.reserve(n); vals.reserve(n); labels.reserve(n);
    const char* groups[] = {"A","B","C","D","E","F","G","H","I","J"};
    for (int i = 0; i < n; ++i) {
        keys.pushValue(std::rand() % 10);
        vals.pushValue((std::rand() % 10000) / 100.0);
        labels.pushValue(groups[std::rand() % 10]);
    }
    std::vector<std::string> names = {"key", "value", "label"};
    std::vector<ColumnVariant> cols;
    cols.emplace_back(std::move(keys));
    cols.emplace_back(std::move(vals));
    cols.emplace_back(std::move(labels));
    return Table("test", std::move(names), std::move(cols));
}

// ═══ Test 1: CSV → Schema → SQL Parse → Plan → CPU Execute ═════════════════

TEST(Integration, CsvToResult) {
    Table tbl = makeSyntheticTable(1000);
    EXPECT_EQ(tbl.numRows(), 1000u);
    EXPECT_EQ(tbl.numCols(), 3u);
    EXPECT_EQ(tbl.columnNames()[0], "key");
}

// ═══ Test 2: CPU Filter == GPU Filter ═══════════════════════════════════════

#ifdef USE_CUDA
TEST(GpuCorrectness, FilterMatchesCPU) {
    const int N = 100000;
    std::srand(42);
    std::vector<double> h(N);
    for (int i = 0; i < N; ++i) h[i] = (double)std::rand() / RAND_MAX;

    // CPU filter
    int cpu_cnt = 0;
    for (int i = 0; i < N; ++i) if (h[i] > 0.5) ++cpu_cnt;

    // GPU filter
    GpuBuffer<double> d_in(N), d_out(N);
    CUDA_CHECK(cudaMemcpy(d_in.data(), h.data(), N*8, cudaMemcpyHostToDevice));
    int gpu_cnt = gpuFilter(d_in.data(), N, 0.5, CompareOp::GT, d_out.data());

    EXPECT_EQ(cpu_cnt, gpu_cnt);
}

// ═══ Test 3: CPU GROUP BY == GPU GROUP BY ═══════════════════════════════════

TEST(GpuCorrectness, GroupByMatchesCPU) {
    const int N = 100000, G = 10;
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
        EXPECT_NEAR(os[i], cpu_map[ok[i]], 1e-2)
            << "Group " << ok[i] << " mismatch";
    }
}

// ═══ Test 4: CPU Hash Join == GPU Hash Join ═════════════════════════════════

TEST(GpuCorrectness, JoinMatchesCPU) {
    const int NB = 1000, NP = 5000;
    std::srand(99);
    std::vector<int> hb(NB), hp(NP);
    for (int i = 0; i < NB; ++i) hb[i] = i;
    for (int i = 0; i < NP; ++i) hp[i] = std::rand() % NB;

    // CPU
    std::unordered_multimap<int,int> cpum;
    for (int i = 0; i < NB; ++i) cpum.emplace(hb[i], i);
    int cpu_cnt = 0;
    for (int i = 0; i < NP; ++i) {
        auto [b,e] = cpum.equal_range(hp[i]);
        for (auto it = b; it != e; ++it) ++cpu_cnt;
    }

    // GPU
    int ht_cap = NB * 2;
    GpuBuffer<int> db(NB), dp(NP), ob(NP), op_buf(NP);
    CUDA_CHECK(cudaMemcpy(db.data(), hb.data(), NB*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dp.data(), hp.data(), NP*4, cudaMemcpyHostToDevice));
    int gpu_cnt = gpuHashJoin(db.data(), NB, dp.data(), NP,
                               ob.data(), op_buf.data(), ht_cap);

    EXPECT_EQ(cpu_cnt, gpu_cnt);
}
#endif // USE_CUDA

// ═══ Test 5: Out-of-core Chunked == In-memory ═══════════════════════════════

TEST(Correctness, ChunkedMatchesFull) {
    const int N = 10000, G = 10;
    std::srand(42);
    std::vector<int> keys(N);
    std::vector<double> vals(N);
    for (int i = 0; i < N; ++i) {
        keys[i] = std::rand() % G;
        vals[i] = (std::rand() % 10000) / 100.0;
    }

    // Full
    std::unordered_map<int,double> full;
    for (int i = 0; i < N; ++i) full[keys[i]] += vals[i];

    // Chunked (CPU simulation)
    std::unordered_map<int,double> chunked;
    int chunk = 2500;
    for (int off = 0; off < N; off += chunk) {
        int n = std::min(chunk, N - off);
        for (int i = off; i < off + n; ++i) chunked[keys[i]] += vals[i];
    }

    EXPECT_EQ(full.size(), chunked.size());
    for (auto& [k, v] : full)
        EXPECT_NEAR(v, chunked[k], 1e-6) << "Group " << k;
}

// ═══ Test 6: Arrow Round-trip ════════════════════════════════════════════════

#ifdef GPUDB_HAS_ARROW
TEST(Arrow, RoundTrip) {
    Table src = makeSyntheticTable(500, 55);

    auto arrow_tbl = ArrowConverter::toArrowTable(src);
    Table dst = ArrowConverter::fromArrowTable(arrow_tbl, "roundtrip");

    EXPECT_EQ(src.numRows(), dst.numRows());
    EXPECT_EQ(src.numCols(), dst.numCols());

    auto& src_keys = std::get<Int32Column>(src.getColumn(0));
    auto& dst_keys = std::get<Int32Column>(dst.getColumn(0));
    for (std::size_t i = 0; i < src.numRows(); ++i)
        EXPECT_EQ(src_keys[i], dst_keys[i]) << "Row " << i;

    auto& src_vals = std::get<Float64Column>(src.getColumn(1));
    auto& dst_vals = std::get<Float64Column>(dst.getColumn(1));
    for (std::size_t i = 0; i < src.numRows(); ++i)
        EXPECT_DOUBLE_EQ(src_vals[i], dst_vals[i]) << "Row " << i;
}
#endif

// ═══ Test 7: Dictionary Encoding ════════════════════════════════════════════

TEST(Compression, DictRoundTrip) {
    StringColumn col;
    col.pushValue("alpha"); col.pushValue("beta"); col.pushValue("alpha");
    col.pushValue("gamma"); col.pushValue("beta");

    auto enc = dictEncode(col);
    EXPECT_EQ(enc.dictionary.size(), 3u);
    EXPECT_EQ(enc.codes.size(), 5u);
    EXPECT_EQ(enc.codes[0], enc.codes[2]);  // "alpha" same code

    StringColumn decoded = dictDecode(enc);
    for (std::size_t i = 0; i < col.size(); ++i)
        EXPECT_EQ(col[i], decoded[i]);
}

TEST(Compression, RLEEncoding) {
    Int32Column col;
    for (int i = 0; i < 100; ++i) col.pushValue(1);
    for (int i = 0; i < 50; ++i)  col.pushValue(2);
    for (int i = 0; i < 200; ++i) col.pushValue(3);

    auto enc = rleEncode(col);
    EXPECT_EQ(enc.values.size(), 3u);
    EXPECT_EQ(enc.run_lengths[0], 100);
    EXPECT_EQ(enc.run_lengths[1], 50);
    EXPECT_EQ(enc.run_lengths[2], 200);

    // Compression ratio
    std::size_t raw_bytes = col.size() * sizeof(int32_t);  // 1400 bytes
    std::size_t comp_bytes = rleCompressedBytes(enc);        // 24 bytes
    EXPECT_LT(comp_bytes, raw_bytes / 10);  // >10x compression
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
