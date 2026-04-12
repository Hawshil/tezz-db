/**
 * @file tick_feed_demo.cpp
 * @brief Self-contained smoke test simulating a live tick feed with
 *        concurrent writer (synthetic tick generator) and reader
 *        (SQL query executor computing SMA(price, 20)).
 *
 * CPU-only — no CUDA required.
 */
#include "core/ring_table.h"
#include "core/schema.h"
#include "query/operator_node.h"
#include "query/planner.h"
#include "sql/lexer.h"
#include "sql/parser.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

using namespace gpudb;

int main() {
    std::printf("\n  ╔═══════════════════════════════════════════════════╗\n");
    std::printf("  ║  GPUDB — Live Tick Feed Smoke Test               ║\n");
    std::printf("  ╚═══════════════════════════════════════════════════╝\n\n");

    // 1. Create a RingTable for tick data (ts, price, volume).
    RingTable ticks("ticks",
        {"ts", "price", "volume"},
        {DataType::INT64, DataType::FLOAT64, DataType::INT64},
        /*chunk_capacity=*/ 50'000,
        /*max_rows=*/       100'000);

    // 2. Writer thread: generate synthetic ticks for 5 seconds.
    std::atomic<bool> stop{false};
    std::thread writer([&]() {
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> price_dist(99.0, 101.0);
        std::uniform_int_distribution<std::int64_t> vol_dist(100, 10'000);
        std::int64_t ts = 0;

        while (!stop) {
            constexpr int BATCH = 1'000;

            std::vector<std::int64_t> ts_data(BATCH);
            std::vector<double>       price_data(BATCH);
            std::vector<std::int64_t> vol_data(BATCH);

            for (int i = 0; i < BATCH; ++i) {
                ts_data[i]    = ts + i;
                price_data[i] = price_dist(rng);
                vol_data[i]   = vol_dist(rng);
            }
            ts += BATCH;

            std::vector<std::vector<std::byte>> batches(3);
            batches[0].resize(BATCH * sizeof(std::int64_t));
            std::memcpy(batches[0].data(), ts_data.data(), batches[0].size());
            batches[1].resize(BATCH * sizeof(double));
            std::memcpy(batches[1].data(), price_data.data(), batches[1].size());
            batches[2].resize(BATCH * sizeof(std::int64_t));
            std::memcpy(batches[2].data(), vol_data.data(), batches[2].size());

            ticks.appendBatch(batches, BATCH);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });

    // 3. Query thread: every 200 ms, snapshot the ring table and run
    //    SELECT SMA(price, 20) OVER (ORDER BY ts ROWS 20 PRECEDING) FROM ticks LIMIT 5
    int sma_count = 0;
    std::thread querier([&]() {
        for (int i = 0; i < 25; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));

            // snapshot() is safe: appendBatch holds a mutex, so reads
            // always see a complete batch boundary.
            Table snap = ticks.snapshot();
            if (snap.numRows() < 20) {
                std::printf("  [%2d] too few rows (%zu), skipping\n",
                            i, snap.numRows());
                continue;
            }

            // Register snapshot in a catalog.
            ExecutionContext ctx;
            ctx.catalog["ticks"] = &snap;

            // Infer schema from snapshot.
            Schema sch = Schema::fromTable(snap);

            // Parse the query.
            Lexer lexer;
            auto tokens = lexer.tokenize(
                "SELECT SMA(price, 20) OVER (ORDER BY ts ROWS 20 PRECEDING) "
                "FROM ticks LIMIT 5");
            Parser parser;
            auto stmt = parser.parse(tokens);

            // Plan and execute.
            QueryPlanner planner;
            auto plan = planner.plan(stmt, sch);
            auto result = plan->execute(ctx);

            if (result.numRows() > 0) {
                // The window function appends a new column at the end.
                auto& col = std::get<Float64Column>(
                    result.getColumn(result.numCols() - 1));
                double sma_val = col[col.size() - 1];
                std::printf("  [%2d] sma20 = %.4f  (ring rows: %zu)\n",
                            i, sma_val, snap.numRows());
                ++sma_count;
            }
        }
    });

    // Let both threads run for 5 seconds.
    std::this_thread::sleep_for(std::chrono::seconds(5));
    stop = true;
    writer.join();
    querier.join();

    std::size_t final_size = ticks.numRows();
    std::printf("\n  Final ring size : %zu rows\n", final_size);
    std::printf("  SMA20 outputs  : %d\n", sma_count);
    std::printf("  Committed rows : %llu\n",
                static_cast<unsigned long long>(ticks.committedRows()));

    // Validate acceptance criteria.
    bool pass = true;
    if (sma_count < 10) {
        std::printf("  FAIL: expected >= 10 SMA outputs, got %d\n", sma_count);
        pass = false;
    }
    if (final_size > 100'000) {
        std::printf("  FAIL: final ring size exceeds max_rows (100K)\n");
        pass = false;
    }

    std::printf("\n  %s\n",
                pass ? "Smoke test PASSED" : "Smoke test FAILED");
    return pass ? 0 : 1;
}
