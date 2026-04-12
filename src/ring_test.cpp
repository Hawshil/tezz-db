/**
 * @file ring_test.cpp
 * @brief Quick compile + correctness smoke test for RingColumn and RingTable.
 */
#include "core/ring_column.h"
#include "core/ring_table.h"
#include "io/arrow_stream_reader.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    using namespace gpudb;

    // ── RingColumn acceptance test ──────────────────────────────────────────
    {
        RingColumn<double> col(/*chunk_cap=*/3, /*max_rows=*/9);
        for (int i = 1; i <= 12; ++i)
            col.push(static_cast<double>(i));

        assert(col.size() == 9);
        // Oldest 3 values (1,2,3) should have been evicted.
        assert(col[0] == 4.0);
        assert(col[8] == 12.0);

        auto vec = col.toVector();
        assert(vec.size() == 9);
        assert(vec[0] == 4.0);
        assert(vec[8] == 12.0);

        auto snap = col.snapshot();
        assert(snap.size() == 9);
        assert(snap[0] == 4.0);
        assert(snap[8] == 12.0);

        std::printf("  [PASS] RingColumn<double> basic push/evict/snapshot\n");
    }

    // ── pushBatch == push() equivalence ─────────────────────────────────────
    {
        RingColumn<double> col_a(/*chunk_cap=*/4, /*max_rows=*/0);
        RingColumn<double> col_b(/*chunk_cap=*/4, /*max_rows=*/0);

        double data[10] = {1,2,3,4,5,6,7,8,9,10};
        col_a.pushBatch(data, 10);
        for (int i = 0; i < 10; ++i)
            col_b.push(data[i]);

        assert(col_a.size() == col_b.size());
        for (std::size_t i = 0; i < col_a.size(); ++i)
            assert(col_a[i] == col_b[i]);

        std::printf("  [PASS] pushBatch == push equivalence\n");
    }

    // ── RingTable acceptance test ───────────────────────────────────────────
    {
        RingTable rt("ticks",
                     {"ts", "price"},
                     {DataType::INT64, DataType::FLOAT64},
                     /*chunk_cap=*/100'000,
                     /*max_rows=*/0);

        const std::size_t batch_rows = 100'000;
        for (int b = 0; b < 5; ++b) {
            std::vector<std::int64_t> ts(batch_rows);
            std::vector<double> prices(batch_rows);
            for (std::size_t i = 0; i < batch_rows; ++i) {
                ts[i] = static_cast<std::int64_t>(b * batch_rows + i);
                prices[i] = 100.0 + i * 0.01;
            }

            std::vector<std::vector<std::byte>> batches(2);
            batches[0].resize(batch_rows * sizeof(std::int64_t));
            std::memcpy(batches[0].data(), ts.data(),
                        batch_rows * sizeof(std::int64_t));
            batches[1].resize(batch_rows * sizeof(double));
            std::memcpy(batches[1].data(), prices.data(),
                        batch_rows * sizeof(double));

            rt.appendBatch(batches, batch_rows);
        }

        assert(rt.numRows() == 500'000);
        Table snap = rt.snapshot();
        assert(snap.numRows() == 500'000);

        std::printf("  [PASS] RingTable 5 batches x 100K rows, snapshot\n");
    }

    // ── ArrowStreamReader compiles with GPUDB_HAS_ARROW=0 ──────────────────
    {
        RingTable rt2("test", {"a"}, {DataType::INT32});
        bool threw = false;
        try {
            ArrowStreamReader::readFile("nonexistent.ipc", rt2);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        assert(threw);
        std::printf("  [PASS] ArrowStreamReader::readFile throws without Arrow\n");
    }

    std::printf("\n  All ring-buffer smoke tests passed.\n");
    return 0;
}
