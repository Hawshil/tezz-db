/**
 * @file ring_buffer_test.cpp
 * @brief Google Test cases for concurrent RingTable access:
 *        1. ConcurrentAppendAndRead
 *        2. SeqLockNeverReadsPartialBatch
 *        3. AppendMutexExcludesMultipleWriters
 */
#include <gtest/gtest.h>

#include "core/ring_table.h"
#include "core/schema.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

using namespace gpudb;

// ── Helper: build a batch of (int64, double) rows ───────────────────────────
static std::vector<std::vector<std::byte>>
makeBatch(std::size_t row_count, std::int64_t ts_start, double price) {
    std::vector<std::int64_t> ts(row_count);
    std::vector<double>       prices(row_count);
    for (std::size_t i = 0; i < row_count; ++i) {
        ts[i]     = ts_start + static_cast<std::int64_t>(i);
        prices[i] = price + i * 0.001;
    }
    std::vector<std::vector<std::byte>> batches(2);
    batches[0].resize(row_count * sizeof(std::int64_t));
    std::memcpy(batches[0].data(), ts.data(), batches[0].size());
    batches[1].resize(row_count * sizeof(double));
    std::memcpy(batches[1].data(), prices.data(), batches[1].size());
    return batches;
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 1: ConcurrentAppendAndRead
// ─────────────────────────────────────────────────────────────────────────────
TEST(RingTable, ConcurrentAppendAndRead) {
    constexpr int NUM_BATCHES  = 10;
    constexpr int BATCH_SIZE   = 50'000;

    RingTable rt("ticks", {"ts", "price"},
                 {DataType::INT64, DataType::FLOAT64},
                 /*chunk_cap=*/ 100'000, /*max_rows=*/ 0);

    // Writer thread: append NUM_BATCHES batches of BATCH_SIZE rows.
    std::thread writer([&]() {
        for (int b = 0; b < NUM_BATCHES; ++b) {
            auto batch = makeBatch(BATCH_SIZE,
                                   static_cast<std::int64_t>(b) * BATCH_SIZE,
                                   100.0);
            rt.appendBatch(batch, BATCH_SIZE);
        }
    });

    // Reader thread: take snapshots while writer is running.
    std::vector<std::size_t> snapshot_sizes;
    std::thread reader([&]() {
        for (int i = 0; i < 20; ++i) {
            auto snap = rt.trySnapshot();
            if (snap.valid)
                snapshot_sizes.push_back(snap.data.numRows());
            std::this_thread::yield();
        }
    });

    writer.join();
    reader.join();

    // After writer finishes, committedRows must equal total.
    EXPECT_EQ(rt.committedRows(),
              static_cast<std::uint64_t>(NUM_BATCHES) * BATCH_SIZE);

    // Every valid snapshot must have a row count that is a multiple of BATCH_SIZE.
    for (auto sz : snapshot_sizes) {
        EXPECT_EQ(sz % BATCH_SIZE, 0u)
            << "Snapshot had " << sz << " rows (not a multiple of "
            << BATCH_SIZE << ")";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 2: SeqLockNeverReadsPartialBatch
// ─────────────────────────────────────────────────────────────────────────────
TEST(RingTable, SeqLockNeverReadsPartialBatch) {
    constexpr int NUM_BATCHES = 10;
    constexpr int BATCH_SIZE  = 50'000;

    RingTable rt("ticks", {"ts", "price"},
                 {DataType::INT64, DataType::FLOAT64},
                 /*chunk_cap=*/ 100'000, /*max_rows=*/ 0);

    std::vector<std::size_t> valid_sizes;
    std::mutex sizes_mtx;

    std::thread writer([&]() {
        for (int b = 0; b < NUM_BATCHES; ++b) {
            auto batch = makeBatch(BATCH_SIZE,
                                   static_cast<std::int64_t>(b) * BATCH_SIZE,
                                   100.0);
            rt.appendBatch(batch, BATCH_SIZE);
        }
    });

    std::thread reader([&]() {
        for (int i = 0; i < 50; ++i) {
            auto snap = rt.trySnapshot();
            if (snap.valid) {
                std::lock_guard<std::mutex> lock(sizes_mtx);
                valid_sizes.push_back(snap.data.numRows());
            }
            std::this_thread::yield();
        }
    });

    writer.join();
    reader.join();

    // All valid snapshots must show whole-batch multiples.
    for (auto sz : valid_sizes) {
        EXPECT_EQ(sz % BATCH_SIZE, 0u)
            << "Valid snapshot had " << sz << " rows";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 3: AppendMutexExcludesMultipleWriters
// ─────────────────────────────────────────────────────────────────────────────
TEST(RingTable, AppendMutexExcludesMultipleWriters) {
    constexpr int NUM_WRITERS      = 4;
    constexpr int ROWS_PER_WRITER  = 100'000;
    constexpr int BATCH_SIZE       = 10'000;

    RingTable rt("ticks", {"ts", "price"},
                 {DataType::INT64, DataType::FLOAT64},
                 /*chunk_cap=*/ 100'000, /*max_rows=*/ 0);

    std::vector<std::thread> writers;
    for (int w = 0; w < NUM_WRITERS; ++w) {
        writers.emplace_back([&, w]() {
            for (int b = 0; b < ROWS_PER_WRITER / BATCH_SIZE; ++b) {
                auto batch = makeBatch(
                    BATCH_SIZE,
                    static_cast<std::int64_t>(w * ROWS_PER_WRITER + b * BATCH_SIZE),
                    100.0 + w);
                rt.appendBatch(batch, BATCH_SIZE);
            }
        });
    }

    for (auto& t : writers) t.join();

    EXPECT_EQ(rt.committedRows(),
              static_cast<std::uint64_t>(NUM_WRITERS) * ROWS_PER_WRITER);
}
