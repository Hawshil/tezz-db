/**
 * @file finance_test.cpp
 * @brief Google Test — financial domain integration tests.
 *
 * Tests window functions (SMA, EMA, RollingStd), ring buffer snapshot,
 * and ASOF join correctness using synthetic market data.
 * CPU-only — no CUDA dependency.
 */
#include <gtest/gtest.h>
#include "core/ring_table.h"
#include "query/operator_node.h"
#include "query/planner.h"
#include "sql/lexer.h"
#include "sql/parser.h"
#include <cmath>
#include <cstring>
#include <numeric>

using namespace gpudb;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Build a tick table with synthetic price data. */
static Table makeTickTable(int n, double base_price = 100.0) {
    Int64Column ts;
    Float64Column price;
    Int32Column symbol;
    for (int i = 0; i < n; ++i) {
        ts.pushValue(static_cast<std::int64_t>(i));
        price.pushValue(base_price + std::sin(i / 10.0) * 5.0);
        symbol.pushValue(0);
    }
    std::vector<std::string> names = {"ts", "price", "symbol"};
    std::vector<ColumnVariant> cols;
    cols.emplace_back(std::move(ts));
    cols.emplace_back(std::move(price));
    cols.emplace_back(std::move(symbol));
    return Table("ticks", std::move(names), std::move(cols));
}

/** Build a quote table at every-other-tick frequency. */
static Table makeQuoteTable(int n) {
    Int64Column ts;
    Float64Column bid, ask;
    Int32Column symbol;
    for (int i = 0; i < n; ++i) {
        ts.pushValue(static_cast<std::int64_t>(i * 2));
        bid.pushValue(99.5 + i * 0.01);
        ask.pushValue(100.5 + i * 0.01);
        symbol.pushValue(0);
    }
    std::vector<std::string> names = {"ts", "bid", "ask", "symbol"};
    std::vector<ColumnVariant> cols;
    cols.emplace_back(std::move(ts));
    cols.emplace_back(std::move(bid));
    cols.emplace_back(std::move(ask));
    cols.emplace_back(std::move(symbol));
    return Table("quotes", std::move(names), std::move(cols));
}

// ─────────────────────────────────────────────────────────────────────────────
// Window function tests
// ─────────────────────────────────────────────────────────────────────────────

TEST(FinanceWindow, SMACorrectness) {
    Table ticks = makeTickTable(100);

    WindowNode::WindowSpec spec{"SMA", "price", "ts", "sma_20", 20};
    WindowNode node(spec);
    auto scan = std::make_unique<ScanNode>("ticks",
        std::vector<std::string>{});
    node.setInput(std::move(scan));

    ExecutionContext ctx;
    ctx.catalog["ticks"] = &ticks;
    Table result = node.execute(ctx);

    // sma_20 column must exist
    ASSERT_TRUE(result.hasColumn("sma_20"));
    auto& col = std::get<Float64Column>(
        result.getColumn(*result.findColumnIndex("sma_20")));

    // SMA[i] must be within the price range [base - amplitude, base + amplitude]
    for (std::size_t i = 0; i < col.size(); ++i) {
        EXPECT_GE(col[i], 90.0) << "Row " << i;
        EXPECT_LE(col[i], 110.0) << "Row " << i;
    }
}

TEST(FinanceWindow, EMAConverges) {
    // Flat price = 100 → EMA should converge to 100
    Int64Column ts;
    Float64Column price;
    for (int i = 0; i < 200; ++i) {
        ts.pushValue(i);
        price.pushValue(100.0);
    }
    std::vector<std::string> names = {"ts", "price"};
    std::vector<ColumnVariant> cols;
    cols.emplace_back(std::move(ts));
    cols.emplace_back(std::move(price));
    Table t("ticks", std::move(names), std::move(cols));

    WindowNode::WindowSpec spec{"EMA", "price", "ts", "ema5", 5};
    WindowNode node(spec);
    auto scan = std::make_unique<ScanNode>("ticks",
        std::vector<std::string>{});
    node.setInput(std::move(scan));

    ExecutionContext ctx;
    ctx.catalog["ticks"] = &t;
    Table result = node.execute(ctx);

    auto& col = std::get<Float64Column>(
        result.getColumn(*result.findColumnIndex("ema5")));

    // Last value should be within 0.01 of 100
    EXPECT_NEAR(col[col.size() - 1], 100.0, 0.01);
}

TEST(FinanceWindow, RollingStdFlatReturnsZero) {
    Int64Column ts;
    Float64Column price;
    for (int i = 0; i < 50; ++i) {
        ts.pushValue(i);
        price.pushValue(50.0);
    }
    std::vector<std::string> names = {"ts", "price"};
    std::vector<ColumnVariant> cols;
    cols.emplace_back(std::move(ts));
    cols.emplace_back(std::move(price));
    Table t("ticks", std::move(names), std::move(cols));

    WindowNode::WindowSpec spec{"ROLLING_STD", "price", "ts", "std10", 10};
    WindowNode node(spec);
    auto scan = std::make_unique<ScanNode>("ticks",
        std::vector<std::string>{});
    node.setInput(std::move(scan));

    ExecutionContext ctx;
    ctx.catalog["ticks"] = &t;
    Table result = node.execute(ctx);

    auto& col = std::get<Float64Column>(
        result.getColumn(*result.findColumnIndex("std10")));

    // Flat series → rolling stddev should be 0 once window is full
    for (std::size_t i = 9; i < col.size(); ++i)
        EXPECT_NEAR(col[i], 0.0, 1e-9) << "Row " << i;
}

// ─────────────────────────────────────────────────────────────────────────────
// Ring buffer tests
// ─────────────────────────────────────────────────────────────────────────────

TEST(FinanceRing, AppendAndSnapshot) {
    RingTable rt("ticks", {"ts", "price"},
        {DataType::INT64, DataType::FLOAT64},
        /*chunk_cap=*/10'000, /*max=*/100'000);

    for (int b = 0; b < 20; ++b) {
        std::vector<std::int64_t> ts(5000);
        std::vector<double> px(5000, 100.0 + b);
        std::iota(ts.begin(), ts.end(), static_cast<std::int64_t>(b) * 5000);

        std::vector<std::vector<std::byte>> batches(2);
        batches[0].resize(5000 * sizeof(std::int64_t));
        batches[1].resize(5000 * sizeof(double));
        std::memcpy(batches[0].data(), ts.data(), 5000 * sizeof(std::int64_t));
        std::memcpy(batches[1].data(), px.data(), 5000 * sizeof(double));
        rt.appendBatch(batches, 5000);
    }

    // max_rows enforced: 20 * 5000 = 100K total, but max_rows = 100K
    EXPECT_LE(rt.numRows(), 100'000u);

    auto snap = rt.trySnapshot();
    EXPECT_TRUE(snap.valid);
    EXPECT_LE(snap.data.numRows(), 100'000u);
    EXPECT_GT(snap.data.numRows(), 0u);
}

// ─────────────────────────────────────────────────────────────────────────────
// ASOF join tests
// ─────────────────────────────────────────────────────────────────────────────

TEST(FinanceAsof, BasicMatch) {
    Table trades = makeTickTable(10);    // ts 0..9
    Table quotes = makeQuoteTable(5);    // ts 0,2,4,6,8

    AsofJoinNode::Spec sp{
        "trades", "quotes", "symbol", "symbol", "ts", "ts", 0};
    AsofJoinNode node(sp);

    ExecutionContext ctx;
    ctx.catalog["trades"] = &trades;
    ctx.catalog["quotes"] = &quotes;
    Table result = node.execute(ctx);

    EXPECT_EQ(result.numRows(), 10u);

    // trade ts=1 → nearest quote ts=0
    // trade ts=2 → nearest quote ts=2
    EXPECT_TRUE(result.hasColumn("quotes.bid"));
    EXPECT_TRUE(result.hasColumn("quotes.ask"));
    EXPECT_TRUE(result.hasColumn("trades.ts"));
}

TEST(FinanceAsof, NoMatchReturnsNull) {
    // trades with ts=5, quotes all have ts > 5
    Int64Column tts;
    tts.pushValue(5LL);
    Float64Column tpx;
    tpx.pushValue(100.0);
    Int32Column tsym;
    tsym.pushValue(0);
    std::vector<std::string> tn = {"ts", "price", "symbol"};
    std::vector<ColumnVariant> tc;
    tc.emplace_back(std::move(tts));
    tc.emplace_back(std::move(tpx));
    tc.emplace_back(std::move(tsym));
    Table trades("trades", std::move(tn), std::move(tc));

    Int64Column qts;
    qts.pushValue(10LL);
    Float64Column qbid;
    qbid.pushValue(99.5);
    Int32Column qsym;
    qsym.pushValue(0);
    std::vector<std::string> qn = {"ts", "bid", "symbol"};
    std::vector<ColumnVariant> qc;
    qc.emplace_back(std::move(qts));
    qc.emplace_back(std::move(qbid));
    qc.emplace_back(std::move(qsym));
    Table quotes("quotes", std::move(qn), std::move(qc));

    AsofJoinNode::Spec sp{
        "trades", "quotes", "symbol", "symbol", "ts", "ts", 0};
    AsofJoinNode node(sp);

    ExecutionContext ctx;
    ctx.catalog["trades"] = &trades;
    ctx.catalog["quotes"] = &quotes;
    Table result = node.execute(ctx);

    EXPECT_EQ(result.numRows(), 1u);

    // quotes.bid column exists but row 0 is NULL (no match)
    auto bid_idx = result.findColumnIndex("quotes.bid");
    ASSERT_TRUE(bid_idx.has_value());
    auto& bid_col = std::get<Float64Column>(result.getColumn(*bid_idx));
    EXPECT_TRUE(bid_col.isNull(0));
}
