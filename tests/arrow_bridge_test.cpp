/**
 * @file arrow_bridge_test.cpp
 * @brief Google Test — Arrow columnar conversion round-trip.
 *
 * Verifies that Table → ArrowTable → Table preserves all values
 * across INT32, FLOAT64, and STRING column types.
 */
#include <gtest/gtest.h>
#include "core/table.h"
#include "core/column.h"

#ifdef GPUDB_HAS_ARROW
#include "io/arrow_bridge.h"
#endif

using namespace gpudb;

// ── Helper: build a synthetic table ─────────────────────────────────────────

static Table makeMockTable(int n) {
    Int32Column keys;
    Float64Column vals;
    StringColumn labels;
    keys.reserve(n); vals.reserve(n); labels.reserve(n);
    const char* groups[] = {"alpha","beta","gamma","delta","epsilon"};
    for (int i = 0; i < n; ++i) {
        keys.pushValue(i * 7 % 100);
        vals.pushValue(i * 3.14159);
        labels.pushValue(groups[i % 5]);
    }
    std::vector<std::string> names = {"id", "value", "label"};
    std::vector<ColumnVariant> cols;
    cols.emplace_back(std::move(keys));
    cols.emplace_back(std::move(vals));
    cols.emplace_back(std::move(labels));
    return Table("mock", std::move(names), std::move(cols));
}

// ── Round-trip: Table → Arrow → Table ───────────────────────────────────────

#ifdef GPUDB_HAS_ARROW
TEST(ArrowBridge, RoundTripPreservesShape) {
    Table src = makeMockTable(1000);
    auto arrow_tbl = ArrowConverter::toArrowTable(src);
    Table dst = ArrowConverter::fromArrowTable(arrow_tbl, "round");

    EXPECT_EQ(src.numRows(), dst.numRows());
    EXPECT_EQ(src.numCols(), dst.numCols());
}

TEST(ArrowBridge, RoundTripPreservesInt32) {
    Table src = makeMockTable(500);
    auto arrow_tbl = ArrowConverter::toArrowTable(src);
    Table dst = ArrowConverter::fromArrowTable(arrow_tbl, "round");

    auto& sk = std::get<Int32Column>(src.getColumn(0));
    auto& dk = std::get<Int32Column>(dst.getColumn(0));
    for (std::size_t i = 0; i < src.numRows(); ++i)
        EXPECT_EQ(sk[i], dk[i]) << "Row " << i;
}

TEST(ArrowBridge, RoundTripPreservesFloat64) {
    Table src = makeMockTable(500);
    auto arrow_tbl = ArrowConverter::toArrowTable(src);
    Table dst = ArrowConverter::fromArrowTable(arrow_tbl, "round");

    auto& sv = std::get<Float64Column>(src.getColumn(1));
    auto& dv = std::get<Float64Column>(dst.getColumn(1));
    for (std::size_t i = 0; i < src.numRows(); ++i)
        EXPECT_DOUBLE_EQ(sv[i], dv[i]) << "Row " << i;
}

TEST(ArrowBridge, RoundTripPreservesStrings) {
    Table src = makeMockTable(500);
    auto arrow_tbl = ArrowConverter::toArrowTable(src);
    Table dst = ArrowConverter::fromArrowTable(arrow_tbl, "round");

    auto& sl = std::get<StringColumn>(src.getColumn(2));
    auto& dl = std::get<StringColumn>(dst.getColumn(2));
    for (std::size_t i = 0; i < src.numRows(); ++i)
        EXPECT_EQ(sl[i], dl[i]) << "Row " << i;
}
#endif

// ── Table construction (always runs, no Arrow dep) ──────────────────────────

TEST(ArrowBridge, MockTableConstruction) {
    Table t = makeMockTable(100);
    EXPECT_EQ(t.numRows(), 100u);
    EXPECT_EQ(t.numCols(), 3u);
    EXPECT_EQ(t.columnNames()[0], "id");
    EXPECT_EQ(t.columnNames()[1], "value");
    EXPECT_EQ(t.columnNames()[2], "label");
}
