/**
 * @file asof_join_test.cpp
 * @brief Standalone smoke test for ASOF JOIN:
 *   1. Parser: verify AsofJoinStmt fields from SQL.
 *   2. AsofJoinNode: verify cpu join correctness on synthetic data.
 *   3. Edge case: left_ts < all right_ts → unmatched row.
 */
#include "sql/lexer.h"
#include "sql/parser.h"
#include "query/operator_node.h"
#include "query/planner.h"
#include "core/schema.h"

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>

using namespace gpudb;

static int tests_passed = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);      \
            return 1;                                                           \
        }                                                                       \
    } while (0)

#define PASS(msg)                                                              \
    do {                                                                        \
        std::printf("  [PASS] %s\n", msg);                                     \
        ++tests_passed;                                                         \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Parser extracts correct fields from ASOF JOIN SQL
// ─────────────────────────────────────────────────────────────────────────────
int test_parser_asof_join() {
    Lexer lex;
    auto tokens = lex.tokenize(
        "SELECT t.ts, t.price, q.bid, q.ask "
        "FROM trades AS t "
        "ASOF JOIN quotes AS q "
        "ON t.symbol = q.symbol "
        "AS OF t.ts >= q.ts");
    Parser parser;
    auto stmt = parser.parseAsofJoin(tokens);

    CHECK(stmt.left_table == "trades",    "left_table == trades");
    CHECK(stmt.left_alias == "t",         "left_alias == t");
    CHECK(stmt.right_table == "quotes",   "right_table == quotes");
    CHECK(stmt.right_alias == "q",        "right_alias == q");
    CHECK(stmt.left_key_col == "symbol",  "left_key == symbol");
    CHECK(stmt.right_key_col == "symbol", "right_key == symbol");
    CHECK(stmt.left_ts_col == "ts",       "left_ts == ts");
    CHECK(stmt.right_ts_col == "ts",      "right_ts == ts");
    CHECK(stmt.tolerance_ns == 0,         "tolerance == 0");
    CHECK(stmt.left_cols.size() == 2,     "left_cols.size == 2");
    CHECK(stmt.right_cols.size() == 2,    "right_cols.size == 2");

    PASS("Parser ASOF JOIN");
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: AsofJoinNode CPU join on synthetic trades/quotes
// ─────────────────────────────────────────────────────────────────────────────
int test_asof_join_node() {
    // Build trades: ts=[100,200,300], symbol=[0,0,0], price=[1,2,3]
    Table trades("trades");
    {
        Int64Column ts;   ts.pushValue(100); ts.pushValue(200); ts.pushValue(300);
        Int32Column sym;  sym.pushValue(0);  sym.pushValue(0);  sym.pushValue(0);
        Float64Column pr; pr.pushValue(1.0); pr.pushValue(2.0); pr.pushValue(3.0);
        trades.addColumn("ts",     std::move(ts));
        trades.addColumn("symbol", std::move(sym));
        trades.addColumn("price",  std::move(pr));
    }

    // Build quotes: ts=[50,150,250], symbol=[0,0,0], bid=[0.9,1.9,2.9]
    Table quotes("quotes");
    {
        Int64Column ts;   ts.pushValue(50); ts.pushValue(150); ts.pushValue(250);
        Int32Column sym;  sym.pushValue(0);  sym.pushValue(0);  sym.pushValue(0);
        Float64Column bd; bd.pushValue(0.9); bd.pushValue(1.9); bd.pushValue(2.9);
        quotes.addColumn("ts",     std::move(ts));
        quotes.addColumn("symbol", std::move(sym));
        quotes.addColumn("bid",    std::move(bd));
    }

    ExecutionContext ctx;
    ctx.catalog["trades"] = &trades;
    ctx.catalog["quotes"] = &quotes;

    AsofJoinNode::Spec spec;
    spec.left_table   = "trades";
    spec.right_table  = "quotes";
    spec.left_key     = "symbol";
    spec.right_key    = "symbol";
    spec.left_ts_col  = "ts";
    spec.right_ts_col = "ts";
    spec.tolerance_ns = 0;

    AsofJoinNode node(spec);
    Table result = node.execute(ctx);

    CHECK(result.numRows() == 3, "3 output rows");

    // Verify quotes.bid column = {0.9, 1.9, 2.9}
    auto bid_idx = result.findColumnIndex("quotes.bid");
    CHECK(bid_idx.has_value(), "quotes.bid column exists");

    const auto& bid_col = std::get<Float64Column>(result.getColumn(*bid_idx));
    CHECK(std::abs(bid_col[0] - 0.9) < 1e-9, "bid[0] == 0.9");
    CHECK(std::abs(bid_col[1] - 1.9) < 1e-9, "bid[1] == 1.9");
    CHECK(std::abs(bid_col[2] - 2.9) < 1e-9, "bid[2] == 2.9");

    PASS("AsofJoinNode CPU join correctness");
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Unmatched row (left_ts < all right_ts)
// ─────────────────────────────────────────────────────────────────────────────
int test_asof_join_unmatched() {
    Table trades("trades");
    {
        Int64Column ts; ts.pushValue(40);
        Int32Column sym; sym.pushValue(0);
        Float64Column pr; pr.pushValue(1.0);
        trades.addColumn("ts", std::move(ts));
        trades.addColumn("symbol", std::move(sym));
        trades.addColumn("price", std::move(pr));
    }

    Table quotes("quotes");
    {
        Int64Column ts; ts.pushValue(50); ts.pushValue(150);
        Int32Column sym; sym.pushValue(0); sym.pushValue(0);
        Float64Column bd; bd.pushValue(0.9); bd.pushValue(1.9);
        quotes.addColumn("ts", std::move(ts));
        quotes.addColumn("symbol", std::move(sym));
        quotes.addColumn("bid", std::move(bd));
    }

    ExecutionContext ctx;
    ctx.catalog["trades"] = &trades;
    ctx.catalog["quotes"] = &quotes;

    AsofJoinNode::Spec spec;
    spec.left_table   = "trades";
    spec.right_table  = "quotes";
    spec.left_key     = "symbol";
    spec.right_key    = "symbol";
    spec.left_ts_col  = "ts";
    spec.right_ts_col = "ts";

    AsofJoinNode node(spec);
    Table result = node.execute(ctx);

    CHECK(result.numRows() == 1, "1 output row");

    // Unmatched → default value (0.0 for double)
    auto bid_idx = result.findColumnIndex("quotes.bid");
    CHECK(bid_idx.has_value(), "quotes.bid column exists");
    const auto& bid_col = std::get<Float64Column>(result.getColumn(*bid_idx));
    CHECK(std::abs(bid_col[0] - 0.0) < 1e-9, "bid[0] == 0.0 (unmatched)");

    PASS("AsofJoinNode unmatched row");
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Tolerance filtering
// ─────────────────────────────────────────────────────────────────────────────
int test_asof_join_tolerance() {
    Table trades("trades");
    {
        Int64Column ts; ts.pushValue(200);
        trades.addColumn("ts", std::move(ts));
    }

    Table quotes("quotes");
    {
        Int64Column ts; ts.pushValue(50); ts.pushValue(150);
        quotes.addColumn("ts", std::move(ts));
    }

    ExecutionContext ctx;
    ctx.catalog["trades"] = &trades;
    ctx.catalog["quotes"] = &quotes;

    // Tolerance = 60: gap(200-150)=50 <= 60 → match index 1
    {
        AsofJoinNode::Spec spec;
        spec.left_table   = "trades";
        spec.right_table  = "quotes";
        spec.left_ts_col  = "ts";
        spec.right_ts_col = "ts";
        spec.tolerance_ns = 60;

        AsofJoinNode node(spec);
        Table result = node.execute(ctx);
        CHECK(result.numRows() == 1, "1 output row");

        // Matched: right.ts = 150
        auto rt_idx = result.findColumnIndex("quotes.ts");
        CHECK(rt_idx.has_value(), "quotes.ts column exists");
        const auto& rt_col = std::get<Int64Column>(result.getColumn(*rt_idx));
        CHECK(rt_col[0] == 150, "matched quotes.ts == 150");
    }

    // Tolerance = 10: gap(200-150)=50 > 10 → no match
    {
        AsofJoinNode::Spec spec;
        spec.left_table   = "trades";
        spec.right_table  = "quotes";
        spec.left_ts_col  = "ts";
        spec.right_ts_col = "ts";
        spec.tolerance_ns = 10;

        AsofJoinNode node(spec);
        Table result = node.execute(ctx);
        auto rt_idx = result.findColumnIndex("quotes.ts");
        CHECK(rt_idx.has_value(), "quotes.ts column exists");
        const auto& rt_col = std::get<Int64Column>(result.getColumn(*rt_idx));
        CHECK(rt_col[0] == 0, "unmatched: quotes.ts == 0 (default)");
    }

    PASS("AsofJoinNode tolerance filtering");
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: planAsofJoin smoke test
// ─────────────────────────────────────────────────────────────────────────────
int test_plan_asof_join() {
    Lexer lex;
    auto tokens = lex.tokenize(
        "SELECT t.ts, t.price, q.bid "
        "FROM trades AS t "
        "ASOF JOIN quotes AS q "
        "ON t.symbol = q.symbol "
        "AS OF t.ts >= q.ts");
    Parser parser;
    auto stmt = parser.parseAsofJoin(tokens);

    Schema trades_schema({
        ColumnDef("ts", DataType::INT64),
        ColumnDef("symbol", DataType::INT32),
        ColumnDef("price", DataType::FLOAT64)
    });

    QueryPlanner planner;
    auto plan = planner.planAsofJoin(stmt, trades_schema);
    CHECK(plan != nullptr, "planAsofJoin returns non-null");

    PASS("planAsofJoin smoke");
    return 0;
}

int main() {
    int rc = 0;
    rc |= test_parser_asof_join();
    rc |= test_asof_join_node();
    rc |= test_asof_join_unmatched();
    rc |= test_asof_join_tolerance();
    rc |= test_plan_asof_join();

    std::printf("\n  %d tests passed.\n", tests_passed);
    std::printf("  %s\n", rc == 0 ? "All ASOF join tests PASSED" : "SOME TESTS FAILED");
    return rc;
}
