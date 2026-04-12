/**
 * @file parser_test.cpp
 * @brief Google Test — SQL parser correctness (AST generation).
 *
 * Tests that the lexer+parser produce correct AST nodes for various SQL
 * statements. Uses deterministic inputs — no randomness needed.
 */
#include <gtest/gtest.h>
#include "sql/lexer.h"
#include "sql/parser.h"

using namespace gpudb;

// ── Basic SELECT ────────────────────────────────────────────────────────────

TEST(Parser, SimpleSelect) {
    Lexer lex;
    auto tokens = lex.tokenize("SELECT name, age FROM users");
    Parser parser(tokens);
    auto stmt = parser.parse();

    EXPECT_EQ(stmt->from_table, "users");
    EXPECT_EQ(stmt->select_list.size(), 2u);
    EXPECT_FALSE(stmt->where_clause);
    EXPECT_TRUE(stmt->group_by.empty());
}

// ── SELECT with WHERE ───────────────────────────────────────────────────────

TEST(Parser, SelectWithWhere) {
    Lexer lex;
    auto tokens = lex.tokenize("SELECT id FROM orders WHERE amount > 100");
    Parser parser(tokens);
    auto stmt = parser.parse();

    EXPECT_EQ(stmt->from_table, "orders");
    ASSERT_TRUE(stmt->where_clause != nullptr);
}

// ── SELECT with GROUP BY ────────────────────────────────────────────────────

TEST(Parser, SelectGroupBy) {
    Lexer lex;
    auto tokens = lex.tokenize(
        "SELECT region, SUM(revenue) FROM sales GROUP BY region");
    Parser parser(tokens);
    auto stmt = parser.parse();

    EXPECT_EQ(stmt->from_table, "sales");
    EXPECT_EQ(stmt->group_by.size(), 1u);
    EXPECT_EQ(stmt->group_by[0], "region");
}

// ── SELECT with ORDER BY ────────────────────────────────────────────────────

TEST(Parser, SelectOrderBy) {
    Lexer lex;
    auto tokens = lex.tokenize(
        "SELECT name FROM users ORDER BY name ASC");
    Parser parser(tokens);
    auto stmt = parser.parse();

    EXPECT_EQ(stmt->order_by_column, "name");
    EXPECT_TRUE(stmt->order_ascending);
}

// ── SELECT with LIMIT ───────────────────────────────────────────────────────

TEST(Parser, SelectLimit) {
    Lexer lex;
    auto tokens = lex.tokenize("SELECT * FROM big_table LIMIT 10");
    Parser parser(tokens);
    auto stmt = parser.parse();

    ASSERT_TRUE(stmt->limit.has_value());
    EXPECT_EQ(*stmt->limit, 10);
}

// ── Aggregate functions ─────────────────────────────────────────────────────

TEST(Parser, AggregateFunctions) {
    Lexer lex;
    auto tokens = lex.tokenize(
        "SELECT COUNT(*), AVG(price), MIN(qty), MAX(qty) FROM products");
    Parser parser(tokens);
    auto stmt = parser.parse();

    EXPECT_EQ(stmt->select_list.size(), 4u);
}

// ── Complex WHERE with AND ──────────────────────────────────────────────────

TEST(Parser, WhereWithAnd) {
    Lexer lex;
    auto tokens = lex.tokenize(
        "SELECT * FROM t WHERE a > 5 AND b < 10");
    Parser parser(tokens);
    auto stmt = parser.parse();

    ASSERT_TRUE(stmt->where_clause != nullptr);
    // The WHERE clause should be a BinaryExpr with AND
    auto* binop = dynamic_cast<BinaryExpr*>(stmt->where_clause.get());
    if (binop) {
        EXPECT_EQ(binop->op, "AND");
    }
}

// ── Empty table name throws ─────────────────────────────────────────────────

TEST(Parser, InvalidSyntaxThrows) {
    Lexer lex;
    auto tokens = lex.tokenize("SELECT FROM");
    Parser parser(tokens);
    EXPECT_THROW(parser.parse(), std::exception);
}

// ── ASOF JOIN ───────────────────────────────────────────────────────────────

TEST(Parser, AsofJoin) {
    Lexer lex;
    auto tokens = lex.tokenize(
        "SELECT t.ts, t.price, q.bid, q.ask "
        "FROM trades AS t "
        "ASOF JOIN quotes AS q "
        "ON t.symbol = q.symbol "
        "AS OF t.ts >= q.ts");
    Parser p;
    auto stmt = p.parseAsofJoin(tokens);

    EXPECT_EQ(stmt.left_table, "trades");
    EXPECT_EQ(stmt.left_alias, "t");
    EXPECT_EQ(stmt.right_table, "quotes");
    EXPECT_EQ(stmt.right_alias, "q");
    EXPECT_EQ(stmt.left_key_col, "symbol");
    EXPECT_EQ(stmt.right_key_col, "symbol");
    EXPECT_EQ(stmt.left_ts_col, "ts");
    EXPECT_EQ(stmt.right_ts_col, "ts");
    EXPECT_EQ(stmt.tolerance_ns, 0);

    // Check projected columns
    ASSERT_EQ(stmt.left_cols.size(), 2u);
    EXPECT_EQ(stmt.left_cols[0], "ts");
    EXPECT_EQ(stmt.left_cols[1], "price");
    ASSERT_EQ(stmt.right_cols.size(), 2u);
    EXPECT_EQ(stmt.right_cols[0], "bid");
    EXPECT_EQ(stmt.right_cols[1], "ask");
}

TEST(Parser, AsofJoinWithTolerance) {
    Lexer lex;
    auto tokens = lex.tokenize(
        "SELECT * FROM trades AS t "
        "ASOF JOIN quotes AS q "
        "ON t.symbol = q.symbol "
        "AS OF t.ts >= q.ts "
        "TOLERANCE 1000");
    Parser p;
    auto stmt = p.parseAsofJoin(tokens);

    EXPECT_EQ(stmt.left_table, "trades");
    EXPECT_EQ(stmt.right_table, "quotes");
    EXPECT_EQ(stmt.tolerance_ns, 1000);
    // SELECT * → empty left/right cols
    EXPECT_TRUE(stmt.left_cols.empty());
    EXPECT_TRUE(stmt.right_cols.empty());
}
