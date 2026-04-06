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
