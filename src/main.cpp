/**
 * @file main.cpp
 * @brief Full end-to-end demo: Schema → Table → SQL Lexer → Parser → Planner → Execute.
 *
 * Demonstrates:
 *   1. Building a sales table with hardcoded data.
 *   2. Tokenizing a SQL query.
 *   3. Parsing tokens into an AST.
 *   4. Planning the query into an operator tree.
 *   5. Executing the plan and printing results.
 */

#include "core/schema.h"
#include "core/column.h"
#include "core/table.h"
#include "sql/lexer.h"
#include "sql/parser.h"
#include "query/planner.h"
#include "query/operator_node.h"

#include <iomanip>
#include <iostream>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// Pretty-print a result table
// ─────────────────────────────────────────────────────────────────────────────

static void printTable(const gpudb::Table& table) {
    using namespace gpudb;
    const auto& names = table.columnNames();
    const std::size_t ncols = table.numCols();
    const std::size_t nrows = table.numRows();

    // Compute column widths.
    std::vector<std::size_t> widths(ncols);
    for (std::size_t c = 0; c < ncols; ++c) {
        widths[c] = names[c].size();
        const auto& col = table.getColumn(c);
        for (std::size_t r = 0; r < nrows; ++r) {
            std::string val = valueToString(getColumnValue(col, r));
            widths[c] = std::max(widths[c], val.size());
        }
        widths[c] = std::min(widths[c], std::size_t(20)); // cap
    }

    // Header
    std::cout << "  ";
    for (std::size_t c = 0; c < ncols; ++c)
        std::cout << std::left << std::setw(static_cast<int>(widths[c] + 2)) << names[c];
    std::cout << "\n  ";
    for (std::size_t c = 0; c < ncols; ++c)
        std::cout << std::string(widths[c], '-') << "  ";
    std::cout << "\n";

    // Rows
    for (std::size_t r = 0; r < nrows; ++r) {
        std::cout << "  ";
        for (std::size_t c = 0; c < ncols; ++c) {
            std::string val = valueToString(getColumnValue(table.getColumn(c), r));
            std::cout << std::left << std::setw(static_cast<int>(widths[c] + 2)) << val;
        }
        std::cout << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    using namespace gpudb;

    std::cout << "\n"
              << "  ╔══════════════════════════════════════════════════════╗\n"
              << "  ║  GPUDB — SQL Query Engine End-to-End Demonstration  ║\n"
              << "  ╚══════════════════════════════════════════════════════╝\n\n";

    // ── 1. Define schema and build a sample sales table ─────────────────────

    Schema schema = SchemaBuilder()
        .addColumn("order_id", DataType::INT64)
        .addColumn("amount",   DataType::FLOAT64)
        .addColumn("region",   DataType::STRING)
        .addColumn("year",     DataType::INT32)
        .build();

    Int64Column   ids({1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010});
    Float64Column amts({250.00, 430.50, 120.75, 890.00, 345.25,
                        560.00, 215.50, 780.00,  95.25, 670.00});
    StringColumn  regions(std::vector<std::string>{
        "North","South","North","East","West",
        "North","South","East","West","North"});
    Int32Column   years({2023, 2024, 2024, 2025, 2023, 2025, 2024, 2024, 2025, 2024});

    Table sales("sales",
                {"order_id", "amount", "region", "year"},
                {std::move(ids), std::move(amts), std::move(regions), std::move(years)});

    std::cout << "═══ Source Table ═══\n" << sales.schemaToString();
    printTable(sales);

    // ── 2. SQL query to execute ─────────────────────────────────────────────

    std::string sql =
        "SELECT region, SUM(amount) AS total_rev, COUNT(order_id) AS num_orders "
        "FROM sales "
        "WHERE year >= 2024 "
        "GROUP BY region "
        "ORDER BY region "
        "LIMIT 10";

    std::cout << "\n═══ SQL Query ═══\n  " << sql << "\n";

    // ── 3. Tokenize ─────────────────────────────────────────────────────────

    Lexer lexer;
    auto tokens = lexer.tokenize(sql);

    std::cout << "\n═══ Tokens (" << tokens.size() << ") ═══\n";
    for (std::size_t i = 0; i < tokens.size(); ++i)
        std::cout << "  [" << std::setw(2) << i << "] " << tokens[i].toString() << "\n";

    // ── 4. Parse into AST ───────────────────────────────────────────────────

    Parser parser;
    SelectStmt ast = parser.parse(tokens);

    std::cout << "\n═══ Abstract Syntax Tree ═══\n" << ast.toString() << "\n";

    // ── 5. Plan the query ───────────────────────────────────────────────────

    QueryPlanner planner;
    auto plan = planner.plan(ast, schema);

    std::cout << "═══ Execution Plan ═══\n" << plan->toString() << "\n";

    // ── 6. Execute ──────────────────────────────────────────────────────────

    ExecutionContext ctx;
    ctx.catalog["sales"] = &sales;

    Table result = plan->execute(ctx);

    std::cout << "\n═══ Query Result ═══\n" << result.schemaToString();
    printTable(result);

    // ── 7. Run a second query: simple filter, no aggregation ────────────────

    std::cout << "\n────────────────────────────────────────────────────────\n";

    std::string sql2 =
        "SELECT order_id, amount, region "
        "FROM sales "
        "WHERE amount > 500 "
        "ORDER BY amount DESC";

    std::cout << "\n═══ SQL Query 2 ═══\n  " << sql2 << "\n";

    auto tokens2 = lexer.tokenize(sql2);
    SelectStmt ast2 = parser.parse(tokens2);
    std::cout << "\n═══ AST ═══\n" << ast2.toString() << "\n";

    auto plan2 = planner.plan(ast2, schema);
    std::cout << "═══ Execution Plan ═══\n" << plan2->toString() << "\n";

    Table result2 = plan2->execute(ctx);
    std::cout << "\n═══ Query Result ═══\n" << result2.schemaToString();
    printTable(result2);

    std::cout << "\n✓ All queries executed successfully.\n";
    return 0;
}
