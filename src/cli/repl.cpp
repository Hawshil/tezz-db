/**
 * @file repl.cpp
 * @brief Interactive CLI REPL for the GPU columnar database engine.
 *
 * Commands:
 *   LOAD <file.csv> AS <table_name>    Load CSV into a named table
 *   SELECT ...                          Parse, plan, execute SQL query
 *   BENCH <query>                       Benchmark query with speedup table
 *   EXPLAIN <query>                     Print physical execution plan
 *   DEVICES                             Print available GPU devices
 *   TABLES                              List loaded tables
 *   EXIT / QUIT                         Exit REPL
 */
#include "core/table.h"
#include "core/schema.h"
#include "io/csv_reader.h"
#include "sql/lexer.h"
#include "sql/parser.h"
#include "query/planner.h"
#include "query/operator_node.h"
#include "backend/gpu_backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <iomanip>

namespace gpudb {

class Repl {
public:
    void run() {
        printBanner();
        auto backend = BackendFactory::createBest();

        std::string line;
        while (true) {
            std::printf("\ngpudb> ");
            std::fflush(stdout);
            if (!std::getline(std::cin, line)) break;

            std::string trimmed = trim(line);
            if (trimmed.empty()) continue;

            std::string upper = toUpper(trimmed);

            if (upper == "EXIT" || upper == "QUIT") {
                std::printf("  Goodbye.\n");
                break;
            } else if (upper == "DEVICES") {
                cmdDevices(backend.get());
            } else if (upper == "TABLES") {
                cmdTables();
            } else if (upper.rfind("LOAD ", 0) == 0) {
                cmdLoad(trimmed);
            } else if (upper.rfind("EXPLAIN ", 0) == 0) {
                cmdExplain(trimmed.substr(8));
            } else if (upper.rfind("BENCH ", 0) == 0) {
                cmdBench(trimmed.substr(6), backend.get());
            } else if (upper.rfind("SELECT", 0) == 0) {
                cmdSelect(trimmed, backend.get());
            } else {
                std::printf("  Unknown command. Type EXIT to quit.\n");
            }
        }
    }

private:
    std::unordered_map<std::string, Table> tables_;

    void printBanner() {
        std::printf("\n");
        std::printf("  ╔═══════════════════════════════════════════════════════╗\n");
        std::printf("  ║  GPUDB — GPU-Accelerated Columnar Database Engine    ║\n");
        std::printf("  ║  Type SQL queries, LOAD data, or DEVICES to begin.   ║\n");
        std::printf("  ╚═══════════════════════════════════════════════════════╝\n");
    }

    void cmdDevices(GpuBackend* backend) {
        auto avail = BackendFactory::availableBackends();
        if (avail.empty()) {
            std::printf("  No GPU backends compiled. Running CPU-only.\n");
        } else {
            std::printf("  Available backends:\n");
            for (auto& b : avail) std::printf("    • %s\n", b.c_str());
        }
        if (backend)
            std::printf("  Active: %s — %s\n", backend->name().c_str(),
                        backend->deviceName().c_str());
    }

    void cmdTables() {
        if (tables_.empty()) { std::printf("  No tables loaded.\n"); return; }
        std::printf("  ┌──────────────────┬──────────┬─────────┐\n");
        std::printf("  │ Table            │ Rows     │ Columns │\n");
        std::printf("  ├──────────────────┼──────────┼─────────┤\n");
        for (auto& [name, tbl] : tables_)
            std::printf("  │ %-16s │ %8zu │ %7zu │\n",
                        name.c_str(), tbl.numRows(), tbl.numCols());
        std::printf("  └──────────────────┴──────────┴─────────┘\n");
    }

    // LOAD sales.csv AS sales
    void cmdLoad(const std::string& cmd) {
        std::istringstream ss(cmd);
        std::string load_kw, filepath, as_kw, tablename;
        ss >> load_kw >> filepath >> as_kw >> tablename;

        if (toUpper(as_kw) != "AS" || tablename.empty()) {
            std::printf("  Usage: LOAD <file.csv> AS <table_name>\n");
            return;
        }

        try {
            // Read CSV header to build schema (default all columns to STRING)
            std::ifstream header_file(filepath);
            if (!header_file.is_open())
                throw std::runtime_error("Cannot open file: " + filepath);
            std::string header_line;
            if (!std::getline(header_file, header_line))
                throw std::runtime_error("Empty CSV file: " + filepath);
            header_file.close();

            // Parse header to get column names
            SchemaBuilder sb;
            std::istringstream hss(header_line);
            std::string col_name;
            while (std::getline(hss, col_name, ',')) {
                // Trim whitespace
                auto a = col_name.find_first_not_of(" \t\r\n\"");
                auto b = col_name.find_last_not_of(" \t\r\n\"");
                if (a != std::string::npos)
                    col_name = col_name.substr(a, b - a + 1);
                sb.addColumn(col_name, DataType::STRING, true);
            }
            Schema schema = sb.build();

            CSVReader reader(filepath, schema);
            auto t0 = std::chrono::high_resolution_clock::now();
            Table tbl = reader.read(tablename);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            std::printf("  ✓ Loaded %zu rows, %zu columns in %.0f ms\n",
                        tbl.numRows(), tbl.numCols(), ms);
            if (reader.hasErrors())
                std::printf("  ⚠ %zu parse errors\n", reader.errors().size());

            tables_[tablename] = std::move(tbl);
        } catch (const std::exception& e) {
            std::printf("  ✗ Error: %s\n", e.what());
        }
    }

    void cmdSelect(const std::string& sql, GpuBackend* backend) {
        try {
            Lexer lexer;
            auto tokens = lexer.tokenize(sql);
            Parser parser;
            SelectStmt ast = parser.parse(tokens);

            // Find the source table
            auto it = tables_.find(ast.from_table);
            if (it == tables_.end()) {
                std::printf("  ✗ Table '%s' not found. Use LOAD first.\n",
                            ast.from_table.c_str());
                return;
            }

            // Build schema from the loaded table
            SchemaBuilder sb;
            const auto& tbl = it->second;
            for (std::size_t i = 0; i < tbl.numCols(); ++i) {
                const auto& col = tbl.getColumn(i);
                DataType dt = DataType::STRING;
                if (std::holds_alternative<Int32Column>(col)) dt = DataType::INT32;
                else if (std::holds_alternative<Int64Column>(col)) dt = DataType::INT64;
                else if (std::holds_alternative<Float64Column>(col)) dt = DataType::FLOAT64;
                sb.addColumn(tbl.columnNames()[i], dt);
            }
            Schema schema = sb.build();

            auto t0 = std::chrono::high_resolution_clock::now();
            QueryPlanner planner;
            auto plan = planner.plan(ast, schema);

            ExecutionContext ctx;
            ctx.catalog[ast.from_table] = &it->second;
            Table result = plan->execute(ctx);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            printResultTable(result);
            std::printf("  %zu rows in %.1f ms", result.numRows(), ms);
            if (backend)
                std::printf(" (GPU: %s)", backend->name().c_str());
            std::printf("\n");
        } catch (const std::exception& e) {
            std::printf("  ✗ Error: %s\n", e.what());
        }
    }

    void cmdExplain(const std::string& sql) {
        try {
            Lexer lexer;
            auto tokens = lexer.tokenize(sql);
            Parser parser;
            SelectStmt ast = parser.parse(tokens);

            std::printf("  Execution Plan:\n");
            std::printf("  ┌─ SELECT ");
            for (auto& item : ast.select_list)
                std::printf("%s ", item.expr->toString().c_str());
            std::printf("\n");
            std::printf("  ├─ FROM %s\n", ast.from_table.c_str());
            if (ast.where_clause)
                std::printf("  ├─ WHERE %s\n", ast.where_clause->toString().c_str());
            if (!ast.group_by.empty()) {
                std::printf("  ├─ GROUP BY ");
                for (auto& c : ast.group_by) std::printf("%s ", c.c_str());
                std::printf("\n");
            }
            if (!ast.order_by_column.empty())
                std::printf("  ├─ ORDER BY %s %s\n", ast.order_by_column.c_str(),
                            ast.order_ascending ? "ASC" : "DESC");
            if (ast.limit)
                std::printf("  └─ LIMIT %lld\n", (long long)*ast.limit);
            else
                std::printf("  └─ (no limit)\n");
        } catch (const std::exception& e) {
            std::printf("  ✗ Parse error: %s\n", e.what());
        }
    }

    void cmdBench(const std::string& sql, GpuBackend* backend) {
        std::printf("  Running benchmark...\n");
        // Run 5 warmup + 10 measured iterations
        std::vector<double> times;
        try {
            Lexer lexer;
            auto tokens = lexer.tokenize(sql);
            Parser parser;
            SelectStmt ast = parser.parse(tokens);

            auto it = tables_.find(ast.from_table);
            if (it == tables_.end()) {
                std::printf("  ✗ Table '%s' not found.\n", ast.from_table.c_str());
                return;
            }

            // Build schema
            SchemaBuilder sb;
            const auto& tbl = it->second;
            for (std::size_t i = 0; i < tbl.numCols(); ++i) {
                const auto& col = tbl.getColumn(i);
                DataType dt = DataType::STRING;
                if (std::holds_alternative<Int32Column>(col)) dt = DataType::INT32;
                else if (std::holds_alternative<Int64Column>(col)) dt = DataType::INT64;
                else if (std::holds_alternative<Float64Column>(col)) dt = DataType::FLOAT64;
                sb.addColumn(tbl.columnNames()[i], dt);
            }
            Schema schema = sb.build();

            QueryPlanner planner;

            // Warmup
            for (int i = 0; i < 5; ++i) {
                auto plan = planner.plan(ast, schema);
                ExecutionContext ctx;
                ctx.catalog[ast.from_table] = &it->second;
                plan->execute(ctx);
            }

            // Measured
            for (int i = 0; i < 10; ++i) {
                auto plan = planner.plan(ast, schema);
                ExecutionContext ctx;
                ctx.catalog[ast.from_table] = &it->second;
                auto t0 = std::chrono::high_resolution_clock::now();
                plan->execute(ctx);
                auto t1 = std::chrono::high_resolution_clock::now();
                times.push_back(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            std::sort(times.begin(), times.end());
            double med = times[5], mean = 0;
            for (auto t : times) mean += t;
            mean /= times.size();

            std::printf("  ┌── Benchmark Results ─────────────────────┐\n");
            std::printf("  │  Mean:   %8.2f ms                      \n", mean);
            std::printf("  │  Median: %8.2f ms                      \n", med);
            std::printf("  │  P95:    %8.2f ms                      \n", times[9]);
            std::printf("  │  Rows:   %8zu                          \n", it->second.numRows());
            std::printf("  └───────────────────────────────────────────┘\n");
            (void)backend;
        } catch (const std::exception& e) {
            std::printf("  ✗ Error: %s\n", e.what());
        }
    }

    void printResultTable(const Table& tbl) {
        if (tbl.numRows() == 0) { std::printf("  (empty result)\n"); return; }
        std::size_t maxRows = std::min(tbl.numRows(), (std::size_t)20);

        // Column widths
        auto& names = tbl.columnNames();
        std::vector<std::size_t> widths(tbl.numCols());
        for (std::size_t c = 0; c < tbl.numCols(); ++c)
            widths[c] = std::max(names[c].size(), (std::size_t)12);

        // Header
        std::printf("  ┌");
        for (std::size_t c = 0; c < tbl.numCols(); ++c) {
            for (std::size_t w = 0; w < widths[c] + 2; ++w) std::printf("─");
            std::printf(c + 1 < tbl.numCols() ? "┬" : "┐\n");
        }
        std::printf("  │");
        for (std::size_t c = 0; c < tbl.numCols(); ++c)
            std::printf(" %-*s│", (int)widths[c], names[c].c_str());
        std::printf("\n  ├");
        for (std::size_t c = 0; c < tbl.numCols(); ++c) {
            for (std::size_t w = 0; w < widths[c] + 2; ++w) std::printf("─");
            std::printf(c + 1 < tbl.numCols() ? "┼" : "┤\n");
        }

        // Rows
        for (std::size_t r = 0; r < maxRows; ++r) {
            std::printf("  │");
            for (std::size_t c = 0; c < tbl.numCols(); ++c) {
                std::string val = valueToString(getColumnValue(tbl.getColumn(c), r));
                std::printf(" %-*s│", (int)widths[c], val.c_str());
            }
            std::printf("\n");
        }

        std::printf("  └");
        for (std::size_t c = 0; c < tbl.numCols(); ++c) {
            for (std::size_t w = 0; w < widths[c] + 2; ++w) std::printf("─");
            std::printf(c + 1 < tbl.numCols() ? "┴" : "┘\n");
        }
        if (tbl.numRows() > maxRows)
            std::printf("  ... and %zu more rows\n", tbl.numRows() - maxRows);
    }

    static std::string trim(const std::string& s) {
        auto a = s.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) return "";
        return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
    }
    static std::string toUpper(const std::string& s) {
        std::string r = s;
        std::transform(r.begin(), r.end(), r.begin(), ::toupper);
        return r;
    }
};

} // namespace gpudb

int main() {
    gpudb::Repl repl;
    repl.run();
    return 0;
}
