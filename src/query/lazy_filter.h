/**
 * @file lazy_filter.h
 * @brief Late materialization: evaluate predicate first, transfer only needed rows.
 *
 * How this mirrors the Vortex approach:
 *   Vortex (VLDB 2024) showed that PCIe is the primary bottleneck for GPU
 *   query processing. Their solution: don't transfer full columns to GPU.
 *   Instead, evaluate the WHERE predicate on CPU (cheap) to produce a selection
 *   vector, then transfer only the columns/rows needed for GROUP BY / SUM.
 *
 *   For a query with 1% selectivity on 100M rows:
 *     Full transfer:  100M × 8B = 800 MB over PCIe
 *     Late material:  1M × 8B   = 8 MB over PCIe  (100x less!)
 *
 *   This makes PCIe bandwidth irrelevant for selective queries.
 */
#pragma once
#include "core/table.h"
#include "query/operator_node.h"
#include <chrono>
#include <cstdio>
#include <functional>
#include <vector>

namespace gpudb {

class LazyFilterPlan {
public:
    using Predicate = std::function<bool(std::size_t row)>;

    LazyFilterPlan(const Table& src, Predicate pred)
        : src_(src), pred_(std::move(pred)) {}

    // Phase 1: CPU-side predicate → selection vector (cheap, no GPU transfer)
    SelectionVector evaluatePredicate() {
        SelectionVector sel;
        sel.reserve(src_.numRows() / 10);  // estimate
        for (std::size_t i = 0; i < src_.numRows(); ++i) {
            if (pred_(i)) sel.push_back(i);
        }
        return sel;
    }

    // Phase 2: materialize only selected rows (small GPU transfer)
    Table materializeSelected(const SelectionVector& sel,
                              const std::string& name = "filtered") {
        return materialize(src_, sel, name);
    }

    // Benchmark: full pipeline vs late materialization
    static void benchmark(const Table& src, Predicate pred,
                          const std::string& col_name) {
        using Clock = std::chrono::high_resolution_clock;

        // ── Full column approach ────────────────────────────────────────────
        auto t0 = Clock::now();
        SelectionVector sel_full;
        sel_full.reserve(src.numRows() / 10);
        for (std::size_t i = 0; i < src.numRows(); ++i)
            if (pred(i)) sel_full.push_back(i);
        Table full = materialize(src, sel_full, "full");
        auto t1 = Clock::now();
        double full_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::size_t full_bytes = src.numRows() * 8 * src.numCols();

        // ── Late materialization approach ────────────────────────────────────
        auto t2 = Clock::now();
        // Phase 1: predicate on CPU (tiny cost)
        SelectionVector sel;
        sel.reserve(src.numRows() / 10);
        for (std::size_t i = 0; i < src.numRows(); ++i)
            if (pred(i)) sel.push_back(i);
        // Phase 2: materialize only needed rows
        Table late = materialize(src, sel, "late");
        auto t3 = Clock::now();
        double late_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        std::size_t late_bytes = sel.size() * 8 * src.numCols();

        double selectivity = 100.0 * sel.size() / src.numRows();
        std::printf("\n═══ Late Materialization Benchmark ═══\n");
        std::printf("  Rows: %zu  Selectivity: %.1f%%  (%zu pass)\n",
                    src.numRows(), selectivity, sel.size());
        std::printf("  Full pipeline:  %7.1f ms  data=%zu MB\n",
                    full_ms, full_bytes / (1024*1024));
        std::printf("  Late material:  %7.1f ms  data=%zu MB\n",
                    late_ms, late_bytes / (1024*1024));
        std::printf("  PCIe savings:   %.0fx less data transferred\n",
                    (double)full_bytes / late_bytes);
    }

private:
    const Table& src_;
    Predicate pred_;
};

} // namespace gpudb
