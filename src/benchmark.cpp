/**
 * @file benchmark.cpp
 * @brief CPU baseline benchmarks: Scan+Filter, Aggregate, Hash Join.
 *
 * Tests on large synthetic datasets (10M / 1M+5M rows) and reports
 * wall-clock time and throughput.  These numbers become the CPU baseline
 * that will later be compared against GPU kernel implementations.
 */

#include "core/column.h"
#include "core/table.h"
#include "query/operator_node.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

using namespace gpudb;
using Clock = std::chrono::high_resolution_clock;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::string fmtNum(std::int64_t n) {
    std::string s = std::to_string(n);
    for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3)
        s.insert(i, ",");
    return s;
}

static double toMs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static void printThroughput(const char* label, std::int64_t rows, double ms) {
    double mrows_sec = rows / (ms / 1000.0) / 1e6;
    std::cout << "  " << std::left << std::setw(28) << label
              << std::right << std::setw(10) << std::fixed << std::setprecision(1)
              << ms << " ms   (" << std::setprecision(1) << mrows_sec << " M rows/sec)\n";
}

static const std::string REGIONS[] = {
    "North", "South", "East", "West", "Central",
    "NorthEast", "NorthWest", "SouthEast", "SouthWest", "Midwest"
};
static constexpr int NUM_REGIONS = 10;

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark 1: Scan + Filter  (10 M rows)
// ─────────────────────────────────────────────────────────────────────────────

static void benchScanFilter() {
    const std::size_t N = 10'000'000;
    std::cout << "\n═══ Benchmark 1: Scan + Filter (" << fmtNum(N) << " rows) ═══\n";

    // ── Generate table ──────────────────────────────────────────────────────
    auto t0 = Clock::now();

    Int64Column   ids;   ids.reserve(N);
    Float64Column amts;  amts.reserve(N);
    StringColumn  regs;  regs.reserve(N);
    Int32Column   yrs;   yrs.reserve(N);

    std::srand(42);
    for (std::size_t i = 0; i < N; ++i) {
        ids.pushValue(static_cast<std::int64_t>(100000 + i));
        amts.pushValue(10.0 + (std::rand() % 99000) / 100.0);
        regs.pushValue(REGIONS[std::rand() % NUM_REGIONS]);
        yrs.pushValue(2020 + std::rand() % 7);  // 2020–2026
    }

    Table sales("sales",
                {"order_id", "amount", "region", "year"},
                {std::move(ids), std::move(amts), std::move(regs), std::move(yrs)});

    auto t1 = Clock::now();
    std::cout << "  Table generation:          " << std::fixed << std::setprecision(1)
              << toMs(t0, t1) << " ms\n";

    // ── Scan: project 2 of 4 columns ────────────────────────────────────────
    auto t2 = Clock::now();

    const auto& amount_col = sales.getTypedColumn<double>("amount");
    const auto& year_col   = sales.getTypedColumn<std::int32_t>("year");

    // "Scan" = just accessing the columns (zero-copy in columnar layout).
    volatile double sink1 = amount_col[0];  // prevent optimisation
    volatile int    sink2 = year_col[0];
    (void)sink1; (void)sink2;

    auto t3 = Clock::now();
    printThroughput("Column projection (2 cols):", N, toMs(t2, t3));

    // ── Filter: build selection vector (year >= 2024) ───────────────────────
    auto t4 = Clock::now();

    SelectionVector sel;
    sel.reserve(N / 2);
    const std::int32_t* yr_data = year_col.data();
    for (std::size_t i = 0; i < N; ++i) {
        if (yr_data[i] >= 2024) sel.push_back(i);
    }

    auto t5 = Clock::now();
    std::cout << "  Selection vector built:    " << std::fixed << std::setprecision(1)
              << toMs(t4, t5) << " ms → " << fmtNum(sel.size()) << " rows pass\n";
    printThroughput("  Filter throughput:", N, toMs(t4, t5));

    // ── Materialize filtered table ──────────────────────────────────────────
    auto t6 = Clock::now();
    Table filtered = materialize(sales, sel, "filtered_sales");
    auto t7 = Clock::now();
    printThroughput("Materialisation:", sel.size(), toMs(t6, t7));

    std::cout << "  Total scan+filter:         " << std::fixed << std::setprecision(1)
              << toMs(t2, t7) << " ms\n";
    printThroughput("End-to-end scan+filter:", N, toMs(t2, t7));
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark 2: Aggregate  (10 M rows)
// ─────────────────────────────────────────────────────────────────────────────

struct AggState {
    double sum   = 0;
    double min_v = std::numeric_limits<double>::max();
    double max_v = std::numeric_limits<double>::lowest();
    std::int64_t count = 0;
    void add(double v) {
        sum += v;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        ++count;
    }
    double avg() const { return count ? sum / static_cast<double>(count) : 0; }
};

static void benchAggregate() {
    const std::size_t N = 10'000'000;
    std::cout << "\n═══ Benchmark 2: Aggregate (" << fmtNum(N) << " rows) ═══\n";
    std::cout << "  Query: GROUP BY region, SUM(amount), COUNT(*), AVG, MIN, MAX\n";

    // ── Generate table ──────────────────────────────────────────────────────
    auto t0 = Clock::now();

    Float64Column amts;  amts.reserve(N);
    StringColumn  regs;  regs.reserve(N);

    std::srand(42);
    for (std::size_t i = 0; i < N; ++i) {
        amts.pushValue(10.0 + (std::rand() % 99000) / 100.0);
        regs.pushValue(REGIONS[std::rand() % NUM_REGIONS]);
    }

    Table sales("sales", {"amount", "region"},
                {std::move(amts), std::move(regs)});

    auto t1 = Clock::now();
    std::cout << "  Table generation:          " << std::fixed << std::setprecision(1)
              << toMs(t0, t1) << " ms\n";

    // ── Aggregate ───────────────────────────────────────────────────────────
    auto t2 = Clock::now();

    const auto& amt_col = sales.getTypedColumn<double>("amount");
    const auto& reg_col = sales.getTypedColumn<std::string>("region");

    std::unordered_map<std::string, AggState> groups;
    groups.reserve(NUM_REGIONS * 2);

    const double*      a_ptr = amt_col.data();
    for (std::size_t i = 0; i < N; ++i) {
        groups[reg_col[i]].add(a_ptr[i]);
    }

    auto t3 = Clock::now();
    printThroughput("Aggregation:", N, toMs(t2, t3));
    std::cout << "  Groups: " << groups.size() << "\n";

    // Print first 5 groups
    int shown = 0;
    std::cout << "  Sample results:\n";
    for (const auto& [key, st] : groups) {
        if (++shown > 5) break;
        std::cout << "    " << std::left << std::setw(12) << key
                  << "  SUM=" << std::fixed << std::setprecision(2) << st.sum
                  << "  COUNT=" << st.count
                  << "  AVG=" << st.avg()
                  << "  MIN=" << st.min_v
                  << "  MAX=" << st.max_v << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark 3: Hash Join  (1 M × 5 M rows)
// ─────────────────────────────────────────────────────────────────────────────

static void benchHashJoin() {
    const std::size_t N_BUILD = 1'000'000;
    const std::size_t N_PROBE = 5'000'000;

    std::cout << "\n═══ Benchmark 3: Hash Join (" << fmtNum(N_BUILD)
              << " × " << fmtNum(N_PROBE) << " rows) ═══\n";

    // ── Generate build table ("orders") ─────────────────────────────────────
    auto t0 = Clock::now();

    Int64Column   order_ids;  order_ids.reserve(N_BUILD);
    Float64Column totals;     totals.reserve(N_BUILD);

    for (std::size_t i = 0; i < N_BUILD; ++i) {
        order_ids.pushValue(static_cast<std::int64_t>(i));
        totals.pushValue(100.0 + (std::rand() % 90000) / 100.0);
    }

    Table orders("orders", {"order_id", "total"},
                 {std::move(order_ids), std::move(totals)});

    // ── Generate probe table ("items") ──────────────────────────────────────
    Int64Column   item_order_ids;  item_order_ids.reserve(N_PROBE);
    Float64Column prices;          prices.reserve(N_PROBE);

    std::srand(99);
    for (std::size_t i = 0; i < N_PROBE; ++i) {
        // Each item references a random order (uniform distribution).
        item_order_ids.pushValue(static_cast<std::int64_t>(std::rand() % N_BUILD));
        prices.pushValue(1.0 + (std::rand() % 50000) / 100.0);
    }

    Table items("items", {"order_id", "price"},
                {std::move(item_order_ids), std::move(prices)});

    auto t1 = Clock::now();
    std::cout << "  Table generation:          " << std::fixed << std::setprecision(1)
              << toMs(t0, t1) << " ms\n";

    // ── Hash Join via typed build-and-probe ──────────────────────────────────
    auto t2 = Clock::now();

    // Build phase
    const auto& build_col = orders.getTypedColumn<std::int64_t>("order_id");
    std::unordered_multimap<std::int64_t, std::size_t> hash_map;
    hash_map.reserve(N_BUILD);
    for (std::size_t i = 0; i < build_col.size(); ++i)
        hash_map.emplace(build_col[i], i);

    auto t3 = Clock::now();
    printThroughput("Build phase:", N_BUILD, toMs(t2, t3));

    // Probe phase
    const auto& probe_col = items.getTypedColumn<std::int64_t>("order_id");
    std::vector<std::pair<std::size_t, std::size_t>> matches;
    matches.reserve(N_PROBE);

    for (std::size_t i = 0; i < probe_col.size(); ++i) {
        auto [b, e] = hash_map.equal_range(probe_col[i]);
        for (auto it = b; it != e; ++it)
            matches.emplace_back(it->second, i);
    }

    auto t4 = Clock::now();
    printThroughput("Probe phase:", N_PROBE, toMs(t3, t4));

    std::cout << "  Matched rows:              " << fmtNum(matches.size()) << "\n";
    printThroughput("Total join:", N_BUILD + N_PROBE, toMs(t2, t4));

    // ── Materialize first 5 result rows for verification ────────────────────
    std::cout << "  Sample output (first 5 matches):\n";
    const auto& ord_id_col  = orders.getTypedColumn<std::int64_t>("order_id");
    const auto& ord_tot_col = orders.getTypedColumn<double>("total");
    const auto& itm_oid_col = items.getTypedColumn<std::int64_t>("order_id");
    const auto& itm_prc_col = items.getTypedColumn<double>("price");

    for (std::size_t i = 0; i < std::min(matches.size(), std::size_t(5)); ++i) {
        auto [br, pr] = matches[i];
        std::cout << "    orders.order_id=" << ord_id_col[br]
                  << "  orders.total=" << std::fixed << std::setprecision(2) << ord_tot_col[br]
                  << "  items.order_id=" << itm_oid_col[pr]
                  << "  items.price=" << itm_prc_col[pr] << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n"
              << "  ╔══════════════════════════════════════════════════════════╗\n"
              << "  ║  GPUDB — CPU Baseline Benchmarks                       ║\n"
              << "  ║  These numbers will be compared against GPU kernels.    ║\n"
              << "  ╚══════════════════════════════════════════════════════════╝\n";

    benchScanFilter();
    benchAggregate();
    benchHashJoin();

    std::cout << "\n✓ All benchmarks completed.\n";
    return 0;
}
