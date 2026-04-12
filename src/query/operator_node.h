/**
 * @file operator_node.h
 * @brief Physical execution plan nodes for the query engine.
 *
 * Each node produces a materialised Table when execute() is called.
 * The plan forms a tree: leaf ScanNodes read from the catalog, and
 * intermediate nodes transform the output of their single child.
 */
#pragma once

#include "../core/column.h"
#include "../core/table.h"
#include "../sql/ast.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace gpudb {

// ── Value type for expression evaluation ────────────────────────────────────

using Value = std::variant<std::monostate, std::int32_t, std::int64_t, double, std::string>;

Value getColumnValue(const ColumnVariant& col, std::size_t row);
double toDouble(const Value& v);
bool   isTruthy(const Value& v);
std::string valueToString(const Value& v);
Value evaluateExpr(const Expr& expr, const Table& table, std::size_t row);

// ── Execution context: catalog of named tables ──────────────────────────────

struct ExecutionContext {
    std::unordered_map<std::string, const Table*> catalog;
};

// ── Aggregate specification ─────────────────────────────────────────────────

struct AggregateSpec {
    std::string func;          // SUM, COUNT, AVG, MIN, MAX
    std::string input_column;  // empty for COUNT(*)
    std::string output_name;   // output column name
};

// ── Base operator node ──────────────────────────────────────────────────────

class OperatorNode {
public:
    virtual ~OperatorNode() = default;
    virtual Table execute(const ExecutionContext& ctx) = 0;
    virtual std::string toString(int indent = 0) const = 0;

    void setInput(std::unique_ptr<OperatorNode> input) {
        input_ = std::move(input);
    }

protected:
    std::unique_ptr<OperatorNode> input_;  // child node (nullptr for leaf)
};

// ── Concrete operator nodes ─────────────────────────────────────────────────

/** Reads a table from the catalog and projects specific columns. */
class ScanNode : public OperatorNode {
public:
    ScanNode(std::string table_name, std::vector<std::string> columns);
    Table execute(const ExecutionContext& ctx) override;
    std::string toString(int indent = 0) const override;
private:
    std::string table_name_;
    std::vector<std::string> columns_;
};

/** Filters rows by evaluating a predicate expression. */
class FilterNode : public OperatorNode {
public:
    explicit FilterNode(ExprPtr predicate);
    Table execute(const ExecutionContext& ctx) override;
    std::string toString(int indent = 0) const override;
private:
    ExprPtr predicate_;
};

/** Computes GROUP BY + aggregate functions. */
class AggregateNode : public OperatorNode {
public:
    AggregateNode(std::vector<std::string> group_by,
                  std::vector<AggregateSpec> aggregates);
    Table execute(const ExecutionContext& ctx) override;
    std::string toString(int indent = 0) const override;
private:
    std::vector<std::string> group_by_;
    std::vector<AggregateSpec> aggregates_;
};

/** Applies a window function over an ordered partition. */
class GpuBackend;  // forward declaration

class WindowNode : public OperatorNode {
public:
    static constexpr int GPU_THRESHOLD = 100'000;

    struct WindowSpec {
        std::string func;         // "SMA", "EMA", "ROLLING_STD"
        std::string input_col;    // e.g. "price"
        std::string order_by_col; // e.g. "ts"
        std::string output_col;   // e.g. "SMA(price,20)"
        int window_size;
    };
    explicit WindowNode(WindowSpec spec);
    Table execute(const ExecutionContext& ctx) override;
    std::string toString(int indent = 0) const override;

    /** Attach an optional GPU backend; if set and n > GPU_THRESHOLD,
     *  window computation is offloaded to the GPU. */
    void setBackend(GpuBackend* b) { gpu_ = b; }

private:
    WindowSpec spec_;
    GpuBackend* gpu_ = nullptr;
    // Helper implementations declared here, defined in .cpp:
    static std::vector<double> computeSMA(
        const std::vector<double>& vals, int w);
    static std::vector<double> computeEMA(
        const std::vector<double>& vals, int w);
    static std::vector<double> computeRollingStd(
        const std::vector<double>& vals, int w);
};

/** Sorts rows by a column. */
class SortNode : public OperatorNode {
public:
    SortNode(std::string column, bool ascending);
    Table execute(const ExecutionContext& ctx) override;
    std::string toString(int indent = 0) const override;
private:
    std::string column_;
    bool ascending_;
};

/** Truncates output to at most N rows. */
class LimitNode : public OperatorNode {
public:
    explicit LimitNode(std::int64_t limit);
    Table execute(const ExecutionContext& ctx) override;
    std::string toString(int indent = 0) const override;
private:
    std::int64_t limit_;
};

// ── Selection Vector ────────────────────────────────────────────────────────

using SelectionVector = std::vector<std::size_t>;

/** Materialize a new Table by picking rows from @p src at the given indices. */
Table materialize(const Table& src, const SelectionVector& sel,
                  const std::string& name);

// ── Hash Join helpers ───────────────────────────────────────────────────────

/** Build-and-probe on typed columns; appends (build_row, probe_row) pairs. */
template<typename T>
void buildAndProbe(const TypedColumn<T>& build_col,
                   const TypedColumn<T>& probe_col,
                   std::vector<std::pair<std::size_t, std::size_t>>& matches) {
    std::unordered_multimap<T, std::size_t> hm;
    hm.reserve(build_col.size());
    for (std::size_t i = 0; i < build_col.size(); ++i)
        hm.emplace(build_col[i], i);
    matches.reserve(probe_col.size());
    for (std::size_t i = 0; i < probe_col.size(); ++i) {
        auto [b, e] = hm.equal_range(probe_col[i]);
        for (auto it = b; it != e; ++it)
            matches.emplace_back(it->second, i);
    }
}

/** Inner hash-join: reads build + probe tables from the catalog. */
class HashJoinNode : public OperatorNode {
public:
    HashJoinNode(std::string build_table, std::string build_col,
                 std::string probe_table, std::string probe_col);
    Table execute(const ExecutionContext& ctx) override;
    std::string toString(int indent = 0) const override;
private:
    std::string build_table_, build_col_;
    std::string probe_table_, probe_col_;
};

/** ASOF join: for each left row find the latest matching right row. */
class AsofJoinNode : public OperatorNode {
public:
    struct Spec {
        std::string left_table,  right_table;
        std::string left_key,    right_key;     // equality columns
        std::string left_ts_col, right_ts_col;  // timestamp columns
        std::int64_t tolerance_ns = 0;
    };
    explicit AsofJoinNode(Spec spec);
    Table execute(const ExecutionContext& ctx) override;
    std::string toString(int indent = 0) const override;

    void setBackend(GpuBackend* b) { gpu_ = b; }
private:
    Spec         spec_;
    GpuBackend*  gpu_ = nullptr;
    // CPU fallback: sort-merge binary search
    Table cpuAsofJoin(const Table& left, const Table& right) const;
};

}  // namespace gpudb
