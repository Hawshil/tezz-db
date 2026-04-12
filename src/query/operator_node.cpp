/**
 * @file operator_node.cpp
 * @brief Execution logic for all physical operator nodes.
 */
#include "operator_node.h"
#include "../core/schema.h"
#include "../backend/gpu_backend.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace gpudb {

// ─────────────────────────────────────────────────────────────────────────────
// Value helpers
// ─────────────────────────────────────────────────────────────────────────────

Value getColumnValue(const ColumnVariant& col, std::size_t row) {
    return std::visit([row](const auto& c) -> Value { return Value(c[row]); }, col);
}

double toDouble(const Value& v) {
    if (auto* i = std::get_if<std::int32_t>(&v)) return static_cast<double>(*i);
    if (auto* i = std::get_if<std::int64_t>(&v)) return static_cast<double>(*i);
    if (auto* d = std::get_if<double>(&v))        return *d;
    throw std::runtime_error("Cannot convert value to double");
}

bool isTruthy(const Value& v) {
    return std::visit([](const auto& val) -> bool {
        using T = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, std::monostate>) return false;
        if constexpr (std::is_same_v<T, std::int32_t>)   return val != 0;
        if constexpr (std::is_same_v<T, std::int64_t>)   return val != 0;
        if constexpr (std::is_same_v<T, double>)          return val != 0.0;
        if constexpr (std::is_same_v<T, std::string>)     return !val.empty();
        return false;
    }, v);
}

std::string valueToString(const Value& v) {
    return std::visit([](const auto& val) -> std::string {
        using T = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, std::monostate>) return "NULL";
        else if constexpr (std::is_same_v<T, std::string>) return val;
        else if constexpr (std::is_same_v<T, std::int32_t>) return std::to_string(val);
        else if constexpr (std::is_same_v<T, std::int64_t>) return std::to_string(val);
        else if constexpr (std::is_same_v<T, double>) return std::to_string(val);
        else return "?";
    }, v);
}

static bool isNumeric(const Value& v) {
    return std::holds_alternative<std::int32_t>(v) ||
           std::holds_alternative<std::int64_t>(v) ||
           std::holds_alternative<double>(v);
}

static Value applyComparison(const std::string& op, const Value& l, const Value& r) {
    // Numeric comparison
    if (isNumeric(l) && isNumeric(r)) {
        double ld = toDouble(l), rd = toDouble(r);
        bool result = false;
        if (op == "=")  result = ld == rd;
        else if (op == "!=" || op == "<>") result = ld != rd;
        else if (op == "<")  result = ld < rd;
        else if (op == ">")  result = ld > rd;
        else if (op == "<=") result = ld <= rd;
        else if (op == ">=") result = ld >= rd;
        return Value(static_cast<std::int32_t>(result ? 1 : 0));
    }
    // String comparison
    if (std::holds_alternative<std::string>(l) && std::holds_alternative<std::string>(r)) {
        const auto& ls = std::get<std::string>(l);
        const auto& rs = std::get<std::string>(r);
        bool result = false;
        if (op == "=")  result = ls == rs;
        else if (op == "!=" || op == "<>") result = ls != rs;
        else if (op == "<")  result = ls < rs;
        else if (op == ">")  result = ls > rs;
        else if (op == "<=") result = ls <= rs;
        else if (op == ">=") result = ls >= rs;
        return Value(static_cast<std::int32_t>(result ? 1 : 0));
    }
    throw std::runtime_error("Cannot compare values of incompatible types");
}

static Value applyArithmetic(const std::string& op, const Value& l, const Value& r) {
    double ld = toDouble(l), rd = toDouble(r);
    double result = 0;
    if (op == "+")      result = ld + rd;
    else if (op == "-") result = ld - rd;
    else if (op == "*") result = ld * rd;
    else if (op == "/") {
        if (rd == 0.0) throw std::runtime_error("Division by zero");
        result = ld / rd;
    }
    return Value(result);
}

Value evaluateExpr(const Expr& expr, const Table& table, std::size_t row) {
    if (auto* col = dynamic_cast<const ColumnRef*>(&expr)) {
        auto idx = table.findColumnIndex(col->column);
        if (!idx) throw std::runtime_error("Column not found: " + col->column);
        return getColumnValue(table.getColumn(*idx), row);
    }
    if (auto* lit = dynamic_cast<const LiteralExpr*>(&expr)) {
        return std::visit([](const auto& v) -> Value { return Value(v); }, lit->value);
    }
    if (auto* bin = dynamic_cast<const BinaryExpr*>(&expr)) {
        if (bin->op == "AND") {
            return Value(static_cast<std::int32_t>(
                isTruthy(evaluateExpr(*bin->left, table, row)) &&
                isTruthy(evaluateExpr(*bin->right, table, row)) ? 1 : 0));
        }
        if (bin->op == "OR") {
            return Value(static_cast<std::int32_t>(
                isTruthy(evaluateExpr(*bin->left, table, row)) ||
                isTruthy(evaluateExpr(*bin->right, table, row)) ? 1 : 0));
        }
        Value lv = evaluateExpr(*bin->left, table, row);
        Value rv = evaluateExpr(*bin->right, table, row);
        if (bin->op == "=" || bin->op == "!=" || bin->op == "<>" ||
            bin->op == "<" || bin->op == ">"  || bin->op == "<=" || bin->op == ">=")
            return applyComparison(bin->op, lv, rv);
        return applyArithmetic(bin->op, lv, rv);
    }
    if (auto* un = dynamic_cast<const UnaryExpr*>(&expr)) {
        Value v = evaluateExpr(*un->operand, table, row);
        if (un->op == "NOT") return Value(static_cast<std::int32_t>(isTruthy(v) ? 0 : 1));
        if (un->op == "-")   return Value(-toDouble(v));
    }
    throw std::runtime_error("Cannot evaluate expression: " + expr.toString());
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build a table from selected row indices
// ─────────────────────────────────────────────────────────────────────────────

static Table selectRows(const Table& src, const std::vector<std::size_t>& rows,
                         const std::string& name) {
    std::vector<std::string> col_names;
    std::vector<ColumnVariant> cols;
    for (std::size_t c = 0; c < src.numCols(); ++c) {
        col_names.push_back(src.columnNames()[c]);
        const auto& src_col = src.getColumn(c);
        std::visit([&](const auto& typed_src) {
            using ColT = std::decay_t<decltype(typed_src)>;
            ColT new_col;
            new_col.reserve(rows.size());
            for (auto r : rows) new_col.pushValue(typed_src[r]);
            cols.emplace_back(std::move(new_col));
        }, src_col);
    }
    return Table(name, std::move(col_names), std::move(cols));
}

// ─────────────────────────────────────────────────────────────────────────────
// ScanNode
// ─────────────────────────────────────────────────────────────────────────────

ScanNode::ScanNode(std::string table_name, std::vector<std::string> columns)
    : table_name_(std::move(table_name)), columns_(std::move(columns)) {}

Table ScanNode::execute(const ExecutionContext& ctx) {
    auto it = ctx.catalog.find(table_name_);
    if (it == ctx.catalog.end())
        throw std::runtime_error("ScanNode: table '" + table_name_ + "' not in catalog");
    const Table& src = *it->second;

    // If no specific columns requested, project all.
    if (columns_.empty()) {
        // Return a full copy.
        std::vector<std::string> names = src.columnNames();
        std::vector<ColumnVariant> cols;
        for (std::size_t i = 0; i < src.numCols(); ++i)
            cols.push_back(src.getColumn(i)); // copy
        return Table(table_name_, std::move(names), std::move(cols));
    }

    Table result(table_name_);
    for (const auto& col_name : columns_) {
        if (!src.hasColumn(col_name))
            throw std::runtime_error("ScanNode: column '" + col_name + "' not in table '" + table_name_ + "'");
        result.addColumn(col_name, src.getColumn(col_name)); // copy
    }
    return result;
}

std::string ScanNode::toString(int indent) const {
    std::string pad(indent * 2, ' ');
    std::string s = pad + "ScanNode(" + table_name_ + " [";
    for (std::size_t i = 0; i < columns_.size(); ++i) {
        if (i) s += ", ";
        s += columns_[i];
    }
    s += "])";
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// FilterNode
// ─────────────────────────────────────────────────────────────────────────────

FilterNode::FilterNode(ExprPtr predicate) : predicate_(std::move(predicate)) {}

Table FilterNode::execute(const ExecutionContext& ctx) {
    Table input = input_->execute(ctx);
    std::vector<std::size_t> matching;
    for (std::size_t r = 0; r < input.numRows(); ++r) {
        if (isTruthy(evaluateExpr(*predicate_, input, r)))
            matching.push_back(r);
    }
    return selectRows(input, matching, input.name());
}

std::string FilterNode::toString(int indent) const {
    std::string pad(indent * 2, ' ');
    std::string s = pad + "FilterNode(" + predicate_->toString() + ")\n";
    if (input_) s += input_->toString(indent + 1);
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// AggregateNode
// ─────────────────────────────────────────────────────────────────────────────

AggregateNode::AggregateNode(std::vector<std::string> group_by,
                             std::vector<AggregateSpec> aggregates)
    : group_by_(std::move(group_by)), aggregates_(std::move(aggregates)) {}

Table AggregateNode::execute(const ExecutionContext& ctx) {
    Table input = input_->execute(ctx);
    const std::size_t nrows = input.numRows();

    // Build groups: key -> row indices
    std::map<std::string, std::vector<std::size_t>> groups;
    for (std::size_t r = 0; r < nrows; ++r) {
        std::string key;
        for (const auto& gb : group_by_) {
            auto idx = input.findColumnIndex(gb);
            if (!idx) throw std::runtime_error("AggregateNode: group-by column '" + gb + "' not found");
            key += valueToString(getColumnValue(input.getColumn(*idx), r)) + "|";
        }
        groups[key].push_back(r);
    }

    // If no group-by columns, single group over all rows.
    if (group_by_.empty() && groups.empty()) {
        std::vector<std::size_t> all_rows(nrows);
        std::iota(all_rows.begin(), all_rows.end(), 0);
        groups[""] = std::move(all_rows);
    }

    // Build output columns.
    // Group-by columns
    std::vector<std::string> out_names;
    std::vector<ColumnVariant> out_cols;

    for (const auto& gb : group_by_) {
        auto src_idx = *input.findColumnIndex(gb);
        // Create output column of same type
        std::visit([&](const auto& typed_src) {
            using ColT = std::decay_t<decltype(typed_src)>;
            ColT new_col;
            for (const auto& [key, rows] : groups)
                new_col.pushValue(typed_src[rows[0]]); // take first row's value
            out_cols.emplace_back(std::move(new_col));
        }, input.getColumn(src_idx));
        out_names.push_back(gb);
    }

    // Aggregate columns
    for (const auto& agg : aggregates_) {
        Float64Column result_col;
        if (agg.func == "COUNT") {
            // COUNT — output as float64 for uniformity
            for (const auto& [key, rows] : groups)
                result_col.pushValue(static_cast<double>(rows.size()));
        } else {
            auto src_idx = input.findColumnIndex(agg.input_column);
            if (!src_idx)
                throw std::runtime_error("AggregateNode: column '" + agg.input_column + "' not found");

            for (const auto& [key, rows] : groups) {
                double acc = 0;
                if (agg.func == "MIN") acc = std::numeric_limits<double>::max();
                if (agg.func == "MAX") acc = std::numeric_limits<double>::lowest();

                for (auto r : rows) {
                    double v = toDouble(getColumnValue(input.getColumn(*src_idx), r));
                    if (agg.func == "SUM" || agg.func == "AVG") acc += v;
                    else if (agg.func == "MIN") acc = std::min(acc, v);
                    else if (agg.func == "MAX") acc = std::max(acc, v);
                }
                if (agg.func == "AVG") acc /= static_cast<double>(rows.size());
                result_col.pushValue(acc);
            }
        }
        out_cols.emplace_back(std::move(result_col));
        out_names.push_back(agg.output_name);
    }

    return Table("aggregate_result", std::move(out_names), std::move(out_cols));
}

std::string AggregateNode::toString(int indent) const {
    std::string pad(indent * 2, ' ');
    std::string s = pad + "AggregateNode(GROUP BY [";
    for (std::size_t i = 0; i < group_by_.size(); ++i) {
        if (i) s += ", ";
        s += group_by_[i];
    }
    s += "]; ";
    for (std::size_t i = 0; i < aggregates_.size(); ++i) {
        if (i) s += ", ";
        s += aggregates_[i].func + "(" + (aggregates_[i].input_column.empty() ? "*" : aggregates_[i].input_column) + ")";
    }
    s += ")\n";
    if (input_) s += input_->toString(indent + 1);
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// WindowNode
// ─────────────────────────────────────────────────────────────────────────────

WindowNode::WindowNode(WindowSpec spec) : spec_(std::move(spec)) {}

std::vector<double> WindowNode::computeSMA(const std::vector<double>& vals, int w) {
    const std::size_t n = vals.size();
    std::vector<double> out(n);
    double running = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        running += vals[i];
        if (static_cast<int>(i) >= w) running -= vals[i - w];
        int count = (static_cast<int>(i) < w) ? static_cast<int>(i) + 1 : w;
        out[i] = running / count;
    }
    return out;
}

std::vector<double> WindowNode::computeEMA(const std::vector<double>& vals, int w) {
    const std::size_t n = vals.size();
    std::vector<double> out(n);
    if (n == 0) return out;
    double alpha = 2.0 / (w + 1);
    out[0] = vals[0];
    for (std::size_t i = 1; i < n; ++i)
        out[i] = alpha * vals[i] + (1.0 - alpha) * out[i - 1];
    return out;
}

std::vector<double> WindowNode::computeRollingStd(const std::vector<double>& vals, int w) {
    const std::size_t n = vals.size();
    std::vector<double> out(n);
    double sumX = 0.0, sumX2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sumX  += vals[i];
        sumX2 += vals[i] * vals[i];
        if (static_cast<int>(i) >= w) {
            sumX  -= vals[i - w];
            sumX2 -= vals[i - w] * vals[i - w];
        }
        int count = (static_cast<int>(i) < w) ? static_cast<int>(i) + 1 : w;
        if (count < 2) {
            out[i] = 0.0;
        } else {
            double mean = sumX / count;
            double var  = sumX2 / count - mean * mean;
            out[i] = (var > 0.0) ? std::sqrt(var) : 0.0;
        }
    }
    return out;
}

Table WindowNode::execute(const ExecutionContext& ctx) {
    Table input = input_->execute(ctx);

    // (a) Sort by order_by_col if specified.
    if (!spec_.order_by_col.empty()) {
        auto col_idx = input.findColumnIndex(spec_.order_by_col);
        if (!col_idx)
            throw std::runtime_error("WindowNode: ORDER BY column '" +
                                     spec_.order_by_col + "' not found");
        std::vector<std::size_t> indices(input.numRows());
        std::iota(indices.begin(), indices.end(), 0);
        const auto& sort_col = input.getColumn(*col_idx);
        std::sort(indices.begin(), indices.end(),
                  [&](std::size_t a, std::size_t b) {
                      Value va = getColumnValue(sort_col, a);
                      Value vb = getColumnValue(sort_col, b);
                      if (isNumeric(va) && isNumeric(vb))
                          return toDouble(va) < toDouble(vb);
                      return valueToString(va) < valueToString(vb);
                  });
        input = selectRows(input, indices, input.name());
    }

    // (b) Extract input_col as vector<double>.
    auto in_idx = input.findColumnIndex(spec_.input_col);
    if (!in_idx)
        throw std::runtime_error("WindowNode: input column '" +
                                 spec_.input_col + "' not found");
    const std::size_t nrows = input.numRows();
    std::vector<double> vals(nrows);
    for (std::size_t i = 0; i < nrows; ++i)
        vals[i] = toDouble(getColumnValue(input.getColumn(*in_idx), i));

    // (c) Dispatch: GPU path if backend is set and row count is large enough.
    std::vector<double> result(nrows);
#ifdef USE_CUDA
    if (gpu_ && static_cast<int>(nrows) > GPU_THRESHOLD) {
        // GPU path
        void* d_in_raw  = gpu_->allocate(nrows * sizeof(double));
        void* d_out_raw = gpu_->allocate(nrows * sizeof(double));
        gpu_->copyToDevice(d_in_raw, vals.data(), nrows * sizeof(double));

        auto* d_in  = static_cast<double*>(d_in_raw);
        auto* d_out = static_cast<double*>(d_out_raw);

        if (spec_.func == "SMA")
            gpu_->windowSMA(d_in, static_cast<int>(nrows), spec_.window_size, d_out);
        else if (spec_.func == "EMA")
            gpu_->windowEMA(d_in, static_cast<int>(nrows), spec_.window_size, d_out);
        else if (spec_.func == "ROLLING_STD")
            gpu_->windowRollingStd(d_in, static_cast<int>(nrows), spec_.window_size, d_out);
        else {
            gpu_->deallocate(d_in_raw);
            gpu_->deallocate(d_out_raw);
            throw std::runtime_error("WindowNode: unknown function '" + spec_.func + "'");
        }

        gpu_->copyToHost(result.data(), d_out_raw, nrows * sizeof(double));
        gpu_->sync();
        gpu_->deallocate(d_in_raw);
        gpu_->deallocate(d_out_raw);
    } else
#endif
    {
        // CPU path
        if (spec_.func == "SMA")
            result = computeSMA(vals, spec_.window_size);
        else if (spec_.func == "EMA")
            result = computeEMA(vals, spec_.window_size);
        else if (spec_.func == "ROLLING_STD")
            result = computeRollingStd(vals, spec_.window_size);
        else
            throw std::runtime_error("WindowNode: unknown function '" +
                                     spec_.func + "'");
    }

    // (d) Append result column to a copy of the table.
    Float64Column out_col(std::move(result));
    input.addColumn(spec_.output_col, ColumnVariant(std::move(out_col)));
    return input;
}

std::string WindowNode::toString(int indent) const {
    std::string pad(indent * 2, ' ');
    std::string s = pad + "WindowNode(" + spec_.func + "(" +
                    spec_.input_col + ", " +
                    std::to_string(spec_.window_size) +
                    ") ORDER BY " + spec_.order_by_col + ")\n";
    if (input_) s += input_->toString(indent + 1);
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// SortNode

SortNode::SortNode(std::string column, bool ascending)
    : column_(std::move(column)), ascending_(ascending) {}

Table SortNode::execute(const ExecutionContext& ctx) {
    Table input = input_->execute(ctx);
    auto col_idx = input.findColumnIndex(column_);
    if (!col_idx) throw std::runtime_error("SortNode: column '" + column_ + "' not found");

    std::vector<std::size_t> indices(input.numRows());
    std::iota(indices.begin(), indices.end(), 0);

    const auto& sort_col = input.getColumn(*col_idx);
    bool asc = ascending_;
    std::sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) {
        Value va = getColumnValue(sort_col, a);
        Value vb = getColumnValue(sort_col, b);
        // Compare: numeric or string
        if (isNumeric(va) && isNumeric(vb)) {
            return asc ? toDouble(va) < toDouble(vb) : toDouble(va) > toDouble(vb);
        }
        auto sa = valueToString(va), sb = valueToString(vb);
        return asc ? sa < sb : sa > sb;
    });

    return selectRows(input, indices, input.name());
}

std::string SortNode::toString(int indent) const {
    std::string pad(indent * 2, ' ');
    std::string s = pad + "SortNode(" + column_ + (ascending_ ? " ASC" : " DESC") + ")\n";
    if (input_) s += input_->toString(indent + 1);
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// LimitNode
// ─────────────────────────────────────────────────────────────────────────────

LimitNode::LimitNode(std::int64_t limit) : limit_(limit) {}

Table LimitNode::execute(const ExecutionContext& ctx) {
    Table input = input_->execute(ctx);
    std::size_t n = std::min(static_cast<std::size_t>(limit_), input.numRows());
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    return selectRows(input, indices, input.name());
}

std::string LimitNode::toString(int indent) const {
    std::string pad(indent * 2, ' ');
    std::string s = pad + "LimitNode(" + std::to_string(limit_) + ")\n";
    if (input_) s += input_->toString(indent + 1);
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// materialize — public selection-vector → Table
// ─────────────────────────────────────────────────────────────────────────────

Table materialize(const Table& src, const SelectionVector& sel,
                  const std::string& name) {
    return selectRows(src, sel, name);
}

// ─────────────────────────────────────────────────────────────────────────────
// HashJoinNode
// ─────────────────────────────────────────────────────────────────────────────

HashJoinNode::HashJoinNode(std::string bt, std::string bc,
                           std::string pt, std::string pc)
    : build_table_(std::move(bt)), build_col_(std::move(bc)),
      probe_table_(std::move(pt)), probe_col_(std::move(pc)) {}

Table HashJoinNode::execute(const ExecutionContext& ctx) {
    auto bit = ctx.catalog.find(build_table_);
    auto pit = ctx.catalog.find(probe_table_);
    if (bit == ctx.catalog.end())
        throw std::runtime_error("HashJoinNode: build table '" + build_table_ + "' not found");
    if (pit == ctx.catalog.end())
        throw std::runtime_error("HashJoinNode: probe table '" + probe_table_ + "' not found");

    const Table& build = *bit->second;
    const Table& probe = *pit->second;

    auto bc_idx = build.findColumnIndex(build_col_);
    auto pc_idx = probe.findColumnIndex(probe_col_);
    if (!bc_idx)
        throw std::runtime_error("HashJoinNode: column '" + build_col_ + "' not in build table");
    if (!pc_idx)
        throw std::runtime_error("HashJoinNode: column '" + probe_col_ + "' not in probe table");

    // ── Build + Probe — dispatch on key type ────────────────────────────────
    const auto& bc = build.getColumn(*bc_idx);
    const auto& pc = probe.getColumn(*pc_idx);

    std::vector<std::pair<std::size_t, std::size_t>> matches;

    if (std::holds_alternative<Int32Column>(bc))
        buildAndProbe(std::get<Int32Column>(bc), std::get<Int32Column>(pc), matches);
    else if (std::holds_alternative<Int64Column>(bc))
        buildAndProbe(std::get<Int64Column>(bc), std::get<Int64Column>(pc), matches);
    else if (std::holds_alternative<StringColumn>(bc))
        buildAndProbe(std::get<StringColumn>(bc), std::get<StringColumn>(pc), matches);
    else
        throw std::runtime_error("HashJoinNode: unsupported join key type (FLOAT64)");

    // ── Assemble result table ───────────────────────────────────────────────
    std::vector<std::string> out_names;
    std::vector<ColumnVariant> out_cols;

    // Build-side columns (prefixed with table name)
    for (std::size_t c = 0; c < build.numCols(); ++c) {
        out_names.push_back(build_table_ + "." + build.columnNames()[c]);
        std::visit([&](const auto& typed_src) {
            using ColT = std::decay_t<decltype(typed_src)>;
            ColT col;
            col.reserve(matches.size());
            for (const auto& [br, pr] : matches) col.pushValue(typed_src[br]);
            out_cols.emplace_back(std::move(col));
        }, build.getColumn(c));
    }

    // Probe-side columns
    for (std::size_t c = 0; c < probe.numCols(); ++c) {
        out_names.push_back(probe_table_ + "." + probe.columnNames()[c]);
        std::visit([&](const auto& typed_src) {
            using ColT = std::decay_t<decltype(typed_src)>;
            ColT col;
            col.reserve(matches.size());
            for (const auto& [br, pr] : matches) col.pushValue(typed_src[pr]);
            out_cols.emplace_back(std::move(col));
        }, probe.getColumn(c));
    }

    return Table("join_result", std::move(out_names), std::move(out_cols));
}

std::string HashJoinNode::toString(int indent) const {
    std::string pad(indent * 2, ' ');
    return pad + "HashJoinNode(" + build_table_ + "." + build_col_ +
           " = " + probe_table_ + "." + probe_col_ + ")";
}

// ─────────────────────────────────────────────────────────────────────────────
// AsofJoinNode
// ─────────────────────────────────────────────────────────────────────────────

AsofJoinNode::AsofJoinNode(Spec spec) : spec_(std::move(spec)) {}

Table AsofJoinNode::cpuAsofJoin(const Table& left, const Table& right) const {
    // Locate timestamp columns
    auto left_ts_idx  = left.findColumnIndex(spec_.left_ts_col);
    auto right_ts_idx = right.findColumnIndex(spec_.right_ts_col);
    if (!left_ts_idx)
        throw std::runtime_error("AsofJoinNode: left ts column '" +
                                 spec_.left_ts_col + "' not found");
    if (!right_ts_idx)
        throw std::runtime_error("AsofJoinNode: right ts column '" +
                                 spec_.right_ts_col + "' not found");

    const auto& left_ts_col  = std::get<Int64Column>(left.getColumn(*left_ts_idx));
    const auto& right_ts_col = std::get<Int64Column>(right.getColumn(*right_ts_idx));
    const std::size_t n_left  = left_ts_col.size();
    const std::size_t n_right = right_ts_col.size();

    // Optional: locate key columns
    bool use_key = !spec_.left_key.empty() && !spec_.right_key.empty();
    const Int32Column* lk_col = nullptr;
    const Int32Column* rk_col = nullptr;
    const Int64Column* lk64_col = nullptr;
    const Int64Column* rk64_col = nullptr;
    bool key_is_int64 = false;

    if (use_key) {
        auto lk_idx = left.findColumnIndex(spec_.left_key);
        auto rk_idx = right.findColumnIndex(spec_.right_key);
        if (!lk_idx || !rk_idx)
            throw std::runtime_error("AsofJoinNode: key columns not found");

        // Support both INT32 and INT64 key columns
        const auto& lk_var = left.getColumn(*lk_idx);
        const auto& rk_var = right.getColumn(*rk_idx);
        if (std::holds_alternative<Int32Column>(lk_var)) {
            lk_col = &std::get<Int32Column>(lk_var);
            rk_col = &std::get<Int32Column>(rk_var);
        } else if (std::holds_alternative<Int64Column>(lk_var)) {
            lk64_col = &std::get<Int64Column>(lk_var);
            rk64_col = &std::get<Int64Column>(rk_var);
            key_is_int64 = true;
        }
    }

    // Build sorted index of right rows by timestamp (already assumed sorted)
    // For each left row, binary search right for the rightmost right_ts <= left_ts
    // with matching key (if use_key).
    std::vector<int> match_idx(n_left, -1);

    for (std::size_t i = 0; i < n_left; ++i) {
        std::int64_t lt = left_ts_col[i];
        int lo = 0, hi = static_cast<int>(n_right) - 1, best = -1;

        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (right_ts_col[mid] <= lt) {
                if (!use_key) {
                    best = mid;
                } else if (key_is_int64) {
                    if ((*lk64_col)[i] == (*rk64_col)[mid])
                        best = mid;
                } else if (lk_col) {
                    if ((*lk_col)[i] == (*rk_col)[mid])
                        best = mid;
                }
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }

        // Apply tolerance
        if (best != -1 && spec_.tolerance_ns > 0 &&
            (lt - right_ts_col[best]) > spec_.tolerance_ns)
            best = -1;

        match_idx[i] = best;
    }

    // Build output table
    std::vector<std::string> out_names;
    std::vector<ColumnVariant> out_cols;

    // Left-side columns (prefixed)
    for (std::size_t c = 0; c < left.numCols(); ++c) {
        out_names.push_back(spec_.left_table + "." + left.columnNames()[c]);
        std::visit([&](const auto& typed_src) {
            using ColT = std::decay_t<decltype(typed_src)>;
            ColT col;
            col.reserve(n_left);
            for (std::size_t i = 0; i < n_left; ++i)
                col.pushValue(typed_src[i]);
            out_cols.emplace_back(std::move(col));
        }, left.getColumn(c));
    }

    // Right-side columns (prefixed, with NULLs for unmatched)
    for (std::size_t c = 0; c < right.numCols(); ++c) {
        out_names.push_back(spec_.right_table + "." + right.columnNames()[c]);
        std::visit([&](const auto& typed_src) {
            using ColT = std::decay_t<decltype(typed_src)>;
            ColT col;
            col.reserve(n_left);
            for (std::size_t i = 0; i < n_left; ++i) {
                if (match_idx[i] >= 0) {
                    col.pushValue(typed_src[match_idx[i]]);
                } else {
                    // Push a NULL for unmatched rows
                    col.pushNull();
                }
            }
            out_cols.emplace_back(std::move(col));
        }, right.getColumn(c));
    }

    return Table("asof_join_result", std::move(out_names), std::move(out_cols));
}

Table AsofJoinNode::execute(const ExecutionContext& ctx) {
    // Fetch tables from catalog
    auto lit = ctx.catalog.find(spec_.left_table);
    auto rit = ctx.catalog.find(spec_.right_table);
    if (lit == ctx.catalog.end())
        throw std::runtime_error("AsofJoinNode: left table '" +
                                 spec_.left_table + "' not in catalog");
    if (rit == ctx.catalog.end())
        throw std::runtime_error("AsofJoinNode: right table '" +
                                 spec_.right_table + "' not in catalog");

    const Table& left  = *lit->second;
    const Table& right = *rit->second;

    // CPU path (GPU path requires USE_CUDA and large tables)
    return cpuAsofJoin(left, right);
}

std::string AsofJoinNode::toString(int indent) const {
    std::string pad(indent * 2, ' ');
    return pad + "AsofJoinNode(" + spec_.left_table + "." +
           spec_.left_ts_col + " >= " + spec_.right_table + "." +
           spec_.right_ts_col + ", key=" + spec_.left_key + "=" +
           spec_.right_key + ")";
}

}  // namespace gpudb
