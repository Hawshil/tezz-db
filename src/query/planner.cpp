/**
 * @file planner.cpp
 * @brief QueryPlanner implementation.
 */
#include "planner.h"
#include <algorithm>
#include <stdexcept>

namespace gpudb {

// ─────────────────────────────────────────────────────────────────────────────
// plan() — main entry point
// ─────────────────────────────────────────────────────────────────────────────

std::unique_ptr<OperatorNode> QueryPlanner::plan(const SelectStmt& stmt,
                                                  const Schema& schema) {
    // 1. Collect all referenced columns for scan projection.
    std::set<std::string> needed;
    for (const auto& item : stmt.select_list)
        collectColumns(item.expr.get(), needed);
    if (stmt.where_clause)
        collectColumns(stmt.where_clause.get(), needed);
    for (const auto& gb : stmt.group_by)
        needed.insert(gb);
    if (!stmt.order_by_column.empty())
        needed.insert(stmt.order_by_column);

    // If SELECT * or no specific columns identified, project all.
    std::vector<std::string> scan_cols;
    if (needed.empty()) {
        for (const auto& def : schema.columns())
            scan_cols.push_back(def.name);
    } else {
        // Verify all needed columns exist in schema.
        for (const auto& col : needed) {
            if (!schema.hasColumn(col))
                throw std::runtime_error("Planner: column '" + col +
                                         "' not found in table schema");
            scan_cols.push_back(col);
        }
    }

    // 2. ScanNode (leaf).
    std::unique_ptr<OperatorNode> node = std::make_unique<ScanNode>(stmt.from_table, scan_cols);

    // 3. FilterNode (if WHERE clause present).
    if (stmt.where_clause) {
        auto filter = std::make_unique<FilterNode>(stmt.where_clause->clone());
        filter->setInput(std::move(node));
        node = std::move(filter);
    }

    // 4. AggregateNode (if GROUP BY or aggregates in SELECT list).
    bool has_agg = hasAggregates(stmt.select_list);
    if (has_agg || !stmt.group_by.empty()) {
        auto agg_specs = extractAggregates(stmt.select_list);
        auto agg_node = std::make_unique<AggregateNode>(stmt.group_by, agg_specs);
        agg_node->setInput(std::move(node));
        node = std::move(agg_node);
    }

    // 5. WindowNode(s) (if window functions in SELECT list).
    if (hasWindowExprs(stmt.select_list)) {
        for (auto& wspec : extractWindowSpecs(stmt.select_list)) {
            auto win = std::make_unique<WindowNode>(wspec);
            win->setInput(std::move(node));
            node = std::move(win);
        }
    }

    // 6. SortNode (if ORDER BY present).
    if (!stmt.order_by_column.empty()) {
        auto sort = std::make_unique<SortNode>(stmt.order_by_column,
                                                stmt.order_ascending);
        sort->setInput(std::move(node));
        node = std::move(sort);
    }

    // 7. LimitNode (if LIMIT present).
    if (stmt.limit) {
        auto lim = std::make_unique<LimitNode>(*stmt.limit);
        lim->setInput(std::move(node));
        node = std::move(lim);
    }

    return node;
}

// ─────────────────────────────────────────────────────────────────────────────
// collectColumns — walk the expression tree gathering ColumnRef names
// ─────────────────────────────────────────────────────────────────────────────

void QueryPlanner::collectColumns(const Expr* expr, std::set<std::string>& cols) {
    if (!expr) return;
    if (auto* c = dynamic_cast<const ColumnRef*>(expr)) {
        cols.insert(c->column);
    } else if (auto* b = dynamic_cast<const BinaryExpr*>(expr)) {
        collectColumns(b->left.get(), cols);
        collectColumns(b->right.get(), cols);
    } else if (auto* u = dynamic_cast<const UnaryExpr*>(expr)) {
        collectColumns(u->operand.get(), cols);
    } else if (auto* a = dynamic_cast<const AggExpr*>(expr)) {
        collectColumns(a->arg.get(), cols);
    } else if (auto* w = dynamic_cast<const WindowExpr*>(expr)) {
        collectColumns(w->arg.get(), cols);
        if (!w->spec.order_by_col.empty())
            cols.insert(w->spec.order_by_col);
    }
    // LiteralExpr, StarExpr — no columns to collect.
}

// ─────────────────────────────────────────────────────────────────────────────
// hasAggregates — check SELECT list for aggregate functions
// ─────────────────────────────────────────────────────────────────────────────

bool QueryPlanner::hasAggregates(const std::vector<SelectItem>& items) const {
    for (const auto& item : items)
        if (dynamic_cast<const AggExpr*>(item.expr.get())) return true;
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// extractAggregates — build AggregateSpec list from SELECT items
// ─────────────────────────────────────────────────────────────────────────────

std::vector<AggregateSpec> QueryPlanner::extractAggregates(
        const std::vector<SelectItem>& items) const {
    std::vector<AggregateSpec> specs;
    for (const auto& item : items) {
        auto* agg = dynamic_cast<const AggExpr*>(item.expr.get());
        if (!agg) continue;

        AggregateSpec spec;
        spec.func = agg->func;
        // Input column (empty for COUNT(*)).
        if (agg->arg) {
            if (auto* col = dynamic_cast<const ColumnRef*>(agg->arg.get()))
                spec.input_column = col->column;
            else
                throw std::runtime_error("Planner: aggregate argument must be "
                                         "a column reference (got complex expression)");
        }
        // Output name: use alias if provided, else auto-generate.
        spec.output_name = item.alias.empty()
            ? agg->toString()
            : item.alias;
        specs.push_back(std::move(spec));
    }
    return specs;
}

// ─────────────────────────────────────────────────────────────────────────────
// hasWindowExprs — check SELECT list for window functions
// ─────────────────────────────────────────────────────────────────────────────

bool QueryPlanner::hasWindowExprs(const std::vector<SelectItem>& items) const {
    for (const auto& item : items)
        if (dynamic_cast<const WindowExpr*>(item.expr.get())) return true;
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// extractWindowSpecs — build WindowNode::WindowSpec list from SELECT items
// ─────────────────────────────────────────────────────────────────────────────

std::vector<WindowNode::WindowSpec> QueryPlanner::extractWindowSpecs(
        const std::vector<SelectItem>& items) const {
    std::vector<WindowNode::WindowSpec> specs;
    for (const auto& item : items) {
        auto* wexpr = dynamic_cast<const WindowExpr*>(item.expr.get());
        if (!wexpr) continue;

        WindowNode::WindowSpec spec;
        spec.func = wexpr->func;
        // Input column
        if (wexpr->arg) {
            if (auto* col = dynamic_cast<const ColumnRef*>(wexpr->arg.get()))
                spec.input_col = col->column;
            else
                throw std::runtime_error("Planner: window function argument must be "
                                         "a column reference (got complex expression)");
        }
        spec.order_by_col = wexpr->spec.order_by_col;
        spec.window_size  = wexpr->window_size;
        // Output name: use alias if provided, else auto-generate.
        spec.output_col = item.alias.empty()
            ? wexpr->toString()
            : item.alias;
        specs.push_back(std::move(spec));
    }
    return specs;
}

// ─────────────────────────────────────────────────────────────────────────────
// planAsofJoin
// ─────────────────────────────────────────────────────────────────────────────

std::unique_ptr<OperatorNode> QueryPlanner::planAsofJoin(
    const AsofJoinStmt& stmt, const Schema& /*left_schema*/)
{
    AsofJoinNode::Spec spec;
    spec.left_table   = stmt.left_table;
    spec.right_table  = stmt.right_table;
    spec.left_key     = stmt.left_key_col;
    spec.right_key    = stmt.right_key_col;
    spec.left_ts_col  = stmt.left_ts_col;
    spec.right_ts_col = stmt.right_ts_col;
    spec.tolerance_ns = stmt.tolerance_ns;
    return std::make_unique<AsofJoinNode>(std::move(spec));
}

}  // namespace gpudb
