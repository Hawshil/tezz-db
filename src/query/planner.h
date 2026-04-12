/**
 * @file planner.h
 * @brief Query planner: translates a SelectStmt AST into an operator DAG.
 */
#pragma once

#include "operator_node.h"
#include "../core/schema.h"
#include "../sql/ast.h"

#include <memory>
#include <set>

namespace gpudb {

class QueryPlanner {
public:
    /**
     * @brief Build a physical execution plan from a parsed SELECT AST.
     *
     * @param stmt   The parsed SELECT statement.
     * @param schema The schema of the FROM table.
     * @return Root of the operator tree (caller owns the pointer).
     */
    std::unique_ptr<OperatorNode> plan(const SelectStmt& stmt,
                                       const Schema& schema);

    /**
     * @brief Build a physical execution plan from an ASOF JOIN AST.
     */
    std::unique_ptr<OperatorNode> planAsofJoin(
        const AsofJoinStmt& stmt, const Schema& left_schema);

private:
    /** Collect all column names referenced anywhere in the query. */
    void collectColumns(const Expr* expr, std::set<std::string>& cols);

    /** Check whether the SELECT list contains aggregate functions. */
    bool hasAggregates(const std::vector<SelectItem>& items) const;

    /** Extract AggregateSpecs from the SELECT list. */
    std::vector<AggregateSpec> extractAggregates(
        const std::vector<SelectItem>& items) const;

    /** Check whether the SELECT list contains window functions. */
    bool hasWindowExprs(const std::vector<SelectItem>& items) const;

    /** Extract WindowNode::WindowSpec list from the SELECT list. */
    std::vector<WindowNode::WindowSpec>
        extractWindowSpecs(const std::vector<SelectItem>& items) const;
};

}  // namespace gpudb
