/**
 * @file ast.h
 * @brief Abstract Syntax Tree node types for parsed SQL queries.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace gpudb {

// ─────────────────────────────────────────────────────────────────────────────
// Expression hierarchy
// ─────────────────────────────────────────────────────────────────────────────

struct Expr;
using ExprPtr = std::unique_ptr<Expr>;

struct Expr {
    virtual ~Expr() = default;
    virtual std::string toString(int indent = 0) const = 0;
    virtual ExprPtr clone() const = 0;
};

/** Column reference (optionally table-qualified: table.column). */
struct ColumnRef : Expr {
    std::string table;   // empty if unqualified
    std::string column;
    ColumnRef() = default;
    ColumnRef(std::string col) : column(std::move(col)) {}
    ColumnRef(std::string tbl, std::string col) : table(std::move(tbl)), column(std::move(col)) {}
    std::string toString(int = 0) const override {
        return table.empty() ? column : table + "." + column;
    }
    ExprPtr clone() const override { return std::make_unique<ColumnRef>(table, column); }
};

/** Literal value: integer, float, or string. */
struct LiteralExpr : Expr {
    using LitValue = std::variant<std::int64_t, double, std::string>;
    LitValue value;
    LiteralExpr() = default;
    explicit LiteralExpr(std::int64_t v) : value(v) {}
    explicit LiteralExpr(double v) : value(v) {}
    explicit LiteralExpr(std::string v) : value(std::move(v)) {}
    std::string toString(int = 0) const override {
        return std::visit([](const auto& v) -> std::string {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::string>) return "'" + v + "'";
            else return std::to_string(v);
        }, value);
    }
    ExprPtr clone() const override {
        auto c = std::make_unique<LiteralExpr>();
        c->value = value;
        return c;
    }
};

/** Binary expression: left OP right (comparison, arithmetic, AND/OR). */
struct BinaryExpr : Expr {
    std::string op;  // =, !=, <, >, <=, >=, +, -, *, /, AND, OR
    ExprPtr left, right;
    BinaryExpr() = default;
    BinaryExpr(std::string o, ExprPtr l, ExprPtr r)
        : op(std::move(o)), left(std::move(l)), right(std::move(r)) {}
    std::string toString(int indent = 0) const override {
        std::string pad(indent * 2, ' ');
        return "(" + left->toString(indent) + " " + op + " " + right->toString(indent) + ")";
    }
    ExprPtr clone() const override {
        return std::make_unique<BinaryExpr>(op, left->clone(), right->clone());
    }
};

/** Unary expression: NOT expr, or -expr. */
struct UnaryExpr : Expr {
    std::string op;
    ExprPtr operand;
    UnaryExpr() = default;
    UnaryExpr(std::string o, ExprPtr e) : op(std::move(o)), operand(std::move(e)) {}
    std::string toString(int indent = 0) const override {
        return op + " " + operand->toString(indent);
    }
    ExprPtr clone() const override {
        return std::make_unique<UnaryExpr>(op, operand->clone());
    }
};

/** Aggregate function: SUM(expr), COUNT(*), AVG(expr), MIN(expr), MAX(expr). */
struct AggExpr : Expr {
    std::string func;  // SUM, COUNT, AVG, MIN, MAX
    ExprPtr arg;       // nullptr for COUNT(*)
    AggExpr() = default;
    AggExpr(std::string f, ExprPtr a) : func(std::move(f)), arg(std::move(a)) {}
    std::string toString(int = 0) const override {
        return func + "(" + (arg ? arg->toString() : "*") + ")";
    }
    ExprPtr clone() const override {
        return std::make_unique<AggExpr>(func, arg ? arg->clone() : nullptr);
    }
};

/** Wildcard (*) in SELECT *. */
struct StarExpr : Expr {
    std::string toString(int = 0) const override { return "*"; }
    ExprPtr clone() const override { return std::make_unique<StarExpr>(); }
};

// ─────────────────────────────────────────────────────────────────────────────
// Statement-level AST
// ─────────────────────────────────────────────────────────────────────────────

/** One item in the SELECT list: expression [AS alias]. */
struct SelectItem {
    ExprPtr     expr;
    std::string alias;  // empty if no AS
};

/** Complete SELECT statement AST. */
struct SelectStmt {
    std::vector<SelectItem>    select_list;
    std::string                from_table;
    ExprPtr                    where_clause;       // nullptr if no WHERE
    std::vector<std::string>   group_by;           // column names
    std::string                order_by_column;    // empty if no ORDER BY
    bool                       order_ascending = true;
    std::optional<std::int64_t> limit;

    /** Pretty-print the full AST tree. */
    std::string toString() const {
        std::string s = "SelectStmt\n";
        s += "  SELECT:\n";
        for (const auto& item : select_list) {
            s += "    " + item.expr->toString();
            if (!item.alias.empty()) s += " AS " + item.alias;
            s += "\n";
        }
        s += "  FROM: " + from_table + "\n";
        if (where_clause)
            s += "  WHERE: " + where_clause->toString() + "\n";
        if (!group_by.empty()) {
            s += "  GROUP BY:";
            for (const auto& g : group_by) s += " " + g;
            s += "\n";
        }
        if (!order_by_column.empty())
            s += "  ORDER BY: " + order_by_column + (order_ascending ? " ASC" : " DESC") + "\n";
        if (limit)
            s += "  LIMIT: " + std::to_string(*limit) + "\n";
        return s;
    }
};

}  // namespace gpudb
