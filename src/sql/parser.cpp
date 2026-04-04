/**
 * @file parser.cpp
 * @brief Recursive-descent SQL parser implementation.
 */
#include "parser.h"
#include <stdexcept>

namespace gpudb {

// ── Token navigation ────────────────────────────────────────────────────────

const Token& Parser::cur() const { return (*tokens_)[pos_]; }
const Token& Parser::peek(std::size_t ahead) const {
    std::size_t idx = pos_ + ahead;
    return idx < tokens_->size() ? (*tokens_)[idx] : tokens_->back();
}
const Token& Parser::advance() { return (*tokens_)[pos_++]; }
bool Parser::check(TokenType t) const { return cur().type == t; }
bool Parser::match(TokenType t) { if (check(t)) { advance(); return true; } return false; }
const Token& Parser::expect(TokenType t, const std::string& ctx) {
    if (!check(t))
        error("Expected " + token_type_name(t) + " in " + ctx +
              ", got " + cur().toString());
    return advance();
}
void Parser::error(const std::string& msg) const {
    throw std::runtime_error("Parser error at " + std::to_string(cur().line) +
                             ":" + std::to_string(cur().col) + " — " + msg);
}

bool Parser::isAggregateKeyword(TokenType t) const {
    return t == TokenType::KW_SUM  || t == TokenType::KW_COUNT ||
           t == TokenType::KW_AVG  || t == TokenType::KW_MIN   ||
           t == TokenType::KW_MAX;
}

// ── Entry point ─────────────────────────────────────────────────────────────

SelectStmt Parser::parse(const std::vector<Token>& tokens) {
    tokens_ = &tokens;
    pos_ = 0;
    return parseSelect();
}

SelectStmt Parser::parseSelect() {
    SelectStmt stmt;
    expect(TokenType::KW_SELECT, "SELECT statement");

    // SELECT list
    stmt.select_list = parseSelectList();

    // FROM
    expect(TokenType::KW_FROM, "SELECT statement");
    stmt.from_table = expect(TokenType::IDENTIFIER, "FROM clause").value;

    // WHERE (optional)
    if (match(TokenType::KW_WHERE)) {
        stmt.where_clause = parseExpr();
    }

    // GROUP BY (optional)
    if (check(TokenType::KW_GROUP)) {
        advance(); // GROUP
        expect(TokenType::KW_BY, "GROUP BY");
        stmt.group_by.push_back(expect(TokenType::IDENTIFIER, "GROUP BY").value);
        while (match(TokenType::COMMA))
            stmt.group_by.push_back(expect(TokenType::IDENTIFIER, "GROUP BY").value);
    }

    // ORDER BY (optional)
    if (check(TokenType::KW_ORDER)) {
        advance(); // ORDER
        expect(TokenType::KW_BY, "ORDER BY");
        stmt.order_by_column = expect(TokenType::IDENTIFIER, "ORDER BY").value;
        stmt.order_ascending = true;
        if (check(TokenType::KW_ASC))  { advance(); stmt.order_ascending = true; }
        else if (check(TokenType::KW_DESC)) { advance(); stmt.order_ascending = false; }
    }

    // LIMIT (optional)
    if (match(TokenType::KW_LIMIT)) {
        const Token& t = expect(TokenType::LIT_INTEGER, "LIMIT");
        stmt.limit = std::stoll(t.value);
    }

    // Optional semicolon
    match(TokenType::SEMICOLON);
    return stmt;
}

// ── SELECT list ─────────────────────────────────────────────────────────────

std::vector<SelectItem> Parser::parseSelectList() {
    std::vector<SelectItem> items;
    items.push_back(parseSelectItem());
    while (match(TokenType::COMMA))
        items.push_back(parseSelectItem());
    return items;
}

SelectItem Parser::parseSelectItem() {
    SelectItem item;
    if (check(TokenType::OP_STAR)) {
        advance();
        item.expr = std::make_unique<StarExpr>();
    } else {
        item.expr = parseExpr();
    }
    // AS alias (optional)
    if (match(TokenType::KW_AS))
        item.alias = expect(TokenType::IDENTIFIER, "AS alias").value;
    return item;
}

// ── Expression parsing (precedence climbing) ────────────────────────────────

ExprPtr Parser::parseExpr() { return parseOrExpr(); }

ExprPtr Parser::parseOrExpr() {
    auto left = parseAndExpr();
    while (check(TokenType::KW_OR)) {
        advance();
        auto right = parseAndExpr();
        left = std::make_unique<BinaryExpr>("OR", std::move(left), std::move(right));
    }
    return left;
}

ExprPtr Parser::parseAndExpr() {
    auto left = parseNotExpr();
    while (check(TokenType::KW_AND)) {
        advance();
        auto right = parseNotExpr();
        left = std::make_unique<BinaryExpr>("AND", std::move(left), std::move(right));
    }
    return left;
}

ExprPtr Parser::parseNotExpr() {
    if (check(TokenType::KW_NOT)) {
        advance();
        auto operand = parseNotExpr();
        return std::make_unique<UnaryExpr>("NOT", std::move(operand));
    }
    return parseComparisonExpr();
}

ExprPtr Parser::parseComparisonExpr() {
    auto left = parseAddSubExpr();
    if (check(TokenType::OP_EQ)  || check(TokenType::OP_NEQ) ||
        check(TokenType::OP_LT)  || check(TokenType::OP_GT)  ||
        check(TokenType::OP_LTE) || check(TokenType::OP_GTE)) {
        std::string op = advance().value;
        auto right = parseAddSubExpr();
        left = std::make_unique<BinaryExpr>(op, std::move(left), std::move(right));
    }
    return left;
}

ExprPtr Parser::parseAddSubExpr() {
    auto left = parseMulDivExpr();
    while (check(TokenType::OP_PLUS) || check(TokenType::OP_MINUS)) {
        std::string op = advance().value;
        auto right = parseMulDivExpr();
        left = std::make_unique<BinaryExpr>(op, std::move(left), std::move(right));
    }
    return left;
}

ExprPtr Parser::parseMulDivExpr() {
    auto left = parseUnaryExpr();
    while (check(TokenType::OP_STAR) || check(TokenType::OP_SLASH)) {
        std::string op = advance().value;
        auto right = parseUnaryExpr();
        left = std::make_unique<BinaryExpr>(op, std::move(left), std::move(right));
    }
    return left;
}

ExprPtr Parser::parseUnaryExpr() {
    if (check(TokenType::OP_MINUS)) {
        advance();
        auto operand = parseUnaryExpr();
        return std::make_unique<UnaryExpr>("-", std::move(operand));
    }
    return parsePrimaryExpr();
}

ExprPtr Parser::parsePrimaryExpr() {
    // Aggregate function
    if (isAggregateKeyword(cur().type)) {
        std::string func = advance().value;
        expect(TokenType::LPAREN, "aggregate function");
        ExprPtr arg = nullptr;
        if (check(TokenType::OP_STAR)) {
            advance(); // consume *
        } else {
            arg = parseExpr();
        }
        expect(TokenType::RPAREN, "aggregate function");
        return std::make_unique<AggExpr>(func, std::move(arg));
    }
    // Parenthesized expression
    if (match(TokenType::LPAREN)) {
        auto expr = parseExpr();
        expect(TokenType::RPAREN, "parenthesized expression");
        return expr;
    }
    // Integer literal
    if (check(TokenType::LIT_INTEGER)) {
        auto val = std::stoll(advance().value);
        return std::make_unique<LiteralExpr>(val);
    }
    // Float literal
    if (check(TokenType::LIT_FLOAT)) {
        auto val = std::stod(advance().value);
        return std::make_unique<LiteralExpr>(val);
    }
    // String literal
    if (check(TokenType::LIT_STRING)) {
        return std::make_unique<LiteralExpr>(advance().value);
    }
    // Identifier (column or table.column)
    if (check(TokenType::IDENTIFIER)) {
        std::string name = advance().value;
        if (match(TokenType::DOT)) {
            std::string col = expect(TokenType::IDENTIFIER, "qualified column").value;
            return std::make_unique<ColumnRef>(name, col);
        }
        return std::make_unique<ColumnRef>(name);
    }
    error("Expected expression, got " + cur().toString());
}

}  // namespace gpudb
