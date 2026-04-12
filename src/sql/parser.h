/**
 * @file parser.h
 * @brief Recursive-descent SQL parser producing an AST.
 */
#pragma once

#include "ast.h"
#include "token.h"
#include <vector>

namespace gpudb {

class Parser {
public:
    /** Parse a token stream into a SelectStmt AST. */
    SelectStmt parse(const std::vector<Token>& tokens);

    /** Parse a token stream into an AsofJoinStmt AST. */
    AsofJoinStmt parseAsofJoin(const std::vector<Token>& tokens);

private:
    const std::vector<Token>* tokens_ = nullptr;
    std::size_t pos_ = 0;

    const Token& cur() const;
    const Token& peek(std::size_t ahead = 0) const;
    const Token& advance();
    bool check(TokenType t) const;
    bool match(TokenType t);
    const Token& expect(TokenType t, const std::string& context);
    [[noreturn]] void error(const std::string& msg) const;

    // Grammar productions
    SelectStmt           parseSelect();
    std::vector<SelectItem> parseSelectList();
    SelectItem           parseSelectItem();
    ExprPtr              parseExpr();
    ExprPtr              parseOrExpr();
    ExprPtr              parseAndExpr();
    ExprPtr              parseNotExpr();
    ExprPtr              parseComparisonExpr();
    ExprPtr              parseAddSubExpr();
    ExprPtr              parseMulDivExpr();
    ExprPtr              parseUnaryExpr();
    ExprPtr              parsePrimaryExpr();
    bool                 isAggregateKeyword(TokenType t) const;
    bool                 isWindowKeyword(TokenType t) const;
};

}  // namespace gpudb
