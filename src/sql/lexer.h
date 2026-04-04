/**
 * @file lexer.h
 * @brief Hand-written SQL lexer (tokeniser).
 */
#pragma once

#include "token.h"
#include <string>
#include <vector>

namespace gpudb {

class Lexer {
public:
    /** Tokenise a SQL string. Returns tokens terminated by TOKEN_EOF. */
    std::vector<Token> tokenize(const std::string& sql);

private:
    std::string src_;
    std::size_t pos_ = 0;
    int line_ = 1, col_ = 1;

    char peek() const;
    char advance();
    bool atEnd() const;
    void skipWhitespace();
    Token readIdentifierOrKeyword();
    Token readNumber();
    Token readString();
    Token makeToken(TokenType t, const std::string& v, int startCol) const;
};

}  // namespace gpudb
