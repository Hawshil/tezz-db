/**
 * @file lexer.cpp
 * @brief SQL lexer implementation.
 */
#include "lexer.h"
#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace gpudb {

char Lexer::peek() const { return atEnd() ? '\0' : src_[pos_]; }
char Lexer::advance() {
    char c = src_[pos_++];
    if (c == '\n') { ++line_; col_ = 1; } else { ++col_; }
    return c;
}
bool Lexer::atEnd() const { return pos_ >= src_.size(); }

void Lexer::skipWhitespace() {
    while (!atEnd() && std::isspace(static_cast<unsigned char>(peek()))) advance();
}

Token Lexer::makeToken(TokenType t, const std::string& v, int startCol) const {
    return Token{t, v, line_, startCol};
}

Token Lexer::readIdentifierOrKeyword() {
    int sc = col_;
    std::string word;
    while (!atEnd() && (std::isalnum(static_cast<unsigned char>(peek())) || peek() == '_'))
        word += advance();
    // Check keyword map (case-insensitive).
    std::string upper = word;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char c){ return static_cast<char>(std::toupper(c)); });
    auto& kw = keyword_map();
    auto it = kw.find(upper);
    if (it != kw.end()) return makeToken(it->second, upper, sc);
    return makeToken(TokenType::IDENTIFIER, word, sc);
}

Token Lexer::readNumber() {
    int sc = col_;
    std::string num;
    bool is_float = false;
    while (!atEnd() && std::isdigit(static_cast<unsigned char>(peek())))
        num += advance();
    if (!atEnd() && peek() == '.') {
        is_float = true;
        num += advance();
        while (!atEnd() && std::isdigit(static_cast<unsigned char>(peek())))
            num += advance();
    }
    return makeToken(is_float ? TokenType::LIT_FLOAT : TokenType::LIT_INTEGER, num, sc);
}

Token Lexer::readString() {
    int sc = col_;
    advance(); // consume opening quote
    std::string s;
    while (!atEnd()) {
        char c = advance();
        if (c == '\'') {
            if (!atEnd() && peek() == '\'') { s += advance(); continue; } // escaped ''
            return makeToken(TokenType::LIT_STRING, s, sc);
        }
        s += c;
    }
    throw std::runtime_error("Lexer: unterminated string literal at line " +
                             std::to_string(line_) + " col " + std::to_string(sc));
}

std::vector<Token> Lexer::tokenize(const std::string& sql) {
    src_ = sql; pos_ = 0; line_ = 1; col_ = 1;
    std::vector<Token> tokens;

    while (true) {
        skipWhitespace();
        if (atEnd()) break;

        int sc = col_;
        char c = peek();

        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            tokens.push_back(readIdentifierOrKeyword());
        } else if (std::isdigit(static_cast<unsigned char>(c))) {
            tokens.push_back(readNumber());
        } else if (c == '\'') {
            tokens.push_back(readString());
        } else if (c == '(') { advance(); tokens.push_back(makeToken(TokenType::LPAREN, "(", sc)); }
          else if (c == ')') { advance(); tokens.push_back(makeToken(TokenType::RPAREN, ")", sc)); }
          else if (c == ',') { advance(); tokens.push_back(makeToken(TokenType::COMMA, ",", sc)); }
          else if (c == ';') { advance(); tokens.push_back(makeToken(TokenType::SEMICOLON, ";", sc)); }
          else if (c == '.') { advance(); tokens.push_back(makeToken(TokenType::DOT, ".", sc)); }
          else if (c == '+') { advance(); tokens.push_back(makeToken(TokenType::OP_PLUS, "+", sc)); }
          else if (c == '-') { advance(); tokens.push_back(makeToken(TokenType::OP_MINUS, "-", sc)); }
          else if (c == '*') { advance(); tokens.push_back(makeToken(TokenType::OP_STAR, "*", sc)); }
          else if (c == '/') { advance(); tokens.push_back(makeToken(TokenType::OP_SLASH, "/", sc)); }
          else if (c == '=') { advance(); tokens.push_back(makeToken(TokenType::OP_EQ, "=", sc)); }
          else if (c == '!') {
              advance();
              if (!atEnd() && peek() == '=') { advance(); tokens.push_back(makeToken(TokenType::OP_NEQ, "!=", sc)); }
              else throw std::runtime_error("Lexer: unexpected '!' at line " + std::to_string(line_) + " col " + std::to_string(sc));
          }
          else if (c == '<') {
              advance();
              if (!atEnd() && peek() == '=') { advance(); tokens.push_back(makeToken(TokenType::OP_LTE, "<=", sc)); }
              else if (!atEnd() && peek() == '>') { advance(); tokens.push_back(makeToken(TokenType::OP_NEQ, "<>", sc)); }
              else tokens.push_back(makeToken(TokenType::OP_LT, "<", sc));
          }
          else if (c == '>') {
              advance();
              if (!atEnd() && peek() == '=') { advance(); tokens.push_back(makeToken(TokenType::OP_GTE, ">=", sc)); }
              else tokens.push_back(makeToken(TokenType::OP_GT, ">", sc));
          }
          else {
              throw std::runtime_error("Lexer: unknown character '" + std::string(1, c) +
                                       "' at line " + std::to_string(line_) + " col " + std::to_string(sc));
          }
    }
    tokens.push_back(makeToken(TokenType::TOKEN_EOF, "", col_));
    return tokens;
}

}  // namespace gpudb
