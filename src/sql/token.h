/**
 * @file token.h
 * @brief Token types and structures for the SQL lexer.
 */
#pragma once

#include <string>
#include <unordered_map>

namespace gpudb {

enum class TokenType {
    // Keywords
    KW_SELECT, KW_FROM, KW_WHERE, KW_GROUP, KW_BY, KW_ORDER, KW_LIMIT,
    KW_AND, KW_OR, KW_NOT, KW_AS,
    KW_JOIN, KW_ON, KW_INNER,
    KW_ASC, KW_DESC,
    // Aggregate keywords
    KW_SUM, KW_COUNT, KW_AVG, KW_MIN, KW_MAX,
    // Window-function keywords
    KW_OVER, KW_PARTITION, KW_ROWS, KW_PRECEDING,
    KW_UNBOUNDED, KW_FOLLOWING, KW_CURRENT,
    KW_SMA, KW_EMA, KW_ROLLING_STD,
    // ASOF join keywords
    KW_ASOF, KW_TOLERANCE,
    // Literals
    LIT_INTEGER, LIT_FLOAT, LIT_STRING,
    // Identifier
    IDENTIFIER,
    // Operators
    OP_EQ, OP_NEQ, OP_LT, OP_GT, OP_LTE, OP_GTE,
    OP_PLUS, OP_MINUS, OP_STAR, OP_SLASH,
    // Punctuation
    LPAREN, RPAREN, COMMA, SEMICOLON, DOT,
    // Special
    TOKEN_EOF, TOKEN_UNKNOWN
};

/** Human-readable name for a TokenType. */
inline std::string token_type_name(TokenType t) {
    switch (t) {
        case TokenType::KW_SELECT:    return "KW_SELECT";
        case TokenType::KW_FROM:      return "KW_FROM";
        case TokenType::KW_WHERE:     return "KW_WHERE";
        case TokenType::KW_GROUP:     return "KW_GROUP";
        case TokenType::KW_BY:        return "KW_BY";
        case TokenType::KW_ORDER:     return "KW_ORDER";
        case TokenType::KW_LIMIT:     return "KW_LIMIT";
        case TokenType::KW_AND:       return "KW_AND";
        case TokenType::KW_OR:        return "KW_OR";
        case TokenType::KW_NOT:       return "KW_NOT";
        case TokenType::KW_AS:        return "KW_AS";
        case TokenType::KW_JOIN:      return "KW_JOIN";
        case TokenType::KW_ON:        return "KW_ON";
        case TokenType::KW_INNER:     return "KW_INNER";
        case TokenType::KW_ASC:       return "KW_ASC";
        case TokenType::KW_DESC:      return "KW_DESC";
        case TokenType::KW_SUM:       return "KW_SUM";
        case TokenType::KW_COUNT:     return "KW_COUNT";
        case TokenType::KW_AVG:       return "KW_AVG";
        case TokenType::KW_MIN:       return "KW_MIN";
        case TokenType::KW_MAX:       return "KW_MAX";
        case TokenType::KW_OVER:      return "KW_OVER";
        case TokenType::KW_PARTITION: return "KW_PARTITION";
        case TokenType::KW_ROWS:      return "KW_ROWS";
        case TokenType::KW_PRECEDING: return "KW_PRECEDING";
        case TokenType::KW_UNBOUNDED: return "KW_UNBOUNDED";
        case TokenType::KW_FOLLOWING: return "KW_FOLLOWING";
        case TokenType::KW_CURRENT:   return "KW_CURRENT";
        case TokenType::KW_SMA:       return "KW_SMA";
        case TokenType::KW_EMA:       return "KW_EMA";
        case TokenType::KW_ROLLING_STD: return "KW_ROLLING_STD";
        case TokenType::KW_ASOF:      return "KW_ASOF";
        case TokenType::KW_TOLERANCE: return "KW_TOLERANCE";
        case TokenType::LIT_INTEGER:  return "LIT_INTEGER";
        case TokenType::LIT_FLOAT:    return "LIT_FLOAT";
        case TokenType::LIT_STRING:   return "LIT_STRING";
        case TokenType::IDENTIFIER:   return "IDENTIFIER";
        case TokenType::OP_EQ:        return "OP_EQ";
        case TokenType::OP_NEQ:       return "OP_NEQ";
        case TokenType::OP_LT:        return "OP_LT";
        case TokenType::OP_GT:        return "OP_GT";
        case TokenType::OP_LTE:       return "OP_LTE";
        case TokenType::OP_GTE:       return "OP_GTE";
        case TokenType::OP_PLUS:      return "OP_PLUS";
        case TokenType::OP_MINUS:     return "OP_MINUS";
        case TokenType::OP_STAR:      return "OP_STAR";
        case TokenType::OP_SLASH:     return "OP_SLASH";
        case TokenType::LPAREN:       return "LPAREN";
        case TokenType::RPAREN:       return "RPAREN";
        case TokenType::COMMA:        return "COMMA";
        case TokenType::SEMICOLON:    return "SEMICOLON";
        case TokenType::DOT:          return "DOT";
        case TokenType::TOKEN_EOF:    return "EOF";
        case TokenType::TOKEN_UNKNOWN:return "UNKNOWN";
    }
    return "???";
}

/** A single lexical token. */
struct Token {
    TokenType   type  = TokenType::TOKEN_UNKNOWN;
    std::string value;
    int         line  = 1;
    int         col   = 1;

    std::string toString() const {
        return token_type_name(type) + " \"" + value + "\" (" +
               std::to_string(line) + ":" + std::to_string(col) + ")";
    }
};

/** Map upper-cased identifiers to keyword TokenTypes. */
inline const std::unordered_map<std::string, TokenType>& keyword_map() {
    static const std::unordered_map<std::string, TokenType> m = {
        {"SELECT", TokenType::KW_SELECT}, {"FROM",  TokenType::KW_FROM},
        {"WHERE",  TokenType::KW_WHERE},  {"GROUP", TokenType::KW_GROUP},
        {"BY",     TokenType::KW_BY},     {"ORDER", TokenType::KW_ORDER},
        {"LIMIT",  TokenType::KW_LIMIT},  {"AND",   TokenType::KW_AND},
        {"OR",     TokenType::KW_OR},     {"NOT",   TokenType::KW_NOT},
        {"AS",     TokenType::KW_AS},     {"JOIN",  TokenType::KW_JOIN},
        {"ON",     TokenType::KW_ON},     {"INNER", TokenType::KW_INNER},
        {"ASC",    TokenType::KW_ASC},    {"DESC",  TokenType::KW_DESC},
        {"SUM",    TokenType::KW_SUM},    {"COUNT", TokenType::KW_COUNT},
        {"AVG",    TokenType::KW_AVG},    {"MIN",   TokenType::KW_MIN},
        {"MAX",    TokenType::KW_MAX},
        {"OVER",       TokenType::KW_OVER},
        {"PARTITION",  TokenType::KW_PARTITION},
        {"ROWS",       TokenType::KW_ROWS},
        {"PRECEDING",  TokenType::KW_PRECEDING},
        {"UNBOUNDED",  TokenType::KW_UNBOUNDED},
        {"FOLLOWING",  TokenType::KW_FOLLOWING},
        {"CURRENT",    TokenType::KW_CURRENT},
        {"SMA",        TokenType::KW_SMA},
        {"EMA",        TokenType::KW_EMA},
        {"ROLLING_STD", TokenType::KW_ROLLING_STD},
        {"ASOF",        TokenType::KW_ASOF},
        {"TOLERANCE",   TokenType::KW_TOLERANCE},
    };
    return m;
}

}  // namespace gpudb
