/**
 * @file csv_reader.cpp
 * @brief Implementation of the CSVReader class.
 */

#include "csv_reader.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace gpudb {

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

CSVReader::CSVReader(std::string file_path, Schema schema,
                     CSVReaderOptions options)
    : file_path_(std::move(file_path)), schema_(std::move(schema)),
      options_(std::move(options)) {}

// ─────────────────────────────────────────────────────────────────────────────
// Main read() entry point
// ─────────────────────────────────────────────────────────────────────────────

Table CSVReader::read(const std::string &table_name) {
  // Reset state from any prior read().
  errors_.clear();
  rows_parsed_ = 0;

  // Open the file.
  std::ifstream file(file_path_);
  if (!file.is_open()) {
    throw std::runtime_error("CSVReader::read — cannot open file: " +
                             file_path_);
  }

  const std::size_t num_cols = schema_.numCols();

  // Prepare per-column accumulators — each is an empty TypedColumn<T>
  // matching the schema's DataType order.
  std::vector<ColumnVariant> columns;
  columns.reserve(num_cols);
  for (std::size_t i = 0; i < num_cols; ++i) {
    switch (schema_[i].dtype) {
    case DataType::INT32:
      columns.emplace_back(Int32Column{});
      break;
    case DataType::INT64:
      columns.emplace_back(Int64Column{});
      break;
    case DataType::FLOAT64:
      columns.emplace_back(Float64Column{});
      break;
    case DataType::STRING:
      columns.emplace_back(StringColumn{});
      break;
    }
  }

  // ── Read lines ──────────────────────────────────────────────────────────

  std::string raw_line;
  std::size_t line_number = 0; // physical line counter
  std::size_t data_row = 0;    // logical row counter (post-header)
  bool header_skipped = false;

  while (std::getline(file, raw_line)) {
    ++line_number;

    // Handle multi-line quoted fields: if the number of unescaped quotes
    // is odd, the logical record continues on the next physical line.
    while (countUnescapedQuotes(raw_line) % 2 != 0) {
      std::string next_line;
      if (!std::getline(file, next_line))
        break; // EOF mid-record
      ++line_number;
      raw_line += '\n';
      raw_line += next_line;
    }

    // Skip the header row (first non-empty line when has_header is on).
    if (options_.has_header && !header_skipped) {
      header_skipped = true;
      continue;
    }

    // Skip empty lines.
    if (raw_line.empty())
      continue;

    // Parse the line into fields.
    std::vector<std::string> fields = parseLine(raw_line);

    // Validate field count.
    if (fields.size() != num_cols) {
      recordError(data_row + 1, 0, "",
                  "Expected " + std::to_string(num_cols) + " fields but got " +
                      std::to_string(fields.size()) + " on line " +
                      std::to_string(line_number));

      if (options_.strict) {
        throw std::runtime_error("CSVReader: strict mode — aborting on field "
                                 "count mismatch at line " +
                                 std::to_string(line_number));
      }

      // Skip this malformed row entirely.
      continue;
    }

    // Convert each field and push into the columnar accumulators.
    bool row_ok = true;
    for (std::size_t c = 0; c < num_cols; ++c) {
      if (!convertAndPush(fields[c], c, data_row + 1, columns)) {
        row_ok = false;
        if (options_.strict) {
          throw std::runtime_error(
              "CSVReader: strict mode — aborting on parse error at row " +
              std::to_string(data_row + 1) + ", col " + std::to_string(c));
        }
      }
    }

    if (row_ok) {
      ++rows_parsed_;
    }
    ++data_row;
  }

  // ── Assemble the Table ──────────────────────────────────────────────────

  std::vector<std::string> col_names;
  col_names.reserve(num_cols);
  for (std::size_t i = 0; i < num_cols; ++i) {
    col_names.push_back(schema_[i].name);
  }

  return Table(table_name, std::move(col_names), std::move(columns));
}

// ─────────────────────────────────────────────────────────────────────────────
// Error reporting accessors
// ─────────────────────────────────────────────────────────────────────────────

bool CSVReader::hasErrors() const noexcept { return !errors_.empty(); }

const std::vector<ParseError> &CSVReader::errors() const noexcept {
  return errors_;
}

std::size_t CSVReader::rowsParsed() const noexcept { return rows_parsed_; }

// ─────────────────────────────────────────────────────────────────────────────
// parseLine — RFC-4180-style field splitting
// ─────────────────────────────────────────────────────────────────────────────

std::vector<std::string> CSVReader::parseLine(const std::string &line) const {
  std::vector<std::string> fields;
  const char delim = options_.delimiter;
  const char quote = options_.quote_char;

  std::string field;
  bool in_quotes = false;

  for (std::size_t i = 0; i < line.size(); ++i) {
    char ch = line[i];

    if (in_quotes) {
      if (ch == quote) {
        // Peek ahead: if the next char is also a quote, it's an
        // escaped quote ("" → ").
        if (i + 1 < line.size() && line[i + 1] == quote) {
          field += quote;
          ++i; // consume the second quote
        } else {
          // End of quoted region.
          in_quotes = false;
        }
      } else {
        field += ch;
      }
    } else {
      if (ch == quote) {
        in_quotes = true;
      } else if (ch == delim) {
        fields.push_back(std::move(field));
        field.clear();
      } else {
        field += ch;
      }
    }
  }

  // Push the last field.
  fields.push_back(std::move(field));
  return fields;
}

// ─────────────────────────────────────────────────────────────────────────────
// convertAndPush — type-directed cell conversion
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Trim leading and trailing whitespace from a string view.
 */
static std::string trimWhitespace(const std::string &s) {
  auto start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos)
    return "";
  auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

bool CSVReader::convertAndPush(const std::string &cell, std::size_t col_idx,
                               std::size_t row_num,
                               std::vector<ColumnVariant> &columns) {
  const ColumnDef &def = schema_[col_idx];
  std::string trimmed = trimWhitespace(cell);

  // ── Handle NULL / empty values ──────────────────────────────────────────
  bool is_empty = trimmed.empty();
  if (is_empty) {
    if (def.nullable) {
      // Push a null marker into the appropriate typed column.
      std::visit([](auto &col) { col.pushNull(); }, columns[col_idx]);
      return true;
    }
    // Non-nullable column with empty value → error.
    recordError(row_num, col_idx, cell,
                "Empty value in non-nullable column \"" + def.name + "\"");
    // Still push a default so column lengths stay aligned.
    std::visit([](auto &col) { col.emplace_back(); }, columns[col_idx]);
    return false;
  }

  // ── Type conversion ─────────────────────────────────────────────────────
  try {
    switch (def.dtype) {
    case DataType::INT32: {
      auto &col = std::get<Int32Column>(columns[col_idx]);
      std::size_t pos = 0;
      std::int32_t val = static_cast<std::int32_t>(std::stoi(trimmed, &pos));
      if (pos != trimmed.size())
        throw std::invalid_argument("trailing chars");
      col.pushValue(val);
      break;
    }
    case DataType::INT64: {
      auto &col = std::get<Int64Column>(columns[col_idx]);
      std::size_t pos = 0;
      std::int64_t val = static_cast<std::int64_t>(std::stoll(trimmed, &pos));
      if (pos != trimmed.size())
        throw std::invalid_argument("trailing chars");
      col.pushValue(val);
      break;
    }
    case DataType::FLOAT64: {
      auto &col = std::get<Float64Column>(columns[col_idx]);
      std::size_t pos = 0;
      double val = std::stod(trimmed, &pos);
      if (pos != trimmed.size())
        throw std::invalid_argument("trailing chars");
      col.pushValue(val);
      break;
    }
    case DataType::STRING: {
      auto &col = std::get<StringColumn>(columns[col_idx]);
      col.pushValue(std::move(trimmed));
      break;
    }
    }
  } catch (const std::exception &e) {
    recordError(row_num, col_idx, cell,
                "Failed to convert \"" + trimmed + "\" to " +
                    type_name(def.dtype) + " for column \"" + def.name +
                    "\": " + e.what());
    // Push a default so column lengths stay aligned.
    std::visit(
        [](auto &col) {
          if (col.hasNullBitmap()) {
            col.pushNull();
          } else {
            col.emplace_back();
          }
        },
        columns[col_idx]);
    return false;
  }

  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// recordError
// ─────────────────────────────────────────────────────────────────────────────

void CSVReader::recordError(std::size_t row, std::size_t col,
                            const std::string &cell, const std::string &msg) {
  if (errors_.size() >= options_.max_errors)
    return;

  std::string col_name = (col < schema_.numCols()) ? schema_[col].name : "?";

  errors_.push_back(ParseError{row, col, std::move(col_name), cell,
                               "[Row " + std::to_string(row) + ", Col " +
                                   std::to_string(col) + " (\"" + col_name +
                                   "\")] " + msg});
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: count unescaped quotes (for multi-line field detection)
// ─────────────────────────────────────────────────────────────────────────────

std::size_t CSVReader::countUnescapedQuotes(const std::string &line) const {
  const char q = options_.quote_char;
  std::size_t count = 0;
  for (std::size_t i = 0; i < line.size(); ++i) {
    if (line[i] == q) {
      // Skip escaped pairs ("").
      if (i + 1 < line.size() && line[i + 1] == q) {
        ++i; // skip the pair
      } else {
        ++count;
      }
    }
  }
  return count;
}

} // namespace gpudb
